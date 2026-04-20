from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import math
from contextlib import nullcontext

import numpy as np

from ..discretization import GridSpec, basis_weights, choose_grid_params
from ..kernels import KernelSpec
from .backends import GPUBackendBundle
from .contexts import GPUDataContext, GPUOperatorContext
from .nufft_adapter import type1_coeffs_rhs, type1_ones_xtx


@dataclass
class EigenspaceConfig:
    q_max: int
    block_size: int
    n_iter: int = 3
    method: str = "subspace_iter"


def _device_array_to_numpy(arr: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Host copy from CuPy (or pass-through numpy). ``dtype`` optional; do not force float on complex."""
    if arr is None:
        raise TypeError("array is None")
    if hasattr(arr, "get"):
        out = np.asarray(arr.get())
    else:
        out = np.asarray(arr)
    if dtype is not None:
        return out.astype(dtype, copy=False)
    return out


def _scale_from_center_numpy(x: np.ndarray, x_center: np.ndarray, h: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    tphx = 2.0 * np.pi * h * (x - x_center)
    upper = math.nextafter(math.pi, 0.0)
    return np.clip(tphx, -math.pi, upper)


def _tphx_scaled_gpu(xp: Any, x: Any, x_center: Any, h: float) -> Any:
    pi = xp.pi
    upper = xp.asarray(float(np.nextafter(float(np.pi), 0.0)), dtype=x.dtype)
    tphx = (2.0 * pi * h) * (x - x_center)
    return xp.clip(tphx, -pi, upper)


def _tphx_scaled_numpy(x: np.ndarray, x_center: np.ndarray, h: float) -> np.ndarray:
    return _scale_from_center_numpy(x, x_center, h)


def _ifft_of_fft_times_Gf_gpu(
    backend: GPUBackendBundle,
    af: Any,
    Gf: Any,
    *,
    gmax_cached: Optional[float] = None,
) -> Any:
    xp = backend.xp
    gmax = gmax_cached if gmax_cached is not None else float(xp.max(xp.abs(Gf)))
    if not math.isfinite(gmax):
        with _errstate(xp, over="ignore", invalid="ignore"):
            return backend.fft.ifftn(af * Gf)
    finf = np.finfo(np.float64).max
    safe = 0.25 * finf / max(gmax, 1e-300)
    # Keep scale on device to avoid per-matvec host sync.
    amax_dev = xp.max(xp.abs(af))
    scale_dev = xp.maximum(1.0, amax_dev / safe)
    af_scaled = af / scale_dev
    vfft = backend.fft.ifftn(af_scaled * Gf)
    return vfft * scale_dev


def _dtype_complex(xp: Any) -> Any:
    return xp.complex128


def _finite_checks_enabled(data_ctx: GPUDataContext) -> bool:
    return bool(data_ctx.meta.get("debug_finite_checks", False))


def _sync_device(xp: Any) -> None:
    cuda = getattr(xp, "cuda", None)
    if cuda is not None:
        cuda.Stream.null.synchronize()


def _errstate(xp: Any, **kwargs: Any):
    try:
        fn = xp.errstate
    except Exception:
        return nullcontext()
    if callable(fn):
        return fn(**kwargs)
    return nullcontext()


def _ensure_workspace_vector(
    op_ctx: GPUOperatorContext,
    xp: Any,
    name: str,
    size: int,
    dtype: Any,
) -> Any:
    buf = getattr(op_ctx, name)
    if buf is None or getattr(buf, "size", 0) != int(size) or buf.dtype != dtype:
        buf = xp.empty((int(size),), dtype=dtype)
        setattr(op_ctx, name, buf)
    return buf


def _ensure_workspace_pad_out(
    op_ctx: GPUOperatorContext,
    xp: Any,
    Gf_shape: tuple[int, ...],
    mtot: int,
    dim: int,
    dtype: Any,
) -> tuple[Any, Any]:
    if op_ctx.pad is None or tuple(op_ctx.pad.shape) != tuple(Gf_shape) or op_ctx.pad.dtype != dtype:
        op_ctx.pad = xp.empty(Gf_shape, dtype=dtype)
    out_size = int(mtot) ** int(dim)
    if op_ctx.work_vec is None or op_ctx.work_vec.size != out_size or op_ctx.work_vec.dtype != dtype:
        op_ctx.work_vec = xp.empty((out_size,), dtype=dtype)
    return op_ctx.pad, op_ctx.work_vec


def gpu_precompute_v1(
    backend: GPUBackendBundle,
    kernel: KernelSpec,
    eps: float,
    nufft_tol: float,
    data_ctx: GPUDataContext,
    op_ctx: Optional[GPUOperatorContext] = None,
    *,
    l2scaled: bool = False,
    force: bool = False,
    chunk_size: Optional[int] = None,
    grid_override: Optional[GridSpec] = None,
) -> GPUDataContext:
    """
    Precompute ``Gf``, ``rhs``, weights on GPU. NUFFT goes through ``nufft_adapter``:
    cuFINUFFT when available (``dim<=3``), else CPU FINUFFT + upload.

    If ``chunk_size`` is set and ``N > chunk_size``, accumulates type-1 transforms chunkwise
    (same spirit as ``EFGPSolver.precompute_streaming``). For ``dim>3``, chunk path uses
    CPU FINUFFT per chunk.

    ``grid_override`` can be used for surrogate grid selection; if provided, its ``h`` and
    ``mtot`` are used instead of ``choose_grid_params``.
    """
    del op_ctx
    if (
        not force
        and data_ctx.gf_gpu is not None
        and data_ctx.rhs_gpu is not None
        and data_ctx.weights_gpu_flat is not None
        and data_ctx.meta.get("mtot") is not None
    ):
        return data_ctx

    xp = backend.xp
    x_gpu = xp.asarray(data_ctx.x_gpu, dtype=xp.float64)
    y_gpu = xp.asarray(data_ctx.y_gpu, dtype=xp.float64).reshape(-1)
    n = int(x_gpu.shape[0])
    if y_gpu.shape[0] != n:
        raise ValueError("x and y length mismatch.")

    x_min = xp.min(x_gpu, axis=0)
    x_max = xp.max(x_gpu, axis=0)
    L = float(xp.max(x_max - x_min))
    x_center_gpu = (x_min + x_max) / 2.0

    if grid_override is None:
        grid = choose_grid_params(kernel, eps, L, l2scaled=l2scaled)
    else:
        grid = grid_override

    weights_np = np.ascontiguousarray(basis_weights(kernel, grid.xis, grid.h).reshape(-1))
    dim = int(kernel.dim)
    mtot = int(grid.mtot)
    ms_xtx = 2 * mtot - 1

    stages: list[str] = []
    use_chunk = chunk_size is not None and int(chunk_size) > 0 and n > int(chunk_size)
    cs = int(chunk_size) if use_chunk else n

    if use_chunk:
        XtXcol_acc = xp.zeros((ms_xtx,) * dim, dtype=_dtype_complex(xp))
        rhs_acc = xp.zeros(mtot**dim, dtype=_dtype_complex(xp))
        for i in range(0, n, cs):
            j = min(i + cs, n)
            x_c = x_gpu[i:j]
            y_c = y_gpu[i:j]
            if dim <= 3 and backend.has_nufft:
                tphx_c = _tphx_scaled_gpu(xp, x_c, x_center_gpu, float(grid.h))
                part_x, s1 = type1_ones_xtx(backend, tphx_c, dim, mtot, nufft_tol, -1)
                stages.append(s1)
                XtXcol_acc += part_x
                coeff = xp.asarray(y_c, dtype=xp.complex128)
                part_r, s2 = type1_coeffs_rhs(backend, tphx_c, coeff, dim, mtot, nufft_tol, -1)
                stages.append(s2)
                rhs_acc += part_r.reshape(-1)
            else:
                x_np_c = _device_array_to_numpy(x_c, np.float64)
                y_np_c = _device_array_to_numpy(y_c, np.float64).reshape(-1)
                x_center_np = _device_array_to_numpy(x_center_gpu, np.float64)
                tphx_np_c = _tphx_scaled_numpy(x_np_c, x_center_np, float(grid.h))
                part_x, s1 = type1_ones_xtx(backend, tphx_np_c, dim, mtot, nufft_tol, -1)
                stages.append(s1)
                XtXcol_acc += part_x
                part_r, s2 = type1_coeffs_rhs(
                    backend,
                    tphx_np_c,
                    np.asarray(y_np_c, dtype=np.complex128),
                    dim,
                    mtot,
                    nufft_tol,
                    -1,
                )
                stages.append(s2)
                rhs_acc += part_r.reshape(-1)
        XtXcol_gpu = XtXcol_acc
        rhs_u = rhs_acc
    else:
        if dim <= 3 and backend.has_nufft:
            tphx_g = _tphx_scaled_gpu(xp, x_gpu, x_center_gpu, float(grid.h))
            XtXcol_gpu, s1 = type1_ones_xtx(backend, tphx_g, dim, mtot, nufft_tol, -1)
            stages.append(s1)
            coeff_g = xp.asarray(y_gpu, dtype=xp.complex128)
            rhs_u, s2 = type1_coeffs_rhs(backend, tphx_g, coeff_g, dim, mtot, nufft_tol, -1)
            stages.append(s2)
        else:
            x_np = _device_array_to_numpy(x_gpu, np.float64)
            y_np = _device_array_to_numpy(y_gpu, np.float64).reshape(-1)
            x_center_np = _device_array_to_numpy(x_center_gpu, np.float64)
            tphx_np = _tphx_scaled_numpy(x_np, x_center_np, float(grid.h))
            XtXcol_gpu, s1 = type1_ones_xtx(backend, tphx_np, dim, mtot, nufft_tol, -1)
            stages.append(s1)
            rhs_u, s2 = type1_coeffs_rhs(
                backend,
                tphx_np,
                np.asarray(y_np, dtype=np.complex128),
                dim,
                mtot,
                nufft_tol,
                -1,
            )
            stages.append(s2)

    Gf_gpu = backend.fft.fftn(XtXcol_gpu)
    Gf_gpu = xp.ascontiguousarray(Gf_gpu)

    w_gpu = xp.asarray(weights_np, dtype=xp.float64)
    weights_flat = w_gpu.reshape(-1)
    weights_nd = weights_flat.reshape((mtot,) * dim)
    rhs_flat = xp.asarray(rhs_u, dtype=_dtype_complex(xp)).reshape(-1)

    data_ctx.weights_gpu_nd = weights_nd
    data_ctx.weights_gpu_flat = weights_flat
    data_ctx.rhs_gpu = xp.multiply(weights_flat, rhs_flat)
    data_ctx.gf_gpu = Gf_gpu
    data_ctx.x_center_gpu = x_center_gpu

    stage_tag = ",".join(sorted(set(stages))) if stages else "unknown"
    data_ctx.meta.update(
        {
            "mtot": mtot,
            "dim": dim,
            "h": float(grid.h),
            "weight_shape": tuple(int(s) for s in weights_nd.shape),
            "gf_shape": tuple(int(s) for s in Gf_gpu.shape),
            "rhs_shape": tuple(int(s) for s in data_ctx.rhs_gpu.shape),
            "nufft_tol": float(nufft_tol),
            "nufft_stage": stage_tag,
            "chunk_size": int(cs) if use_chunk else None,
            "gf_absmax": float(xp.max(xp.abs(Gf_gpu))),
            "debug_finite_checks": bool(data_ctx.meta.get("debug_finite_checks", False)),
            "grid_override": grid_override is not None,
        }
    )
    return data_ctx


def estimate_top_eigenspace_v3(
    backend: GPUBackendBundle,
    apply_A_block_gpu: Callable[[Any], Any],
    size: int,
    cfg: EigenspaceConfig,
    *,
    init_Q: Optional[Any] = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """
    GPU eigenspace estimation via block subspace iteration.
    Returns eigenvalues/eigenvectors in descending order plus diagnostics.
    """
    if cfg.q_max < 1:
        raise ValueError("q_max must be >= 1")
    if size <= cfg.q_max:
        raise ValueError("q_max must be < size.")
    if cfg.block_size <= cfg.q_max:
        raise ValueError("block_size should be > q_max for oversampling")
    if cfg.n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    if cfg.method != "subspace_iter":
        raise NotImplementedError("Only subspace_iter is implemented for now.")

    xp = backend.xp
    block = min(int(cfg.block_size), int(size))
    if block < cfg.q_max:
        raise ValueError("block_size must be >= q_max.")

    rng = None
    randn = None
    try:
        rng = xp.random.RandomState(0)
    except Exception:
        rng = None
    if rng is not None:
        randn = rng.standard_normal
    else:
        randn = xp.random.standard_normal

    init_used = False
    init_cols = 0
    if init_Q is not None:
        Q = xp.asarray(init_Q, dtype=xp.complex128)
        if Q.ndim == 1:
            Q = Q.reshape(-1, 1)
        if Q.shape[0] != int(size):
            raise ValueError("init_Q shape mismatch.")
        if Q.shape[1] < block:
            extra = randn((int(size), block - Q.shape[1])).astype(xp.float64)
            extra = extra + 1j * randn((int(size), block - Q.shape[1])).astype(xp.float64)
            extra = extra.astype(xp.complex128)
            Q = xp.concatenate([Q, extra], axis=1)
        if Q.shape[1] > block:
            Q = Q[:, :block]
        Q, _ = xp.linalg.qr(Q)
        Q = xp.ascontiguousarray(Q)
        init_used = True
        init_cols = int(Q.shape[1])
    else:
        X = randn((int(size), block)).astype(xp.float64)
        X = X + 1j * randn((int(size), block)).astype(xp.float64)
        X = X.astype(xp.complex128)
        Q, _ = xp.linalg.qr(X)
        Q = xp.ascontiguousarray(Q)

    for _ in range(int(cfg.n_iter)):
        Y = apply_A_block_gpu(Q)
        if Y.shape != Q.shape:
            raise ValueError("apply_A_block_gpu returned shape mismatch.")
        if Y.dtype != Q.dtype:
            raise ValueError("apply_A_block_gpu returned dtype mismatch.")
        Q, _ = xp.linalg.qr(Y)
        Q = xp.ascontiguousarray(Q)

    Y = apply_A_block_gpu(Q)
    if Y.shape != Q.shape:
        raise ValueError("apply_A_block_gpu returned shape mismatch.")
    if Y.dtype != Q.dtype:
        raise ValueError("apply_A_block_gpu returned dtype mismatch.")
    B = Q.conj().T @ Y
    B = 0.5 * (B + B.conj().T)
    vals, vecs = xp.linalg.eigh(B)
    order = xp.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    eigvecs = Q @ vecs

    q = int(cfg.q_max)
    eigvals = vals[:q]
    eigvecs = xp.ascontiguousarray(eigvecs[:, :q])
    AU = apply_A_block_gpu(eigvecs)
    if AU.shape != eigvecs.shape:
        raise ValueError("apply_A_block_gpu returned shape mismatch on eigvecs.")
    if AU.dtype != eigvecs.dtype:
        raise ValueError("apply_A_block_gpu returned dtype mismatch on eigvecs.")
    resid = AU - eigvecs * eigvals.reshape(1, -1)
    res_norm = float(xp.linalg.norm(resid))
    res_cols = xp.linalg.norm(resid, axis=0)
    au_norm = float(xp.linalg.norm(AU))
    au_cols = xp.linalg.norm(AU, axis=0)
    denom = max(au_norm, 1e-30)
    res_norm_rel = float(res_norm / denom)
    res_cols_rel = res_cols / xp.maximum(au_cols, 1e-30)
    asnumpy = getattr(xp, "asnumpy", None)
    res_cols_host = asnumpy(res_cols) if callable(asnumpy) else res_cols
    res_cols_rel_host = asnumpy(res_cols_rel) if callable(asnumpy) else res_cols_rel
    diag = {
        "method": cfg.method,
        "n_iter": int(cfg.n_iter),
        "block_size": int(block),
        "residual_fro": res_norm,
        "residual_fro_rel": res_norm_rel,
        "residual_cols": res_cols_host,
        "residual_cols_rel": res_cols_rel_host,
        "init_used": bool(init_used),
        "init_cols": int(init_cols),
    }
    return eigvals, eigvecs, diag
