from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..discretization import basis_weights, choose_grid_params
from ..kernels import KernelSpec
from .backends import GPUBackendBundle
from .contexts import GPUDataContext, GPUOperatorContext
from .nufft_adapter import type1_coeffs_rhs, type1_ones_xtx, type2_eval
from .iterative_solvers import cg_solve_gpu


@dataclass
class V1Outputs:
    """V1 pipeline result: weights on GPU, optional CPU mirror for tooling."""

    beta_gpu: Any
    diagnostics: dict[str, Any]
    backend: Optional[GPUBackendBundle] = None
    data_ctx: Optional[GPUDataContext] = None


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


def _ctx_complex_dtype(xp: Any, data_ctx: Optional[GPUDataContext]) -> Any:
    """
    Complex working dtype for the matvec, optionally lowered via
    ``data_ctx.meta['complex_dtype']`` (e.g. for a low-fidelity float32 operator).
    Defaults to complex128; the high-fidelity path is unaffected.
    """
    cd = None
    if data_ctx is not None:
        cd = data_ctx.meta.get("complex_dtype")
    if cd is None:
        return xp.complex128
    if cd in ("complex64", "single", "c8") or cd is getattr(xp, "complex64", None):
        return xp.complex64
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


def _apply_A_hot_1d_gpu(
    backend: GPUBackendBundle,
    data_ctx: GPUDataContext,
    v_gpu: Any,
    reg_lambda: float,
    op_ctx: GPUOperatorContext,
) -> Any:
    xp = backend.xp
    w_flat = data_ctx.weights_gpu_flat
    gf = data_ctx.gf_gpu
    if w_flat is None or gf is None:
        raise RuntimeError("gpu_precompute_v1 must run before apply_A_v1.")
    mtot = int(data_ctx.meta["mtot"])
    dtype = _ctx_complex_dtype(xp, data_ctx)

    w_buf = _ensure_workspace_vector(op_ctx, xp, "cg_tmp", v_gpu.size, dtype)
    xp.multiply(w_flat, v_gpu, out=w_buf)
    if _finite_checks_enabled(data_ctx) and not bool(xp.all(xp.isfinite(w_buf))):
        return xp.full(v_gpu.shape, xp.nan + 0.0j, dtype=dtype)

    pad, out = _ensure_workspace_pad_out(op_ctx, xp, tuple(int(s) for s in gf.shape), mtot, 1, gf.dtype)
    pad.fill(0)
    pad[:mtot] = w_buf
    af = backend.fft.fftn(pad)
    vfft = _ifft_of_fft_times_Gf_gpu(
        backend,
        af,
        gf,
        gmax_cached=data_ctx.meta.get("gf_absmax"),
    )
    xp.copyto(out, vfft[mtot - 1 :])
    with _errstate(xp, invalid="ignore", over="ignore"):
        xp.multiply(w_flat, out, out=out)
    if _finite_checks_enabled(data_ctx) and not bool(xp.all(xp.isfinite(out))):
        out.fill(xp.nan)
        return out
    out += reg_lambda * v_gpu
    return out


def _apply_A_hot_2d_gpu(
    backend: GPUBackendBundle,
    data_ctx: GPUDataContext,
    v_gpu: Any,
    reg_lambda: float,
    op_ctx: GPUOperatorContext,
) -> Any:
    xp = backend.xp
    w_flat = data_ctx.weights_gpu_flat
    gf = data_ctx.gf_gpu
    if w_flat is None or gf is None:
        raise RuntimeError("gpu_precompute_v1 must run before apply_A_v1.")
    mtot = int(data_ctx.meta["mtot"])
    dtype = _ctx_complex_dtype(xp, data_ctx)

    w_buf = _ensure_workspace_vector(op_ctx, xp, "cg_tmp", v_gpu.size, dtype)
    xp.multiply(w_flat, v_gpu, out=w_buf)
    if _finite_checks_enabled(data_ctx) and not bool(xp.all(xp.isfinite(w_buf))):
        return xp.full(v_gpu.shape, xp.nan + 0.0j, dtype=dtype)

    pad, out = _ensure_workspace_pad_out(op_ctx, xp, tuple(int(s) for s in gf.shape), mtot, 2, gf.dtype)
    pad.fill(0)
    pad[:mtot, :mtot] = w_buf.reshape(mtot, mtot)
    af = backend.fft.fftn(pad)
    vfft = _ifft_of_fft_times_Gf_gpu(
        backend,
        af,
        gf,
        gmax_cached=data_ctx.meta.get("gf_absmax"),
    )
    t = vfft[mtot - 1 : 2 * mtot - 1, mtot - 1 : 2 * mtot - 1]
    xp.copyto(out, t.reshape(-1))
    with _errstate(xp, invalid="ignore", over="ignore"):
        xp.multiply(w_flat, out, out=out)
    if _finite_checks_enabled(data_ctx) and not bool(xp.all(xp.isfinite(out))):
        out.fill(xp.nan)
        return out
    out += reg_lambda * v_gpu
    return out


def _apply_A_nd_gpu(
    backend: GPUBackendBundle,
    data_ctx: GPUDataContext,
    v_gpu: Any,
    reg_lambda: float,
    op_ctx: GPUOperatorContext,
    dim: int,
) -> Any:
    xp = backend.xp
    w_flat = data_ctx.weights_gpu_flat
    gf = data_ctx.gf_gpu
    if w_flat is None or gf is None:
        raise RuntimeError("gpu_precompute_v1 must run before apply_A_v1.")
    mtot = int(data_ctx.meta["mtot"])
    dtype = _ctx_complex_dtype(xp, data_ctx)

    w_buf = _ensure_workspace_vector(op_ctx, xp, "cg_tmp", v_gpu.size, dtype)
    xp.multiply(w_flat, v_gpu, out=w_buf)
    if _finite_checks_enabled(data_ctx) and not bool(xp.all(xp.isfinite(w_buf))):
        return xp.full(v_gpu.shape, xp.nan + 0.0j, dtype=dtype)

    pad, out = _ensure_workspace_pad_out(op_ctx, xp, tuple(int(s) for s in gf.shape), mtot, dim, gf.dtype)
    pad.fill(0)
    shape = (mtot,) * dim
    pad[tuple(slice(0, mtot) for _ in range(dim))] = w_buf.reshape(shape)
    af = backend.fft.fftn(pad)
    v = _ifft_of_fft_times_Gf_gpu(
        backend,
        af,
        gf,
        gmax_cached=data_ctx.meta.get("gf_absmax"),
    )
    slicer = tuple(slice(mtot - 1, 2 * mtot - 1) for _ in range(dim))
    t = v[slicer].reshape(-1)
    xp.multiply(w_flat, t, out=out)
    if _finite_checks_enabled(data_ctx) and not bool(xp.all(xp.isfinite(out))):
        out.fill(xp.nan)
        return out
    out += reg_lambda * v_gpu
    return out


def apply_A_v1(
    backend: GPUBackendBundle,
    data_ctx: GPUDataContext,
    v_gpu: Any,
    reg_lambda: float,
    op_ctx: GPUOperatorContext,
    out: Optional[Any] = None,
) -> Any:
    """
    GPU matvec: D (F^* F) D v + lambda v, aligned with ``EFGPSolver._apply_A``.
    If ``out`` is given, the result is written there (same shape as ``v``).
    """
    xp = backend.xp
    v_gpu = xp.asarray(v_gpu, dtype=_ctx_complex_dtype(xp, data_ctx)).reshape(-1)
    dim = int(data_ctx.meta.get("dim", 0))
    if dim < 1:
        raise RuntimeError("Invalid spatial dimension in GPUDataContext meta['dim'].")
    if dim == 1:
        res = _apply_A_hot_1d_gpu(backend, data_ctx, v_gpu, reg_lambda, op_ctx)
    elif dim == 2:
        res = _apply_A_hot_2d_gpu(backend, data_ctx, v_gpu, reg_lambda, op_ctx)
    else:
        res = _apply_A_nd_gpu(backend, data_ctx, v_gpu, reg_lambda, op_ctx, dim)
    if out is not None:
        xp.copyto(out, res)
        return out
    return res


def _gpu_free_memory_bytes(xp: Any) -> Optional[int]:
    try:
        cuda = getattr(xp, "cuda", None)
        if cuda is None:
            return None
        free_bytes, _total_bytes = cuda.runtime.memGetInfo()
        return int(free_bytes)
    except Exception:
        return None


def _choose_block_cols(
    xp: Any,
    gf_shape: tuple[int, ...],
    dtype: Any,
    requested: Any,
    n_cols: int,
    max_workspace_GB: Optional[float],
) -> int:
    if requested not in (None, "auto"):
        return max(1, min(int(requested), int(n_cols)))
    itemsize = int(xp.dtype(dtype).itemsize)
    spatial = 1
    for s in gf_shape:
        spatial *= int(s)
    budget = None
    if max_workspace_GB is not None:
        budget = float(max_workspace_GB) * (1024.0**3)
    else:
        free_bytes = _gpu_free_memory_bytes(xp)
        if free_bytes is not None:
            budget = 0.5 * float(free_bytes)
    if budget is None or budget <= 0.0:
        return max(1, min(8, int(n_cols)))
    factor = 5.0
    b = int(max(1, budget // max(spatial * itemsize * factor, 1)))
    return max(1, min(b, int(n_cols)))


def _ensure_workspace_pad_block(
    op_ctx: GPUOperatorContext,
    xp: Any,
    shape: tuple[int, ...],
    dtype: Any,
) -> Any:
    buf = getattr(op_ctx, "pad_block", None)
    if buf is None or tuple(getattr(buf, "shape", ())) != tuple(shape) or buf.dtype != dtype:
        buf = xp.empty(shape, dtype=dtype)
        setattr(op_ctx, "pad_block", buf)
    return buf


def _apply_A_block_v1_once(
    backend: GPUBackendBundle,
    data_ctx: GPUDataContext,
    V: Any,
    reg_lambda: float,
    op_ctx: GPUOperatorContext,
) -> Any:
    xp = backend.xp
    w_flat = data_ctx.weights_gpu_flat
    gf = data_ctx.gf_gpu
    if w_flat is None or gf is None:
        raise RuntimeError("gpu_precompute_v1 must run before apply_A_block_v1.")
    mtot = int(data_ctx.meta["mtot"])
    dim = int(data_ctx.meta["dim"])
    dtype = _ctx_complex_dtype(xp, data_ctx)
    V = xp.asarray(V, dtype=dtype)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    if V.ndim != 2:
        raise ValueError("V must be 1D or 2D.")
    n, b = int(V.shape[0]), int(V.shape[1])
    expected_n = int(mtot) ** int(dim)
    if n != expected_n:
        raise ValueError(f"V has incompatible row count {n}; expected {expected_n}.")

    U = xp.asarray(w_flat, dtype=dtype).reshape(-1, 1) * V
    U_nd = U.reshape((mtot,) * dim + (b,))
    pad_shape = tuple(int(s) for s in gf.shape) + (b,)
    pad = _ensure_workspace_pad_block(op_ctx, xp, pad_shape, xp.dtype(gf.dtype))
    pad.fill(0)
    pad[tuple(slice(0, mtot) for _ in range(dim)) + (slice(None),)] = U_nd

    axes = tuple(range(dim))
    af = backend.fft.fftn(pad, axes=axes)
    gf_exp = xp.asarray(gf)[(slice(None),) * dim + (None,)]
    gmax = data_ctx.meta.get("gf_absmax")
    gmax = float(gmax) if gmax is not None else float(xp.max(xp.abs(gf)))
    if math.isfinite(gmax):
        finf = np.finfo(np.float64).max
        safe = 0.25 * finf / max(gmax, 1e-300)
        amax_dev = xp.max(xp.abs(af))
        scale_dev = xp.maximum(1.0, amax_dev / safe)
        ypad = backend.fft.ifftn((af / scale_dev) * gf_exp, axes=axes) * scale_dev
    else:
        with _errstate(xp, over="ignore", invalid="ignore"):
            ypad = backend.fft.ifftn(af * gf_exp, axes=axes)
    slicer = tuple(slice(mtot - 1, 2 * mtot - 1) for _ in range(dim)) + (slice(None),)
    Y = ypad[slicer].reshape(n, b)
    with _errstate(xp, invalid="ignore", over="ignore"):
        Y = xp.asarray(w_flat, dtype=dtype).reshape(-1, 1) * Y
    Y = Y + float(reg_lambda) * V
    return xp.ascontiguousarray(Y)


def apply_A_block_v1(
    backend: GPUBackendBundle,
    data_ctx: GPUDataContext,
    V_gpu: Any,
    reg_lambda: float,
    op_ctx: GPUOperatorContext,
    *,
    block_cols: Any = "auto",
    max_workspace_GB: Optional[float] = None,
    out: Optional[Any] = None,
) -> Any:
    """
    Batched GPU matvec for multiple columns.

    Always returns a 2D array of shape ``(n, b)``.  A 1D input is treated as
    ``(n, 1)``.  The implementation chooses a safe column block size and falls
    back to smaller blocks / scalar ``apply_A_v1`` if a batched FFT allocation
    fails.
    """
    xp = backend.xp
    dtype = _ctx_complex_dtype(xp, data_ctx)
    V = xp.asarray(V_gpu, dtype=dtype)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    if V.ndim != 2:
        raise ValueError("V_gpu must be 1D or 2D.")
    n, btot = int(V.shape[0]), int(V.shape[1])
    out_arr = out
    if out_arr is None:
        out_arr = xp.empty((n, btot), dtype=dtype)
    else:
        if tuple(out_arr.shape) != (n, btot):
            raise ValueError("out must have shape (n, b).")
    gf = data_ctx.gf_gpu
    if gf is None:
        raise RuntimeError("gpu_precompute_v1 must run before apply_A_block_v1.")
    step0 = _choose_block_cols(
        xp,
        tuple(int(s) for s in gf.shape),
        gf.dtype,
        block_cols,
        btot,
        max_workspace_GB,
    )
    lo = 0
    while lo < btot:
        step = min(step0, btot - lo)
        while True:
            hi = lo + step
            try:
                Y = _apply_A_block_v1_once(
                    backend, data_ctx, V[:, lo:hi], reg_lambda, op_ctx
                )
                out_arr[:, lo:hi] = Y
                break
            except Exception:
                if step <= 1:
                    apply_A_v1(
                        backend,
                        data_ctx,
                        V[:, lo],
                        reg_lambda,
                        op_ctx,
                        out=out_arr[:, lo],
                    )
                    break
                step = max(1, step // 2)
        lo += step
    return out_arr


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
) -> GPUDataContext:
    """
    Precompute ``Gf``, ``rhs``, weights on GPU. NUFFT goes through ``nufft_adapter``:
    cuFINUFFT when available (``dim<=3``), else CPU FINUFFT + upload.

    If ``chunk_size`` is set and ``N > chunk_size``, accumulates type-1 transforms chunkwise
    (same spirit as ``EFGPSolver.precompute_streaming``). For ``dim>3``, chunk path uses
    CPU FINUFFT per chunk.
    """
    del op_ctx
    if (
        not force
        and data_ctx.gf_gpu is not None
        and data_ctx.rhs_gpu is not None
        and data_ctx.weights_gpu_flat is not None
        and data_ctx.meta.get("mtot") is not None
    ):
        if getattr(data_ctx, "xtxcol_gpu", None) is None and data_ctx.gf_gpu is not None:
            xp0 = backend.xp
            try:
                data_ctx.xtxcol_gpu = xp0.ascontiguousarray(backend.fft.ifftn(data_ctx.gf_gpu))
            except Exception:
                pass
        if getattr(data_ctx, "weights_np_flat", None) is None and data_ctx.weights_gpu_flat is not None:
            try:
                data_ctx.weights_np_flat = np.ascontiguousarray(
                    _device_array_to_numpy(data_ctx.weights_gpu_flat, np.float64).reshape(-1)
                )
            except Exception:
                pass
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
    grid = choose_grid_params(kernel, eps, L, l2scaled=l2scaled)
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
    data_ctx.xtxcol_gpu = xp.ascontiguousarray(XtXcol_gpu)

    w_gpu = xp.asarray(weights_np, dtype=xp.float64)
    weights_flat = w_gpu.reshape(-1)
    weights_nd = weights_flat.reshape((mtot,) * dim)
    rhs_flat = xp.asarray(rhs_u, dtype=_dtype_complex(xp)).reshape(-1)

    data_ctx.weights_gpu_nd = weights_nd
    data_ctx.weights_gpu_flat = weights_flat
    data_ctx.weights_np_flat = np.ascontiguousarray(np.asarray(weights_np, dtype=np.float64).reshape(-1))
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
        }
    )
    return data_ctx


def predict_v1(
    backend: GPUBackendBundle,
    data_ctx: GPUDataContext,
    x_eval: Any,
    beta_gpu: Any,
    *,
    return_nufft_stage: bool = False,
    allow_cpu_fallback: bool = True,
) -> Any:
    """
    Predict at ``x_eval`` (``numpy`` or CuPy ``(N, d)``), same formula as ``EFGPSolver.predict``.
    Type-2 NUFFT goes through ``nufft_adapter.type2_eval`` (cuFINUFFT when available for ``dim<=3``).
    Returns a real-valued CuPy vector on the active device.  With
    ``return_nufft_stage=True`` it also returns the actual backend tag.
    ``allow_cpu_fallback=False`` makes strict audit callers fail immediately
    on a missing or failed cuFINUFFT path instead of first running CPU FINUFFT.
    """
    xp = backend.xp
    if data_ctx.x_center_gpu is None or data_ctx.weights_gpu_flat is None:
        raise RuntimeError("gpu_precompute_v1 must run before predict_v1.")

    x_eval_gpu = xp.asarray(x_eval, dtype=xp.float64)
    if x_eval_gpu.ndim == 1:
        x_eval_gpu = x_eval_gpu.reshape(-1, 1)
    if x_eval_gpu.ndim != 2:
        raise ValueError("x_eval must be 2D (N, d).")
    dim = int(data_ctx.meta["dim"])
    if x_eval_gpu.shape[1] != dim:
        raise ValueError("x_eval.shape[1] must match precomputed spatial dimension.")

    h = float(data_ctx.meta["h"])
    mtot = int(data_ctx.meta["mtot"])
    nufft_tol = float(data_ctx.meta.get("nufft_tol", 1e-12))

    tphx = _tphx_scaled_gpu(xp, x_eval_gpu, data_ctx.x_center_gpu, h)
    beta_c = xp.asarray(beta_gpu, dtype=xp.complex128).reshape(-1)
    wbeta = data_ctx.weights_gpu_flat * beta_c
    yhat, stage = type2_eval(
        backend,
        tphx,
        wbeta,
        dim,
        mtot,
        nufft_tol,
        +1,
        allow_cpu_fallback=bool(allow_cpu_fallback),
    )
    prediction = xp.real(yhat)
    if return_nufft_stage:
        return prediction, str(stage)
    return prediction


def solve_beta_plain_cg_v1(
    backend: GPUBackendBundle,
    data_ctx: GPUDataContext,
    reg_lambda: float,
    op_ctx: GPUOperatorContext,
    tol: float,
    maxiter: int,
    *,
    return_stats: bool = False,
    profile_components: bool = True,
    trace_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> tuple[Any, int, float] | tuple[Any, int, float, dict[str, float]]:
    """
    Plain CG on GPU with reusable buffers on ``op_ctx``.
    """
    if data_ctx.rhs_gpu is None:
        raise RuntimeError("gpu_precompute_v1 must run before solve_beta_plain_cg_v1.")

    def _matvec(v: Any, out: Any) -> None:
        apply_A_v1(backend, data_ctx, v, reg_lambda, op_ctx, out=out)

    return cg_solve_gpu(
        backend,
        _matvec,
        data_ctx.rhs_gpu,
        op_ctx,
        tol,
        maxiter,
        return_stats=return_stats,
        work_prefix="cg",
        profile_components=profile_components,
        trace_callback=trace_callback,
    )
