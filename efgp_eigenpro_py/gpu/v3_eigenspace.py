from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
import math
import time

import numpy as np
from .backends import GPUBackendBundle
from .cupy_eigenspace_methods import cupy_eigsh, rand_subspace_rr


@dataclass
class EigenspaceConfig:
    q_max: int
    block_size: int
    n_iter: int = 3
    method: str = "subspace_iter"
    # When None, dispatch uses ``method`` (legacy). Set to ``subspace_iter`` / ``eigenpro_nystrom`` / …
    eig_method: Optional[str] = None
    method_cfg: Optional[dict[str, Any]] = None
    surrogate_size: Optional[int] = None
    surrogate_oversample: int = 10
    surrogate_lowfreq_ratio: float = 0.25
    surrogate_seed: int = 0
    # None: auto from Nyström size s and target GPU memory (see _auto_block_rows).
    surrogate_block_rows: Optional[int] = None
    surrogate_eig_scale: Optional[float] = None
    surrogate_ritz_refine: bool = True
    surrogate_ritz_block_cols: int = 8
    # False: only embed W0 eigenvectors on coordinate indices S (no K[:,S] T^-1 lift); for profiling.
    surrogate_lift: bool = True
    eig_floor: float = 1e-12


def _diag_method_name(cfg: EigenspaceConfig) -> str:
    return str(cfg.eig_method if cfg.eig_method is not None else cfg.method)


def mu_for_precond_from_eig(
    vals_gpu: Any, q: int, eig_diag: dict[str, Any]
) -> float:
    """
    EigenPro / dominant-subspace style threshold :math:`\\mu \\approx \\lambda_{q+1}`.

    Prefer ``eig_diag['surrogate_mu']`` when present (e.g. Nyström+Ritz, cupy with q+1
    values); otherwise use ``vals_gpu[q]`` if there are at least q+1 values, else
    ``vals_gpu[-1]`` (less reliable).
    """
    sm = eig_diag.get("surrogate_mu", eig_diag.get("mu"))
    if sm is not None and math.isfinite(float(sm)):
        return float(sm)
    n = int(getattr(vals_gpu, "size", len(vals_gpu)))
    if n > int(q):
        return float(vals_gpu[int(q)])
    return float(vals_gpu[-1])


def estimate_top_eigenspace_v3(
    backend: GPUBackendBundle,
    apply_A_block_gpu: Callable[[Any], Any],
    size: int,
    cfg: EigenspaceConfig,
    *,
    init_Q: Optional[Any] = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """
    GPU eigenspace estimation. ``method`` selects the algorithm; see ``EigenspaceConfig``.
    Returns leading eigenvalue/eigenvector blocks (for preconditioning) plus diagnostics.
    """
    if cfg.q_max < 1:
        raise ValueError("q_max must be >= 1")
    if size <= cfg.q_max:
        raise ValueError("q_max must be < size.")
    method_name = cfg.eig_method if cfg.eig_method is not None else cfg.method
    method = str(method_name or "subspace_iter").lower()
    if method == "subspace_iter":
        if cfg.block_size <= cfg.q_max:
            raise ValueError("block_size should be > q_max for oversampling")
        if cfg.n_iter < 1:
            raise ValueError("n_iter must be >= 1")
    if method in ("cupy_eigsh", "rand_subspace_rr"):
        if cfg.n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        return _estimate_via_cupy_methods(
            backend, apply_A_block_gpu, int(size), cfg, method=method, init_Q=init_Q
        )
    if method in ("eigenpro_nystrom", "nystrom", "ep_nystrom"):
        return _estimate_eigenpro_nystrom(
            backend=backend,
            apply_A_block_gpu=apply_A_block_gpu,
            size=int(size),
            cfg=cfg,
        )
    if method != "subspace_iter":
        raise NotImplementedError(f"Unknown eigenspace method: {method_name!r}.")

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
    b_mat = Q.conj().T @ Y
    b_mat = 0.5 * (b_mat + b_mat.conj().T)
    vals, vecs = xp.linalg.eigh(b_mat)
    order = xp.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    eigvecs = Q @ vecs

    q = int(cfg.q_max)
    eigvals = vals[:q]
    eigvecs = xp.ascontiguousarray(eigvecs[:, :q])
    return _residual_output(apply_A_block_gpu, xp, cfg, eigvals, eigvecs, {
        "n_iter": int(cfg.n_iter),
        "block_size": int(block),
        "init_used": bool(init_used),
        "init_cols": int(init_cols),
    })


def _residual_output(
    apply_A_block_gpu: Callable[[Any], Any],
    xp: Any,
    cfg: EigenspaceConfig,
    eigvals: Any,
    eigvecs: Any,
    extra: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    q = int(cfg.q_max)
    AU = apply_A_block_gpu(eigvecs)
    if AU.shape != eigvecs.shape:
        raise ValueError("apply_A_block_gpu returned shape mismatch on eigvecs.")
    if AU.dtype != eigvecs.dtype:
        raise ValueError("apply_A_block_gpu returned dtype mismatch on eigvecs.")
    res_norm, res_norm_rel = _fro_residual_norm(eigvecs, eigvals, AU, q, xp)
    diag: dict[str, Any] = {
        "method": _diag_method_name(cfg),
        "n_iter": int(extra.get("n_iter", cfg.n_iter)),
        "block_size": int(extra.get("block_size", cfg.block_size)),
        "residual_fro": res_norm,
        "residual_fro_rel": res_norm_rel,
        "residual_cols": None,
        "residual_cols_rel": None,
        "init_used": bool(extra.get("init_used", False)),
        "init_cols": int(extra.get("init_cols", 0)),
    }
    e_vals = eigvals[:q] if int(eigvals.size) > q else eigvals
    e_vecs = eigvecs[:, :q] if eigvecs.shape[1] > q else eigvecs
    return e_vals, e_vecs, diag


def _fro_residual_norm(
    eigvecs: Any, eigvals: Any, AU: Any, q: int, xp: Any
) -> tuple[float, float]:
    lam = xp.asarray(eigvals, dtype=xp.complex128)[:q].reshape(1, -1)
    U = xp.asarray(eigvecs, dtype=xp.complex128)[:, :q]
    resid = AU[:, :q] - U * lam
    res_norm = float(xp.linalg.norm(resid))
    au_norm = float(xp.linalg.norm(AU[:, :q]))
    denom = max(au_norm, 1e-30)
    return res_norm, float(res_norm / denom)


def _estimate_via_cupy_methods(
    backend: GPUBackendBundle,
    apply_A_block_gpu: Callable[[Any], Any],
    size: int,
    cfg: EigenspaceConfig,
    *,
    method: str,
    init_Q: Optional[Any] = None,
) -> tuple[Any, Any, dict[str, Any]]:
    xp = backend.xp
    q = int(cfg.q_max)
    mcfg: dict[str, Any] = dict(cfg.method_cfg or {})
    if "maxiter" not in mcfg:
        mcfg["maxiter"] = None if method == "cupy_eigsh" else int(cfg.n_iter)
    mcfg.setdefault("tol", 1e-6)
    mcfg.setdefault("min_oversample", mcfg.get("min_oversample", 16))
    if init_Q is not None and "init_Q" not in mcfg:
        mcfg["init_Q"] = init_Q
    if method == "cupy_eigsh":
        res = cupy_eigsh(
            None,
            size,
            cfg.q_max,
            matvec_block=apply_A_block_gpu,
            block_size=cfg.block_size,
            oversample=int(mcfg.get("oversample", 2)),
            xp=xp,
            cfg=mcfg,
        )
    elif method == "rand_subspace_rr":
        res = rand_subspace_rr(
            None,
            size,
            cfg.q_max,
            matvec_block=apply_A_block_gpu,
            block_size=cfg.block_size,
            oversample=int(mcfg.get("oversample", 2)),
            cfg=mcfg,
            xp=xp,
        )
    else:
        raise NotImplementedError(method)

    vals_full = xp.real(xp.asarray(res.values, dtype=xp.float64).reshape(-1))
    vecs_full = xp.asarray(res.vectors, dtype=xp.complex128)
    need = int(min(q + 1, int(vals_full.size)))
    if need < 1:
        raise RuntimeError("eigenspace solve returned no eigenvalues.")
    take_vals = vals_full[:need]
    eigvecs = xp.ascontiguousarray(vecs_full[:, :q])
    AU = apply_A_block_gpu(eigvecs)
    if AU.shape != eigvecs.shape:
        raise ValueError("apply_A_block_gpu returned shape mismatch on eigvecs.")
    if AU.dtype != eigvecs.dtype:
        raise ValueError("apply_A_block_gpu returned dtype mismatch on eigvecs.")
    res_norm, res_norm_rel = _fro_residual_norm(eigvecs, take_vals[:q], AU, q, xp)
    out_vals = take_vals[:q]
    out_vecs = eigvecs
    if int(take_vals.size) > q:
        mu_cupy = float(take_vals[q])
    else:
        mu_cupy = float(take_vals[-1])
    ic = 0
    if init_Q is not None and hasattr(init_Q, "shape"):
        iq = init_Q
        if len(iq.shape) == 2:
            ic = int(iq.shape[1])
        elif len(iq.shape) == 1:
            ic = 1
    diag: dict[str, Any] = {
        "method": _diag_method_name(cfg),
        "n_iter": int(cfg.n_iter),
        "block_size": int(min(cfg.block_size, size)),
        "residual_fro": res_norm,
        "residual_fro_rel": res_norm_rel,
        "residual_cols": None,
        "residual_cols_rel": None,
        "init_used": init_Q is not None,
        "init_cols": int(ic),
        "surrogate_mu": mu_cupy,
        "mu": mu_cupy,
    }
    return out_vals, out_vecs, diag


def _auto_block_rows(s: int, *, target_mb: int = 256) -> int:
    """Row block size for Nyström cross-lift to cap peak memory (~t, c per block)."""
    bytes_per_complex = 16
    bytes_per_row = 3 * int(s) * bytes_per_complex
    return max(1, int(target_mb * 1024**2 // max(bytes_per_row, 1)))


def _cfg_get(cfg: EigenspaceConfig, name: str, default: Any) -> Any:
    if hasattr(cfg, name):
        val = getattr(cfg, name)
        if val is not None:
            return val
    mcfg = dict(cfg.method_cfg or {})
    return mcfg.get(name, default)


def _asnumpy(x: Any) -> np.ndarray:
    if hasattr(x, "get"):
        return np.asarray(x.get())
    return np.asarray(x)


def _unravel_indices_np(indices: np.ndarray, mtot: int, dim: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    return np.stack(np.unravel_index(idx, (mtot,) * dim), axis=1).astype(np.int64, copy=False)


def _sample_frequency_indices(
    weights_np: np.ndarray,
    *,
    q_max: int,
    surrogate_size: Optional[int],
    oversample: int,
    lowfreq_ratio: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    score = np.square(np.asarray(weights_np, dtype=np.float64).reshape(-1))
    m = int(score.size)
    need = int(q_max) + 1
    s = int(surrogate_size) if surrogate_size is not None else max(int(oversample) * need, need + 16)
    s = max(need, min(s, m))

    finite = np.isfinite(score) & (score > 0.0)
    if not np.any(finite):
        return rng.choice(m, size=s, replace=False).astype(np.int64)

    n_det = int(round(float(lowfreq_ratio) * s))
    n_det = max(0, min(n_det, s))
    det = np.argpartition(score, -n_det)[-n_det:] if n_det > 0 else np.empty((0,), dtype=np.int64)

    remain = s - int(det.size)
    if remain <= 0:
        out = np.unique(det)
        if out.size < s:
            pool = np.setdiff1d(np.arange(m), out, assume_unique=False)
            out = np.concatenate([out, rng.choice(pool, size=s - out.size, replace=False)])
        return out[:s].astype(np.int64, copy=False)

    prob = np.where(finite, score, 0.0)
    if det.size:
        prob[det] = 0.0
    psum = float(prob.sum())
    if (not math.isfinite(psum)) or psum <= 0.0:
        pool = np.setdiff1d(np.arange(m), det, assume_unique=False)
        rnd = rng.choice(pool, size=remain, replace=False)
    else:
        pnorm = prob / psum
        nz = np.flatnonzero(pnorm > 0.0)
        if int(nz.size) < remain:
            pool = np.setdiff1d(np.arange(m), det, assume_unique=False)
            rnd = rng.choice(pool, size=remain, replace=False)
        else:
            rnd = rng.choice(m, size=remain, replace=False, p=pnorm)

    out = np.unique(np.concatenate([det, rnd]).astype(np.int64))
    if out.size < s:
        pool = np.setdiff1d(np.arange(m), out, assume_unique=False)
        out = np.concatenate([out, rng.choice(pool, size=s - out.size, replace=False).astype(np.int64)])
    return out[:s].astype(np.int64, copy=False)


def _toeplitz_submatrix_gpu(
    xp: Any,
    xtxcol_gpu: Any,
    weights_gpu: Any,
    s_idx_np: np.ndarray,
    *,
    mtot: int,
    dim: int,
) -> Any:
    multi = _unravel_indices_np(s_idx_np, mtot, dim)
    shift = int(mtot) - 1
    diff = multi[None, :, :] - multi[:, None, :] + shift
    idx = tuple(xp.asarray(diff[..., k], dtype=xp.int64) for k in range(dim))
    t = xtxcol_gpu[idx]
    ws = weights_gpu[xp.asarray(s_idx_np, dtype=xp.int64)]
    w0 = ws[:, None] * t * ws[None, :]
    return 0.5 * (w0 + w0.conj().T)


def _apply_matvec_columns(
    xp: Any,
    matvec: Callable[[Any], Any],
    u: Any,
    *,
    block_cols: int = 8,
) -> Any:
    """Apply a block matvec to matrix columns; optional column batching to limit memory."""
    m, q = int(u.shape[0]), int(u.shape[1])
    au = xp.empty((m, q), dtype=u.dtype)
    step = int(max(1, block_cols))
    for lo in range(0, q, step):
        hi = min(lo + step, q)
        xb = u[:, lo:hi]
        try:
            yb = matvec(xb)
            if getattr(yb, "shape", None) == xb.shape:
                au[:, lo:hi] = yb
                continue
        except Exception:
            pass
        for j in range(lo, hi):
            au[:, j] = matvec(u[:, j])
    return au


def _toeplitz_cross_lift_gpu(
    xp: Any,
    xtxcol_gpu: Any,
    weights_gpu: Any,
    s_idx_np: np.ndarray,
    v_gpu: Any,
    tau_gpu: Any,
    *,
    mtot: int,
    dim: int,
    block_rows: int,
    eig_floor: float,
) -> Any:
    multi_s = _unravel_indices_np(s_idx_np, mtot, dim)
    shift = int(mtot) - 1
    m = int(weights_gpu.size)
    q = int(v_gpu.shape[1])
    tau_safe = xp.maximum(tau_gpu, xp.asarray(float(eig_floor), dtype=tau_gpu.dtype))
    b = v_gpu / tau_safe[None, :]
    ws = weights_gpu[xp.asarray(s_idx_np, dtype=xp.int64)]
    out = xp.empty((m, q), dtype=xp.complex128)

    rows_step = int(max(1, block_rows))
    for lo in range(0, m, rows_step):
        hi = min(lo + rows_step, m)
        rows_np = np.arange(lo, hi, dtype=np.int64)
        multi_r = _unravel_indices_np(rows_np, mtot, dim)

        diff = multi_s[None, :, :] - multi_r[:, None, :] + shift
        idx = tuple(xp.asarray(diff[..., k], dtype=xp.int64) for k in range(dim))
        t = xtxcol_gpu[idx]
        wr = weights_gpu[lo:hi]
        c = wr[:, None] * t * ws[None, :]
        out[lo:hi, :] = c @ b
    return out


def _estimate_eigenpro_nystrom(
    *,
    backend: GPUBackendBundle,
    apply_A_block_gpu: Callable[[Any], Any],
    size: int,
    cfg: EigenspaceConfig,
) -> tuple[Any, Any, dict[str, Any]]:
    t0 = time.perf_counter()
    mcfg = dict(cfg.method_cfg or {})
    data_ctx = mcfg.get("data_ctx")
    if data_ctx is None:
        raise ValueError("eigenpro_nystrom requires method_cfg['data_ctx'].")
    reg_lambda = float(mcfg.get("reg_lambda", 0.0))
    if data_ctx.gf_gpu is None or data_ctx.weights_gpu_flat is None:
        raise ValueError("data_ctx must include gf_gpu and weights_gpu_flat.")
    so = int(_cfg_get(cfg, "surrogate_oversample", 10))
    lfr = float(_cfg_get(cfg, "surrogate_lowfreq_ratio", 0.25))

    mtot = int(data_ctx.meta["mtot"])
    dim = int(data_ctx.meta["dim"])
    xp = backend.xp
    q_max = int(cfg.q_max)
    m = int(size)
    weights_gpu = xp.asarray(data_ctx.weights_gpu_flat, dtype=xp.float64).reshape(-1)
    if int(weights_gpu.size) != int(size):
        raise ValueError(
            f"size mismatch: size={size}, but weights_gpu_flat.size={weights_gpu.size}. "
            "Check data_ctx / grid_override / precompute cache."
        )
    weights_np = getattr(data_ctx, "weights_np_flat", None)
    if weights_np is None:
        weights_np = _asnumpy(weights_gpu)
        try:
            data_ctx.weights_np_flat = weights_np
        except Exception:
            pass

    s_idx_np = _sample_frequency_indices(
        weights_np,
        q_max=q_max,
        surrogate_size=_cfg_get(cfg, "surrogate_size", None),
        oversample=int(_cfg_get(cfg, "surrogate_oversample", 10)),
        lowfreq_ratio=float(_cfg_get(cfg, "surrogate_lowfreq_ratio", 0.25)),
        seed=int(_cfg_get(cfg, "surrogate_seed", 0)),
    )
    s = int(s_idx_np.size)
    if s < (q_max + 1):
        raise ValueError(f"surrogate_size={s} is too small for q_max={q_max}.")

    xtxcol_gpu = getattr(data_ctx, "xtxcol_gpu", None)
    if xtxcol_gpu is None:
        xtxcol_gpu = xp.ascontiguousarray(backend.fft.ifftn(data_ctx.gf_gpu))
        try:
            data_ctx.xtxcol_gpu = xtxcol_gpu
        except Exception:
            pass
    w0 = _toeplitz_submatrix_gpu(xp, xtxcol_gpu, weights_gpu, s_idx_np, mtot=mtot, dim=dim)
    ew, ev = xp.linalg.eigh(w0)
    order = xp.argsort(ew)[::-1]
    n_top = int(min(q_max + 1, int(ew.size)))
    pos = order[:n_top]
    tau_lift = xp.real(ew[pos])
    v_lift = ev[:, pos]
    q_lift = int(min(q_max + 1, int(v_lift.shape[1])))

    use_full_lift = bool(_cfg_get(cfg, "surrogate_lift", True))
    block_rows = 0
    if use_full_lift:
        block_rows_cfg = _cfg_get(cfg, "surrogate_block_rows", None)
        if block_rows_cfg is None:
            block_rows = _auto_block_rows(s, target_mb=256)
        else:
            block_rows = int(block_rows_cfg)

        u = _toeplitz_cross_lift_gpu(
            xp,
            xtxcol_gpu,
            weights_gpu,
            s_idx_np,
            v_lift[:, :q_lift],
            tau_lift[:q_lift],
            mtot=mtot,
            dim=dim,
            block_rows=block_rows,
            eig_floor=float(_cfg_get(cfg, "eig_floor", 1e-12)),
        )
    else:
        # Coordinate subspace: U = I[:,S] V_q (no Nyström cross-lift); weak precondition, fast setup.
        u = xp.zeros((m, q_max), dtype=xp.complex128)
        s_gpu = xp.asarray(s_idx_np, dtype=xp.int64)
        u[s_gpu, :] = v_lift[:, :q_max]
    u, _ = xp.linalg.qr(u, mode="reduced")
    u = xp.ascontiguousarray(u)

    scale_cfg = _cfg_get(cfg, "surrogate_eig_scale", None)
    scale = float(m / max(1, s)) if scale_cfg is None else float(scale_cfg)
    use_ritz = bool(_cfg_get(cfg, "surrogate_ritz_refine", True))
    ritz_cols = int(_cfg_get(cfg, "surrogate_ritz_block_cols", 8))
    residual_fro = float("nan")
    residual_fro_rel = float("nan")

    if use_ritz:
        au = _apply_matvec_columns(
            xp, apply_A_block_gpu, u, block_cols=max(1, ritz_cols)
        )
        h = u.conj().T @ au
        h = 0.5 * (h + h.conj().T)
        re, rvec = xp.linalg.eigh(h)
        rord = xp.argsort(re)[::-1]
        evals_all = xp.real(re[rord])
        rvec = rvec[:, rord]
        u_all = xp.ascontiguousarray(u @ rvec)
        au_rot = au @ rvec
        eigvals_out = evals_all[:q_max]
        eigvecs_out = u_all[:, :q_max]
        res = au_rot[:, :q_max] - eigvecs_out * eigvals_out.reshape(1, -1)
        residual_fro = float(xp.linalg.norm(res))
        denom = float(xp.linalg.norm(au_rot[:, :q_max]))
        residual_fro_rel = residual_fro / max(denom, 1e-30)
        if int(evals_all.size) > q_max:
            mu = float(evals_all[q_max])
        else:
            mu = float(eigvals_out[-1])
    else:
        tau_use = tau_lift[:q_max]
        eigvals_out = scale * xp.maximum(tau_use, 0.0) + reg_lambda
        eigvecs_out = u[:, :q_max]
        if int(tau_lift.size) > q_max:
            mu = float(scale * float(tau_lift[q_max]) + reg_lambda)
        else:
            mu = float(scale * float(tau_lift[-1]) + reg_lambda)

    t_eig = float(time.perf_counter() - t0)
    diag: dict[str, Any] = {
        "method": _diag_method_name(cfg),
        "eig_method": "eigenpro_nystrom",
        "n_iter": 0,
        "block_size": int(s),
        "residual_fro": residual_fro,
        "residual_fro_rel": residual_fro_rel,
        "residual_cols": None,
        "residual_cols_rel": None,
        "init_used": False,
        "init_cols": 0,
        "eig_nystrom_kernel_s": t_eig,
        "time_eigenspace": t_eig,
        "surrogate_tag": (
            f"nystrom_s{s}_q{q_max}_oversample{so}_low{lfr:g}"
            + ("" if use_full_lift else "_coord_nolift")
        ),
        "surrogate_indices": s_idx_np,
        "surrogate_mu": mu,
        "mu": mu,
        "surrogate_scale": float(scale),
        "surrogate_block_rows_used": int(block_rows),
        "surrogate_lift": bool(use_full_lift),
    }
    if (not use_ritz) and scale_cfg is None:
        diag["surrogate_scale_warning"] = (
            "M/s scale is heuristic under weighted sampling; prefer surrogate_ritz_refine=True."
        )
    return xp.asarray(eigvals_out, dtype=xp.float64), eigvecs_out, diag


def estimate_top_eigenspace_eigenpro_nystrom(
    backend: GPUBackendBundle,
    data_ctx: Any,
    reg_lambda: float,
    cfg: EigenspaceConfig,
    *,
    apply_A_block_gpu: Callable[[Any], Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """
    Public entry: EigenPro-style Nyström top-:math:`q` surrogate eigenspace for
    :math:`A = D(F^*F)D + \\lambda I`, using a ``gpu_precompute_v1``-style ``data_ctx``.

    This wraps :func:`_estimate_eigenpro_nystrom` (same as ``eig_method='eigenpro_nystrom'`` in
    :func:`estimate_top_eigenspace_v3`). Rayleigh–Ritz refinement uses ``apply_A_block_gpu`` when
    ``surrogate_ritz_refine`` is true.
    """
    mcfg = dict(cfg.method_cfg or {})
    mcfg["data_ctx"] = data_ctx
    mcfg["reg_lambda"] = float(reg_lambda)
    cfg2 = replace(cfg, method_cfg=mcfg, eig_method="eigenpro_nystrom")
    return _estimate_eigenpro_nystrom(
        backend=backend,
        apply_A_block_gpu=apply_A_block_gpu,
        size=int(data_ctx.weights_gpu_flat.size),
        cfg=cfg2,
    )
