from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .backends import GPUBackendBundle
from .cupy_eigenspace_methods import cupy_eigsh, rand_subspace_rr


@dataclass
class EigenspaceConfig:
    q_max: int
    block_size: int
    n_iter: int = 3
    method: str = "subspace_iter"
    method_cfg: Optional[dict[str, Any]] = None


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
    if cfg.block_size <= cfg.q_max:
        raise ValueError("block_size should be > q_max for oversampling")
    if cfg.n_iter < 1:
        raise ValueError("n_iter must be >= 1")

    method = str(cfg.method or "subspace_iter").lower()
    if method in ("cupy_eigsh", "rand_subspace_rr"):
        return _estimate_via_cupy_methods(
            backend, apply_A_block_gpu, int(size), cfg, method=method, init_Q=init_Q
        )
    if method != "subspace_iter":
        raise NotImplementedError(f"Unknown eigenspace method: {cfg.method!r}.")

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
        "method": str(cfg.method),
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
    res_norm, res_norm_rel = _fro_residual_norm(eigvecs, take_vals, AU, q, xp)
    out_vals = take_vals
    out_vecs = eigvecs
    ic = 0
    if init_Q is not None and hasattr(init_Q, "shape"):
        iq = init_Q
        if len(iq.shape) == 2:
            ic = int(iq.shape[1])
        elif len(iq.shape) == 1:
            ic = 1
    diag: dict[str, Any] = {
        "method": str(cfg.method),
        "n_iter": int(cfg.n_iter),
        "block_size": int(min(cfg.block_size, size)),
        "residual_fro": res_norm,
        "residual_fro_rel": res_norm_rel,
        "residual_cols": None,
        "residual_cols_rel": None,
        "init_used": init_Q is not None,
        "init_cols": int(ic),
    }
    return out_vals, out_vecs, diag
