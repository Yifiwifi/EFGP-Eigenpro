from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .backends import GPUBackendBundle


@dataclass
class EigenspaceConfig:
    q_max: int
    block_size: int
    n_iter: int = 3
    method: str = "subspace_iter"


def estimate_top_eigenspace_v3(
    backend: GPUBackendBundle,
    apply_A_block_gpu: Callable[[Any], Any],
    size: int,
    cfg: EigenspaceConfig,
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
    au_norm = float(xp.linalg.norm(AU))
    denom = max(au_norm, 1e-30)
    res_norm_rel = float(res_norm / denom)
    diag = {
        "method": cfg.method,
        "n_iter": int(cfg.n_iter),
        "block_size": int(block),
        "residual_fro": res_norm,
        "residual_fro_rel": res_norm_rel,
        "residual_cols": None,
        "residual_cols_rel": None,
    }
    return eigvals, eigvecs, diag
