from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class EigenPairs:
    values: np.ndarray
    vectors: np.ndarray


def estimate_top_eigenspace(
    matvec: Callable[[np.ndarray], np.ndarray],
    size: int,
    top_q: int,
    method: str = "eigsh",
    tol: float = 1e-6,
    maxiter: int | None = None,
    *,
    matvec_block: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    block_size: Optional[int] = None,
    oversample: int = 2,
) -> EigenPairs:
    """
    Estimate top eigenpairs using matrix-free matvec.
    Requires scipy.sparse.linalg for eigsh.
    """
    if method not in ("eigsh", "lobpcg", "subspace_iter"):
        raise ValueError("method must be one of: eigsh, lobpcg, subspace_iter.")

    if top_q < 1 or top_q >= size:
        raise ValueError("top_q must satisfy 1 <= top_q < size.")

    # matrix-free eigensolvers
    try:
        from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for eigensolvers.") from exc

    def _matvec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.complex128)
        if v.ndim == 2 and v.shape[1] == 1:
            v = v.reshape(-1)
        return np.asarray(matvec(v), dtype=np.complex128).reshape(-1)

    def _matmat(V: np.ndarray) -> np.ndarray:
        V = np.asarray(V, dtype=np.complex128)
        if V.ndim == 1:
            V = V.reshape(-1, 1)
        if matvec_block is not None:
            return np.asarray(matvec_block(V), dtype=np.complex128)
        cols = [np.asarray(matvec(V[:, i]), dtype=np.complex128) for i in range(V.shape[1])]
        return np.stack(cols, axis=1)

    A = LinearOperator((size, size), matvec=_matvec, matmat=_matmat, dtype=np.complex128)
    k = min(top_q + 1, size - 1)

    if method == "eigsh":
        vals, vecs = eigsh(A, k=k, which="LA", tol=tol, maxiter=maxiter)
        order = np.argsort(vals)[::-1]
        return EigenPairs(values=vals[order], vectors=vecs[:, order])

    block = block_size or min(max(k + int(oversample), 2), size)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((size, block)) + 1j * rng.standard_normal((size, block))

    if method == "lobpcg":
        vals, vecs = lobpcg(A, X, largest=True, tol=tol, maxiter=maxiter)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        return EigenPairs(values=vals[:k], vectors=vecs[:, :k])

    # randomized subspace iteration
    n_iter = 2 if maxiter is None else max(1, int(maxiter))
    for _ in range(n_iter):
        Y = _matmat(X)
        Q, _ = np.linalg.qr(Y)
        X = Q
    B = Q.conj().T @ _matmat(Q)
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vecs_full = Q @ vecs
    return EigenPairs(values=vals[:k], vectors=vecs_full[:, :k])
    
# Consider randomized SVD / operator Nyström method