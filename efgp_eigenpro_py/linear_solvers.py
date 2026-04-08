from typing import Callable, Optional, Tuple

import numpy as np


def pcg(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    tol: float,
    maxiter: int,
    precond: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, int, float]:
    """
    Preconditioned CG using scipy.sparse.linalg.cg.
    """
    try:
        from scipy.sparse.linalg import LinearOperator, cg  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for cg.") from exc

    n = b.size
    dtype = np.result_type(b.dtype, np.complex128)
    A = LinearOperator((n, n), matvec=matvec, dtype=dtype)

    M = None
    if precond is not None:
        M = LinearOperator((n, n), matvec=precond, dtype=dtype)

    it_count = 0

    def _cb(_xk: np.ndarray) -> None:
        nonlocal it_count
        it_count += 1

    x, info = cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, M=M, callback=_cb)
    # info == 0 means success, otherwise info is iterations or negative for error.
    relres = np.linalg.norm(matvec(x) - b) / max(np.linalg.norm(b), 1e-12)
    return x, it_count, relres


def richardson(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: np.ndarray,
    eta: float,
    tol: float,
    maxiter: int,
    precond: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, int, float]:
    """
    Preconditioned Richardson iteration.
    """
    x = x0.copy()
    r = matvec(x) - b
    relres = np.linalg.norm(r) / max(np.linalg.norm(b), 1e-12)
    if relres <= tol or maxiter <= 0:
        return x, 0, relres

    for it in range(maxiter):
        z = precond(r) if precond is not None else r
        Az = matvec(z)
        x = x - eta * z
        r = r - eta * Az
        relres = np.linalg.norm(r) / max(np.linalg.norm(b), 1e-12)
        if relres <= tol:
            return x, it + 1, relres
    return x, maxiter, relres


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 8
    M = rng.standard_normal((n, n))
    A = M.T @ M + 0.5 * np.eye(n)
    b = rng.standard_normal(n)
    x_ref = np.linalg.solve(A, b)

    def matvec(v: np.ndarray) -> np.ndarray:
        return A @ v

    diag_inv = 1.0 / np.diag(A)

    def precond(v: np.ndarray) -> np.ndarray:
        return diag_inv * v

    try:
        _ = __import__("scipy")
        x_cg, it_cg, relres_cg = pcg(matvec, b, tol=1e-8, maxiter=200, precond=precond)
        err_cg = np.linalg.norm(x_cg - x_ref) / max(np.linalg.norm(x_ref), 1e-12)
        print("pcg: it=%d relres=%.2e relerr=%.2e" % (it_cg, relres_cg, err_cg))
    except Exception as exc:
        print("pcg: skipped (%s)" % exc)

    eta = 0.9 / np.linalg.norm(A, 2)
    x0 = np.zeros_like(b)
    x_rich, it_rich, relres_rich = richardson(
        matvec, b, x0, eta=eta, tol=1e-8, maxiter=500, precond=precond
    )
    err_rich = np.linalg.norm(x_rich - x_ref) / max(np.linalg.norm(x_ref), 1e-12)
    print("richardson: it=%d relres=%.2e relerr=%.2e" % (it_rich, relres_rich, err_rich))
