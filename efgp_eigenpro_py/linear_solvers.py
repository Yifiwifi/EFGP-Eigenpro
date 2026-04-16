from typing import Callable, Iterable, List, Optional, Tuple

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
    *,
    relres_check_every: int = 1,
) -> Tuple[np.ndarray, int, float]:
    """
    Preconditioned Richardson iteration: x <- x - eta * P r, r <- r - eta * A P r.

    Convergence needs 0 < eta < 2 / lambda_max(PA) (roughly, when PA is SPD-like);
    a fixed eta=1.0 is often suboptimal or unstable — use ``tune_richardson_eta`` or
    set ``eta`` from a spectral estimate for fair comparisons.

    relres_check_every: if > 1, only compute ||r||/||b|| every k steps (faster for
    timing); the returned relres is always computed on the final iterate.
    """
    if relres_check_every < 1:
        raise ValueError("relres_check_every must be >= 1")

    norm_b = max(float(np.linalg.norm(b)), 1e-12)
    x = x0.copy()
    r = matvec(x) - b
    relres = float(np.linalg.norm(r) / norm_b)
    if relres <= tol or maxiter <= 0:
        return x, 0, relres

    for it in range(maxiter):
        z = precond(r) if precond is not None else r
        if not np.all(np.isfinite(z)):
            relres = float("inf")
            return x, it, relres
        Az = matvec(z)
        if not np.all(np.isfinite(Az)):
            relres = float("inf")
            return x, it + 1, relres
        x = x - eta * z
        r = r - eta * Az
        if not np.all(np.isfinite(r)):
            relres = float("inf")
            return x, it + 1, relres
        step = it + 1
        if step % relres_check_every == 0 or step == maxiter:
            relres = float(np.linalg.norm(r) / norm_b)
            if relres <= tol:
                return x, step, relres
    relres = float(np.linalg.norm(r) / norm_b)
    return x, maxiter, relres


def tune_richardson_eta(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: np.ndarray,
    etas: Iterable[float],
    tol: float,
    maxiter: int,
    precond: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    *,
    pilot_iters: Optional[int] = None,
    relres_check_every: int = 1,
) -> Tuple[float, List[Tuple[float, int, float]]]:
    """
    Pilot each candidate eta with the same iteration cap and pick the one with the
    lowest relative residual (tie-break: fewer iterations).

    If pilot_iters is set, each candidate is run for at most that many steps instead
    of maxiter (cheap screening before a full solve).
    """
    etas_list: List[float] = list(etas)
    if not etas_list:
        raise ValueError("etas must be non-empty")

    cap = maxiter if pilot_iters is None else min(int(pilot_iters), maxiter)
    records: List[Tuple[float, int, float]] = []
    best_eta = etas_list[0]
    best_rel = float("inf")
    best_it = cap + 1

    for eta in etas_list:
        _x, it, rr = richardson(
            matvec,
            b,
            x0,
            eta,
            tol,
            cap,
            precond=precond,
            relres_check_every=relres_check_every,
        )
        records.append((float(eta), int(it), float(rr)))
        if rr < best_rel or (rr == best_rel and it < best_it):
            best_rel = rr
            best_it = it
            best_eta = float(eta)

    return best_eta, records


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
