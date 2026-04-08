from __future__ import annotations

import csv
import itertools
import math
import os
import platform
import time
from typing import Callable, Iterable, Optional

import numpy as np

from .efgp_solver import EFGPSolver, PrecomputeState
from .discretization import choose_grid_params, basis_weights, generate_multi_index
from .eigenspace import EigenPairs, estimate_top_eigenspace
from .eigenpro_precond import build_preconditioner
from .kernels import KernelSpec, make_squared_exponential, make_matern
from .toeplitz import make_toeplitz_workspace


BENCHMARK_COLUMNS = [
    "task_id",
    "method",
    "dim",
    "N",
    "kernel_family",
    "lengthscale",
    "nu",
    "reg_lambda",
    "eps",
    "top_q",
    "mtot",
    "M",
    "train_RMSE",
    "test_RMSE",
    "r2",
    "mae",
    "pred_relerr_KRR",
    "rhs_relerr",
    "applyA_relerr",
    "beta_relerr_denseW",
    "pred_relerr_denseW",
    "eigval_relerr",
    "proj_err_fro",
    "sin_theta_max",
    "plain_it",
    "prec_it",
    "plain_final_relres",
    "prec_final_relres",
    "lambda_max",
    "lambda_q",
    "lambda_q_plus_1",
    "lambda_min",
    "cond_A",
    "cond_PA",
    "time_precompute_total",
    "time_nufft_v",
    "time_fft_embed",
    "time_nufft_rhs",
    "time_eigenspace",
    "time_precond_build",
    "time_solve_plain",
    "time_solve_prec",
    "time_predict_plain",
    "time_predict_prec",
    "time_total_plain",
    "time_total_prec",
    "precompute_throughput",
    "solve_speedup",
    "total_speedup",
    "it_speedup",
    "time_matvec_avg",
    "time_solve_baseline",
    "time_predict_baseline",
    "time_feature_build",
    "time_factor_solve",
    "bytes_per_vector",
    "peak_memory_mb",
    "cpu_name",
    "num_threads",
]


def compute_rmse(yhat: np.ndarray, ytrue: np.ndarray) -> float:
    yhat = np.asarray(yhat)
    ytrue = np.asarray(ytrue)
    return float(np.sqrt(np.mean((yhat - ytrue) ** 2)))


def compute_mae(yhat: np.ndarray, ytrue: np.ndarray) -> float:
    yhat = np.asarray(yhat)
    ytrue = np.asarray(ytrue)
    return float(np.mean(np.abs(yhat - ytrue)))


def compute_r2(yhat: np.ndarray, ytrue: np.ndarray) -> float:
    yhat = np.asarray(yhat)
    ytrue = np.asarray(ytrue)
    denom = np.sum((ytrue - np.mean(ytrue)) ** 2)
    if denom <= 0:
        return float("nan")
    num = np.sum((ytrue - yhat) ** 2)
    return float(1.0 - num / denom)


def get_cpu_name() -> str:
    name = platform.processor()
    if not name:
        name = os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    return name


def get_num_threads() -> int:
    return os.cpu_count() or 1


def get_rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def bytes_per_vector_m(M: int) -> int:
    return int(M) * 16


def bytes_per_vector_n(N: int) -> int:
    return int(N) * 8


def true_func_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    return np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x)


def true_func_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return (
        np.sin(2 * np.pi * x[:, 0])
        + 0.5 * np.cos(2 * np.pi * x[:, 1])
        + 0.2 * np.sin(2 * np.pi * (x[:, 0] + x[:, 1]))
    )


def true_func_3d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return (
        np.sin(2 * np.pi * x[:, 0])
        + 0.3 * np.cos(2 * np.pi * x[:, 1])
        + 0.2 * np.sin(2 * np.pi * x[:, 2])
    )


def rough_func_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    return np.sin(2 * np.pi * x) + 0.3 * (x > 0.55)


def rough_func_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.sin(2 * np.pi * x[:, 0]) + 0.4 * (x[:, 0] + x[:, 1] > 1.0)


def make_dataset(dim: int, n: int, func: Callable[[np.ndarray], np.ndarray], noise: float = 0.02, seed: int = 0):
    rng = np.random.default_rng(seed)
    if dim == 1:
        x = np.sort(rng.uniform(0.0, 1.0, size=n))[:, None]
    else:
        x = rng.uniform(0.0, 1.0, size=(n, dim))
    f = func(x)
    y = f + noise * rng.standard_normal(n)
    return x, y


def make_test_set(dim: int, n: int, func: Callable[[np.ndarray], np.ndarray], seed: int = 0):
    rng = np.random.default_rng(seed)
    if dim == 1:
        x = np.linspace(0.0, 1.0, n)[:, None]
    else:
        x = rng.uniform(0.0, 1.0, size=(n, dim))
    y = func(x)
    return x, y


def make_clustered_2d(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n1 = int(0.6 * n)
    n2 = n - n1
    x1 = rng.normal(loc=(0.3, 0.3), scale=0.05, size=(n1, 2))
    x2 = rng.normal(loc=(0.75, 0.65), scale=0.08, size=(n2, 2))
    x = np.vstack([x1, x2])
    x = np.clip(x, 0.0, 1.0)
    return x


def precompute_with_timing(solver: EFGPSolver, x: np.ndarray, y: np.ndarray) -> tuple[PrecomputeState, dict]:
    t0 = time.perf_counter()
    x = solver._ensure_2d(x)
    L = np.max(np.max(x, axis=0) - np.min(x, axis=0))
    grid = choose_grid_params(solver.kernel, solver.eps, L)
    tphx, x_center = solver._center_and_scale(x, grid.h)
    weights = np.ascontiguousarray(basis_weights(solver.kernel, grid.xis, grid.h).reshape(-1))

    dim = solver.kernel.dim
    if dim == 1:
        c = np.ones(x.shape[0], dtype=complex)
        t1 = time.perf_counter()
        from .nufft_ops import nufft1d1

        XtXcol = nufft1d1(tphx[:, 0], c, 2 * grid.mtot - 1, solver.nufft_tol, -1)
        t2 = time.perf_counter()
        Gf = np.fft.fftn(XtXcol)
        t3 = time.perf_counter()
        rhs = nufft1d1(tphx[:, 0], y, grid.mtot, solver.nufft_tol, -1)
        t4 = time.perf_counter()
        rhs = np.multiply(weights, rhs)
    elif dim == 2:
        c = np.ones(x.shape[0], dtype=complex)
        t1 = time.perf_counter()
        from .nufft_ops import nufft2d1

        XtXcol = nufft2d1(
            tphx[:, 0], tphx[:, 1], c, 2 * grid.mtot - 1, 2 * grid.mtot - 1, solver.nufft_tol, -1
        )
        t2 = time.perf_counter()
        Gf = np.fft.fftn(XtXcol)
        t3 = time.perf_counter()
        rhs = nufft2d1(tphx[:, 0], tphx[:, 1], y, grid.mtot, grid.mtot, solver.nufft_tol, -1)
        t4 = time.perf_counter()
        rhs = np.ascontiguousarray(rhs.reshape(-1))
        np.multiply(weights, rhs, out=rhs)
    elif dim == 3:
        c = np.ones(x.shape[0], dtype=complex)
        t1 = time.perf_counter()
        from .nufft_ops import nufft3d1

        XtXcol = nufft3d1(
            tphx[:, 0],
            tphx[:, 1],
            tphx[:, 2],
            c,
            2 * grid.mtot - 1,
            2 * grid.mtot - 1,
            2 * grid.mtot - 1,
            solver.nufft_tol,
            -1,
        )
        t2 = time.perf_counter()
        Gf = np.fft.fftn(XtXcol)
        t3 = time.perf_counter()
        rhs = nufft3d1(
            tphx[:, 0], tphx[:, 1], tphx[:, 2], y, grid.mtot, grid.mtot, grid.mtot, solver.nufft_tol, -1
        )
        t4 = time.perf_counter()
        rhs = np.ascontiguousarray(rhs.reshape(-1))
        np.multiply(weights, rhs, out=rhs)
    else:
        c = np.ones(x.shape[0], dtype=complex)
        ms_xtx = 2 * grid.mtot - 1
        t1 = time.perf_counter()
        from .nufft_ops import nufftnd1

        XtXcol = nufftnd1(tphx, c, ms_xtx, solver.nufft_tol, -1)
        XtXcol = XtXcol.reshape((ms_xtx,) * dim)
        t2 = time.perf_counter()
        Gf = np.fft.fftn(XtXcol)
        t3 = time.perf_counter()
        rhs = nufftnd1(tphx, y, grid.mtot, solver.nufft_tol, -1)
        t4 = time.perf_counter()
        rhs = np.ascontiguousarray(rhs.reshape(-1))
        np.multiply(weights, rhs, out=rhs)

    Gf = np.ascontiguousarray(Gf)
    rhs = np.ascontiguousarray(rhs.reshape(-1))
    toeplitz_ws = make_toeplitz_workspace(Gf, grid.mtot, dim, dtype=Gf.dtype)
    apply_w = np.empty(rhs.size, dtype=np.result_type(Gf.dtype, np.complex128))
    state = PrecomputeState(
        grid=grid,
        weights=weights,
        Gf=Gf,
        rhs=rhs,
        x_center=x_center,
        toeplitz_ws=toeplitz_ws,
        apply_w=apply_w,
    )
    t5 = time.perf_counter()
    times = {
        "time_nufft_v": t2 - t1,
        "time_fft_embed": t3 - t2,
        "time_nufft_rhs": t4 - t3,
        "time_precompute_total": t5 - t0,
    }
    return state, times


def build_precond_from_state(
    solver: EFGPSolver,
    state: PrecomputeState,
    top_q: int,
    *,
    eig_method: str = "subspace_iter",
    eig_tol: float = 1e-2,
    eig_maxiter: int | None = 20,
    eig_block_size: int | None = None,
    eig_oversample: int = 2,
):
    if top_q is None or top_q <= 0:
        return None, None, None, 0.0, 0.0, 0
    size = int(state.rhs.size)
    safe_max = size - 3
    if safe_max <= 0:
        return None, None, None, 0.0, 0.0, 0
    top_q_used = min(int(top_q), safe_max)
    if top_q_used <= 0:
        return None, None, None, 0.0, 0.0, 0
    t0 = time.perf_counter()
    eigpairs = estimate_top_eigenspace(
        lambda v: solver._apply_A(state, v),
        size=state.rhs.size,
        top_q=top_q_used,
        method=eig_method,
        tol=eig_tol,
        maxiter=eig_maxiter,
        matvec_block=lambda V: solver._apply_A_block(state, V),
        block_size=eig_block_size,
        oversample=eig_oversample,
    )
    t1 = time.perf_counter()
    mu = float(eigpairs.values[top_q_used])
    precond = build_preconditioner(eigpairs.values[:top_q_used], eigpairs.vectors[:, :top_q_used], mu)
    t2 = time.perf_counter()
    return precond, eigpairs, mu, (t1 - t0), (t2 - t1), top_q_used


def build_dense_phi(solver: EFGPSolver, x: np.ndarray, state: PrecomputeState) -> np.ndarray:
    x = solver._ensure_2d(x)
    tphx = 2.0 * np.pi * state.grid.h * (x - state.x_center)
    m = (state.grid.mtot - 1) // 2
    J = generate_multi_index(m, solver.kernel.dim)
    w = state.weights.reshape(-1)
    phase = np.exp(1j * (tphx @ J.T))
    return phase * w[None, :]


def build_dense_A_rhs(solver: EFGPSolver, x: np.ndarray, y: np.ndarray, state: PrecomputeState):
    Phi = build_dense_phi(solver, x, state)
    A = Phi.conj().T @ Phi + solver.reg_lambda * np.eye(Phi.shape[1], dtype=complex)
    rhs = Phi.conj().T @ y
    return Phi, A, rhs


def spectral_metrics_dense(A_dense: np.ndarray, eigpairs: Optional[EigenPairs], top_q: int):
    eigvals = np.linalg.eigvalsh(A_dense)
    lam_max = float(eigvals[-1])
    lam_min = float(eigvals[0])
    cond_A = lam_max / max(lam_min, 1e-30)
    lam_q = None
    lam_q_plus_1 = None
    cond_PA = None
    if eigpairs is not None and top_q and top_q > 0 and eigpairs.values.size > top_q:
        lam_q = float(eigpairs.values[top_q - 1])
        lam_q_plus_1 = float(eigpairs.values[top_q])
        cond_PA = lam_q_plus_1 / max(lam_min, 1e-30)
    return {
        "lambda_max": lam_max,
        "lambda_q": lam_q,
        "lambda_q_plus_1": lam_q_plus_1,
        "lambda_min": lam_min,
        "cond_A": cond_A,
        "cond_PA": cond_PA,
    }


def eigenspace_estimation_metrics(A_dense: np.ndarray, solver: EFGPSolver, state: PrecomputeState, top_q: int):
    evals, evecs = np.linalg.eigh(A_dense)
    evals_desc = evals[::-1]
    evecs_desc = evecs[:, ::-1]

    eigpairs = estimate_top_eigenspace(lambda v: solver._apply_A(state, v), size=state.rhs.size, top_q=top_q)

    true_vals = evals_desc[: top_q + 1]
    est_vals = eigpairs.values[: top_q + 1]
    rel_err = np.abs(est_vals - true_vals) / np.maximum(np.abs(true_vals), 1e-15)

    U_true = evecs_desc[:, :top_q]
    U_est = eigpairs.vectors[:, :top_q]
    proj_true = U_true @ U_true.conj().T
    proj_est = U_est @ U_est.conj().T
    proj_err_fro = np.linalg.norm(proj_est - proj_true, ord="fro")

    s = np.linalg.svd(U_true.conj().T @ U_est, compute_uv=False)
    sin_theta = float(np.sqrt(max(0.0, 1.0 - float(np.min(s)) ** 2)))

    return {
        "eigval_relerr": rel_err,
        "proj_err_fro": proj_err_fro,
        "sin_theta_max": sin_theta,
    }


def kernel_matrix_dense(x: np.ndarray, kernel: KernelSpec) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    diff = x[:, None, :] - x[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    return kernel.k(r)


def kernel_matvec_block(x: np.ndarray, v: np.ndarray, kernel: KernelSpec, block_size: int = 256) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    n = x.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(0, n, block_size):
        j = min(i + block_size, n)
        xb = x[i:j]
        diff = xb[:, None, :] - x[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        Kb = kernel.k(r)
        out[i:j] = Kb @ v
    return out


def kernel_predict_block(
    x_train: np.ndarray, alpha: np.ndarray, x_eval: np.ndarray, kernel: KernelSpec, block_size: int = 256
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    m = x_eval.shape[0]
    out = np.zeros(m, dtype=float)
    for i in range(0, m, block_size):
        j = min(i + block_size, m)
        xb = x_eval[i:j]
        diff = xb[:, None, :] - x_train[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        Kb = kernel.k(r)
        out[i:j] = Kb @ alpha
    return out


def cg_basic(matvec: Callable[[np.ndarray], np.ndarray], b: np.ndarray, tol: float = 1e-6, maxiter: int = 20000):
    b = np.asarray(b)
    x = np.zeros_like(b)
    r = b - matvec(x)
    p = r.copy()
    rsold = np.vdot(r, r).real
    norm_b = max(np.linalg.norm(b), 1e-15)
    it = 0
    for it in range(1, maxiter + 1):
        Ap = matvec(p)
        denom = np.vdot(p, Ap).real
        alpha = rsold / max(denom, 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.vdot(r, r).real
        if math.sqrt(rsnew) / norm_b < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    relres = np.linalg.norm(b - matvec(x)) / norm_b
    return x, it, float(relres)


def kernel_cg_solve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    kernel: KernelSpec,
    reg_lambda: float,
    tol: float = 1e-6,
    maxiter: int = 20000,
    block_size: int = 256,
):
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    matvec_time = 0.0
    matvec_count = 0

    def matvec(v: np.ndarray) -> np.ndarray:
        nonlocal matvec_time, matvec_count
        t0 = time.perf_counter()
        Kv = kernel_matvec_block(x_train, v, kernel, block_size=block_size)
        t1 = time.perf_counter()
        matvec_time += t1 - t0
        matvec_count += 1
        return Kv + reg_lambda * v

    t0 = time.perf_counter()
    it = 0
    try:
        from scipy.sparse.linalg import LinearOperator, cg  # type: ignore

        n = y_train.size
        Aop = LinearOperator((n, n), matvec=matvec, dtype=float)

        def cb(_xk):
            nonlocal it
            it += 1

        x, _info = cg(Aop, y_train, rtol=tol, atol=0.0, maxiter=maxiter, callback=cb)
    except Exception:
        x, it, _ = cg_basic(matvec, y_train, tol=tol, maxiter=maxiter)
    t1 = time.perf_counter()

    relres = np.linalg.norm(matvec(x) - y_train) / max(np.linalg.norm(y_train), 1e-15)
    time_solve = t1 - t0
    time_matvec_avg = matvec_time / max(matvec_count, 1)
    return x, it, float(relres), time_matvec_avg, time_solve


def nystrom_krr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel: KernelSpec,
    reg_lambda: float,
    m: int,
    seed: int = 0,
    block_size: int = 256,
    max_entries: int = 20000000,
):
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    n = x_train.shape[0]

    if n * m > max_entries:
        return None

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    x_land = x_train[idx]

    t0 = time.perf_counter()
    K_mm = kernel_matrix_dense(x_land, kernel)
    jitter = 1e-10 * np.eye(m)
    L = np.linalg.cholesky(K_mm + jitter)

    K_nm = np.zeros((n, m), dtype=float)
    for i in range(0, n, block_size):
        j = min(i + block_size, n)
        xb = x_train[i:j]
        diff = xb[:, None, :] - x_land[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        K_nm[i:j] = kernel.k(r)

    L_inv = np.linalg.solve(L, np.eye(m))
    K_mm_inv_sqrt = L_inv.T
    Phi = K_nm @ K_mm_inv_sqrt
    t1 = time.perf_counter()

    A = Phi.T @ Phi + reg_lambda * np.eye(m)
    b = Phi.T @ y_train
    w = np.linalg.solve(A, b)
    t2 = time.perf_counter()

    K_tm = np.zeros((x_test.shape[0], m), dtype=float)
    for i in range(0, x_test.shape[0], block_size):
        j = min(i + block_size, x_test.shape[0])
        xb = x_test[i:j]
        diff = xb[:, None, :] - x_land[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        K_tm[i:j] = kernel.k(r)

    Phi_test = K_tm @ K_mm_inv_sqrt
    yhat_train = Phi @ w
    yhat = Phi_test @ w
    t3 = time.perf_counter()

    return {
        "time_feature_build": t1 - t0,
        "time_solve": t2 - t1,
        "time_predict": t3 - t2,
        "yhat": yhat,
        "yhat_train": yhat_train,
    }


def pcg_solve(
    matvec,
    b,
    tol: float = 1e-10,
    maxiter: int = 20000,
    precond=None,
    store_history: bool = False,
    return_stats: bool = False,
):
    b = np.asarray(b, dtype=np.complex128)
    it_count = 0
    history = []
    n_matvec = 0
    n_precond = 0
    t_matvec_total = 0.0
    t_precond_total = 0.0

    def _matvec(v):
        nonlocal n_matvec, t_matvec_total
        v = np.asarray(v, dtype=np.complex128)
        t0 = time.perf_counter()
        out = np.asarray(matvec(v), dtype=np.complex128)
        t_matvec_total += time.perf_counter() - t0
        n_matvec += 1
        return out

    def _precond(v):
        nonlocal n_precond, t_precond_total
        v = np.asarray(v, dtype=np.complex128)
        t0 = time.perf_counter()
        out = np.asarray(precond(v), dtype=np.complex128)
        t_precond_total += time.perf_counter() - t0
        n_precond += 1
        return out

    try:
        from scipy.sparse.linalg import LinearOperator, cg  # type: ignore

        n = b.size
        Aop = LinearOperator((n, n), matvec=_matvec, dtype=np.complex128)
        Mop = None if precond is None else LinearOperator((n, n), matvec=_precond, dtype=np.complex128)

        def cb(xk):
            nonlocal it_count
            it_count += 1
            if store_history:
                rk = _matvec(xk) - b
                history.append(np.linalg.norm(rk) / max(np.linalg.norm(b), 1e-15))

        x, _info = cg(Aop, b, rtol=tol, atol=0.0, maxiter=maxiter, M=Mop, callback=cb)
    except Exception:
        x, it_count, _ = cg_basic(_matvec, b, tol=tol, maxiter=maxiter)

    relres = np.linalg.norm(_matvec(x) - b) / max(np.linalg.norm(b), 1e-15)
    if store_history and len(history) == 0:
        history.append(relres)
    hist = np.asarray(history) if store_history else None
    if not return_stats:
        return x, it_count, float(relres), hist
    stats = {
        "n_matvec": int(n_matvec),
        "t_matvec_total": float(t_matvec_total),
        "n_precond": int(n_precond),
        "t_precond_total": float(t_precond_total),
    }
    return x, it_count, float(relres), hist, stats


def richardson_solve(matvec, b, eta, tol: float = 1e-8, maxiter: int = 20000, precond=None, store_history: bool = False):
    b = np.asarray(b, dtype=np.complex128)
    x = np.zeros_like(b)
    r = matvec(x) - b
    norm_b = max(np.linalg.norm(b), 1e-15)
    it = 0
    history = [np.linalg.norm(r) / norm_b] if store_history else None
    for it in range(1, maxiter + 1):
        z = precond(r) if precond is not None else r
        x = x - eta * z
        r = matvec(x) - b
        rel = np.linalg.norm(r) / norm_b
        if store_history:
            history.append(rel)
        if rel < tol:
            break
    relres = np.linalg.norm(matvec(x) - b) / norm_b
    hist = np.asarray(history) if store_history else None
    return x, it, float(relres), hist


def run_dense_krr(
    task_id: str,
    dim: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    kernel: KernelSpec,
    reg_lambda: float,
):
    peak = get_rss_mb()
    t0 = time.perf_counter()
    K = kernel_matrix_dense(x_train, kernel)
    t1 = time.perf_counter()
    alpha = np.linalg.solve(K + reg_lambda * np.eye(K.shape[0]), y_train)
    t2 = time.perf_counter()
    yhat_train = K @ alpha
    yhat_test = kernel_predict_block(x_train, alpha, x_test, kernel)
    t3 = time.perf_counter()

    row = {
        "task_id": task_id,
        "method": "dense_krr",
        "dim": dim,
        "N": x_train.shape[0],
        "kernel_family": kernel.fam,
        "lengthscale": kernel.lengthscale,
        "nu": kernel.nu,
        "reg_lambda": reg_lambda,
        "eps": None,
        "top_q": None,
        "mtot": None,
        "M": None,
        "train_RMSE": compute_rmse(yhat_train, y_train),
        "test_RMSE": compute_rmse(yhat_test, y_test),
        "r2": compute_r2(yhat_test, y_test),
        "mae": compute_mae(yhat_test, y_test),
        "pred_relerr_KRR": None,
        "time_solve_baseline": t2 - t1,
        "time_predict_baseline": t3 - t2,
        "time_feature_build": t1 - t0,
        "time_factor_solve": t2 - t1,
        "bytes_per_vector": bytes_per_vector_n(x_train.shape[0]),
        "peak_memory_mb": peak,
        "cpu_name": get_cpu_name(),
        "num_threads": get_num_threads(),
    }
    return row, yhat_test


def run_kernel_cg(
    task_id: str,
    dim: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    kernel: KernelSpec,
    reg_lambda: float,
):
    peak = get_rss_mb()
    alpha, it, relres, time_matvec_avg, time_solve = kernel_cg_solve(
        x_train, y_train, kernel, reg_lambda, tol=1e-6, maxiter=20000, block_size=256
    )
    t0 = time.perf_counter()
    yhat_train = kernel_predict_block(x_train, alpha, x_train, kernel)
    yhat_test = kernel_predict_block(x_train, alpha, x_test, kernel)
    t1 = time.perf_counter()

    row = {
        "task_id": task_id,
        "method": "kernel_cg",
        "dim": dim,
        "N": x_train.shape[0],
        "kernel_family": kernel.fam,
        "lengthscale": kernel.lengthscale,
        "nu": kernel.nu,
        "reg_lambda": reg_lambda,
        "eps": None,
        "top_q": None,
        "mtot": None,
        "M": None,
        "train_RMSE": compute_rmse(yhat_train, y_train),
        "test_RMSE": compute_rmse(yhat_test, y_test),
        "r2": compute_r2(yhat_test, y_test),
        "mae": compute_mae(yhat_test, y_test),
        "pred_relerr_KRR": None,
        "plain_it": it,
        "plain_final_relres": relres,
        "time_matvec_avg": time_matvec_avg,
        "time_solve_baseline": time_solve,
        "time_predict_baseline": t1 - t0,
        "bytes_per_vector": bytes_per_vector_n(x_train.shape[0]),
        "peak_memory_mb": peak,
        "cpu_name": get_cpu_name(),
        "num_threads": get_num_threads(),
    }
    return row


def run_nystrom(
    task_id: str,
    dim: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    kernel: KernelSpec,
    reg_lambda: float,
    m: int,
    max_entries: int = 20000000,
):
    peak = get_rss_mb()
    res = nystrom_krr(x_train, y_train, x_test, kernel, reg_lambda, m=m, seed=0, max_entries=max_entries)
    if res is None:
        return None
    yhat_test = res["yhat"]
    yhat_train = res.get("yhat_train")
    train_rmse = None if yhat_train is None else compute_rmse(yhat_train, y_train)
    row = {
        "task_id": task_id,
        "method": "nystrom_krr",
        "dim": dim,
        "N": x_train.shape[0],
        "kernel_family": kernel.fam,
        "lengthscale": kernel.lengthscale,
        "nu": kernel.nu,
        "reg_lambda": reg_lambda,
        "eps": None,
        "top_q": None,
        "mtot": None,
        "M": None,
        "train_RMSE": train_rmse,
        "test_RMSE": compute_rmse(yhat_test, y_test),
        "r2": compute_r2(yhat_test, y_test),
        "mae": compute_mae(yhat_test, y_test),
        "pred_relerr_KRR": None,
        "time_solve_baseline": res["time_solve"],
        "time_predict_baseline": res["time_predict"],
        "time_feature_build": res["time_feature_build"],
        "time_factor_solve": res["time_solve"],
        "bytes_per_vector": bytes_per_vector_n(x_train.shape[0]),
        "peak_memory_mb": peak,
        "cpu_name": get_cpu_name(),
        "num_threads": get_num_threads(),
    }
    return row


def run_efgp_case(
    task_id: str,
    dim: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    kernel: KernelSpec,
    reg_lambda: float,
    eps: float,
    nufft_tol: float = 1e-12,
    top_q: int = 0,
    tol: float = 1e-10,
    maxiter: int = 20000,
    solver_type: str = "pcg",
    eta: Optional[float] = None,
    compute_dense: bool = False,
    dense_max_m: int = 5000,
    dense_max_n: int = 3000,
    krr_pred: Optional[np.ndarray] = None,
    store_history: bool = False,
):
    solver = EFGPSolver(kernel, reg_lambda=reg_lambda, eps=eps, nufft_tol=nufft_tol)
    peak = get_rss_mb()

    state, pre_times = precompute_with_timing(solver, x_train, y_train)
    M = state.grid.mtot ** dim
    mtot = state.grid.mtot

    precond = None
    eigpairs = None
    mu = None
    t_eig = 0.0
    t_precond = 0.0
    top_q_used = 0
    if top_q is not None and top_q > 0:
        top_q_used = min(int(top_q), max(1, state.rhs.size - 1))
        t_e0 = time.perf_counter()
        eigpairs = estimate_top_eigenspace(lambda v: solver._apply_A(state, v), size=state.rhs.size, top_q=top_q_used)
        t_e1 = time.perf_counter()
        mu = float(eigpairs.values[top_q_used])
        precond = build_preconditioner(eigpairs.values[:top_q_used], eigpairs.vectors[:, :top_q_used], mu)
        t_e2 = time.perf_counter()
        t_eig = t_e1 - t_e0
        t_precond = t_e2 - t_e1

    matvec = lambda v: solver._apply_A(state, v)

    t_plain0 = time.perf_counter()
    if solver_type == "richardson":
        eta_use = 0.8 if eta is None else float(eta)
        beta_plain, it_plain, rr_plain, hist_plain = richardson_solve(
            matvec, state.rhs, eta=eta_use, tol=tol, maxiter=maxiter, precond=None, store_history=store_history
        )
    else:
        beta_plain, it_plain, rr_plain, hist_plain = pcg_solve(
            matvec, state.rhs, tol=tol, maxiter=maxiter, precond=None, store_history=store_history
        )
    t_plain1 = time.perf_counter()

    t_prec0 = time.perf_counter()
    if solver_type == "richardson":
        eta_use = 0.8 if eta is None else float(eta)
        beta_prec, it_prec, rr_prec, hist_prec = richardson_solve(
            matvec,
            state.rhs,
            eta=eta_use,
            tol=tol,
            maxiter=maxiter,
            precond=None if precond is None else precond.apply,
            store_history=store_history,
        )
    else:
        beta_prec, it_prec, rr_prec, hist_prec = pcg_solve(
            matvec,
            state.rhs,
            tol=tol,
            maxiter=maxiter,
            precond=None if precond is None else precond.apply,
            store_history=store_history,
        )
    t_prec1 = time.perf_counter()

    t_pred0 = time.perf_counter()
    yhat_plain = solver.predict(x_test, beta_plain, state)
    t_pred1 = time.perf_counter()
    t_pred2 = time.perf_counter()
    yhat_prec = solver.predict(x_test, beta_prec, state)
    t_pred3 = time.perf_counter()

    train_rmse_plain = compute_rmse(solver.predict(x_train, beta_plain, state), y_train)
    test_rmse_plain = compute_rmse(yhat_plain, y_test)
    train_rmse_prec = compute_rmse(solver.predict(x_train, beta_prec, state), y_train)
    test_rmse_prec = compute_rmse(yhat_prec, y_test)

    pred_relerr_krr_plain = None
    pred_relerr_krr_prec = None
    if krr_pred is not None:
        pred_relerr_krr_plain = float(np.linalg.norm(yhat_plain - krr_pred) / max(np.linalg.norm(krr_pred), 1e-15))
        pred_relerr_krr_prec = float(np.linalg.norm(yhat_prec - krr_pred) / max(np.linalg.norm(krr_pred), 1e-15))

    rhs_relerr = None
    applyA_relerr = None
    beta_relerr_denseW_plain = None
    beta_relerr_denseW_prec = None
    pred_relerr_denseW_plain = None
    pred_relerr_denseW_prec = None
    eigval_relerr = None
    proj_err_fro = None
    sin_theta_max = None
    spec = {
        "lambda_max": None,
        "lambda_q": None,
        "lambda_q_plus_1": None,
        "lambda_min": None,
        "cond_A": None,
        "cond_PA": None,
    }

    dense_ok = compute_dense and (x_train.shape[0] <= dense_max_n) and (M <= dense_max_m)
    if dense_ok:
        Phi, A_dense, rhs_dense = build_dense_A_rhs(solver, x_train, y_train, state)
        rhs_relerr = np.linalg.norm(state.rhs - rhs_dense) / np.linalg.norm(rhs_dense)
        rng = np.random.default_rng(123)
        v = rng.normal(size=rhs_dense.size) + 1j * rng.normal(size=rhs_dense.size)
        Av_fft = solver._apply_A(state, v)
        Av_dense = A_dense @ v
        applyA_relerr = np.linalg.norm(Av_fft - Av_dense) / np.linalg.norm(Av_dense)
        beta_dense = np.linalg.solve(A_dense, rhs_dense)
        beta_relerr_denseW_plain = np.linalg.norm(beta_plain - beta_dense) / np.linalg.norm(beta_dense)
        beta_relerr_denseW_prec = np.linalg.norm(beta_prec - beta_dense) / np.linalg.norm(beta_dense)
        yhat_dense = np.real(Phi @ beta_dense)
        pred_relerr_denseW_plain = np.linalg.norm(yhat_plain - yhat_dense) / np.linalg.norm(yhat_dense)
        pred_relerr_denseW_prec = np.linalg.norm(yhat_prec - yhat_dense) / np.linalg.norm(yhat_dense)
        spec = spectral_metrics_dense(A_dense, eigpairs, top_q_used)
        if top_q_used is not None and top_q_used > 0:
            est = eigenspace_estimation_metrics(A_dense, solver, state, top_q_used)
            eigval_relerr = est.get("eigval_relerr")
            proj_err_fro = est.get("proj_err_fro")
            sin_theta_max = est.get("sin_theta_max")

    time_solve_plain = t_plain1 - t_plain0
    time_solve_prec = t_prec1 - t_prec0
    time_predict_plain = t_pred1 - t_pred0
    time_predict_prec = t_pred3 - t_pred2
    time_total_plain = pre_times["time_precompute_total"] + time_solve_plain + time_predict_plain
    time_total_prec = pre_times["time_precompute_total"] + t_eig + t_precond + time_solve_prec + time_predict_prec

    precompute_throughput = None
    if pre_times["time_precompute_total"] > 0:
        precompute_throughput = x_train.shape[0] / pre_times["time_precompute_total"]

    solve_speedup = None
    total_speedup = None
    it_speedup = None
    if time_solve_prec > 0:
        solve_speedup = time_solve_plain / time_solve_prec
    if time_total_prec > 0:
        total_speedup = time_total_plain / time_total_prec
    if it_prec > 0:
        it_speedup = it_plain / it_prec

    base = {
        "task_id": task_id,
        "dim": dim,
        "N": x_train.shape[0],
        "kernel_family": kernel.fam,
        "lengthscale": kernel.lengthscale,
        "nu": kernel.nu,
        "reg_lambda": reg_lambda,
        "eps": eps,
        "top_q": top_q_used,
        "mtot": mtot,
        "M": M,
        "lambda_max": spec["lambda_max"],
        "lambda_q": spec["lambda_q"],
        "lambda_q_plus_1": spec["lambda_q_plus_1"],
        "lambda_min": spec["lambda_min"],
        "cond_A": spec["cond_A"],
        "cond_PA": spec["cond_PA"],
        "time_precompute_total": pre_times["time_precompute_total"],
        "time_nufft_v": pre_times["time_nufft_v"],
        "time_fft_embed": pre_times["time_fft_embed"],
        "time_nufft_rhs": pre_times["time_nufft_rhs"],
        "time_eigenspace": t_eig,
        "time_precond_build": t_precond,
        "precompute_throughput": precompute_throughput,
        "bytes_per_vector": bytes_per_vector_m(M),
        "peak_memory_mb": peak,
        "cpu_name": get_cpu_name(),
        "num_threads": get_num_threads(),
        "rhs_relerr": rhs_relerr,
        "applyA_relerr": applyA_relerr,
        "eigval_relerr": eigval_relerr,
        "proj_err_fro": proj_err_fro,
        "sin_theta_max": sin_theta_max,
    }

    row_plain = base.copy()
    row_plain.update(
        {
            "method": f"efgp_plain_{solver_type}",
            "train_RMSE": train_rmse_plain,
            "test_RMSE": test_rmse_plain,
            "r2": compute_r2(yhat_plain, y_test),
            "mae": compute_mae(yhat_plain, y_test),
            "pred_relerr_KRR": pred_relerr_krr_plain,
            "beta_relerr_denseW": beta_relerr_denseW_plain,
            "pred_relerr_denseW": pred_relerr_denseW_plain,
            "plain_it": it_plain,
            "plain_final_relres": rr_plain,
            "time_solve_plain": time_solve_plain,
            "time_predict_plain": time_predict_plain,
            "time_total_plain": time_total_plain,
        }
    )

    row_prec = base.copy()
    row_prec.update(
        {
            "method": f"efgp_prec_{solver_type}",
            "train_RMSE": train_rmse_prec,
            "test_RMSE": test_rmse_prec,
            "r2": compute_r2(yhat_prec, y_test),
            "mae": compute_mae(yhat_prec, y_test),
            "pred_relerr_KRR": pred_relerr_krr_prec,
            "beta_relerr_denseW": beta_relerr_denseW_prec,
            "pred_relerr_denseW": pred_relerr_denseW_prec,
            "prec_it": it_prec,
            "prec_final_relres": rr_prec,
            "time_solve_prec": time_solve_prec,
            "time_predict_prec": time_predict_prec,
            "time_total_prec": time_total_prec,
            "solve_speedup": solve_speedup,
            "total_speedup": total_speedup,
            "it_speedup": it_speedup,
        }
    )

    if store_history:
        row_plain["residual_history"] = hist_plain
        row_prec["residual_history"] = hist_prec

    return [row_plain, row_prec]


def run_case(method: str, **kwargs):
    if method == "dense_krr":
        return [run_dense_krr(**kwargs)[0]]
    if method == "kernel_cg":
        return [run_kernel_cg(**kwargs)]
    if method == "nystrom_krr":
        row = run_nystrom(**kwargs)
        return [] if row is None else [row]
    if method.startswith("efgp_"):
        return run_efgp_case(**kwargs)
    raise ValueError("unknown method: " + str(method))


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], (list, tuple)) else [grid[k]] for k in keys]
    out = []
    for combo in itertools.product(*values):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


def benchmark_grid(cases: Iterable[dict], output_csv: Optional[str] = None, columns: Optional[list] = None):
    rows = []
    cols = BENCHMARK_COLUMNS if columns is None else columns
    for case in cases:
        case_rows = run_case(**case)
        for row in case_rows:
            full = {k: row.get(k) for k in cols}
            rows.append(full)
            if output_csv is not None:
                write_csv([full], output_csv, columns=cols, append=True)
    return rows


def write_csv(rows: list[dict], path: str, columns: Optional[list] = None, append: bool = False):
    cols = BENCHMARK_COLUMNS if columns is None else columns
    file_exists = os.path.exists(path)
    mode = "a" if append else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if not append or not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in cols})
