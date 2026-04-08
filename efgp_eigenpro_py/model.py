from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import pickle
import time

import numpy as np

from .kernels import KernelSpec
from .efgp_solver import EFGPSolver, PrecomputeState
from .eigenspace import EigenPairs, estimate_top_eigenspace
from .eigenpro_precond import build_preconditioner
from .linear_solvers import pcg as pcg_solve, richardson as richardson_solve
from .discretization import generate_multi_index


@dataclass
class SolverConfig:
    reg_lambda: float = 1e-3
    eps: float = 1e-4
    nufft_tol: float = 1e-10
    solver: str = "pcg"          # "pcg", "richardson", "direct"
    tol: float = 1e-10
    maxiter: int = 5000
    eta: Optional[float] = None  # for Richardson
    preconditioner: str = "none" # "none", "eigenpro"
    top_q: int = 0
    eig_backend: str = "auto"    # "auto", "dense", "eigsh"
    eig_method: str = "subspace_iter"  # "eigsh", "lobpcg", "subspace_iter"
    eig_tol: float = 1e-2
    eig_maxiter: int = 20
    eig_block_size: Optional[int] = None
    eig_oversample: int = 2
    verbose: bool = False
    store_history: bool = False
    allow_direct: bool = False   # explicit opt-in for direct solve
    direct_max_m: int = 4000     # only use direct when M <= direct_max_m


@dataclass
class FitDiagnostics:
    solver: str
    preconditioner: str
    top_q: int
    n_train: int
    dim: int
    mtot: int
    M: int
    n_iter: Optional[int]
    final_relres: Optional[float]
    lambda_max: Optional[float] = None
    lambda_q: Optional[float] = None
    lambda_q_plus_1: Optional[float] = None
    lambda_min: Optional[float] = None
    cond_A: Optional[float] = None
    cond_PA: Optional[float] = None
    time_precompute: Optional[float] = None
    time_eigenspace: Optional[float] = None
    time_precond: Optional[float] = None
    time_solve: Optional[float] = None
    time_predict: Optional[float] = None
    time_total: Optional[float] = None
    residual_history: Optional[np.ndarray] = None


@dataclass
class EFGPModel:
    kernel: KernelSpec
    config: SolverConfig
    state: PrecomputeState
    beta: np.ndarray
    diagnostics: FitDiagnostics

    def predict(self, x: np.ndarray) -> np.ndarray:
        solver = EFGPSolver(
            self.kernel,
            reg_lambda=self.config.reg_lambda,
            eps=self.config.eps,
            nufft_tol=self.config.nufft_tol,
        )
        return solver.predict(x, self.beta, self.state)


class WeightSpaceOperator:
    def __init__(self, solver: EFGPSolver, state: PrecomputeState) -> None:
        self._solver = solver
        self._state = state
        self.shape = (state.rhs.size, state.rhs.size)
        self.dtype = np.result_type(state.rhs.dtype, np.complex128)

    def apply(self, v: np.ndarray) -> np.ndarray:
        return self._solver._apply_A(self._state, v)

    def apply_block(self, v: np.ndarray) -> np.ndarray:
        return self._solver._apply_A_block(self._state, v)


def precompute_state(
    x: np.ndarray,
    y: np.ndarray,
    kernel: KernelSpec,
    eps: float = 1e-4,
    nufft_tol: float = 1e-10,
    reg_lambda: float = 1e-3,
) -> PrecomputeState:
    solver = EFGPSolver(kernel, reg_lambda=reg_lambda, eps=eps, nufft_tol=nufft_tol)
    return solver.precompute(x, y)


def make_weight_space_operator(state: PrecomputeState, solver: EFGPSolver) -> WeightSpaceOperator:
    return WeightSpaceOperator(solver, state)


def _pcg_history(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    tol: float,
    maxiter: int,
    precond: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    store_history: bool = False,
) -> tuple[np.ndarray, int, float, Optional[np.ndarray]]:
    try:
        from scipy.sparse.linalg import LinearOperator, cg  # type: ignore
    except Exception as exc:
        x, it, relres = pcg_solve(matvec, b, tol=tol, maxiter=maxiter, precond=precond)
        return x, it, relres, None

    n = b.size
    dtype = np.result_type(b.dtype, np.complex128)
    A = LinearOperator((n, n), matvec=matvec, dtype=dtype)
    M = None
    if precond is not None:
        M = LinearOperator((n, n), matvec=precond, dtype=dtype)

    it_count = 0
    history = []
    norm_b = max(np.linalg.norm(b), 1e-12)

    def _cb(xk: np.ndarray) -> None:
        nonlocal it_count
        it_count += 1
        if store_history:
            r = matvec(xk) - b
            history.append(np.linalg.norm(r) / norm_b)

    x, _info = cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, M=M, callback=_cb)
    relres = np.linalg.norm(matvec(x) - b) / norm_b
    hist = np.asarray(history) if store_history else None
    return x, it_count, float(relres), hist


def _richardson_history(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    eta: float,
    tol: float,
    maxiter: int,
    precond: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    store_history: bool = False,
) -> tuple[np.ndarray, int, float, Optional[np.ndarray]]:
    x = np.zeros_like(b)
    r = matvec(x) - b
    norm_b = max(np.linalg.norm(b), 1e-12)
    relres = np.linalg.norm(r) / norm_b
    history = [relres] if store_history else None

    if relres <= tol or maxiter <= 0:
        return x, 0, float(relres), np.asarray(history) if store_history else None

    for it in range(maxiter):
        z = precond(r) if precond is not None else r
        Az = matvec(z)
        x = x - eta * z
        r = r - eta * Az
        relres = np.linalg.norm(r) / norm_b
        if store_history:
            history.append(relres)
        if relres <= tol:
            return x, it + 1, float(relres), np.asarray(history) if store_history else None
    return x, maxiter, float(relres), np.asarray(history) if store_history else None


def _dense_phi(solver: EFGPSolver, x: np.ndarray, state: PrecomputeState) -> np.ndarray:
    x = solver._ensure_2d(x)
    tphx = 2.0 * np.pi * state.grid.h * (x - state.x_center)
    m = (state.grid.mtot - 1) // 2
    J = generate_multi_index(m, solver.kernel.dim)
    w = state.weights.reshape(-1)
    phase = np.exp(1j * (tphx @ J.T))
    return phase * w[None, :]


def _dense_A_rhs(solver: EFGPSolver, x: np.ndarray, y: np.ndarray, state: PrecomputeState) -> tuple[np.ndarray, np.ndarray]:
    Phi = _dense_phi(solver, x, state)
    A = Phi.conj().T @ Phi + solver.reg_lambda * np.eye(Phi.shape[1], dtype=complex)
    rhs = Phi.conj().T @ y
    return A, rhs


def _spectral_metrics_from_dense(A: np.ndarray, eigpairs: Optional[EigenPairs], top_q: int) -> dict:
    eigvals = np.linalg.eigvalsh(A)
    lam_max = float(eigvals[-1])
    lam_min = float(eigvals[0])
    out = {
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "cond_A": lam_max / max(lam_min, 1e-30),
        "lambda_q": None,
        "lambda_q_plus_1": None,
        "cond_PA": None,
    }
    if eigpairs is not None and top_q > 0 and eigpairs.values.size > top_q:
        out["lambda_q"] = float(eigpairs.values[top_q - 1])
        out["lambda_q_plus_1"] = float(eigpairs.values[top_q])
        out["cond_PA"] = float(eigpairs.values[top_q]) / max(lam_min, 1e-30)
    return out


def solve_weight_space(
    op: WeightSpaceOperator,
    rhs: np.ndarray,
    solver: str = "pcg",
    tol: float = 1e-10,
    maxiter: int = 5000,
    precond: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    eta: Optional[float] = None,
    store_history: bool = False,
    direct: Optional[tuple[np.ndarray, np.ndarray]] = None,
    allow_direct: bool = False,
) -> tuple[np.ndarray, FitDiagnostics]:
    if solver not in ("pcg", "richardson", "direct"):
        raise ValueError("solver must be 'pcg', 'richardson', or 'direct'.")

    if solver == "direct" and not allow_direct:
        solver = "pcg"
    if solver == "direct":
        if direct is None:
            raise ValueError("direct solver requires dense (A, rhs).")
        A_dense, rhs_dense = direct
        beta = np.linalg.solve(A_dense, rhs_dense)
        diag = FitDiagnostics(
            solver=solver,
            preconditioner="none",
            top_q=0,
            n_train=rhs.size,
            dim=0,
            mtot=0,
            M=rhs.size,
            n_iter=1,
            final_relres=None,
        )
        return beta, diag

    if solver == "pcg":
        beta, it, relres, history = _pcg_history(
            op.apply, rhs, tol=tol, maxiter=maxiter, precond=precond, store_history=store_history
        )
    else:
        eta_use = eta if eta is not None else 0.8
        beta, it, relres, history = _richardson_history(
            op.apply, rhs, eta=eta_use, tol=tol, maxiter=maxiter, precond=precond, store_history=store_history
        )

    diag = FitDiagnostics(
        solver=solver,
        preconditioner="none" if precond is None else "eigenpro",
        top_q=0,
        n_train=rhs.size,
        dim=0,
        mtot=0,
        M=rhs.size,
        n_iter=it,
        final_relres=relres,
        residual_history=history,
    )
    return beta, diag


def predict_from_beta(x: np.ndarray, beta: np.ndarray, state: PrecomputeState, kernel: KernelSpec, config: SolverConfig) -> np.ndarray:
    solver = EFGPSolver(kernel, reg_lambda=config.reg_lambda, eps=config.eps, nufft_tol=config.nufft_tol)
    return solver.predict(x, beta, state)


class EFGPRegressor:
    def __init__(self, kernel: KernelSpec, config: Optional[SolverConfig] = None, **kwargs) -> None:
        self.kernel = kernel
        if config is None:
            self.config = SolverConfig(**kwargs)
        else:
            self.config = config
        self.model_: Optional[EFGPModel] = None
        self.beta_: Optional[np.ndarray] = None
        self.state_: Optional[PrecomputeState] = None
        self.diagnostics_: Optional[FitDiagnostics] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EFGPRegressor":
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        solver = EFGPSolver(
            self.kernel,
            reg_lambda=self.config.reg_lambda,
            eps=self.config.eps,
            nufft_tol=self.config.nufft_tol,
        )

        t0 = time.perf_counter()
        state = solver.precompute(x, y)
        t1 = time.perf_counter()

        op = make_weight_space_operator(state, solver)
        matvec = op.apply

        eigpairs = None
        precond = None
        mu = None
        t_eig = 0.0
        t_precond = 0.0
        top_q_used = 0

        M = state.rhs.size
        mtot = state.grid.mtot
        n_train = solver._ensure_2d(x).shape[0]

        if self.config.preconditioner == "eigenpro" and self.config.top_q > 0:
            top_q_used = int(self.config.top_q)
            t_e0 = time.perf_counter()
            use_dense = False
            if self.config.eig_backend == "dense":
                use_dense = True
            elif self.config.eig_backend == "auto" and M <= 2000:
                use_dense = True

            if use_dense:
                if M > 2000:
                    raise ValueError("dense eig_backend requires M <= 2000.")
                A_dense, _rhs = _dense_A_rhs(solver, x, y, state)
                vals, vecs = np.linalg.eigh(A_dense)
                order = np.argsort(vals)[::-1]
                eigpairs = EigenPairs(values=vals[order], vectors=vecs[:, order])
            else:
                eigpairs = estimate_top_eigenspace(
                    matvec,
                    M,
                    top_q=top_q_used,
                    method=self.config.eig_method,
                    tol=self.config.eig_tol,
                    maxiter=self.config.eig_maxiter,
                    matvec_block=op.apply_block if hasattr(op, "apply_block") else None,
                    block_size=self.config.eig_block_size,
                    oversample=self.config.eig_oversample,
                )

            t_e1 = time.perf_counter()
            mu = float(eigpairs.values[top_q_used])
            precond = build_preconditioner(
                eigpairs.values[:top_q_used],
                eigpairs.vectors[:, :top_q_used],
                mu,
            )
            t_e2 = time.perf_counter()
            t_eig = t_e1 - t_e0
            t_precond = t_e2 - t_e1

        t_s0 = time.perf_counter()
        solver_kind = self.config.solver
        if solver_kind == "direct" and not self.config.allow_direct:
            solver_kind = "pcg"
        if solver_kind == "direct":
            if M > self.config.direct_max_m:
                solver_kind = "pcg"
            else:
                A_dense, rhs_dense = _dense_A_rhs(solver, x, y, state)
                beta = np.linalg.solve(A_dense, rhs_dense)
                it = 1
                relres = None
                history = None
        if solver_kind == "pcg":
            beta, it, relres, history = _pcg_history(
                matvec,
                state.rhs,
                tol=self.config.tol,
                maxiter=self.config.maxiter,
                precond=None if precond is None else precond.apply,
                store_history=self.config.store_history,
            )
        elif solver_kind == "richardson":
            eta_use = self.config.eta if self.config.eta is not None else 0.8
            beta, it, relres, history = _richardson_history(
                matvec,
                state.rhs,
                eta=eta_use,
                tol=self.config.tol,
                maxiter=self.config.maxiter,
                precond=None if precond is None else precond.apply,
                store_history=self.config.store_history,
            )
        t_s1 = time.perf_counter()

        spec = {
            "lambda_max": None,
            "lambda_q": None,
            "lambda_q_plus_1": None,
            "lambda_min": None,
            "cond_A": None,
            "cond_PA": None,
        }

        if self.config.eig_backend == "dense" or (self.config.eig_backend == "auto" and M <= 2000):
            if M > 2000:
                raise ValueError("dense eig_backend requires M <= 2000.")
            A_dense, _rhs = _dense_A_rhs(solver, x, y, state)
            spec = _spectral_metrics_from_dense(A_dense, eigpairs, top_q_used)
        elif eigpairs is not None and eigpairs.values.size > 0:
            spec["lambda_max"] = float(eigpairs.values[0])
            if top_q_used > 0 and eigpairs.values.size > top_q_used:
                spec["lambda_q"] = float(eigpairs.values[top_q_used - 1])
                spec["lambda_q_plus_1"] = float(eigpairs.values[top_q_used])

        diag = FitDiagnostics(
            solver=solver_kind,
            preconditioner=self.config.preconditioner,
            top_q=top_q_used,
            n_train=n_train,
            dim=self.kernel.dim,
            mtot=mtot,
            M=M,
            n_iter=it,
            final_relres=None if relres is None else float(relres),
            lambda_max=spec["lambda_max"],
            lambda_q=spec["lambda_q"],
            lambda_q_plus_1=spec["lambda_q_plus_1"],
            lambda_min=spec["lambda_min"],
            cond_A=spec["cond_A"],
            cond_PA=spec["cond_PA"],
            time_precompute=t1 - t0,
            time_eigenspace=t_eig,
            time_precond=t_precond,
            time_solve=t_s1 - t_s0,
            time_predict=None,
            time_total=(t1 - t0) + t_eig + t_precond + (t_s1 - t_s0),
            residual_history=history,
        )

        self.model_ = EFGPModel(self.kernel, self.config, state, beta, diag)
        self.beta_ = beta
        self.state_ = state
        self.diagnostics_ = diag
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        t0 = time.perf_counter()
        yhat = self.model_.predict(x)
        t1 = time.perf_counter()
        if self.diagnostics_ is not None:
            self.diagnostics_.time_predict = t1 - t0
            if self.diagnostics_.time_total is not None:
                self.diagnostics_.time_total = self.diagnostics_.time_total + (t1 - t0)
        return yhat

    def fit_predict(self, x: np.ndarray, y: np.ndarray, x_test: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(x, y)
        x_eval = x if x_test is None else x_test
        return self.predict(x_eval)

    def score(self, x: np.ndarray, y: np.ndarray, metric: str = "rmse") -> float:
        y = np.asarray(y)
        yhat = self.predict(x)
        if metric == "rmse":
            return float(np.sqrt(np.mean((yhat - y) ** 2)))
        if metric == "mae":
            return float(np.mean(np.abs(yhat - y)))
        if metric == "r2":
            denom = np.sum((y - np.mean(y)) ** 2)
            if denom <= 0:
                return float("nan")
            return float(1.0 - np.sum((y - yhat) ** 2) / denom)
        raise ValueError("metric must be one of: rmse, mae, r2")

    def summary(self) -> None:
        if self.diagnostics_ is None:
            print("no diagnostics; call fit() first")
            return
        d = self.diagnostics_
        print("N=", d.n_train, "dim=", d.dim, "M=", d.M, "mtot=", d.mtot)
        print("solver=", d.solver, "preconditioner=", d.preconditioner, "top_q=", d.top_q)
        print("n_iter=", d.n_iter, "final_relres=", d.final_relres)
        print(
            "time_precompute=", d.time_precompute,
            "time_eigenspace=", d.time_eigenspace,
            "time_precond=", d.time_precond,
            "time_solve=", d.time_solve,
            "time_predict=", d.time_predict,
            "time_total=", d.time_total,
        )

    def get_diagnostics(self) -> Optional[FitDiagnostics]:
        return self.diagnostics_

    def save(self, path: str) -> None:
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        with open(path, "wb") as f:
            pickle.dump(self.model_, f)

    @staticmethod
    def load(path: str) -> EFGPModel:
        with open(path, "rb") as f:
            return pickle.load(f)
