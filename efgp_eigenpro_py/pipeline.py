import numpy as np

from .kernels import KernelSpec
from .efgp_solver import EFGPSolver


def run_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel: KernelSpec,
    reg_lambda: float,
    eps: float,
    nufft_tol: float,
    top_q: int | None = None,
    solver_type: str | None = None,
    allow_direct: bool = False,
) -> np.ndarray:
    """
    Minimal pipeline wrapper.
    """
    solver = EFGPSolver(kernel, reg_lambda=reg_lambda, eps=eps, nufft_tol=nufft_tol)
    beta, state = solver.solve(
        x_train,
        y_train,
        top_q=top_q,
        solver_type=solver_type,
        allow_direct=allow_direct,
    )
    return solver.predict(x_test, beta, state)
