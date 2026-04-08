"""
EFGP + EigenPro weight-space solver skeleton.
"""

from .kernels import KernelSpec
from .efgp_solver import EFGPSolver, PrecomputeState
from .model import (
    SolverConfig,
    FitDiagnostics,
    EFGPModel,
    EFGPRegressor,
    precompute_state,
    make_weight_space_operator,
    solve_weight_space,
    predict_from_beta,
)
from .benchmark import (
    BENCHMARK_COLUMNS,
    run_case,
    benchmark_grid,
    expand_grid,
    write_csv,
)

__all__ = [
    "KernelSpec",
    "EFGPSolver",
    "PrecomputeState",
    "SolverConfig",
    "FitDiagnostics",
    "EFGPModel",
    "EFGPRegressor",
    "precompute_state",
    "make_weight_space_operator",
    "solve_weight_space",
    "predict_from_beta",
    "BENCHMARK_COLUMNS",
    "run_case",
    "benchmark_grid",
    "expand_grid",
    "write_csv",
]
