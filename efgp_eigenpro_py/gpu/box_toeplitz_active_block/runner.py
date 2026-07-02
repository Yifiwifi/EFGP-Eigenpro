from __future__ import annotations

import time
from typing import Any
from typing import Callable

from ..iterative_solvers import fgmres_solve_gpu, pcg_solve_gpu
from ..v1_ops import apply_A_v1
from .config import BTABConfig
from .preconditioner import (
    apply_box_toeplitz_preconditioner,
    build_box_toeplitz_preconditioner,
)
from .box_eigenpro import (
    apply_box_eigenpro_preconditioner,
    build_box_eigenpro_preconditioner,
)


def _resolve_outer_solver(btab_cfg: BTABConfig, solve_mode: str) -> str:
    mode = str(getattr(btab_cfg, "outer_solver", "auto")).strip().lower()
    if mode not in ("auto", "pcg", "fgmres"):
        raise ValueError(f"unknown outer_solver={btab_cfg.outer_solver!r}; expected 'auto', 'pcg', or 'fgmres'.")
    if mode == "auto":
        return "fgmres" if str(solve_mode).lower() == "inner_pcg" else "pcg"
    return mode


def solve_box_toeplitz_active_block(
    backend: Any,
    data_ctx: Any,
    reg_lambda: float,
    rhs_gpu: Any,
    op_ctx: Any,
    *,
    tol: float,
    maxiter: int,
    btab_cfg: BTABConfig,
    profile_components: bool = True,
    return_precond_data: bool = False,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Any, int, float, dict[str, Any], dict[str, Any]] | tuple[Any, int, float, dict[str, Any], dict[str, Any], Any]:
    t0 = time.perf_counter()
    precond_data = build_box_toeplitz_preconditioner(
        backend,
        data_ctx,
        reg_lambda,
        btab_cfg,
        profile_apply_components=profile_components,
    )
    t1 = time.perf_counter()

    def _matvec(v: Any, out: Any) -> None:
        apply_A_v1(backend, data_ctx, v, float(reg_lambda), op_ctx, out=out)

    def _precond(v: Any, out: Any) -> None:
        apply_box_toeplitz_preconditioner(backend, precond_data, v, out=out)

    outer_solver = _resolve_outer_solver(btab_cfg, str(precond_data.solve_mode))
    if outer_solver == "fgmres":
        beta_gpu, it, relres, stats = fgmres_solve_gpu(
            backend,
            _matvec,
            _precond,
            rhs_gpu,
            op_ctx,
            tol,
            maxiter,
            restart=int(getattr(btab_cfg, "outer_gmres_restart", 50)),
            return_stats=True,
            profile_components=profile_components,
        )
    else:
        beta_gpu, it, relres, stats = pcg_solve_gpu(
            backend,
            _matvec,
            _precond,
            rhs_gpu,
            op_ctx,
            tol,
            maxiter,
            return_stats=True,
            profile_components=profile_components,
            trace_callback=trace_callback,
        )
    setup_diag = dict(precond_data.diagnostics)
    setup_diag.update(
        {
            "outer_solver": outer_solver,
            "outer_gmres_restart": int(getattr(btab_cfg, "outer_gmres_restart", 50)),
            "precond_apply_calls": int(precond_data.runtime_stats.get("precond_apply_calls", 0)),
            "inner_total_iters": int(precond_data.runtime_stats.get("inner_total_iters", 0)),
            "inner_total_matvec": int(precond_data.runtime_stats.get("inner_total_matvec", 0)),
            "inner_total_precond": int(precond_data.runtime_stats.get("inner_total_precond", 0)),
            "inner_last_relres": float(precond_data.runtime_stats.get("inner_last_relres", float("nan"))),
            "inner_max_iters": int(precond_data.runtime_stats.get("inner_max_iters", 0)),
            "time_diag_total": float(precond_data.runtime_stats.get("time_diag_total", 0.0)),
            "time_box_gather_total": float(precond_data.runtime_stats.get("time_box_gather_total", 0.0)),
            "time_box_solve_total": float(precond_data.runtime_stats.get("time_box_solve_total", 0.0)),
            "time_box_scatter_total": float(precond_data.runtime_stats.get("time_box_scatter_total", 0.0)),
        }
    )
    n_apply = max(int(precond_data.runtime_stats.get("precond_apply_calls", 0)), 1)
    setup_diag["time_diag_avg"] = float(setup_diag["time_diag_total"] / n_apply)
    setup_diag["time_box_gather_avg"] = float(setup_diag["time_box_gather_total"] / n_apply)
    setup_diag["time_box_solve_avg"] = float(setup_diag["time_box_solve_total"] / n_apply)
    setup_diag["time_box_scatter_avg"] = float(setup_diag["time_box_scatter_total"] / n_apply)
    setup_diag["outer_status"] = str(stats.get("status", "ok"))
    setup_diag["time_precond_build"] = float(t1 - t0)
    if return_precond_data:
        return beta_gpu, int(it), float(relres), stats, setup_diag, precond_data
    return beta_gpu, int(it), float(relres), stats, setup_diag


def solve_box_eigenpro_active_block(
    backend: Any,
    data_ctx: Any,
    reg_lambda: float,
    rhs_gpu: Any,
    op_ctx: Any,
    *,
    tol: float,
    maxiter: int,
    btab_cfg: BTABConfig,
    profile_components: bool = True,
    return_precond_data: bool = False,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Any, int, float, dict[str, Any], dict[str, Any]] | tuple[Any, int, float, dict[str, Any], dict[str, Any], Any]:
    t0 = time.perf_counter()
    precond_data = build_box_eigenpro_preconditioner(
        backend,
        data_ctx,
        reg_lambda,
        btab_cfg,
        q=int(getattr(btab_cfg, "eig_q", 64)),
        profile_apply_components=profile_components,
    )
    t1 = time.perf_counter()

    def _matvec(v: Any, out: Any) -> None:
        apply_A_v1(backend, data_ctx, v, float(reg_lambda), op_ctx, out=out)

    def _precond(v: Any, out: Any) -> None:
        apply_box_eigenpro_preconditioner(backend, precond_data, v, out=out)

    outer_solver = _resolve_outer_solver(btab_cfg, str(precond_data.solve_mode))
    if outer_solver == "fgmres":
        beta_gpu, it, relres, stats = fgmres_solve_gpu(
            backend,
            _matvec,
            _precond,
            rhs_gpu,
            op_ctx,
            tol,
            maxiter,
            restart=int(getattr(btab_cfg, "outer_gmres_restart", 50)),
            return_stats=True,
            profile_components=profile_components,
        )
    else:
        beta_gpu, it, relres, stats = pcg_solve_gpu(
            backend,
            _matvec,
            _precond,
            rhs_gpu,
            op_ctx,
            tol,
            maxiter,
            return_stats=True,
            profile_components=profile_components,
            trace_callback=trace_callback,
        )
    setup_diag = dict(precond_data.diagnostics)
    setup_diag.update(
        {
            "outer_solver": outer_solver,
            "outer_gmres_restart": int(getattr(btab_cfg, "outer_gmres_restart", 50)),
            "precond_apply_calls": int(precond_data.runtime_stats.get("precond_apply_calls", 0)),
            "time_diag_total": float(precond_data.runtime_stats.get("time_diag_total", 0.0)),
            "time_box_gather_total": float(precond_data.runtime_stats.get("time_box_gather_total", 0.0)),
            "time_box_solve_total": float(precond_data.runtime_stats.get("time_box_solve_total", 0.0)),
            "time_box_scatter_total": float(precond_data.runtime_stats.get("time_box_scatter_total", 0.0)),
            "inner_total_iters": 0,
            "inner_total_matvec": 0,
            "inner_total_precond": 0,
            "inner_last_relres": float("nan"),
            "inner_max_iters": 0,
        }
    )
    n_apply = max(int(precond_data.runtime_stats.get("precond_apply_calls", 0)), 1)
    setup_diag["time_diag_avg"] = float(setup_diag["time_diag_total"] / n_apply)
    setup_diag["time_box_gather_avg"] = float(setup_diag["time_box_gather_total"] / n_apply)
    setup_diag["time_box_solve_avg"] = float(setup_diag["time_box_solve_total"] / n_apply)
    setup_diag["time_box_scatter_avg"] = float(setup_diag["time_box_scatter_total"] / n_apply)
    setup_diag["outer_status"] = str(stats.get("status", "ok"))
    setup_diag["time_precond_build"] = float(t1 - t0)
    if return_precond_data:
        return beta_gpu, int(it), float(relres), stats, setup_diag, precond_data
    return beta_gpu, int(it), float(relres), stats, setup_diag
