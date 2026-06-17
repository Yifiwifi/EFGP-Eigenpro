from __future__ import annotations

import os
import time
import traceback
from datetime import datetime

import numpy as np


def _log_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(here, f"btab_validate_{stamp}.txt")


class _Logger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        line = str(msg)
        print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _fmt(d: dict, keys: list[str]) -> str:
    out = []
    for key in keys:
        val = d.get(key)
        if isinstance(val, float):
            out.append(f"{key}={val:.4g}")
        else:
            out.append(f"{key}={val}")
    return ", ".join(out)


def main() -> None:
    log = _Logger(_log_path())
    log(f"# BTAB validation  {datetime.now().isoformat()}")
    log(f"# log file: {log.path}")
    try:
        from ...efgp_solver import EFGPSolver
        from ...kernels import make_squared_exponential
        from ...toy_data import generate_toy_data
        from ..box_toeplitz_active_block.config import BTABConfig
        from ..v3_eigenspace import EigenspaceConfig
        from ..versions import (
            GPURunConfig,
            run_v1_pure_efgp,
            run_v3_full_gpu_eigenspace,
            run_v6_box_toeplitz_active_block,
        )
    except Exception:
        log("IMPORT FAILED:")
        log(traceback.format_exc())
        log.close()
        return

    n_samples = 6000
    lengthscale = 0.02
    reg_lambda = 1e-4
    eps = 1e-6
    nufft_tol = 1e-10
    cg_tol = 1e-6
    maxiter = 1000

    xi, yi = generate_toy_data(n_samples=n_samples, nu=3.0, sigma_eps=0.1, seed=7)
    x = np.asarray(xi, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(yi, dtype=np.float64).reshape(-1)
    kernel = make_squared_exponential(lengthscale=lengthscale, dim=1, variance=1.0)
    solver = EFGPSolver(kernel, reg_lambda=reg_lambda, eps=eps, nufft_tol=nufft_tol)
    cfg = GPURunConfig(reg_lambda=reg_lambda, tol=cg_tol, maxiter=maxiter)

    log("")
    log(
        f"problem: n_samples={n_samples}, l={lengthscale}, reg_lambda={reg_lambda}, "
        f"eps={eps}, cg_tol={cg_tol}"
    )

    try:
        t0 = time.perf_counter()
        out1 = run_v1_pure_efgp(solver, x, y, cfg)
        dt = time.perf_counter() - t0
        d1 = out1.diagnostics
        log("")
        log("[V1 plain CG]")
        log("  " + _fmt(d1, ["cg_iters", "cg_relres", "n_matvec", "time_total"]))
        log(f"  wall={dt:.3f}s")
    except Exception:
        log("[V1] FAILED:")
        log(traceback.format_exc())
        log.close()
        return

    try:
        eig_cfg = EigenspaceConfig(q_max=16, block_size=24, n_iter=6, method="subspace_iter")
        t0 = time.perf_counter()
        out3 = run_v3_full_gpu_eigenspace(solver, x, y, cfg, eig_cfg=eig_cfg)
        dt = time.perf_counter() - t0
        d3 = out3.diagnostics
        log("")
        log("[V3 EigenPro-PCG]")
        log("  " + _fmt(d3, ["cg_iters", "cg_relres", "eigen_n_matvec", "cg_n_matvec", "time_total"]))
        log(f"  wall={dt:.3f}s")
    except Exception:
        log("[V3] FAILED (continuing):")
        log(traceback.format_exc())

    for topk in (64, 128, 256):
        try:
            btab_cfg = BTABConfig(
                active_mode="topk",
                active_topk=int(topk),
                box_budget=3000,
                exact_box_max_size=3000,
            )
            t0 = time.perf_counter()
            out6 = run_v6_box_toeplitz_active_block(solver, x, y, cfg, btab_cfg=btab_cfg)
            dt = time.perf_counter() - t0
            d6 = out6.diagnostics
            log("")
            log(f"[V6 BTAB topk={topk}]")
            log("  " + _fmt(d6, [
                "cg_iters",
                "cg_relres",
                "n_matvec",
                "n_precond",
                "active_size",
                "box_size",
                "tail_size",
                "time_precond_build",
                "time_total",
            ]))
            log(f"  wall={dt:.3f}s")
        except Exception:
            log(f"[V6 topk={topk}] FAILED (continuing):")
            log(traceback.format_exc())

    try:
        btab_cfg = BTABConfig(
            active_mode="topk",
            active_topk=64,
            solve_mode="inner_pcg",
            outer_solver="fgmres",
            outer_gmres_restart=20,
            exact_box_max_size=1,
            inner_tol=1e-3,
            inner_maxiter=20,
            inner_precond="diag",
            box_budget=3000,
        )
        t0 = time.perf_counter()
        out6 = run_v6_box_toeplitz_active_block(solver, x, y, cfg, btab_cfg=btab_cfg)
        dt = time.perf_counter() - t0
        d6 = out6.diagnostics
        log("")
        log("[V6 BTAB inner_pcg topk=64]")
        log("  " + _fmt(d6, [
            "solve_mode",
            "outer_solver",
            "cg_iters",
            "cg_relres",
            "n_matvec",
            "n_precond",
            "box_size",
            "inner_total_iters",
            "inner_total_matvec",
            "time_precond_build",
            "time_total",
        ]))
        log(f"  wall={dt:.3f}s")
    except Exception:
        log("[V6 inner_pcg topk=64] FAILED (continuing):")
        log(traceback.format_exc())

    log("")
    log("validation complete")
    log.close()


if __name__ == "__main__":
    main()
