from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..backends import BackendConfig


@dataclass
class BTABConfig:
    active_mode: str = "topk"
    active_topk: Optional[int] = 1024
    active_tau: Optional[float] = None
    box_budget: Optional[int] = 5000
    solve_mode: str = "auto"  # "auto", "exact", "inner_pcg"
    exact_box_max_size: Optional[int] = 5000
    exact_apply_mode: str = "inverse"  # "inverse", "chol_solve"
    outer_solver: str = "auto"  # "auto", "pcg", "fgmres"
    outer_gmres_restart: int = 50
    inner_tol: float = 1e-3
    inner_maxiter: int = 50
    inner_precond: str = "diag"  # "diag", "identity"
    chol_jitter: float = 1e-12
    diag_floor: float = 1e-30
    keep_box_matrix: bool = False


@dataclass
class BTABExperimentConfig:
    dataset_stems: list[str] = field(
        default_factory=lambda: [
            "synthetic_true_func_2d_n100000",
            "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain100000",
        ]
    )
    kernel_family: str = "matern"
    kernel_lengthscale: float = 0.1
    kernel_nu: float = 1.5
    kernel_variance: float = 1.0
    reg_lambda: float = 0.1
    eps: float = 1e-5
    nufft_tol: float = 1e-10
    l2_scaled: bool = True
    tol: float = 1e-6
    maxiter: int = 6000
    chunk_size: Optional[int] = None
    profile_components: bool = True
    debug_finite_checks: bool = False
    eigenpro_topq_list: list[int] = field(default_factory=lambda: [45, 90])
    btab_active_mode: str = "topk"
    btab_topk_list: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    btab_tau_list: list[float] = field(default_factory=lambda: [1e-1, 1e-2])
    btab_box_budget: Optional[int] = 5000
    btab_solve_mode: str = "auto"
    btab_exact_box_max_size: Optional[int] = 5000
    btab_exact_apply_mode: str = "inverse"
    btab_outer_solver: str = "auto"
    btab_outer_gmres_restart: int = 50
    btab_inner_tol: float = 1e-3
    btab_inner_maxiter: int = 50
    btab_inner_precond: str = "diag"
    btab_keep_box_matrix: bool = False
    seed: int = 0
    output_dir: str = ""
    run_tag: str = field(
        default_factory=lambda: datetime.now().strftime("btab_gpu_%Y%m%d_%H%M%S")
    )
    backend: BackendConfig = field(default_factory=BackendConfig)

    def resolve_output_dir(self, base_dir: str | Path) -> Path:
        if self.output_dir:
            return Path(self.output_dir).expanduser().resolve()
        return Path(base_dir).resolve() / "outputs" / self.run_tag
