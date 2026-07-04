from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
    eig_q: int = 64
    eig_tol: float = 1e-3
    eig_maxiter: Optional[int] = None
    eig_ncv: Optional[int] = None
    eig_apply_batch_cols: Optional[int] = None
    diagnostic_mode: str = "cheap"  # "none", "cheap", "full"
    diagnostic_power_iter: int = 30
    diagnostic_tol: float = 1e-2


@dataclass
class BTABExperimentConfig:
    dataset_stems: list[str] = field(
        default_factory=lambda: [
            "synthetic_true_func_2d_n100000",
            "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain100000",
        ]
    )
    n_train_list: list[int] = field(default_factory=list)
    kernel_family: str = "matern"
    kernel_family_list: list[str] = field(default_factory=list)
    kernel_params_by_family: dict[str, dict[str, Any]] = field(default_factory=dict)
    kernel_lengthscale: float = 0.1
    kernel_nu: float = 1.5
    kernel_variance: float = 1.0
    reg_lambda: float = 0.1
    eps: float = 1e-5
    eps_list: list[float] = field(default_factory=lambda: [1e-5])
    nufft_tol: float = 1e-10
    l2_scaled: bool = True
    tol: float = 1e-6
    maxiter: int = 6000
    non_v1_maxiter: int = 5000
    chunk_size: Optional[int] = None
    profile_components: bool = True
    debug_finite_checks: bool = False
    warmup_repeats: int = 0
    measured_repeats: int = 1
    eigenpro_topq_list: list[int] = field(default_factory=lambda: [64,128, 256,320,384,448])
    btab_active_mode: str = "topk"
    btab_experiment_route: str = "cartesian"
    btab_experiment_routes: list[str] = field(default_factory=list)
    btab_topk_list: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    btab_tau_list: list[float] = field(default_factory=lambda: [1e-1, 1e-2])
    btab_inverse_topk_list: Optional[list[int]] = None
    btab_boxeig_topk_q_pairs: Optional[list[tuple[int, int]]] = None
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
    btab_eig_q_list: list[int] = field(default_factory=lambda: [32, 64, 128])
    btab_eig_tol: float = 1e-3
    btab_eig_maxiter: Optional[int] = None
    btab_eig_ncv: Optional[int] = None
    btab_eig_apply_batch_cols: Optional[int] = None
    btab_diagnostic_mode: str = "cheap"
    btab_diagnostic_power_iter: int = 30
    btab_diagnostic_tol: float = 1e-2
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


_BTAB_ROUTE_ALIASES = {
    "full": "cartesian",
    "legacy": "cartesian",
    "a": "group_a",
    "exact": "group_a",
    "exact_inverse": "group_a",
    "b": "group_b",
    "boxeig": "group_b",
    "idea_validation": "group_b",
    "c": "group_c",
    "large": "group_c",
    "large_scale": "group_c",
    "auto": "schedule",
    "n_schedule": "schedule",
}

_BTAB_NAMED_GROUPS = {"group_a", "group_b", "group_c"}


def normalize_btab_experiment_route(route: str) -> str:
    route = str(route or "cartesian").strip().lower()
    return _BTAB_ROUTE_ALIASES.get(route, route)


def expand_btab_experiment_routes(cfg: BTABExperimentConfig) -> list[BTABExperimentConfig]:
    """Expand a config with optional multiple named route groups.

    ``btab_experiment_routes`` is intended for running several named presets in
    one notebook invocation. Non-named routes stay single-route to preserve the
    historical Cartesian/custom/schedule semantics.
    """
    raw_routes = cfg.btab_experiment_routes or [cfg.btab_experiment_route]
    routes = [normalize_btab_experiment_route(route) for route in raw_routes if str(route).strip()]
    if not routes:
        routes = [normalize_btab_experiment_route(cfg.btab_experiment_route)]
    if len(routes) > 1:
        invalid = [route for route in routes if route not in _BTAB_NAMED_GROUPS]
        if invalid:
            raise ValueError(
                "btab_experiment_routes supports multiple entries only for "
                f"group_a, group_b, and group_c; got {invalid!r}."
            )
    expanded: list[BTABExperimentConfig] = []
    for route in routes:
        expanded.append(
            replace(
                cfg,
                btab_experiment_route=route,
                btab_experiment_routes=[],
            )
        )
    return expanded


def resolve_btab_experiment_route(
    cfg: BTABExperimentConfig,
    *,
    n_train: Optional[int] = None,
) -> BTABExperimentConfig:
    """Resolve a named BTAB experiment route into an exact method shortlist.

    ``cartesian`` preserves the historical Cartesian-product behavior.
    ``custom`` uses the explicit inverse top-k values and Box-EigenPro
    ``(top-k, q)`` pairs stored on the config. Named routes provide curated
    non-Cartesian shortlists so large experiments do not accidentally run
    every combination.
    """
    route = normalize_btab_experiment_route(cfg.btab_experiment_route)
    if route == "cartesian":
        return replace(
            cfg,
            btab_experiment_route="cartesian",
            btab_inverse_topk_list=None,
            btab_boxeig_topk_q_pairs=None,
        )
    if route == "custom":
        if cfg.btab_inverse_topk_list is None:
            raise ValueError(
                "btab_experiment_route='custom' requires "
                "btab_inverse_topk_list."
            )
        if cfg.btab_boxeig_topk_q_pairs is None:
            raise ValueError(
                "btab_experiment_route='custom' requires "
                "btab_boxeig_topk_q_pairs."
            )
        return replace(
            cfg,
            btab_experiment_route="custom",
            btab_active_mode="topk",
            btab_tau_list=[],
        )
    if route in {"schedule_small", "schedule_medium", "schedule_large"}:
        return cfg

    exact_apply_mode = "chol_solve"
    route_preset: dict[str, Any] = {}
    if route == "schedule":
        if n_train is None:
            raise ValueError("btab_experiment_route='schedule' requires n_train.")
        if int(n_train) <= 3_000_000:
            topk = [512, 1024, 2048, 4096]
            inverse = [1024, 2048, 4096]
            boxeig = [
                (1024, 64),
                (1024, 128),
                (2048, 128),
                (4096, 128),
                (4096, 192),
            ]
            eig_q = [64, 128, 192]
            box_budget = 6000
            resolved_route = "schedule_small"
        elif int(n_train) <= 30_000_000:
            topk = [1024, 2048, 4096, 8192]
            inverse = [2048, 4096]
            boxeig = [(8192, 128), (8192, 192)]
            eig_q = [128, 192]
            box_budget = 12000
            resolved_route = "schedule_medium"
        else:
            topk = [2048, 4096, 8192, 16384]
            inverse = [4096]
            boxeig = [
                (8192, 128),
                (8192, 192),
                (16384, 192),
                (16384, 256),
            ]
            eig_q = [128, 192, 256]
            box_budget = 25000
            resolved_route = "schedule_large"
    elif route == "group_a":
        # for matern kernel with small sample, M=35721
        topk = [512, 1024, 2048, 4096,8192]
        inverse = [512,728, 1024, 2048, 4096]
        boxeig = [
    # small active blocks: useful for N=1e6, 3e6

    (2048, 192),

    # medium active blocks: useful for N=3e6, 1e7

    (4096, 192),
    (4096, 256),

    (8192, 192),
    (8192, 256),
    (8192, 320),

    # large active blocks: useful for N=1e7, 3e7
 
    (16384, 192),
    (16384, 256),
    (16384, 320),
]

        eig_q = [64,128,192]
        box_budget = 80000
        exact_apply_mode = "inverse"
        resolved_route = route
        route_preset = {
            "dataset_stems": [
                "synthetic_true_func_2d_n1000000",
                "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain1000000",
            ],
            "n_train_list": [3_000_000, 1_000_000,30_000_000, 10_000_000],
            "kernel_family": "matern",
            "kernel_family_list": ["matern"],
            "kernel_params_by_family": {
                "matern": {
                    "kernel_lengthscale": 0.1,
                    "kernel_nu": 1.5,
                    "kernel_variance": 1.0,
                }
            },
            "kernel_lengthscale": 0.1,
            "kernel_nu": 1.5,
            "kernel_variance": 1.0,
            "eps": 1e-5,
            "eps_list": [1e-5],
        }
    elif route == "group_b":
        # for SE kernel, M=1225
        # Keep this route separate from the exact-inverse sweep.
        topk = [512,1024]
        inverse = [512,1024, 1225]
        boxeig = [
            (256, 64),
            (512, 128),
            (768, 128),
        ]
        eig_q = [16,32,64,128, 192,256]
        box_budget = 80000
        exact_apply_mode = "inverse"
        resolved_route = route
        route_preset = {
            "dataset_stems": [
                "synthetic_true_func_2d_n1000000",
                "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain1000000",
            ],
            "n_train_list": [300_000_000, 100_000_000,3_000_000, 1_000_000,30_000_000, 10_000_000],
            "kernel_family": "SE",
            "kernel_family_list": ["SE"],
            "kernel_params_by_family": {
                "SE": {
                    "kernel_lengthscale": 0.1,
                    "kernel_variance": 1.0,
                }
            },
            "kernel_lengthscale": 0.1,
            "kernel_nu": 1.5,
            "kernel_variance": 1.0,
            "eps": 1e-5,
            "eps_list": [1e-5],
        }
    elif route == "group_c":
        # for matern kernel with large sample, M=35721
        topk = [1024, 2048, 4096, 8192, 16384]
        inverse = [4096, 8192 ,12288]    #[1024, 2048,4096]
        boxeig = [
            (20480, 192),
            (20480, 256),
            (20480, 320),
            (20480, 384),
            (20480, 448),

            (25720, 192),
            (25720, 256),
            (25720, 320),
            (25720, 384),
            (25720, 448),

            (35721, 192),
            (35721, 256),
            (35721, 320),
            (35721, 384),
            (35721, 448),
        ]
        #''' (4096, 192),(4096, 256),(8192, 192),            (8192, 256),(16384, 192),            (16384, 256),'''

        eig_q = [128, 192,256,320,384,448]
        box_budget = 80000
        exact_apply_mode = "inverse"
        resolved_route = route
        route_preset = {
            "dataset_stems": [
                "synthetic_true_func_2d_n1000000",
                "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain1000000",
            ],
            "n_train_list": [300_000_000, 100_000_000],
            "kernel_family": "matern",
            "kernel_family_list": ["matern"],
            "kernel_params_by_family": {
                "matern": {
                    "kernel_lengthscale": 0.1,
                    "kernel_nu": 1.5,
                    "kernel_variance": 1.0,
                }
            },
            "kernel_lengthscale": 0.1,
            "kernel_nu": 1.5,
            "kernel_variance": 1.0,
            "eps": 1e-5,
            "eps_list": [1e-5],
        }
    else:
        raise ValueError(
            "Unknown btab_experiment_route "
            f"{cfg.btab_experiment_route!r}; expected cartesian, custom, "
            "group_a, group_b, group_c, or schedule."
        )

    return replace(
        cfg,
        **route_preset,
        btab_experiment_route=resolved_route,
        btab_experiment_routes=[],
        btab_active_mode="topk",
        btab_topk_list=list(topk),
        btab_tau_list=[],
        btab_inverse_topk_list=list(inverse),
        btab_boxeig_topk_q_pairs=list(boxeig),
        btab_box_budget=box_budget,
        btab_solve_mode="auto",
        btab_exact_box_max_size=20000,
        btab_exact_apply_mode=exact_apply_mode,
        btab_eig_q_list=list(eig_q),
    )
