from __future__ import annotations

import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.config import (
    BTABExperimentConfig,
    expand_btab_experiment_routes,
    resolve_btab_experiment_route,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block import run_experiments as run_experiments_module
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.run_experiments import (
    _parse_topk_q_pairs,
    build_gpu_run_config,
    method_rows_for_dataset,
)


def test_cartesian_route_preserves_legacy_lists_and_clears_shortlists():
    cfg = BTABExperimentConfig(
        btab_experiment_route="cartesian",
        btab_topk_list=[7, 11],
        btab_eig_q_list=[3, 5],
        btab_inverse_topk_list=[7],
        btab_boxeig_topk_q_pairs=[(11, 5)],
    )
    got = resolve_btab_experiment_route(cfg, n_train=1_000_000)
    assert got.btab_topk_list == [7, 11]
    assert got.btab_eig_q_list == [3, 5]
    assert got.btab_inverse_topk_list is None
    assert got.btab_boxeig_topk_q_pairs is None


def test_custom_route_preserves_explicit_non_cartesian_shortlist():
    cfg = BTABExperimentConfig(
        btab_experiment_route="custom",
        btab_topk_list=[7, 11],
        btab_tau_list=[0.1],
        btab_eig_q_list=[3, 5],
        btab_inverse_topk_list=[7],
        btab_boxeig_topk_q_pairs=[(11, 5)],
    )
    got = resolve_btab_experiment_route(cfg, n_train=1_000_000)
    assert got.btab_active_mode == "topk"
    assert got.btab_tau_list == []
    assert got.btab_inverse_topk_list == [7]
    assert got.btab_boxeig_topk_q_pairs == [(11, 5)]


@pytest.mark.parametrize(
    ("missing_field", "message"),
    [
        ("inverse", "btab_inverse_topk_list"),
        ("boxeig", "btab_boxeig_topk_q_pairs"),
    ],
)
def test_custom_route_requires_both_explicit_shortlists(missing_field, message):
    kwargs = {
        "btab_experiment_route": "custom",
        "btab_inverse_topk_list": [1024],
        "btab_boxeig_topk_q_pairs": [(4096, 128)],
    }
    if missing_field == "inverse":
        kwargs["btab_inverse_topk_list"] = None
    else:
        kwargs["btab_boxeig_topk_q_pairs"] = None
    with pytest.raises(ValueError, match=message):
        resolve_btab_experiment_route(BTABExperimentConfig(**kwargs))


@pytest.mark.parametrize(
    ("route", "budget", "inverse", "boxeig"),
    [
        (
            "group_a",
            80000,
            [512, 728, 1024, 2048, 4096],
            [
                (2048, 192),
                (4096, 192),
                (4096, 256),
                (8192, 192),
                (8192, 256),
                (8192, 320),
                (16384, 192),
                (16384, 256),
                (16384, 320),
            ],
        ),
        (
            "group_b",
            80000,
            [512, 1024],
            [
                (256, 64),
                (512, 128),
                (768, 128),
            ],
        ),
        (
            "group_c",
            80000,
            [2048, 4096],
            [
                (20480, 320),
                (20480, 384),
                (32768, 192),
                (32768, 256),
                (32768, 320),
                (32768, 384),
                (32768, 448),
                (35721, 192),
                (35721, 256),
                (35721, 320),
                (35721, 384),
                (35721, 448),
            ],
        ),
    ],
)
def test_named_routes_resolve_exact_shortlists(route, budget, inverse, boxeig):
    got = resolve_btab_experiment_route(
        BTABExperimentConfig(btab_experiment_route=route),
        n_train=10_000_000,
    )
    assert got.btab_box_budget == budget
    assert got.btab_inverse_topk_list == inverse
    assert got.btab_boxeig_topk_q_pairs == boxeig
    assert got.btab_exact_box_max_size == 20000
    assert got.btab_exact_apply_mode == (
        "chol_solve" if route == "group_b" else "inverse"
    )


@pytest.mark.parametrize(
    ("route", "kernel_family", "n_train_list"),
    [
        ("group_a", "matern", [3_000_000, 1_000_000, 30_000_000, 10_000_000]),
        (
            "group_b",
            "SE",
            [300_000_000, 100_000_000, 3_000_000, 1_000_000, 30_000_000, 10_000_000],
        ),
        ("group_c", "matern", [300_000_000, 100_000_000]),
    ],
)
def test_named_routes_apply_experiment_presets(route, kernel_family, n_train_list):
    cfg = BTABExperimentConfig(
        btab_experiment_route=route,
        dataset_stems=["manual_dataset"],
        n_train_list=[123],
        kernel_family="manual_kernel",
        kernel_family_list=["manual_kernel"],
        kernel_params_by_family={"manual_kernel": {"kernel_lengthscale": 9.0}},
        eps=9e-3,
        eps_list=[9e-3],
    )
    got = resolve_btab_experiment_route(cfg, n_train=10_000_000)
    assert got.kernel_family == kernel_family
    assert got.kernel_family_list == [kernel_family]
    assert got.dataset_stems == [
        "synthetic_true_func_2d_n1000000",
        "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain1000000",
    ]
    assert got.n_train_list == n_train_list
    assert got.eps == 1e-5
    assert got.eps_list == [1e-5]
    assert cfg.dataset_stems == ["manual_dataset"]
    assert cfg.n_train_list == [123]
    assert cfg.eps_list == [9e-3]
    assert got.dataset_stems is not cfg.dataset_stems
    assert got.n_train_list is not cfg.n_train_list
    assert got.eps_list is not cfg.eps_list


def test_expand_btab_experiment_routes_preserves_input_order():
    cfg = BTABExperimentConfig(
        btab_experiment_routes=["group_c", "group_a", "group_b"],
    )
    got = expand_btab_experiment_routes(cfg)
    assert [item.btab_experiment_route for item in got] == [
        "group_c",
        "group_a",
        "group_b",
    ]
    assert all(item.btab_experiment_routes == [] for item in got)


def test_expand_btab_experiment_routes_rejects_mixed_multi_route_modes():
    cfg = BTABExperimentConfig(btab_experiment_routes=["group_a", "custom"])
    with pytest.raises(ValueError, match="multiple entries only"):
        expand_btab_experiment_routes(cfg)


@pytest.mark.parametrize(
    ("n_train", "resolved_route", "budget", "inverse", "boxeig"),
    [
        (
            3_000_000,
            "schedule_small",
            6000,
            [1024, 2048, 4096],
            [(1024, 64), (1024, 128), (2048, 128), (4096, 128), (4096, 192)],
        ),
        (
            10_000_000,
            "schedule_medium",
            12000,
            [2048, 4096],
            [(8192, 128), (8192, 192)],
        ),
        (
            100_000_000,
            "schedule_large",
            25000,
            [4096],
            [(8192, 128), (8192, 192), (16384, 192), (16384, 256)],
        ),
    ],
)
def test_schedule_resolves_from_n_train(n_train, resolved_route, budget, inverse, boxeig):
    got = resolve_btab_experiment_route(
        BTABExperimentConfig(btab_experiment_route="schedule"),
        n_train=n_train,
    )
    assert got.btab_experiment_route == resolved_route
    assert got.btab_box_budget == budget
    assert got.btab_inverse_topk_list == inverse
    assert got.btab_boxeig_topk_q_pairs == boxeig


def test_schedule_requires_n_train():
    with pytest.raises(ValueError, match="requires n_train"):
        resolve_btab_experiment_route(
            BTABExperimentConfig(btab_experiment_route="schedule")
        )


def test_parse_custom_boxeig_pairs():
    assert _parse_topk_q_pairs("4096:128,8192:192") == [
        (4096, 128),
        (8192, 192),
    ]
    assert _parse_topk_q_pairs("") is None


def test_parse_custom_boxeig_pairs_rejects_cartesian_style_input():
    with pytest.raises(ValueError, match="topk:q"):
        _parse_topk_q_pairs("4096,128")


def test_build_gpu_run_config_allows_maxiter_override():
    cfg = BTABExperimentConfig(maxiter=80000, non_v1_maxiter=5000)
    assert build_gpu_run_config(cfg).maxiter == 80000
    assert build_gpu_run_config(cfg, maxiter=cfg.non_v1_maxiter).maxiter == 5000


def test_method_rows_use_long_maxiter_only_for_v1(monkeypatch):
    seen: dict[str, list[int]] = {"v1": [], "v3": [], "v6": [], "v7": []}

    class DummyOut:
        backend = None
        data_ctx = None
        beta_gpu = None
        diagnostics = {"status": "ok"}

    def record(name):
        def _fn(*args, **kwargs):
            run_cfg = args[3]
            seen[name].append(int(run_cfg.maxiter))
            return DummyOut()

        return _fn

    monkeypatch.setattr(run_experiments_module, "EFGPSolver", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_experiments_module, "make_kernel", lambda *args, **kwargs: object())
    monkeypatch.setattr(run_experiments_module, "evaluate_output", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(run_experiments_module, "run_v1_pure_efgp", record("v1"))
    monkeypatch.setattr(run_experiments_module, "run_v3_full_gpu_eigenspace", record("v3"))
    monkeypatch.setattr(run_experiments_module, "run_v6_box_toeplitz_active_block", record("v6"))
    monkeypatch.setattr(run_experiments_module, "run_v7_box_eigenpro_active_block", record("v7"))

    cfg = BTABExperimentConfig(
        btab_experiment_route="custom",
        maxiter=80000,
        non_v1_maxiter=5000,
        eigenpro_topq_list=[4],
        btab_inverse_topk_list=[8],
        btab_boxeig_topk_q_pairs=[(8, 4)],
    )
    payload = {
        "stem": "synthetic_true_func_2d_n10",
        "dim": 2,
        "n_train": 10,
        "n_test": 5,
        "x_train": [[0.0, 0.0]],
        "y_train": [0.0],
        "x_test": [[0.0, 0.0]],
        "y_test": [0.0],
    }

    rows = method_rows_for_dataset(cfg, payload)

    assert seen == {"v1": [80000], "v3": [5000], "v6": [5000], "v7": [5000]}
    maxiters_by_method = {row["method"]: row["maxiter"] for row in rows}
    assert maxiters_by_method["plain_cg"] == 80000
    assert maxiters_by_method["eigenpro_pcg_q4"] == 5000
    assert maxiters_by_method["btab_auto_topk_8"] == 5000
    assert maxiters_by_method["btab_boxeig_topk_8_q4"] == 5000
    assert {row["v1_maxiter"] for row in rows} == {80000}
    assert {row["non_v1_maxiter"] for row in rows} == {5000}
