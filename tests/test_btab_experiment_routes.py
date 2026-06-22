from __future__ import annotations

import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.config import (
    BTABExperimentConfig,
    resolve_btab_experiment_route,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.run_experiments import (
    _parse_topk_q_pairs,
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
            6000,
            [1024, 2048, 4096],
            [],
        ),
        (
            "group_b",
            12000,
            [],
            [
                (4096, 128),
                (4096, 192),
                (8192, 128),
                (8192, 192),
            ],
        ),
        (
            "group_c",
            25000,
            [1024, 2048, 4096],
            [
                (4096, 128),
                (4096, 192),
                (8192, 128),
                (8192, 192),
                (16384, 192),
                (16384, 256),
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
    assert got.btab_exact_box_max_size == 6000
    assert got.btab_exact_apply_mode == (
        "inverse" if route == "group_a" else "chol_solve"
    )


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
