from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark import (
    ControlledConfig,
    PreparedSystem,
    _load_dataset,
    _normalize_n_train,
    _validate_config,
    pairwise_comparisons,
    resolve_method_specs,
    resolve_score_box_rule,
    summarize_rows,
    system_fingerprint,
)


def _fake_system() -> PreparedSystem:
    mtot = 7
    backend = SimpleNamespace(xp=np, fft=np.fft)
    xtxcol = np.zeros(2 * mtot - 1, dtype=np.complex128)
    xtxcol[mtot - 1] = 100.0
    weights = np.array([0.1, 0.3, 1.0, 2.0, 1.0, 0.3, 0.1])
    data_ctx = SimpleNamespace(
        xtxcol_gpu=xtxcol,
        gf_gpu=np.fft.fftn(xtxcol),
        weights_gpu_flat=weights,
        weights_np_flat=weights.copy(),
        rhs_gpu=np.arange(mtot, dtype=np.complex128),
        meta={"mtot": mtot, "dim": 1},
    )
    fingerprint = system_fingerprint(data_ctx, 0.1)
    return PreparedSystem(
        backend=backend,
        data_ctx=data_ctx,
        rhs_gpu=data_ctx.rhs_gpu,
        reg_lambda=0.1,
        setup_seconds=1.0,
        system_id=fingerprint,
        manifest={"M": mtot, "mtot": mtot, "dim": 1},
    )


def test_score_rule_uses_only_declared_memory_cap() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "default", "full-eig"),
        score_tau=1.0,
        box_budget=3,
        inverse_max_size=3,
        rank=2,
        measured_repeats=5,
    )

    rule = resolve_score_box_rule(system, cfg)
    specs, _ = resolve_method_specs(system, cfg)

    assert rule.raw_tau_box_size > 3
    assert int(rule.active.box_idx.size) <= 3
    assert rule.selection_rule == "score_ranked_memory_capped_box"
    default = next(spec for spec in specs if spec.label == "default")
    assert default.kind == "active-inverse"


def test_full_inverse_spec_uses_the_complete_fourier_grid() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "full-inverse"),
        score_tau=1.0,
        box_budget=3,
        inverse_max_size=3,
        rank=2,
        measured_repeats=5,
    )

    specs, _ = resolve_method_specs(system, cfg)
    full = next(spec for spec in specs if spec.label == "full-inverse")

    assert full.kind == "active-inverse"
    assert full.selection_rule == "full_grid"
    assert full.result_role == "baseline"
    assert full.btab_config is not None
    assert full.btab_config.active_mode == "topk"
    assert full.btab_config.active_topk == system.manifest["M"]
    assert full.btab_config.box_budget == system.manifest["M"]
    assert full.btab_config.exact_box_max_size == system.manifest["M"]


def test_system_fingerprint_detects_rhs_change() -> None:
    system = _fake_system()
    before = system_fingerprint(system.data_ctx, system.reg_lambda)
    system.data_ctx.rhs_gpu = system.data_ctx.rhs_gpu.copy()
    system.data_ctx.rhs_gpu[0] += 1.0
    after = system_fingerprint(system.data_ctx, system.reg_lambda)
    assert before != after


def test_empty_score_box_makes_default_fall_back_to_jacobi() -> None:
    system = _fake_system()
    system.reg_lambda = 1e12
    cfg = ControlledConfig(
        methods=("cg", "default"),
        score_tau=1.0,
        box_budget=3,
        inverse_max_size=3,
        rank=2,
        measured_repeats=5,
    )

    specs, rule = resolve_method_specs(system, cfg)
    default = next(spec for spec in specs if spec.label == "default")

    assert rule.active.box_idx.size == 0
    assert default.kind == "jacobi"
    assert "empty_fallback_jacobi" in default.selection_rule


def test_mixed32_rejects_tolerance_near_machine_epsilon() -> None:
    cfg = ControlledConfig(
        methods=("cg",),
        precision="mixed32",
        tol=1e-7,
        measured_repeats=5,
    )
    with pytest.raises(ValueError, match="machine eps"):
        _validate_config(cfg)


def test_zero_n_train_means_all_but_negative_is_invalid() -> None:
    assert _normalize_n_train(None) is None
    assert _normalize_n_train(0) is None
    assert _normalize_n_train(12) == 12
    with pytest.raises(ValueError, match="n_train"):
        _normalize_n_train(-1)


def test_loader_rejects_requested_rows_beyond_source(tmp_path) -> None:
    np.savez(
        tmp_path / "tiny.npz",
        x_train=np.zeros((3, 2), dtype=np.float32),
        y_train=np.zeros(3, dtype=np.float32),
    )
    (tmp_path / "tiny.json").write_text(
        '{"shapes":{"n_train":3,"dim":2}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exceeds"):
        _load_dataset("tiny", 4, 0, str(tmp_path))


def test_summary_uses_paired_cold_speedup_and_requested_order() -> None:
    rows = []
    for repeat, cg_time, pcg_total in [(0, 10.0, 5.0), (1, 20.0, 5.0)]:
        rows.extend(
            [
                {
                    "method": "pcg",
                    "method_kind": "active-inverse",
                    "repeat_idx": repeat,
                    "is_warmup": False,
                    "build_seconds": 1.0,
                    "solve_seconds": pcg_total - 1.0,
                    "build_plus_solve_seconds": pcg_total,
                    "iterations": 2,
                    "true_relres": 1e-9,
                    "status": "converged",
                },
                {
                    "method": "cg",
                    "method_kind": "cg",
                    "repeat_idx": repeat,
                    "is_warmup": False,
                    "build_seconds": 0.0,
                    "solve_seconds": cg_time,
                    "build_plus_solve_seconds": cg_time,
                    "iterations": 10,
                    "true_relres": 1e-9,
                    "status": "converged",
                },
            ]
        )

    summary = summarize_rows(rows, 100.0, 1e-7, method_order=("cg", "pcg"))

    assert [row["method"] for row in summary] == ["cg", "pcg"]
    pcg = summary[1]
    assert pcg["cold_speedup_median"] == pytest.approx(3.0)
    assert pcg["paired_wins_over_cg"] == 2
    comparison = pairwise_comparisons(rows)[0]
    assert comparison["reference_method"] == "cg"
    assert comparison["candidate_method"] == "pcg"
    assert comparison["total_speedup_median"] == pytest.approx(3.0)


def test_unconverged_candidate_is_excluded_from_speedup_claims() -> None:
    rows = [
        {
            "method": "cg",
            "method_kind": "cg",
            "repeat_idx": 0,
            "is_warmup": False,
            "build_seconds": 0.0,
            "solve_seconds": 10.0,
            "build_plus_solve_seconds": 10.0,
            "iterations": 10,
            "true_relres": 5e-8,
            "status": "converged",
            "tol": 1e-7,
        },
        {
            "method": "fast-but-wrong",
            "method_kind": "jacobi",
            "repeat_idx": 0,
            "is_warmup": False,
            "build_seconds": 0.1,
            "solve_seconds": 0.1,
            "build_plus_solve_seconds": 0.2,
            "iterations": 1,
            "true_relres": 2e-7,
            "status": "maxiter",
            "tol": 1e-7,
        },
    ]

    summary = summarize_rows(
        rows,
        1.0,
        1e-7,
        method_order=("cg", "fast-but-wrong"),
    )
    bad = summary[1]
    assert bad["converged_repeats"] == 0
    assert bad["paired_comparisons"] == 0
    assert not bad["performance_claim_eligible"]
    assert np.isnan(bad["cold_speedup_median"])

    comparison = pairwise_comparisons(rows, 1e-7)[0]
    assert comparison["available_paired_repeats"] == 1
    assert comparison["paired_repeats"] == 0
    assert not comparison["performance_claim_eligible"]
    assert np.isnan(comparison["total_speedup_median"])


def test_error_repeat_cannot_disappear_from_pairwise_eligibility() -> None:
    rows = []
    for repeat in range(5):
        rows.append(
            {
                "method": "cg",
                "repeat_idx": repeat,
                "is_warmup": False,
                "build_plus_solve_seconds": 10.0,
                "solve_seconds": 10.0,
                "true_relres": 5e-8,
                "status": "converged",
                "tol": 1e-7,
            }
        )
        rows.append(
            {
                "method": "candidate",
                "repeat_idx": repeat,
                "is_warmup": False,
                "build_plus_solve_seconds": 5.0 if repeat else np.nan,
                "solve_seconds": 4.0 if repeat else np.nan,
                "true_relres": 5e-8 if repeat else np.nan,
                "status": "converged" if repeat else "error",
                "tol": 1e-7,
            }
        )

    comparison = pairwise_comparisons(rows, 1e-7)[0]
    assert comparison["available_paired_repeats"] == 5
    assert comparison["paired_repeats"] == 4
    assert not comparison["performance_claim_eligible"]


def test_pairwise_reports_crossover_for_higher_build_lower_solve() -> None:
    rows = []
    for repeat in range(5):
        rows.extend(
            [
                {
                    "method": "cg",
                    "repeat_idx": repeat,
                    "is_warmup": False,
                    "build_seconds": 0.0,
                    "solve_seconds": 10.0,
                    "build_plus_solve_seconds": 10.0,
                    "true_relres": 1e-9,
                    "status": "converged",
                    "tol": 1e-7,
                },
                {
                    "method": "candidate",
                    "repeat_idx": repeat,
                    "is_warmup": False,
                    "build_seconds": 12.0,
                    "solve_seconds": 4.0,
                    "build_plus_solve_seconds": 16.0,
                    "true_relres": 1e-9,
                    "status": "converged",
                    "tol": 1e-7,
                },
            ]
        )

    comparison = pairwise_comparisons(rows, 1e-7)[0]
    assert comparison["crossover_status"] == "candidate_higher_build_lower_solve"
    assert comparison["cold_to_reuse_crossover_rhs"] == pytest.approx(2.0)
    assert np.isnan(comparison["candidate_faster_through_rhs"])
