from __future__ import annotations

import math
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end import (
    END_TO_END_METHODS,
    PROTOCOL_FAMILY,
    STAGE2_SYSTEM_CONFIG_FIELDS,
    TIMING_SCOPE,
    EndToEndConfig,
    _validate_config,
    choose_rpcholesky_landmarks,
    choose_uniform_landmarks,
    fit_restricted_krr,
    kernel_cross,
    predict_restricted_krr,
    summarize_pipeline_rows,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end_suite import (
    _load_completed_case,
    build_profile_plan,
    load_suite_config,
    materialize_robustness_plan,
    require_complete_plan,
    select_target_regime,
)


def test_protocol_methods_are_complete_krr_pipelines() -> None:
    assert PROTOCOL_FAMILY == "end_to_end_krr"
    assert set(END_TO_END_METHODS) == {
        "nystrom-krr",
        "rpcholesky-krr",
        "efgp-standard-cg",
        "efgp-standard-jacobi",
        "efgp-standard-full-eig",
        "ours-binned-default",
    }
    assert "cg" not in END_TO_END_METHODS
    assert "nystrom" not in END_TO_END_METHODS
    assert "rpcholesky" not in END_TO_END_METHODS


def test_mixed_pipeline_config_rejects_mismatched_regularization_scaling() -> None:
    with pytest.raises(ValueError, match="mismatched ridge scaling"):
        _validate_config(EndToEndConfig(regularization_convention="mean_loss"))


def test_kernel_cross_matches_closed_form_matern_three_halves() -> None:
    x = np.asarray([[0.0, 0.0], [0.3, 0.4]], dtype=np.float64)
    z = np.asarray([[0.0, 0.0]], dtype=np.float64)
    result = kernel_cross(
        np,
        x,
        z,
        family="matern",
        lengthscale=0.2,
        nu=1.5,
        variance=1.7,
        dtype=np.float64,
    ).reshape(-1)
    scaled = math.sqrt(3.0) * 0.5 / 0.2
    expected = np.asarray([1.7, 1.7 * (1.0 + scaled) * math.exp(-scaled)])
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


def test_streamed_restricted_krr_matches_dense_normal_equations() -> None:
    rng = np.random.default_rng(9)
    x = rng.uniform(size=(31, 2))
    y = np.sin(2.0 * np.pi * x[:, 0]) + 0.2 * x[:, 1]
    cfg = EndToEndConfig(
        methods=("nystrom-krr",),
        nystrom_rank=7,
        low_rank_chunk_size=6,
        low_rank_dtype="fp64",
        kernel_family="matern",
        nu=1.5,
        lengthscale=0.25,
        reg_lambda=0.07,
    )
    indices = choose_uniform_landmarks(len(x), 7, seed=4)
    coefficients, landmarks, diagnostics = fit_restricted_krr(np, x, y, indices, cfg)

    C = kernel_cross(
        np,
        x,
        x[indices],
        family=cfg.kernel_family,
        lengthscale=cfg.lengthscale,
        nu=cfg.nu,
        variance=cfg.variance,
        dtype=np.float64,
    )
    W = kernel_cross(
        np,
        x[indices],
        x[indices],
        family=cfg.kernel_family,
        lengthscale=cfg.lengthscale,
        nu=cfg.nu,
        variance=cfg.variance,
        dtype=np.float64,
    )
    dense_system = C.T @ C + cfg.reg_lambda * W
    # Reproduce the documented numerical jitter.
    dense_system += diagnostics["jitter"] * np.eye(len(indices))
    expected = np.linalg.solve(dense_system, C.T @ y)
    np.testing.assert_allclose(coefficients, expected, rtol=2e-11, atol=2e-11)
    predicted = predict_restricted_krr(np, x, landmarks, coefficients, cfg)
    np.testing.assert_allclose(predicted, C @ expected, rtol=2e-12, atol=2e-12)
    assert diagnostics["regularization_convention"] == "absolute"


def test_exact_rpcholesky_is_reproducible_and_resource_gated() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(40, 2))
    cfg = EndToEndConfig(
        methods=("rpcholesky-krr",),
        rpcholesky_rank=6,
        low_rank_chunk_size=11,
        low_rank_dtype="fp64",
        rpcholesky_max_factor_bytes=10_000_000,
    )
    first, diag_first = choose_rpcholesky_landmarks(
        np, x, cfg, rank=6, dtype=np.float64
    )
    second, diag_second = choose_rpcholesky_landmarks(
        np, x, cfg, rank=6, dtype=np.float64
    )
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == len(first)
    assert diag_first["selection_algorithm"] == "exact_simple_rpcholesky"
    assert diag_first["relative_trace_final"] <= 1.0
    assert diag_first == diag_second

    gated = EndToEndConfig(
        methods=("rpcholesky-krr",),
        rpcholesky_rank=6,
        rpcholesky_max_factor_bytes=1,
    )
    with pytest.raises(MemoryError, match="will record resource_limit") as exc_info:
        choose_rpcholesky_landmarks(np, x, gated, rank=6, dtype=np.float64)
    assert exc_info.value.required_bytes == 6 * len(x) * 8
    assert exc_info.value.effective_cap_bytes == 1


def test_rpcholesky_stops_cleanly_when_residual_is_exhausted() -> None:
    x = np.zeros((12, 2), dtype=np.float64)
    cfg = EndToEndConfig(
        methods=("rpcholesky-krr",),
        rpcholesky_rank=6,
        rpcholesky_max_factor_bytes=10_000_000,
    )
    pivots, diagnostics = choose_rpcholesky_landmarks(
        np, x, cfg, rank=6, dtype=np.float64
    )
    assert len(pivots) == 1
    assert diagnostics["effective_rank"] == 1
    assert diagnostics["relative_trace_final"] <= 1e-12


def test_summary_speedup_requires_accuracy_gate() -> None:
    cfg = EndToEndConfig(
        methods=(
            "nystrom-krr",
            "efgp-standard-full-eig",
            "ours-binned-default",
        ),
        measured_repeats=1,
        accuracy_relative_tolerance=0.01,
    )
    common = {
        "protocol_family": PROTOCOL_FAMILY,
        "dataset_stem": "toy",
        "n_train": 100,
        "n_test": 20,
        "dim": 2,
        "kernel_family": "matern",
        "lengthscale": 0.1,
        "nu": 1.5,
        "reg_lambda": 0.1,
        "box_budget": 64,
        "repeat_idx": 0,
        "is_warmup": False,
        "status": "converged",
        "prediction_seconds": 0.1,
        "test_mae": 0.1,
        "test_r2": 0.9,
    }
    rows = [
        {
            **common,
            "method": "efgp-standard-full-eig",
            "setup_seconds": 4.0,
            "solver_build_seconds": 1.0,
            "iterative_solve_seconds": 5.0,
            "solving_phase_seconds": 6.0,
            "train_total_seconds": 10.0,
            "test_rmse": 1.0,
        },
        {
            **common,
            "method": "ours-binned-default",
            "setup_seconds": 1.0,
            "solver_build_seconds": 1.0,
            "iterative_solve_seconds": 2.0,
            "solving_phase_seconds": 3.0,
            "train_total_seconds": 4.0,
            "test_rmse": 1.005,
        },
        {
            **common,
            "method": "nystrom-krr",
            "setup_seconds": 1.0,
            "solver_build_seconds": 0.0,
            "iterative_solve_seconds": 1.0,
            "solving_phase_seconds": 1.0,
            "train_total_seconds": 2.0,
            "test_rmse": 1.2,
        },
    ]
    summary = {row["method"]: row for row in summarize_pipeline_rows(rows, cfg)}
    assert summary["ours-binned-default"]["accuracy_eligible"] is True
    assert summary["efgp-standard-full-eig"]["ours_total_speedup"] == 2.5
    assert summary["efgp-standard-full-eig"]["fourier_eps"] == cfg.fourier_eps
    assert (
        summary["efgp-standard-full-eig"]["setup_seconds_at_median_total"]
        + summary["efgp-standard-full-eig"]["solving_phase_seconds_at_median_total"]
        == summary["efgp-standard-full-eig"]["train_total_seconds_median"]
    )
    assert summary["nystrom-krr"]["accuracy_eligible"] is False
    assert math.isnan(summary["nystrom-krr"]["ours_total_speedup"])


def test_accuracy_gate_requires_every_paired_repeat_and_absolute_quality() -> None:
    cfg = EndToEndConfig(
        methods=("efgp-standard-full-eig", "ours-binned-default"),
        measured_repeats=3,
        accuracy_relative_tolerance=0.01,
        accuracy_max_rmse=1.1,
        accuracy_min_r2=0.5,
    )
    common = {
        "protocol_family": PROTOCOL_FAMILY,
        "dataset_stem": "toy",
        "dataset_family": "Toy",
        "n_train": 100,
        "n_test": 20,
        "dim": 2,
        "kernel_family": "matern",
        "lengthscale": 0.1,
        "nu": 1.5,
        "variance": 1.0,
        "reg_lambda": 0.1,
        "regularization_convention": "absolute",
        "fourier_eps": 1e-5,
        "box_budget": 64,
        "is_warmup": False,
        "status": "converged",
        "setup_seconds": 1.0,
        "solver_build_seconds": 0.25,
        "iterative_solve_seconds": 0.75,
        "solving_phase_seconds": 1.0,
        "train_total_seconds": 2.0,
        "prediction_seconds": 0.1,
        "test_mae": 0.1,
    }
    rows = []
    for repeat_idx in range(3):
        rows.append(
            {
                **common,
                "method": "efgp-standard-full-eig",
                "repeat_idx": repeat_idx,
                "test_rmse": 1.0,
                "test_r2": 0.8,
            }
        )
        rows.append(
            {
                **common,
                "method": "ours-binned-default",
                "repeat_idx": repeat_idx,
                "test_rmse": 2.0 if repeat_idx == 2 else 0.9,
                "test_r2": 0.1 if repeat_idx == 2 else 0.85,
            }
        )
    summary = {row["method"]: row for row in summarize_pipeline_rows(rows, cfg)}
    assert summary["efgp-standard-full-eig"]["accuracy_eligible"] is True
    assert summary["ours-binned-default"]["test_rmse_median"] == 0.9
    assert summary["ours-binned-default"]["accuracy_passed_repeats"] == 2
    assert summary["ours-binned-default"]["accuracy_eligible"] is False
    assert summary["ours-binned-default"]["performance_claim_eligible"] is False
    assert math.isnan(summary["ours-binned-default"]["ours_total_speedup"])


def _target_rows(n_train: int, cg_iterations: int) -> list[dict[str, object]]:
    common: dict[str, object] = {
        "dataset_stem": "synthetic_true_func_2d_n300000000",
        "n_train": n_train,
        "kernel_family": "matern",
        "lengthscale": 0.1,
        "nu": 1.5,
        "reg_lambda": 0.1,
        "fourier_eps": 1e-5,
        "status": "ok",
        "accuracy_eligible": True,
    }
    rows = [{**common, "method": method} for method in END_TO_END_METHODS]
    for row in rows:
        if row["method"] == "efgp-standard-cg":
            row["iterations_median"] = cg_iterations
    return rows


def test_target_selection_is_frozen_and_uses_largest_eligible_n() -> None:
    rows = [
        *_target_rows(10_000_000, 3500),
        *_target_rows(30_000_000, 5500),
        *_target_rows(100_000_000, 9000),
    ]
    selected = select_target_regime(rows)
    assert selected["n_train"] == 30_000_000
    assert selected["eligible_candidate_count"] == 2
    assert selected["rejected_candidate_count"] == 1


def test_target_selection_uses_prospectively_declared_equal_n_priority() -> None:
    synthetic = _target_rows(10_000_000, 3500)
    winnebago = [
        {
            **row,
            "dataset_stem": "winnebago_master",
        }
        for row in _target_rows(10_000_000, 3500)
    ]
    selected = select_target_regime(
        [*synthetic, *winnebago],
        dataset_priority=("winnebago_master", "synthetic_true_func_2d_n300000000"),
    )
    assert selected["dataset_stem"] == "winnebago_master"


def test_target_selection_accepts_declared_rpcholesky_resource_limit() -> None:
    rows = [*_target_rows(10_000_000, 4000), *_target_rows(30_000_000, 5000)]
    for row in rows:
        if row["method"] == "rpcholesky-krr" and row["n_train"] == 30_000_000:
            row["status"] = "resource_limit"
    selected = select_target_regime(rows)
    assert selected["n_train"] == 30_000_000
    assert selected["resource_limited_methods"] == ["rpcholesky-krr"]


def test_target_selection_fails_closed_for_undeclared_resource_limit() -> None:
    rows = _target_rows(10_000_000, 4000)
    next(row for row in rows if row["method"] == "efgp-standard-jacobi")[
        "status"
    ] = "resource_limit"
    with pytest.raises(RuntimeError, match="Do not cherry-pick"):
        select_target_regime(rows)


def test_suite_refuses_partial_plan_before_downstream_work() -> None:
    plan = [{"case_id": "case-a"}, {"case_id": "case-b"}]
    partial_index = [
        {
            "case_id": "case-a",
            "all_rows_present": True,
            "status": "claim_eligible_complete",
            "error_type": None,
        }
    ]
    with pytest.raises(RuntimeError, match="refusing target selection/downstream"):
        require_complete_plan(plan, partial_index, phase="test scale")

    complete_index = [
        {
            "case_id": case_id,
            "all_rows_present": True,
            "status": "complete_with_resource_limits",
            "error_type": None,
        }
        for case_id in ("case-a", "case-b")
    ]
    require_complete_plan(plan, complete_index, phase="test scale")


def test_shipped_suite_covers_10m_to_300m_and_materializes_robustness(
    tmp_path: Path,
) -> None:
    suite = load_suite_config()
    scale = build_profile_plan(
        suite,
        "scale_10m_300m",
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    assert {item["config"].n_train for item in scale} == {
        10_000_000,
        30_000_000,
        100_000_000,
        300_000_000,
    }
    assert {item["dataset_family"] for item in scale} == {
        "Synthetic",
        "Winnebago",
    }
    target = select_target_regime(_target_rows(10_000_000, 4000))
    robust = materialize_robustness_plan(
        suite,
        target,
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    configs = [item["config"] for item in robust]
    assert {cfg.reg_lambda for cfg in configs} >= {0.01, 0.1, 1.0}
    assert {cfg.lengthscale for cfg in configs} >= {0.05, 0.1, 0.2}
    assert {cfg.box_budget for cfg in configs} >= {4096, 8192, 16384}
    assert {cfg.dataset_stem for cfg in configs} >= {
        "synthetic_true_func_2d_n300000000",
        "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain10000000",
    }


def test_robustness_uses_exact_dataset_artifact_at_selected_n(tmp_path: Path) -> None:
    suite = load_suite_config()
    target = select_target_regime(_target_rows(30_000_000, 4000))
    robust = materialize_robustness_plan(
        suite,
        target,
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    winnebago = [item for item in robust if item.get("dataset_family") == "Winnebago"]
    assert len(winnebago) == 1
    assert winnebago[0]["config"].n_train == 30_000_000
    assert winnebago[0]["config"].dataset_stem.endswith("ntrain30000000")
    assert winnebago[0]["config"].accuracy_max_rmse == 0.15


def test_resume_reuses_resource_outcome_but_not_execution_error(tmp_path: Path) -> None:
    cfg = EndToEndConfig(
        methods=("rpcholesky-krr",),
        output_dir=str(tmp_path),
    )
    config_payload = asdict(cfg)
    config_payload["methods"] = list(cfg.methods)
    (tmp_path / "experiment_config.json").write_text(
        json.dumps(config_payload), encoding="utf-8"
    )
    summary = {
        "method": "rpcholesky-krr",
        "status": "resource_limit",
        "timing_scope": TIMING_SCOPE,
        **{field: getattr(cfg, field) for field in STAGE2_SYSTEM_CONFIG_FIELDS},
        "accuracy_relative_tolerance": cfg.accuracy_relative_tolerance,
        "accuracy_max_rmse": cfg.accuracy_max_rmse,
        "accuracy_min_r2": cfg.accuracy_min_r2,
        "expected_measured_repeats": cfg.measured_repeats,
        "accuracy_evaluated_repeats": 0,
        "accuracy_passed_repeats": 0,
        "setup_seconds_at_median_total": None,
        "solving_phase_seconds_at_median_total": None,
    }
    (tmp_path / "pipeline_summary.json").write_text(
        json.dumps([summary]), encoding="utf-8"
    )
    (tmp_path / "pipeline_runs.csv").write_text(
        "method,status\nrpcholesky-krr,resource_limit\n", encoding="utf-8"
    )
    completion = {
        "protocol_family": PROTOCOL_FAMILY,
        "timing_scope": TIMING_SCOPE,
        "methods": ["rpcholesky-krr"],
        "artifact_complete": True,
        "all_rows_present": True,
        "error_methods": [],
    }
    completion_path = tmp_path / "run_complete.json"
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    assert _load_completed_case(cfg) is not None

    completion["error_methods"] = ["rpcholesky-krr"]
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    assert _load_completed_case(cfg) is None
