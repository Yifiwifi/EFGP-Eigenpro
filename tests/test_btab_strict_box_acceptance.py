from __future__ import annotations

import json
from pathlib import Path

import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.strict_box_acceptance import (
    AcceptanceInputError,
    evaluate_strict_box_run,
    main,
)


def _write_run(
    run_dir: Path,
    *,
    repeats: int = 5,
    include_diagnostics: bool = True,
    memory_capped: bool = False,
) -> None:
    run_dir.mkdir(parents=True)
    manifest = {
        "system_id": "one-fixed-system",
        "final_system_id": "one-fixed-system",
        "system_unchanged": True,
        "precision_mode": "fp64",
        "x_host_dtype": "float64",
        "weights_dtype": "float64",
        "gf_dtype": "complex128",
        "rhs_solve_dtype": "complex128",
        "real_component_dtype": "float64",
        "tolerance": 1e-7,
        "M": 100,
        "score_tau_raw_box_size": 20,
        "score_box_size": 20,
        "score_cap_excludes_requested_threshold_modes": memory_capped,
    }
    (run_dir / "system_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    rows = []
    cold_times = {"cg": 6.0, "default": 4.0, "full-eig": 5.2}
    for repeat_idx in range(repeats):
        for method in ("cg", "default", "full-eig"):
            rows.append(
                {
                    "system_id": "one-fixed-system",
                    "method": method,
                    "repeat_idx": repeat_idx,
                    "is_warmup": False,
                    "tol": 1e-7,
                    "precision_mode": "fp64",
                    "solve_dtype": "complex128",
                    "true_residual_audit_dtype": "complex128",
                    "status": "converged",
                    "true_relres": 5e-8,
                    "build_plus_solve_seconds": cold_times[method],
                    "rank": 32 if method != "cg" else None,
                    "box_size": 20 if method == "default" else None,
                }
            )
    (run_dir / "matched_runs.json").write_text(json.dumps(rows), encoding="utf-8")
    if include_diagnostics:
        diagnostics = [
            {
                "system_id": "one-fixed-system",
                "method": "full-eig",
                "diagnostic_status": "ok",
                "score_box_leverage_capture": 0.92,
                "score_box_fraction": 0.20,
                "score_box_size": 20,
            }
        ]
        (run_dir / "post_diagnostics.json").write_text(
            json.dumps(diagnostics), encoding="utf-8"
        )


def test_complete_run_passes_all_predeclared_gates(tmp_path: Path) -> None:
    run_dir = tmp_path / "passing"
    _write_run(run_dir)

    report = evaluate_strict_box_run(run_dir)

    assert report["status"] == "pass"
    assert report["eligible"] is True
    assert report["failed_criteria"] == []
    assert report["criteria"]["cold_speedup"]["evidence"][
        "median_speedup_over_same_rank_full_eig"
    ] == pytest.approx(1.3)
    assert report["criteria"]["score_leverage"]["evidence"][
        "leverage_enrichment"
    ] == pytest.approx(4.6)


def test_absent_score_diagnostics_are_pending_not_pass(tmp_path: Path) -> None:
    run_dir = tmp_path / "pending"
    _write_run(run_dir, include_diagnostics=False)

    report = evaluate_strict_box_run(run_dir)

    assert report["status"] == "pending"
    assert report["eligible"] is False
    assert report["pending_criteria"] == ["score_leverage"]
    assert report["criteria"]["score_leverage"]["status"] == "pending"


def test_memory_capped_threshold_and_slow_full_comparison_fail(tmp_path: Path) -> None:
    run_dir = tmp_path / "failing"
    _write_run(run_dir, memory_capped=True)
    rows = json.loads((run_dir / "matched_runs.json").read_text(encoding="utf-8"))
    for row in rows:
        if row["method"] == "full-eig":
            row["build_plus_solve_seconds"] = 4.4
    (run_dir / "matched_runs.json").write_text(json.dumps(rows), encoding="utf-8")

    report = evaluate_strict_box_run(run_dir)

    assert report["status"] == "fail"
    assert set(report["failed_criteria"]) >= {
        "cold_speedup",
        "threshold_not_memory_capped",
    }
    assert report["criteria"]["cold_speedup"]["evidence"][
        "median_speedup_over_same_rank_full_eig"
    ] == pytest.approx(1.1)


def test_mismatched_repeat_sets_and_failed_true_residual_are_rejected(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "bad_pairs"
    _write_run(run_dir)
    rows = json.loads((run_dir / "matched_runs.json").read_text(encoding="utf-8"))
    rows = [
        row
        for row in rows
        if not (row["method"] == "full-eig" and row["repeat_idx"] == 4)
    ]
    next(row for row in rows if row["method"] == "default")["true_relres"] = 2e-7
    (run_dir / "matched_runs.json").write_text(json.dumps(rows), encoding="utf-8")

    report = evaluate_strict_box_run(run_dir)

    assert report["status"] == "fail"
    assert "paired_repeats" in report["failed_criteria"]
    assert "audited_convergence" in report["failed_criteria"]


def test_missing_core_file_raises_for_module_and_cli_emits_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(AcceptanceInputError):
        evaluate_strict_box_run(tmp_path)

    exit_code = main([str(tmp_path), "--compact"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "fail"
    assert payload["failed_criteria"] == ["input_files"]


def test_zero_box_fraction_is_a_json_serializable_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "zero_fraction"
    _write_run(run_dir)
    diagnostics = json.loads(
        (run_dir / "post_diagnostics.json").read_text(encoding="utf-8")
    )
    diagnostics[0]["score_box_fraction"] = 0.0
    (run_dir / "post_diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8"
    )

    exit_code = main([str(run_dir), "--compact"])

    assert exit_code == 1


def test_enrichment_below_three_is_descriptive_not_a_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "moderate_enrichment"
    _write_run(run_dir)
    diagnostics = json.loads(
        (run_dir / "post_diagnostics.json").read_text(encoding="utf-8")
    )
    diagnostics[0].update(
        {
            "score_box_leverage_capture": 0.95,
            "score_box_fraction": 0.40,
            "score_box_size": 40,
        }
    )
    manifest = json.loads(
        (run_dir / "system_manifest.json").read_text(encoding="utf-8")
    )
    manifest["score_box_size"] = 40
    manifest["score_tau_raw_box_size"] = 40
    (run_dir / "post_diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8"
    )
    (run_dir / "system_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    rows = json.loads((run_dir / "matched_runs.json").read_text(encoding="utf-8"))
    for row in rows:
        if row["method"] == "default":
            row["box_size"] = 40
    (run_dir / "matched_runs.json").write_text(json.dumps(rows), encoding="utf-8")

    report = evaluate_strict_box_run(run_dir)

    assert report["status"] == "pass"
    evidence = report["criteria"]["score_leverage"]["evidence"]
    assert evidence["leverage_enrichment"] == pytest.approx(2.375)
    assert "minimum_enrichment" not in evidence


def test_score_box_fraction_above_one_half_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "large_box"
    _write_run(run_dir)
    diagnostics = json.loads(
        (run_dir / "post_diagnostics.json").read_text(encoding="utf-8")
    )
    diagnostics[0].update(
        {
            "score_box_leverage_capture": 0.95,
            "score_box_fraction": 0.60,
            "score_box_size": 60,
        }
    )
    manifest = json.loads(
        (run_dir / "system_manifest.json").read_text(encoding="utf-8")
    )
    manifest["score_box_size"] = 60
    manifest["score_tau_raw_box_size"] = 60
    (run_dir / "post_diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8"
    )
    (run_dir / "system_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    rows = json.loads((run_dir / "matched_runs.json").read_text(encoding="utf-8"))
    for row in rows:
        if row["method"] == "default":
            row["box_size"] = 60
    (run_dir / "matched_runs.json").write_text(json.dumps(rows), encoding="utf-8")

    report = evaluate_strict_box_run(run_dir)

    assert report["status"] == "fail"
    assert "score_leverage" in report["failed_criteria"]
    assert report["criteria"]["score_leverage"]["evidence"][
        "maximum_score_box_fraction"
    ] == pytest.approx(0.50)
