from __future__ import annotations

import csv
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.family_parameter_sweep_reporting import (
    EXPECTED_SUCCESSFUL_REPEATS,
    SELECTION_RULE,
    collect_family_parameter_sweep_candidates,
    load_pipeline_summary,
    main,
    select_fastest_successful_medians,
    write_family_parameter_sweep_reports,
)


COMMON = {
    "dataset_family": "usgs",
    "dataset_stem": "USGS_example",
    "n_train": 10_000_000,
    "kernel_family": "matern",
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FamilyParameterSweepReportingTests(unittest.TestCase):
    def test_csv_and_json_candidates_use_only_the_declared_selection_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_case = root / "csv_case"
            json_case = root / "json_case"
            csv_case.mkdir()
            json_case.mkdir()

            csv_rows = [
                {
                    **COMMON,
                    "method": "ours-binned-inverse",
                    "status": "ok",
                    "successful_repeats": 3,
                    "measured_repeats": 3,
                    "train_total_seconds_median": 4.0,
                    "test_rmse_median": 0.01,
                    "iterations_median": 17,
                    "effective_active_box_size": 121,
                    "effective_active_topk": 100,
                },
                {
                    **COMMON,
                    "method": "ours-binned-inverse",
                    "status": "error",
                    "successful_repeats": 3,
                    "measured_repeats": 3,
                    "train_total_seconds_median": 0.1,
                    "test_rmse_median": 0.001,
                    "effective_active_box_size": 81,
                },
                {
                    **COMMON,
                    "method": "ours-binned-inverse",
                    "status": "ok",
                    "successful_repeats": 2,
                    "measured_repeats": 3,
                    "train_total_seconds_median": 0.2,
                    "test_rmse_median": 0.001,
                    "effective_active_box_size": 100,
                },
                {
                    **COMMON,
                    "method": "ours-binned-inverse",
                    "status": "ok",
                    "successful_repeats": 3,
                    "measured_repeats": 3,
                    "train_total_seconds_median": "nan",
                    "test_rmse_median": 0.001,
                    "effective_active_box_size": 144,
                },
                {
                    **COMMON,
                    "method": "efgp-standard-full-eig",
                    "status": "ok",
                    "successful_repeats": 3,
                    "measured_repeats": 3,
                    "train_total_seconds_median": 2.0,
                    "test_rmse_median": 0.001,
                    "iterations_median": 5,
                    "effective_active_box_size": 400,
                    "effective_active_topk": 400,
                    "effective_active_rank": 32,
                },
                {
                    **COMMON,
                    "method": "nystrom-krr",
                    "status": "ok",
                    "successful_repeats": 3,
                    "train_total_seconds_median": 0.01,
                },
            ]
            _write_csv(csv_case / "pipeline_summary.csv", csv_rows)

            json_rows = [
                {
                    **COMMON,
                    "method": "ours-binned-inverse",
                    "status": "ok",
                    "successful_repeats": 3,
                    # Deliberately not three: measured repeat count and RMSE are
                    # preserved, but neither is an additional selection gate.
                    "measured_repeats": 99,
                    "train_total_seconds_median": 3.0,
                    "test_rmse_median": 500.0,
                    "iterations_median": 13,
                },
                {
                    **COMMON,
                    "method": "ours-binned-active-eig",
                    "status": "ok",
                    "successful_repeats": 3,
                    "measured_repeats": 3,
                    "train_total_seconds_median": 1.5,
                    # It must still win over the slower full-grid candidate.
                    "test_rmse_median": 100.0,
                    "iterations_median": 7,
                    "effective_active_box_size": 361,
                    "effective_active_topk": 324,
                    "effective_active_rank": 16,
                },
            ]
            (json_case / "pipeline_summary.json").write_text(
                json.dumps(json_rows), encoding="utf-8"
            )

            plan = [
                {
                    "profile": "sweep",
                    "case_id": "csv",
                    "dataset_family": "usgs",
                    "config": SimpleNamespace(
                        output_dir=str(csv_case),
                        dataset_stem="USGS_example",
                        n_train=10_000_000,
                        kernel_family="matern",
                        measured_repeats=3,
                        full_eig_rank=32,
                        expected_active_box_size=400,
                        active_topk=400,
                    ),
                },
                {
                    "profile": "sweep",
                    "case_id": "json",
                    "dataset_family": "usgs",
                    "config": {
                        "output_dir": str(json_case),
                        "dataset_stem": "USGS_example",
                        "n_train": 10_000_000,
                        "kernel_family": "matern",
                        "measured_repeats": 3,
                        "inverse_expected_active_box_size": 225,
                        "inverse_active_topk": 196,
                        "active_eig_rank": 16,
                    },
                },
            ]

            candidates = collect_family_parameter_sweep_candidates(plan)
            self.assertEqual(len(candidates), 7)
            self.assertEqual(
                sum(row["selection_eligible"] for row in candidates), 4
            )
            self.assertTrue(all(row["selection_rule"] == SELECTION_RULE for row in candidates))
            self.assertTrue(all(row["rmse_used_for_selection"] is False for row in candidates))

            nan_row = next(
                row
                for row in candidates
                if row["method"] == "ours-binned-inverse" and row["B"] == 144
            )
            self.assertIsNone(nan_row["train_total_seconds_median"])
            self.assertEqual(
                nan_row["selection_ineligibility_reason"], "train_total_not_finite"
            )

            fallback_inverse = next(
                row
                for row in candidates
                if row["method"] == "ours-binned-inverse"
                and row["case_id"] == "json"
            )
            self.assertEqual(fallback_inverse["B"], 225)
            self.assertEqual(fallback_inverse["active_topk"], 196)
            self.assertIsNone(fallback_inverse["q"])
            self.assertTrue(fallback_inverse["selection_eligible"])

            winners = select_fastest_successful_medians(candidates)
            self.assertEqual(len(winners), 2)
            by_family = {row["parameter_family"]: row for row in winners}
            self.assertEqual(by_family["inverse"]["case_id"], "json")
            self.assertEqual(by_family["inverse"]["train_total_seconds_median"], 3.0)
            self.assertEqual(by_family["inverse"]["test_rmse_median"], 500.0)
            self.assertEqual(by_family["eigen"]["method"], "ours-binned-active-eig")
            self.assertEqual(by_family["eigen"]["B"], 361)
            self.assertEqual(by_family["eigen"]["q"], 16)
            self.assertEqual(by_family["eigen"]["test_rmse_median"], 100.0)

    def test_writer_emits_csv_json_and_auditable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "case"
            report_dir = root / "report"
            case_dir.mkdir()
            rows = [
                {
                    **COMMON,
                    "method": "ours-binned-active-eig",
                    "status": "ok",
                    "successful_repeats": EXPECTED_SUCCESSFUL_REPEATS,
                    "measured_repeats": 3,
                    "train_total_seconds_median": 1.25,
                    "setup_seconds_median": 0.25,
                    "solving_phase_seconds_median": 1.0,
                    "test_rmse_median": 0.5,
                    "iterations_median": 8,
                    "effective_active_box_size": 289,
                    "effective_active_topk": 256,
                    "effective_active_rank": 12,
                }
            ]
            (case_dir / "pipeline_summary.json").write_text(
                json.dumps({"pipeline_summary": rows}), encoding="utf-8"
            )
            plan = [
                {
                    "profile": "synthetic-matern-sweep",
                    "case_id": "n10m-b289-q12",
                    "dataset_family": "synthetic",
                    "config": {
                        "output_dir": str(case_dir),
                        "measured_repeats": 3,
                    },
                }
            ]

            result = write_family_parameter_sweep_reports(plan, report_dir)
            for path in result["paths"].values():
                self.assertTrue(path.is_file(), path)

            written_candidates = json.loads(
                result["paths"]["all_candidates_json"].read_text(encoding="utf-8")
            )
            written_winners = json.loads(
                result["paths"]["selected_winners_json"].read_text(encoding="utf-8")
            )
            manifest = json.loads(
                result["paths"]["manifest_json"].read_text(encoding="utf-8")
            )
            self.assertEqual(len(written_candidates), 1)
            self.assertEqual(len(written_winners), 1)
            self.assertEqual(written_winners[0]["selection_rank"], 1)
            self.assertEqual(manifest["candidate_count"], 1)
            self.assertEqual(manifest["selection_eligible_count"], 1)
            self.assertEqual(manifest["winner_count"], 1)
            self.assertFalse(manifest["rmse_used_for_selection"])
            self.assertIn("RMSE is reported only", manifest["selection_rule"])

            csv_rows = load_pipeline_summary(result["paths"]["all_candidates_csv"])
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["B"], "289")
            self.assertEqual(csv_rows[0]["q"], "12")

    def test_cli_accepts_serialized_plan_and_writes_required_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "case"
            report_dir = root / "report"
            case_dir.mkdir()
            (case_dir / "pipeline_summary.json").write_text(
                json.dumps(
                    [
                        {
                            **COMMON,
                            "method": "ours-binned-inverse",
                            "status": "ok",
                            "successful_repeats": 3,
                            "measured_repeats": 3,
                            "train_total_seconds_median": 2.5,
                            "test_rmse_median": 0.4,
                            "iterations_median": 9,
                            "effective_active_box_size": 625,
                            "effective_active_topk": 512,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "plan": [
                            {
                                "profile": "sweep",
                                "case_id": "inverse-b625",
                                "dataset_family": "Winnebago",
                                "config": {"output_dir": str(case_dir)},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = main(
                    [
                        "--plan-json",
                        str(plan_path),
                        "--output-dir",
                        str(report_dir),
                    ]
                )
            self.assertEqual(return_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["winner_count"], 1)
            for name in (
                "all_candidates.csv",
                "all_candidates.json",
                "selected_winners.csv",
                "selected_winners.json",
                "family_parameter_sweep_manifest.json",
            ):
                self.assertTrue((report_dir / name).is_file(), name)

    def test_cli_reconstructs_plan_from_suite_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_output = root / "suite-runs"
            case_dir = suite_output / "one-family-case" / "synthetic-n10m-inverse"
            report_dir = root / "report"
            case_dir.mkdir(parents=True)
            (case_dir / "pipeline_summary.json").write_text(
                json.dumps(
                    [
                        {
                            **COMMON,
                            "dataset_family": "synthetic source stem",
                            "dataset_stem": "synthetic_true_func_2d_ntrain10000000",
                            "method": "ours-binned-inverse",
                            "status": "ok",
                            "successful_repeats": 3,
                            "measured_repeats": 3,
                            "train_total_seconds_median": 1.75,
                            "test_rmse_median": 0.25,
                            "iterations_median": 6,
                            "effective_active_box_size": 625,
                            "effective_active_topk": 512,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            suite_path = root / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "base": {
                            "methods": ["ours-binned-inverse"],
                            "measured_repeats": 3,
                        },
                        "profiles": {
                            "one-family-case": {
                                "cases": [
                                    {
                                        "id": "synthetic-n10m-inverse",
                                        "dataset_family": "Synthetic",
                                        "dataset_stem": (
                                            "synthetic_true_func_2d_ntrain10000000"
                                        ),
                                        "n_train": 10_000_000,
                                        "kernel_family": "matern",
                                        "inverse_active_topk": 512,
                                        "inverse_expected_active_box_size": 625,
                                    }
                                ]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                return_code = main(
                    [
                        "--suite-config",
                        str(suite_path),
                        "--profile",
                        "one-family-case",
                        "--suite-output-root",
                        str(suite_output),
                        "--output-dir",
                        str(report_dir),
                    ]
                )
            self.assertEqual(return_code, 0)
            winners = json.loads(
                (report_dir / "selected_winners.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(winners), 1)
            # The suite plan's stable family label wins over the summary's
            # dataset-stem-like legacy dataset_family field.
            self.assertEqual(winners[0]["dataset_family"], "Synthetic")

    def test_tied_fastest_time_has_deterministic_parameter_tie_break(self) -> None:
        shared = {
            **COMMON,
            "parameter_family": "inverse",
            "method": "ours-binned-inverse",
            "route": "localized",
            "status": "ok",
            "successful_repeats": 3,
            "train_total_seconds_median": 1.0,
            "selection_eligible": True,
        }
        winners = select_fastest_successful_medians(
            [
                {**shared, "B": 225, "case_id": "later"},
                {**shared, "B": 121, "case_id": "earlier"},
            ]
        )
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]["B"], 121)
        self.assertEqual(winners[0]["fastest_time_tie_count"], 2)

    def test_missing_summary_fails_with_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = [{"config": {"output_dir": str(Path(tmp) / "missing")}}]
            with self.assertRaisesRegex(FileNotFoundError, "pipeline_summary"):
                collect_family_parameter_sweep_candidates(plan)

    def test_nonfinite_hand_built_candidate_cannot_bypass_selector(self) -> None:
        winners = select_fastest_successful_medians(
            [
                {
                    **COMMON,
                    "parameter_family": "inverse",
                    "status": "ok",
                    "successful_repeats": 3,
                    "selection_eligible": True,
                    "train_total_seconds_median": math.inf,
                }
            ]
        )
        self.assertEqual(winners, [])

    def test_selector_rechecks_status_and_successful_repeat_count(self) -> None:
        shared = {
            **COMMON,
            "parameter_family": "inverse",
            "method": "ours-binned-inverse",
            "selection_eligible": True,
            "train_total_seconds_median": 0.1,
        }
        winners = select_fastest_successful_medians(
            [
                {**shared, "status": "error", "successful_repeats": 3},
                {**shared, "status": "ok", "successful_repeats": 2},
            ]
        )
        self.assertEqual(winners, [])


if __name__ == "__main__":
    unittest.main()
