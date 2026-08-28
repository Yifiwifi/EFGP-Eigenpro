from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    build_colab_all_experiments_notebook as notebook_builder,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.suite import (
    load_suite_config,
)


CONTROLLED_DIR = Path(notebook_builder.__file__).resolve().parent
SUITE_PATH = CONTROLLED_DIR / "colab_all_experiments_suite.json"
NOTEBOOK_PATH = CONTROLLED_DIR.parent / "colab_all_experiments_10m_300m.ipynb"


def _all_source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_colab_suite_declares_dataset_family_for_every_case() -> None:
    suite = load_suite_config(SUITE_PATH)
    allowed = {"Synthetic", "Winnebago", "Manitowoc"}
    for profile_name, profile in suite["profiles"].items():
        for case in profile["cases"]:
            assert case["dataset_family"] in allowed, (profile_name, case["id"])


def test_box_budget_profile_is_a_fixed_three_method_sweep() -> None:
    suite = load_suite_config(SUITE_PATH)
    profile = suite["profiles"]["winnebago_box_budget_n10m"]
    assert profile["overrides"]["methods"] == ["cg", "default", "full-eig"]
    assert [case["box_budget"] for case in profile["cases"]] == [4096, 8192, 16384]
    assert {case["dataset_stem"] for case in profile["cases"]} == {
        "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain10000000"
    }
    assert {case["expected_n_train"] for case in profile["cases"]} == {10_000_000}


def test_one_click_plan_is_strictly_two_stage() -> None:
    notebook = notebook_builder.build_notebook()
    source = _all_source(notebook)
    assert "RUN_ALL_FORMAL_EXPERIMENTS = True" in source
    assert "RUN_STAGE1_END_TO_END_KRR = RUN_ALL_FORMAL_EXPERIMENTS" in source
    assert "RUN_STAGE2_FIXED_AB_SOLVERS = RUN_ALL_FORMAL_EXPERIMENTS" in source
    assert 'STAGE1_SCALE_PROFILE = "scale_10m_300m"' in source
    assert '"nystrom-krr", "rpcholesky-krr", "efgp-standard-cg"' in source
    assert '"efgp-standard-jacobi", "efgp-standard-full-eig"' in source
    assert '"ours-binned-default"' in source
    assert "stage1_suite.build_profile_plan(" in source
    assert "stage1_suite.select_target_regime(" in source
    assert "canonical_reporting.load_stage1_summaries(" in source
    assert source.index("canonical_reporting.load_stage1_summaries(") < source.index(
        "stage1_suite.select_target_regime("
    )
    assert "Stage-1 scale campaign is incomplete; refusing target" in source
    assert "allowed_resource_limit_methods=selection.get(" in source
    assert "stage1_suite.materialize_robustness_plan(" in source
    assert (
        'expected_config = normalize_stage1_config_value(asdict(item["config"]))'
        in source
    )
    assert "observed_config == expected_config" in source
    assert "Winnebago 10/30/100/300M exact artifacts" in source
    assert (
        'frame["declared_dataset_family"] = declared_stage1_dataset_family(item)'
        in source
    )
    assert 'frame["suite_profile"] = item["profile"]' in source
    assert 'frame["robustness_axes"] = json.dumps(' in source
    assert "robustness_axes, ensure_ascii=False" in source
    assert 'frame["fourier_eps"] = float(cfg.fourier_eps)' in source
    assert "(STAGE1_SCALE_SUMMARY_PATH, stage1_scale_summary)" not in source
    assert 'globals().get("STAGE1_SCALE_SUMMARY_PATH")' in source
    assert '"artifact_complete": artifact_complete' in source
    assert '"scientific_eligible": scientific_eligible' in source
    assert 'status = str(completion_payload["formal_result_status"])' in source
    assert '"complete_with_resource_limits"' in source
    assert '"complete_with_usability_ineligible_methods"' in source
    assert "stage1_scale_artifacts_complete" in source
    assert "stage1_scale_scientifically_eligible" in source
    assert 'target_profile_name = "fixed_ab_selected_target"' in source
    assert 'PREDICTION_AUDIT_PROFILES = ["fixed_ab_selected_target"]' in source
    assert "PREDICTION_AUDIT_MAX_TEST_N = 2_500_000" in source
    assert 'payload.get("timing_solutions_reused") is not True' in source
    assert (
        'DATA_MANIFEST_SNAPSHOT = DRIVE_RUN_ROOT / "data_manifest_snapshot.json"'
        in source
    )
    assert '"data_manifest_snapshot": str(DATA_MANIFEST_SNAPSHOT)' in source
    assert '"first_run_campaign_elapsed_seconds"' in source
    assert 'print("Verifying selected local cache SHA-256:", basename)' in source
    assert "local_link.symlink_to(source)" in source
    assert '"prediction_summary_complete"' in source
    assert '"plot_artifacts_complete"' in source
    assert "expected_selected_controlled_pairs" in source
    assert "controlled_plot = selected_controlled.copy()" in source
    assert '"stage1_krr_train_total_10m_300m.png"' in source
    assert '"stage1_krr_setup_solving_breakdown.png"' in source
    assert '"stage1_krr_accuracy_tradeoff.png"' in source
    assert '"usability_eligible"' in source
    assert '"reference_equivalent"' in source
    assert '"pareto_nondominated"' in source
    assert "descriptive quality trade-off, not a speed gate" in source
    assert "all paired speedups (x = outside broad usable range)" in source
    assert '"stage1_krr_robustness.png"' in source
    assert '"stage2_fixed_ab_solver_total.png"' in source
    assert '"stage2_formal_solver_totals.csv"' in source
    assert "CONTROLLED ARTIFACT AUDIT FAILED; refusing every formal" in source
    assert '"solver_total_seconds = selection + preconditioner build + solve"' in source
    assert "selected_target_path=str(STAGE1_TARGET_PATH)" in source
    assert "stage1_suite_path=str(STAGE1_SUITE_CONFIG)" in source
    assert "stage2_feasibility_path=str(STAGE2_FEASIBILITY_PATH)" in source
    assert (
        'STAGE2_FEASIBILITY_PATH = DRIVE_RUN_ROOT / "stage2_feasibility.json"' in source
    )
    assert '"prospective declared active-box upper bound before timing"' in source
    assert "STAGE2_SYSTEM_CONFIG_FIELDS = (" in source
    assert "STAGE2_METHOD_CONFIG_FIELDS = (" in source
    assert '"rank", "full_eig_rank", "active_topk"' in source
    assert '"allow_frozen_topk_capacity_adaptation"' in source
    assert '"parameter_selection_policy", "parameter_source"' in source
    assert "effective_active_topk" in source
    assert "effective_active_box_size" in source
    assert "active_selection_rule" in source
    assert "capacity_adapted" in source
    assert '"precision", "nufft_backend", "precompute_chunk_size"' in source
    assert "inverse_feasible = active_box_upper_bound <= inverse_max_size" in source
    assert '"active_box_upper_bound": active_box_upper_bound' in source
    assert '"methods": list(STAGE2_FEASIBLE_METHODS)' in source
    assert '"stage2_feasibility_decision": STAGE2_FEASIBILITY' in source
    assert "STAGE2_FORBIDDEN_METHODS" in source
    assert '"fourier-nystrom-precond"' in source
    assert '"fourier-rpcholesky-precond"' in source
    assert '"pending_data_generation_and_prefix_verification"' in source
    assert "check=False" in source
    assert "NO_ELIGIBLE_TARGET_FAIL_CLOSED" in source


def test_committed_notebook_matches_generator() -> None:
    committed = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert committed == notebook_builder.build_notebook()


@pytest.mark.parametrize(
    (
        "box_budget",
        "active_box_upper_bound",
        "inverse_max_size",
        "active_inverse_feasible",
    ),
    [
        (80_000, 10_609, 8_192, False),
        (80_000, 10_609, 16_384, True),
    ],
)
def test_one_click_orchestrator_runs_only_fixed_ab_at_frozen_target(
    tmp_path: Path,
    box_budget: int,
    active_box_upper_bound: int,
    inverse_max_size: int,
    active_inverse_feasible: bool,
) -> None:
    notebook = notebook_builder.build_notebook()
    orchestrator_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def select_profile_cases" in "".join(cell["source"])
    )
    repository_root = CONTROLLED_DIR.parents[3]

    def fake_run_cmd(args, **kwargs):
        args = [str(value) for value in args]
        config_path = Path(args[args.index("--config") + 1])
        output_root = Path(args[args.index("--output-root") + 1])
        profile_name = args[args.index("--profile") + 1]
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        cases = payload["profiles"][profile_name]["cases"]
        is_resume = (output_root / "suite_status.json").is_file()
        status_rows = []
        for case in cases:
            run_dir = output_root / case["id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            status_row = {
                "case_id": case["id"],
                "status": "resumed_existing" if is_resume else "completed",
            }
            status_rows.append(status_row)
        (output_root / "suite_status.json").write_text(
            json.dumps(status_rows),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    namespace = {
        "Path": Path,
        "json": json,
        "pd": pd,
        "sys": sys,
        "time": time,
        "LOCAL_REPO": repository_root,
        "LOCAL_DATA_DIR": tmp_path / "data",
        "DRIVE_RUN_ROOT": tmp_path / "run",
        "RUN_ALL_FORMAL_EXPERIMENTS": True,
        "RUN_STAGE2_FIXED_AB_SOLVERS": True,
        "STAGE1_SCALE_PROFILE": "scale_10m_300m",
        "STAGE2_METHODS": [
            "cg",
            "jacobi",
            "default",
            "active-inverse",
            "active-eig",
            "full-eig",
        ],
        "STAGE2_MANDATORY_METHODS": [
            "cg",
            "jacobi",
            "default",
            "active-eig",
            "full-eig",
        ],
        "END_TO_END_TARGET": {
            "dataset_stem": "synthetic_true_func_2d_n300000000",
            "n_train": 30_000_000,
            "subset_seed": 0,
            "subset_mode": "prefix",
            "kernel_family": "matern",
            "lengthscale": 0.1,
            "nu": 1.5,
            "variance": 1.0,
            "reg_lambda": 0.1,
            "fourier_eps": 1e-5,
            "nufft_tol": 1e-10,
            "l2_scaled": True,
            "precision": "fp64",
            "nufft_backend": "cufinufft",
            "precompute_chunk_size": 1_000_000,
            "rank": 320,
            "full_eig_rank": 256,
            "active_topk": 8_192,
            "expected_active_box_size": active_box_upper_bound,
            "allow_frozen_topk_capacity_adaptation": False,
            "box_budget": box_budget,
            "inverse_max_size": inverse_max_size,
            "parameter_selection_policy": (
                "historical_selected_transfer_no_current_scan"
            ),
            "parameter_source": "test frozen historical selection",
        },
        "stage1_config": {
            "base": {
                "dataset_stem": "synthetic_true_func_2d_n300000000",
                "n_train": 10_000_000,
                "rank": 320,
                "full_eig_rank": 256,
                "active_topk": 8_192,
                "expected_active_box_size": active_box_upper_bound,
                "allow_frozen_topk_capacity_adaptation": False,
                "box_budget": box_budget,
                "inverse_max_size": 1_024,
                "parameter_selection_policy": (
                    "historical_selected_transfer_no_current_scan"
                ),
                "parameter_source": "test frozen historical selection",
            },
            "stage2_fixed_ab": {
                "inverse_max_size": inverse_max_size,
                "default_inverse_max_size": 1_024,
            },
            "profiles": {
                "scale_10m_300m": {
                    "cases": [
                        {
                            "id": "synthetic_matern_n30m",
                            "dataset_stem": "synthetic_true_func_2d_n300000000",
                            "n_train": 30_000_000,
                        }
                    ],
                },
            },
        },
        "stage1_scale_summary": pd.DataFrame(
            [
                {
                    "dataset_stem": "synthetic_true_func_2d_n300000000",
                    "n_train": 30_000_000,
                    "dataset_family": "Synthetic",
                }
            ]
        ),
        "stage1_campaign_rows": [
            {
                "job_id": "stage1_scale_synthetic_n30m",
                "profile": "scale_10m_300m",
                "dataset_family": "Synthetic",
                "n_train": 30_000_000,
                "mandatory": True,
                "status": "complete_with_resource_limits",
                "reason": "declared resource-limit methods: rpcholesky-krr",
                "artifact_complete": True,
                "scientific_eligible": False,
                "resource_limit_methods": "rpcholesky-krr",
                "error_methods": "",
                "case_count": 1,
                "invocation_mode": "resumed_existing",
                "resumed_case_count": 1,
                "executed_case_count": 0,
            }
        ],
        "FORMAL_SCALE_SIZES": [10_000_000, 30_000_000, 100_000_000, 300_000_000],
        "RUN_PLUMBING_SMOKE": True,
        "SMOKE_OK": True,
        "SMOKE_RETURN_CODE": 0,
        "CAN_RUN_300M": True,
        "PROFILE_DATASET_FAMILIES": {
            "scale_development_masters": ["Synthetic"],
            "scale_archived_exact": ["Winnebago"],
        },
        "ACTIVE_CASE_IDS": [],
        "run_cmd": fake_run_cmd,
        "display": lambda value: None,
        "validate_archived_synthetic_inputs": lambda sizes: None,
    }
    exec(compile(orchestrator_source, "<one-click-orchestrator>", "exec"), namespace)

    records = namespace["selected_case_records"]
    assert len(records) == 1
    assert records[0]["suite_profile"] == "fixed_ab_selected_target"
    assert records[0]["dataset_family"] == "Synthetic"
    assert records[0]["case_id"] == "fixed_ab_target_n30000000"
    runtime_payload = json.loads(
        (
            tmp_path / "run" / "runtime_configs" / "fixed_ab_selected_target.json"
        ).read_text(encoding="utf-8")
    )
    target_profile = runtime_payload["profiles"]["fixed_ab_selected_target"]
    expected_methods = [
        method
        for method in namespace["STAGE2_METHODS"]
        if method != "active-inverse" or active_inverse_feasible
    ]
    assert target_profile["overrides"]["methods"] == expected_methods
    assert target_profile["overrides"]["box_budget"] == box_budget
    assert target_profile["overrides"]["inverse_max_size"] == inverse_max_size
    assert target_profile["overrides"]["default_inverse_max_size"] == 1_024
    method_config_fields = (
        "rank",
        "full_eig_rank",
        "active_topk",
        "expected_active_box_size",
        "allow_frozen_topk_capacity_adaptation",
        "box_budget",
        "parameter_selection_policy",
        "parameter_source",
    )
    for field in method_config_fields:
        assert target_profile["overrides"][field] == namespace["END_TO_END_TARGET"][
            field
        ]
    system_fields = (
        "dataset_stem",
        "n_train",
        "subset_seed",
        "subset_mode",
        "kernel_family",
        "lengthscale",
        "nu",
        "variance",
        "reg_lambda",
        "fourier_eps",
        "nufft_tol",
        "l2_scaled",
        "precision",
        "nufft_backend",
        "precompute_chunk_size",
    )
    for field in system_fields:
        assert field in namespace["STAGE2_FEASIBILITY"]
        assert (
            namespace["STAGE2_FEASIBILITY"][field]
            == namespace["END_TO_END_TARGET"][field]
        )
        if field not in {"dataset_stem", "n_train"}:
            assert (
                target_profile["overrides"][field]
                == namespace["END_TO_END_TARGET"][field]
            )
    assert set(target_profile["overrides"]["methods"]).isdisjoint(
        {
            "nystrom",
            "rpcholesky",
            "fourier-nystrom-precond",
            "fourier-rpcholesky-precond",
            "nystrom-krr",
            "rpcholesky-krr",
        }
    )
    feasibility_path = tmp_path / "run" / "stage2_feasibility.json"
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    assert feasibility["protocol_family"] == "controlled_fixed_system"
    assert feasibility["dataset_stem"] == namespace["END_TO_END_TARGET"]["dataset_stem"]
    assert feasibility["n_train"] == 30_000_000
    assert feasibility["box_budget"] == box_budget
    assert feasibility["active_box_upper_bound"] == active_box_upper_bound
    assert feasibility["allow_frozen_topk_capacity_adaptation"] is False
    assert feasibility["inverse_max_size"] == inverse_max_size
    assert feasibility["default_inverse_max_size"] == 1_024
    assert feasibility["default_resolved_kind"] == "active-eig"
    assert feasibility["decision_basis"] == (
        "prospective declared active-box upper bound before timing"
    )
    assert (
        feasibility["methods"]["active-inverse"]["feasible"] is active_inverse_feasible
    )
    assert all(
        feasibility["methods"][method]["feasible"] is True
        for method in namespace["STAGE2_MANDATORY_METHODS"]
    )
    job_status = {
        row["job_id"]: row["status"] for row in namespace["campaign_job_rows"]
    }
    assert job_status["fixed_ab_selected_target"] == "PASS"
    assert job_status["stage1_scale_synthetic_n30m"] == "complete_with_resource_limits"
    campaign_rows = {row["job_id"]: row for row in namespace["campaign_job_rows"]}
    assert campaign_rows["fixed_ab_selected_target"]["invocation_mode"] == "executed"
    assert campaign_rows["fixed_ab_selected_target"]["resumed_case_count"] == 0
    assert campaign_rows["fixed_ab_selected_target"]["executed_case_count"] == 1
    assert campaign_rows["fixed_ab_selected_target"]["first_run_elapsed_seconds"] > 0
    assert campaign_rows["fixed_ab_selected_target"]["planned_methods"] == ",".join(
        expected_methods
    )
    assert campaign_rows["fixed_ab_selected_target"]["stage2_feasibility_path"] == str(
        feasibility_path
    )
    assert (
        "not first-run"
        in campaign_rows["fixed_ab_selected_target"]["elapsed_seconds_scope"]
    )
    assert (
        "method timing"
        in campaign_rows["fixed_ab_selected_target"]["elapsed_seconds_scope"]
    )
    assert (tmp_path / "run" / "campaign_jobs.json").is_file()

    first_run_elapsed = campaign_rows["fixed_ab_selected_target"][
        "first_run_elapsed_seconds"
    ]
    exec(
        compile(orchestrator_source, "<one-click-orchestrator-resume>", "exec"),
        namespace,
    )
    resumed_rows = {row["job_id"]: row for row in namespace["campaign_job_rows"]}
    assert (
        resumed_rows["fixed_ab_selected_target"]["invocation_mode"]
        == "resumed_existing"
    )
    assert resumed_rows["fixed_ab_selected_target"]["resumed_case_count"] == 1
    assert resumed_rows["fixed_ab_selected_target"]["executed_case_count"] == 0
    assert (
        resumed_rows["fixed_ab_selected_target"]["first_run_elapsed_seconds"]
        == first_run_elapsed
    )
    assert (
        resumed_rows["fixed_ab_selected_target"]["first_run_elapsed_seconds_source"]
        == "preserved successful campaign checkpoint"
    )
