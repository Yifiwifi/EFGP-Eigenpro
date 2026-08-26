from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

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


def test_one_click_plan_filters_unsafe_development_family() -> None:
    notebook = notebook_builder.build_notebook()
    source = _all_source(notebook)
    assert "RUN_ALL_FORMAL_EXPERIMENTS = True" in source
    assert '("Synthetic", "scale_development_masters", "synthetic_nested")' in source
    assert '("Winnebago", "scale_archived_exact", "winnebago_exact")' in source
    assert '("Winnebago", "scale_development_masters"' not in source
    assert '"job_id": "winnebago_box_budget_n10m"' in source
    assert 'PREDICTION_AUDIT_PROFILES = ["paper_10m"]' in source
    assert "check=False" in source
    assert "SCIENTIFIC_FAIL" in source


def test_committed_notebook_matches_generator() -> None:
    committed = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert committed == notebook_builder.build_notebook()


def test_one_click_orchestrator_runs_all_jobs_with_family_filters(tmp_path: Path) -> None:
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
        scientific_failure = bool(
            profile_name == "scale_development_masters"
            and int(cases[0]["expected_n_train"]) == 30_000_000
        )
        status_rows = []
        for case_index, case in enumerate(cases):
            run_dir = output_root / case["id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            status_row = {"case_id": case["id"], "status": "completed"}
            if scientific_failure and case_index == 0:
                status_row["status"] = "completed_with_ineligible_methods"
                status_row["ineligible_methods"] = ["default"]
            status_rows.append(status_row)
            if profile_name.startswith("scale_"):
                pd.DataFrame(
                    [
                        {
                            "method": method,
                            "performance_claim_eligible": not (
                                scientific_failure and method == "default"
                            ),
                            "converged_repeats": (
                                0 if scientific_failure and method == "default" else 5
                            ),
                            "true_relres_max": (
                                float("nan")
                                if scientific_failure and method == "default" else 1e-8
                            ),
                        }
                        for method in ("cg", "default", "full-eig")
                    ]
                ).to_csv(run_dir / "matched_summary.csv", index=False)
                (run_dir / "experiment_config.json").write_text(
                    json.dumps({"measured_repeats": 5, "tol": 1e-7}),
                    encoding="utf-8",
                )
        (output_root / "suite_status.json").write_text(
            json.dumps(status_rows),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=2 if scientific_failure else 0
        )

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
    assert len(records) == 17
    development = [
        record for record in records
        if record["suite_profile"] == "scale_development_masters"
    ]
    archived = [
        record for record in records
        if record["suite_profile"] == "scale_archived_exact"
    ]
    assert {record["dataset_family"] for record in development} == {"Synthetic"}
    assert {record["dataset_family"] for record in archived} == {"Winnebago"}
    assert len(development) == 2
    assert len(archived) == 4
    job_status = {row["job_id"]: row["status"] for row in namespace["campaign_job_rows"]}
    assert job_status["synthetic_nested_30m"] == "SCIENTIFIC_FAIL"
    assert job_status["synthetic_nested_100m"] == "SKIPPED_UPSTREAM_GATE"
    assert job_status["synthetic_nested_300m"] == "SKIPPED_UPSTREAM_GATE"
    assert job_status["winnebago_box_budget_n10m"] == "PASS"
    assert job_status["winnebago_exact_300m"] == "PASS"
    assert (tmp_path / "run" / "campaign_jobs.json").is_file()
