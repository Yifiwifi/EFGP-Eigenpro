from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    build_colab_all_experiments_notebook as notebook_builder,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.prediction_audit import (
    PREDICTION_AUDIT_COMPLETION_FILENAME,
    prediction_source_manifest,
)


matplotlib.use("Agg", force=True)


def _cell_source(notebook: dict, marker: str) -> str:
    matches = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    ]
    assert len(matches) == 1, marker
    return matches[0]


def _write_case(
    run_dir: Path,
    *,
    dataset_stem: str,
    n_train: int,
    speedup: float,
    reg_lambda: float = 0.1,
    lengthscale: float = 0.1,
    source_bundle_sha256: str = "source",
    content_sha256: str = "content",
    metadata_sha256: str = "metadata",
    include_component_hashes: bool = True,
    box_budget: int = 8192,
    box_size: int = 8099,
    system_id: str | None = None,
    extra_methods: tuple[str, ...] = (),
) -> None:
    run_dir.mkdir(parents=True)
    config = {
        "kernel_family": "matern",
        "lengthscale": lengthscale,
        "nu": 1.5,
        "variance": 1.0,
        "reg_lambda": reg_lambda,
        "fourier_eps": 1e-5,
        "nufft_tol": 1e-10,
        "l2_scaled": True,
        "tol": 1e-7,
        "maxiter": 20_000,
        "precision": "fp64",
        "subset_mode": "prefix",
        "subset_seed": 0,
        "score_tau": 1.0,
        "box_budget": box_budget,
        "inverse_max_size": 1024,
        "rank": 256,
        "nystrom_rank": 256,
        "rpcholesky_rank": 256,
        "eig_tol": 1e-3,
        "eig_maxiter": 1280,
        "measured_repeats": 5,
        "warmup_repeats": 1,
        "precompute_chunk_size": 1_000_000,
        "strict_gpu_eig": True,
        "methods": ["cg", "default", *extra_methods],
    }
    manifest = {
        "system_id": system_id or hashlib.sha256(str(run_dir).encode()).hexdigest(),
        "dataset_stem": dataset_stem,
        "n_train": n_train,
        "system_unchanged": True,
        "nufft_backend_resolved": "cufinufft",
        "nufft_stage": "cufinufft",
        "precision_mode": "fp64",
        "reg_lambda": reg_lambda,
        "device_name": "NVIDIA A100-SXM4-40GB",
        "compute_capability": "8.0",
        "timing_runtime_sha256": "a100-runtime-hash",
        "setup_seconds": 1.25,
        "prepared_system_loaded_from_artifact": False,
        "setup_timing_source": "measured_in_current_process",
        "setup_inclusive_timing_eligible": True,
        "source_bundle_sha256": source_bundle_sha256,
        "dataset_content_index_sha256": content_sha256,
        "dataset_metadata_sha256": metadata_sha256,
    }
    if include_component_hashes:
        manifest.update({
            "weights_sha256": "weights-hash",
            "gf_sha256": "gf-hash",
            "rhs_sha256": "rhs-hash",
            "rhs_storage_sha256": "rhs-storage-hash",
        })
    common = {
        "measured_repeats": 5,
        "converged_repeats": 5,
        "performance_claim_eligible": True,
        "true_relres_max": 9e-8,
        "cold_speedup_min": 1.0,
        "cold_speedup_max": 1.0,
        "shared_fourier_setup_plus_method_speedup_median": 1.0,
        "iterations_median": 4000,
        "iterations_min": 4000,
        "iterations_max": 4000,
        "build_plus_solve_seconds_median": 4.0,
        "build_seconds_median": 0.0,
        "build_seconds_max": 0.0,
        "preconditioner_storage_bytes": None,
        "box_size": box_size,
    }
    rows = [
        {**common, "method": "cg", "cold_speedup_median": 1.0},
        {
            **common,
            "method": "default",
            "cold_speedup_median": speedup,
            "cold_speedup_min": speedup * 0.95,
            "cold_speedup_max": speedup * 1.05,
            "shared_fourier_setup_plus_method_speedup_median": speedup * 0.9,
            "iterations_median": 800,
            "iterations_min": 790,
            "iterations_max": 810,
            "build_plus_solve_seconds_median": 1.3,
            "build_seconds_median": 0.5,
            "build_seconds_max": 0.55,
            "preconditioner_storage_bytes": 64 * 2**20,
        },
    ]
    for method in extra_methods:
        rows.append({
            **common,
            "method": method,
            "cold_speedup_median": speedup * 1.1,
            "cold_speedup_min": speedup * 1.04,
            "cold_speedup_max": speedup * 1.16,
            "shared_fourier_setup_plus_method_speedup_median": speedup,
            "iterations_median": 600,
            "iterations_min": 590,
            "iterations_max": 610,
            "build_plus_solve_seconds_median": 1.0,
            "build_seconds_median": 0.5,
            "build_seconds_max": 0.55,
            "preconditioner_storage_bytes": 128 * 2**20,
        })
    (run_dir / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "system_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "run_complete.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(rows).to_csv(run_dir / "matched_summary.csv", index=False)


def _write_prediction_audit(run_dir: Path) -> None:
    config_path = run_dir / "experiment_config.json"
    manifest = json.loads((run_dir / "system_manifest.json").read_text(encoding="utf-8"))
    rows = [
        {
            "system_id": manifest["system_id"],
            "dataset": manifest["dataset_stem"],
            "method": "cg",
            "method_kind": "cg",
            "solve_status": "converged",
            "true_relres": 8e-8,
            "test_rmse": 0.2,
            "test_rmse_ratio_vs_cg": 1.0,
            "test_rmse_diff_vs_cg": 0.0,
            "prediction_equivalent_to_cg": True,
            "n_test": 2_500_000,
        },
        {
            "system_id": manifest["system_id"],
            "dataset": manifest["dataset_stem"],
            "method": "default",
            "method_kind": "active-eig",
            "solve_status": "converged",
            "true_relres": 9e-8,
            "test_rmse": 0.20000002,
            "test_rmse_ratio_vs_cg": 1.0000001,
            "test_rmse_diff_vs_cg": 2e-8,
            "prediction_equivalent_to_cg": True,
            "n_test": 2_500_000,
        },
    ]
    audit_dir = run_dir / "prediction_audit"
    audit_dir.mkdir()
    csv_path = audit_dir / "prediction_audit.csv"
    json_path = audit_dir / "prediction_audit.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    prediction_source_sha256 = prediction_source_manifest()[
        "prediction_source_bundle_sha256"
    ]
    payload = {
        "schema_version": 2,
        "system_id": manifest["system_id"],
        "weights_sha256": manifest["weights_sha256"],
        "gf_sha256": manifest["gf_sha256"],
        "rhs_sha256": manifest["rhs_sha256"],
        "rhs_storage_sha256": manifest["rhs_storage_sha256"],
        "reg_lambda": manifest["reg_lambda"],
        "audit_rebuilt_system": False,
        "timing_system_reused": True,
        "timing_solutions_reused": True,
        "timing_system_hashes_exact": True,
        "timing_solution_hashes_verified": True,
        "audit_solve_count": 0,
        "audit_solves_per_method": 0,
        "audit_pass": True,
        "prediction_source_bundle_sha256": prediction_source_sha256,
        "test_dataset_content_index_verified": True,
        "test_dataset_metadata_verified": True,
        "strict_prediction_nufft": True,
        "observed_prediction_nufft_stages": ["cufinufft"],
        "dataset_content_index_sha256": manifest["dataset_content_index_sha256"],
        "source_bundle_sha256": manifest["source_bundle_sha256"],
        "config_source_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "evaluated_n_test": 2_500_000,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    completion = {
        "schema_version": 1,
        "system_id": manifest["system_id"],
        "methods": ["cg", "default"],
        "row_count": len(rows),
        "audit_pass": True,
        "evaluated_n_test": 2_500_000,
        "prediction_source_bundle_sha256": prediction_source_sha256,
        "prediction_audit_json": json_path.name,
        "prediction_audit_json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "prediction_audit_csv": csv_path.name,
        "prediction_audit_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    (audit_dir / PREDICTION_AUDIT_COMPLETION_FILENAME).write_text(
        json.dumps(completion), encoding="utf-8"
    )


def test_reporting_uses_exact_selected_cases_and_isolates_profiles(tmp_path, monkeypatch) -> None:
    controlled_root = tmp_path / "controlled_fixed_system"
    paper_run = controlled_root / "paper_10m" / "manitowoc_n10m_matern"
    stale_paper_run = controlled_root / "paper_10m" / "old_manitowoc_n10m_matern"
    stale_run = controlled_root / "screen_10m" / "screen_manitowoc_n10m"
    oat_root = controlled_root / "winnebago_oat_n10m"
    oat_ref = oat_root / "winnebago_oat_ref_lam0p1_ell0p1"
    oat_lambda = oat_root / "winnebago_oat_lam0p01"
    oat_lengthscale = oat_root / "winnebago_oat_ell0p2"
    stem = "USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000"
    _write_case(paper_run, dataset_stem=stem, n_train=10_000_000, speedup=3.2)
    _write_prediction_audit(paper_run)
    _write_case(
        stale_paper_run, dataset_stem=stem, n_train=10_000_000, speedup=9.7,
    )
    _write_case(
        stale_run, dataset_stem=stem, n_train=10_000_000, speedup=9.9,
        include_component_hashes=False,
    )
    _write_case(oat_ref, dataset_stem=stem, n_train=10_000_000, speedup=2.9)
    _write_case(
        oat_lambda, dataset_stem=stem, n_train=10_000_000, speedup=3.4, reg_lambda=0.01,
    )
    _write_case(
        oat_lengthscale, dataset_stem=stem, n_train=10_000_000, speedup=2.8, lengthscale=0.2,
    )
    output_root = tmp_path / "report"
    output_root.mkdir()
    selected_case_records = [{
        "output_group": "paper_10m",
        "suite_profile": "paper_10m",
        "case_id": paper_run.name,
        "run_dir": paper_run,
        "scale_role": "independent replication",
    }]
    notebook = notebook_builder.build_notebook()
    namespace = {
        "Path": Path,
        "pd": pd,
        "json": json,
        "hashlib": hashlib,
        "CONTROLLED_OUTPUT_ROOT": controlled_root,
        "DRIVE_RUN_ROOT": output_root,
        "selected_case_records": selected_case_records,
        "legacy_all": pd.DataFrame(),
        "display": lambda _value: None,
    }

    audit_source = _cell_source(notebook, "ignored_stale_dirs =")
    exec(compile(audit_source, "audit-cell", "exec"), namespace)
    audit = namespace["controlled_artifact_audit"]
    assert audit[["output_group", "case", "status"]].to_dict("records") == [{
        "output_group": "paper_10m",
        "case": paper_run.name,
        "status": "PASS",
    }]
    assert audit.loc[0, "rhs_storage_sha256"] == "rhs-storage-hash"
    assert audit.loc[0, "device_name"] == "NVIDIA A100-SXM4-40GB"
    assert audit.loc[0, "compute_capability"] == "8.0"
    assert audit.loc[0, "timing_runtime_sha256"] == "a100-runtime-hash"
    assert set(namespace["ignored_stale_dirs"]) == {
        stale_paper_run.resolve(), stale_run.resolve(), oat_ref.resolve(),
        oat_lambda.resolve(), oat_lengthscale.resolve(),
    }

    index_source = _cell_source(notebook, "def dataset_family_label")
    exec(compile(index_source, "index-cell", "exec"), namespace)
    catalog = namespace["controlled_catalog"]
    assert set(catalog["output_group"]) == {
        "paper_10m", "screen_10m", "winnebago_oat_n10m",
    }
    assert catalog.loc[catalog["selected_in_this_invocation"], "output_group"].eq("paper_10m").all()
    assert catalog.loc[catalog["case_id"].eq(paper_run.name), "rhs_sha256"].eq("rhs-hash").all()
    paper_catalog = catalog.loc[catalog["case_id"].eq(paper_run.name)]
    assert paper_catalog["rhs_storage_sha256"].eq("rhs-storage-hash").all()
    assert paper_catalog["cfg_reg_lambda"].eq(0.1).all()
    assert paper_catalog["device_name"].eq("NVIDIA A100-SXM4-40GB").all()
    assert paper_catalog["compute_capability"].eq("8.0").all()
    assert paper_catalog["timing_runtime_sha256"].eq("a100-runtime-hash").all()
    assert paper_catalog["prepared_system_loaded_from_artifact"].eq(False).all()
    assert paper_catalog["setup_inclusive_timing_eligible"].eq(True).all()
    assert catalog.loc[catalog["case_id"].eq(stale_run.name), "rhs_sha256"].isna().all()
    assert not catalog.loc[
        catalog["case_id"].eq(stale_paper_run.name), "selected_in_this_invocation"
    ].any()

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    plot_source = _cell_source(notebook, "METHOD_ORDER = [")
    exec(compile(plot_source, "plot-cell", "exec"), namespace)
    paper_plot = namespace["paper_plot"]
    assert set(paper_plot["output_group"]) == {"paper_10m"}
    assert paper_plot.set_index("method").loc["default", "cold_speedup_median"] == 3.2
    assert (output_root / "controlled_10m_method_speedup.png").is_file()
    assert (output_root / "controlled_10m_speed_memory_pareto.png").is_file()
    assert not (output_root / "winnebago_oat_10m_speed_memory.png").exists()
    assert (output_root / "prediction_accuracy_vs_cg.png").is_file()
    prediction_summary = pd.read_csv(output_root / "prediction_accuracy_summary.csv")
    assert set(prediction_summary["method"]) == {"cg", "default"}
    assert prediction_summary["exact_timing_system_match"].eq(True).all()
    assert prediction_summary["timing_solutions_reused"].eq(True).all()
    assert prediction_summary["prediction_completion_valid"].eq(True).all()
    assert prediction_summary["audit_rhs_storage_sha256"].eq(
        "rhs-storage-hash"
    ).all()


def test_controlled_audit_rejects_truncated_method_summary(tmp_path: Path) -> None:
    controlled_root = tmp_path / "controlled_fixed_system"
    run_dir = controlled_root / "paper_10m" / "synthetic_n10m_matern"
    _write_case(
        run_dir,
        dataset_stem="synthetic_true_func_2d_ntrain10000000",
        n_train=10_000_000,
        speedup=3.0,
    )
    summary_path = run_dir / "matched_summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[summary["method"].eq("cg")].to_csv(summary_path, index=False)
    output_root = tmp_path / "report"
    output_root.mkdir()
    selected_case_records = [{
        "output_group": "paper_10m",
        "suite_profile": "paper_10m",
        "case_id": run_dir.name,
        "run_dir": run_dir,
    }]
    notebook = notebook_builder.build_notebook()
    namespace = {
        "Path": Path,
        "pd": pd,
        "json": json,
        "CONTROLLED_OUTPUT_ROOT": controlled_root,
        "DRIVE_RUN_ROOT": output_root,
        "selected_case_records": selected_case_records,
        "display": lambda _value: None,
    }
    exec(
        compile(_cell_source(notebook, "ignored_stale_dirs ="), "audit-cell", "exec"),
        namespace,
    )
    status = namespace["controlled_artifact_audit"].loc[0, "status"]
    assert status.startswith("FAIL:")
    assert "method coverage differs" in status


def _execute_reporting_cells(
    controlled_root: Path,
    output_root: Path,
    *,
    selected_run_dirs: tuple[Path, ...] = (),
) -> tuple[dict, str]:
    notebook = notebook_builder.build_notebook()
    selected_case_records = [
        {
            "output_group": run_dir.resolve().relative_to(
                controlled_root.resolve()
            ).parts[0],
            "suite_profile": run_dir.resolve().relative_to(
                controlled_root.resolve()
            ).parts[0],
            "case_id": run_dir.name,
            "run_dir": run_dir,
            "scale_role": None,
        }
        for run_dir in selected_run_dirs
    ]
    namespace = {
        "Path": Path,
        "pd": pd,
        "json": json,
        "hashlib": hashlib,
        "CONTROLLED_OUTPUT_ROOT": controlled_root,
        "DRIVE_RUN_ROOT": output_root,
        "selected_case_records": selected_case_records,
        "legacy_all": pd.DataFrame(),
        "display": lambda _value: None,
    }
    index_source = _cell_source(notebook, "def dataset_family_label")
    exec(compile(index_source, "index-cell", "exec"), namespace)
    return namespace, _cell_source(notebook, "METHOD_ORDER = [")


def test_plot_cell_skips_when_no_controlled_rows_are_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled_root = tmp_path / "controlled_fixed_system"
    output_root = tmp_path / "report"
    output_root.mkdir()
    namespace, plot_source = _execute_reporting_cells(controlled_root, output_root)

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    exec(compile(plot_source, "plot-cell", "exec"), namespace)

    assert namespace["controlled_plot"].empty
    assert namespace["scale_plot"].empty
    assert namespace["GENERATED_PLOT_PATHS"] == []


def test_scale_plot_accepts_one_archived_series_with_exact_per_n_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled_root = tmp_path / "controlled_fixed_system"
    scale_root = controlled_root / "scale_archived_exact"
    _write_case(
        scale_root / "synthetic_n10m",
        dataset_stem="synthetic_true_func_2d_ntrain10000000",
        n_train=10_000_000,
        speedup=3.0,
        content_sha256="content-10m",
        metadata_sha256="metadata-10m",
        extra_methods=("full-eig",),
    )
    _write_case(
        scale_root / "synthetic_n30m",
        dataset_stem="synthetic_true_func_2d_ntrain30000000",
        n_train=30_000_000,
        speedup=3.5,
        content_sha256="content-30m",
        metadata_sha256="metadata-30m",
    )
    output_root = tmp_path / "report"
    output_root.mkdir()
    selected = (
        scale_root / "synthetic_n10m",
        scale_root / "synthetic_n30m",
    )
    namespace, plot_source = _execute_reporting_cells(
        controlled_root, output_root, selected_run_dirs=selected
    )

    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    monkeypatch.setattr(plt, "show", lambda: None)
    errorbar_calls = []
    original_errorbar = Axes.errorbar

    def tracked_errorbar(self, *args, **kwargs):
        errorbar_calls.append(kwargs.get("yerr"))
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", tracked_errorbar)
    exec(compile(plot_source, "plot-cell", "exec"), namespace)
    assert (output_root / "scale_archived_exact_cold_speedup.png").is_file()
    assert (output_root / "scale_archived_exact_setup_inclusive_speedup.png").is_file()
    assert errorbar_calls and all(value is not None for value in errorbar_calls)
    scale_catalog = namespace["scale_plot"]
    assert scale_catalog["device_name"].eq("NVIDIA A100-SXM4-40GB").all()
    assert scale_catalog["compute_capability"].eq("8.0").all()
    assert scale_catalog["setup_inclusive_timing_eligible"].eq(True).all()
    availability = pd.read_csv(output_root / "scale_method_availability.csv")
    row_30m = availability.loc[availability["N"].eq(30_000_000)].iloc[0]
    assert bool(row_30m["is_subset_vs_profile_union"])
    assert row_30m["methods"] == "cg,default"


def test_scale_plot_rejects_mixed_source_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled_root = tmp_path / "controlled_fixed_system"
    scale_root = controlled_root / "scale_archived_exact"
    _write_case(
        scale_root / "synthetic_n10m",
        dataset_stem="synthetic_true_func_2d_ntrain10000000",
        n_train=10_000_000,
        speedup=3.0,
        source_bundle_sha256="source-a",
    )
    _write_case(
        scale_root / "synthetic_n30m",
        dataset_stem="synthetic_true_func_2d_ntrain30000000",
        n_train=30_000_000,
        speedup=3.5,
        source_bundle_sha256="source-b",
    )
    output_root = tmp_path / "report"
    output_root.mkdir()
    selected = (
        scale_root / "synthetic_n10m",
        scale_root / "synthetic_n30m",
    )
    namespace, plot_source = _execute_reporting_cells(
        controlled_root, output_root, selected_run_dirs=selected
    )

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    with pytest.raises(RuntimeError, match="mixes or lacks source bundles"):
        exec(compile(plot_source, "plot-cell", "exec"), namespace)


def test_scale_plot_rejects_mixed_dataset_series(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled_root = tmp_path / "controlled_fixed_system"
    scale_root = controlled_root / "scale_archived_exact"
    _write_case(
        scale_root / "synthetic_n10m",
        dataset_stem="synthetic_true_func_2d_ntrain10000000",
        n_train=10_000_000,
        speedup=3.0,
    )
    _write_case(
        scale_root / "synthetic_variant_n30m",
        dataset_stem="synthetic_true_func_2d_variant_ntrain30000000",
        n_train=30_000_000,
        speedup=3.5,
    )
    output_root = tmp_path / "report"
    output_root.mkdir()
    selected = (
        scale_root / "synthetic_n10m",
        scale_root / "synthetic_variant_n30m",
    )
    namespace, plot_source = _execute_reporting_cells(
        controlled_root, output_root, selected_run_dirs=selected
    )

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    with pytest.raises(RuntimeError, match="mixes or lacks dataset series"):
        exec(compile(plot_source, "plot-cell", "exec"), namespace)


def test_box_budget_report_checks_fixed_system_and_writes_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled_root = tmp_path / "controlled_fixed_system"
    budget_root = controlled_root / "winnebago_box_budget_n10m"
    stem = "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain10000000"
    for budget, actual_size, speedup in [
        (4096, 4000, 2.4),
        (8192, 8099, 3.0),
        (16384, 16000, 3.2),
    ]:
        _write_case(
            budget_root / f"winnebago_box_budget_{budget}_n10m",
            dataset_stem=stem,
            n_train=10_000_000,
            speedup=speedup,
            box_budget=budget,
            box_size=actual_size,
            system_id="shared-fixed-system",
        )
    output_root = tmp_path / "report"
    output_root.mkdir()
    selected = tuple(
        budget_root / f"winnebago_box_budget_{budget}_n10m"
        for budget in (4096, 8192, 16384)
    )
    namespace, plot_source = _execute_reporting_cells(
        controlled_root, output_root, selected_run_dirs=selected
    )

    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    monkeypatch.setattr(plt, "show", lambda: None)
    errorbar_calls = []
    original_errorbar = Axes.errorbar

    def tracked_errorbar(self, *args, **kwargs):
        errorbar_calls.append(kwargs.get("yerr"))
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", tracked_errorbar)
    exec(compile(plot_source, "plot-cell", "exec"), namespace)
    assert namespace["BOX_BUDGET_SYSTEM_MATCH"] is True
    assert len(errorbar_calls) == 2
    assert all(value is not None for value in errorbar_calls)
    assert (output_root / "winnebago_box_budget_10m.png").is_file()
    summary = pd.read_csv(output_root / "winnebago_box_budget_10m_summary.csv")
    assert set(summary["cfg_box_budget"]) == {4096, 8192, 16384}
    assert summary["system_id"].eq("shared-fixed-system").all()
    assert summary["rhs_storage_sha256"].eq("rhs-storage-hash").all()
    assert summary["cfg_reg_lambda"].eq(0.1).all()
    assert summary["device_name"].eq("NVIDIA A100-SXM4-40GB").all()
    assert summary["compute_capability"].astype(str).eq("8.0").all()
