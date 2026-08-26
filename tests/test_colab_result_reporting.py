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
    }
    manifest = {
        "system_id": system_id or hashlib.sha256(str(run_dir).encode()).hexdigest(),
        "dataset_stem": dataset_stem,
        "n_train": n_train,
        "system_unchanged": True,
        "nufft_backend_resolved": "cufinufft",
        "nufft_stage": "cufinufft",
        "precision_mode": "fp64",
        "source_bundle_sha256": source_bundle_sha256,
        "dataset_content_index_sha256": content_sha256,
        "dataset_metadata_sha256": metadata_sha256,
    }
    if include_component_hashes:
        manifest.update({
            "weights_sha256": "weights-hash",
            "gf_sha256": "gf-hash",
            "rhs_sha256": "rhs-hash",
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
            "build_plus_solve_seconds_median": 1.3,
            "build_seconds_median": 0.5,
            "build_seconds_max": 0.55,
            "preconditioner_storage_bytes": 64 * 2**20,
        },
    ]
    (run_dir / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "system_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "run_complete.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(rows).to_csv(run_dir / "matched_summary.csv", index=False)


def test_reporting_uses_exact_selected_cases_and_isolates_profiles(tmp_path, monkeypatch) -> None:
    controlled_root = tmp_path / "controlled_fixed_system"
    paper_run = controlled_root / "paper_10m" / "manitowoc_n10m_matern"
    stale_run = controlled_root / "screen_10m" / "screen_manitowoc_n10m"
    oat_root = controlled_root / "winnebago_oat_n10m"
    oat_ref = oat_root / "winnebago_oat_ref_lam0p1_ell0p1"
    oat_lambda = oat_root / "winnebago_oat_lam0p01"
    oat_lengthscale = oat_root / "winnebago_oat_ell0p2"
    stem = "USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000"
    _write_case(paper_run, dataset_stem=stem, n_train=10_000_000, speedup=3.2)
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
    assert set(namespace["ignored_stale_dirs"]) == {
        stale_run.resolve(), oat_ref.resolve(), oat_lambda.resolve(), oat_lengthscale.resolve(),
    }

    index_source = _cell_source(notebook, "def dataset_family_label")
    exec(compile(index_source, "index-cell", "exec"), namespace)
    catalog = namespace["controlled_catalog"]
    assert set(catalog["output_group"]) == {
        "paper_10m", "screen_10m", "winnebago_oat_n10m",
    }
    assert catalog.loc[catalog["selected_in_this_invocation"], "output_group"].eq("paper_10m").all()
    assert catalog.loc[catalog["case_id"].eq(paper_run.name), "rhs_sha256"].eq("rhs-hash").all()
    assert catalog.loc[catalog["case_id"].eq(stale_run.name), "rhs_sha256"].isna().all()

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    plot_source = _cell_source(notebook, "METHOD_ORDER = [")
    exec(compile(plot_source, "plot-cell", "exec"), namespace)
    paper_plot = namespace["paper_plot"]
    assert set(paper_plot["output_group"]) == {"paper_10m"}
    assert paper_plot.set_index("method").loc["default", "cold_speedup_median"] == 3.2
    assert (output_root / "controlled_10m_method_speedup.png").is_file()
    assert (output_root / "controlled_10m_speed_memory_pareto.png").is_file()
    assert (output_root / "winnebago_oat_10m_speed_memory.png").is_file()


def _execute_reporting_cells(controlled_root: Path, output_root: Path) -> tuple[dict, str]:
    notebook = notebook_builder.build_notebook()
    namespace = {
        "Path": Path,
        "pd": pd,
        "json": json,
        "hashlib": hashlib,
        "CONTROLLED_OUTPUT_ROOT": controlled_root,
        "DRIVE_RUN_ROOT": output_root,
        "selected_case_records": [],
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
    namespace, plot_source = _execute_reporting_cells(controlled_root, output_root)

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    exec(compile(plot_source, "plot-cell", "exec"), namespace)
    assert (output_root / "scale_archived_exact_cold_speedup.png").is_file()


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
    namespace, plot_source = _execute_reporting_cells(controlled_root, output_root)

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
    namespace, plot_source = _execute_reporting_cells(controlled_root, output_root)

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
    namespace, plot_source = _execute_reporting_cells(controlled_root, output_root)

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    exec(compile(plot_source, "plot-cell", "exec"), namespace)
    assert namespace["BOX_BUDGET_SYSTEM_MATCH"] is True
    assert (output_root / "winnebago_box_budget_10m.png").is_file()
    summary = pd.read_csv(output_root / "winnebago_box_budget_10m_summary.csv")
    assert set(summary["cfg_box_budget"]) == {4096, 8192, 16384}
