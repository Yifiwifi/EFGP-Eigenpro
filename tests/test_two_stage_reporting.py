from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark import (
    _artifact_array_descriptor,
    system_component_fingerprints,
    system_fingerprint,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end import (
    END_TO_END_METHODS,
    EndToEndConfig,
    TIMING_SCOPE as RUNNER_STAGE1_TIMING_SCOPE,
    summarize_pipeline_rows,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end_suite import (
    BUDGET_ADAPTIVE_PARAMETER_POLICY,
    BUDGET_ADAPTIVE_PARAMETER_SOURCE,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.two_stage_reporting import (
    EXPECTED_SOLVER_TOTAL_DEFINITION,
    EXPECTED_STAGE1_TIMING_SCOPE,
    ReportSchemaError,
    STAGE1_FORMAL_METHODS,
    STAGE1_PROTOCOL,
    STAGE1_TABLE_COLUMNS,
    TwoStageReportConfig,
    build_two_stage_report,
    load_stage1_summaries,
)


def test_stage1_timing_scope_contract_matches_runner() -> None:
    assert EXPECTED_STAGE1_TIMING_SCOPE == RUNNER_STAGE1_TIMING_SCOPE


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


FROZEN_METHOD_CONFIG = {
    "rank": 256,
    "full_eig_rank": 256,
    "active_topk": 4096,
    "expected_active_box_size": 5329,
    "parameter_selection_policy": "historical_selected_transfer_no_current_scan",
    "parameter_source": "test fixture: frozen historical selected configuration",
}


def _suite_payload() -> dict[str, object]:
    base = {
        "dataset_stem": "synthetic_master",
        "n_train": 10_000_000,
        "subset_seed": 0,
        "subset_mode": "prefix",
        "kernel_family": "se",
        "nu": 1.5,
        "variance": 1.0,
        "reg_lambda": 0.1,
        "lengthscale": 0.1,
        "fourier_eps": 1e-5,
        "nufft_tol": 1e-10,
        "l2_scaled": True,
        "precision": "fp64",
        "nufft_backend": "cufinufft",
        "precompute_chunk_size": 1_000_000,
        "box_budget": 8192,
        **FROZEN_METHOD_CONFIG,
        "inverse_max_size": 1024,
        "accuracy_max_rmse": 1.1,
        "accuracy_min_r2": 0.5,
    }
    return {
        "schema_version": 1,
        "protocol_family": "end_to_end_krr",
        "base": base,
        "stage2_fixed_ab": {
            "inverse_max_size": 16384,
            "default_inverse_max_size": 1024,
        },
        "profiles": {
            "scale_10m_300m": {
                "cases": [
                    {
                        "id": f"synthetic_n{n // 1_000_000}m",
                        "dataset_family": "Synthetic",
                        "dataset_stem": "synthetic_master",
                        "n_train": n,
                    }
                    for n in (10_000_000, 30_000_000, 100_000_000, 300_000_000)
                ]
            },
            "robustness_at_selected_target": {
                "lambda_values": [0.01, 0.1, 1.0],
                "lengthscale_values": [0.05, 0.1, 0.2],
                "box_budget_values": [4096, 8192, 16384],
                "datasets": [
                    {
                        "dataset_family": "Synthetic",
                        "dataset_stem": "synthetic_master",
                        "accuracy_max_rmse": 1.1,
                    },
                    {
                        "dataset_family": "Winnebago",
                        "dataset_stems_by_n_train": {"10000000": "winnebago_10m"},
                        "accuracy_max_rmse": 2.0,
                    },
                ],
            },
        },
        "target_selection": {
            "cg_iteration_min": 3000,
            "cg_iteration_max": 6000,
            "dataset_priority": ["synthetic_master"],
            "allowed_resource_limit_methods": ["rpcholesky-krr"],
        },
    }


def _target_payload() -> dict[str, object]:
    return {
        "dataset_stem": "synthetic_master",
        "declared_dataset_family": "Synthetic",
        "n_train": 10_000_000,
        "subset_seed": 0,
        "subset_mode": "prefix",
        "kernel_family": "se",
        "nu": 1.5,
        "variance": 1.0,
        "reg_lambda": 0.1,
        "lengthscale": 0.1,
        "fourier_eps": 1e-5,
        "nufft_tol": 1e-10,
        "l2_scaled": True,
        "precision": "fp64",
        "nufft_backend": "cufinufft",
        "precompute_chunk_size": 1_000_000,
        **FROZEN_METHOD_CONFIG,
        "box_budget": 8192,
        "accuracy_max_rmse": 1.1,
        "accuracy_min_r2": 0.5,
        "selection_rule": (
            "largest N with all six declared pipeline rows present; successful "
            "EFGP/Nystrom rows (a declared RPCholesky hardware resource-limit row "
            "is retained as a valid scalability outcome); full-eig/ours inside "
            "the declared broad absolute usable-quality range (the "
            "reference-equivalence label is descriptive only); and EFGP-CG "
            "iterations in the predeclared [3000,6000] interval"
        ),
    }


def _stage1_method_rows(
    *,
    profile: str,
    case_id: str,
    axes: list[str],
    dataset_stem: str,
    dataset_family: str,
    n_train: int,
    reg_lambda: float = 0.1,
    lengthscale: float = 0.1,
    box_budget: int = 8192,
    accuracy_max_rmse: float = 1.1,
) -> list[dict[str, object]]:
    totals = {
        "nystrom-krr": 8.0,
        "rpcholesky-krr": 9.0,
        "efgp-standard-cg": 12.0,
        "efgp-standard-jacobi": 11.0,
        "efgp-standard-full-eig": 10.0,
        "ours-binned-default": 5.0,
    }
    is_robustness = profile == "robustness_at_selected_target"
    is_box_budget_axis = any(str(axis).startswith("box_budget_") for axis in axes)
    configured_active_topk = (
        None if is_box_budget_axis else FROZEN_METHOD_CONFIG["active_topk"]
    )
    configured_expected_box_size = (
        None
        if is_robustness
        else FROZEN_METHOD_CONFIG["expected_active_box_size"]
    )
    parameter_policy = (
        BUDGET_ADAPTIVE_PARAMETER_POLICY
        if is_box_budget_axis
        else FROZEN_METHOD_CONFIG["parameter_selection_policy"]
    )
    parameter_source = (
        BUDGET_ADAPTIVE_PARAMETER_SOURCE
        if is_box_budget_axis
        else FROZEN_METHOD_CONFIG["parameter_source"]
    )
    output: list[dict[str, object]] = []
    for method in sorted(STAGE1_FORMAL_METHODS):
        total = totals[method]
        accuracy = method != "nystrom-krr"
        output.append(
            {
                "protocol_family": "end_to_end_krr",
                "timing_scope": EXPECTED_STAGE1_TIMING_SCOPE,
                "suite_profile": profile,
                "case_id": case_id,
                "robustness_axes": json.dumps(axes),
                "dataset_stem": dataset_stem,
                "declared_dataset_family": dataset_family,
                "dataset_family": dataset_family,
                "n_train": n_train,
                "subset_seed": 0,
                "subset_mode": "prefix",
                "kernel_family": "se",
                "nu": 1.5,
                "variance": 1.0,
                "reg_lambda": reg_lambda,
                "lengthscale": lengthscale,
                "fourier_eps": 1e-5,
                "nufft_tol": 1e-10,
                "l2_scaled": True,
                "precision": "fp64",
                "nufft_backend": "cufinufft",
                "precompute_chunk_size": 1_000_000,
                "box_budget": box_budget,
                "configured_active_rank": FROZEN_METHOD_CONFIG["rank"],
                "configured_full_eig_rank": FROZEN_METHOD_CONFIG["full_eig_rank"],
                "configured_active_topk": configured_active_topk,
                "configured_expected_active_box_size": configured_expected_box_size,
                "parameter_selection_policy": parameter_policy,
                "parameter_source": parameter_source,
                "method": method,
                "status": "ok",
                "execution_eligible": True,
                "usability_eligible": accuracy,
                "usability_evaluated_repeats": 5,
                "usability_passed_repeats": 5 if accuracy else 0,
                "reference_equivalent": accuracy,
                "reference_evaluated_repeats": 5,
                "reference_equivalent_repeats": 5 if accuracy else 0,
                "quality_qualified_performance_eligible": accuracy,
                "ours_speedup_complete_pairing": True,
                "ours_speedup_claim_eligible": accuracy,
                "ours_total_speedup": total / 5.0,
                "ours_setup_speedup": total / 5.0,
                "ours_solving_speedup": total / 5.0,
                "comparison_rmse_ratio_to_ours": 1.0 if accuracy else 1.5,
                "comparison_rmse_delta_from_ours": 0.0 if accuracy else 0.5,
                "accuracy_eligible": accuracy,
                "performance_claim_eligible": accuracy,
                "accuracy_max_rmse": accuracy_max_rmse,
                "accuracy_min_r2": 0.5,
                "accuracy_relative_tolerance": 0.01,
                "measured_repeats": 5,
                "expected_measured_repeats": 5,
                "successful_repeats": 5,
                "accuracy_evaluated_repeats": 5,
                "accuracy_passed_repeats": 5 if accuracy else 0,
                "resource_required_bytes": "",
                "resource_effective_cap_bytes": "",
                "resource_declared_cap_bytes": "",
                "resource_available_device_bytes": "",
                "setup_seconds_median": total * 0.4,
                "solving_phase_seconds_median": total * 0.6,
                "setup_seconds_at_median_total": total * 0.4,
                "solving_phase_seconds_at_median_total": total * 0.6,
                "train_total_seconds_median": total,
                "test_rmse_median": 1.0 if accuracy else 1.5,
                "test_r2_median": 0.6,
                "iterations_median": (
                    4000.0
                    if method == "efgp-standard-cg" and n_train == 10_000_000
                    else 7000.0 if method == "efgp-standard-cg" else ""
                ),
            }
        )
    return output


def _all_stage1_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in (10_000_000, 30_000_000, 100_000_000, 300_000_000):
        rows.extend(
            _stage1_method_rows(
                profile="scale_10m_300m",
                case_id=f"synthetic_n{n // 1_000_000}m",
                axes=[],
                dataset_stem="synthetic_master",
                dataset_family="Synthetic",
                n_train=n,
            )
        )
    robust_cases = [
        (
            "reference",
            ["lambda_0p1", "lengthscale_0p1", "dataset_synthetic"],
            "synthetic_master",
            "Synthetic",
            0.1,
            0.1,
            8192,
            1.1,
        ),
        (
            "box_budget_8192",
            ["box_budget_8192"],
            "synthetic_master",
            "Synthetic",
            0.1,
            0.1,
            8192,
            1.1,
        ),
        (
            "lambda_0p01",
            ["lambda_0p01"],
            "synthetic_master",
            "Synthetic",
            0.01,
            0.1,
            8192,
            1.1,
        ),
        (
            "lambda_1p0",
            ["lambda_1p0"],
            "synthetic_master",
            "Synthetic",
            1.0,
            0.1,
            8192,
            1.1,
        ),
        (
            "lengthscale_0p05",
            ["lengthscale_0p05"],
            "synthetic_master",
            "Synthetic",
            0.1,
            0.05,
            8192,
            1.1,
        ),
        (
            "lengthscale_0p2",
            ["lengthscale_0p2"],
            "synthetic_master",
            "Synthetic",
            0.1,
            0.2,
            8192,
            1.1,
        ),
        (
            "box_budget_4096",
            ["box_budget_4096"],
            "synthetic_master",
            "Synthetic",
            0.1,
            0.1,
            4096,
            1.1,
        ),
        (
            "box_budget_16384",
            ["box_budget_16384"],
            "synthetic_master",
            "Synthetic",
            0.1,
            0.1,
            16384,
            1.1,
        ),
        (
            "dataset_winnebago",
            ["dataset_winnebago"],
            "winnebago_10m",
            "Winnebago",
            0.1,
            0.1,
            8192,
            2.0,
        ),
    ]
    for case in robust_cases:
        rows.extend(
            _stage1_method_rows(
                profile="robustness_at_selected_target",
                case_id=case[0],
                axes=case[1],
                dataset_stem=case[2],
                dataset_family=case[3],
                n_train=10_000_000,
                reg_lambda=case[4],
                lengthscale=case[5],
                box_budget=case[6],
                accuracy_max_rmse=case[7],
            )
        )
    return rows


STAGE2_METHODS = (
    "cg",
    "jacobi",
    "default",
    "active-inverse",
    "active-eig",
    "full-eig",
    "nystrom",
)


def _stage2_artifacts(root: Path) -> Path:
    run_dir = root / "fixed_ab"
    arrays = {
        "weights_flat": np.asarray([1.25, 0.75, 2.0], dtype=np.float64),
        "weights_np_flat": np.asarray([1.25, 0.75, 2.0], dtype=np.float64),
        "gf": np.asarray([1.0 + 0.5j, 0.25 - 0.75j, -0.5j], dtype=np.complex128),
        "rhs_storage": np.asarray(
            [0.5 + 0.25j, -1.0 + 0.5j, 0.75 - 0.25j], dtype=np.complex128
        ),
        "rhs_solve": np.asarray(
            [0.5 + 0.25j, -1.0 + 0.5j, 0.75 - 0.25j], dtype=np.complex128
        ),
    }
    numpy_context = SimpleNamespace(
        weights_gpu_flat=arrays["weights_flat"],
        weights_np_flat=arrays["weights_np_flat"],
        gf_gpu=arrays["gf"],
        rhs_gpu=arrays["rhs_storage"],
    )
    system_id = system_fingerprint(
        numpy_context,
        0.1,
        solve_rhs_gpu=arrays["rhs_solve"],
    )
    component_hashes = system_component_fingerprints(
        numpy_context,
        solve_rhs_gpu=arrays["rhs_solve"],
    )
    base_totals = {
        "cg": 10.0,
        "jacobi": 11.0,
        "default": 5.0,
        "active-inverse": 5.5,
        "active-eig": 5.2,
        "full-eig": 6.0,
        "nystrom": 2.0,
    }
    runs: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for method in STAGE2_METHODS:
        total_mid = base_totals[method]
        totals = [
            total_mid - 1.0,
            total_mid - 0.5,
            total_mid,
            total_mid + 1.0,
            total_mid + 3.0,
        ]
        active = method in {"default", "active-inverse", "active-eig"}
        selections = [0.1, 0.15, 0.2, 0.25, 0.3] if active else [0.0] * 5
        if method == "cg":
            builds = [0.0] * 5
        elif method == "nystrom":
            builds = [0.2, 0.15, 0.1, 0.15, 0.2]
        else:
            builds = [1.5, 0.4, 0.2, 0.8, 1.0]
        solves = [
            total - selection - build
            for total, selection, build in zip(totals, selections, builds)
        ]
        for repeat, (total, selection, build, solve) in enumerate(
            zip(totals, selections, builds, solves)
        ):
            runs.append(
                {
                    "method": method,
                    "system_id": system_id,
                    "repeat_idx": repeat,
                    "is_warmup": False,
                    "status": "converged",
                    "tol": 1e-7,
                    "maxiter": 6000,
                    "zero_initial_vector": True,
                    "true_relres": 1e-8,
                    "solver_total_definition": EXPECTED_SOLVER_TOTAL_DEFINITION,
                    "selection_seconds": selection,
                    "preconditioner_build_seconds": build,
                    "solve_seconds": solve,
                    "solver_total_seconds": total,
                }
            )
        runs.append(
            {
                **runs[-5],
                # Match benchmark.py: warmups occupy negative repeat IDs and
                # measured repetitions start at zero.
                "repeat_idx": -1,
                "is_warmup": True,
            }
        )
        summaries.append(
            {
                "method": method,
                "method_kind": (
                    "active-eig" if method == "default" else method
                ),
                "result_role": (
                    "deployable_default" if method == "default" else "baseline"
                ),
                "measured_repeats": 5,
                "performance_claim_eligible": True,
                "solver_total_definition": EXPECTED_SOLVER_TOTAL_DEFINITION,
                "selection_seconds_median": float(np.median(selections)),
                "preconditioner_build_seconds_median": float(np.median(builds)),
                "solve_seconds_median": float(np.median(solves)),
                "solver_total_seconds_median": total_mid,
                "build_plus_solve_seconds_median": total_mid,
            }
        )
    summary_path = _write_csv(run_dir / "matched_summary.csv", summaries)
    _write_csv(run_dir / "matched_runs.csv", runs)
    system_config = {
        "dataset_stem": "synthetic_master",
        "n_train": 10_000_000,
        "subset_seed": 0,
        "subset_mode": "prefix",
        "kernel_family": "se",
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
    }
    system_config_sha = hashlib.sha256(
        json.dumps(
            system_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    artifact_path = run_dir / "timing_system.npz"
    nested_system_manifest = {
        "system_id": system_id,
        **component_hashes,
        "system_config": system_config,
        "system_config_sha256": system_config_sha,
        "reg_lambda": 0.1,
    }
    artifact_manifest = json.dumps(
        {
            "schema_version": 1,
            "system_id": system_id,
            **component_hashes,
            "system_config": system_config,
            "system_config_sha256": system_config_sha,
            "reg_lambda": 0.1,
            "system_manifest": nested_system_manifest,
            "arrays": {
                name: _artifact_array_descriptor(name, value)
                for name, value in arrays.items()
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    np.savez(
        artifact_path,
        **arrays,
        artifact_manifest_json=np.frombuffer(artifact_manifest, dtype=np.uint8),
    )
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    _write_json(
        run_dir / "system_manifest.json",
        {
            "system_id": system_id,
            "final_system_id": system_id,
            "system_unchanged": True,
            **component_hashes,
            "system_artifact_sha256": artifact_sha,
            "system_config": system_config,
            "system_config_sha256": system_config_sha,
            "dataset_stem": "synthetic_master",
            "dataset_task_type": "Synthetic",
            "n_train": 10_000_000,
            "subset_seed": 0,
            "subset_mode": "prefix",
            "kernel_family": "se",
            "kernel_nu": 1.5,
            "kernel_variance": 1.0,
            "kernel_lengthscale": 0.1,
            "reg_lambda": 0.1,
            "fourier_eps": 1e-5,
            "nufft_tol": 1e-10,
            "l2_scaled": True,
            "precision_mode": "fp64",
            "nufft_backend_requested": "cufinufft",
            "precompute_chunk_size": 1_000_000,
            "setup_seconds": 12.5,
            "selection_timing_protocol": "selection rerun in every measured repeat",
            "score_selection_seconds": 0.2,
            "selection_seconds_median_by_method": {
                "default": 0.2,
                "active-inverse": 0.2,
                "active-eig": 0.2,
            },
            "score_protocol_freeze_selection_seconds": 0.15,
        },
    )
    _write_json(
        run_dir / "experiment_config.json",
        {
            "methods": list(STAGE2_METHODS),
            **FROZEN_METHOD_CONFIG,
            "box_budget": 8192,
            "inverse_max_size": 16384,
            "default_inverse_max_size": 1024,
            "warmup_repeats": 1,
            "measured_repeats": 5,
            "tol": 1e-7,
            "maxiter": 6000,
            "zero_initial_vector": True,
        },
    )
    _write_json(
        run_dir / "run_complete.json",
        {
            "system_id": system_id,
            "methods": list(STAGE2_METHODS),
            "warmup_repeats": 1,
            "measured_repeats": 5,
            "tol": 1e-7,
            "maxiter": 6000,
            "zero_initial_vector": True,
            "run_row_count": len(runs),
            "summary_row_count": len(summaries),
            "timing_system_artifact": artifact_path.name,
            "timing_system_artifact_sha256": artifact_sha,
        },
    )
    return summary_path


def _materialize_stage1_evidence(root: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["suite_profile"]), str(row["case_id"])), []).append(
            row
        )
    for (profile, case_id), case_rows in grouped.items():
        first = case_rows[0]
        run_dir = root / "stage1_runs" / profile / case_id
        cfg = EndToEndConfig(
            dataset_stem=str(first["dataset_stem"]),
            dataset_dir=str(root),
            n_train=int(first["n_train"]),
            subset_seed=int(first["subset_seed"]),
            subset_mode=str(first["subset_mode"]),
            kernel_family=str(first["kernel_family"]),
            lengthscale=float(first["lengthscale"]),
            nu=float(first["nu"]),
            variance=float(first["variance"]),
            reg_lambda=float(first["reg_lambda"]),
            fourier_eps=float(first["fourier_eps"]),
            nufft_tol=float(first["nufft_tol"]),
            l2_scaled=bool(first["l2_scaled"]),
            precision=str(first["precision"]),
            methods=END_TO_END_METHODS,
            rank=int(first["configured_active_rank"]),
            full_eig_rank=int(first["configured_full_eig_rank"]),
            active_topk=(
                None
                if first["configured_active_topk"] in (None, "")
                else int(first["configured_active_topk"])
            ),
            expected_active_box_size=(
                None
                if first["configured_expected_active_box_size"] in (None, "")
                else int(first["configured_expected_active_box_size"])
            ),
            box_budget=int(first["box_budget"]),
            parameter_selection_policy=str(first["parameter_selection_policy"]),
            parameter_source=str(first["parameter_source"]),
            inverse_max_size=16384,
            warmup_repeats=1,
            measured_repeats=5,
            accuracy_relative_tolerance=0.01,
            accuracy_max_rmse=float(first["accuracy_max_rmse"]),
            accuracy_min_r2=float(first["accuracy_min_r2"]),
            nufft_backend=str(first["nufft_backend"]),
            precompute_chunk_size=int(first["precompute_chunk_size"]),
            output_dir=str(run_dir),
        )
        for row in case_rows:
            row["run_dir"] = str(run_dir)
        run_rows: list[dict[str, object]] = []
        for is_warmup, repeats in ((True, 1), (False, 5)):
            for repeat_idx in range(repeats):
                for summary in case_rows:
                    total = float(summary["train_total_seconds_median"])
                    method = str(summary["method"])
                    record: dict[str, object] = {
                        "protocol_family": "end_to_end_krr",
                        "timing_scope": EXPECTED_STAGE1_TIMING_SCOPE,
                        "method": method,
                        "repeat_idx": repeat_idx,
                        "is_warmup": is_warmup,
                        "status": "converged",
                        "dataset_stem": cfg.dataset_stem,
                        "n_train": cfg.n_train,
                        "subset_seed": cfg.subset_seed,
                        "subset_mode": cfg.subset_mode,
                        "kernel_family": cfg.kernel_family,
                        "lengthscale": cfg.lengthscale,
                        "nu": cfg.nu,
                        "variance": cfg.variance,
                        "reg_lambda": cfg.reg_lambda,
                        "fourier_eps": cfg.fourier_eps,
                        "nufft_tol": cfg.nufft_tol,
                        "l2_scaled": cfg.l2_scaled,
                        "precision": cfg.precision,
                        "nufft_backend": cfg.nufft_backend,
                        "precompute_chunk_size": cfg.precompute_chunk_size,
                        "box_budget": cfg.box_budget,
                        "configured_active_rank": cfg.rank,
                        "configured_full_eig_rank": cfg.full_eig_rank,
                        "configured_active_topk": cfg.active_topk,
                        "configured_expected_active_box_size": (
                            cfg.expected_active_box_size
                        ),
                        "parameter_selection_policy": (
                            cfg.parameter_selection_policy
                        ),
                        "parameter_source": cfg.parameter_source,
                        "accuracy_max_rmse": cfg.accuracy_max_rmse,
                        "accuracy_min_r2": cfg.accuracy_min_r2,
                        "setup_seconds": total * 0.4,
                        "solving_phase_seconds": total * 0.6,
                        "train_total_seconds": total,
                        "test_rmse": 1.5 if method == "nystrom-krr" else 1.0,
                        "test_r2": 0.6,
                    }
                    if method == "efgp-standard-cg":
                        record["iterations"] = (
                            4000 if int(cfg.n_train or 0) == 10_000_000 else 7000
                        )
                    run_rows.append(record)
        recomputed = {
            str(summary["method"]): summary
            for summary in summarize_pipeline_rows(run_rows, cfg)
        }
        for row in case_rows:
            row.update(recomputed[str(row["method"])])
        _write_csv(run_dir / "pipeline_runs.csv", run_rows)
        _write_json(run_dir / "experiment_config.json", asdict(cfg))
        expected_count = len(END_TO_END_METHODS) * 6
        _write_json(
            run_dir / "run_complete.json",
            {
                "protocol_family": "end_to_end_krr",
                "methods": list(END_TO_END_METHODS),
                "expected_row_count": expected_count,
                "observed_row_count": expected_count,
                "all_rows_present": True,
                "artifact_complete": True,
            },
        )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    suite = _write_json(tmp_path / "suite.json", _suite_payload())
    target = _write_json(tmp_path / "selected_target_regime.json", _target_payload())
    stage1_rows = _all_stage1_rows()
    _materialize_stage1_evidence(tmp_path, stage1_rows)
    stage1 = _write_csv(tmp_path / "stage1" / "pipeline_summary.csv", stage1_rows)
    stage2 = _stage2_artifacts(tmp_path)
    return suite, target, stage1, stage2


def test_stage1_loader_preserves_protocol_and_timing_reporting_contract(
    tmp_path: Path,
) -> None:
    raw_rows = _all_stage1_rows()
    _materialize_stage1_evidence(tmp_path, raw_rows)
    summary_path = _write_csv(tmp_path / "stage1" / "pipeline_summary.csv", raw_rows)

    normalized = load_stage1_summaries((summary_path,))

    assert normalized
    assert {row["protocol_family"] for row in normalized} == {STAGE1_PROTOCOL}
    assert {row["timing_scope"] for row in normalized} == {
        EXPECTED_STAGE1_TIMING_SCOPE
    }
    # Exercise the same canonical projection used by the Stage 1 report tables.
    projected = [
        {column: row[column] for column in STAGE1_TABLE_COLUMNS}
        for row in normalized
    ]
    assert all(row["protocol_family"] == STAGE1_PROTOCOL for row in projected)
    assert all(
        row["timing_scope"] == EXPECTED_STAGE1_TIMING_SCOPE for row in projected
    )


def _sync_stage2_artifact_sha(stage2: Path, artifact_path: Path) -> None:
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_path = stage2.parent / "system_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["system_artifact_sha256"] = artifact_sha
    _write_json(manifest_path, manifest)
    completion_path = stage2.parent / "run_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["timing_system_artifact_sha256"] = artifact_sha
    _write_json(completion_path, completion)


def _build(
    tmp_path: Path,
    *,
    stage1: Path,
    stage2: Path,
    suite: Path,
    target: Path,
    feasibility: Path | None = None,
    plots: bool = False,
) -> dict[str, object]:
    if feasibility is None:
        target_payload = json.loads(target.read_text(encoding="utf-8"))
        suite_payload = json.loads(suite.read_text(encoding="utf-8"))
        box_budget = int(suite_payload["base"]["box_budget"])
        active_box_upper_bound = int(
            suite_payload["base"].get("expected_active_box_size", box_budget)
        )
        inverse_max_size = int(
            suite_payload.get("stage2_fixed_ab", {}).get(
                "inverse_max_size", suite_payload["base"]["inverse_max_size"]
            )
        )
        default_inverse_max_size = int(
            suite_payload.get("stage2_fixed_ab", {}).get(
                "default_inverse_max_size",
                suite_payload["base"]["inverse_max_size"],
            )
        )
        feasibility = _write_json(
            tmp_path / "stage2_feasibility.json",
            {
                "schema_version": 1,
                "protocol_family": "controlled_fixed_system",
                "decision_basis": (
                    "prospective declared active-box upper bound before timing"
                ),
                **{
                    field: target_payload[field]
                    for field in (
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
                },
                **{
                    field: target_payload[field]
                    for field in FROZEN_METHOD_CONFIG
                },
                "box_budget": box_budget,
                "active_box_upper_bound": active_box_upper_bound,
                "inverse_max_size": inverse_max_size,
                "default_inverse_max_size": default_inverse_max_size,
                "default_resolved_kind": (
                    "active-inverse"
                    if active_box_upper_bound <= default_inverse_max_size
                    else "active-eig"
                ),
                "methods": {
                    **{
                        method: {
                            "feasible": True,
                            "reason": "prospectively mandatory test method",
                        }
                        for method in (
                            "cg",
                            "jacobi",
                            "default",
                            "active-eig",
                            "full-eig",
                        )
                    },
                    "active-inverse": {
                        "feasible": active_box_upper_bound <= inverse_max_size,
                        "reason": (
                            "prospective cap permits inverse"
                            if active_box_upper_bound <= inverse_max_size
                            else "prospective active box bound exceeds inverse cap"
                        ),
                    },
                },
            },
        )
    return build_two_stage_report(
        TwoStageReportConfig(
            stage1_paths=(str(stage1),),
            stage2_paths=(str(stage2),),
            output_dir=str(tmp_path / "report"),
            selected_target_path=str(target),
            stage1_suite_path=str(suite),
            stage2_feasibility_path=str(feasibility) if feasibility else None,
            make_plots=plots,
        )
    )


def _claims(result: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["claim_id"]): row for row in result["claims"]}  # type: ignore[index]


def test_full_report_uses_declared_profiles_target_and_paired_totals(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    result = _build(
        tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target, plots=True
    )
    claims = _claims(result)
    assert claims["stage1_scale_10m_to_300m"]["status"] == "supported"
    assert claims["stage1_robustness_oat_design_complete"]["status"] == "supported"
    for axis in ("lambda", "lengthscale", "box_budget", "dataset"):
        assert claims[f"stage1_robust_across_{axis}"]["status"] == "supported"
    assert claims["stage2_same_A_b_verified"]["status"] == "supported"
    assert claims["stage2_formal_method_matrix_complete"]["status"] == "supported"
    assert claims["stage2_corrected_totals_verified"]["status"] == "supported"
    assert (
        claims["stage2_primary_ours_beats_best_baseline_total"]["status"] == "supported"
    )
    default = next(row for row in result["stage2"] if row["method"] == "default")  # type: ignore[index]
    assert default["solver_total_seconds"] == pytest.approx(5.0)
    assert default["preconditioner_build_seconds"] == pytest.approx(0.2)
    assert default["selection_seconds"] + default[
        "preconditioner_build_seconds"
    ] + default["solve_seconds"] == pytest.approx(default["solver_total_seconds"])
    assert default["solver_total_speedup_over_cg_median"] == pytest.approx(2.0)
    assert default["paired_comparisons"] == 5
    fourier = next(row for row in result["stage2"] if row["method"] == "nystrom")  # type: ignore[index]
    assert fourier["formal_included"] is False
    output = Path(result["output_dir"])
    for png in output.glob("*.png"):
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    scale_rows = list(
        csv.DictReader(
            (output / "stage1_scale_10m_300m.csv").open(newline="", encoding="utf-8")
        )
    )
    assert len(scale_rows) == 4 * len(STAGE1_FORMAL_METHODS)
    assert {row["suite_profile"] for row in scale_rows} == {"scale_10m_300m"}
    nystrom = next(row for row in scale_rows if row["method"] == "nystrom-krr")
    assert float(nystrom["speedup_vs_ours"]) == pytest.approx(8.0 / 5.0)
    assert nystrom["usability_eligible"] == "False"
    assert nystrom["reference_equivalent"] == "False"
    assert nystrom["speedup_claim_eligible"] == "False"


def test_stage2_speedup_uses_median_of_matched_repeat_ratios(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    runs_path = stage2.parent / "matched_runs.csv"
    run_rows = list(csv.DictReader(runs_path.open(newline="", encoding="utf-8")))
    replacement_totals = {
        "cg": [2.0, 2.0, 100.0, 100.0, 100.0],
        "default": [4.0, 4.0, 5.0, 200.0, 200.0],
    }
    for row in run_rows:
        if str(row["is_warmup"]).lower() == "true":
            continue
        method = row["method"]
        if method not in replacement_totals:
            continue
        total = replacement_totals[method][int(row["repeat_idx"])]
        row["solver_total_seconds"] = total
        row["solve_seconds"] = (
            total
            - float(row["selection_seconds"])
            - float(row["preconditioner_build_seconds"])
        )
    _write_csv(runs_path, run_rows)

    summary_rows = list(csv.DictReader(stage2.open(newline="", encoding="utf-8")))
    for row in summary_rows:
        method = row["method"]
        if method not in replacement_totals:
            continue
        method_runs = [
            run
            for run in run_rows
            if run["method"] == method and str(run["is_warmup"]).lower() != "true"
        ]
        row["solver_total_seconds_median"] = float(
            np.median(replacement_totals[method])
        )
        row["build_plus_solve_seconds_median"] = row["solver_total_seconds_median"]
        row["selection_seconds_median"] = float(
            np.median([float(run["selection_seconds"]) for run in method_runs])
        )
        row["preconditioner_build_seconds_median"] = float(
            np.median(
                [float(run["preconditioner_build_seconds"]) for run in method_runs]
            )
        )
        row["solve_seconds_median"] = float(
            np.median([float(run["solve_seconds"]) for run in method_runs])
        )
    _write_csv(stage2, summary_rows)

    result = _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)
    default = next(
        row for row in result["stage2"] if row["method"] == "default"  # type: ignore[index]
    )
    assert 100.0 / 5.0 == pytest.approx(20.0)  # ratio of medians: wrong protocol
    assert default["solver_total_speedup_over_cg_median"] == pytest.approx(0.5)
    assert default["solver_total_speedup_over_cg_min"] == pytest.approx(0.5)
    assert default["solver_total_speedup_over_cg_max"] == pytest.approx(20.0)
    assert default["paired_comparisons"] == 5
    assert default["paired_wins_over_cg"] == 1
    assert (
        _claims(result)["stage2_primary_ours_beats_best_baseline_total"]["status"]
        == "not_supported"
    )
    formal_rows = list(
        csv.DictReader(
            (Path(result["output_dir"]) / "stage2_formal_solver_totals.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    formal_default = next(row for row in formal_rows if row["method"] == "default")
    assert float(
        formal_default["solver_total_speedup_over_cg_median"]
    ) == pytest.approx(0.5)


def test_stage2_requires_negative_production_warmup_repeat_ids(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    runs_path = stage2.parent / "matched_runs.csv"
    rows = list(csv.DictReader(runs_path.open(newline="", encoding="utf-8")))
    for row in rows:
        if str(row["is_warmup"]).lower() == "true":
            row["repeat_idx"] = 0
    _write_csv(runs_path, rows)
    with pytest.raises(ReportSchemaError, match="warmup/measured repeat coverage"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


def test_stage1_rejects_blank_profile_metadata(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    rows = list(csv.DictReader(stage1.open(newline="", encoding="utf-8")))
    rows[0]["suite_profile"] = ""
    bad = _write_csv(tmp_path / "bad_stage1.csv", rows)
    with pytest.raises(ReportSchemaError, match="suite_profile"):
        _build(tmp_path, stage1=bad, stage2=stage2, suite=suite, target=target)


def test_stage1_scale_requires_every_predeclared_case_and_method(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    rows = list(csv.DictReader(stage1.open(newline="", encoding="utf-8")))
    missing_case = [row for row in rows if row["case_id"] != "synthetic_n300m"]
    with pytest.raises(ReportSchemaError, match="exactly match"):
        _build(
            tmp_path,
            stage1=_write_csv(tmp_path / "missing_case.csv", missing_case),
            stage2=stage2,
            suite=suite,
            target=target,
        )

    missing_method = [
        row
        for row in rows
        if not (
            row["case_id"] == "synthetic_n300m"
            and row["method"] == "efgp-standard-jacobi"
        )
    ]
    with pytest.raises(ReportSchemaError, match="exactly the six"):
        _build(
            tmp_path,
            stage1=_write_csv(tmp_path / "missing_method.csv", missing_method),
            stage2=stage2,
            suite=suite,
            target=target,
        )


def test_scale_resource_limit_row_is_preserved_and_plot_remains_auditable(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    rows = list(csv.DictReader(stage1.open(newline="", encoding="utf-8")))
    limited = next(
        row
        for row in rows
        if row["case_id"] == "synthetic_n300m" and row["method"] == "rpcholesky-krr"
    )
    limited.update(
        {
            "status": "resource_limit",
            "execution_eligible": "False",
            "usability_eligible": "False",
            "usability_evaluated_repeats": "0",
            "usability_passed_repeats": "0",
            "reference_equivalent": "False",
            "reference_evaluated_repeats": "0",
            "reference_equivalent_repeats": "0",
            "quality_qualified_performance_eligible": "False",
            "ours_speedup_complete_pairing": "False",
            "ours_speedup_claim_eligible": "False",
            "ours_total_speedup": "",
            "ours_setup_speedup": "",
            "ours_solving_speedup": "",
            "comparison_rmse_ratio_to_ours": "",
            "comparison_rmse_delta_from_ours": "",
            "accuracy_eligible": "False",
            "performance_claim_eligible": "False",
            "accuracy_evaluated_repeats": "0",
            "accuracy_passed_repeats": "0",
            "successful_repeats": "0",
            "setup_seconds_median": "",
            "solving_phase_seconds_median": "",
            "setup_seconds_at_median_total": "",
            "solving_phase_seconds_at_median_total": "",
            "train_total_seconds_median": "",
            "test_rmse_median": "",
            "test_r2_median": "",
            "iterations_median": "",
            "resource_required_bytes": str(80 * 2**30),
            "resource_effective_cap_bytes": str(48 * 2**30),
            "resource_declared_cap_bytes": str(48 * 2**30),
            "resource_available_device_bytes": str(40 * 2**30),
        }
    )
    runs_path = Path(limited["run_dir"]) / "pipeline_runs.csv"
    run_rows = list(csv.DictReader(runs_path.open(newline="", encoding="utf-8")))
    for run in run_rows:
        if run["method"] == "rpcholesky-krr":
            run.update(
                {
                    "status": "resource_limit",
                    "setup_seconds": "",
                    "solving_phase_seconds": "",
                    "train_total_seconds": "",
                    "test_rmse": "",
                    "test_r2": "",
                    "resource_required_bytes": str(80 * 2**30),
                    "resource_effective_cap_bytes": str(48 * 2**30),
                    "resource_declared_cap_bytes": str(48 * 2**30),
                    "resource_available_device_bytes": str(40 * 2**30),
                }
            )
    _write_csv(runs_path, run_rows)
    result = _build(
        tmp_path,
        stage1=_write_csv(tmp_path / "resource.csv", rows),
        stage2=stage2,
        suite=suite,
        target=target,
        plots=True,
    )
    written = list(
        csv.DictReader(
            (Path(result["output_dir"]) / "stage1_scale_10m_300m.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    row = next(
        item
        for item in written
        if item["declared_case_id"] == "synthetic_n300m"
        and item["method"] == "rpcholesky-krr"
    )
    assert row["status"] == "resource_limit"
    assert int(row["resource_required_bytes"]) == 80 * 2**30
    assert (Path(result["output_dir"]) / "stage1_scale_10m_300m.png").is_file()


def test_missing_predeclared_oat_value_is_not_claimed_robust(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    rows = [
        row
        for row in csv.DictReader(stage1.open(newline="", encoding="utf-8"))
        if row["case_id"] != "lambda_1p0"
    ]
    result = _build(
        tmp_path,
        stage1=_write_csv(tmp_path / "reduced.csv", rows),
        stage2=stage2,
        suite=suite,
        target=target,
    )
    claims = _claims(result)
    assert claims["stage1_robustness_oat_design_complete"]["status"] == "not_supported"
    assert claims["stage1_robust_across_lambda"]["status"] == "not_evaluable"


def test_oat_non_axis_change_fails_design_audit(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    rows = list(csv.DictReader(stage1.open(newline="", encoding="utf-8")))
    for row in rows:
        if row["case_id"] == "lambda_1p0":
            row["lengthscale"] = "0.2"
    with pytest.raises(ReportSchemaError, match="summary lengthscale disagrees"):
        _build(
            tmp_path,
            stage1=_write_csv(tmp_path / "changed.csv", rows),
            stage2=stage2,
            suite=suite,
            target=target,
        )


def test_incomplete_accuracy_repeat_evidence_removes_speed_claim(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    rows = list(csv.DictReader(stage1.open(newline="", encoding="utf-8")))
    for row in rows:
        if (
            row["suite_profile"] == "scale_10m_300m"
            and row["method"] == "ours-binned-default"
        ):
            row["accuracy_evaluated_repeats"] = "2"
            row["accuracy_passed_repeats"] = "2"
    with pytest.raises(
        ReportSchemaError, match="summary disagrees with repeat evidence"
    ):
        _build(
            tmp_path,
            stage1=_write_csv(tmp_path / "accuracy_changed.csv", rows),
            stage2=stage2,
            suite=suite,
            target=target,
        )


def test_stage2_target_mismatch_is_rejected(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    manifest_path = stage2.parent / "system_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reg_lambda"] = 0.2
    _write_json(manifest_path, manifest)
    with pytest.raises(ReportSchemaError, match="regularization differ"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


def test_unknown_stage2_method_spelling_is_rejected(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    rows = list(csv.DictReader(stage2.open(newline="", encoding="utf-8")))
    rows[0]["method"] = "JABOBI"
    _write_csv(stage2, rows)
    with pytest.raises(ReportSchemaError, match="unknown Stage 2 method"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


def test_explicit_feasibility_matrix_can_document_missing_inverse(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    suite_payload = json.loads(suite.read_text(encoding="utf-8"))
    suite_payload["base"]["inverse_max_size"] = 1024
    suite_payload["stage2_fixed_ab"]["inverse_max_size"] = 1024
    _write_json(suite, suite_payload)
    summary_rows = [
        row
        for row in csv.DictReader(stage2.open(newline="", encoding="utf-8"))
        if row["method"] != "active-inverse"
    ]
    run_path = stage2.parent / "matched_runs.csv"
    run_rows = [
        row
        for row in csv.DictReader(run_path.open(newline="", encoding="utf-8"))
        if row["method"] != "active-inverse"
    ]
    _write_csv(stage2, summary_rows)
    _write_csv(run_path, run_rows)
    executed_methods = [
        method for method in STAGE2_METHODS if method != "active-inverse"
    ]
    config_path = stage2.parent / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["methods"] = executed_methods
    config["inverse_max_size"] = 1024
    _write_json(config_path, config)
    manifest_path = stage2.parent / "system_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection_seconds_median_by_method"].pop("active-inverse")
    _write_json(manifest_path, manifest)
    completion_path = stage2.parent / "run_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion.update(
        {
            "methods": executed_methods,
            "run_row_count": len(run_rows),
            "summary_row_count": len(summary_rows),
        }
    )
    _write_json(completion_path, completion)
    result = _build(
        tmp_path,
        stage1=stage1,
        stage2=stage2,
        suite=suite,
        target=target,
    )
    claim = _claims(result)["stage2_formal_method_matrix_complete"]
    assert claim["status"] == "supported"
    assert claim["details"]["feasibility"]["active-inverse"]["feasible"] is False


def test_stage2_runtime_inverse_cap_must_match_prospective_feasibility(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    config_path = stage2.parent / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["inverse_max_size"] = 1024
    _write_json(config_path, config)

    with pytest.raises(ReportSchemaError, match="inverse_max_size"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


def test_stage2_runtime_default_cap_must_preserve_frozen_default_route(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    config_path = stage2.parent / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["default_inverse_max_size"] = 16384
    _write_json(config_path, config)

    with pytest.raises(ReportSchemaError, match="default_inverse_max_size"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


def test_stage2_default_method_kind_must_match_declared_inverse_route(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    rows = list(csv.DictReader(stage2.open(newline="", encoding="utf-8")))
    for row in rows:
        if row["method"] == "default":
            row["method_kind"] = "active-inverse"
    _write_csv(stage2, rows)

    with pytest.raises(ReportSchemaError, match="declared Stage-2 routing"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


def test_stage2_measured_run_component_mismatch_is_rejected(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    run_path = stage2.parent / "matched_runs.csv"
    rows = list(csv.DictReader(run_path.open(newline="", encoding="utf-8")))
    rows[0]["solve_seconds"] = str(float(rows[0]["solve_seconds"]) + 1.0)
    _write_csv(run_path, rows)
    with pytest.raises(ReportSchemaError, match="component sum"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


def test_stage2_summary_total_must_match_recomputed_measured_median(
    tmp_path: Path,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    rows = list(csv.DictReader(stage2.open(newline="", encoding="utf-8")))
    next(row for row in rows if row["method"] == "default")[
        "solver_total_seconds_median"
    ] = "99"
    _write_csv(stage2, rows)
    with pytest.raises(ReportSchemaError, match="measured-run median"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


def test_stage2_requires_measured_selection_protocol(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    manifest_path = stage2.parent / "system_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_timing_protocol"] = ""
    _write_json(manifest_path, manifest)
    with pytest.raises(ReportSchemaError, match="selection timing protocol"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


def test_stage2_rejects_changed_system_or_repeat_residual(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    manifest_path = stage2.parent / "system_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_system_id"] = "changed-system"
    _write_json(manifest_path, manifest)
    with pytest.raises(ReportSchemaError, match="identical non-empty"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)

    suite, target, stage1, stage2 = _fixture(tmp_path / "residual")
    runs_path = stage2.parent / "matched_runs.csv"
    runs = list(csv.DictReader(runs_path.open(newline="", encoding="utf-8")))
    run = next(
        row
        for row in runs
        if row["method"] == "default" and row["is_warmup"].lower() == "false"
    )
    run["status"] = "maxiter"
    run["true_relres"] = "1e-3"
    _write_csv(runs_path, runs)
    with pytest.raises(ReportSchemaError, match="repeat-level status"):
        _build(
            tmp_path / "residual",
            stage1=stage1,
            stage2=stage2,
            suite=suite,
            target=target,
        )


def test_stage2_verifies_artifact_hash_and_per_method_selection(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    manifest_path = stage2.parent / "system_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The pooled value is descriptive only; formal verification uses each
    # method's own median and therefore must not impose a cross-method 2% gate.
    manifest["score_selection_seconds"] = 999.0
    _write_json(manifest_path, manifest)
    _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)

    manifest["selection_seconds_median_by_method"]["default"] = 9.0
    _write_json(manifest_path, manifest)
    with pytest.raises(ReportSchemaError, match="own measured per-repeat"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)

    suite, target, stage1, stage2 = _fixture(tmp_path / "artifact")
    artifact_path = stage2.parent / "timing_system.npz"
    with artifact_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ReportSchemaError, match="SHA-256"):
        _build(
            tmp_path / "artifact",
            stage1=stage1,
            stage2=stage2,
            suite=suite,
            target=target,
        )


def test_stage2_rejects_metadata_only_or_array_checksum_tamper(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path / "metadata_only")
    artifact_path = stage2.parent / "timing_system.npz"
    with np.load(artifact_path, allow_pickle=False) as artifact:
        manifest_bytes = np.asarray(
            artifact["artifact_manifest_json"], dtype=np.uint8
        ).copy()
    np.savez(artifact_path, artifact_manifest_json=manifest_bytes)
    _sync_stage2_artifact_sha(stage2, artifact_path)
    with pytest.raises(ReportSchemaError, match="invalid timing-system array artifact"):
        _build(
            tmp_path / "metadata_only",
            stage1=stage1,
            stage2=stage2,
            suite=suite,
            target=target,
        )

    suite, target, stage1, stage2 = _fixture(tmp_path / "array_tamper")
    artifact_path = stage2.parent / "timing_system.npz"
    with np.load(artifact_path, allow_pickle=False) as artifact:
        payload = {name: np.asarray(artifact[name]).copy() for name in artifact.files}
    payload["gf"][0] += 1.0
    np.savez(artifact_path, **payload)
    _sync_stage2_artifact_sha(stage2, artifact_path)
    with pytest.raises(ReportSchemaError, match="checksum changed"):
        _build(
            tmp_path / "array_tamper",
            stage1=stage1,
            stage2=stage2,
            suite=suite,
            target=target,
        )


@pytest.mark.parametrize(
    "manifest_layer",
    (
        "embedded artifact manifest",
        "nested system manifest",
        "external system manifest",
    ),
)
def test_stage2_rejects_component_hash_drift_in_each_manifest_layer(
    tmp_path: Path,
    manifest_layer: str,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    artifact_path = stage2.parent / "timing_system.npz"
    if manifest_layer == "external system manifest":
        manifest_path = stage2.parent / "system_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["gf_sha256"] = "0" * 64
        _write_json(manifest_path, manifest)
    else:
        with np.load(artifact_path, allow_pickle=False) as artifact:
            payload = {
                name: np.asarray(artifact[name]).copy() for name in artifact.files
            }
        metadata = json.loads(
            np.asarray(payload["artifact_manifest_json"], dtype=np.uint8)
            .tobytes()
            .decode("utf-8")
        )
        target_manifest = (
            metadata
            if manifest_layer == "embedded artifact manifest"
            else metadata["system_manifest"]
        )
        target_manifest["gf_sha256"] = "0" * 64
        payload["artifact_manifest_json"] = np.frombuffer(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            dtype=np.uint8,
        )
        np.savez(artifact_path, **payload)
        _sync_stage2_artifact_sha(stage2, artifact_path)
    with pytest.raises(ReportSchemaError, match=manifest_layer):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (
        ("tol", "1e-3"),
        ("maxiter", "12000"),
        ("zero_initial_vector", "False"),
    ),
)
def test_stage2_rejects_per_run_solver_protocol_drift(
    tmp_path: Path,
    field: str,
    changed_value: str,
) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    runs_path = stage2.parent / "matched_runs.csv"
    runs = list(csv.DictReader(runs_path.open(newline="", encoding="utf-8")))
    runs[0][field] = changed_value
    _write_csv(runs_path, runs)
    with pytest.raises(ReportSchemaError, match="common tol/maxiter"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)


def test_reporter_recomputes_the_frozen_target_winner(tmp_path: Path) -> None:
    suite, target, stage1, stage2 = _fixture(tmp_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["n_train"] = 30_000_000
    _write_json(target, payload)
    with pytest.raises(ReportSchemaError, match="not the winner recomputed"):
        _build(tmp_path, stage1=stage1, stage2=stage2, suite=suite, target=target)
