"""Out-of-timing prediction audit for a controlled fixed-system experiment.

The timed benchmark persists the exact Fourier-system arrays and one canonical
measured solution per method.  This module loads those artifacts verbatim and
only performs chunked prediction.  It never rebuilds ``A, b``, never re-solves
the system, and never includes prediction time in a speedup claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ...benchmark_dataset.stored_npz import (
    StoredNpzError,
    inspect_stored_npy_member,
    load_stored_npz_prefix,
)
from ...v1_ops import predict_v1
from .benchmark import (
    ControlledConfig,
    PreparedSystem,
    TIMING_SOLUTIONS_ARTIFACT_FILENAME,
    TIMING_SOLUTIONS_MANIFEST_FILENAME,
    TIMING_SYSTEM_ARTIFACT_FILENAME,
    _npz_content_index_sha256,
    load_prepared_system_artifact,
    system_component_fingerprints,
    system_fingerprint,
)


_HERE = Path(__file__).resolve().parent
_TIMING_SYSTEM_ARTIFACT = TIMING_SYSTEM_ARTIFACT_FILENAME
_TIMING_SOLUTIONS_ARTIFACT = TIMING_SOLUTIONS_ARTIFACT_FILENAME
_TIMING_SOLUTIONS_MANIFEST = TIMING_SOLUTIONS_MANIFEST_FILENAME
PREDICTION_AUDIT_JSON_FILENAME = "prediction_audit.json"
PREDICTION_AUDIT_CSV_FILENAME = "prediction_audit.csv"
PREDICTION_AUDIT_COMPLETION_FILENAME = "prediction_audit_complete.json"
_ROW_FIELDS = (
    "system_id",
    "dataset",
    "method",
    "method_kind",
    "solve_status",
    "true_relres",
    "test_rmse",
    "test_rmse_ratio_vs_cg",
    "test_rmse_diff_vs_cg",
    "test_rmse_relative_diff_vs_cg",
    "prediction_equivalent_to_cg",
    "prediction_seconds",
    "prediction_nufft_stages",
    "prediction_nufft_strict",
    "iterations",
    "timing_repeat_idx",
    "timing_order_position",
    "timing_solution_sha256",
    "timing_solution_build_seconds",
    "timing_solution_solve_seconds",
    "timing_system_reused",
    "timing_solution_reused",
    "n_test",
    "prediction_chunk_size",
    "audit_solve_build_seconds",
    "audit_solve_seconds",
    "audit_only_not_for_speed_claim",
)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_source_manifest() -> dict[str, Any]:
    """Hash the code path that loads timed betas and produces predictions."""
    repo_root = _HERE.parents[3]
    paths = [
        Path(__file__).resolve(),
        (_HERE / "benchmark.py").resolve(),
        (_HERE.parents[1] / "v1_ops.py").resolve(),
        (_HERE.parents[1] / "nufft_adapter.py").resolve(),
        (_HERE.parents[1] / "backends.py").resolve(),
        (_HERE.parents[1] / "contexts.py").resolve(),
        (_HERE.parents[1] / "benchmark_dataset" / "stored_npz.py").resolve(),
    ]
    combined = hashlib.sha256()
    hashes: dict[str, str] = {}
    for path in paths:
        relative = str(path.relative_to(repo_root)).replace("\\", "/")
        digest = _file_sha256(path)
        hashes[relative] = digest
        combined.update(relative.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return {
        "prediction_source_bundle_sha256": combined.hexdigest(),
        "prediction_source_files_sha256": hashes,
    }


def verify_test_dataset_provenance(
    dataset_path: str | Path,
    timing_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the current test NPZ/metadata match the timed data."""
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"timing dataset is no longer available: {path}")
    expected_content = timing_manifest.get("dataset_content_index_sha256")
    if not expected_content:
        raise ValueError("timing manifest lacks dataset_content_index_sha256")
    actual_content = _npz_content_index_sha256(path)
    if actual_content != expected_content:
        raise ValueError(
            "current test NPZ content index differs from the exact timing dataset"
        )
    expected_size = timing_manifest.get("dataset_file_size_bytes")
    actual_size = int(path.stat().st_size)
    if expected_size is not None and actual_size != int(expected_size):
        raise ValueError("current test NPZ byte size differs from the timing dataset")

    metadata_path = path.with_suffix(".json")
    actual_metadata_sha256 = (
        _file_sha256(metadata_path) if metadata_path.is_file() else None
    )
    if actual_metadata_sha256 != timing_manifest.get("dataset_metadata_sha256"):
        raise ValueError("current test metadata differs from the timing dataset")
    return {
        "test_dataset_path": str(path),
        "test_dataset_file_size_bytes": actual_size,
        "test_dataset_content_index_sha256": actual_content,
        "test_dataset_metadata_path": (
            str(metadata_path) if metadata_path.is_file() else None
        ),
        "test_dataset_metadata_sha256": actual_metadata_sha256,
        "test_dataset_content_index_verified": True,
        "test_dataset_metadata_verified": True,
    }


def _artifact_array_sha256(name: str, value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    hasher = hashlib.sha256()
    hasher.update(str(name).encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.view(np.uint8))
    return hasher.hexdigest()


def _load_timing_solutions(
    timing_run_dir: Path,
    timing_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_name = str(
        timing_manifest.get("timing_solution_manifest")
        or _TIMING_SOLUTIONS_MANIFEST
    )
    artifact_name = str(
        timing_manifest.get("timing_solution_artifact")
        or _TIMING_SOLUTIONS_ARTIFACT
    )
    solution_manifest_path = timing_run_dir / manifest_name
    solution_artifact_path = timing_run_dir / artifact_name
    if not solution_manifest_path.is_file() or not solution_artifact_path.is_file():
        raise FileNotFoundError(
            "timing prediction artifacts are missing; rerun the controlled timing case "
            "with the current benchmark before running its prediction audit"
        )
    expected_manifest_sha = timing_manifest.get("timing_solution_manifest_sha256")
    if not expected_manifest_sha:
        raise ValueError("system_manifest.json lacks timing solution manifest checksum")
    if _file_sha256(solution_manifest_path) != expected_manifest_sha:
        raise ValueError("timing solution manifest checksum does not match system_manifest.json")
    payload = json.loads(solution_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("timing solution manifest must be a JSON object")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported timing solution schema {payload.get('schema_version')!r}")
    expected_artifact_sha = timing_manifest.get("timing_solution_artifact_sha256")
    payload_artifact_sha = payload.get("timing_solution_artifact_sha256")
    if not expected_artifact_sha or payload_artifact_sha != expected_artifact_sha:
        raise ValueError("timing solution artifact checksum anchors are missing or disagree")
    actual_artifact_sha = _file_sha256(solution_artifact_path)
    if expected_artifact_sha != actual_artifact_sha:
        raise ValueError("timing solution artifact checksum mismatch")
    for field in (
        "system_id",
        "weights_sha256",
        "gf_sha256",
        "rhs_sha256",
        "rhs_storage_sha256",
        "system_config_sha256",
        "source_bundle_sha256",
        "dataset_content_index_sha256",
        "dataset_metadata_sha256",
    ):
        if payload.get(field) != timing_manifest.get(field):
            raise ValueError(f"timing solution manifest {field} differs from timing system")
    expected_system_artifact_sha = timing_manifest.get("system_artifact_sha256")
    if not expected_system_artifact_sha or payload.get(
        "timing_system_artifact_sha256"
    ) != expected_system_artifact_sha:
        raise ValueError("timing solution manifest is not anchored to the exact system artifact")

    loaded_solutions: dict[str, dict[str, Any]] = {}
    with np.load(solution_artifact_path, allow_pickle=False) as loaded:
        raw_solutions = payload.get("solutions")
        if not isinstance(raw_solutions, list):
            raise ValueError("timing solution manifest solutions must be a list")
        for raw_record in raw_solutions:
            if not isinstance(raw_record, dict):
                raise ValueError("timing solution records must be JSON objects")
            record = dict(raw_record)
            method = str(record.get("method", ""))
            if not method or method in loaded_solutions:
                raise ValueError("timing solution manifest has an empty or duplicate method")
            if not bool(record.get("available")):
                loaded_solutions[method] = record
                continue
            array_key = str(record.get("array_key", ""))
            descriptor = record.get("beta", {})
            if array_key not in loaded.files or not isinstance(descriptor, dict):
                raise ValueError(f"timing beta for method {method!r} is missing")
            beta = np.ascontiguousarray(loaded[array_key])
            if str(beta.dtype) != str(descriptor.get("dtype")):
                raise ValueError(f"timing beta dtype changed for method {method!r}")
            if [int(size) for size in beta.shape] != list(descriptor.get("shape", [])):
                raise ValueError(f"timing beta shape changed for method {method!r}")
            if beta.ndim != 1 or int(beta.size) != int(timing_manifest.get("M", -1)):
                raise ValueError(f"timing beta length differs from M for method {method!r}")
            if _artifact_array_sha256(array_key, beta) != descriptor.get("sha256"):
                raise ValueError(f"timing beta checksum changed for method {method!r}")
            timing_row = record.get("timing_row")
            if not isinstance(timing_row, dict):
                raise ValueError(f"timing row is missing for method {method!r}")
            if timing_row.get("method") != method or timing_row.get(
                "system_id"
            ) != timing_manifest.get("system_id"):
                raise ValueError(f"timing row provenance differs for method {method!r}")
            record["beta_host"] = beta
            loaded_solutions[method] = record
    payload["timing_solution_manifest_path"] = str(solution_manifest_path)
    payload["timing_solution_manifest_sha256"] = _file_sha256(solution_manifest_path)
    payload["timing_solution_artifact_path"] = str(solution_artifact_path)
    return payload, loaded_solutions


def load_timing_prediction_inputs(
    cfg: ControlledConfig,
    timing_run_dir: str | Path,
) -> tuple[PreparedSystem, dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and verify the exact timed system and canonical measured betas."""
    run_dir = Path(timing_run_dir).expanduser().resolve()
    timing_manifest_path = run_dir / "system_manifest.json"
    if not timing_manifest_path.is_file():
        raise FileNotFoundError(f"timing system manifest is missing: {timing_manifest_path}")
    timing_manifest = json.loads(timing_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(timing_manifest, dict):
        raise ValueError("timing system manifest must be a JSON object")
    system_artifact_name = str(
        timing_manifest.get("timing_system_artifact")
        or timing_manifest.get("system_artifact_filename")
        or _TIMING_SYSTEM_ARTIFACT
    )
    system_artifact_path = run_dir / system_artifact_name
    if not system_artifact_path.is_file():
        raise FileNotFoundError(
            "exact timing-system artifact is missing; rerun the controlled timing case "
            "with the current benchmark"
        )
    expected_system_sha = timing_manifest.get("system_artifact_sha256")
    if not expected_system_sha:
        raise ValueError("system_manifest.json lacks exact system artifact checksum")
    if _file_sha256(system_artifact_path) != expected_system_sha:
        raise ValueError("timing system artifact checksum does not match system_manifest.json")
    completion_path = run_dir / "run_complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError(
            "timing run has no run_complete.json; prediction audit requires a "
            "fully committed controlled timing case"
        )
    try:
        timing_completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"timing run completion is unreadable: {exc}") from exc
    if not isinstance(timing_completion, dict):
        raise ValueError("timing run completion must be a JSON object")
    system = load_prepared_system_artifact(
        cfg,
        system_artifact_path,
        expected_source_sha256=timing_manifest.get("source_bundle_sha256"),
        expected_dataset_content_index_sha256=timing_manifest.get(
            "dataset_content_index_sha256"
        ),
        expected_dataset_metadata_sha256=timing_manifest.get(
            "dataset_metadata_sha256"
        ),
    )
    actual_components = system_component_fingerprints(
        system.data_ctx,
        solve_rhs_gpu=system.rhs_gpu,
    )
    if system.system_id != timing_manifest.get("system_id"):
        raise ValueError("loaded timing artifact system_id differs from system_manifest.json")
    for field, actual in actual_components.items():
        if actual != timing_manifest.get(field):
            raise ValueError(f"loaded timing artifact {field} differs from system_manifest.json")
    if getattr(system.data_ctx, "x_center_gpu", None) is None:
        raise ValueError("timing system artifact lacks x_center required for prediction")

    solution_payload, solutions = _load_timing_solutions(run_dir, timing_manifest)
    if solution_payload.get("system_id") != system.system_id:
        raise ValueError("timing solution artifact belongs to a different system_id")
    for field, actual in actual_components.items():
        if solution_payload.get(field) != actual:
            raise ValueError(f"timing solution artifact {field} differs from exact system")
    missing_methods = [str(method) for method in cfg.methods if str(method) not in solutions]
    if missing_methods:
        raise ValueError(f"timing solutions are missing requested methods: {missing_methods}")
    completion_methods = timing_completion.get("methods")
    if not isinstance(completion_methods, list) or not set(cfg.methods).issubset(
        {str(method) for method in completion_methods}
    ):
        raise ValueError("timing run completion does not cover the requested methods")
    try:
        completion_schema_version = int(
            timing_completion.get("schema_version", -1)
        )
        completion_solution_count = int(
            timing_completion.get("timing_solution_count", -1)
        )
        payload_solution_count = int(solution_payload.get("solution_count", -1))
    except (TypeError, ValueError, OverflowError):
        completion_schema_version = -1
        completion_solution_count = -1
        payload_solution_count = -2
    completion_checks = (
        completion_schema_version == 1,
        timing_completion.get("system_id") == timing_manifest.get("system_id"),
        timing_completion.get("source_bundle_sha256")
        == timing_manifest.get("source_bundle_sha256"),
        timing_completion.get("dataset_content_index_sha256")
        == timing_manifest.get("dataset_content_index_sha256"),
        timing_completion.get("dataset_metadata_sha256")
        == timing_manifest.get("dataset_metadata_sha256"),
        timing_completion.get("timing_system_artifact") == system_artifact_name,
        timing_completion.get("timing_system_artifact_sha256")
        == expected_system_sha,
        timing_completion.get("timing_solution_artifact")
        == str(timing_manifest.get("timing_solution_artifact")),
        timing_completion.get("timing_solution_artifact_sha256")
        == timing_manifest.get("timing_solution_artifact_sha256"),
        timing_completion.get("timing_solution_manifest")
        == str(timing_manifest.get("timing_solution_manifest")),
        timing_completion.get("timing_solution_manifest_sha256")
        == timing_manifest.get("timing_solution_manifest_sha256"),
        completion_solution_count == payload_solution_count,
    )
    if not all(completion_checks):
        raise ValueError(
            "timing run completion does not exactly anchor the system/solution artifacts"
        )
    timing_manifest["timing_manifest_path"] = str(timing_manifest_path)
    timing_manifest["timing_manifest_sha256"] = _file_sha256(timing_manifest_path)
    timing_manifest["timing_system_artifact_path"] = str(system_artifact_path)
    timing_manifest["timing_system_artifact_sha256"] = _file_sha256(
        system_artifact_path
    )
    timing_manifest["timing_run_complete_path"] = str(completion_path)
    timing_manifest["timing_run_complete_sha256"] = _file_sha256(completion_path)
    return system, {**timing_manifest, "solution_artifact": solution_payload}, solutions


def _synchronize(backend: Any) -> None:
    cuda = getattr(backend.xp, "cuda", None)
    if cuda is not None:
        cuda.runtime.deviceSynchronize()


def _parse_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(part.strip().lower() for part in str(raw).split(",") if part.strip())
    if not methods:
        raise argparse.ArgumentTypeError("methods must contain at least one name")
    return methods


def load_controlled_config(
    path: str | Path,
    *,
    methods: Sequence[str] | None = None,
) -> ControlledConfig:
    """Load a benchmark ``experiment_config.json`` without changing its system."""
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("controlled config must be a JSON object")
    valid_fields = {field.name for field in fields(ControlledConfig)}
    unknown = sorted(set(payload) - valid_fields)
    if unknown:
        raise ValueError(f"unknown ControlledConfig fields: {unknown}")
    if "methods" in payload:
        payload["methods"] = tuple(str(value) for value in payload["methods"])
    if "diagnostic_topk" in payload:
        payload["diagnostic_topk"] = tuple(int(value) for value in payload["diagnostic_topk"])
    cfg = ControlledConfig(**payload)
    if methods is not None:
        cfg = replace(cfg, methods=tuple(str(method).strip().lower() for method in methods))
    if "cg" not in cfg.methods:
        raise ValueError("prediction audit requires method 'cg' for RMSE comparisons")
    return cfg


def load_test_arrays(
    dataset_path: str | Path,
    *,
    max_test: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load host-resident test arrays from the exact NPZ used for training."""
    path = Path(dataset_path).expanduser().resolve()
    try:
        x_info = inspect_stored_npy_member(path, "x_test")
        y_info = inspect_stored_npy_member(path, "y_test")
        if not x_info.shape or not y_info.shape:
            raise StoredNpzError("x_test and y_test must have a row dimension")
        full_n_test = int(x_info.shape[0])
        if int(y_info.shape[0]) != full_n_test:
            raise ValueError("x_test and y_test row counts differ")
        limit = full_n_test if max_test is None else min(int(max_test), full_n_test)
        if limit <= 0:
            raise ValueError("max_test must be positive or None")
        x_test = load_stored_npz_prefix(path, "x_test", limit)
        y_test = load_stored_npz_prefix(
            path,
            "y_test",
            limit,
            dtype=np.float64,
        ).reshape(-1)
    except StoredNpzError:
        with np.load(path) as loaded:
            missing = [name for name in ("x_test", "y_test") if name not in loaded.files]
            if missing:
                raise KeyError(f"dataset {path} is missing test arrays: {missing}")
            x_test = np.asarray(loaded["x_test"])
            y_test = np.asarray(loaded["y_test"], dtype=np.float64).reshape(-1)
        full_n_test = int(y_test.size)
        if max_test is not None:
            limit = int(max_test)
            if limit <= 0:
                raise ValueError("max_test must be positive or None")
            limit = min(limit, full_n_test)
            x_test = x_test[:limit]
            y_test = y_test[:limit]
    if x_test.ndim != 2:
        raise ValueError("x_test must be a two-dimensional array")
    if x_test.shape[0] != y_test.size:
        raise ValueError("x_test and y_test row counts differ")
    if y_test.size == 0:
        raise ValueError("test set is empty")
    return x_test, y_test, full_n_test


def chunked_test_rmse(
    system: PreparedSystem,
    beta_host: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    chunk_size: int,
    predict_fn: Callable[[Any, Any, Any, Any], Any] | None = None,
    strict_nufft_stage: str | None = None,
    observed_nufft_stages: set[str] | None = None,
) -> tuple[float, float]:
    """Compute RMSE while placing at most ``chunk_size`` test rows on the GPU."""
    size = int(chunk_size)
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if int(x_test.shape[0]) != int(y_test.size):
        raise ValueError("x_test and y_test row counts differ")

    backend = system.backend
    xp = backend.xp
    use_stage_reporting = predict_fn is None
    if predict_fn is None:
        predict_fn = predict_v1
    beta_gpu = xp.asarray(beta_host)
    squared_error_sum = 0.0
    _synchronize(backend)
    start_ns = time.perf_counter_ns()
    for start in range(0, int(y_test.size), size):
        stop = min(start + size, int(y_test.size))
        prediction_result = (
            predict_fn(
                backend,
                system.data_ctx,
                x_test[start:stop],
                beta_gpu,
                return_nufft_stage=True,
                allow_cpu_fallback=strict_nufft_stage is None,
            )
            if use_stage_reporting
            else predict_fn(backend, system.data_ctx, x_test[start:stop], beta_gpu)
        )
        prediction_stage = None
        if use_stage_reporting:
            prediction_value, prediction_stage = prediction_result
            prediction_stage = str(prediction_stage)
            if observed_nufft_stages is not None:
                observed_nufft_stages.add(prediction_stage)
            if strict_nufft_stage is not None and prediction_stage != str(
                strict_nufft_stage
            ):
                raise RuntimeError(
                    f"prediction NUFFT used {prediction_stage!r}, required "
                    f"{strict_nufft_stage!r}"
                )
        else:
            prediction_value = prediction_result
        prediction_gpu = xp.asarray(prediction_value, dtype=xp.float64).reshape(-1)
        target_gpu = xp.asarray(y_test[start:stop], dtype=xp.float64)
        if int(prediction_gpu.size) != stop - start:
            raise ValueError("prediction row count differs from the requested chunk")
        finite_prediction = xp.all(xp.isfinite(prediction_gpu))
        if not bool(
            finite_prediction.item()
            if hasattr(finite_prediction, "item")
            else finite_prediction
        ):
            raise FloatingPointError("prediction contains NaN or infinity")
        chunk_sse = xp.sum((prediction_gpu - target_gpu) ** 2)
        chunk_sse_value = float(
            chunk_sse.item() if hasattr(chunk_sse, "item") else chunk_sse
        )
        if not math.isfinite(chunk_sse_value):
            raise FloatingPointError("prediction squared error is not finite")
        squared_error_sum += chunk_sse_value
        del prediction_gpu, target_gpu, chunk_sse
    _synchronize(backend)
    prediction_seconds = (time.perf_counter_ns() - start_ns) * 1e-9
    rmse = math.sqrt(squared_error_sum / int(y_test.size))
    if not math.isfinite(rmse):
        raise FloatingPointError("prediction RMSE is not finite")
    return float(rmse), float(prediction_seconds)


def attach_cg_rmse_comparisons(
    rows: list[dict[str, Any]],
    *,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-10,
) -> None:
    """Add method/CG RMSE comparisons and an explicit equivalence decision."""
    if not (
        math.isfinite(float(relative_tolerance))
        and math.isfinite(float(absolute_tolerance))
    ):
        raise ValueError("RMSE equivalence tolerances must be finite")
    if float(relative_tolerance) < 0.0 or float(absolute_tolerance) < 0.0:
        raise ValueError("RMSE equivalence tolerances must be nonnegative")
    cg_rows = [row for row in rows if row.get("method") == "cg"]
    cg_rmse = _finite_or_none(cg_rows[0].get("test_rmse")) if len(cg_rows) == 1 else None
    for row in rows:
        rmse = _finite_or_none(row.get("test_rmse"))
        if row.get("method") == "cg" and rmse is not None:
            row["test_rmse_ratio_vs_cg"] = 1.0
            row["test_rmse_diff_vs_cg"] = 0.0
            row["test_rmse_relative_diff_vs_cg"] = 0.0
            row["prediction_equivalent_to_cg"] = True
        elif rmse is not None and cg_rmse is not None:
            row["test_rmse_ratio_vs_cg"] = rmse / cg_rmse if cg_rmse > 0.0 else None
            row["test_rmse_diff_vs_cg"] = rmse - cg_rmse
            row["test_rmse_relative_diff_vs_cg"] = (
                abs(rmse - cg_rmse) / abs(cg_rmse) if cg_rmse != 0.0 else None
            )
            row["prediction_equivalent_to_cg"] = bool(
                abs(rmse - cg_rmse)
                <= float(absolute_tolerance) + float(relative_tolerance) * abs(cg_rmse)
            )
        else:
            row["test_rmse_ratio_vs_cg"] = None
            row["test_rmse_diff_vs_cg"] = None
            row["test_rmse_relative_diff_vs_cg"] = None
            row["prediction_equivalent_to_cg"] = False


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_ROW_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_prediction_audit(
    cfg: ControlledConfig,
    *,
    timing_run_dir: str | Path | None = None,
    output_dir: str | Path,
    prediction_chunk_size: int = 100_000,
    warmup_solves: int = 0,
    max_test: int | None = None,
    config_source: str | Path | None = None,
    rmse_relative_tolerance: float = 1e-3,
    rmse_absolute_tolerance: float = 1e-10,
    strict_prediction_nufft: bool = False,
) -> Path:
    """Predict canonical timed solutions and write accuracy-only JSON/CSV evidence."""
    if int(warmup_solves) != 0:
        raise ValueError(
            "prediction audit no longer performs warmup or audit solves; "
            "set warmup_solves=0"
        )
    if int(prediction_chunk_size) <= 0:
        raise ValueError("prediction_chunk_size must be positive")
    if not (
        math.isfinite(float(rmse_relative_tolerance))
        and math.isfinite(float(rmse_absolute_tolerance))
    ):
        raise ValueError("RMSE equivalence tolerances must be finite")
    if float(rmse_relative_tolerance) < 0.0 or float(rmse_absolute_tolerance) < 0.0:
        raise ValueError("RMSE equivalence tolerances must be nonnegative")
    if "cg" not in cfg.methods:
        raise ValueError("prediction audit requires method 'cg' for RMSE comparisons")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    completion_path = destination / PREDICTION_AUDIT_COMPLETION_FILENAME
    completion_path.unlink(missing_ok=True)

    if timing_run_dir is None:
        if config_source is None:
            raise ValueError("timing_run_dir or config_source must identify the timing case")
        timing_run_dir = Path(config_source).expanduser().resolve().parent
    system, timing_evidence, timing_solutions = load_timing_prediction_inputs(
        cfg,
        timing_run_dir,
    )
    dataset_path = Path(str(system.manifest["dataset_path"])).resolve()
    test_dataset_provenance = verify_test_dataset_provenance(
        dataset_path,
        system.manifest,
    )
    x_test, y_test, full_n_test = load_test_arrays(dataset_path, max_test=max_test)
    rows: list[dict[str, Any]] = []

    for method in cfg.methods:
        solution_record = timing_solutions[str(method)]
        solve_row = dict(solution_record.get("timing_row") or {})
        beta_host = solution_record.get("beta_host")
        rmse: float | None = None
        prediction_seconds: float | None = None
        observed_prediction_stages: set[str] = set()
        solve_status = str(solve_row.get("status", "missing_timing_solution"))
        if beta_host is not None:
            try:
                rmse, prediction_seconds = chunked_test_rmse(
                    system,
                    beta_host,
                    x_test,
                    y_test,
                    chunk_size=int(prediction_chunk_size),
                    strict_nufft_stage=(
                        "cufinufft" if bool(strict_prediction_nufft) else None
                    ),
                    observed_nufft_stages=observed_prediction_stages,
                )
            except Exception as exc:
                solve_status = f"prediction_error:{type(exc).__name__}:{exc}"
        rows.append(
            {
                "system_id": system.system_id,
                "dataset": str(system.manifest.get("dataset_stem", cfg.dataset_stem)),
                "method": str(method),
                "method_kind": solution_record.get("method_kind")
                or solve_row.get("method_kind"),
                "solve_status": solve_status,
                "true_relres": _finite_or_none(solve_row.get("true_relres")),
                "test_rmse": rmse,
                "test_rmse_ratio_vs_cg": None,
                "test_rmse_diff_vs_cg": None,
                "test_rmse_relative_diff_vs_cg": None,
                "prediction_equivalent_to_cg": False,
                "prediction_seconds": prediction_seconds,
                "prediction_nufft_stages": ",".join(
                    sorted(observed_prediction_stages)
                ),
                "prediction_nufft_strict": bool(strict_prediction_nufft),
                "iterations": int(solve_row.get("iterations", -1)),
                "timing_repeat_idx": int(solution_record.get("timing_repeat_idx", -1)),
                "timing_order_position": int(
                    solution_record.get("timing_order_position", -1)
                ),
                "timing_solution_sha256": (
                    solution_record.get("beta", {}).get("sha256")
                    if isinstance(solution_record.get("beta"), dict)
                    else None
                ),
                "timing_solution_build_seconds": _finite_or_none(
                    solve_row.get("build_seconds")
                ),
                "timing_solution_solve_seconds": _finite_or_none(
                    solve_row.get("solve_seconds")
                ),
                "timing_system_reused": True,
                "timing_solution_reused": beta_host is not None,
                "n_test": int(y_test.size),
                "prediction_chunk_size": int(prediction_chunk_size),
                "audit_solve_build_seconds": None,
                "audit_solve_seconds": None,
                "audit_only_not_for_speed_claim": True,
            }
        )

    attach_cg_rmse_comparisons(
        rows,
        relative_tolerance=float(rmse_relative_tolerance),
        absolute_tolerance=float(rmse_absolute_tolerance),
    )
    final_system_id = system_fingerprint(
        system.data_ctx,
        float(system.reg_lambda),
        solve_rhs_gpu=system.rhs_gpu,
    )
    if final_system_id != system.system_id:
        raise RuntimeError(
            "the arrays defining A,b changed during prediction audit: "
            f"{system.system_id} -> {final_system_id}"
        )

    all_timing_solutions_reused = all(
        bool(row.get("timing_solution_reused")) for row in rows
    )
    audit_failures: list[str] = []
    for row in rows:
        method = str(row["method"])
        true_relres = _finite_or_none(row.get("true_relres"))
        if not str(row.get("solve_status", "")).lower().startswith("converged"):
            audit_failures.append(
                f"{method}: canonical timing solution status={row.get('solve_status')!r}"
            )
        elif true_relres is None or true_relres > float(cfg.tol):
            audit_failures.append(
                f"{method}: canonical timing true_relres={true_relres!r} exceeds {cfg.tol}"
            )
        if _finite_or_none(row.get("test_rmse")) is None:
            audit_failures.append(f"{method}: prediction RMSE is unavailable")
        if not bool(row.get("prediction_equivalent_to_cg")):
            audit_failures.append(
                f"{method}: RMSE is outside the declared CG-equivalence tolerance"
            )

    config_source_path = Path(config_source).expanduser().resolve() if config_source else None
    payload = {
        "schema_version": 2,
        "audit_role": (
            "prediction accuracy of canonical timed solutions only; "
            "no solve or prediction speed claim"
        ),
        "audit_pass": not audit_failures,
        "audit_failure_reasons": audit_failures,
        "system_id": system.system_id,
        "weights_sha256": system.manifest.get("weights_sha256"),
        "gf_sha256": system.manifest.get("gf_sha256"),
        "rhs_sha256": system.manifest.get("rhs_sha256"),
        "rhs_storage_sha256": system.manifest.get("rhs_storage_sha256"),
        "reg_lambda": float(system.reg_lambda),
        "system_unchanged": True,
        "audit_rebuilt_system": False,
        "timing_system_reused": True,
        "timing_solutions_reused": all_timing_solutions_reused,
        "timing_system_hashes_exact": True,
        "timing_solution_hashes_verified": True,
        "dataset": str(system.manifest.get("dataset_stem", cfg.dataset_stem)),
        "dataset_path": str(dataset_path),
        "source_bundle_sha256": system.manifest.get("source_bundle_sha256"),
        "dataset_content_index_sha256": system.manifest.get("dataset_content_index_sha256"),
        "dataset_metadata_sha256": system.manifest.get("dataset_metadata_sha256"),
        **test_dataset_provenance,
        **prediction_source_manifest(),
        "config_source": str(config_source_path) if config_source_path else None,
        "config_source_sha256": (
            hashlib.sha256(config_source_path.read_bytes()).hexdigest()
            if config_source_path is not None
            else None
        ),
        "timing_run_dir": str(Path(timing_run_dir).expanduser().resolve()),
        "timing_manifest_path": timing_evidence.get("timing_manifest_path"),
        "timing_manifest_sha256": timing_evidence.get("timing_manifest_sha256"),
        "timing_system_artifact_path": timing_evidence.get(
            "timing_system_artifact_path"
        ),
        "timing_system_artifact_sha256": timing_evidence.get(
            "timing_system_artifact_sha256"
        ),
        "timing_run_complete_path": timing_evidence.get(
            "timing_run_complete_path"
        ),
        "timing_run_complete_sha256": timing_evidence.get(
            "timing_run_complete_sha256"
        ),
        "timing_solution_manifest_path": timing_evidence.get(
            "solution_artifact", {}
        ).get("timing_solution_manifest_path"),
        "timing_solution_manifest_sha256": timing_evidence.get(
            "solution_artifact", {}
        ).get("timing_solution_manifest_sha256"),
        "timing_solution_artifact_path": timing_evidence.get(
            "solution_artifact", {}
        ).get("timing_solution_artifact_path"),
        "timing_solution_artifact_sha256": timing_evidence.get(
            "solution_artifact", {}
        ).get("timing_solution_artifact_sha256"),
        "timing_solution_selection_policy": timing_evidence.get(
            "solution_artifact", {}
        ).get("selection_policy"),
        "controlled_config": asdict(cfg),
        "warmup_solves_per_method": 0,
        "audit_solves_per_method": 0,
        "audit_solve_count": 0,
        "rmse_equivalence_relative_tolerance": float(rmse_relative_tolerance),
        "rmse_equivalence_absolute_tolerance": float(rmse_absolute_tolerance),
        "test_array_source": "x_test/y_test from the exact training NPZ",
        "test_target_scale": "stored NPZ y_test scale",
        "full_n_test": int(full_n_test),
        "evaluated_n_test": int(y_test.size),
        "test_subset_policy": "all" if int(y_test.size) == full_n_test else "first_n_prefix",
        "prediction_chunk_size": int(prediction_chunk_size),
        "strict_prediction_nufft": bool(strict_prediction_nufft),
        "required_prediction_nufft_stage": (
            "cufinufft" if bool(strict_prediction_nufft) else None
        ),
        "observed_prediction_nufft_stages": sorted(
            {
                stage
                for row in rows
                for stage in str(row.get("prediction_nufft_stages", "")).split(",")
                if stage
            }
        ),
        "rows": rows,
    }
    json_path = destination / PREDICTION_AUDIT_JSON_FILENAME
    csv_path = destination / PREDICTION_AUDIT_CSV_FILENAME
    json_temporary = destination / f".{PREDICTION_AUDIT_JSON_FILENAME}.tmp"
    csv_temporary = destination / f".{PREDICTION_AUDIT_CSV_FILENAME}.tmp"
    json_temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(csv_temporary, rows)
    json_temporary.replace(json_path)
    csv_temporary.replace(csv_path)
    completion_payload = {
        "schema_version": 1,
        "system_id": system.system_id,
        "methods": [str(method) for method in cfg.methods],
        "row_count": len(rows),
        "audit_pass": bool(payload["audit_pass"]),
        "evaluated_n_test": int(y_test.size),
        "prediction_source_bundle_sha256": payload[
            "prediction_source_bundle_sha256"
        ],
        "prediction_audit_json": PREDICTION_AUDIT_JSON_FILENAME,
        "prediction_audit_json_sha256": _file_sha256(json_path),
        "prediction_audit_csv": PREDICTION_AUDIT_CSV_FILENAME,
        "prediction_audit_csv_sha256": _file_sha256(csv_path),
    }
    completion_temporary = destination / f".{PREDICTION_AUDIT_COMPLETION_FILENAME}.tmp"
    completion_temporary.write_text(
        json.dumps(completion_payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    completion_temporary.replace(completion_path)
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an out-of-timing, chunked test-RMSE audit by reusing the exact "
            "system arrays and canonical measured betas from a controlled timing run."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--timing-run-dir",
        type=Path,
        default=None,
        help="Timing case directory; defaults to the parent of --config.",
    )
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=None,
        help="Optional comma-separated subset; it must include cg.",
    )
    parser.add_argument("--prediction-chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--warmup-solves",
        type=int,
        default=0,
        help="Compatibility option; must be zero because timed betas are reused.",
    )
    parser.add_argument("--rmse-relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--rmse-absolute-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--strict-prediction-nufft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require every prediction chunk to use cuFINUFFT; reject CPU fallback.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=0,
        help="First-N smoke-test cap; zero evaluates the entire stored test set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to controlled/outputs/prediction_audit_TIMESTAMP.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_controlled_config(args.config, methods=args.methods)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _HERE / "outputs" / datetime.now().strftime("prediction_audit_%Y%m%d_%H%M%S")
    destination = run_prediction_audit(
        cfg,
        timing_run_dir=args.timing_run_dir or args.config.parent,
        output_dir=output_dir,
        prediction_chunk_size=int(args.prediction_chunk_size),
        warmup_solves=int(args.warmup_solves),
        max_test=None if int(args.max_test) == 0 else int(args.max_test),
        config_source=args.config,
        rmse_relative_tolerance=float(args.rmse_relative_tolerance),
        rmse_absolute_tolerance=float(args.rmse_absolute_tolerance),
        strict_prediction_nufft=bool(args.strict_prediction_nufft),
    )
    print(f"Wrote prediction audit to {destination}")
    payload = json.loads(
        (destination / PREDICTION_AUDIT_JSON_FILENAME).read_text(encoding="utf-8")
    )
    return 0 if bool(payload.get("audit_pass")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
