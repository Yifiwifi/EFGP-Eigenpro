"""Fail-closed reporting for the two distinct experiment protocols.

Stage 1 consumes complete KRR-pipeline ``pipeline_summary.csv`` files.  Stage 2
consumes matched fixed-system ``matched_summary.csv`` files.  The schemas are
deliberately incompatible: a data-space KRR method can never silently appear
in a fixed-``A,b`` solver chart, and a Fourier preconditioner can never be
reported as a Nyström or RPCholesky KRR pipeline.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .benchmark import (
    _load_system_artifact_payload,
    system_component_fingerprints,
    system_fingerprint,
)
from .end_to_end import (
    END_TO_END_METHODS,
    STAGE2_SYSTEM_CONFIG_FIELDS,
    EndToEndConfig,
    summarize_pipeline_rows,
)


STAGE1_PROTOCOL = "end_to_end_krr"
STAGE2_PROTOCOL = "controlled_fixed_system"
STAGE1_OURS = "ours-binned-default"
STAGE2_PRIMARY_OURS = "default"
EXPECTED_SOLVER_TOTAL_DEFINITION = (
    "score selection + preconditioner construction + CG/PCG solve"
)
EXPECTED_STAGE1_TIMING_SCOPE = (
    "Each method owns algorithmic model setup and solve. train_total_seconds = "
    "setup_seconds + solving_phase_seconds. This is a method-owned algorithmic "
    "training total, not process wall clock: common dataset I/O, backend creation, "
    "and host-to-device staging are excluded, and prediction is separate."
)

STAGE1_FORMAL_METHODS = {
    "nystrom-krr",
    "rpcholesky-krr",
    "efgp-standard-cg",
    "efgp-standard-jacobi",
    "efgp-standard-full-eig",
    STAGE1_OURS,
}
STAGE2_OURS_METHODS = {"default", "active-eig", "active-inverse"}
STAGE2_FORMAL_ALLOWLIST = {
    "cg",
    "jacobi",
    "default",
    "active-inverse",
    "active-eig",
    "full-eig",
    "full-inverse",
}
STAGE2_REQUIRED_METHODS = {
    "cg",
    "jacobi",
    "default",
    "active-inverse",
    "active-eig",
    "full-eig",
}
FOURIER_ADAPTATION_ALIASES = {
    "nystrom",
    "rpcholesky",
    "fourier-nystrom-precond",
    "fourier-rpcholesky-precond",
}

STAGE1_REQUIRED = {
    "protocol_family",
    "timing_scope",
    "suite_profile",
    "case_id",
    "run_dir",
    "robustness_axes",
    "dataset_stem",
    "method",
    "n_train",
    "subset_seed",
    "subset_mode",
    "kernel_family",
    "nu",
    "variance",
    "reg_lambda",
    "lengthscale",
    "fourier_eps",
    "nufft_tol",
    "l2_scaled",
    "precision",
    "nufft_backend",
    "precompute_chunk_size",
    "box_budget",
    "setup_seconds_median",
    "solving_phase_seconds_median",
    "setup_seconds_at_median_total",
    "solving_phase_seconds_at_median_total",
    "train_total_seconds_median",
    "test_rmse_median",
    "accuracy_eligible",
    "performance_claim_eligible",
    "accuracy_max_rmse",
    "accuracy_min_r2",
    "accuracy_relative_tolerance",
    "measured_repeats",
    "successful_repeats",
    "expected_measured_repeats",
    "accuracy_evaluated_repeats",
    "accuracy_passed_repeats",
    "iterations_median",
    "test_r2_median",
    "resource_required_bytes",
    "resource_effective_cap_bytes",
    "resource_declared_cap_bytes",
    "resource_available_device_bytes",
}
STAGE1_RUN_REQUIRED = {
    "protocol_family",
    "timing_scope",
    "method",
    "repeat_idx",
    "is_warmup",
    "status",
    "setup_seconds",
    "solving_phase_seconds",
    "train_total_seconds",
    "test_rmse",
    "test_r2",
}
STAGE2_REQUIRED = {
    "method",
    "measured_repeats",
    "performance_claim_eligible",
    "solver_total_definition",
    "solver_total_seconds_median",
    "selection_seconds_median",
    "preconditioner_build_seconds_median",
    "solve_seconds_median",
}
STAGE2_RUN_REQUIRED = {
    "method",
    "system_id",
    "repeat_idx",
    "is_warmup",
    "status",
    "tol",
    "maxiter",
    "zero_initial_vector",
    "true_relres",
    "solver_total_definition",
    "solver_total_seconds",
    "selection_seconds",
    "preconditioner_build_seconds",
    "solve_seconds",
}

STAGE2_SYSTEM_ARTIFACT_REQUIRED_ARRAYS = {
    "weights_flat",
    "weights_np_flat",
    "gf",
    "rhs_storage",
    "rhs_solve",
}
STAGE2_SYSTEM_COMPONENT_HASH_FIELDS = (
    "weights_sha256",
    "gf_sha256",
    "rhs_sha256",
    "rhs_storage_sha256",
)

STAGE1_TABLE_COLUMNS = (
    "case_id",
    "declared_case_id",
    "source_file",
    "run_dir",
    "suite_profile",
    "robustness_axes",
    "dataset_stem",
    "dataset",
    "n_train",
    "subset_seed",
    "subset_mode",
    "kernel_family",
    "nu",
    "variance",
    "reg_lambda",
    "lengthscale",
    "fourier_eps",
    "nufft_tol",
    "l2_scaled",
    "precision",
    "nufft_backend",
    "precompute_chunk_size",
    "box_budget",
    "method",
    "formal_method",
    "status",
    "accuracy_eligible",
    "performance_claim_eligible",
    "accuracy_evidence_complete",
    "accuracy_max_rmse",
    "accuracy_min_r2",
    "accuracy_relative_tolerance",
    "expected_measured_repeats",
    "accuracy_evaluated_repeats",
    "accuracy_passed_repeats",
    "iterations_median",
    "resource_required_bytes",
    "resource_effective_cap_bytes",
    "resource_declared_cap_bytes",
    "resource_available_device_bytes",
    "speedup_claim_eligible",
    "test_rmse",
    "test_r2",
    "iterations_median",
    "setup_seconds",
    "solving_phase_seconds",
    "train_total_seconds",
    "speedup_vs_ours",
)

STAGE2_TABLE_COLUMNS = (
    "case_id",
    "source_file",
    "dataset",
    "dataset_stem",
    "n_train",
    "subset_seed",
    "subset_mode",
    "kernel_family",
    "nu",
    "variance",
    "reg_lambda",
    "lengthscale",
    "fourier_eps",
    "nufft_tol",
    "l2_scaled",
    "precision",
    "nufft_backend",
    "precompute_chunk_size",
    "method",
    "method_kind",
    "result_role",
    "reporting_class",
    "measured_repeats",
    "formal_included",
    "performance_claim_eligible",
    "fixed_system_verified",
    "system_id",
    "selection_seconds",
    "preconditioner_build_seconds",
    "solve_seconds",
    "solver_total_seconds",
    "solver_total_speedup_over_cg_median",
    "solver_total_speedup_over_cg_min",
    "solver_total_speedup_over_cg_max",
    "paired_comparisons",
    "paired_wins_over_cg",
    "solver_total_speedup_source",
    "solver_total_speedup_vs_best_baseline_median",
    "solver_total_speedup_vs_best_baseline_min",
    "solver_total_speedup_vs_best_baseline_max",
    "best_baseline_paired_comparisons",
    "best_baseline_speedup_source",
    "solver_total_source",
    "solver_total_definition",
    "component_sum_seconds",
    "component_sum_relative_error",
    "corrected_total_eligible",
    "corrected_total_verified",
    "target_regime_match",
    "method_matrix_complete",
    "selection_timing_protocol",
    "score_selection_seconds",
    "score_protocol_freeze_selection_seconds",
    "shared_ab_setup_seconds",
)


class ReportSchemaError(ValueError):
    """Raised when a summary is passed to the wrong reporting stage."""


@dataclass(frozen=True)
class TwoStageReportConfig:
    stage1_paths: tuple[str, ...]
    stage2_paths: tuple[str, ...]
    output_dir: str
    selected_target_path: str
    stage1_suite_path: str
    stage2_feasibility_path: str | None = None
    stage1_ours_method: str = STAGE1_OURS
    stage2_primary_ours_method: str = STAGE2_PRIMARY_OURS
    include_fourier_adaptations_in_formal_stage2: bool = False
    component_sum_relative_tolerance: float = 0.02
    make_plots: bool = True


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ReportSchemaError(f"{path}: CSV has no header")
        return [dict(row) for row in reader]


def _require_columns(
    path: Path, rows: list[dict[str, str]], required: set[str]
) -> None:
    if not rows:
        raise ReportSchemaError(f"{path}: summary is empty")
    missing = required.difference(rows[0])
    if missing:
        raise ReportSchemaError(
            f"{path}: missing required columns: {', '.join(sorted(missing))}"
        )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(number) if math.isfinite(number) else None


def _finite_positive(value: Any) -> bool:
    number = _as_float(value)
    return math.isfinite(number) and number > 0.0


def _finite_nonnegative(value: Any) -> bool:
    number = _as_float(value)
    return math.isfinite(number) and number >= 0.0


def _numbers_match(left: Any, right: Any, *, rel_tol: float = 1e-12) -> bool:
    lhs = _as_float(left)
    rhs = _as_float(right)
    return bool(
        math.isfinite(lhs)
        and math.isfinite(rhs)
        and math.isclose(lhs, rhs, rel_tol=rel_tol, abs_tol=1e-12)
    )


def _optional_numbers_match(left: Any, right: Any) -> bool:
    lhs = _as_float(left)
    rhs = _as_float(right)
    if not math.isfinite(lhs) and not math.isfinite(rhs):
        return True
    return _numbers_match(lhs, rhs)


_SYSTEM_STRING_FIELDS = {
    "dataset_stem",
    "subset_mode",
    "kernel_family",
    "precision",
    "nufft_backend",
}
_SYSTEM_INTEGER_FIELDS = {"n_train", "subset_seed", "precompute_chunk_size"}
_SYSTEM_BOOLEAN_FIELDS = {"l2_scaled"}


def _system_field_matches(field: str, observed: Any, expected: Any) -> bool:
    if field in _SYSTEM_STRING_FIELDS:
        return str(observed).strip() == str(expected).strip()
    if field in _SYSTEM_INTEGER_FIELDS:
        return _as_int(observed) == _as_int(expected)
    if field in _SYSTEM_BOOLEAN_FIELDS:
        return _as_bool(observed) == _as_bool(expected)
    return _numbers_match(observed, expected)


def _verify_stage2_system_artifact(
    artifact_path: Path,
    *,
    expected_system_id: str,
    external_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the fixed Fourier system from materialized NPZ arrays.

    A file-level digest only proves that a file did not change after its digest
    was recorded.  Formal Stage 2 evidence additionally needs the arrays that
    define ``A,b``.  The benchmark loader validates every declared descriptor
    and checksum; this function then recomputes the canonical system/component
    fingerprints from a NumPy-backed context and requires the embedded
    envelope, nested build manifest, and external timing manifest to agree.
    """
    try:
        artifact_metadata, arrays = _load_system_artifact_payload(artifact_path)
    except ValueError as exc:
        raise ReportSchemaError(
            f"{artifact_path}: invalid timing-system array artifact: {exc}"
        ) from exc

    missing_arrays = sorted(STAGE2_SYSTEM_ARTIFACT_REQUIRED_ARRAYS.difference(arrays))
    empty_arrays = sorted(
        name
        for name in STAGE2_SYSTEM_ARTIFACT_REQUIRED_ARRAYS.intersection(arrays)
        if np.asarray(arrays[name]).size == 0
    )
    if missing_arrays or empty_arrays:
        raise ReportSchemaError(
            f"{artifact_path}: timing-system evidence requires materialized arrays; "
            f"missing={missing_arrays}, empty={empty_arrays}"
        )

    nested_manifest = artifact_metadata.get("system_manifest")
    if not isinstance(nested_manifest, dict):
        raise ReportSchemaError(
            f"{artifact_path}: embedded nested system_manifest is required"
        )

    reg_lambda = _as_float(artifact_metadata.get("reg_lambda"))
    if (
        not _finite_positive(reg_lambda)
        or not _numbers_match(nested_manifest.get("reg_lambda"), reg_lambda)
        or not _numbers_match(external_manifest.get("reg_lambda"), reg_lambda)
    ):
        raise ReportSchemaError(
            f"{artifact_path}: embedded, nested, and external regularization differ"
        )

    data_ctx = SimpleNamespace(
        weights_gpu_flat=np.ascontiguousarray(arrays["weights_flat"]),
        weights_np_flat=np.ascontiguousarray(arrays["weights_np_flat"]),
        gf_gpu=np.ascontiguousarray(arrays["gf"]),
        rhs_gpu=np.ascontiguousarray(arrays["rhs_storage"]),
    )
    rhs_solve = np.ascontiguousarray(arrays["rhs_solve"])
    recomputed_system_id = system_fingerprint(
        data_ctx,
        reg_lambda,
        solve_rhs_gpu=rhs_solve,
    )
    recomputed_components = system_component_fingerprints(
        data_ctx,
        solve_rhs_gpu=rhs_solve,
    )
    if recomputed_system_id != expected_system_id:
        raise ReportSchemaError(
            f"{artifact_path}: materialized arrays do not reproduce the timed system_id"
        )

    layers = (
        ("embedded artifact manifest", artifact_metadata),
        ("nested system manifest", nested_manifest),
        ("external system manifest", external_manifest),
    )
    for label, layer in layers:
        if str(layer.get("system_id", "")).strip() != recomputed_system_id:
            raise ReportSchemaError(
                f"{artifact_path}: {label} system_id disagrees with materialized arrays"
            )
        mismatched_components = [
            field
            for field in STAGE2_SYSTEM_COMPONENT_HASH_FIELDS
            if str(layer.get(field, "")).strip() != str(recomputed_components[field])
        ]
        if mismatched_components:
            raise ReportSchemaError(
                f"{artifact_path}: {label} component fingerprints disagree with "
                f"materialized arrays: {mismatched_components}"
            )

    return artifact_metadata, nested_manifest


def _parse_axes(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        axes = [str(item).strip() for item in value]
    else:
        text = str(value or "").strip()
        if not text or text == "[]":
            return ()
        parsed: Any = None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    parsed = None
        if isinstance(parsed, (list, tuple)):
            axes = [str(item).strip() for item in parsed]
        else:
            axes = [part.strip() for part in text.split(",")]
    result = tuple(axis for axis in axes if axis)
    if len(result) != len(set(result)):
        raise ReportSchemaError(f"duplicate robustness axis labels: {result!r}")
    return result


def _load_required_json_object(path_value: str | Path, label: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ReportSchemaError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportSchemaError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportSchemaError(f"{label} must be a JSON object: {path}")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _case_id(path: Path, values: Sequence[Any]) -> str:
    payload = "|".join([str(path.resolve()), *(str(value) for value in values)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{path.parent.name}-{digest}"


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            values: dict[str, Any] = {}
            for key in columns:
                value = row.get(key, "")
                if isinstance(value, (float, np.floating)) and not math.isfinite(
                    float(value)
                ):
                    value = ""
                values[key] = value
            writer.writerow(values)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _stage1_group_key(row: Mapping[str, str]) -> tuple[Any, ...]:
    return (
        row.get("suite_profile", ""),
        row.get("case_id", ""),
    )


def _verify_stage1_case_evidence(
    source_path: Path, case_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    run_dirs = {str(row.get("run_dir", "")).strip() for row in case_rows}
    if len(run_dirs) != 1 or not next(iter(run_dirs)):
        raise ReportSchemaError(
            f"{source_path}: every Stage 1 case requires one explicit run_dir"
        )
    run_dir = Path(next(iter(run_dirs))).expanduser().resolve()
    config_path = run_dir / "experiment_config.json"
    completion_path = run_dir / "run_complete.json"
    runs_path = run_dir / "pipeline_runs.csv"
    config_payload = _load_required_json_object(
        config_path, "Stage 1 experiment config"
    )
    completion = _load_required_json_object(completion_path, "Stage 1 completion")
    run_rows = _read_csv(runs_path)
    _require_columns(runs_path, run_rows, STAGE1_RUN_REQUIRED)
    try:
        config_payload = dict(config_payload)
        config_payload["methods"] = tuple(config_payload.get("methods", ()))
        cfg = EndToEndConfig(**config_payload)
    except (TypeError, ValueError) as exc:
        raise ReportSchemaError(
            f"{config_path}: invalid EndToEndConfig payload"
        ) from exc
    if (
        int(cfg.warmup_repeats) < 1
        or int(cfg.measured_repeats) < 5
        or tuple(cfg.methods) != tuple(END_TO_END_METHODS)
    ):
        raise ReportSchemaError(
            f"{config_path}: formal Stage 1 requires one warmup, at least five "
            "measured repeats, and the exact six-method protocol"
        )
    normalized_runs: list[dict[str, Any]] = []
    observed_keys: list[tuple[str, bool, int]] = []
    for raw in run_rows:
        row = dict(raw)
        row["is_warmup"] = _as_bool(raw.get("is_warmup"))
        repeat_idx = _as_int(raw.get("repeat_idx"))
        if repeat_idx is None:
            raise ReportSchemaError(f"{runs_path}: repeat_idx must be an integer")
        row["repeat_idx"] = repeat_idx
        if (
            str(row.get("protocol_family", "")).strip() != STAGE1_PROTOCOL
            or str(row.get("timing_scope", "")).strip() != EXPECTED_STAGE1_TIMING_SCOPE
        ):
            raise ReportSchemaError(
                f"{runs_path}: repeat row has the wrong protocol/timing scope"
            )
        for field in TARGET_REGIME_FIELDS:
            if not _system_field_matches(field, row.get(field), getattr(cfg, field)):
                raise ReportSchemaError(
                    f"{runs_path}: repeat row {field} differs from experiment config"
                )
        if str(row.get("status", "")).lower() in {"ok", "converged"}:
            setup = _as_float(row.get("setup_seconds"))
            solve = _as_float(row.get("solving_phase_seconds"))
            total = _as_float(row.get("train_total_seconds"))
            if not (
                _finite_nonnegative(setup)
                and _finite_nonnegative(solve)
                and _finite_positive(total)
                and _numbers_match(setup + solve, total, rel_tol=1e-9)
            ):
                raise ReportSchemaError(
                    f"{runs_path}: successful repeat setup+solve does not equal total"
                )
        observed_keys.append(
            (str(row.get("method", "")), bool(row["is_warmup"]), repeat_idx)
        )
        normalized_runs.append(row)
    expected_keys = {
        (method, True, repeat_idx)
        for method in cfg.methods
        for repeat_idx in range(int(cfg.warmup_repeats))
    } | {
        (method, False, repeat_idx)
        for method in cfg.methods
        for repeat_idx in range(int(cfg.measured_repeats))
    }
    if (
        len(observed_keys) != len(expected_keys)
        or len(set(observed_keys)) != len(observed_keys)
        or set(observed_keys) != expected_keys
        or completion.get("artifact_complete") is not True
        or completion.get("all_rows_present") is not True
        or tuple(completion.get("methods", ())) != tuple(cfg.methods)
        or _as_int(completion.get("expected_row_count")) != len(expected_keys)
        or _as_int(completion.get("observed_row_count")) != len(normalized_runs)
    ):
        raise ReportSchemaError(
            f"{run_dir}: Stage 1 completion and repeat coverage are inconsistent"
        )
    recomputed = summarize_pipeline_rows(normalized_runs, cfg)
    recomputed_by_method = {str(row["method"]): row for row in recomputed}
    if set(recomputed_by_method) != set(STAGE1_FORMAL_METHODS):
        raise ReportSchemaError(
            f"{runs_path}: repeat evidence does not contain the exact six methods"
        )
    summary_by_method = {str(row.get("method", "")): row for row in case_rows}
    for observed in case_rows:
        for field in TARGET_REGIME_FIELDS:
            if not _system_field_matches(
                field, observed.get(field), getattr(cfg, field)
            ):
                raise ReportSchemaError(
                    f"{source_path}: summary {field} disagrees with experiment config"
                )
        if (
            _as_int(observed.get("box_budget")) != int(cfg.box_budget)
            or not _optional_numbers_match(
                observed.get("accuracy_max_rmse"), cfg.accuracy_max_rmse
            )
            or not _optional_numbers_match(
                observed.get("accuracy_min_r2"), cfg.accuracy_min_r2
            )
        ):
            raise ReportSchemaError(
                f"{source_path}: summary gates/budget disagree with experiment config"
            )
    numeric_fields = (
        "setup_seconds_median",
        "solving_phase_seconds_median",
        "setup_seconds_at_median_total",
        "solving_phase_seconds_at_median_total",
        "train_total_seconds_median",
        "test_rmse_median",
        "test_r2_median",
        "iterations_median",
        "accuracy_relative_tolerance",
        "accuracy_max_rmse",
        "accuracy_min_r2",
    )
    integer_fields = (
        "measured_repeats",
        "expected_measured_repeats",
        "successful_repeats",
        "accuracy_evaluated_repeats",
        "accuracy_passed_repeats",
    )
    boolean_fields = ("accuracy_eligible", "performance_claim_eligible")
    for method, expected in recomputed_by_method.items():
        observed = summary_by_method.get(method)
        if observed is None or str(observed.get("status", "")) != str(
            expected.get("status", "")
        ):
            raise ReportSchemaError(
                f"{source_path}: {method} summary status disagrees with pipeline_runs"
            )
        mismatches = [
            field
            for field in numeric_fields
            if not _optional_numbers_match(observed.get(field), expected.get(field))
        ]
        mismatches.extend(
            field
            for field in integer_fields
            if _as_int(observed.get(field)) != _as_int(expected.get(field))
        )
        mismatches.extend(
            field
            for field in boolean_fields
            if _as_bool(observed.get(field)) != bool(expected.get(field))
        )
        if mismatches:
            raise ReportSchemaError(
                f"{source_path}: {method} summary disagrees with repeat evidence: "
                f"{sorted(set(mismatches))}"
            )
    return recomputed_by_method


def load_stage1_summaries(
    paths: Iterable[str | Path], ours_method: str = STAGE1_OURS
) -> list[dict[str, Any]]:
    """Load and normalize only complete KRR-pipeline summaries."""
    normalized: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        rows = _read_csv(path)
        _require_columns(path, rows, STAGE1_REQUIRED)
        protocols = {str(row.get("protocol_family", "")).strip() for row in rows}
        if protocols != {STAGE1_PROTOCOL}:
            raise ReportSchemaError(
                f"{path}: Stage 1 requires protocol_family={STAGE1_PROTOCOL!r}; "
                f"found {sorted(protocols)!r}"
            )
        if any("solver_total_seconds_median" in row for row in rows):
            raise ReportSchemaError(
                f"{path}: fixed-system solver total found in Stage 1 input"
            )
        timing_scopes = {str(row.get("timing_scope", "")).strip() for row in rows}
        if timing_scopes != {EXPECTED_STAGE1_TIMING_SCOPE}:
            raise ReportSchemaError(
                f"{path}: Stage 1 timing_scope must state method-owned setup+solve "
                "with common dataset I/O/H2D and prediction excluded"
            )

        groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
        for row in rows:
            groups.setdefault(_stage1_group_key(row), []).append(row)
        for group_key, case_rows in groups.items():
            suite_profile = str(group_key[0]).strip()
            declared_case_id = str(group_key[1]).strip()
            if suite_profile not in {
                "scale_10m_300m",
                "robustness_at_selected_target",
            }:
                raise ReportSchemaError(
                    f"{path}: unsupported or blank suite_profile={suite_profile!r}"
                )
            if not declared_case_id:
                raise ReportSchemaError(f"{path}: Stage 1 case_id must be non-empty")
            case = f"{suite_profile}:{declared_case_id}"
            metadata_fields = (
                *TARGET_REGIME_FIELDS,
                "box_budget",
                "robustness_axes",
            )
            inconsistent = [
                field
                for field in metadata_fields
                if len({str(row.get(field, "")).strip() for row in case_rows}) != 1
            ]
            if inconsistent:
                raise ReportSchemaError(
                    f"{path}: inconsistent per-method case metadata: {inconsistent}"
                )
            axes = _parse_axes(case_rows[0].get("robustness_axes"))
            if suite_profile == "scale_10m_300m" and axes:
                raise ReportSchemaError(
                    f"{path}: scale case {declared_case_id!r} cannot declare robustness axes"
                )
            if suite_profile == "robustness_at_selected_target" and not axes:
                raise ReportSchemaError(
                    f"{path}: robustness case {declared_case_id!r} requires explicit axes"
                )
            dataset_stem = str(case_rows[0].get("dataset_stem", "")).strip()
            dataset_family = str(
                case_rows[0].get("declared_dataset_family")
                or case_rows[0].get("dataset_family")
                or ""
            ).strip()
            if not dataset_stem or not dataset_family:
                raise ReportSchemaError(
                    f"{path}: dataset_stem and declared dataset family are required"
                )
            template = case_rows[0]
            if not str(template.get("kernel_family", "")).strip():
                raise ReportSchemaError(f"{path}: kernel_family must be non-empty")
            if not (
                (_as_int(template.get("n_train")) or 0) > 0
                and (_as_int(template.get("subset_seed")) or 0) >= 0
                and str(template.get("subset_mode", "")).strip()
                and _finite_positive(template.get("nu"))
                and _finite_positive(template.get("variance"))
                and _finite_nonnegative(template.get("reg_lambda"))
                and _finite_positive(template.get("lengthscale"))
                and _finite_positive(template.get("fourier_eps"))
                and _finite_positive(template.get("nufft_tol"))
                and str(template.get("precision", "")).strip()
                and str(template.get("nufft_backend", "")).strip()
                and (_as_int(template.get("precompute_chunk_size")) or 0) > 0
                and (_as_int(template.get("box_budget")) or 0) > 0
            ):
                raise ReportSchemaError(
                    f"{path}: invalid numeric Stage 1 case metadata for {declared_case_id}"
                )
            duplicates = [
                method
                for method in {str(row["method"]) for row in case_rows}
                if sum(str(row["method"]) == method for row in case_rows) > 1
            ]
            if duplicates:
                raise ReportSchemaError(
                    f"{path}: duplicate Stage 1 method(s) in one case: {duplicates}"
                )
            unknown = sorted(
                {str(row["method"]) for row in case_rows}.difference(
                    STAGE1_FORMAL_METHODS
                )
            )
            if unknown:
                raise ReportSchemaError(
                    f"{path}: unknown Stage 1 method label(s): {unknown}"
                )
            present_methods = {str(row["method"]) for row in case_rows}
            if present_methods != STAGE1_FORMAL_METHODS:
                raise ReportSchemaError(
                    f"{path}: Stage 1 case must contain exactly the six formal KRR "
                    f"methods; missing={sorted(STAGE1_FORMAL_METHODS - present_methods)}"
                )
            verified_by_method = _verify_stage1_case_evidence(path, case_rows)
            case_rows = [
                {
                    **row,
                    **verified_by_method[str(row["method"])],
                }
                for row in case_rows
            ]
            ours = next(
                (row for row in case_rows if str(row["method"]) == ours_method), None
            )

            def accuracy_evidence(row: Mapping[str, Any]) -> bool:
                max_rmse = _as_float(row.get("accuracy_max_rmse"))
                min_r2 = _as_float(row.get("accuracy_min_r2"))
                expected = _as_int(row.get("expected_measured_repeats"))
                evaluated = _as_int(row.get("accuracy_evaluated_repeats"))
                passed = _as_int(row.get("accuracy_passed_repeats"))
                has_absolute_gate = math.isfinite(max_rmse) or math.isfinite(min_r2)
                return bool(
                    has_absolute_gate
                    and expected is not None
                    and expected > 0
                    and evaluated == expected
                    and passed == expected
                )

            ours_total = (
                _as_float(ours.get("train_total_seconds_median")) if ours else math.nan
            )
            ours_accuracy = bool(
                ours
                and _as_bool(ours.get("accuracy_eligible"))
                and accuracy_evidence(ours)
            )
            ours_performance = bool(
                ours and _as_bool(ours.get("performance_claim_eligible"))
            )
            for row in case_rows:
                evidence_complete = accuracy_evidence(row)
                accuracy = bool(
                    _as_bool(row.get("accuracy_eligible")) and evidence_complete
                )
                performance = _as_bool(row.get("performance_claim_eligible"))
                total = _as_float(row.get("train_total_seconds_median"))
                paired_setup = _as_float(row.get("setup_seconds_at_median_total"))
                paired_solve = _as_float(
                    row.get("solving_phase_seconds_at_median_total")
                )
                if math.isfinite(total) and not (
                    _finite_nonnegative(paired_setup)
                    and _finite_nonnegative(paired_solve)
                    and _numbers_match(paired_setup + paired_solve, total, rel_tol=1e-9)
                ):
                    raise ReportSchemaError(
                        f"{path}: {declared_case_id}/{row['method']} paired setup+solve "
                        "does not equal train_total_seconds_median"
                    )
                speedup_eligible = bool(
                    accuracy
                    and performance
                    and ours_accuracy
                    and ours_performance
                    and _finite_positive(total)
                    and _finite_positive(ours_total)
                )
                normalized.append(
                    {
                        "case_id": case,
                        "declared_case_id": declared_case_id,
                        "source_file": str(path),
                        "run_dir": str(row.get("run_dir", "")),
                        "suite_profile": suite_profile,
                        "robustness_axes": axes,
                        "dataset": dataset_family,
                        "dataset_stem": dataset_stem,
                        "n_train": _as_int(row.get("n_train")),
                        "subset_seed": _as_int(row.get("subset_seed")),
                        "subset_mode": str(row.get("subset_mode", "")),
                        "kernel_family": str(row.get("kernel_family", "")),
                        "nu": _as_float(row.get("nu")),
                        "variance": _as_float(row.get("variance")),
                        "reg_lambda": _as_float(row.get("reg_lambda")),
                        "lengthscale": _as_float(row.get("lengthscale")),
                        "fourier_eps": _as_float(row.get("fourier_eps")),
                        "nufft_tol": _as_float(row.get("nufft_tol")),
                        "l2_scaled": _as_bool(row.get("l2_scaled")),
                        "precision": str(row.get("precision", "")),
                        "nufft_backend": str(row.get("nufft_backend", "")),
                        "precompute_chunk_size": _as_int(
                            row.get("precompute_chunk_size")
                        ),
                        "box_budget": _as_int(row.get("box_budget")),
                        "method": str(row["method"]),
                        "formal_method": str(row["method"]) in STAGE1_FORMAL_METHODS,
                        "status": str(row.get("status", "")),
                        "accuracy_eligible": accuracy,
                        "performance_claim_eligible": performance,
                        "accuracy_evidence_complete": evidence_complete,
                        "accuracy_max_rmse": _as_float(row.get("accuracy_max_rmse")),
                        "accuracy_min_r2": _as_float(row.get("accuracy_min_r2")),
                        "accuracy_relative_tolerance": _as_float(
                            row.get("accuracy_relative_tolerance")
                        ),
                        "expected_measured_repeats": _as_int(
                            row.get("expected_measured_repeats")
                        ),
                        "accuracy_evaluated_repeats": _as_int(
                            row.get("accuracy_evaluated_repeats")
                        ),
                        "accuracy_passed_repeats": _as_int(
                            row.get("accuracy_passed_repeats")
                        ),
                        "iterations_median": _as_float(row.get("iterations_median")),
                        "resource_required_bytes": _as_int(
                            row.get("resource_required_bytes")
                        ),
                        "resource_effective_cap_bytes": _as_int(
                            row.get("resource_effective_cap_bytes")
                        ),
                        "resource_declared_cap_bytes": _as_int(
                            row.get("resource_declared_cap_bytes")
                        ),
                        "resource_available_device_bytes": _as_int(
                            row.get("resource_available_device_bytes")
                        ),
                        "speedup_claim_eligible": speedup_eligible,
                        "test_rmse": _as_float(row.get("test_rmse_median")),
                        "test_r2": _as_float(row.get("test_r2_median")),
                        # Paired components from the repeat attaining median total.
                        "setup_seconds": paired_setup,
                        "solving_phase_seconds": paired_solve,
                        "train_total_seconds": total,
                        # Comparator total / our total; above one favors ours.
                        "speedup_vs_ours": (
                            total / ours_total if speedup_eligible else math.nan
                        ),
                    }
                )
    return normalized


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReportSchemaError(f"{path}: expected a JSON object")
    return payload


TARGET_REGIME_FIELDS = tuple(STAGE2_SYSTEM_CONFIG_FIELDS)


def load_selected_target(path: str | Path) -> dict[str, Any]:
    target = _load_required_json_object(path, "selected_target_regime")
    missing = [field for field in TARGET_REGIME_FIELDS if target.get(field) is None]
    if missing:
        raise ReportSchemaError(
            f"selected_target_regime is missing fields: {', '.join(missing)}"
        )
    if not str(target.get("selection_rule", "")).strip():
        raise ReportSchemaError(
            "selected_target_regime must record the frozen selection_rule"
        )
    subset_seed = _as_int(target.get("subset_seed"))
    precompute_chunk_size = _as_int(target.get("precompute_chunk_size"))
    if not (
        all(str(target[field]).strip() for field in _SYSTEM_STRING_FIELDS)
        and str(target["subset_mode"]).strip().lower() in {"prefix", "random"}
        and (_as_int(target["n_train"]) or 0) > 0
        and subset_seed is not None
        and subset_seed >= 0
        and precompute_chunk_size is not None
        and precompute_chunk_size > 0
        and isinstance(target.get("l2_scaled"), bool)
        and _finite_positive(target["nu"])
        and _finite_positive(target["variance"])
        and _finite_nonnegative(target["reg_lambda"])
        and _finite_positive(target["lengthscale"])
        and _finite_positive(target["fourier_eps"])
        and _finite_positive(target["nufft_tol"])
    ):
        raise ReportSchemaError("selected_target_regime contains invalid values")
    return target


def load_stage1_suite(path: str | Path) -> dict[str, Any]:
    suite = _load_required_json_object(path, "Stage 1 suite")
    if suite.get("protocol_family") != STAGE1_PROTOCOL:
        raise ReportSchemaError(
            f"Stage 1 suite protocol_family must be {STAGE1_PROTOCOL!r}"
        )
    profiles = suite.get("profiles")
    if not isinstance(profiles, dict):
        raise ReportSchemaError("Stage 1 suite must contain profiles")
    scale = profiles.get("scale_10m_300m")
    robust = profiles.get("robustness_at_selected_target")
    if not isinstance(scale, dict) or not isinstance(robust, dict):
        raise ReportSchemaError(
            "Stage 1 suite requires scale_10m_300m and robustness_at_selected_target"
        )
    base = suite.get("base")
    if not isinstance(base, dict) or (_as_int(base.get("box_budget")) or 0) <= 0:
        raise ReportSchemaError("Stage 1 suite base.box_budget must be positive")
    for field in (
        "lambda_values",
        "lengthscale_values",
        "box_budget_values",
        "datasets",
    ):
        values = robust.get(field)
        if not isinstance(values, list) or not values:
            raise ReportSchemaError(
                f"Stage 1 robustness profile requires non-empty {field}"
            )
    for dataset in robust["datasets"]:
        if not isinstance(dataset, dict):
            raise ReportSchemaError(
                "every declared robustness dataset must be an object"
            )
        stem = str(dataset.get("dataset_stem", "")).strip()
        stems_by_n = dataset.get("dataset_stems_by_n_train")
        if not stem and not isinstance(stems_by_n, dict):
            raise ReportSchemaError(
                "every declared robustness dataset requires dataset_stem or "
                "dataset_stems_by_n_train"
            )
        if not str(dataset.get("dataset_family", "")).strip():
            raise ReportSchemaError(
                "every declared robustness dataset requires dataset_family"
            )
    return suite


def load_stage2_feasibility(
    path: str | Path | None,
    *,
    selected_target: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if path is None:
        raise ReportSchemaError(
            "Stage 2 requires a prospective feasibility artifact bound to the frozen target"
        )
    payload = _load_required_json_object(path, "Stage 2 feasibility matrix")
    if payload.get("protocol_family") != STAGE2_PROTOCOL:
        raise ReportSchemaError(
            f"Stage 2 feasibility protocol_family must be {STAGE2_PROTOCOL!r}"
        )
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        raise ReportSchemaError("Stage 2 feasibility matrix requires a methods object")
    if str(payload.get("decision_basis", "")).strip() != (
        "prospective configured box-budget cap before timing"
    ):
        raise ReportSchemaError(
            "Stage 2 feasibility matrix has an invalid or post-hoc decision basis"
        )
    target_mismatches = [
        field
        for field in TARGET_REGIME_FIELDS
        if not _system_field_matches(
            field, payload.get(field), selected_target.get(field)
        )
    ]
    if target_mismatches:
        raise ReportSchemaError(
            "Stage 2 feasibility matrix does not match the frozen target: "
            f"{target_mismatches}"
        )
    declared_matches: list[dict[str, Any]] = []
    for case in suite["profiles"]["scale_10m_300m"].get("cases", []):
        declared = dict(suite["base"])
        declared.update(case)
        if all(
            _system_field_matches(
                field, declared.get(field), selected_target.get(field)
            )
            for field in TARGET_REGIME_FIELDS
        ):
            declared_matches.append(declared)
    if len(declared_matches) != 1:
        raise ReportSchemaError(
            "frozen target must match exactly one suite-declared scale case for "
            "Stage 2 feasibility"
        )
    declared = declared_matches[0]
    box_budget = _as_int(payload.get("box_budget"))
    inverse_max_size = _as_int(payload.get("inverse_max_size"))
    if (
        box_budget != _as_int(declared.get("box_budget"))
        or inverse_max_size != _as_int(declared.get("inverse_max_size"))
        or box_budget is None
        or inverse_max_size is None
        or box_budget <= 0
        or inverse_max_size <= 0
    ):
        raise ReportSchemaError(
            "Stage 2 feasibility caps do not match the suite-declared frozen target"
        )
    unknown = sorted(set(methods).difference(STAGE2_FORMAL_ALLOWLIST))
    if unknown:
        raise ReportSchemaError(
            f"Stage 2 feasibility matrix has unknown method(s): {unknown}"
        )
    missing = sorted(STAGE2_REQUIRED_METHODS.difference(methods))
    if missing:
        raise ReportSchemaError(
            f"Stage 2 feasibility matrix omits required method(s): {missing}"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for method, raw in methods.items():
        if isinstance(raw, bool):
            entry = {"feasible": raw, "reason": ""}
        elif isinstance(raw, dict):
            entry = dict(raw)
        else:
            raise ReportSchemaError(
                f"Stage 2 feasibility entry for {method!r} must be bool or object"
            )
        feasible = _as_bool(entry.get("feasible"))
        reason = str(entry.get("reason", "")).strip()
        if not feasible and not reason:
            raise ReportSchemaError(
                f"infeasible Stage 2 method {method!r} requires a reason"
            )
        normalized[str(method)] = {**entry, "feasible": feasible, "reason": reason}
    for mandatory in ("cg", "jacobi", "default", "active-eig", "full-eig"):
        if not normalized[mandatory]["feasible"]:
            raise ReportSchemaError(
                f"Stage 2 mandatory method {mandatory!r} cannot be declared infeasible"
            )
    expected_inverse_feasible = box_budget <= inverse_max_size
    if normalized["active-inverse"]["feasible"] != expected_inverse_feasible:
        raise ReportSchemaError(
            "active-inverse feasibility must equal the prospective rule "
            "box_budget <= inverse_max_size"
        )
    return normalized


def _stage2_method_class(method: str, method_kind: str, result_role: str) -> str:
    if (
        method in FOURIER_ADAPTATION_ALIASES
        or method_kind in FOURIER_ADAPTATION_ALIASES
    ):
        return "exploratory_fourier_preconditioner"
    if result_role == "diagnostic_only":
        return "diagnostic_only"
    if method in STAGE2_OURS_METHODS:
        return "proposed_fixed_system_solver"
    return "formal_fixed_system_baseline"


def load_stage2_summaries(
    paths: Iterable[str | Path],
    *,
    selected_target: Mapping[str, Any],
    feasibility: Mapping[str, Mapping[str, Any]],
    component_sum_relative_tolerance: float = 0.02,
    include_fourier_adaptations_in_formal: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load fixed-``A,b`` summaries and their separately reported shared setup."""
    if include_fourier_adaptations_in_formal:
        raise ReportSchemaError(
            "Fourier Nystrom/RPCholesky adaptations cannot enter the formal Stage 2 table"
        )
    if not (0.0 <= float(component_sum_relative_tolerance) <= 0.1):
        raise ReportSchemaError("component_sum_relative_tolerance must lie in [0, 0.1]")
    expected_methods = {
        method for method, entry in feasibility.items() if bool(entry.get("feasible"))
    }
    normalized: list[dict[str, Any]] = []
    setup_rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        rows = _read_csv(path)
        _require_columns(path, rows, STAGE2_REQUIRED)
        if any(
            "train_total_seconds_median" in row or "accuracy_eligible" in row
            for row in rows
        ):
            raise ReportSchemaError(
                f"{path}: end-to-end KRR fields found in Stage 2 input"
            )
        protocols = {
            str(row.get("protocol_family", "")).strip()
            for row in rows
            if str(row.get("protocol_family", "")).strip()
        }
        if protocols and protocols != {STAGE2_PROTOCOL}:
            raise ReportSchemaError(
                f"{path}: Stage 2 protocol must be {STAGE2_PROTOCOL!r}; "
                f"found {sorted(protocols)!r}"
            )

        labels = [str(row.get("method", "")).strip() for row in rows]
        duplicates = sorted({method for method in labels if labels.count(method) > 1})
        if duplicates:
            raise ReportSchemaError(
                f"{path}: duplicate Stage 2 method row(s): {duplicates}"
            )
        accepted = STAGE2_FORMAL_ALLOWLIST | FOURIER_ADAPTATION_ALIASES
        unknown = sorted(set(labels).difference(accepted))
        if unknown:
            raise ReportSchemaError(
                f"{path}: unknown Stage 2 method spelling(s): {unknown}"
            )
        present_formal = set(labels).intersection(STAGE2_FORMAL_ALLOWLIST)
        missing_expected = sorted(expected_methods.difference(present_formal))
        undeclared_formal = sorted(present_formal.difference(feasibility))
        unexpected_infeasible = sorted(
            method
            for method in present_formal
            if method in feasibility and not bool(feasibility[method].get("feasible"))
        )
        if missing_expected or undeclared_formal or unexpected_infeasible:
            raise ReportSchemaError(
                f"{path}: Stage 2 method matrix mismatch; missing feasible methods="
                f"{missing_expected}, undeclared formal methods={undeclared_formal}, "
                f"present-but-declared-infeasible={unexpected_infeasible}"
            )

        runs_path = path.parent / "matched_runs.csv"
        run_rows = _read_csv(runs_path)
        _require_columns(runs_path, run_rows, STAGE2_RUN_REQUIRED)
        measured_runs = [row for row in run_rows if not _as_bool(row.get("is_warmup"))]
        runs_by_method: dict[str, list[dict[str, str]]] = {}
        for run in measured_runs:
            run_method = str(run.get("method", "")).strip()
            if run_method not in set(labels):
                raise ReportSchemaError(
                    f"{runs_path}: measured run has no summary row: {run_method!r}"
                )
            if str(run.get("solver_total_definition", "")).strip() != (
                EXPECTED_SOLVER_TOTAL_DEFINITION
            ):
                raise ReportSchemaError(
                    f"{runs_path}: {run_method} has an invalid solver_total_definition"
                )
            components = tuple(
                _as_float(run.get(key))
                for key in (
                    "selection_seconds",
                    "preconditioner_build_seconds",
                    "solve_seconds",
                )
            )
            run_total = _as_float(run.get("solver_total_seconds"))
            if not _finite_positive(run_total) or not all(
                _finite_nonnegative(value) for value in components
            ):
                raise ReportSchemaError(
                    f"{runs_path}: {run_method} measured total/components must be finite"
                )
            if not math.isclose(
                sum(components), run_total, rel_tol=1e-12, abs_tol=1e-9
            ):
                raise ReportSchemaError(
                    f"{runs_path}: {run_method} measured total is not the component sum"
                )
            runs_by_method.setdefault(run_method, []).append(run)

        manifest_path = path.parent / "system_manifest.json"
        manifest = _load_json_if_present(manifest_path)
        if not manifest:
            raise ReportSchemaError(
                f"{path}: sibling system_manifest.json is required for fixed-A,b verification"
            )
        initial_system_id = str(manifest.get("system_id") or "").strip()
        final_system_id = str(manifest.get("final_system_id") or "").strip()
        system_id = final_system_id
        fixed_verified = bool(
            manifest.get("system_unchanged") is True
            and initial_system_id
            and final_system_id
            and initial_system_id == final_system_id
        )
        if not fixed_verified:
            raise ReportSchemaError(
                f"{manifest_path}: system_unchanged=true and identical non-empty "
                "initial/final system ids are required"
            )
        for run in run_rows:
            if str(run.get("system_id", "")).strip() != system_id:
                raise ReportSchemaError(
                    f"{runs_path}: Stage 2 run system_id does not match the verified "
                    f"manifest id for method {run.get('method')!r}"
                )

        completion_path = path.parent / "run_complete.json"
        completion = _load_json_if_present(completion_path)
        if not completion:
            raise ReportSchemaError(
                f"{completion_path}: completion artifact is required for fixed-A,b verification"
            )
        experiment_config_path = path.parent / "experiment_config.json"
        experiment_config = _load_json_if_present(experiment_config_path)
        if not experiment_config:
            raise ReportSchemaError(
                f"{experiment_config_path}: exact Stage 2 experiment config is required"
            )
        formal_measured_repeats = _as_int(experiment_config.get("measured_repeats"))
        formal_warmup_repeats = _as_int(experiment_config.get("warmup_repeats"))
        configured_tol = _as_float(experiment_config.get("tol"))
        configured_maxiter = _as_int(experiment_config.get("maxiter"))
        configured_methods = [
            str(method) for method in experiment_config.get("methods", [])
        ]
        if (
            formal_measured_repeats is None
            or formal_measured_repeats < 5
            or formal_warmup_repeats is None
            or formal_warmup_repeats < 1
            or not _finite_positive(configured_tol)
            or configured_maxiter is None
            or configured_maxiter < 1
            or experiment_config.get("zero_initial_vector") is not True
            or configured_methods != labels
            or _as_int(completion.get("measured_repeats")) != formal_measured_repeats
            or _as_int(completion.get("warmup_repeats")) != formal_warmup_repeats
            or not _numbers_match(completion.get("tol"), configured_tol)
            or _as_int(completion.get("maxiter")) != configured_maxiter
            or completion.get("zero_initial_vector") is not True
            or [str(method) for method in completion.get("methods", [])] != labels
            or _as_int(completion.get("summary_row_count")) != len(labels)
            or _as_int(completion.get("run_row_count")) != len(run_rows)
        ):
            raise ReportSchemaError(
                f"{completion_path}: Stage 2 config/completion must agree on at least "
                "one warmup, five measured repeats, methods, tol/maxiter, the zero "
                "initial vector protocol, and row counts"
            )
        for run in run_rows:
            run_method = str(run.get("method", "")).strip()
            if run_method not in set(labels):
                raise ReportSchemaError(
                    f"{runs_path}: Stage 2 run has no summary row: {run_method!r}"
                )
            if (
                not _numbers_match(run.get("tol"), configured_tol)
                or _as_int(run.get("maxiter")) != configured_maxiter
                or not _as_bool(run.get("zero_initial_vector"))
            ):
                raise ReportSchemaError(
                    f"{runs_path}: every warmup/measured row must use the configured "
                    "common tol/maxiter and zero_initial_vector=True"
                )
        artifact_name = str(completion.get("timing_system_artifact", "")).strip()
        if not artifact_name or Path(artifact_name).name != artifact_name:
            raise ReportSchemaError(
                f"{completion_path}: timing_system_artifact must be a local filename"
            )
        artifact_path = path.parent / artifact_name
        expected_artifact_sha = str(
            completion.get("timing_system_artifact_sha256", "")
        ).strip()
        manifest_artifact_sha = str(manifest.get("system_artifact_sha256", "")).strip()
        if (
            str(completion.get("system_id", "")).strip() != system_id
            or not artifact_path.is_file()
            or not expected_artifact_sha
            or expected_artifact_sha != manifest_artifact_sha
        ):
            raise ReportSchemaError(
                f"{completion_path}: completion/system-artifact identity is incomplete"
            )
        observed_artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if observed_artifact_sha != expected_artifact_sha:
            raise ReportSchemaError(
                f"{artifact_path}: SHA-256 does not match completion/manifest"
            )
        artifact_metadata, nested_artifact_manifest = _verify_stage2_system_artifact(
            artifact_path,
            expected_system_id=system_id,
            external_manifest=manifest,
        )
        artifact_system_config = artifact_metadata.get("system_config")
        nested_system_config = nested_artifact_manifest.get("system_config")
        manifest_system_config = manifest.get("system_config")
        if (
            not isinstance(artifact_system_config, dict)
            or not isinstance(nested_system_config, dict)
            or not isinstance(manifest_system_config, dict)
        ):
            raise ReportSchemaError(
                f"{artifact_path}: canonical system_config metadata is required"
            )
        encoded_system_config = json.dumps(
            artifact_system_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        observed_config_sha = hashlib.sha256(encoded_system_config).hexdigest()
        if (
            artifact_system_config != nested_system_config
            or artifact_system_config != manifest_system_config
            or observed_config_sha
            != str(artifact_metadata.get("system_config_sha256", ""))
            or observed_config_sha
            != str(nested_artifact_manifest.get("system_config_sha256", ""))
            or observed_config_sha != str(manifest.get("system_config_sha256", ""))
            or any(
                not _system_field_matches(
                    field,
                    artifact_system_config.get(field),
                    selected_target.get(field),
                )
                for field in TARGET_REGIME_FIELDS
            )
        ):
            raise ReportSchemaError(
                f"{artifact_path}: canonical system config/hash does not match the "
                "manifest and frozen target"
            )
        dataset_stem = str(manifest.get("dataset_stem", "")).strip()
        dataset = str(manifest.get("dataset_task_type") or dataset_stem)
        n_train = _as_int(manifest.get("n_train"))
        kernel_family = str(manifest.get("kernel_family", "")).strip()
        nu = _as_float(manifest.get("kernel_nu"))
        reg_lambda = _as_float(manifest.get("reg_lambda"))
        lengthscale = _as_float(manifest.get("kernel_lengthscale"))
        fourier_eps = _as_float(manifest.get("fourier_eps"))
        observed_target = {
            "dataset_stem": dataset_stem,
            "n_train": n_train,
            "subset_seed": _as_int(manifest.get("subset_seed")),
            "subset_mode": str(manifest.get("subset_mode", "")).strip(),
            "kernel_family": kernel_family,
            "lengthscale": lengthscale,
            "nu": nu,
            "variance": _as_float(manifest.get("kernel_variance")),
            "reg_lambda": reg_lambda,
            "fourier_eps": fourier_eps,
            "nufft_tol": _as_float(manifest.get("nufft_tol")),
            "l2_scaled": manifest.get("l2_scaled"),
            "precision": str(manifest.get("precision_mode", "")).strip(),
            "nufft_backend": str(manifest.get("nufft_backend_requested", "")).strip(),
            "precompute_chunk_size": _as_int(manifest.get("precompute_chunk_size")),
        }
        target_mismatches: list[str] = []
        for field in TARGET_REGIME_FIELDS:
            observed = observed_target[field]
            expected = selected_target[field]
            matches = _system_field_matches(field, observed, expected)
            if not matches:
                target_mismatches.append(
                    f"{field}: observed={observed!r}, selected={expected!r}"
                )
        if target_mismatches:
            raise ReportSchemaError(
                f"{manifest_path}: Stage 2 does not match frozen target: "
                + "; ".join(target_mismatches)
            )
        selection_timing_protocol = str(
            manifest.get("selection_timing_protocol", "")
        ).strip()
        measured_score_selection = _as_float(manifest.get("score_selection_seconds"))
        selection_medians_by_method = manifest.get("selection_seconds_median_by_method")
        frozen_protocol_selection = _as_float(
            manifest.get("score_protocol_freeze_selection_seconds")
        )
        if (
            not isinstance(selection_medians_by_method, dict)
            or not selection_timing_protocol
            or not (
                _finite_nonnegative(measured_score_selection)
                and _finite_nonnegative(frozen_protocol_selection)
            )
        ):
            raise ReportSchemaError(
                f"{manifest_path}: measured selection timing protocol metadata is required"
            )
        expected_selection_methods = set(labels).intersection(STAGE2_OURS_METHODS)
        if set(selection_medians_by_method) != expected_selection_methods:
            raise ReportSchemaError(
                f"{manifest_path}: per-method selection medians must match the executed "
                f"active methods; expected={sorted(expected_selection_methods)}"
            )
        shared_setup = _as_float(manifest.get("setup_seconds"))
        if not _finite_nonnegative(shared_setup):
            raise ReportSchemaError(
                f"{manifest_path}: setup_seconds must be finite and nonnegative"
            )
        case = _case_id(path, (system_id, dataset, n_train, kernel_family, reg_lambda))
        setup_rows.append(
            {
                "case_id": case,
                "source_file": str(path),
                "dataset": dataset,
                "dataset_stem": dataset_stem,
                "n_train": n_train,
                "subset_seed": observed_target["subset_seed"],
                "subset_mode": observed_target["subset_mode"],
                "kernel_family": kernel_family,
                "nu": nu,
                "variance": observed_target["variance"],
                "reg_lambda": reg_lambda,
                "lengthscale": lengthscale,
                "fourier_eps": fourier_eps,
                "nufft_tol": observed_target["nufft_tol"],
                "l2_scaled": observed_target["l2_scaled"],
                "precision": observed_target["precision"],
                "nufft_backend": observed_target["nufft_backend"],
                "precompute_chunk_size": observed_target["precompute_chunk_size"],
                "system_id": system_id,
                "fixed_system_verified": fixed_verified,
                "target_regime_match": True,
                "method_matrix_complete": True,
                "selection_timing_protocol": selection_timing_protocol,
                "score_selection_seconds": measured_score_selection,
                "score_protocol_freeze_selection_seconds": frozen_protocol_selection,
                "shared_ab_setup_seconds": shared_setup,
                "timing_scope": "shared Fourier A,b setup; excluded from Stage 2 headline solver total",
            }
        )
        case_row_start = len(normalized)
        case_totals_by_method: dict[str, dict[int | None, float]] = {}
        case_eligible_by_method: dict[str, bool] = {}
        case_formal_by_method: dict[str, bool] = {}
        for row in rows:
            method = str(row["method"])
            method_kind = str(row.get("method_kind") or method)
            result_role = str(row.get("result_role", ""))
            reporting_class = _stage2_method_class(method, method_kind, result_role)
            definition = str(row.get("solver_total_definition", "")).strip()
            if definition != EXPECTED_SOLVER_TOTAL_DEFINITION:
                raise ReportSchemaError(
                    f"{path}: {method} has invalid solver_total_definition={definition!r}"
                )
            method_runs = runs_by_method.get(method, [])
            method_warmups = [
                run
                for run in run_rows
                if _as_bool(run.get("is_warmup"))
                and str(run.get("method", "")).strip() == method
            ]
            expected_repeats = _as_int(row.get("measured_repeats"))
            repeat_ids = [_as_int(run.get("repeat_idx")) for run in method_runs]
            warmup_ids = [_as_int(run.get("repeat_idx")) for run in method_warmups]
            if (
                expected_repeats is None
                or expected_repeats != formal_measured_repeats
                or len(method_runs) != expected_repeats
                or None in repeat_ids
                or len(repeat_ids) != len(set(repeat_ids))
                or set(repeat_ids) != set(range(formal_measured_repeats))
                or len(method_warmups) != formal_warmup_repeats
                or None in warmup_ids
                # benchmark.py numbers warmups as -W, ..., -1 and measured
                # repeats as 0, ..., R-1.  Validate that production convention
                # exactly instead of accepting a synthetic zero-based warmup.
                or set(warmup_ids) != set(range(-formal_warmup_repeats, 0))
            ):
                raise ReportSchemaError(
                    f"{runs_path}: {method} warmup/measured repeat coverage does not "
                    "match the benchmark protocol"
                )
            cg_runs = runs_by_method.get("cg", [])
            cg_repeat_ids = {_as_int(run.get("repeat_idx")) for run in cg_runs}

            def converged(run: Mapping[str, Any]) -> bool:
                tol = _as_float(run.get("tol"))
                residual = _as_float(run.get("true_relres"))
                return bool(
                    str(run.get("status", "")).lower().startswith("converged")
                    and _finite_positive(tol)
                    and _finite_nonnegative(residual)
                    and residual <= tol
                )

            recomputed_performance_eligible = bool(
                method_runs
                and len(cg_runs) == expected_repeats
                and set(repeat_ids) == cg_repeat_ids
                and all(converged(run) for run in method_runs)
                and all(converged(run) for run in cg_runs)
            )
            if _as_bool(row.get("performance_claim_eligible")) != (
                recomputed_performance_eligible
            ):
                raise ReportSchemaError(
                    f"{path}: {method} performance eligibility disagrees with "
                    "repeat-level status/true-residual evidence"
                )
            total_values = np.asarray(
                [_as_float(run["solver_total_seconds"]) for run in method_runs],
                dtype=float,
            )
            cg_total_by_repeat = {
                _as_int(run.get("repeat_idx")): _as_float(
                    run.get("solver_total_seconds")
                )
                for run in cg_runs
            }
            method_total_by_repeat = {
                _as_int(run.get("repeat_idx")): _as_float(
                    run.get("solver_total_seconds")
                )
                for run in method_runs
            }
            case_totals_by_method[method] = method_total_by_repeat
            paired_speedups = np.asarray(
                (
                    [
                        cg_total_by_repeat[repeat_idx]
                        / method_total_by_repeat[repeat_idx]
                        for repeat_idx in range(formal_measured_repeats)
                    ]
                    if recomputed_performance_eligible
                    else []
                ),
                dtype=float,
            )
            if recomputed_performance_eligible and (
                paired_speedups.size != formal_measured_repeats
                or not np.all(np.isfinite(paired_speedups))
                or not np.all(paired_speedups > 0.0)
            ):
                raise ReportSchemaError(
                    f"{runs_path}: {method} lacks a finite positive paired CG/"
                    "method total ratio for every measured repeat"
                )
            selection_values = np.asarray(
                [_as_float(run["selection_seconds"]) for run in method_runs],
                dtype=float,
            )
            build_values = np.asarray(
                [_as_float(run["preconditioner_build_seconds"]) for run in method_runs],
                dtype=float,
            )
            solve_values = np.asarray(
                [_as_float(run["solve_seconds"]) for run in method_runs],
                dtype=float,
            )
            solver_total = float(np.median(total_values))
            summary_total = _as_float(row.get("solver_total_seconds_median"))
            if not _numbers_match(solver_total, summary_total, rel_tol=1e-9):
                raise ReportSchemaError(
                    f"{path}: {method} canonical summary total does not match measured-run median"
                )
            summary_component_pairs = (
                ("selection_seconds_median", float(np.median(selection_values))),
                (
                    "preconditioner_build_seconds_median",
                    float(np.median(build_values)),
                ),
                ("solve_seconds_median", float(np.median(solve_values))),
            )
            for summary_key, recomputed in summary_component_pairs:
                if not _numbers_match(row.get(summary_key), recomputed, rel_tol=1e-9):
                    raise ReportSchemaError(
                        f"{path}: {method} {summary_key} does not match measured runs"
                    )
            order = np.argsort(total_values, kind="stable")
            middle = len(order) // 2
            representative_indices = (
                [int(order[middle])]
                if len(order) % 2
                else [int(order[middle - 1]), int(order[middle])]
            )
            selection = float(np.mean(selection_values[representative_indices]))
            preconditioner_build = float(np.mean(build_values[representative_indices]))
            solve = float(np.mean(solve_values[representative_indices]))
            component_sum = selection + preconditioner_build + solve
            if not _numbers_match(component_sum, solver_total, rel_tol=1e-9):
                raise ReportSchemaError(
                    f"{runs_path}: {method} median-total representative components do not add"
                )
            selection_median = float(np.median(selection_values))
            if method in STAGE2_OURS_METHODS:
                declared_method_selection = _as_float(
                    selection_medians_by_method.get(method)
                )
                if not np.all(selection_values > 0.0) or not _numbers_match(
                    selection_median, declared_method_selection, rel_tol=1e-9
                ):
                    raise ReportSchemaError(
                        f"{path}: {method} must include its own measured per-repeat "
                        "score-selection component"
                    )
            elif method in {
                "cg",
                "jacobi",
                "full-eig",
                "full-inverse",
                *FOURIER_ADAPTATION_ALIASES,
            } and not np.allclose(selection_values, 0.0, rtol=0.0, atol=1e-12):
                raise ReportSchemaError(
                    f"{path}: non-active method {method} must have zero selection time"
                )
            relative_error = abs(component_sum - solver_total) / solver_total
            formal = reporting_class not in {
                "exploratory_fourier_preconditioner",
                "diagnostic_only",
            }
            case_eligible_by_method[method] = recomputed_performance_eligible
            case_formal_by_method[method] = formal
            normalized.append(
                {
                    "case_id": case,
                    "source_file": str(path),
                    "dataset": dataset,
                    "dataset_stem": dataset_stem,
                    "n_train": n_train,
                    "subset_seed": observed_target["subset_seed"],
                    "subset_mode": observed_target["subset_mode"],
                    "kernel_family": kernel_family,
                    "nu": nu,
                    "variance": observed_target["variance"],
                    "reg_lambda": reg_lambda,
                    "lengthscale": lengthscale,
                    "fourier_eps": fourier_eps,
                    "nufft_tol": observed_target["nufft_tol"],
                    "l2_scaled": observed_target["l2_scaled"],
                    "precision": observed_target["precision"],
                    "nufft_backend": observed_target["nufft_backend"],
                    "precompute_chunk_size": observed_target["precompute_chunk_size"],
                    "method": method,
                    "method_kind": method_kind,
                    "result_role": result_role,
                    "reporting_class": reporting_class,
                    "measured_repeats": formal_measured_repeats,
                    "formal_included": formal,
                    "performance_claim_eligible": recomputed_performance_eligible,
                    "fixed_system_verified": fixed_verified,
                    "target_regime_match": True,
                    "method_matrix_complete": True,
                    "selection_timing_protocol": selection_timing_protocol,
                    "score_selection_seconds": measured_score_selection,
                    "score_protocol_freeze_selection_seconds": frozen_protocol_selection,
                    "system_id": system_id,
                    "selection_seconds": selection,
                    "preconditioner_build_seconds": preconditioner_build,
                    "solve_seconds": solve,
                    "solver_total_seconds": solver_total,
                    "solver_total_speedup_over_cg_median": (
                        float(np.median(paired_speedups))
                        if paired_speedups.size
                        else math.nan
                    ),
                    "solver_total_speedup_over_cg_min": (
                        float(np.min(paired_speedups))
                        if paired_speedups.size
                        else math.nan
                    ),
                    "solver_total_speedup_over_cg_max": (
                        float(np.max(paired_speedups))
                        if paired_speedups.size
                        else math.nan
                    ),
                    "paired_comparisons": int(paired_speedups.size),
                    "paired_wins_over_cg": int(np.sum(paired_speedups > 1.0)),
                    "solver_total_speedup_source": (
                        "median of matched-repeat CG_i / method_i solver totals"
                    ),
                    "solver_total_source": (
                        "median of measured matched_runs totals; validated against summary; "
                        "components are paired median-total representatives"
                    ),
                    "solver_total_definition": definition,
                    "component_sum_seconds": component_sum,
                    "component_sum_relative_error": relative_error,
                    "corrected_total_verified": True,
                    "corrected_total_eligible": bool(
                        fixed_verified and recomputed_performance_eligible and formal
                    ),
                    "shared_ab_setup_seconds": shared_setup,
                }
            )
        eligible_baseline_methods = [
            method
            for method in labels
            if method not in STAGE2_OURS_METHODS
            and case_formal_by_method.get(method, False)
            and case_eligible_by_method.get(method, False)
        ]
        for normalized_row in normalized[case_row_start:]:
            method = str(normalized_row["method"])
            best_baseline_speedups = np.asarray([], dtype=float)
            if (
                method in STAGE2_OURS_METHODS
                and case_eligible_by_method.get(method, False)
                and eligible_baseline_methods
            ):
                method_totals = case_totals_by_method[method]
                best_baseline_speedups = np.asarray(
                    [
                        min(
                            case_totals_by_method[baseline][repeat_idx]
                            for baseline in eligible_baseline_methods
                        )
                        / method_totals[repeat_idx]
                        for repeat_idx in range(formal_measured_repeats)
                    ],
                    dtype=float,
                )
            if best_baseline_speedups.size and (
                best_baseline_speedups.size != formal_measured_repeats
                or not np.all(np.isfinite(best_baseline_speedups))
                or not np.all(best_baseline_speedups > 0.0)
            ):
                raise ReportSchemaError(
                    f"{runs_path}: {method} lacks a finite positive matched-repeat "
                    "speedup against the per-repeat best formal baseline"
                )
            normalized_row.update(
                {
                    "solver_total_speedup_vs_best_baseline_median": (
                        float(np.median(best_baseline_speedups))
                        if best_baseline_speedups.size
                        else math.nan
                    ),
                    "solver_total_speedup_vs_best_baseline_min": (
                        float(np.min(best_baseline_speedups))
                        if best_baseline_speedups.size
                        else math.nan
                    ),
                    "solver_total_speedup_vs_best_baseline_max": (
                        float(np.max(best_baseline_speedups))
                        if best_baseline_speedups.size
                        else math.nan
                    ),
                    "best_baseline_paired_comparisons": int(
                        best_baseline_speedups.size
                    ),
                    "best_baseline_speedup_source": (
                        "median of matched-repeat min(formal baseline_i) / method_i "
                        "solver totals"
                    ),
                }
            )
    return normalized, setup_rows


def _case_maps(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    return grouped


def _claim(
    claim_id: str,
    stage: str,
    statement: str,
    definition: str,
    *,
    evaluated: int,
    supporting: int,
    contradicting: int,
    missing: int,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if evaluated == 0:
        status = "not_evaluable"
    elif contradicting > 0:
        status = "not_supported"
    elif missing > 0:
        status = "not_evaluable"
    elif supporting == evaluated:
        status = "supported"
    else:
        status = "not_supported"
    return {
        "claim_id": claim_id,
        "stage": stage,
        "status": status,
        "statement": statement,
        "definition": definition,
        "evaluated_cases": int(evaluated),
        "supporting_cases": int(supporting),
        "contradicting_cases": int(contradicting),
        "missing_cases": int(missing),
        "details": dict(details or {}),
    }


def build_stage1_robustness(
    rows: Sequence[Mapping[str, Any]],
    selected_target: Mapping[str, Any],
    suite: Mapping[str, Any],
    ours_method: str = STAGE1_OURS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the prospectively declared OAT design, then summarize evidence."""
    robustness_rows = [
        row
        for row in rows
        if str(row.get("suite_profile", "")) == "robustness_at_selected_target"
    ]
    cases = _case_maps(robustness_rows)
    case_evidence: list[dict[str, Any]] = []
    for case_id, case_rows in cases.items():
        ours = next((row for row in case_rows if row["method"] == ours_method), None)
        comparators = [
            row
            for row in case_rows
            if row["method"] != ours_method
            and bool(row.get("formal_method"))
            and bool(row.get("speedup_claim_eligible"))
        ]
        ours_ok = bool(
            ours
            and ours.get("speedup_claim_eligible")
            and _finite_positive(ours.get("train_total_seconds"))
        )
        best_comparator = min(
            (_as_float(row["train_total_seconds"]) for row in comparators),
            default=math.nan,
        )
        ours_total = _as_float(ours.get("train_total_seconds")) if ours else math.nan
        comparable = bool(ours_ok and _finite_positive(best_comparator))
        speedup = best_comparator / ours_total if comparable else math.nan
        template = ours or case_rows[0]
        case_evidence.append(
            {
                "case_id": case_id,
                "robustness_axes": tuple(template.get("robustness_axes", ())),
                "dataset": template.get("dataset"),
                "dataset_stem": template.get("dataset_stem"),
                "n_train": template.get("n_train"),
                "subset_seed": template.get("subset_seed"),
                "subset_mode": template.get("subset_mode"),
                "kernel_family": template.get("kernel_family"),
                "nu": template.get("nu"),
                "variance": template.get("variance"),
                "reg_lambda": template.get("reg_lambda"),
                "lengthscale": template.get("lengthscale"),
                "fourier_eps": template.get("fourier_eps"),
                "nufft_tol": template.get("nufft_tol"),
                "l2_scaled": template.get("l2_scaled"),
                "precision": template.get("precision"),
                "nufft_backend": template.get("nufft_backend"),
                "precompute_chunk_size": template.get("precompute_chunk_size"),
                "box_budget": template.get("box_budget"),
                "accuracy_max_rmse": template.get("accuracy_max_rmse"),
                "accuracy_min_r2": template.get("accuracy_min_r2"),
                "ours_accuracy_and_timing_eligible": ours_ok,
                "eligible_comparator_count": len(comparators),
                "speedup_vs_best_eligible_comparator": speedup,
                "ours_faster": bool(comparable and speedup > 1.0),
                "comparable": comparable,
            }
        )
    robust_spec = suite["profiles"]["robustness_at_selected_target"]
    reference = {
        **{field: selected_target[field] for field in TARGET_REGIME_FIELDS},
        "box_budget": _as_int(suite["base"]["box_budget"]),
        "accuracy_max_rmse": _as_float(selected_target.get("accuracy_max_rmse")),
        "accuracy_min_r2": _as_float(selected_target.get("accuracy_min_r2")),
    }
    axes = {
        "lambda": {
            "field": "reg_lambda",
            "prefix": "lambda_",
            "values": list(robust_spec["lambda_values"]),
        },
        "lengthscale": {
            "field": "lengthscale",
            "prefix": "lengthscale_",
            "values": list(robust_spec["lengthscale_values"]),
        },
        "box_budget": {
            "field": "box_budget",
            "prefix": "box_budget_",
            "values": list(robust_spec["box_budget_values"]),
        },
        "dataset": {
            "field": "dataset_stem",
            "prefix": "dataset_",
            "values": list(robust_spec["datasets"]),
        },
    }

    def declared_dataset_stem(declared: Mapping[str, Any]) -> str:
        direct = str(declared.get("dataset_stem", "")).strip()
        if direct:
            return direct
        stems = declared.get("dataset_stems_by_n_train", {})
        return str(stems.get(str(reference["n_train"]), "")).strip()

    def value_matches(observed: Any, declared: Any, dimension: str) -> bool:
        if dimension == "dataset":
            return str(observed) == declared_dataset_stem(declared)
        if dimension == "box_budget":
            return _as_int(observed) == _as_int(declared)
        return _numbers_match(observed, declared)

    fixed_fields = tuple(reference)
    table: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    design_support = design_bad = 0
    for dimension, spec in axes.items():
        field = str(spec["field"])
        axis_prefix = str(spec["prefix"])
        declared_values = list(spec["values"])
        dimension_evidence = [
            row
            for row in case_evidence
            if any(
                str(label).startswith(axis_prefix)
                for label in row.get("robustness_axes", ())
            )
        ]
        dimension_rows: list[dict[str, Any]] = []
        matched_case_ids: set[str] = set()
        for declared in declared_values:
            subset = [
                row
                for row in dimension_evidence
                if value_matches(row.get(field), declared, dimension)
            ]
            design_errors: list[str] = []
            if len(subset) != 1:
                design_errors.append(
                    f"expected exactly one case, observed {len(subset)}"
                )
            candidate = subset[0] if len(subset) == 1 else None
            if candidate is not None:
                matched_case_ids.add(str(candidate["case_id"]))
                for fixed_field in fixed_fields:
                    if fixed_field == field or (
                        dimension == "dataset"
                        and fixed_field in {"accuracy_max_rmse", "accuracy_min_r2"}
                    ):
                        continue
                    expected = reference[fixed_field]
                    observed = candidate.get(fixed_field)
                    if fixed_field in TARGET_REGIME_FIELDS:
                        matches = _system_field_matches(fixed_field, observed, expected)
                    elif fixed_field == "box_budget":
                        matches = _as_int(observed) == _as_int(expected)
                    elif fixed_field in {
                        "accuracy_max_rmse",
                        "accuracy_min_r2",
                    }:
                        matches = _optional_numbers_match(observed, expected)
                    else:
                        matches = _numbers_match(observed, expected)
                    if not matches:
                        design_errors.append(
                            f"non-axis {fixed_field} changed: {observed!r} != {expected!r}"
                        )
                if dimension == "dataset" and str(candidate.get("dataset")) != str(
                    declared["dataset_family"]
                ):
                    design_errors.append(
                        "declared dataset_family does not match the suite"
                    )
                if dimension == "dataset":
                    for threshold in ("accuracy_max_rmse", "accuracy_min_r2"):
                        expected_threshold = declared.get(
                            threshold, reference[threshold]
                        )
                        if not _optional_numbers_match(
                            candidate.get(threshold), expected_threshold
                        ):
                            design_errors.append(
                                f"{threshold} does not match the declared dataset gate"
                            )
            design_eligible = not design_errors
            comparable = bool(candidate and candidate["comparable"] and design_eligible)
            speedup = (
                _as_float(candidate["speedup_vs_best_eligible_comparator"])
                if comparable
                else math.nan
            )
            display_value = (
                str(declared["dataset_family"])
                if dimension == "dataset"
                else str(declared)
            )
            record = {
                "dimension": dimension,
                "value": display_value,
                "declared_value": (
                    declared_dataset_stem(declared)
                    if dimension == "dataset"
                    else declared
                ),
                "expected_case_count": 1,
                "observed_case_count": len(subset),
                "case_id": candidate.get("case_id") if candidate else "",
                "design_eligible": design_eligible,
                "design_errors": "; ".join(design_errors),
                "ours_eligible_cases": int(
                    bool(candidate and candidate["ours_accuracy_and_timing_eligible"])
                ),
                "comparable_cases": int(comparable),
                "ours_win_cases": int(
                    comparable and bool(candidate and candidate["ours_faster"])
                ),
                "speedup_vs_best_min": speedup,
                "speedup_vs_best_median": speedup,
                "speedup_vs_best_max": speedup,
            }
            dimension_rows.append(record)
            table.append(record)
        extras = [
            row
            for row in dimension_evidence
            if str(row["case_id"]) not in matched_case_ids
        ]
        if extras:
            dimension_rows.append(
                {
                    "dimension": dimension,
                    "value": "<undeclared-extra>",
                    "declared_value": "",
                    "expected_case_count": 0,
                    "observed_case_count": len(extras),
                    "case_id": ",".join(str(row["case_id"]) for row in extras),
                    "design_eligible": False,
                    "design_errors": "case value is not in the predeclared OAT values",
                    "ours_eligible_cases": 0,
                    "comparable_cases": 0,
                    "ours_win_cases": 0,
                    "speedup_vs_best_min": math.nan,
                    "speedup_vs_best_median": math.nan,
                    "speedup_vs_best_max": math.nan,
                }
            )
            table.append(dimension_rows[-1])
        axis_design_ok = bool(
            len(dimension_rows) == len(declared_values)
            and all(bool(row["design_eligible"]) for row in dimension_rows)
        )
        design_support += int(axis_design_ok)
        design_bad += int(not axis_design_ok)
        evaluated = sum(int(row["comparable_cases"]) for row in dimension_rows)
        supporting = sum(int(row["ours_win_cases"]) for row in dimension_rows)
        missing = len(declared_values) - evaluated
        if not axis_design_ok:
            missing = max(missing, 1)
        contradicting = evaluated - supporting
        claims.append(
            _claim(
                f"stage1_robust_across_{dimension}",
                "stage1_end_to_end_krr",
                f"The proposed full KRR pipeline remains accurate and faster across {dimension}.",
                "Every prospectively declared OAT value appears exactly once; all non-axis "
                "target fields remain frozen; every case has an accuracy/timing-eligible "
                "proposed row and comparator; proposed algorithmic training total is lower "
                "than the best eligible formal comparator.",
                evaluated=evaluated,
                supporting=supporting,
                contradicting=contradicting,
                missing=missing,
                details={
                    "declared_values": [
                        (
                            value.get("dataset_stem")
                            or value.get("dataset_stems_by_n_train", {}).get(
                                str(reference["n_train"])
                            )
                            if isinstance(value, dict)
                            else value
                        )
                        for value in declared_values
                    ],
                    "design_complete": axis_design_ok,
                },
            )
        )
    claims.insert(
        0,
        _claim(
            "stage1_robustness_oat_design_complete",
            "stage1_end_to_end_krr",
            "The robustness campaign is a complete predeclared OAT study at the frozen target.",
            "All four axes exactly match their suite-declared values and change only the "
            "named axis while N, kernel settings, and other target fields stay fixed.",
            evaluated=4,
            supporting=design_support,
            contradicting=design_bad,
            missing=0,
        ),
    )
    return table, claims


def audit_claims(
    stage1_rows: Sequence[Mapping[str, Any]],
    stage2_rows: Sequence[Mapping[str, Any]],
    *,
    selected_target: Mapping[str, Any],
    suite: Mapping[str, Any],
    feasibility: Mapping[str, Mapping[str, Any]],
    stage1_ours_method: str = STAGE1_OURS,
    stage2_primary_ours_method: str = STAGE2_PRIMARY_OURS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims: list[dict[str, Any]] = []
    scale_rows = [
        row for row in stage1_rows if str(row.get("suite_profile")) == "scale_10m_300m"
    ]
    stage1_cases = _case_maps(scale_rows)

    accuracy_eval = accuracy_support = accuracy_bad = accuracy_missing = 0
    faster_eval = faster_support = faster_bad = faster_missing = 0
    covered_n: set[int] = set()
    matrix_eval = matrix_support = matrix_bad = 0
    for case_rows in stage1_cases.values():
        present_methods = {str(row["method"]) for row in case_rows}
        matrix_eval += 1
        matrix_ok = bool(
            STAGE1_FORMAL_METHODS.issubset(present_methods)
            and all(
                str(row.get("status", "")).lower() == "ok"
                for row in case_rows
                if row["method"] in STAGE1_FORMAL_METHODS
            )
        )
        if matrix_ok:
            matrix_support += 1
        else:
            matrix_bad += 1
        ours = next(
            (row for row in case_rows if row["method"] == stage1_ours_method), None
        )
        if ours is None:
            accuracy_missing += 1
            faster_missing += 1
            continue
        accuracy_eval += 1
        if bool(ours.get("accuracy_eligible")) and bool(
            ours.get("performance_claim_eligible")
        ):
            accuracy_support += 1
            n_train = ours.get("n_train")
            if isinstance(n_train, int):
                covered_n.add(n_train)
        else:
            accuracy_bad += 1
        eligible_comparators = [
            row
            for row in case_rows
            if row["method"] != stage1_ours_method
            and bool(row.get("formal_method"))
            and bool(row.get("speedup_claim_eligible"))
            and _finite_positive(row.get("train_total_seconds"))
        ]
        if not bool(ours.get("speedup_claim_eligible")) or not eligible_comparators:
            faster_missing += 1
            continue
        faster_eval += 1
        ours_total = _as_float(ours["train_total_seconds"])
        best = min(
            _as_float(row["train_total_seconds"]) for row in eligible_comparators
        )
        if ours_total < best:
            faster_support += 1
        else:
            faster_bad += 1

    claims.append(
        _claim(
            "stage1_full_method_matrix_complete",
            "stage1_end_to_end_krr",
            "Every Stage 1 case contains successful rows for all six declared KRR pipelines.",
            "The six formal method labels are present and status=ok; resource-limit and "
            "failed rows remain visible but do not count as a complete successful matrix.",
            evaluated=matrix_eval,
            supporting=matrix_support,
            contradicting=matrix_bad,
            missing=0,
        )
    )
    claims.append(
        _claim(
            "stage1_proposed_accuracy_preserved",
            "stage1_end_to_end_krr",
            "The proposed full KRR pipeline satisfies the frozen accuracy gate.",
            "At least one prospective absolute RMSE/R2 gate is present and every expected "
            "measured repeat is evaluated and passes; reported accuracy/performance eligibility "
            "must also hold in every scale case.",
            evaluated=accuracy_eval,
            supporting=accuracy_support,
            contradicting=accuracy_bad,
            missing=accuracy_missing,
        )
    )
    claims.append(
        _claim(
            "stage1_proposed_fastest_train_total",
            "stage1_end_to_end_krr",
            "The proposed full KRR pipeline has the lowest method-owned algorithmic training total.",
            "In every comparable case, proposed train_total_seconds is lower than the "
            "best accuracy-eligible formal KRR comparator.",
            evaluated=faster_eval,
            supporting=faster_support,
            contradicting=faster_bad,
            missing=faster_missing,
        )
    )
    required_scale = {10_000_000, 30_000_000, 100_000_000, 300_000_000}
    scale_supported = required_scale.issubset(covered_n)
    claims.append(
        _claim(
            "stage1_scale_10m_to_300m",
            "stage1_end_to_end_krr",
            "Accuracy-qualified proposed-pipeline results cover 10M through 300M training rows.",
            "Eligible proposed runs include each of 10M, 30M, 100M, and 300M; "
            "no extrapolated or pilot rows count.",
            evaluated=1 if covered_n else 0,
            supporting=1 if scale_supported else 0,
            contradicting=1 if covered_n and not scale_supported else 0,
            missing=0 if covered_n else 1,
            details={"eligible_n_train": sorted(covered_n)},
        )
    )

    robustness_table, robustness_claims = build_stage1_robustness(
        stage1_rows, selected_target, suite, stage1_ours_method
    )
    claims.extend(robustness_claims)

    stage2_cases = _case_maps(stage2_rows)
    primary_eval = primary_support = primary_bad = primary_missing = 0
    fixed_eval = fixed_support = fixed_bad = 0
    for case_rows in stage2_cases.values():
        verified = all(
            bool(row.get("fixed_system_verified"))
            and bool(row.get("target_regime_match"))
            and bool(row.get("method_matrix_complete"))
            for row in case_rows
        )
        fixed_eval += 1
        if verified:
            fixed_support += 1
        else:
            fixed_bad += 1
        primary = next(
            (row for row in case_rows if row["method"] == stage2_primary_ours_method),
            None,
        )
        baselines = [
            row
            for row in case_rows
            if bool(row.get("formal_included"))
            and row["method"] not in STAGE2_OURS_METHODS
            and bool(row.get("performance_claim_eligible"))
            and bool(row.get("corrected_total_eligible"))
            and _finite_positive(row.get("solver_total_seconds"))
        ]
        primary_ok = bool(
            primary
            and primary.get("formal_included")
            and primary.get("performance_claim_eligible")
            and primary.get("fixed_system_verified")
            and primary.get("corrected_total_eligible")
            and _finite_positive(primary.get("solver_total_seconds"))
            and _finite_positive(
                primary.get("solver_total_speedup_vs_best_baseline_median")
            )
            and _as_int(primary.get("best_baseline_paired_comparisons"))
            == _as_int(primary.get("measured_repeats"))
        )
        if not primary_ok or not baselines:
            primary_missing += 1
            continue
        primary_eval += 1
        paired_speedup = _as_float(
            primary["solver_total_speedup_vs_best_baseline_median"]
        )
        if paired_speedup > 1.0:
            primary_support += 1
        else:
            primary_bad += 1

    claims.append(
        _claim(
            "stage2_same_A_b_verified",
            "stage2_fixed_A_b",
            "Every Stage 2 comparison uses one saved, unchanged A,b within its case.",
            "Initial/final/run/embedded-artifact system ids agree, the timing-system "
            "artifact passes SHA-256 verification, and all fifteen system-defining "
            "configuration fields match selected_target_regime.json.",
            evaluated=fixed_eval,
            supporting=fixed_support,
            contradicting=fixed_bad,
            missing=0,
        )
    )
    claims.append(
        _claim(
            "stage2_formal_method_matrix_complete",
            "stage2_fixed_A_b",
            "Every Stage 2 case contains the complete prospectively feasible formal method matrix.",
            "CG, Jacobi, default, active inverse, active eig, and full eig must appear unless "
            "the supplied feasibility matrix prospectively marks a method infeasible with a reason; "
            "unknown spellings are rejected.",
            evaluated=len(stage2_cases),
            supporting=sum(
                all(bool(row.get("method_matrix_complete")) for row in case_rows)
                for case_rows in stage2_cases.values()
            ),
            contradicting=0,
            missing=0,
            details={"feasibility": feasibility},
        )
    )
    claims.append(
        _claim(
            "stage2_corrected_totals_verified",
            "stage2_fixed_A_b",
            "Every formal Stage 2 headline total is a validated corrected solver total.",
            "solver_total_definition is frozen and selection, preconditioner build, and solve "
            "are finite/nonnegative and sum numerically to solver_total_seconds_median.",
            evaluated=sum(bool(row.get("formal_included")) for row in stage2_rows),
            supporting=sum(
                bool(row.get("formal_included"))
                and bool(row.get("corrected_total_verified"))
                for row in stage2_rows
            ),
            contradicting=sum(
                bool(row.get("formal_included"))
                and not bool(row.get("corrected_total_verified"))
                for row in stage2_rows
            ),
            missing=0,
        )
    )
    claims.append(
        _claim(
            "stage2_primary_ours_beats_best_baseline_total",
            "stage2_fixed_A_b",
            "The primary proposed solver beats the best formal baseline in total solver time.",
            "In every verified fixed-A,b case, the median matched-repeat ratio "
            "min(formal baseline total_i) / primary total_i is greater than one. "
            "Each total includes selection, preconditioner build, and solve; shared A,b "
            "setup is excluded and reported separately.",
            evaluated=primary_eval,
            supporting=primary_support,
            contradicting=primary_bad,
            missing=primary_missing,
            details={"primary_method": stage2_primary_ours_method},
        )
    )
    return claims, robustness_table


def _claim_csv_rows(claims: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for claim in claims:
        row = dict(claim)
        row["details"] = json.dumps(
            _json_safe(row.get("details", {})), ensure_ascii=False
        )
        output.append(row)
    return output


def _validate_stage1_scale_design(
    rows: Sequence[Mapping[str, Any]],
    selected_target: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> None:
    scale_rows = [row for row in rows if row.get("suite_profile") == "scale_10m_300m"]
    if not scale_rows:
        raise ReportSchemaError("Stage 1 contains no scale_10m_300m rows")
    duplicate_keys = [
        key
        for key in {(str(row["case_id"]), str(row["method"])) for row in scale_rows}
        if sum((str(row["case_id"]), str(row["method"])) == key for row in scale_rows)
        > 1
    ]
    if duplicate_keys:
        raise ReportSchemaError(
            f"duplicate Stage 1 scale case/method rows across inputs: {duplicate_keys}"
        )
    declared_cases = {
        str(case["id"]): case
        for case in suite["profiles"]["scale_10m_300m"].get("cases", [])
    }
    if not declared_cases:
        raise ReportSchemaError("Stage 1 suite scale profile has no declared cases")
    case_groups = _case_maps(scale_rows)
    supplied_declared_ids = [
        str(case_rows[0].get("declared_case_id", ""))
        for case_rows in case_groups.values()
    ]
    duplicated_declared_ids = sorted(
        {
            declared_id
            for declared_id in supplied_declared_ids
            if supplied_declared_ids.count(declared_id) > 1
        }
    )
    missing_cases = sorted(set(declared_cases).difference(supplied_declared_ids))
    extra_cases = sorted(set(supplied_declared_ids).difference(declared_cases))
    if duplicated_declared_ids or missing_cases or extra_cases:
        raise ReportSchemaError(
            "Stage 1 scale coverage must exactly match the suite declaration; "
            f"missing={missing_cases}, extra={extra_cases}, "
            f"duplicated={duplicated_declared_ids}"
        )
    base = suite["base"]
    target_found = False
    for case_rows in case_groups.values():
        template = case_rows[0]
        declared_id = str(template.get("declared_case_id", ""))
        if declared_id not in declared_cases:
            raise ReportSchemaError(
                f"Stage 1 scale case {declared_id!r} is not declared in the suite"
            )
        expected = dict(base)
        expected.update(declared_cases[declared_id])
        comparisons = {
            **{
                field: _system_field_matches(
                    field, template.get(field), expected.get(field)
                )
                for field in TARGET_REGIME_FIELDS
            },
            "box_budget": _as_int(template.get("box_budget"))
            == _as_int(expected.get("box_budget")),
            "accuracy_max_rmse": _optional_numbers_match(
                template.get("accuracy_max_rmse"),
                expected.get("accuracy_max_rmse"),
            ),
            "accuracy_min_r2": _optional_numbers_match(
                template.get("accuracy_min_r2"),
                expected.get("accuracy_min_r2"),
            ),
        }
        failures = [field for field, matches in comparisons.items() if not matches]
        if failures:
            raise ReportSchemaError(
                f"Stage 1 scale case {declared_id!r} differs from its suite declaration: "
                f"{failures}"
            )
        present_methods = {str(row.get("method", "")) for row in case_rows}
        if present_methods != STAGE1_FORMAL_METHODS:
            raise ReportSchemaError(
                f"Stage 1 scale case {declared_id!r} must contain exactly the six "
                f"formal KRR methods; missing="
                f"{sorted(STAGE1_FORMAL_METHODS.difference(present_methods))}, "
                f"extra={sorted(present_methods.difference(STAGE1_FORMAL_METHODS))}"
            )
        target_matches = True
        for field in TARGET_REGIME_FIELDS:
            observed = template.get(field)
            expected_target = selected_target[field]
            matches = _system_field_matches(field, observed, expected_target)
            target_matches = target_matches and matches
        target_found = target_found or target_matches
    if not target_found:
        raise ReportSchemaError(
            "selected_target_regime does not match any supplied scale_10m_300m case"
        )
    selection = suite.get("target_selection")
    if not isinstance(selection, dict):
        raise ReportSchemaError("Stage 1 suite requires target_selection metadata")
    from .end_to_end_suite import TargetSelectionError, select_target_regime

    try:
        recomputed_target = select_target_regime(
            scale_rows,
            cg_iteration_min=int(selection.get("cg_iteration_min", 3000)),
            cg_iteration_max=int(selection.get("cg_iteration_max", 6000)),
            dataset_priority=selection.get("dataset_priority", ()),
            allowed_resource_limit_methods=selection.get(
                "allowed_resource_limit_methods", ("rpcholesky-krr",)
            ),
        )
    except TargetSelectionError as exc:
        raise ReportSchemaError(
            "Stage 1 rows do not yield a target under the suite-declared selection rule"
        ) from exc
    mismatched_target_fields = [
        field
        for field in TARGET_REGIME_FIELDS
        if not _system_field_matches(
            field, selected_target.get(field), recomputed_target.get(field)
        )
    ]
    for field in ("accuracy_max_rmse", "accuracy_min_r2"):
        if not _optional_numbers_match(
            selected_target.get(field), recomputed_target.get(field)
        ):
            mismatched_target_fields.append(field)
    if (
        mismatched_target_fields
        or str(selected_target.get("selection_rule", "")).strip()
        != str(recomputed_target.get("selection_rule", "")).strip()
    ):
        raise ReportSchemaError(
            "selected_target_regime is not the winner recomputed from the complete "
            f"Stage 1 scale evidence; mismatched={mismatched_target_fields}"
        )


def _empty_plot(path: Path, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_plots(
    output: Path,
    stage1: Sequence[Mapping[str, Any]],
    stage2: Sequence[Mapping[str, Any]],
    robustness: Sequence[Mapping[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    artifacts: list[str] = []
    scale_stage1 = [
        row for row in stage1 if str(row.get("suite_profile")) == "scale_10m_300m"
    ]
    resource_limits = [
        row
        for row in scale_stage1
        if str(row.get("status", "")).lower() == "resource_limit"
    ]

    def resource_note(row: Mapping[str, Any]) -> str:
        required = _as_float(row.get("resource_required_bytes"))
        cap = _as_float(row.get("resource_effective_cap_bytes"))
        required_text = (
            f"{required / 2**30:.1f} GiB" if math.isfinite(required) else "unknown"
        )
        cap_text = f"{cap / 2**30:.1f} GiB" if math.isfinite(cap) else "unknown"
        return (
            f"{row.get('method')} @ N={row.get('n_train')}: "
            f"resource limit ({required_text} required, {cap_text} cap)"
        )

    accuracy_path = output / "stage1_accuracy_vs_train_total.png"
    valid = [
        row
        for row in scale_stage1
        if _finite_positive(row.get("train_total_seconds"))
        and _finite_positive(row.get("test_rmse"))
    ]
    if valid:
        fig, axis = plt.subplots(figsize=(8.5, 5.2))
        for method in sorted({str(row["method"]) for row in valid}):
            subset = [row for row in valid if row["method"] == method]
            eligible = [row for row in subset if bool(row["accuracy_eligible"])]
            ineligible = [row for row in subset if not bool(row["accuracy_eligible"])]
            if eligible:
                axis.scatter(
                    [row["train_total_seconds"] for row in eligible],
                    [row["test_rmse"] for row in eligible],
                    label=method,
                    s=42,
                )
            if ineligible:
                axis.scatter(
                    [row["train_total_seconds"] for row in ineligible],
                    [row["test_rmse"] for row in ineligible],
                    marker="x",
                    color="0.45",
                    s=48,
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Method-owned algorithmic training total (s)")
        axis.set_ylabel("Test RMSE")
        axis.set_title("Stage 1: accuracy vs method-owned algorithmic training total")
        axis.grid(True, which="both", alpha=0.25)
        if resource_limits:
            axis.scatter([], [], marker="v", color="crimson", label="resource limit")
            axis.text(
                0.01,
                0.01,
                "\n".join(resource_note(row) for row in resource_limits),
                transform=axis.transAxes,
                fontsize=6.5,
                color="crimson",
                va="bottom",
            )
        axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(accuracy_path, dpi=180)
        plt.close(fig)
    else:
        _empty_plot(
            accuracy_path,
            "Stage 1: accuracy vs training total",
            "No finite Stage 1 rows",
        )
    artifacts.append(accuracy_path.name)

    breakdown_path = output / "stage1_setup_solving_breakdown.png"
    breakdown = [
        row
        for row in scale_stage1
        if math.isfinite(_as_float(row.get("setup_seconds")))
        and math.isfinite(_as_float(row.get("solving_phase_seconds")))
    ]
    if breakdown:
        labels = [f"{row['case_id'][:8]}\n{row['method']}" for row in breakdown]
        x = np.arange(len(breakdown))
        setup = np.asarray([row["setup_seconds"] for row in breakdown], dtype=float)
        solve = np.asarray(
            [row["solving_phase_seconds"] for row in breakdown], dtype=float
        )
        fig, axis = plt.subplots(figsize=(max(9, len(labels) * 0.55), 5.2))
        axis.bar(x, setup, label="setup")
        axis.bar(x, solve, bottom=setup, label="solving phase")
        axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
        axis.set_ylabel("Seconds")
        axis.set_title("Stage 1: method-owned setup and solving breakdown")
        if resource_limits:
            axis.text(
                0.01,
                0.99,
                "\n".join(resource_note(row) for row in resource_limits),
                transform=axis.transAxes,
                fontsize=6.5,
                color="crimson",
                va="top",
            )
        axis.legend()
        fig.tight_layout()
        fig.savefig(breakdown_path, dpi=180)
        plt.close(fig)
    else:
        _empty_plot(
            breakdown_path, "Stage 1 timing breakdown", "No finite timing breakdown"
        )
    artifacts.append(breakdown_path.name)

    scale_path = output / "stage1_scale_10m_300m.png"
    scale = [
        row
        for row in scale_stage1
        if isinstance(row.get("n_train"), int)
        and 10_000_000 <= int(row["n_train"]) <= 300_000_000
        and _finite_positive(row.get("train_total_seconds"))
    ]
    if scale:
        fig, axis = plt.subplots(figsize=(8.5, 5.2))
        for method in sorted({str(row["method"]) for row in scale}):
            subset = sorted(
                (row for row in scale if row["method"] == method),
                key=lambda row: int(row["n_train"]),
            )
            axis.plot(
                [row["n_train"] for row in subset],
                [row["train_total_seconds"] for row in subset],
                marker="o",
                label=method,
            )
        if resource_limits:
            marker_height = (
                max(float(row["train_total_seconds"]) for row in scale) * 1.4
            )
            for index, row in enumerate(resource_limits):
                axis.scatter(
                    [row["n_train"]],
                    [marker_height],
                    marker="v",
                    color="crimson",
                    s=55,
                    label="resource limit" if index == 0 else None,
                )
                axis.annotate(
                    str(row["method"]),
                    (row["n_train"], marker_height),
                    xytext=(3, 5),
                    textcoords="offset points",
                    fontsize=6.5,
                    color="crimson",
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Training rows N")
        axis.set_ylabel("Method-owned algorithmic training total (s)")
        axis.set_title("Stage 1: measured scale results (10M–300M only)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(scale_path, dpi=180)
        plt.close(fig)
    else:
        _empty_plot(
            scale_path,
            "Stage 1: 10M–300M scale",
            "No measured rows in the requested range",
        )
    artifacts.append(scale_path.name)

    robustness_path = output / "stage1_robustness.png"
    robustness_valid = [
        row for row in robustness if _finite_positive(row.get("speedup_vs_best_median"))
    ]
    if robustness_valid:
        labels = [f"{row['dimension']}={row['value']}" for row in robustness_valid]
        values = [row["speedup_vs_best_median"] for row in robustness_valid]
        fig, axis = plt.subplots(figsize=(max(8, len(labels) * 0.65), 4.8))
        axis.bar(np.arange(len(labels)), values)
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.set_xticks(
            np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=7
        )
        axis.set_ylabel("Proposed speedup vs best eligible comparator")
        axis.set_title("Stage 1: robustness evidence (accuracy-gated)")
        fig.tight_layout()
        fig.savefig(robustness_path, dpi=180)
        plt.close(fig)
    else:
        _empty_plot(
            robustness_path,
            "Stage 1 robustness",
            "No accuracy-gated robustness comparison",
        )
    artifacts.append(robustness_path.name)

    stage2_path = output / "stage2_solver_total.png"
    fixed = [
        row
        for row in stage2
        if bool(row.get("formal_included"))
        and bool(row.get("performance_claim_eligible"))
        and bool(row.get("corrected_total_eligible"))
        and _finite_positive(row.get("solver_total_seconds"))
    ]
    if fixed:
        labels = [f"{row['case_id'][:8]}\n{row['method']}" for row in fixed]
        totals = [row["solver_total_seconds"] for row in fixed]
        fig, axis = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5.0))
        axis.bar(np.arange(len(labels)), totals)
        axis.set_xticks(
            np.arange(len(labels)), labels, rotation=50, ha="right", fontsize=7
        )
        axis.set_ylabel("Solver total (s)")
        axis.set_title(
            "Stage 2: fixed A,b total = selection + preconditioner build + solve"
        )
        fig.tight_layout()
        fig.savefig(stage2_path, dpi=180)
        plt.close(fig)
    else:
        _empty_plot(
            stage2_path,
            "Stage 2 fixed A,b solver total",
            "No eligible formal Stage 2 rows",
        )
    artifacts.append(stage2_path.name)
    return artifacts


def build_two_stage_report(config: TwoStageReportConfig) -> dict[str, Any]:
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_target = load_selected_target(config.selected_target_path)
    suite = load_stage1_suite(config.stage1_suite_path)
    feasibility = load_stage2_feasibility(
        config.stage2_feasibility_path,
        selected_target=selected_target,
        suite=suite,
    )
    stage1 = load_stage1_summaries(
        config.stage1_paths, ours_method=config.stage1_ours_method
    )
    duplicate_stage1 = [
        key
        for key in {(str(row["case_id"]), str(row["method"])) for row in stage1}
        if sum((str(row["case_id"]), str(row["method"])) == key for row in stage1) > 1
    ]
    if duplicate_stage1:
        raise ReportSchemaError(
            f"duplicate Stage 1 case/method rows across inputs: {duplicate_stage1}"
        )
    _validate_stage1_scale_design(stage1, selected_target, suite)
    stage2, shared_setup = load_stage2_summaries(
        config.stage2_paths,
        selected_target=selected_target,
        feasibility=feasibility,
        component_sum_relative_tolerance=config.component_sum_relative_tolerance,
        include_fourier_adaptations_in_formal=(
            config.include_fourier_adaptations_in_formal_stage2
        ),
    )
    claims, robustness = audit_claims(
        stage1,
        stage2,
        selected_target=selected_target,
        suite=suite,
        feasibility=feasibility,
        stage1_ours_method=config.stage1_ours_method,
        stage2_primary_ours_method=config.stage2_primary_ours_method,
    )

    scale = [row for row in stage1 if row.get("suite_profile") == "scale_10m_300m"]
    _write_csv(output / "stage1_accuracy_vs_total.csv", scale, STAGE1_TABLE_COLUMNS)
    _write_csv(
        output / "stage1_timing_breakdown.csv",
        scale,
        (
            "case_id",
            "dataset",
            "n_train",
            "method",
            "setup_seconds",
            "solving_phase_seconds",
            "train_total_seconds",
            "accuracy_eligible",
        ),
    )
    _write_csv(output / "stage1_scale_10m_300m.csv", scale, STAGE1_TABLE_COLUMNS)
    robustness_columns = (
        "dimension",
        "value",
        "declared_value",
        "expected_case_count",
        "observed_case_count",
        "case_id",
        "design_eligible",
        "design_errors",
        "ours_eligible_cases",
        "comparable_cases",
        "ours_win_cases",
        "speedup_vs_best_min",
        "speedup_vs_best_median",
        "speedup_vs_best_max",
    )
    _write_csv(output / "stage1_robustness.csv", robustness, robustness_columns)
    _write_csv(output / "stage2_solver_totals.csv", stage2, STAGE2_TABLE_COLUMNS)
    _write_csv(
        output / "stage2_formal_solver_totals.csv",
        [
            row
            for row in stage2
            if bool(row["formal_included"]) and bool(row["corrected_total_eligible"])
        ],
        STAGE2_TABLE_COLUMNS,
    )
    _write_csv(
        output / "stage2_shared_ab_setup.csv",
        shared_setup,
        (
            "case_id",
            "source_file",
            "dataset",
            "n_train",
            "system_id",
            "fixed_system_verified",
            "target_regime_match",
            "method_matrix_complete",
            "selection_timing_protocol",
            "score_selection_seconds",
            "score_protocol_freeze_selection_seconds",
            "shared_ab_setup_seconds",
            "timing_scope",
        ),
    )
    claim_columns = (
        "claim_id",
        "stage",
        "status",
        "statement",
        "definition",
        "evaluated_cases",
        "supporting_cases",
        "contradicting_cases",
        "missing_cases",
        "details",
    )
    _write_csv(output / "claim_audit.csv", _claim_csv_rows(claims), claim_columns)
    _write_json(output / "claim_audit.json", claims)
    _write_json(
        output / "stage1_report.json",
        {
            "protocol_family": STAGE1_PROTOCOL,
            "timing_scope": EXPECTED_STAGE1_TIMING_SCOPE,
            "accuracy_vs_total": scale,
            "scale_10m_300m": scale,
            "robustness": robustness,
        },
    )
    _write_json(
        output / "stage2_report.json",
        {
            "protocol_family": STAGE2_PROTOCOL,
            "solver_total_definition": (EXPECTED_SOLVER_TOTAL_DEFINITION),
            "solver_totals": stage2,
            "shared_ab_setup": shared_setup,
        },
    )

    plot_artifacts = (
        _make_plots(output, stage1, stage2, robustness) if config.make_plots else []
    )
    artifacts = [
        "stage1_accuracy_vs_total.csv",
        "stage1_timing_breakdown.csv",
        "stage1_scale_10m_300m.csv",
        "stage1_robustness.csv",
        "stage2_solver_totals.csv",
        "stage2_formal_solver_totals.csv",
        "stage2_shared_ab_setup.csv",
        "claim_audit.csv",
        "claim_audit.json",
        "stage1_report.json",
        "stage2_report.json",
        *plot_artifacts,
    ]
    manifest = {
        "schema_version": 2,
        "protocols": {
            "stage1": STAGE1_PROTOCOL,
            "stage2": STAGE2_PROTOCOL,
        },
        "timing_definitions": {
            "stage1_train_total": (
                "method-owned algorithmic setup + solve; common dataset I/O, "
                "backend/H2D staging, and prediction excluded"
            ),
            "stage2_solver_total": (
                "selection + preconditioner construction + CG/PCG solve; "
                "shared A,b setup is separate"
            ),
        },
        "stage1_inputs": [
            str(Path(path).expanduser().resolve()) for path in config.stage1_paths
        ],
        "stage2_inputs": [
            str(Path(path).expanduser().resolve()) for path in config.stage2_paths
        ],
        "selected_target_path": str(
            Path(config.selected_target_path).expanduser().resolve()
        ),
        "stage1_suite_path": str(Path(config.stage1_suite_path).expanduser().resolve()),
        "stage2_feasibility_path": (
            str(Path(config.stage2_feasibility_path).expanduser().resolve())
            if config.stage2_feasibility_path
            else None
        ),
        "selected_target": selected_target,
        "stage2_feasibility": feasibility,
        "stage1_rows": len(stage1),
        "stage2_rows": len(stage2),
        "formal_stage2_rows": sum(bool(row["formal_included"]) for row in stage2),
        "claim_status_counts": {
            status: sum(claim["status"] == status for claim in claims)
            for status in ("supported", "not_supported", "not_evaluable")
        },
        "artifacts": [*artifacts, "report_manifest.json"],
    }
    _write_json(output / "report_manifest.json", manifest)
    return {
        "output_dir": str(output),
        "stage1": stage1,
        "stage2": stage2,
        "shared_setup": shared_setup,
        "robustness": robustness,
        "claims": claims,
        "manifest": manifest,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build strict two-stage experiment reports."
    )
    parser.add_argument(
        "--stage1", action="append", default=[], help="pipeline_summary.csv; repeatable"
    )
    parser.add_argument(
        "--stage2", action="append", default=[], help="matched_summary.csv; repeatable"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-target", required=True)
    parser.add_argument("--stage1-suite", required=True)
    parser.add_argument("--stage2-feasibility")
    parser.add_argument("--stage1-ours-method", default=STAGE1_OURS)
    parser.add_argument("--stage2-primary-ours-method", default=STAGE2_PRIMARY_OURS)
    parser.add_argument(
        "--include-fourier-adaptations-in-formal-stage2",
        action="store_true",
        help="Opt-in only; by default Fourier Nystrom/RPCholesky adaptations are exploratory.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.stage1:
        raise SystemExit("at least one --stage1 pipeline_summary.csv is required")
    if not args.stage2:
        raise SystemExit("at least one --stage2 matched_summary.csv is required")
    result = build_two_stage_report(
        TwoStageReportConfig(
            stage1_paths=tuple(args.stage1),
            stage2_paths=tuple(args.stage2),
            output_dir=args.output_dir,
            selected_target_path=args.selected_target,
            stage1_suite_path=args.stage1_suite,
            stage2_feasibility_path=args.stage2_feasibility,
            stage1_ours_method=args.stage1_ours_method,
            stage2_primary_ours_method=args.stage2_primary_ours_method,
            include_fourier_adaptations_in_formal_stage2=(
                args.include_fourier_adaptations_in_formal_stage2
            ),
            make_plots=not args.no_plots,
        )
    )
    print(json.dumps(result["manifest"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
