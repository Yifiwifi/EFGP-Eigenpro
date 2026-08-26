from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
import zipfile
from collections import Counter
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark import (
    ControlledConfig,
    PreparedSystem,
    TIMING_SOLUTIONS_ARTIFACT_FILENAME,
    TIMING_SOLUTIONS_MANIFEST_FILENAME,
    TIMING_SYSTEM_ARTIFACT_FILENAME,
    _npz_content_index_sha256,
    _resolve_dataset_dir,
    _sanitize_json,
    _source_manifest,
    load_prepared_system_artifact,
    prepare_shared_system,
    probe_timing_runtime,
    run_controlled_experiment,
    save_prepared_system_artifact,
    system_config_fingerprint,
)


_HERE = Path(__file__).resolve().parent
_DEFAULT_SUITE_CONFIG = _HERE / "three_dataset_suite.json"
_CASE_METADATA_KEYS = {
    "id",
    "dataset_alias",
    "dataset_family",
    "expected_n_train",
    "scale_role",
    "note",
}
_ALIAS_KEYS = {
    "dataset_stem",
    "dataset_stem_glob",
    "metadata_required",
    "metadata_equals",
    "metadata_minimums",
    "n_train_consistency_paths",
}


def _normalize_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cooked = dict(payload)
    if "methods" in cooked:
        cooked["methods"] = tuple(cooked["methods"])
    if "diagnostic_topk" in cooked:
        cooked["diagnostic_topk"] = tuple(cooked["diagnostic_topk"])
    if "n_train" in cooked and int(cooked["n_train"]) == 0:
        cooked["n_train"] = None
    if "precompute_chunk_size" in cooked and int(cooked["precompute_chunk_size"]) == 0:
        cooked["precompute_chunk_size"] = None
    return cooked


def _validate_controlled_fields(payload: dict[str, Any], *, source: str) -> None:
    allowed = {field.name for field in fields(ControlledConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown ControlledConfig fields in {source}: {unknown}")


def load_suite_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("base"), dict) or not isinstance(payload.get("profiles"), dict):
        raise ValueError("suite config must contain object-valued 'base' and 'profiles'.")
    _validate_controlled_fields(payload["base"], source=f"{config_path}:base")
    aliases = payload.get("dataset_aliases", {})
    if not isinstance(aliases, dict):
        raise ValueError("suite config 'dataset_aliases' must be an object when present.")
    for alias_name, alias in aliases.items():
        if not isinstance(alias, dict):
            raise ValueError(f"dataset alias {alias_name!r} must be an object.")
        unknown = sorted(set(alias) - _ALIAS_KEYS)
        if unknown:
            raise ValueError(f"dataset alias {alias_name!r} has unknown fields {unknown}.")
        selectors = [key for key in ("dataset_stem", "dataset_stem_glob") if alias.get(key)]
        if len(selectors) != 1:
            raise ValueError(
                f"dataset alias {alias_name!r} must define exactly one of "
                "'dataset_stem' and 'dataset_stem_glob'."
            )
        selector = str(alias[selectors[0]])
        if Path(selector).name != selector:
            raise ValueError(f"dataset alias {alias_name!r} must select a filename stem only.")
        for field_name in ("metadata_equals", "metadata_minimums"):
            if not isinstance(alias.get(field_name, {}), dict):
                raise ValueError(
                    f"dataset alias {alias_name!r} field {field_name!r} must be an object."
                )
        for field_name in ("metadata_required", "n_train_consistency_paths"):
            value = alias.get(field_name, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(
                    f"dataset alias {alias_name!r} field {field_name!r} must be a string list."
                )
    for profile_name, profile in payload["profiles"].items():
        if not isinstance(profile, dict):
            raise ValueError(f"profile {profile_name!r} must be an object.")
        overrides = profile.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"profile {profile_name!r} overrides must be an object.")
        _validate_controlled_fields(overrides, source=f"{config_path}:{profile_name}:overrides")
        cases = profile.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"profile {profile_name!r} must contain nonempty cases.")
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError(f"profile {profile_name!r} contains a non-object case.")
            required = {"id", "expected_n_train"}
            missing = sorted(required - set(case))
            if missing:
                raise ValueError(f"case in {profile_name!r} is missing {missing}.")
            selectors = [key for key in ("dataset_stem", "dataset_alias") if case.get(key)]
            if len(selectors) != 1:
                raise ValueError(
                    f"case {case.get('id', '?')!r} in {profile_name!r} must define "
                    "exactly one of 'dataset_stem' and 'dataset_alias'."
                )
            if "dataset_alias" in case and str(case["dataset_alias"]) not in aliases:
                raise ValueError(
                    f"case {case.get('id', '?')!r} uses undefined dataset alias "
                    f"{case['dataset_alias']!r}."
                )
            controlled = {
                key: value
                for key, value in case.items()
                if key not in _CASE_METADATA_KEYS
            }
            _validate_controlled_fields(
                controlled,
                source=f"{config_path}:{profile_name}:{case.get('id', '?')}",
            )
    payload["config_path"] = str(config_path)
    return payload


def _sidecar_n_train(metadata: dict[str, Any]) -> int:
    shapes = metadata.get("shapes", {})
    if "n_train" not in shapes:
        raise ValueError("dataset sidecar does not contain shapes.n_train.")
    return int(shapes["n_train"])


def _metadata_value(metadata: dict[str, Any], path: str) -> Any:
    value: Any = metadata
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _resolve_case_dataset(
    case: dict[str, Any],
    *,
    dataset_dir: Path,
    aliases: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    direct_stem = case.get("dataset_stem")
    if direct_stem:
        return str(direct_stem), "", {}

    alias_name = str(case["dataset_alias"])
    if alias_name not in aliases:
        raise ValueError(f"case {case['id']!r} uses undefined dataset alias {alias_name!r}.")
    alias = aliases[alias_name]
    exact_stem = alias.get("dataset_stem")
    if exact_stem:
        return str(exact_stem), alias_name, alias

    pattern = str(alias["dataset_stem_glob"])
    candidates = sorted(
        path.stem
        for path in dataset_dir.glob(f"{pattern}.npz")
        if (dataset_dir / f"{path.stem}.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"case {case['id']!r} dataset alias {alias_name!r} found no matching "
            f"NPZ/JSON pair for {pattern!r} in {dataset_dir}."
        )
    if len(candidates) != 1:
        raise ValueError(
            f"case {case['id']!r} dataset alias {alias_name!r} is ambiguous: "
            f"matched {candidates}. Use an exact alias stem."
        )
    return candidates[0], alias_name, alias


def _validate_alias_metadata(
    *,
    alias_name: str,
    alias: dict[str, Any],
    metadata: dict[str, Any],
    source_n: int,
    json_path: Path,
) -> None:
    for path in alias.get("metadata_required", []):
        try:
            _metadata_value(metadata, str(path))
        except KeyError as exc:
            raise ValueError(
                f"dataset alias {alias_name!r} requires metadata field {path!r} "
                f"in {json_path.name}."
            ) from exc

    for path, expected in alias.get("metadata_equals", {}).items():
        try:
            actual = _metadata_value(metadata, str(path))
        except KeyError as exc:
            raise ValueError(
                f"dataset alias {alias_name!r} requires metadata field {path!r} "
                f"in {json_path.name}."
            ) from exc
        if actual != expected:
            raise ValueError(
                f"dataset alias {alias_name!r} rejects {json_path.name}: metadata "
                f"{path!r} is {actual!r}, expected {expected!r}."
            )

    for path, minimum in alias.get("metadata_minimums", {}).items():
        try:
            actual = _metadata_value(metadata, str(path))
        except KeyError as exc:
            raise ValueError(
                f"dataset alias {alias_name!r} requires metadata field {path!r} "
                f"in {json_path.name}."
            ) from exc
        if float(actual) < float(minimum):
            raise ValueError(
                f"dataset alias {alias_name!r} rejects {json_path.name}: metadata "
                f"{path!r} is {actual!r}, below {minimum!r}."
            )

    for path in alias.get("n_train_consistency_paths", []):
        try:
            actual = int(_metadata_value(metadata, str(path)))
        except KeyError as exc:
            raise ValueError(
                f"dataset alias {alias_name!r} requires unique-row count field {path!r} "
                f"in {json_path.name}."
            ) from exc
        if actual != source_n:
            raise ValueError(
                f"dataset alias {alias_name!r} rejects {json_path.name}: {path!r}={actual} "
                f"but shapes.n_train={source_n}; replicated or truncated rows are not allowed."
            )


def validate_suite_case(
    case: dict[str, Any],
    *,
    dataset_dir: Path,
    aliases: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stem, alias_name, alias = _resolve_case_dataset(
        case,
        dataset_dir=dataset_dir,
        aliases=aliases or {},
    )
    npz_path = dataset_dir / f"{stem}.npz"
    json_path = dataset_dir / f"{stem}.json"
    if not npz_path.is_file() or not json_path.is_file():
        alias_text = f" through dataset alias {alias_name!r}" if alias_name else ""
        raise FileNotFoundError(
            f"case {case['id']!r}{alias_text} requires both {npz_path.name} "
            f"and {json_path.name} in {dataset_dir}."
        )
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    source_n = _sidecar_n_train(metadata)
    if alias_name:
        _validate_alias_metadata(
            alias_name=alias_name,
            alias=alias,
            metadata=metadata,
            source_n=source_n,
            json_path=json_path,
        )
    requested_raw = case.get("n_train", 0)
    requested_n = source_n if requested_raw in (None, 0) else int(requested_raw)
    expected_n = int(case["expected_n_train"])
    if requested_n > source_n:
        raise ValueError(
            f"case {case['id']!r} requests {requested_n} rows from a {source_n}-row file."
        )
    if requested_n != expected_n:
        raise ValueError(
            f"case {case['id']!r} resolves to N={requested_n}, expected {expected_n}."
        )
    with zipfile.ZipFile(npz_path, "r") as archive:
        members = set(archive.namelist())
    missing_arrays = sorted({"x_train.npy", "y_train.npy"} - members)
    if missing_arrays:
        raise ValueError(f"{npz_path.name} is missing {missing_arrays}.")
    generation = metadata.get("generation", {})
    return {
        "case_id": str(case["id"]),
        "dataset_alias": alias_name,
        "dataset_family": str(case.get("dataset_family", "")),
        "dataset_stem": stem,
        "dataset_path": str(npz_path),
        "dataset_file_size_bytes": int(npz_path.stat().st_size),
        "dataset_content_index_sha256": _npz_content_index_sha256(npz_path),
        "dataset_metadata_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "source_n_train": int(source_n),
        "n_train": int(requested_n),
        "dim": int(metadata.get("shapes", {}).get("dim", 0)),
        "task_type": metadata.get("task_type"),
        "target_definition": metadata.get("target_definition"),
        "noise_std": generation.get("noise_std"),
        "n_test": metadata.get("shapes", {}).get("n_test"),
        "scale_role": case.get("scale_role", ""),
        "note": case.get("note", ""),
        "estimated_resident_x_y_float64_bytes": int(
            requested_n * (int(metadata.get("shapes", {}).get("dim", 0)) + 1) * 8
        ),
    }


def build_suite_plan(
    suite: dict[str, Any],
    *,
    profile_name: str,
    dataset_dir_override: str = "",
    output_root: Path,
    nufft_backend_override: str = "",
    strict_gpu_eig: bool = False,
) -> list[tuple[dict[str, Any], ControlledConfig]]:
    profiles = suite["profiles"]
    if profile_name not in profiles:
        raise ValueError(
            f"unknown profile {profile_name!r}; choices are {sorted(profiles)}."
        )
    profile = profiles[profile_name]
    common = dict(suite["base"])
    common.update(profile.get("overrides", {}))
    if dataset_dir_override:
        common["dataset_dir"] = dataset_dir_override
    if nufft_backend_override:
        common["nufft_backend"] = nufft_backend_override
    if strict_gpu_eig:
        common["strict_gpu_eig"] = True
    common = _normalize_config_payload(common)
    _validate_controlled_fields(common, source=f"profile {profile_name!r}")
    dataset_dir = _resolve_dataset_dir(str(common.get("dataset_dir", "")))

    plan: list[tuple[dict[str, Any], ControlledConfig]] = []
    seen_ids: set[str] = set()
    for case in profile["cases"]:
        case_id = str(case["id"])
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id {case_id!r} in profile {profile_name!r}.")
        seen_ids.add(case_id)
        validation = validate_suite_case(
            case,
            dataset_dir=dataset_dir,
            aliases=suite.get("dataset_aliases", {}),
        )
        controlled_case = {
            key: value
            for key, value in case.items()
            if key not in _CASE_METADATA_KEYS
        }
        values = dict(common)
        values.update(controlled_case)
        values["dataset_stem"] = validation["dataset_stem"]
        values["dataset_dir"] = str(dataset_dir)
        values["output_dir"] = str((output_root / case_id).resolve())
        values = _normalize_config_payload(values)
        config = ControlledConfig(**values)
        plan.append((validation, config))
    return plan


def _expected_config_payload(config: ControlledConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["methods"] = list(config.methods)
    return _sanitize_json(payload)


def _complete_run(
    run_dir: Path,
    expected_n: int,
    *,
    expected_config: ControlledConfig,
    expected_source_sha256: str,
    expected_dataset_content_index_sha256: str,
    expected_dataset_metadata_sha256: str,
    expected_timing_runtime_sha256: str | None = None,
) -> bool:
    required_paths = [
        run_dir / "system_manifest.json",
        run_dir / "experiment_config.json",
        run_dir / "matched_runs.json",
        run_dir / "matched_runs.csv",
        run_dir / "matched_summary.json",
        run_dir / "matched_summary.csv",
        run_dir / "matched_comparisons.json",
        run_dir / "matched_comparisons.csv",
        run_dir / TIMING_SYSTEM_ARTIFACT_FILENAME,
        run_dir / TIMING_SOLUTIONS_ARTIFACT_FILENAME,
        run_dir / TIMING_SOLUTIONS_MANIFEST_FILENAME,
        run_dir / "run_complete.json",
    ]
    if not all(path.is_file() for path in required_paths):
        return False
    manifest_path = run_dir / "system_manifest.json"
    summary_path = run_dir / "matched_summary.json"
    config_path = run_dir / "experiment_config.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        runs = json.loads((run_dir / "matched_runs.json").read_text(encoding="utf-8"))
        comparisons = json.loads(
            (run_dir / "matched_comparisons.json").read_text(encoding="utf-8")
        )
        completion = json.loads(
            (run_dir / "run_complete.json").read_text(encoding="utf-8")
        )
        timing_solution_manifest = json.loads(
            (run_dir / TIMING_SOLUTIONS_MANIFEST_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        with (run_dir / "matched_summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            summary_csv = list(csv.DictReader(handle))
    except Exception:
        return False
    if not (
        isinstance(manifest, dict)
        and isinstance(saved_config, dict)
        and isinstance(completion, dict)
        and isinstance(timing_solution_manifest, dict)
        and isinstance(summary, list)
        and all(isinstance(row, dict) for row in summary)
        and isinstance(summary_csv, list)
        and all(isinstance(row, dict) for row in summary_csv)
        and isinstance(runs, list)
        and all(isinstance(row, dict) for row in runs)
        and isinstance(comparisons, list)
        and all(isinstance(row, dict) for row in comparisons)
    ):
        return False

    def resume_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    if not (
        manifest.get("system_unchanged")
        and resume_int(manifest.get("n_train")) == int(expected_n)
        and manifest.get("source_bundle_sha256") == expected_source_sha256
        and manifest.get("dataset_content_index_sha256")
        == expected_dataset_content_index_sha256
        and manifest.get("dataset_metadata_sha256") == expected_dataset_metadata_sha256
        and (
            expected_timing_runtime_sha256 is None
            or manifest.get("timing_runtime_sha256")
            == expected_timing_runtime_sha256
        )
        and saved_config == _expected_config_payload(expected_config)
        and summary
    ):
        return False

    expected_methods = tuple(str(method) for method in expected_config.methods)
    summary_methods = [str(row.get("method")) for row in summary]
    summary_csv_methods = [str(row.get("method")) for row in summary_csv]
    if (
        Counter(summary_methods) != Counter(expected_methods)
        or Counter(summary_csv_methods) != Counter(expected_methods)
    ):
        return False
    by_method = {str(row.get("method")): row for row in summary}
    if not all(
        resume_int(by_method[method].get("measured_repeats"))
        == int(expected_config.measured_repeats)
        for method in expected_methods
    ):
        return False

    system_id = str(manifest.get("system_id", ""))
    timing_system_path = run_dir / TIMING_SYSTEM_ARTIFACT_FILENAME
    timing_solutions_path = run_dir / TIMING_SOLUTIONS_ARTIFACT_FILENAME
    timing_solution_manifest_path = run_dir / TIMING_SOLUTIONS_MANIFEST_FILENAME
    timing_system_sha256 = hashlib.sha256(timing_system_path.read_bytes()).hexdigest()
    timing_solutions_sha256 = hashlib.sha256(
        timing_solutions_path.read_bytes()
    ).hexdigest()
    timing_solution_manifest_sha256 = hashlib.sha256(
        timing_solution_manifest_path.read_bytes()
    ).hexdigest()
    if not (
        system_id
        and all(
            bool(manifest.get(field))
            for field in (
                "weights_sha256",
                "gf_sha256",
                "rhs_sha256",
                "rhs_storage_sha256",
                "system_config_sha256",
            )
        )
        and manifest.get("system_artifact_sha256") == timing_system_sha256
        and manifest.get("timing_solution_artifact_sha256")
        == timing_solutions_sha256
        and manifest.get("timing_solution_manifest_sha256")
        == timing_solution_manifest_sha256
        and resume_int(manifest.get("timing_solution_count"))
        == resume_int(timing_solution_manifest.get("solution_count"))
        and resume_int(manifest.get("timing_solution_count")) is not None
        and timing_solution_manifest.get("system_id") == system_id
        and timing_solution_manifest.get("weights_sha256")
        == manifest.get("weights_sha256")
        and timing_solution_manifest.get("gf_sha256") == manifest.get("gf_sha256")
        and timing_solution_manifest.get("rhs_sha256") == manifest.get("rhs_sha256")
        and timing_solution_manifest.get("rhs_storage_sha256")
        == manifest.get("rhs_storage_sha256")
        and timing_solution_manifest.get("system_config_sha256")
        == manifest.get("system_config_sha256")
        and timing_solution_manifest.get("source_bundle_sha256")
        == expected_source_sha256
        and timing_solution_manifest.get("dataset_content_index_sha256")
        == expected_dataset_content_index_sha256
        and timing_solution_manifest.get("dataset_metadata_sha256")
        == expected_dataset_metadata_sha256
        and timing_solution_manifest.get("timing_system_artifact_sha256")
        == timing_system_sha256
        and timing_solution_manifest.get("timing_solution_artifact_sha256")
        == timing_solutions_sha256
    ):
        return False
    expected_run_count = len(expected_methods) * (
        int(expected_config.warmup_repeats) + int(expected_config.measured_repeats)
    )
    if len(runs) != expected_run_count:
        return False
    for method in expected_methods:
        method_rows = [row for row in runs if str(row.get("method")) == method]
        warmups = [row for row in method_rows if bool(row.get("is_warmup"))]
        measured = [row for row in method_rows if not bool(row.get("is_warmup"))]
        if len(warmups) != int(expected_config.warmup_repeats):
            return False
        if len(measured) != int(expected_config.measured_repeats):
            return False
        if {resume_int(row.get("repeat_idx")) for row in measured} != set(
            range(int(expected_config.measured_repeats))
        ):
            return False
        if any(str(row.get("system_id", "")) != system_id for row in method_rows):
            return False

    return bool(
        resume_int(completion.get("schema_version")) == 1
        and completion.get("system_id") == system_id
        and completion.get("source_bundle_sha256") == expected_source_sha256
        and completion.get("dataset_content_index_sha256")
        == expected_dataset_content_index_sha256
        and completion.get("dataset_metadata_sha256")
        == expected_dataset_metadata_sha256
        and completion.get("timing_system_artifact")
        == TIMING_SYSTEM_ARTIFACT_FILENAME
        and completion.get("timing_system_artifact_sha256")
        == timing_system_sha256
        and completion.get("timing_solution_artifact")
        == TIMING_SOLUTIONS_ARTIFACT_FILENAME
        and completion.get("timing_solution_artifact_sha256")
        == timing_solutions_sha256
        and completion.get("timing_solution_manifest")
        == TIMING_SOLUTIONS_MANIFEST_FILENAME
        and completion.get("timing_solution_manifest_sha256")
        == timing_solution_manifest_sha256
        and resume_int(completion.get("timing_solution_count"))
        == resume_int(timing_solution_manifest.get("solution_count"))
        and resume_int(completion.get("timing_solution_count")) is not None
        and completion.get("methods") == list(expected_methods)
        and resume_int(completion.get("warmup_repeats"))
        == int(expected_config.warmup_repeats)
        and resume_int(completion.get("measured_repeats"))
        == int(expected_config.measured_repeats)
        and resume_int(completion.get("run_row_count")) == len(runs)
        and resume_int(completion.get("summary_row_count")) == len(summary)
        and resume_int(completion.get("summary_row_count")) == len(summary_csv)
        and resume_int(completion.get("comparison_row_count")) == len(comparisons)
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            _sanitize_json(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    _write_rows(temporary, rows)
    temporary.replace(path)


def _write_suite_checkpoint(
    output_root: Path,
    status_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
) -> None:
    """Atomically replace suite state so interrupted runs cannot expose stale status."""
    _write_json_atomic(output_root / "suite_status.json", status_rows)
    _write_json_atomic(output_root / "suite_index.json", index_rows)
    _write_rows_atomic(output_root / "suite_status.csv", status_rows)
    _write_rows_atomic(output_root / "suite_index.csv", index_rows)


def run_suite(
    suite: dict[str, Any],
    *,
    profile_name: str,
    dataset_dir_override: str,
    output_root: Path,
    execute: bool,
    resume: bool,
    nufft_backend_override: str,
    strict_gpu_eig: bool,
) -> tuple[Path, bool]:
    output_root.mkdir(parents=True, exist_ok=True)
    plan = build_suite_plan(
        suite,
        profile_name=profile_name,
        dataset_dir_override=dataset_dir_override,
        output_root=output_root,
        nufft_backend_override=nufft_backend_override,
        strict_gpu_eig=strict_gpu_eig,
    )
    system_config_ids = [system_config_fingerprint(config) for _, config in plan]
    system_config_counts = Counter(system_config_ids)
    plan_rows = []
    for (validation, config), system_config_id in zip(plan, system_config_ids):
        shares_frozen_system = system_config_counts[system_config_id] > 1
        plan_rows.append(
            {
                **validation,
                "output_dir": config.output_dir,
                "system_config_sha256": system_config_id,
                "shared_frozen_system": shares_frozen_system,
                "shared_system_case_count": int(system_config_counts[system_config_id]),
                "shared_system_artifact": (
                    str(
                        (
                            output_root
                            / "_shared_systems"
                            / f"{system_config_id}.npz"
                        ).resolve()
                    )
                    if shares_frozen_system
                    else ""
                ),
                "methods": list(config.methods),
                "fourier_eps": config.fourier_eps,
                "tol": config.tol,
                "precision": config.precision,
                "precompute_chunk_size": config.precompute_chunk_size,
                "nufft_backend": config.nufft_backend,
            }
        )
    _write_json_atomic(output_root / "suite_plan.json", plan_rows)
    if not execute:
        return output_root, False

    expected_source_sha256 = str(_source_manifest()["source_bundle_sha256"])
    shared_systems: dict[str, PreparedSystem] = {}
    shared_group_records: dict[str, dict[str, Any]] = {}
    current_timing_runtimes: dict[str, dict[str, Any]] = {}

    def _current_timing_runtime_sha256(config: ControlledConfig) -> str:
        backend_key = str(config.nufft_backend)
        if backend_key not in current_timing_runtimes:
            current_timing_runtimes[backend_key] = probe_timing_runtime(backend_key)
        runtime_sha256 = current_timing_runtimes[backend_key].get(
            "timing_runtime_sha256"
        )
        if not runtime_sha256:
            raise RuntimeError("current timing runtime probe lacks timing_runtime_sha256")
        return str(runtime_sha256)

    def _shared_system_for_case(
        validation: dict[str, Any],
        config: ControlledConfig,
    ) -> tuple[PreparedSystem | None, Path | None]:
        system_config_id = system_config_fingerprint(config)
        if system_config_counts[system_config_id] <= 1:
            return None, None
        artifact_path = (
            output_root / "_shared_systems" / f"{system_config_id}.npz"
        ).resolve()
        if system_config_id in shared_systems:
            return shared_systems[system_config_id], artifact_path

        system: PreparedSystem | None = None
        if artifact_path.is_file():
            try:
                system = load_prepared_system_artifact(
                    config,
                    artifact_path,
                    expected_source_sha256=expected_source_sha256,
                    expected_dataset_content_index_sha256=str(
                        validation["dataset_content_index_sha256"]
                    ),
                    expected_dataset_metadata_sha256=str(
                        validation["dataset_metadata_sha256"]
                    ),
                )
                print(
                    f"[suite] LOAD shared frozen system {system.system_id[:12]} "
                    f"from {artifact_path}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[suite] REBUILD invalid shared-system artifact {artifact_path}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if system is None:
            system = prepare_shared_system(config)
            for manifest_key, validation_key in (
                ("dataset_content_index_sha256", "dataset_content_index_sha256"),
                ("dataset_metadata_sha256", "dataset_metadata_sha256"),
            ):
                if str(system.manifest.get(manifest_key)) != str(
                    validation[validation_key]
                ):
                    raise RuntimeError(
                        f"prepared shared system {manifest_key} does not match suite validation"
                    )
            if str(system.manifest.get("source_bundle_sha256")) != expected_source_sha256:
                raise RuntimeError(
                    "prepared shared system source bundle does not match suite source"
                )
            save_prepared_system_artifact(system, config, artifact_path)
            print(
                f"[suite] SAVE shared frozen system {system.system_id[:12]} "
                f"to {artifact_path}",
                flush=True,
            )
        shared_systems[system_config_id] = system
        return system, artifact_path

    status_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    had_failure = False
    _write_suite_checkpoint(output_root, status_rows, index_rows)
    for case_index, (validation, config) in enumerate(plan, start=1):
        case_id = str(validation["case_id"])
        run_dir = Path(config.output_dir)
        started_utc = datetime.now().astimezone().isoformat()
        status_rows.append(
            {
                **validation,
                "status": "running",
                "case_index": int(case_index),
                "case_count": int(len(plan)),
                "started_utc": started_utc,
                "run_dir": str(run_dir),
            }
        )
        _write_suite_checkpoint(output_root, status_rows, index_rows)
        print(
            f"[suite] START case {case_index}/{len(plan)}: {case_id} "
            f"(N={validation['n_train']}, methods={list(config.methods)})",
            flush=True,
        )
        case_stage = "shared_system"
        try:
            shared_system, shared_artifact_path = _shared_system_for_case(
                validation, config
            )
            expected_fixed_components = (
                {
                    key: shared_system.manifest.get(key)
                    for key in (
                        "system_id",
                        "weights_sha256",
                        "gf_sha256",
                        "rhs_sha256",
                        "rhs_storage_sha256",
                        "reg_lambda",
                        "device_name",
                        "compute_capability",
                        "timing_runtime_sha256",
                    )
                }
                if shared_system is not None
                else None
            )
            expected_timing_runtime_sha256 = (
                str(shared_system.manifest.get("timing_runtime_sha256"))
                if shared_system is not None
                else _current_timing_runtime_sha256(config)
            )
            case_stage = "resume_check"
            resumed = resume and _complete_run(
                run_dir,
                int(validation["n_train"]),
                expected_config=config,
                expected_source_sha256=expected_source_sha256,
                expected_dataset_content_index_sha256=str(
                    validation["dataset_content_index_sha256"]
                ),
                expected_dataset_metadata_sha256=str(
                    validation["dataset_metadata_sha256"]
                ),
                expected_timing_runtime_sha256=expected_timing_runtime_sha256,
            )
            if resumed and expected_fixed_components is not None:
                resumed_manifest = json.loads(
                    (run_dir / "system_manifest.json").read_text(encoding="utf-8")
                )
                actual_fixed_components = {
                    key: resumed_manifest.get(key)
                    for key in expected_fixed_components
                }
                if actual_fixed_components != expected_fixed_components:
                    resumed = False
                    print(
                        f"[suite] RERUN case {case_id!r}: its resumed A,b hashes do "
                        "not match the suite's shared frozen system.",
                        file=sys.stderr,
                        flush=True,
                    )
            if resumed:
                status = "resumed_existing"
            else:
                case_stage = "experiment"
                if shared_system is None:
                    run_controlled_experiment(config)
                else:
                    run_controlled_experiment(
                        config,
                        prepared_system=shared_system,
                    )
                status = "completed"

            case_stage = "artifact_validation"
            manifest = json.loads(
                (run_dir / "system_manifest.json").read_text(encoding="utf-8")
            )
            if int(manifest.get("n_train", -1)) != int(validation["n_train"]):
                raise RuntimeError(
                    f"case {case_id!r} wrote N={manifest.get('n_train')}, "
                    f"expected {validation['n_train']}."
                )
            shared_system_verified = False
            if expected_fixed_components is not None:
                actual_fixed_components = {
                    key: manifest.get(key) for key in expected_fixed_components
                }
                if actual_fixed_components != expected_fixed_components:
                    raise RuntimeError(
                        f"case {case_id!r} did not use the exact shared A,b: "
                        f"{actual_fixed_components} != {expected_fixed_components}"
                    )
                shared_system_verified = True
                group_id = system_config_fingerprint(config)
                record = shared_group_records.setdefault(
                    group_id,
                    {
                        "system_config_sha256": group_id,
                        "system_id": shared_system.system_id,
                        "weights_sha256": shared_system.manifest.get(
                            "weights_sha256"
                        ),
                        "gf_sha256": shared_system.manifest.get("gf_sha256"),
                        "rhs_sha256": shared_system.manifest.get("rhs_sha256"),
                        "rhs_storage_sha256": shared_system.manifest.get(
                            "rhs_storage_sha256"
                        ),
                        "reg_lambda": float(shared_system.reg_lambda),
                        "device_name": shared_system.manifest.get("device_name"),
                        "compute_capability": shared_system.manifest.get(
                            "compute_capability"
                        ),
                        "timing_runtime_sha256": shared_system.manifest.get(
                            "timing_runtime_sha256"
                        ),
                        "artifact_relative_path": str(
                            shared_artifact_path.relative_to(output_root)
                        ).replace("\\", "/"),
                        "artifact_sha256": hashlib.sha256(
                            shared_artifact_path.read_bytes()
                        ).hexdigest(),
                        "expected_case_count": int(system_config_counts[group_id]),
                        "case_ids": [],
                        "case_run_dirs": [],
                        "all_cases_exact_match": True,
                    },
                )
                record["case_ids"].append(case_id)
                record["case_run_dirs"].append(str(run_dir))
            summaries = json.loads(
                (run_dir / "matched_summary.json").read_text(encoding="utf-8")
            )
            ineligible_details = [
                {
                    key: row.get(key)
                    for key in (
                        "method",
                        "method_kind",
                        "measured_repeats",
                        "converged_repeats",
                        "iterations_max",
                        "true_relres_max",
                    )
                }
                for row in summaries
                if not bool(row.get("performance_claim_eligible"))
            ]
            ineligible = [str(row.get("method")) for row in ineligible_details]

            diagnostic_errors: list[dict[str, Any]] = []
            diagnostics_path = run_dir / "post_diagnostics.json"
            if diagnostics_path.is_file():
                diagnostic_rows = json.loads(
                    diagnostics_path.read_text(encoding="utf-8")
                )
                diagnostic_errors = [
                    {
                        key: row.get(key)
                        for key in (
                            "method",
                            "method_kind",
                            "diagnostic_error_stage",
                            "error_type",
                            "error_message",
                        )
                    }
                    for row in diagnostic_rows
                    if str(row.get("diagnostic_status", "")).lower()
                    not in {"", "ok"}
                ]

            status_row: dict[str, Any] = {
                **validation,
                "status": status,
                "case_index": int(case_index),
                "case_count": int(len(plan)),
                "started_utc": started_utc,
                "finished_utc": datetime.now().astimezone().isoformat(),
                "run_dir": str(run_dir),
            }
            if expected_fixed_components is not None:
                status_row.update(
                    {
                        "shared_system_config_sha256": system_config_fingerprint(
                            config
                        ),
                        "shared_system_id": shared_system.system_id,
                        "shared_system_artifact": str(shared_artifact_path),
                        "shared_system_exact_match": shared_system_verified,
                    }
                )
            if ineligible:
                had_failure = True
                status_row["ineligible_methods"] = ineligible
                status_row["ineligible_method_details"] = ineligible_details
                print(
                    f"[suite] INELIGIBLE case={case_id!r}: "
                    f"{json.dumps(ineligible_details, ensure_ascii=False)}",
                    file=sys.stderr,
                    flush=True,
                )
            if diagnostic_errors:
                had_failure = True
                status_row["diagnostic_errors"] = diagnostic_errors
                status_row["post_diagnostics_path"] = str(diagnostics_path)
                print(
                    f"[suite] DIAGNOSTIC ERROR case={case_id!r}: "
                    f"{json.dumps(diagnostic_errors, ensure_ascii=False)}",
                    file=sys.stderr,
                    flush=True,
                )
            if ineligible and diagnostic_errors:
                status_row["status"] = (
                    f"{status}_with_ineligible_methods_and_diagnostic_errors"
                )
            elif ineligible:
                status_row["status"] = f"{status}_with_ineligible_methods"
            elif diagnostic_errors:
                status_row["status"] = f"{status}_with_diagnostic_errors"
            status_rows[-1] = status_row

            for summary in summaries:
                index_rows.append(
                    {
                        "case_id": case_id,
                        "dataset_family": validation.get("dataset_family", ""),
                        "dataset_stem": validation["dataset_stem"],
                        "task_type": validation["task_type"],
                        "scale_role": validation["scale_role"],
                        "n_train": validation["n_train"],
                        "M": manifest.get("M"),
                        "system_id": manifest.get("system_id"),
                        "shared_system_config_sha256": (
                            system_config_fingerprint(config)
                            if expected_fixed_components is not None
                            else ""
                        ),
                        "shared_system_exact_match": bool(shared_system_verified),
                        "run_dir": str(run_dir),
                        **summary,
                    }
                )
            _write_suite_checkpoint(output_root, status_rows, index_rows)
            print(
                f"[suite] END case {case_index}/{len(plan)}: {case_id} "
                f"status={status_row['status']}",
                flush=True,
            )
        except Exception as exc:
            had_failure = True
            failure_traceback = traceback.format_exc()
            error_row: dict[str, Any] = {
                **validation,
                "status": "error",
                "case_index": int(case_index),
                "case_count": int(len(plan)),
                "started_utc": started_utc,
                "finished_utc": datetime.now().astimezone().isoformat(),
                "error_stage": case_stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": failure_traceback,
                "run_dir": str(run_dir),
            }
            solver_breakdown = getattr(exc, "diagnostics", None)
            if isinstance(solver_breakdown, dict):
                error_row["solver_breakdown"] = dict(solver_breakdown)
            status_rows[-1] = error_row
            _write_suite_checkpoint(output_root, status_rows, index_rows)
            print(
                f"[suite] ERROR case {case_index}/{len(plan)}: {case_id} "
                f"stage={case_stage}: {type(exc).__name__}: {exc}\n"
                f"{failure_traceback}",
                file=sys.stderr,
                flush=True,
            )

    shared_group_rows: list[dict[str, Any]] = []
    for system_config_id, expected_count in sorted(system_config_counts.items()):
        if expected_count <= 1:
            continue
        planned_case_ids = [
            str(validation["case_id"])
            for (validation, _), candidate_id in zip(plan, system_config_ids)
            if candidate_id == system_config_id
        ]
        record = dict(
            shared_group_records.get(
                system_config_id,
                {
                    "system_config_sha256": system_config_id,
                    "system_id": None,
                    "weights_sha256": None,
                    "gf_sha256": None,
                    "rhs_sha256": None,
                    "rhs_storage_sha256": None,
                    "reg_lambda": None,
                    "device_name": None,
                    "compute_capability": None,
                    "timing_runtime_sha256": None,
                    "artifact_relative_path": str(
                        Path("_shared_systems") / f"{system_config_id}.npz"
                    ).replace("\\", "/"),
                    "artifact_sha256": None,
                    "case_ids": [],
                    "case_run_dirs": [],
                },
            )
        )
        observed_count = len(record.get("case_ids", []))
        record.update(
            {
                "expected_case_count": int(expected_count),
                "observed_verified_case_count": int(observed_count),
                "planned_case_ids": planned_case_ids,
                "all_cases_exact_match": bool(observed_count == expected_count),
                "verification_rule": (
                    "Exact equality of system_id, weights_sha256, gf_sha256, "
                    "solve/storage rhs hashes, reg_lambda, device_name, "
                    "compute_capability, and the complete timing-runtime hash; "
                    "no tolerance is used."
                ),
            }
        )
        shared_group_rows.append(record)
    _write_json_atomic(output_root / "shared_system_groups.json", shared_group_rows)
    _write_rows_atomic(output_root / "shared_system_groups.csv", shared_group_rows)
    _write_suite_checkpoint(output_root, status_rows, index_rows)
    return output_root, had_failure


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or run matched fixed-system experiments on GeoLife, USGS, and synthetic data."
        )
    )
    parser.add_argument("--config", default=str(_DEFAULT_SUITE_CONFIG))
    parser.add_argument("--profile", default="demo")
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Override the processed-data directory, for example a mounted Google Drive path.",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--nufft-backend",
        choices=("", "auto", "cufinufft", "none"),
        default="",
    )
    parser.add_argument("--strict-gpu-eig", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the cases. Without this flag, only validate data and write suite_plan.json.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


SUITE_EXIT_OK = 0
SUITE_EXIT_EXECUTION_ERROR = 1
SUITE_EXIT_SCIENTIFIC_FAILURE = 2


def _suite_exit_code(output_root: Path, *, had_failure: bool) -> int:
    """Classify a completed suite without conflating scientific and runtime failures.

    A controlled case can finish writing all required artifacts while one method is
    ineligible or a post diagnostic fails.  That is a scientific result, not an
    orchestration crash.  Case/config/data errors remain hard execution failures.
    """

    if not had_failure:
        return SUITE_EXIT_OK

    status_path = output_root / "suite_status.json"
    if not status_path.is_file():
        return SUITE_EXIT_EXECUTION_ERROR
    try:
        status_rows = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return SUITE_EXIT_EXECUTION_ERROR
    if not isinstance(status_rows, list) or not status_rows:
        return SUITE_EXIT_EXECUTION_ERROR

    terminal_statuses = [str(row.get("status", "")) for row in status_rows]
    if any(status == "error" for status in terminal_statuses):
        return SUITE_EXIT_EXECUTION_ERROR
    if any(status in {"", "running"} for status in terminal_statuses):
        return SUITE_EXIT_EXECUTION_ERROR
    return SUITE_EXIT_SCIENTIFIC_FAILURE


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    suite = load_suite_config(args.config)
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        tag = datetime.now().strftime(f"suite_{args.profile}_%Y%m%d_%H%M%S")
        output_root = (_HERE / "outputs" / tag).resolve()
    result, failed = run_suite(
        suite,
        profile_name=str(args.profile),
        dataset_dir_override=str(args.dataset_dir),
        output_root=output_root,
        execute=bool(args.execute),
        resume=bool(args.resume),
        nufft_backend_override=str(args.nufft_backend),
        strict_gpu_eig=bool(args.strict_gpu_eig),
    )
    print(f"Wrote controlled-suite files to {result}")
    return _suite_exit_code(result, had_failure=failed)


if __name__ == "__main__":
    raise SystemExit(main())
