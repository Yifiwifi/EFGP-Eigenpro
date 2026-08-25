from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark import (
    ControlledConfig,
    _npz_content_index_sha256,
    _resolve_dataset_dir,
    _sanitize_json,
    _source_manifest,
    run_controlled_experiment,
)


_HERE = Path(__file__).resolve().parent
_DEFAULT_SUITE_CONFIG = _HERE / "three_dataset_suite.json"
_CASE_METADATA_KEYS = {
    "id",
    "dataset_alias",
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
    except Exception:
        return False
    if not (
        manifest.get("system_unchanged")
        and int(manifest.get("n_train", -1)) == int(expected_n)
        and manifest.get("source_bundle_sha256") == expected_source_sha256
        and manifest.get("dataset_content_index_sha256")
        == expected_dataset_content_index_sha256
        and manifest.get("dataset_metadata_sha256") == expected_dataset_metadata_sha256
        and saved_config == _expected_config_payload(expected_config)
        and isinstance(summary, list)
        and summary
        and isinstance(runs, list)
        and isinstance(comparisons, list)
        and isinstance(completion, dict)
    ):
        return False

    by_method = {
        str(row.get("method")): row for row in summary if isinstance(row, dict)
    }
    expected_methods = tuple(str(method) for method in expected_config.methods)
    if set(by_method) != set(expected_methods):
        return False
    if not all(
        int(by_method[method].get("measured_repeats", -1))
        == int(expected_config.measured_repeats)
        for method in expected_methods
    ):
        return False

    system_id = str(manifest.get("system_id", ""))
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
        if {int(row.get("repeat_idx", -1)) for row in measured} != set(
            range(int(expected_config.measured_repeats))
        ):
            return False
        if any(str(row.get("system_id", "")) != system_id for row in method_rows):
            return False

    return bool(
        int(completion.get("schema_version", -1)) == 1
        and completion.get("system_id") == system_id
        and completion.get("source_bundle_sha256") == expected_source_sha256
        and completion.get("dataset_content_index_sha256")
        == expected_dataset_content_index_sha256
        and completion.get("dataset_metadata_sha256")
        == expected_dataset_metadata_sha256
        and completion.get("methods") == list(expected_methods)
        and int(completion.get("warmup_repeats", -1))
        == int(expected_config.warmup_repeats)
        and int(completion.get("measured_repeats", -1))
        == int(expected_config.measured_repeats)
        and int(completion.get("run_row_count", -1)) == len(runs)
        and int(completion.get("summary_row_count", -1)) == len(summary)
        and int(completion.get("comparison_row_count", -1)) == len(comparisons)
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
    plan_rows = []
    for validation, config in plan:
        plan_rows.append(
            {
                **validation,
                "output_dir": config.output_dir,
                "methods": list(config.methods),
                "fourier_eps": config.fourier_eps,
                "tol": config.tol,
                "precision": config.precision,
                "precompute_chunk_size": config.precompute_chunk_size,
                "nufft_backend": config.nufft_backend,
            }
        )
    (output_root / "suite_plan.json").write_text(
        json.dumps(_sanitize_json(plan_rows), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if not execute:
        return output_root, False

    expected_source_sha256 = str(_source_manifest()["source_bundle_sha256"])
    status_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    had_failure = False
    for validation, config in plan:
        case_id = str(validation["case_id"])
        run_dir = Path(config.output_dir)
        if resume and _complete_run(
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
        ):
            status = "resumed_existing"
        else:
            try:
                run_controlled_experiment(config)
                status = "completed"
            except Exception as exc:
                had_failure = True
                status_rows.append(
                    {
                        **validation,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "run_dir": str(run_dir),
                    }
                )
                continue
        manifest = json.loads((run_dir / "system_manifest.json").read_text(encoding="utf-8"))
        if int(manifest.get("n_train", -1)) != int(validation["n_train"]):
            raise RuntimeError(
                f"case {case_id!r} wrote N={manifest.get('n_train')}, "
                f"expected {validation['n_train']}."
            )
        summaries = json.loads((run_dir / "matched_summary.json").read_text(encoding="utf-8"))
        ineligible = [
            str(row.get("method"))
            for row in summaries
            if not bool(row.get("performance_claim_eligible"))
        ]
        if ineligible:
            had_failure = True
            status = f"{status}_with_ineligible_methods"
        status_rows.append({**validation, "status": status, "run_dir": str(run_dir)})
        for summary in summaries:
            index_rows.append(
                {
                    "case_id": case_id,
                    "dataset_stem": validation["dataset_stem"],
                    "task_type": validation["task_type"],
                    "scale_role": validation["scale_role"],
                    "n_train": validation["n_train"],
                    "M": manifest.get("M"),
                    "system_id": manifest.get("system_id"),
                    "run_dir": str(run_dir),
                    **summary,
                }
            )

    (output_root / "suite_status.json").write_text(
        json.dumps(_sanitize_json(status_rows), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (output_root / "suite_index.json").write_text(
        json.dumps(_sanitize_json(index_rows), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_rows(output_root / "suite_status.csv", status_rows)
    _write_rows(output_root / "suite_index.csv", index_rows)
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
