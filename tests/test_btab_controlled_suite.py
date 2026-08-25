from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.suite import (
    _complete_run,
    build_suite_plan,
    load_suite_config,
)


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "efgp_eigenpro_py"
        / "gpu"
        / "box_toeplitz_active_block"
        / "controlled"
        / "three_dataset_suite.json"
    )


def _set_metadata_path(metadata: dict, path: str, value: object) -> None:
    parent = metadata
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent.setdefault(part, {})
    parent[parts[-1]] = value


def _write_stub_dataset(
    dataset_dir: Path,
    *,
    stem: str,
    n_train: int,
    metadata: dict | None = None,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": stem,
        "task_type": "fixture_regression",
        "target_definition": "fixture target",
        "shapes": {"n_train": n_train, "n_test": max(1, n_train // 4), "dim": 2},
    }
    if metadata:
        payload.update(metadata)
    with zipfile.ZipFile(dataset_dir / f"{stem}.npz", "w") as archive:
        archive.writestr("x_train.npy", b"fixture-x")
        archive.writestr("y_train.npy", b"fixture-y")
    (dataset_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_profile_fixtures(suite: dict, profile_name: str, dataset_dir: Path) -> None:
    aliases = suite.get("dataset_aliases", {})
    for case in suite["profiles"][profile_name]["cases"]:
        alias_name = case.get("dataset_alias")
        if alias_name:
            alias = aliases[alias_name]
            stem = alias["dataset_stem"]
            metadata: dict = {}
            for path, value in alias.get("metadata_equals", {}).items():
                _set_metadata_path(metadata, path, value)
            metadata["source_file"] = "fixture://local-source"
            metadata["processed_file"] = str(dataset_dir / f"{stem}.npz")
            n_train = int(metadata["shapes"]["n_train"])
            _write_stub_dataset(
                dataset_dir,
                stem=stem,
                n_train=n_train,
                metadata=metadata,
            )
            continue

        stem = case["dataset_stem"]
        expected_n = int(case["expected_n_train"])
        source_n = max(expected_n, 100_000 if profile_name == "demo" else expected_n)
        metadata = {}
        if stem.startswith("synthetic_true_func_2d"):
            metadata = {
                "task_type": "2d_synthetic_regression",
                "generation": {"noise_std": 0.02},
                "shapes": {"n_train": source_n, "n_test": source_n // 4, "dim": 2},
            }
        _write_stub_dataset(
            dataset_dir,
            stem=stem,
            n_train=source_n,
            metadata=metadata,
        )


def test_three_dataset_profiles_use_exact_large_stems_without_replication(tmp_path) -> None:
    suite = load_suite_config(_config_path())
    _write_profile_fixtures(suite, "scale_10m", tmp_path)
    plan = build_suite_plan(
        suite,
        profile_name="scale_10m",
        dataset_dir_override=str(tmp_path),
        output_root=tmp_path,
    )
    validation = {row[0]["case_id"]: row[0] for row in plan}
    configs = {row[0]["case_id"]: row[1] for row in plan}

    assert validation["synthetic_n10000000"]["n_train"] == 10_000_000
    assert validation["usgs_n10000000"]["n_train"] == 10_000_000
    assert validation["geolife_n10000000"]["n_train"] == 10_000_000
    assert validation["geolife_n10000000"]["dataset_alias"] == "geolife_n10000000"
    assert configs["synthetic_n10000000"].n_train is None
    assert configs["usgs_n10000000"].precompute_chunk_size == 1_000_000
    assert "rpcholesky" in configs["geolife_n10000000"].methods
    assert configs["geolife_n10000000"].kernel_family == "se"
    assert configs["geolife_n10000000"].lengthscale == 0.02
    assert configs["synthetic_n10000000"].kernel_family == "matern"

    scale_100m_ids = {
        case["id"] for case in suite["profiles"]["scale_100m"]["cases"]
    }
    assert scale_100m_ids == {"synthetic_n100000000", "usgs_n100000000"}


def test_demo_profile_carries_synthetic_provenance(tmp_path) -> None:
    suite = load_suite_config(_config_path())
    _write_profile_fixtures(suite, "demo", tmp_path)
    plan = build_suite_plan(
        suite,
        profile_name="demo",
        dataset_dir_override=str(tmp_path),
        output_root=tmp_path,
    )
    synthetic = next(row for row, _ in plan if row["case_id"] == "synthetic_n5000")
    geolife_config = next(
        config for row, config in plan if row["case_id"] == "geolife_n5000"
    )
    assert synthetic["task_type"] == "2d_synthetic_regression"
    assert synthetic["noise_std"] == 0.02
    assert synthetic["n_test"] == 25_000
    assert geolife_config.box_budget == 4096
    assert geolife_config.inverse_max_size == 256


def test_missing_geolife_alias_and_wrong_sidecar_fail_clearly(tmp_path) -> None:
    suite = load_suite_config(_config_path())
    case = suite["profiles"]["demo"]["cases"][0]
    alias = suite["dataset_aliases"][case["dataset_alias"]]

    try:
        build_suite_plan(
            suite,
            profile_name="demo",
            dataset_dir_override=str(tmp_path),
            output_root=tmp_path / "out",
        )
    except FileNotFoundError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing GeoLife alias unexpectedly validated")
    assert case["dataset_alias"] in message
    assert alias["dataset_stem"] in message

    metadata: dict = {}
    for path, value in alias["metadata_equals"].items():
        _set_metadata_path(metadata, path, value)
    metadata["source_file"] = "fixture://local-source"
    metadata["processed_file"] = str(tmp_path / f"{alias['dataset_stem']}.npz")
    metadata["frozen_policy"]["lon_min"] = 116.09
    _write_stub_dataset(
        tmp_path,
        stem=alias["dataset_stem"],
        n_train=100_000,
        metadata=metadata,
    )
    try:
        build_suite_plan(
            suite,
            profile_name="demo",
            dataset_dir_override=str(tmp_path),
            output_root=tmp_path / "out",
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("wrong GeoLife crop unexpectedly validated")
    assert "frozen_policy.lon_min" in message
    assert "116.1" in message

    metadata["frozen_policy"]["lon_min"] = 116.1
    metadata["sampling"]["split_unit"] = "source record"
    _write_stub_dataset(
        tmp_path,
        stem=alias["dataset_stem"],
        n_train=100_000,
        metadata=metadata,
    )
    try:
        build_suite_plan(
            suite,
            profile_name="demo",
            dataset_dir_override=str(tmp_path),
            output_root=tmp_path / "out",
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("point-level GeoLife split unexpectedly validated")
    assert "sampling.split_unit" in message
    assert "complete PLT trajectory" in message


def test_resume_requires_matching_config_source_and_repeat_count(tmp_path) -> None:
    suite = load_suite_config(_config_path())
    dataset_dir = tmp_path / "data"
    _write_profile_fixtures(suite, "demo", dataset_dir)
    _, config = build_suite_plan(
        suite,
        profile_name="demo",
        dataset_dir_override=str(dataset_dir),
        output_root=tmp_path,
    )[0]
    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True)

    config_payload = dict(config.__dict__)
    config_payload["methods"] = list(config.methods)
    config_payload["diagnostic_topk"] = list(config.diagnostic_topk)
    (run_dir / "experiment_config.json").write_text(
        json.dumps(config_payload), encoding="utf-8"
    )
    (run_dir / "system_manifest.json").write_text(
        json.dumps(
            {
                "system_id": "fixed-system",
                "system_unchanged": True,
                "n_train": 5000,
                "source_bundle_sha256": "frozen-source",
                "dataset_content_index_sha256": "frozen-data",
                "dataset_metadata_sha256": "frozen-metadata",
            }
        ),
        encoding="utf-8",
    )
    summaries = [
        {"method": method, "measured_repeats": config.measured_repeats}
        for method in config.methods
    ]
    (run_dir / "matched_summary.json").write_text(
        json.dumps(
            summaries
        ),
        encoding="utf-8",
    )
    runs = []
    for method in config.methods:
        for repeat_idx in range(-config.warmup_repeats, config.measured_repeats):
            runs.append(
                {
                    "method": method,
                    "repeat_idx": repeat_idx,
                    "is_warmup": repeat_idx < 0,
                    "system_id": "fixed-system",
                }
            )
    (run_dir / "matched_runs.json").write_text(json.dumps(runs), encoding="utf-8")
    comparisons = [{"reference_method": "cg", "candidate_method": "default"}]
    (run_dir / "matched_comparisons.json").write_text(
        json.dumps(comparisons), encoding="utf-8"
    )
    for name in ("matched_runs.csv", "matched_summary.csv", "matched_comparisons.csv"):
        (run_dir / name).write_text("header\n", encoding="utf-8")
    (run_dir / "run_complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "system_id": "fixed-system",
                "source_bundle_sha256": "frozen-source",
                "dataset_content_index_sha256": "frozen-data",
                "dataset_metadata_sha256": "frozen-metadata",
                "methods": list(config.methods),
                "warmup_repeats": config.warmup_repeats,
                "measured_repeats": config.measured_repeats,
                "run_row_count": len(runs),
                "summary_row_count": len(summaries),
                "comparison_row_count": len(comparisons),
            }
        ),
        encoding="utf-8",
    )

    assert _complete_run(
        run_dir,
        5000,
        expected_config=config,
        expected_source_sha256="frozen-source",
        expected_dataset_content_index_sha256="frozen-data",
        expected_dataset_metadata_sha256="frozen-metadata",
    )
    assert not _complete_run(
        run_dir,
        5000,
        expected_config=replace(config, rpcholesky_rank=config.rpcholesky_rank + 1),
        expected_source_sha256="frozen-source",
        expected_dataset_content_index_sha256="frozen-data",
        expected_dataset_metadata_sha256="frozen-metadata",
    )
    assert not _complete_run(
        run_dir,
        5000,
        expected_config=config,
        expected_source_sha256="different-source",
        expected_dataset_content_index_sha256="frozen-data",
        expected_dataset_metadata_sha256="frozen-metadata",
    )
    (run_dir / "matched_runs.csv").unlink()
    assert not _complete_run(
        run_dir,
        5000,
        expected_config=config,
        expected_source_sha256="frozen-source",
        expected_dataset_content_index_sha256="frozen-data",
        expected_dataset_metadata_sha256="frozen-metadata",
    )
