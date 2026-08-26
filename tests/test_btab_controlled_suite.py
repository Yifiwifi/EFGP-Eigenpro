from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    suite as suite_module,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark import (
    ControlledConfig,
    TIMING_SOLUTIONS_ARTIFACT_FILENAME,
    TIMING_SOLUTIONS_MANIFEST_FILENAME,
    TIMING_SYSTEM_ARTIFACT_FILENAME,
)
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


def test_resume_requires_matching_evidence_and_rejects_malformed_valid_json(
    tmp_path: Path,
) -> None:
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
    timing_system_path = run_dir / TIMING_SYSTEM_ARTIFACT_FILENAME
    timing_solution_path = run_dir / TIMING_SOLUTIONS_ARTIFACT_FILENAME
    timing_solution_manifest_path = run_dir / TIMING_SOLUTIONS_MANIFEST_FILENAME
    timing_system_path.write_bytes(b"exact-system-artifact")
    timing_solution_path.write_bytes(b"canonical-timing-betas")
    timing_system_sha256 = hashlib.sha256(timing_system_path.read_bytes()).hexdigest()
    timing_solution_sha256 = hashlib.sha256(
        timing_solution_path.read_bytes()
    ).hexdigest()
    timing_solution_manifest_path.write_text(
        json.dumps(
            {
                "system_id": "fixed-system",
                "weights_sha256": "weights-sha",
                "gf_sha256": "gf-sha",
                "rhs_sha256": "rhs-sha",
                "rhs_storage_sha256": "rhs-storage-sha",
                "system_config_sha256": "system-config-sha",
                "source_bundle_sha256": "frozen-source",
                "dataset_content_index_sha256": "frozen-data",
                "dataset_metadata_sha256": "frozen-metadata",
                "timing_system_artifact_sha256": timing_system_sha256,
                "timing_solution_artifact_sha256": timing_solution_sha256,
                "solution_count": len(config.methods),
            }
        ),
        encoding="utf-8",
    )
    timing_solution_manifest_sha256 = hashlib.sha256(
        timing_solution_manifest_path.read_bytes()
    ).hexdigest()
    (run_dir / "system_manifest.json").write_text(
        json.dumps(
            {
                "system_id": "fixed-system",
                "system_unchanged": True,
                "n_train": 5000,
                "source_bundle_sha256": "frozen-source",
                "dataset_content_index_sha256": "frozen-data",
                "dataset_metadata_sha256": "frozen-metadata",
                "weights_sha256": "weights-sha",
                "gf_sha256": "gf-sha",
                "rhs_sha256": "rhs-sha",
                "rhs_storage_sha256": "rhs-storage-sha",
                "system_config_sha256": "system-config-sha",
                "timing_runtime_sha256": "fixture-runtime",
                "system_artifact_sha256": timing_system_sha256,
                "timing_solution_artifact_sha256": timing_solution_sha256,
                "timing_solution_manifest_sha256": timing_solution_manifest_sha256,
                "timing_solution_count": len(config.methods),
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
    (run_dir / "matched_runs.csv").write_text("header\n", encoding="utf-8")
    (run_dir / "matched_summary.csv").write_text(
        "method,measured_repeats\n"
        + "".join(
            f"{method},{config.measured_repeats}\n" for method in config.methods
        ),
        encoding="utf-8",
    )
    (run_dir / "matched_comparisons.csv").write_text("header\n", encoding="utf-8")
    (run_dir / "run_complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "system_id": "fixed-system",
                "source_bundle_sha256": "frozen-source",
                "dataset_content_index_sha256": "frozen-data",
                "dataset_metadata_sha256": "frozen-metadata",
                "timing_system_artifact": TIMING_SYSTEM_ARTIFACT_FILENAME,
                "timing_system_artifact_sha256": timing_system_sha256,
                "timing_solution_artifact": TIMING_SOLUTIONS_ARTIFACT_FILENAME,
                "timing_solution_artifact_sha256": timing_solution_sha256,
                "timing_solution_manifest": TIMING_SOLUTIONS_MANIFEST_FILENAME,
                "timing_solution_manifest_sha256": timing_solution_manifest_sha256,
                "timing_solution_count": len(config.methods),
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
    assert _complete_run(
        run_dir,
        5000,
        expected_config=config,
        expected_source_sha256="frozen-source",
        expected_dataset_content_index_sha256="frozen-data",
        expected_dataset_metadata_sha256="frozen-metadata",
        expected_timing_runtime_sha256="fixture-runtime",
    )
    original_summary_csv = (run_dir / "matched_summary.csv").read_bytes()
    (run_dir / "matched_summary.csv").write_text(
        f"method,measured_repeats\ncg,{config.measured_repeats}\n",
        encoding="utf-8",
    )
    assert not _complete_run(
        run_dir,
        5000,
        expected_config=config,
        expected_source_sha256="frozen-source",
        expected_dataset_content_index_sha256="frozen-data",
        expected_dataset_metadata_sha256="frozen-metadata",
    )
    (run_dir / "matched_summary.csv").write_bytes(original_summary_csv)
    assert not _complete_run(
        run_dir,
        5000,
        expected_config=config,
        expected_source_sha256="frozen-source",
        expected_dataset_content_index_sha256="frozen-data",
        expected_dataset_metadata_sha256="frozen-metadata",
        expected_timing_runtime_sha256="different-runtime",
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
    malformed_payloads = (
        (run_dir / "system_manifest.json", []),
        (run_dir / "matched_summary.json", {}),
        (run_dir / "matched_runs.json", None),
        (timing_solution_manifest_path, "valid JSON, wrong structure"),
        (run_dir / "run_complete.json", []),
    )
    for malformed_path, malformed_payload in malformed_payloads:
        original_payload = malformed_path.read_bytes()
        malformed_path.write_text(
            json.dumps(malformed_payload),
            encoding="utf-8",
        )
        assert not _complete_run(
            run_dir,
            5000,
            expected_config=config,
            expected_source_sha256="frozen-source",
            expected_dataset_content_index_sha256="frozen-data",
            expected_dataset_metadata_sha256="frozen-metadata",
        )
        malformed_path.write_bytes(original_payload)
    original_solution_artifact = timing_solution_path.read_bytes()
    timing_solution_path.write_bytes(original_solution_artifact + b"tampered")
    assert not _complete_run(
        run_dir,
        5000,
        expected_config=config,
        expected_source_sha256="frozen-source",
        expected_dataset_content_index_sha256="frozen-data",
        expected_dataset_metadata_sha256="frozen-metadata",
    )
    timing_solution_path.write_bytes(original_solution_artifact)
    (run_dir / "matched_runs.csv").unlink()
    assert not _complete_run(
        run_dir,
        5000,
        expected_config=config,
        expected_source_sha256="frozen-source",
        expected_dataset_content_index_sha256="frozen-data",
        expected_dataset_metadata_sha256="frozen-metadata",
    )


@pytest.mark.parametrize(
    ("failure_stage", "expected_message"),
    [
        ("experiment", "experiment exploded"),
        ("artifact_validation", "wrote N=999, expected 7"),
    ],
)
def test_run_suite_records_case_traceback_and_returns_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_stage: str,
    expected_message: str,
) -> None:
    output_root = tmp_path / "suite-output"
    run_dir = output_root / "case-a"
    config = ControlledConfig(output_dir=str(run_dir))
    validation = {
        "case_id": "case-a",
        "dataset_stem": "fixture",
        "task_type": "fixture_regression",
        "scale_role": "test",
        "n_train": 7,
        "dataset_content_index_sha256": "content-sha",
        "dataset_metadata_sha256": "metadata-sha",
    }

    monkeypatch.setattr(
        suite_module,
        "build_suite_plan",
        lambda *args, **kwargs: [(validation, config)],
    )
    monkeypatch.setattr(
        suite_module,
        "_source_manifest",
        lambda: {"source_bundle_sha256": "source-sha"},
    )

    def fake_run_controlled_experiment(supplied_config):
        assert supplied_config is config
        if failure_stage == "experiment":
            raise RuntimeError("experiment exploded")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "system_manifest.json").write_text(
            json.dumps({"n_train": 999}),
            encoding="utf-8",
        )
        return run_dir

    monkeypatch.setattr(
        suite_module,
        "run_controlled_experiment",
        fake_run_controlled_experiment,
    )

    result, had_failure = suite_module.run_suite(
        {},
        profile_name="fixture-profile",
        dataset_dir_override="",
        output_root=output_root,
        execute=True,
        resume=False,
        nufft_backend_override="",
        strict_gpu_eig=False,
    )

    assert result == output_root
    assert had_failure is True
    status_rows = json.loads(
        (output_root / "suite_status.json").read_text(encoding="utf-8")
    )
    assert len(status_rows) == 1
    failed = status_rows[0]
    assert failed["case_id"] == "case-a"
    assert failed["status"] == "error"
    assert failed["error_type"] == "RuntimeError"
    assert expected_message in failed["error_message"]
    assert "Traceback (most recent call last):" in failed["traceback"]
    assert "RuntimeError" in failed["traceback"]
    assert expected_message in failed["traceback"]

    captured = capsys.readouterr()
    console = captured.out + captured.err
    assert "case-a" in console
    assert "RuntimeError" in console
    assert expected_message in console


@pytest.mark.parametrize(
    ("statuses", "had_failure", "expected"),
    [
        ([{"status": "completed"}], False, suite_module.SUITE_EXIT_OK),
        (
            [{"status": "completed_with_ineligible_methods"}],
            True,
            suite_module.SUITE_EXIT_SCIENTIFIC_FAILURE,
        ),
        (
            [{"status": "completed_with_diagnostic_errors"}],
            True,
            suite_module.SUITE_EXIT_SCIENTIFIC_FAILURE,
        ),
        ([{"status": "error"}], True, suite_module.SUITE_EXIT_EXECUTION_ERROR),
        ([{"status": "running"}], True, suite_module.SUITE_EXIT_EXECUTION_ERROR),
    ],
)
def test_suite_exit_code_distinguishes_scientific_and_execution_failures(
    tmp_path: Path,
    statuses: list[dict[str, str]],
    had_failure: bool,
    expected: int,
) -> None:
    (tmp_path / "suite_status.json").write_text(
        json.dumps(statuses),
        encoding="utf-8",
    )
    assert suite_module._suite_exit_code(
        tmp_path,
        had_failure=had_failure,
    ) == expected


def test_suite_exit_code_treats_missing_status_as_execution_error(tmp_path: Path) -> None:
    assert suite_module._suite_exit_code(
        tmp_path,
        had_failure=True,
    ) == suite_module.SUITE_EXIT_EXECUTION_ERROR
