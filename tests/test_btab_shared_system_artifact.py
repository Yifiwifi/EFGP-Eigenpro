from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    benchmark as benchmark_module,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    suite as suite_module,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark import (
    ControlledConfig,
    PreparedSystem,
    load_prepared_system_artifact,
    save_prepared_system_artifact,
    system_component_fingerprints,
    system_config_fingerprint,
    system_config_payload,
    system_fingerprint,
)


def _config(*, output_dir: str = "", box_budget: int = 4096) -> ControlledConfig:
    return ControlledConfig(
        dataset_stem="CaseSensitive_Winnebago_10M",
        dataset_dir="/not-part-of-the-system-key",
        n_train=10_000_000,
        subset_mode="prefix",
        methods=("cg", "default", "full-eig"),
        box_budget=box_budget,
        inverse_max_size=1024,
        rank=32,
        measured_repeats=5,
        output_dir=output_dir,
        nufft_backend="none",
    )


_BUILD_RUNTIME = {
    "device_name": "fixture-build-gpu",
    "device_id": 0,
    "supports_fp64": True,
    "supports_complex128": True,
    "cupy_version": "fixture-build-cupy",
    "cuda_runtime_version": 12010,
    "cuda_driver_version": 12020,
    "compute_capability": "8.0",
    "nufft_backend_resolved": "none",
    "timing_runtime_sha256": "fixture-build-runtime-sha",
}


def _system(
    cfg: ControlledConfig,
    *,
    distinct_solve_rhs: bool = False,
) -> PreparedSystem:
    mtot = 3
    weights = np.asarray([0.25, 1.0, 0.25], dtype=np.float64)
    xtxcol = np.asarray([0.1, 0.2, 3.0, 0.2, 0.1], dtype=np.complex128)
    gf = np.ascontiguousarray(np.fft.fftn(xtxcol))
    rhs = np.asarray([1.0 + 0.5j, -2.0j, 3.0 - 0.25j], dtype=np.complex128)
    solve_rhs = rhs.astype(np.complex64) if distinct_solve_rhs else rhs.copy()
    data_ctx = SimpleNamespace(
        x_gpu=np.empty((0, 1), dtype=np.float64),
        y_gpu=np.empty((0,), dtype=np.float64),
        weights_gpu_nd=weights.reshape((mtot,)),
        weights_gpu_flat=weights,
        weights_np_flat=weights.copy(),
        rhs_gpu=rhs,
        gf_gpu=gf,
        xtxcol_gpu=xtxcol,
        x_center_gpu=np.asarray([42.0], dtype=np.float64),
        meta={
            "mtot": mtot,
            "dim": 1,
            "h": 0.1,
            "weight_shape": (mtot,),
            "gf_shape": gf.shape,
            "rhs_shape": rhs.shape,
            "nufft_tol": cfg.nufft_tol,
            "nufft_stage": "fixture",
            "debug_finite_checks": False,
        },
    )
    system_id = system_fingerprint(
        data_ctx,
        cfg.reg_lambda,
        solve_rhs_gpu=solve_rhs,
    )
    return PreparedSystem(
        backend=SimpleNamespace(
            xp=np,
            fft=np.fft,
            nufft_name="none",
            device_name=_BUILD_RUNTIME["device_name"],
            device_id=_BUILD_RUNTIME["device_id"],
            supports_fp64=True,
            supports_complex128=True,
        ),
        data_ctx=data_ctx,
        rhs_gpu=solve_rhs,
        reg_lambda=cfg.reg_lambda,
        setup_seconds=12.5,
        system_id=system_id,
        manifest={
            "system_id": system_id,
            **system_component_fingerprints(
                data_ctx,
                solve_rhs_gpu=solve_rhs,
            ),
            "system_config": system_config_payload(cfg),
            "system_config_sha256": system_config_fingerprint(cfg),
            "source_bundle_sha256": "source-sha",
            "dataset_content_index_sha256": "dataset-sha",
            "dataset_metadata_sha256": "metadata-sha",
            "n_train": 10_000_000,
            "M": mtot,
            "mtot": mtot,
            "dim": 1,
            "reg_lambda": cfg.reg_lambda,
            **_BUILD_RUNTIME,
            "system_build_runtime": dict(_BUILD_RUNTIME),
            "current_timing_runtime": dict(_BUILD_RUNTIME),
            "setup_timing_source": "measured_during_system_build",
            "setup_inclusive_timing_eligible": True,
        },
    )


def _numpy_backend_with_runtime(
    *,
    device_name: str,
    compute_capability: tuple[int, int],
) -> SimpleNamespace:
    runtime = SimpleNamespace(
        runtimeGetVersion=lambda: 13010,
        driverGetVersion=lambda: 13020,
        getDeviceProperties=lambda _device_id: {
            "major": compute_capability[0],
            "minor": compute_capability[1],
        },
    )
    xp = SimpleNamespace(
        __version__="fixture-current-cupy",
        cuda=SimpleNamespace(runtime=runtime),
        asarray=np.asarray,
        ascontiguousarray=np.ascontiguousarray,
        empty=np.empty,
        float64=np.float64,
    )
    return SimpleNamespace(
        xp=xp,
        fft=np.fft,
        nufft_name="none",
        device_name=device_name,
        device_id=0,
        supports_fp64=True,
        supports_complex128=True,
    )


def test_system_config_key_preserves_case_and_excludes_box_settings() -> None:
    base = _config(box_budget=4096)
    other_budget = _config(box_budget=16384)
    other_kernel = ControlledConfig(
        **{**base.__dict__, "lengthscale": base.lengthscale * 2.0}
    )

    assert system_config_payload(base)["dataset_stem"] == "CaseSensitive_Winnebago_10M"
    assert system_config_fingerprint(base) == system_config_fingerprint(other_budget)
    assert system_config_fingerprint(base) != system_config_fingerprint(other_kernel)


def test_prepared_system_artifact_round_trips_exact_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    original = _system(cfg, distinct_solve_rhs=True)
    original.manifest["timing_solution_artifact_sha256"] = "stale-previous-case"
    artifact = save_prepared_system_artifact(
        original, cfg, tmp_path / "shared-system.npz"
    )
    monkeypatch.setattr(
        benchmark_module,
        "build_gpu_backend_bundle",
        lambda _config: _numpy_backend_with_runtime(
            device_name="fixture-current-gpu",
            compute_capability=(9, 0),
        ),
    )

    restored = load_prepared_system_artifact(
        _config(box_budget=16384),
        artifact,
        expected_source_sha256="source-sha",
        expected_dataset_content_index_sha256="dataset-sha",
        expected_dataset_metadata_sha256="metadata-sha",
    )

    assert restored.system_id == original.system_id
    restored_components = system_component_fingerprints(
        restored.data_ctx,
        solve_rhs_gpu=restored.rhs_gpu,
    )
    original_components = system_component_fingerprints(
        original.data_ctx,
        solve_rhs_gpu=original.rhs_gpu,
    )
    assert restored_components == original_components
    assert restored_components["rhs_sha256"] != restored_components[
        "rhs_storage_sha256"
    ]
    np.testing.assert_array_equal(
        restored.data_ctx.weights_gpu_flat, original.data_ctx.weights_gpu_flat
    )
    np.testing.assert_array_equal(restored.data_ctx.gf_gpu, original.data_ctx.gf_gpu)
    np.testing.assert_array_equal(restored.data_ctx.rhs_gpu, original.data_ctx.rhs_gpu)
    np.testing.assert_array_equal(restored.rhs_gpu, original.rhs_gpu)
    assert restored.manifest["system_artifact_loaded"] is True
    assert restored.manifest["prepared_system_loaded_from_artifact"] is True
    assert restored.manifest["prepared_system_origin_artifact_path"] == str(
        artifact.resolve()
    )
    assert restored.manifest["system_build_runtime"] == _BUILD_RUNTIME
    assert restored.manifest["system_build_runtime"]["device_name"] == (
        "fixture-build-gpu"
    )
    assert restored.manifest["current_timing_runtime"]["device_name"] == (
        "fixture-current-gpu"
    )
    assert restored.manifest["current_timing_runtime"]["compute_capability"] == "9.0"
    assert restored.manifest["device_name"] == "fixture-current-gpu"
    assert restored.manifest["compute_capability"] == "9.0"
    assert restored.manifest["setup_timing_source"] == (
        "reused_from_prepared_system_artifact"
    )
    assert restored.manifest["setup_inclusive_timing_eligible"] is False
    assert "timing_solution_artifact_sha256" not in restored.manifest

    wrong_kernel = ControlledConfig(
        **{**cfg.__dict__, "lengthscale": cfg.lengthscale * 2.0}
    )
    with pytest.raises(ValueError, match="config does not match"):
        load_prepared_system_artifact(wrong_kernel, artifact)


@pytest.mark.parametrize(
    ("manifest_field", "bad_value", "message"),
    [
        ("system_config_sha256", None, "config hash"),
        ("rhs_sha256", "not-the-solve-rhs", "component hashes"),
        ("rhs_storage_sha256", "not-the-storage-rhs", "component hashes"),
    ],
)
def test_prepared_system_validation_fails_closed_on_missing_or_stale_hashes(
    tmp_path: Path,
    manifest_field: str,
    bad_value: str | None,
    message: str,
) -> None:
    cfg = _config()
    system = _system(cfg, distinct_solve_rhs=True)
    if bad_value is None:
        system.manifest.pop(manifest_field)
    else:
        system.manifest[manifest_field] = bad_value

    with pytest.raises(ValueError, match=message):
        save_prepared_system_artifact(
            system,
            cfg,
            tmp_path / f"invalid-{manifest_field}.npz",
        )


def test_artifact_rejects_nested_provenance_that_differs_from_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    artifact = save_prepared_system_artifact(
        _system(cfg, distinct_solve_rhs=True),
        cfg,
        tmp_path / "nested-provenance.npz",
    )
    with np.load(artifact, allow_pickle=False) as loaded:
        arrays = {
            name: np.ascontiguousarray(loaded[name])
            for name in loaded.files
            if name != "artifact_manifest_json"
        }
        artifact_manifest = json.loads(
            np.asarray(loaded["artifact_manifest_json"], dtype=np.uint8)
            .tobytes()
            .decode("utf-8")
        )
    artifact_manifest["system_manifest"]["source_bundle_sha256"] = "nested-stale"
    arrays["artifact_manifest_json"] = np.frombuffer(
        json.dumps(
            artifact_manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8"),
        dtype=np.uint8,
    )
    with artifact.open("wb") as handle:
        np.savez(handle, **arrays)
    monkeypatch.setattr(
        benchmark_module,
        "build_gpu_backend_bundle",
        lambda _config: _numpy_backend_with_runtime(
            device_name="fixture-current-gpu",
            compute_capability=(9, 0),
        ),
    )

    with pytest.raises(ValueError, match="provenance envelope"):
        load_prepared_system_artifact(
            cfg,
            artifact,
            expected_source_sha256="source-sha",
            expected_dataset_content_index_sha256="dataset-sha",
            expected_dataset_metadata_sha256="metadata-sha",
        )


def test_prepared_system_validation_hashes_the_actual_solve_rhs(
    tmp_path: Path,
) -> None:
    cfg = _config()
    system = _system(cfg, distinct_solve_rhs=True)
    system.rhs_gpu = np.ascontiguousarray(system.rhs_gpu.copy())
    system.rhs_gpu[0] += np.complex64(0.125 + 0.0j)

    with pytest.raises(ValueError, match="recorded system_id"):
        save_prepared_system_artifact(
            system,
            cfg,
            tmp_path / "mutated-solve-rhs.npz",
        )


def test_suite_reuses_one_system_and_reruns_mismatched_resumed_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "suite"
    configs = [
        _config(
            output_dir=str(output_root / f"budget-{budget}"),
            box_budget=budget,
        )
        for budget in (4096, 8192, 16384)
    ]
    validations = [
        {
            "case_id": f"budget-{budget}",
            "dataset_family": "Winnebago",
            "dataset_stem": config.dataset_stem,
            "task_type": "regression",
            "scale_role": "fixed-system box-budget ablation",
            "n_train": 10_000_000,
            "dataset_content_index_sha256": "dataset-sha",
            "dataset_metadata_sha256": "metadata-sha",
        }
        for budget, config in zip((4096, 8192, 16384), configs)
    ]
    plan = list(zip(validations, configs))
    prepared = _system(configs[0])
    prepare_calls: list[ControlledConfig] = []
    experiment_systems: list[PreparedSystem] = []

    for config in configs:
        run_dir = Path(config.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "system_manifest.json").write_text(
            json.dumps(
                {
                    "n_train": 10_000_000,
                    "system_id": "old-nondeterministic-system",
                    "weights_sha256": "old-weights",
                    "gf_sha256": "old-gf",
                    "rhs_sha256": "old-rhs",
                    "rhs_storage_sha256": "old-storage-rhs",
                    "reg_lambda": config.reg_lambda,
                    "device_name": "old-gpu",
                    "compute_capability": "0.0",
                    "timing_runtime_sha256": "old-runtime-sha",
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(suite_module, "build_suite_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        suite_module,
        "_source_manifest",
        lambda: {"source_bundle_sha256": "source-sha"},
    )
    monkeypatch.setattr(suite_module, "_complete_run", lambda *args, **kwargs: True)

    def fake_prepare(config: ControlledConfig) -> PreparedSystem:
        prepare_calls.append(config)
        return prepared

    def fake_experiment(
        config: ControlledConfig,
        *,
        prepared_system: PreparedSystem | None = None,
    ) -> Path:
        assert prepared_system is prepared
        experiment_systems.append(prepared_system)
        run_dir = Path(config.output_dir)
        manifest = dict(prepared_system.manifest)
        manifest["n_train"] = 10_000_000
        (run_dir / "system_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (run_dir / "matched_summary.json").write_text(
            json.dumps(
                [
                    {
                        "method": "cg",
                        "performance_claim_eligible": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
        return run_dir

    monkeypatch.setattr(suite_module, "prepare_shared_system", fake_prepare)
    monkeypatch.setattr(suite_module, "run_controlled_experiment", fake_experiment)

    result, had_failure = suite_module.run_suite(
        {},
        profile_name="box-budget",
        dataset_dir_override="",
        output_root=output_root,
        execute=True,
        resume=True,
        nufft_backend_override="",
        strict_gpu_eig=False,
    )

    assert result == output_root
    assert had_failure is False
    assert len(prepare_calls) == 1
    assert experiment_systems == [prepared, prepared, prepared]
    statuses = json.loads((output_root / "suite_status.json").read_text())
    assert {row["shared_system_id"] for row in statuses} == {prepared.system_id}
    assert all(row["shared_system_exact_match"] is True for row in statuses)
    groups = json.loads((output_root / "shared_system_groups.json").read_text())
    assert len(groups) == 1
    assert groups[0]["all_cases_exact_match"] is True
    assert groups[0]["observed_verified_case_count"] == 3
    assert groups[0]["system_id"] == prepared.system_id
    assert groups[0]["weights_sha256"] == prepared.manifest["weights_sha256"]
    assert groups[0]["gf_sha256"] == prepared.manifest["gf_sha256"]
    assert groups[0]["rhs_sha256"] == prepared.manifest["rhs_sha256"]
    assert groups[0]["rhs_storage_sha256"] == prepared.manifest[
        "rhs_storage_sha256"
    ]
    assert groups[0]["device_name"] == _BUILD_RUNTIME["device_name"]
    assert groups[0]["compute_capability"] == _BUILD_RUNTIME[
        "compute_capability"
    ]
    assert groups[0]["timing_runtime_sha256"] == _BUILD_RUNTIME[
        "timing_runtime_sha256"
    ]
