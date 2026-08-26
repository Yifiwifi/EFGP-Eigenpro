from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import prediction_audit
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    benchmark as benchmark_module,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark import (
    ControlledConfig,
    PreparedSystem,
)
from efgp_eigenpro_py.gpu import nufft_adapter


def _numpy_system() -> SimpleNamespace:
    return SimpleNamespace(
        backend=SimpleNamespace(xp=np),
        data_ctx=SimpleNamespace(),
        rhs_gpu=np.zeros(1, dtype=np.complex128),
        reg_lambda=0.1,
        setup_seconds=0.0,
        system_id="fixed-system",
        manifest={
            "dataset_stem": "tiny",
            "weights_sha256": "weights-hash",
            "gf_sha256": "gf-hash",
            "rhs_sha256": "rhs-hash",
            "rhs_storage_sha256": "rhs-storage-hash",
            "reg_lambda": 0.1,
            "source_bundle_sha256": "source-hash",
            "dataset_metadata_sha256": "metadata-hash",
        },
    )


def _single_cg_timing_inputs(
    tmp_path: Path,
) -> tuple[SimpleNamespace, dict, dict[str, dict]]:
    dataset_path = tmp_path / "tiny.npz"
    x_test = np.arange(1, 4, dtype=np.float64).reshape(-1, 1)
    np.savez(
        dataset_path,
        x_train=np.zeros((3, 1)),
        y_train=np.zeros(3),
        x_test=x_test,
        y_test=2.0 * x_test[:, 0],
    )
    system = _numpy_system()
    system.manifest.update(
        {
            "dataset_path": str(dataset_path),
            "dataset_file_size_bytes": dataset_path.stat().st_size,
            "dataset_content_index_sha256": benchmark_module._npz_content_index_sha256(
                dataset_path
            ),
            "dataset_metadata_sha256": None,
        }
    )
    timing_evidence = {
        "timing_manifest_path": str(tmp_path / "system_manifest.json"),
        "timing_manifest_sha256": "timing-manifest-hash",
        "timing_system_artifact_path": str(tmp_path / "timing_system_arrays.npz"),
        "timing_system_artifact_sha256": "system-artifact-hash",
        "solution_artifact": {
            "timing_solution_manifest_path": str(
                tmp_path / "timing_prediction_solutions.json"
            ),
            "timing_solution_manifest_sha256": "solution-manifest-hash",
            "timing_solution_artifact_path": str(
                tmp_path / "timing_prediction_solutions.npz"
            ),
            "timing_solution_artifact_sha256": "solution-artifact-hash",
            "selection_policy": "lowest converged repeat",
        },
    }
    timing_solutions = {
        "cg": {
            "method": "cg",
            "method_kind": "cg",
            "available": True,
            "selection_eligible": True,
            "timing_repeat_idx": 0,
            "timing_order_position": 0,
            "beta": {"sha256": "cg-beta-hash"},
            "beta_host": np.asarray([1.0]),
            "timing_row": {
                "status": "converged",
                "true_relres": 1e-8,
                "iterations": 7,
                "build_seconds": 0.0,
                "solve_seconds": 0.2,
            },
        }
    }
    return system, timing_evidence, timing_solutions


def test_chunked_rmse_never_predicts_more_than_declared_chunk() -> None:
    system = _numpy_system()
    x_test = np.arange(7, dtype=np.float64).reshape(-1, 1)
    y_test = 2.0 * x_test[:, 0]
    chunk_sizes: list[int] = []

    def fake_predict(backend, data_ctx, x_chunk, beta_gpu):
        del backend, data_ctx
        chunk_sizes.append(int(x_chunk.shape[0]))
        return x_chunk[:, 0] * float(np.real(beta_gpu[0]))

    rmse, seconds = prediction_audit.chunked_test_rmse(
        system,
        np.asarray([1.0]),
        x_test,
        y_test,
        chunk_size=3,
        predict_fn=fake_predict,
    )

    assert chunk_sizes == [3, 3, 1]
    assert rmse == pytest.approx(np.sqrt(np.mean(x_test[:, 0] ** 2)))
    assert seconds >= 0.0


def test_cg_rmse_ratios_and_zero_denominator_are_explicit() -> None:
    rows = [
        {"method": "cg", "test_rmse": 2.0},
        {"method": "default", "test_rmse": 2.5},
    ]
    prediction_audit.attach_cg_rmse_comparisons(rows)
    assert rows[0]["test_rmse_ratio_vs_cg"] == 1.0
    assert rows[0]["test_rmse_diff_vs_cg"] == 0.0
    assert rows[1]["test_rmse_ratio_vs_cg"] == pytest.approx(1.25)
    assert rows[1]["test_rmse_diff_vs_cg"] == pytest.approx(0.5)
    assert rows[1]["test_rmse_relative_diff_vs_cg"] == pytest.approx(0.25)
    assert rows[1]["prediction_equivalent_to_cg"] is False

    zero_rows = [
        {"method": "cg", "test_rmse": 0.0},
        {"method": "default", "test_rmse": 1.0},
    ]
    prediction_audit.attach_cg_rmse_comparisons(zero_rows)
    assert zero_rows[1]["test_rmse_ratio_vs_cg"] is None
    assert zero_rows[1]["test_rmse_diff_vs_cg"] == 1.0
    assert zero_rows[1]["prediction_equivalent_to_cg"] is False


def test_config_and_test_arrays_come_from_declared_npz(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_stem": "tiny",
                "methods": ["cg", "default"],
                "diagnostic_topk": [8, 16],
                "measured_repeats": 5,
            }
        ),
        encoding="utf-8",
    )
    cfg = prediction_audit.load_controlled_config(
        config_path,
        methods=("cg",),
    )
    assert cfg.methods == ("cg",)
    assert cfg.diagnostic_topk == (8, 16)

    dataset_path = tmp_path / "tiny.npz"
    np.savez(
        dataset_path,
        x_train=np.zeros((2, 1)),
        y_train=np.zeros(2),
        x_test=np.arange(10, dtype=np.float64).reshape(5, 2),
        y_test=np.arange(5, dtype=np.float64),
    )
    x_test, y_test, full_n = prediction_audit.load_test_arrays(
        dataset_path,
        max_test=3,
    )
    assert x_test.shape == (3, 2)
    assert y_test.tolist() == [0.0, 1.0, 2.0]
    assert full_n == 5


def test_current_test_npz_provenance_mismatch_is_rejected(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tiny.npz"
    np.savez(
        dataset_path,
        x_train=np.zeros((2, 1)),
        y_train=np.zeros(2),
        x_test=np.arange(3, dtype=np.float64).reshape(-1, 1),
        y_test=np.arange(3, dtype=np.float64),
    )
    timing_manifest = {
        "dataset_file_size_bytes": dataset_path.stat().st_size,
        "dataset_content_index_sha256": benchmark_module._npz_content_index_sha256(
            dataset_path
        ),
        "dataset_metadata_sha256": None,
    }

    # Preserve the path and array shapes while changing the actual test targets.
    np.savez(
        dataset_path,
        x_train=np.zeros((2, 1)),
        y_train=np.zeros(2),
        x_test=np.arange(3, dtype=np.float64).reshape(-1, 1),
        y_test=np.asarray([10.0, 11.0, 12.0]),
    )

    with pytest.raises(ValueError, match="content index differs"):
        prediction_audit.verify_test_dataset_provenance(
            dataset_path,
            timing_manifest,
        )


def test_missing_timing_hash_anchors_are_rejected(tmp_path: Path) -> None:
    system_artifact = tmp_path / prediction_audit._TIMING_SYSTEM_ARTIFACT
    np.savez(system_artifact, placeholder=np.asarray([1]))
    (tmp_path / "system_manifest.json").write_text(
        json.dumps(
            {
                "timing_system_artifact": system_artifact.name,
                "system_id": "fixed-system",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lacks exact system artifact checksum"):
        prediction_audit.load_timing_prediction_inputs(
            ControlledConfig(methods=("cg",), nufft_backend="none"),
            tmp_path,
        )

    solution_manifest = tmp_path / prediction_audit._TIMING_SOLUTIONS_MANIFEST
    solution_manifest.write_text(
        json.dumps({"schema_version": 1, "solutions": []}),
        encoding="utf-8",
    )
    np.savez(
        tmp_path / prediction_audit._TIMING_SOLUTIONS_ARTIFACT,
        placeholder=np.asarray([1]),
    )
    with pytest.raises(ValueError, match="lacks timing solution manifest checksum"):
        prediction_audit._load_timing_solutions(tmp_path, {})


def test_cli_defaults_to_timing_artifact_reuse_without_audit_solves() -> None:
    args = prediction_audit.build_arg_parser().parse_args(
        ["--config", "timing-case/experiment_config.json"]
    )
    assert args.timing_run_dir is None
    assert args.warmup_solves == 0
    assert args.rmse_relative_tolerance == pytest.approx(1e-3)


def test_timing_prediction_inputs_round_trip_exact_system_and_beta_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ControlledConfig(
        dataset_stem="tiny",
        dataset_dir=str(tmp_path),
        n_train=3,
        methods=("cg", "default"),
        nufft_backend="none",
    )
    backend = SimpleNamespace(xp=np, fft=np.fft, nufft_name="none")
    weights = np.asarray([1.0], dtype=np.float64)
    gf = np.asarray([2.0 + 0.0j], dtype=np.complex128)
    rhs = np.asarray([3.0 + 0.0j], dtype=np.complex128)
    data_ctx = SimpleNamespace(
        x_gpu=np.empty((0, 1)),
        y_gpu=np.empty((0,)),
        weights_gpu_nd=weights.copy(),
        weights_gpu_flat=weights.copy(),
        weights_np_flat=weights.copy(),
        gf_gpu=gf.copy(),
        rhs_gpu=rhs.copy(),
        xtxcol_gpu=np.asarray([2.0 + 0.0j]),
        x_center_gpu=np.asarray([0.5]),
        meta={
            "mtot": 1,
            "dim": 1,
            "h": 0.1,
            "weight_shape": (1,),
            "gf_shape": (1,),
            "rhs_shape": (1,),
            "nufft_tol": cfg.nufft_tol,
        },
    )
    system_id = benchmark_module.system_fingerprint(data_ctx, cfg.reg_lambda)
    components = benchmark_module.system_component_fingerprints(data_ctx)
    system = PreparedSystem(
        backend=backend,
        data_ctx=data_ctx,
        rhs_gpu=rhs.copy(),
        reg_lambda=cfg.reg_lambda,
        setup_seconds=1.0,
        system_id=system_id,
        manifest={
            "system_id": system_id,
            **components,
            "system_config_sha256": benchmark_module.system_config_fingerprint(cfg),
            "reg_lambda": cfg.reg_lambda,
            "dataset_stem": "tiny",
            "dataset_path": str(tmp_path / "tiny.npz"),
            "source_bundle_sha256": "source-hash",
            "dataset_content_index_sha256": "data-hash",
            "dataset_metadata_sha256": "metadata-hash",
            "n_train": 3,
            "M": 1,
            "mtot": 1,
            "dim": 1,
        },
    )
    benchmark_module.save_prepared_system_artifact(
        system,
        cfg,
        tmp_path / benchmark_module.TIMING_SYSTEM_ARTIFACT_FILENAME,
    )
    rows = []
    saved = []
    for method, beta_value in (("cg", 1.0), ("default", 1.001)):
        row = {
            "system_id": system_id,
            "method": method,
            "method_kind": "cg" if method == "cg" else "active-eig",
            "repeat_idx": 0,
            "order_position": 0,
            "is_warmup": False,
            "status": "converged",
            "true_relres": 1e-9,
            "iterations": 2,
            "build_seconds": 0.1,
            "solve_seconds": 0.2,
        }
        rows.append(row)
        saved.append((row, np.asarray([beta_value], dtype=np.complex128)))
    solution_payload = benchmark_module.save_timing_prediction_solutions(
        system,
        cfg,
        rows,
        saved,
        tmp_path,
    )
    (tmp_path / "system_manifest.json").write_text(
        json.dumps(system.manifest), encoding="utf-8"
    )
    (tmp_path / "run_complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "system_id": system_id,
                "source_bundle_sha256": "source-hash",
                "dataset_content_index_sha256": "data-hash",
                "dataset_metadata_sha256": "metadata-hash",
                "methods": ["cg", "default"],
                "timing_system_artifact": (
                    benchmark_module.TIMING_SYSTEM_ARTIFACT_FILENAME
                ),
                "timing_system_artifact_sha256": system.manifest[
                    "system_artifact_sha256"
                ],
                "timing_solution_artifact": (
                    benchmark_module.TIMING_SOLUTIONS_ARTIFACT_FILENAME
                ),
                "timing_solution_artifact_sha256": solution_payload[
                    "timing_solution_artifact_sha256"
                ],
                "timing_solution_manifest": (
                    benchmark_module.TIMING_SOLUTIONS_MANIFEST_FILENAME
                ),
                "timing_solution_manifest_sha256": solution_payload[
                    "timing_solution_manifest_sha256"
                ],
                "timing_solution_count": solution_payload["solution_count"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        benchmark_module,
        "build_gpu_backend_bundle",
        lambda _cfg: backend,
    )

    restored, evidence, solutions = prediction_audit.load_timing_prediction_inputs(
        cfg,
        tmp_path,
    )

    assert restored.system_id == system_id
    assert benchmark_module.system_component_fingerprints(restored.data_ctx) == components
    assert evidence["timing_system_artifact_sha256"] == system.manifest[
        "system_artifact_sha256"
    ]
    np.testing.assert_array_equal(solutions["cg"]["beta_host"], saved[0][1])
    np.testing.assert_array_equal(solutions["default"]["beta_host"], saved[1][1])

    (tmp_path / "run_complete.json").unlink()
    with pytest.raises(FileNotFoundError, match="no run_complete.json"):
        prediction_audit.load_timing_prediction_inputs(cfg, tmp_path)


def test_prediction_audit_reuses_timed_system_and_betas_without_solving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "tiny.npz"
    x_test = np.arange(1, 6, dtype=np.float64).reshape(-1, 1)
    y_test = 2.0 * x_test[:, 0]
    np.savez(
        dataset_path,
        x_train=np.zeros((3, 1)),
        y_train=np.zeros(3),
        x_test=x_test,
        y_test=y_test,
    )
    system = _numpy_system()
    system.manifest.update(
        {
            "dataset_path": str(dataset_path),
            "dataset_file_size_bytes": dataset_path.stat().st_size,
            "dataset_content_index_sha256": benchmark_module._npz_content_index_sha256(
                dataset_path
            ),
            "dataset_metadata_sha256": None,
        }
    )
    def timing_solution(method: str, beta: float, iterations: int):
        return {
            "method": method,
            "method_kind": "cg" if method == "cg" else "active-eig",
            "available": True,
            "selection_eligible": True,
            "timing_repeat_idx": 0,
            "timing_order_position": 1,
            "beta": {"sha256": f"{method}-beta-hash"},
            "beta_host": np.asarray([beta]),
            "timing_row": {
                "status": "converged",
                "true_relres": 1e-8,
                "iterations": iterations,
                "build_seconds": 0.1,
                "solve_seconds": 0.2,
            },
        }

    timing_solutions = {
        "cg": timing_solution("cg", 1.0, 7),
        "default": timing_solution("default", 0.0, 3),
    }
    timing_evidence = {
        "timing_manifest_path": str(tmp_path / "system_manifest.json"),
        "timing_manifest_sha256": "timing-manifest-hash",
        "timing_system_artifact_path": str(tmp_path / "timing_system_arrays.npz"),
        "timing_system_artifact_sha256": "system-artifact-hash",
        "solution_artifact": {
            "timing_solution_manifest_path": str(
                tmp_path / "timing_prediction_solutions.json"
            ),
            "timing_solution_manifest_sha256": "solution-manifest-hash",
            "timing_solution_artifact_path": str(
                tmp_path / "timing_prediction_solutions.npz"
            ),
            "timing_solution_artifact_sha256": "solution-artifact-hash",
            "selection_policy": "lowest converged repeat",
        },
    }

    predict_chunk_sizes: list[int] = []

    def fake_predict(
        backend,
        data_ctx,
        x_chunk,
        beta_gpu,
        *,
        return_nufft_stage=False,
        allow_cpu_fallback=True,
    ):
        del backend, data_ctx
        predict_chunk_sizes.append(int(x_chunk.shape[0]))
        prediction = x_chunk[:, 0] * float(beta_gpu[0])
        return (prediction, "cufinufft") if return_nufft_stage else prediction

    monkeypatch.setattr(
        prediction_audit,
        "load_timing_prediction_inputs",
        lambda cfg, timing_run_dir: (system, timing_evidence, timing_solutions),
    )
    monkeypatch.setattr(prediction_audit, "predict_v1", fake_predict)
    monkeypatch.setattr(
        prediction_audit,
        "system_fingerprint",
        lambda data_ctx, reg_lambda, **kwargs: "fixed-system",
    )

    output_dir = prediction_audit.run_prediction_audit(
        ControlledConfig(methods=("cg", "default"), measured_repeats=5),
        timing_run_dir=tmp_path,
        output_dir=tmp_path / "audit",
        prediction_chunk_size=2,
        warmup_solves=0,
        rmse_relative_tolerance=1.1,
    )

    assert predict_chunk_sizes == [2, 2, 1, 2, 2, 1]
    payload = json.loads((output_dir / "prediction_audit.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["audit_pass"] is True
    assert payload["system_unchanged"] is True
    assert payload["audit_rebuilt_system"] is False
    assert payload["timing_system_reused"] is True
    assert payload["timing_solutions_reused"] is True
    assert payload["timing_system_hashes_exact"] is True
    assert payload["timing_solution_hashes_verified"] is True
    assert payload["audit_solves_per_method"] == 0
    assert payload["audit_solve_count"] == 0
    assert payload["weights_sha256"] == "weights-hash"
    assert payload["gf_sha256"] == "gf-hash"
    assert payload["rhs_sha256"] == "rhs-hash"
    assert payload["rhs_storage_sha256"] == "rhs-storage-hash"
    assert payload["source_bundle_sha256"] == "source-hash"
    assert payload["test_subset_policy"] == "all"
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["method"] == "cg"
    assert payload["rows"][0]["test_rmse_ratio_vs_cg"] == 1.0
    assert payload["rows"][1]["test_rmse_ratio_vs_cg"] == pytest.approx(2.0)
    assert payload["rows"][1]["test_rmse_diff_vs_cg"] > 0.0
    assert payload["rows"][1]["audit_only_not_for_speed_claim"] is True
    assert payload["rows"][1]["timing_solution_reused"] is True
    assert payload["rows"][1]["timing_solution_sha256"] == "default-beta-hash"

    with (output_dir / "prediction_audit.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["method"] for row in csv_rows] == ["cg", "default"]
    assert "prediction_seconds" in csv_rows[0]
    completion = json.loads(
        (output_dir / prediction_audit.PREDICTION_AUDIT_COMPLETION_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert completion["methods"] == ["cg", "default"]
    assert completion["row_count"] == 2


@pytest.mark.parametrize(
    ("label", "nonfinite_value"),
    (("nan", np.nan), ("infinity", np.inf)),
)
def test_nonfinite_prediction_writes_audit_failure_instead_of_invalid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    nonfinite_value: float,
) -> None:
    system, timing_evidence, timing_solutions = _single_cg_timing_inputs(tmp_path)

    def nonfinite_predict(
        backend,
        data_ctx,
        x_chunk,
        beta_gpu,
        *,
        return_nufft_stage=False,
        allow_cpu_fallback=True,
    ):
        del backend, data_ctx, beta_gpu
        prediction = np.full(int(x_chunk.shape[0]), nonfinite_value)
        return (prediction, "cufinufft") if return_nufft_stage else prediction

    monkeypatch.setattr(
        prediction_audit,
        "load_timing_prediction_inputs",
        lambda cfg, timing_run_dir: (system, timing_evidence, timing_solutions),
    )
    monkeypatch.setattr(prediction_audit, "predict_v1", nonfinite_predict)
    monkeypatch.setattr(
        prediction_audit,
        "system_fingerprint",
        lambda data_ctx, reg_lambda, **kwargs: "fixed-system",
    )

    output_dir = prediction_audit.run_prediction_audit(
        ControlledConfig(methods=("cg",), measured_repeats=5),
        timing_run_dir=tmp_path,
        output_dir=tmp_path / f"audit_{label}",
        prediction_chunk_size=2,
    )

    payload = json.loads(
        (output_dir / prediction_audit.PREDICTION_AUDIT_JSON_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert payload["audit_pass"] is False
    assert payload["rows"][0]["test_rmse"] is None
    assert payload["rows"][0]["solve_status"].startswith(
        "prediction_error:FloatingPointError:"
    )
    assert any(
        "prediction RMSE is unavailable" in reason
        for reason in payload["audit_failure_reasons"]
    )
    completion = json.loads(
        (
            output_dir / prediction_audit.PREDICTION_AUDIT_COMPLETION_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert completion["audit_pass"] is False


def test_strict_cufinufft_prediction_rejects_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system, timing_evidence, timing_solutions = _single_cg_timing_inputs(tmp_path)

    def cpu_fallback_predict(
        backend,
        data_ctx,
        x_chunk,
        beta_gpu,
        *,
        return_nufft_stage=False,
        allow_cpu_fallback=True,
    ):
        del backend, data_ctx, beta_gpu
        assert allow_cpu_fallback is False
        prediction = 2.0 * x_chunk[:, 0]
        return (prediction, "cpu_finufft") if return_nufft_stage else prediction

    monkeypatch.setattr(
        prediction_audit,
        "load_timing_prediction_inputs",
        lambda cfg, timing_run_dir: (system, timing_evidence, timing_solutions),
    )
    monkeypatch.setattr(prediction_audit, "predict_v1", cpu_fallback_predict)
    monkeypatch.setattr(
        prediction_audit,
        "system_fingerprint",
        lambda data_ctx, reg_lambda, **kwargs: "fixed-system",
    )

    output_dir = prediction_audit.run_prediction_audit(
        ControlledConfig(methods=("cg",), measured_repeats=5),
        timing_run_dir=tmp_path,
        output_dir=tmp_path / "strict_cpu_fallback",
        strict_prediction_nufft=True,
    )

    payload = json.loads(
        (output_dir / prediction_audit.PREDICTION_AUDIT_JSON_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    row = payload["rows"][0]
    assert payload["audit_pass"] is False
    assert payload["strict_prediction_nufft"] is True
    assert payload["required_prediction_nufft_stage"] == "cufinufft"
    assert payload["observed_prediction_nufft_stages"] == ["cpu_finufft"]
    assert row["prediction_nufft_stages"] == "cpu_finufft"
    assert "used 'cpu_finufft', required 'cufinufft'" in row["solve_status"]
    assert row["test_rmse"] is None


def test_type2_strict_mode_fails_before_cpu_fallback() -> None:
    backend = SimpleNamespace(xp=np, has_nufft=False, nufft=None)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        nufft_adapter.type2_eval(
            backend,
            np.asarray([[0.0]], dtype=np.float64),
            np.asarray([1.0 + 0.0j], dtype=np.complex128),
            1,
            1,
            1e-10,
            1,
            allow_cpu_fallback=False,
        )
