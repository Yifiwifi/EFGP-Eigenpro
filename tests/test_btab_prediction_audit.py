from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import prediction_audit
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark import (
    ControlledConfig,
)


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
            "source_bundle_sha256": "source-hash",
            "dataset_metadata_sha256": "metadata-hash",
        },
    )


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

    zero_rows = [
        {"method": "cg", "test_rmse": 0.0},
        {"method": "default", "test_rmse": 1.0},
    ]
    prediction_audit.attach_cg_rmse_comparisons(zero_rows)
    assert zero_rows[1]["test_rmse_ratio_vs_cg"] is None
    assert zero_rows[1]["test_rmse_diff_vs_cg"] == 1.0


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


def test_prediction_audit_runs_warmup_and_one_audit_solve_and_writes_rows(
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
            "dataset_content_index_sha256": "tiny-hash",
        }
    )
    specs = [
        SimpleNamespace(label="cg", kind="cg"),
        SimpleNamespace(label="default", kind="active-eig"),
    ]
    solve_calls: list[tuple[str, bool]] = []

    def fake_run_one_method(
        supplied_system,
        cfg,
        spec,
        *,
        repeat_idx,
        order_position,
        is_warmup,
    ):
        del supplied_system, cfg, repeat_idx, order_position
        solve_calls.append((spec.label, bool(is_warmup)))
        beta = np.asarray([1.0 if spec.label == "cg" else 0.0])
        return (
            {
                "status": "converged",
                "true_relres": 1e-8,
                "iterations": 7 if spec.label == "cg" else 3,
                "build_seconds": 0.1,
                "solve_seconds": 0.2,
            },
            beta,
        )

    predict_chunk_sizes: list[int] = []

    def fake_predict(backend, data_ctx, x_chunk, beta_gpu):
        del backend, data_ctx
        predict_chunk_sizes.append(int(x_chunk.shape[0]))
        return x_chunk[:, 0] * float(beta_gpu[0])

    monkeypatch.setattr(prediction_audit, "prepare_shared_system", lambda cfg: system)
    monkeypatch.setattr(
        prediction_audit,
        "resolve_method_specs",
        lambda supplied_system, cfg: (specs, None),
    )
    monkeypatch.setattr(prediction_audit, "run_one_method", fake_run_one_method)
    monkeypatch.setattr(prediction_audit, "predict_v1", fake_predict)
    monkeypatch.setattr(
        prediction_audit,
        "system_fingerprint",
        lambda data_ctx, reg_lambda: "fixed-system",
    )

    output_dir = prediction_audit.run_prediction_audit(
        ControlledConfig(methods=("cg", "default"), measured_repeats=5),
        output_dir=tmp_path / "audit",
        prediction_chunk_size=2,
        warmup_solves=1,
    )

    assert solve_calls == [
        ("cg", True),
        ("cg", False),
        ("default", True),
        ("default", False),
    ]
    assert predict_chunk_sizes == [2, 2, 1, 2, 2, 1]
    payload = json.loads((output_dir / "prediction_audit.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["system_unchanged"] is True
    assert payload["weights_sha256"] == "weights-hash"
    assert payload["gf_sha256"] == "gf-hash"
    assert payload["rhs_sha256"] == "rhs-hash"
    assert payload["source_bundle_sha256"] == "source-hash"
    assert payload["test_subset_policy"] == "all"
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["method"] == "cg"
    assert payload["rows"][0]["test_rmse_ratio_vs_cg"] == 1.0
    assert payload["rows"][1]["test_rmse_ratio_vs_cg"] == pytest.approx(2.0)
    assert payload["rows"][1]["test_rmse_diff_vs_cg"] > 0.0
    assert payload["rows"][1]["audit_only_not_for_speed_claim"] is True

    with (output_dir / "prediction_audit.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["method"] for row in csv_rows] == ["cg", "default"]
    assert "prediction_seconds" in csv_rows[0]
