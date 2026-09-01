from __future__ import annotations

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    original_krr_nystrom as original,
)


def _dense_matern32(
    x_left: np.ndarray,
    x_right: np.ndarray,
    *,
    lengthscale: float,
    variance: float,
) -> np.ndarray:
    delta = x_left[:, None, :] - x_right[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    scaled = np.sqrt(3.0) * distance / lengthscale
    return variance * (1.0 + scaled) * np.exp(-scaled)


def _config(**overrides: object) -> original.OriginalKRRNystromConfig:
    values: dict[str, object] = {
        "rank": 5,
        "seed": 7,
        "absolute_ridge": 0.07,
        "tolerance": 1e-11,
        "maxiter": 100,
        "lengthscale": 0.23,
        "kernel_variance": 1.4,
        "backend": "numpy",
        "matvec_row_chunk_size": 3,
        "matvec_column_chunk_size": 4,
        "nystrom_row_chunk_size": 4,
        "prediction_row_chunk_size": 2,
        "prediction_column_chunk_size": 3,
        "max_exact_matvec_pairs": 10**9,
        "max_prediction_pairs": 10**9,
        "max_preconditioner_bytes": 10**9,
    }
    values.update(overrides)
    return original.OriginalKRRNystromConfig(**values)


def test_exact_double_blocked_matern_matvec_matches_dense_kernel() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(size=(11, 2))
    vector = rng.normal(size=11)
    cfg = _config()

    observed = original.exact_matern32_cross_matvec(
        x,
        x,
        vector,
        lengthscale=cfg.lengthscale,
        variance=cfg.kernel_variance,
        row_chunk_size=3,
        column_chunk_size=4,
        array_module=np,
    )
    dense = _dense_matern32(
        x,
        x,
        lengthscale=cfg.lengthscale,
        variance=cfg.kernel_variance,
    )

    np.testing.assert_allclose(observed, dense @ vector, rtol=2e-14, atol=2e-14)


def test_column_nystrom_factor_matches_c_pinv_w_ct() -> None:
    rng = np.random.default_rng(5)
    x = rng.uniform(size=(13, 2))
    cfg = _config(rank=6, seed=19)
    dense = _dense_matern32(
        x,
        x,
        lengthscale=cfg.lengthscale,
        variance=cfg.kernel_variance,
    )

    preconditioner = original.build_column_nystrom_preconditioner(
        x,
        cfg,
        array_module=np,
    )
    indices = preconditioner.sample_indices
    cross = dense[:, indices]
    center = dense[np.ix_(indices, indices)]
    expected = cross @ np.linalg.pinv(center, rcond=cfg.nystrom_rcond) @ cross.T
    observed = (
        preconditioner.basis * preconditioner.eigenvalues[None, :]
    ) @ preconditioner.basis.T

    np.testing.assert_allclose(observed, expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(
        preconditioner.basis.T @ preconditioner.basis,
        np.eye(preconditioner.effective_rank),
        rtol=2e-10,
        atol=2e-10,
    )


def test_pcg_solves_full_original_krr_and_ridge_is_not_n_scaled() -> None:
    rng = np.random.default_rng(11)
    x_train = rng.uniform(size=(17, 2))
    y_train = rng.normal(size=17)
    cfg = _config(rank=7, absolute_ridge=0.09, tolerance=1e-12)
    kernel = _dense_matern32(
        x_train,
        x_train,
        lengthscale=cfg.lengthscale,
        variance=cfg.kernel_variance,
    )
    expected = np.linalg.solve(
        kernel + cfg.absolute_ridge * np.eye(x_train.shape[0]),
        y_train,
    )
    incorrectly_scaled = np.linalg.solve(
        kernel + x_train.shape[0] * cfg.absolute_ridge * np.eye(x_train.shape[0]),
        y_train,
    )

    model, diagnostics = original.fit_original_krr_nystrom_pcg(
        x_train,
        y_train,
        cfg,
        array_module=np,
    )

    np.testing.assert_allclose(model.alpha, expected, rtol=2e-10, atol=2e-10)
    assert np.linalg.norm(model.alpha - expected) < 1e-6 * np.linalg.norm(
        model.alpha - incorrectly_scaled
    )
    assert diagnostics["converged"] is True
    assert diagnostics["true_relative_residual"] <= cfg.tolerance * 1.05
    assert diagnostics["exact_matvec_count"] == diagnostics["iterations"] + 1
    assert diagnostics["kernel_pair_evaluations"] == (
        diagnostics["exact_matvec_count"] * x_train.shape[0] ** 2
    )


def test_complete_run_reports_exact_system_timing_metrics_and_citation() -> None:
    rng = np.random.default_rng(23)
    x_train = rng.uniform(size=(15, 2))
    y_train = np.sin(2.0 * x_train[:, 0]) - 0.3 * x_train[:, 1]
    x_test = rng.uniform(size=(6, 2))
    y_test = np.sin(2.0 * x_test[:, 0]) - 0.3 * x_test[:, 1]
    cfg = _config(rank=6, tolerance=1e-10)

    row = original.run_original_krr_nystrom_pcg(
        x_train,
        y_train,
        x_test,
        y_test,
        cfg,
        array_module=np,
    )
    dense_train = _dense_matern32(
        x_train,
        x_train,
        lengthscale=cfg.lengthscale,
        variance=cfg.kernel_variance,
    )
    alpha = np.linalg.solve(
        dense_train + cfg.absolute_ridge * np.eye(x_train.shape[0]),
        y_train,
    )
    expected_prediction = _dense_matern32(
        x_test,
        x_train,
        lengthscale=cfg.lengthscale,
        variance=cfg.kernel_variance,
    ) @ alpha
    expected_residual = expected_prediction - y_test

    assert row["status"] == "converged"
    assert row["method"] == "column-randomized-nystrom-pcg-original-krr"
    assert row["operator_approximation"] is False
    assert row["solved_system"] == "original_data_space_K_plus_absolute_ridge_I"
    assert row["preconditioner_sketch"].startswith("uniform_random_column")
    assert row["regularization_convention"] == "absolute"
    assert row["absolute_ridge"] == cfg.absolute_ridge
    assert row["citations"] == ["frangella2023randomized"]
    assert row["setup_seconds"] >= 0.0
    assert row["solve_seconds"] >= 0.0
    assert row["train_total_seconds"] == pytest.approx(
        row["setup_seconds"] + row["solve_seconds"]
    )
    assert row["prediction_seconds"] >= 0.0
    assert row["rmse"] == pytest.approx(
        np.sqrt(np.mean(expected_residual**2)), rel=2e-9, abs=2e-11
    )
    assert row["mae"] == pytest.approx(
        np.mean(np.abs(expected_residual)), rel=2e-9, abs=2e-11
    )
    assert np.isfinite(row["r2"])
    assert row["true_relative_residual"] <= cfg.tolerance * 1.05
    assert row["prediction_kernel_pair_evaluations"] == (
        x_train.shape[0] * x_test.shape[0]
    )
    assert "sample_indices" not in row
    assert "prediction" not in row


def test_resource_gate_rejects_10m_and_300m_by_exact_pair_cap() -> None:
    cfg = original.OriginalKRRNystromConfig(rank=128)
    for n_train in (10_000_000, 300_000_000):
        with pytest.raises(original.OriginalKRRResourceLimit) as error:
            original.preflight_original_krr_resources(n_train, 1_000, cfg)
        assert error.value.reason == "exact_matvec_pair_cap"
        assert error.value.audit["exact_matvec_pairs"] == n_train**2
        assert error.value.audit["resource_preflight_before_backend"] is True
        assert error.value.as_dict()["status"] == "resource_limit"


def test_factor_memory_gate_is_independent_and_auditable() -> None:
    n_train = 2_000
    rank = 100
    required = n_train * rank * np.dtype(np.float64).itemsize
    cfg = _config(
        rank=rank,
        max_exact_matvec_pairs=10**12,
        max_preconditioner_bytes=required - 1,
    )

    with pytest.raises(original.OriginalKRRResourceLimit) as error:
        original.preflight_original_krr_resources(n_train, 0, cfg)

    assert error.value.reason == "preconditioner_factor_memory_cap"
    assert error.value.audit["preconditioner_factor_bytes"] == required
    assert error.value.audit["max_preconditioner_bytes"] == required - 1


def test_all_in_one_gate_runs_before_backend_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    class ShapeOnly:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    def forbidden_backend(*_args: object, **_kwargs: object) -> tuple[object, str]:
        raise AssertionError("backend resolution must not run before resource preflight")

    monkeypatch.setattr(original, "_resolve_array_module", forbidden_backend)
    cfg = original.OriginalKRRNystromConfig(rank=128, backend="cupy")

    with pytest.raises(original.OriginalKRRResourceLimit) as error:
        original.run_original_krr_nystrom_pcg(
            ShapeOnly((10_000_000, 2)),
            ShapeOnly((10_000_000,)),
            ShapeOnly((1_000, 2)),
            ShapeOnly((1_000,)),
            cfg,
        )

    assert error.value.reason == "exact_matvec_pair_cap"


def test_fp64_is_required_for_original_krr_auditability() -> None:
    with pytest.raises(ValueError, match="precision='fp64'"):
        original.preflight_original_krr_resources(
            20,
            5,
            _config(precision="fp32"),
        )
