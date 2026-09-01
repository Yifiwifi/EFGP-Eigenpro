from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    literature_baselines as baselines,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.literature_baselines import (
    FalkonKRRConfig,
    MaternRFFRidgeConfig,
    NativeFalkonKRRConfig,
    falkon_penalty_from_absolute_ridge,
    fit_matern_rff_ridge,
    fit_native_falkon_krr,
    matern_rff_features,
    run_falkon_krr,
    run_matern_rff_ridge,
    run_native_falkon_krr,
    sample_matern_rff_parameters,
)


def _matern32(x: np.ndarray, z: np.ndarray, lengthscale: float) -> np.ndarray:
    squared = (
        np.sum(x * x, axis=1).reshape(-1, 1)
        + np.sum(z * z, axis=1).reshape(1, -1)
        - 2.0 * (x @ z.T)
    )
    radius = np.sqrt(np.maximum(squared, 0.0))
    scaled = math.sqrt(3.0) * radius / lengthscale
    return (1.0 + scaled) * np.exp(-scaled)


def test_falkon_absolute_ridge_conversion_is_exact() -> None:
    assert falkon_penalty_from_absolute_ridge(0.1, 300_000_000) == pytest.approx(
        1.0 / 3_000_000_000
    )
    assert 300_000_000 * falkon_penalty_from_absolute_ridge(
        0.1, 300_000_000
    ) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="absolute_ridge"):
        falkon_penalty_from_absolute_ridge(0.0, 10)
    with pytest.raises(ValueError, match="n_train"):
        falkon_penalty_from_absolute_ridge(0.1, 0)


def test_native_falkon_streamed_cg_matches_dense_restricted_krr() -> None:
    rng = np.random.default_rng(12)
    x = rng.uniform(size=(53, 2))
    y = np.sin(4.0 * x[:, 0]) - 0.3 * np.cos(3.0 * x[:, 1])
    x_test = rng.uniform(size=(17, 2))
    y_test = np.sin(4.0 * x_test[:, 0]) - 0.3 * np.cos(3.0 * x_test[:, 1])
    cfg = NativeFalkonKRRConfig(
        nystrom_centers=9,
        maxiter=50,
        tolerance=1e-11,
        seed=7,
        lengthscale=0.3,
        nu=1.5,
        absolute_ridge=0.07,
        train_chunk_size=8,
        prediction_chunk_size=5,
        precision="fp64",
        backend="numpy",
        preconditioner_jitter=1e-12,
    )

    model, diagnostics = fit_native_falkon_krr(x, y, cfg, array_module=np)
    centers = np.asarray(model.centers)
    cross = _matern32(x, centers, cfg.lengthscale)
    center_kernel = _matern32(centers, centers, cfg.lengthscale)
    expected = np.linalg.solve(
        cross.T @ cross + cfg.absolute_ridge * center_kernel,
        cross.T @ y,
    )

    np.testing.assert_allclose(model.coefficients, expected, rtol=2e-8, atol=2e-9)
    assert diagnostics["converged"] is True
    assert diagnostics["iterations"] <= cfg.maxiter
    assert diagnostics["relative_residual"] <= cfg.tolerance
    assert diagnostics["matvec_passes"] == diagnostics["iterations"]
    assert diagnostics["streamed_kernel_chunks"] == diagnostics[
        "chunks_per_data_pass"
    ] * (1 + diagnostics["iterations"])

    row = run_native_falkon_krr(x, y, x_test, y_test, cfg, array_module=np)
    expected_prediction = _matern32(x_test, centers, cfg.lengthscale) @ expected
    expected_rmse = float(np.sqrt(np.mean((expected_prediction - y_test) ** 2)))
    assert row["implementation"] == "native_falkon_algorithm"
    assert row["official_falkon_package"] is False
    assert row["status"] == "converged"
    assert row["test_rmse"] == pytest.approx(expected_rmse, rel=2e-8, abs=2e-9)
    assert row["rmse"] == row["test_rmse"]
    assert math.isfinite(row["test_mae"])
    assert math.isfinite(row["test_r2"])
    assert row["train_total_seconds"] == pytest.approx(
        row["setup_seconds"]
        + row["solver_build_seconds"]
        + row["iterative_solve_seconds"]
    )

    maxiter_row = run_native_falkon_krr(
        x,
        y,
        x_test,
        y_test,
        replace(cfg, maxiter=1, tolerance=1e-15),
        array_module=np,
    )
    assert maxiter_row["converged"] is False
    assert maxiter_row["iterations"] == 1
    assert maxiter_row["status"] == "maxiter"


def test_matern_rff_streamed_statistics_match_dense_features() -> None:
    rng = np.random.default_rng(31)
    x = rng.normal(size=(37, 3))
    y = 0.5 * x[:, 0] - x[:, 1] ** 2 + 0.2 * x[:, 2]
    cfg = MaternRFFRidgeConfig(
        num_features=13,
        seed=19,
        lengthscale=0.7,
        nu=1.5,
        kernel_variance=1.3,
        absolute_ridge=0.2,
        train_chunk_size=6,
        prediction_chunk_size=4,
        precision="fp64",
        backend="numpy",
    )

    model, diagnostics = fit_matern_rff_ridge(x, y, cfg, array_module=np)
    frequencies, phases = sample_matern_rff_parameters(x.shape[1], cfg)
    dense_features = matern_rff_features(
        x,
        frequencies,
        phases,
        kernel_variance=cfg.kernel_variance,
        array_module=np,
        dtype=np.float64,
    )
    dense_coefficients = np.linalg.solve(
        dense_features.T @ dense_features
        + cfg.absolute_ridge * np.eye(cfg.num_features),
        dense_features.T @ y,
    )

    np.testing.assert_array_equal(model.frequencies, frequencies)
    np.testing.assert_array_equal(model.phases, phases)
    np.testing.assert_allclose(
        model.coefficients, dense_coefficients, rtol=2e-12, atol=2e-12
    )
    assert diagnostics["train_chunks"] == math.ceil(len(x) / cfg.train_chunk_size)


def test_matern_rff_student_t_spectrum_matches_matern32_kernel() -> None:
    cfg = MaternRFFRidgeConfig(
        num_features=100_000,
        seed=2,
        lengthscale=0.7,
        nu=1.5,
        backend="numpy",
    )
    frequencies, _ = sample_matern_rff_parameters(2, cfg)
    displacement = np.asarray([0.4, 0.0])
    estimated = float(np.cos(frequencies @ displacement).mean())
    scaled = math.sqrt(3.0) * 0.4 / cfg.lengthscale
    exact = (1.0 + scaled) * math.exp(-scaled)
    assert estimated == pytest.approx(exact, abs=0.01)


def test_matern_rff_run_is_deterministic_and_reports_streamed_metrics() -> None:
    rng = np.random.default_rng(44)
    x = rng.uniform(size=(42, 2))
    y = np.sin(2.0 * np.pi * x[:, 0]) + 0.1 * x[:, 1]
    x_test = rng.uniform(size=(15, 2))
    y_test = np.sin(2.0 * np.pi * x_test[:, 0]) + 0.1 * x_test[:, 1]
    cfg = MaternRFFRidgeConfig(
        num_features=17,
        seed=5,
        lengthscale=0.25,
        nu=1.5,
        absolute_ridge=0.1,
        train_chunk_size=9,
        prediction_chunk_size=4,
        backend="numpy",
    )

    first = run_matern_rff_ridge(x, y, x_test, y_test, cfg, array_module=np)
    second = run_matern_rff_ridge(x, y, x_test, y_test, cfg, array_module=np)

    assert first["test_rmse"] == pytest.approx(second["test_rmse"], abs=1e-14)
    assert first["test_mae"] == pytest.approx(second["test_mae"], abs=1e-14)
    assert first["test_r2"] == pytest.approx(second["test_r2"], abs=1e-14)
    assert first["prediction_chunks"] == 4
    assert first["train_chunks"] == 5
    assert first["feature_system"] == "Phi.T@Phi + absolute_ridge*I"


class _FakeFalkonOptions:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeMaternKernel:
    def __init__(self, *, sigma: float, nu: float, opt: object) -> None:
        self.sigma = sigma
        self.nu = nu
        self.options = opt


class _FakeGaussianKernel:
    def __init__(self, *, sigma: float, opt: object) -> None:
        self.sigma = sigma
        self.options = opt


class _FakeOfficialFalkon:
    instances: list["_FakeOfficialFalkon"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.predict_calls = 0
        type(self).instances.append(self)

    def fit(self, x: object, y: object) -> "_FakeOfficialFalkon":
        torch = pytest.importorskip("torch")
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().reshape(-1)
        count = int(self.kwargs["M"])
        self.centers = x_np[:count]
        kernel = self.kwargs["kernel"]
        cross = _matern32(x_np, self.centers, kernel.sigma)
        center_kernel = _matern32(self.centers, self.centers, kernel.sigma)
        absolute_ridge = len(x_np) * float(self.kwargs["penalty"])
        self.coefficients = np.linalg.solve(
            cross.T @ cross + absolute_ridge * center_kernel,
            cross.T @ y_np,
        )
        self.torch = torch
        return self

    def predict(self, x: object) -> object:
        self.predict_calls += 1
        x_np = x.detach().cpu().numpy()
        kernel = self.kwargs["kernel"]
        prediction = _matern32(x_np, self.centers, kernel.sigma) @ self.coefficients
        return self.torch.from_numpy(prediction.reshape(-1, 1))


def test_official_falkon_adapter_passes_converted_penalty_and_chunks_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    _FakeOfficialFalkon.instances.clear()
    fake_module = SimpleNamespace(
        __version__="test-double",
        FalkonOptions=_FakeFalkonOptions,
        Falkon=_FakeOfficialFalkon,
        kernels=SimpleNamespace(
            MaternKernel=_FakeMaternKernel,
            GaussianKernel=_FakeGaussianKernel,
        ),
    )
    monkeypatch.setattr(baselines, "_load_falkon_module", lambda: fake_module)

    rng = np.random.default_rng(6)
    x = rng.uniform(size=(29, 2)).astype(np.float64)
    y = (np.sin(x[:, 0]) + x[:, 1]).astype(np.float64)
    x_test = rng.uniform(size=(11, 2)).astype(np.float64)
    y_test = (np.sin(x_test[:, 0]) + x_test[:, 1]).astype(np.float64)
    cfg = FalkonKRRConfig(
        nystrom_centers=7,
        maxiter=4,
        lengthscale=0.4,
        absolute_ridge=0.3,
        prediction_chunk_size=3,
        use_cpu=True,
        precision="fp64",
    )

    row = run_falkon_krr(x, y, x_test, y_test, cfg)
    instance = _FakeOfficialFalkon.instances[-1]
    assert instance.kwargs["penalty"] == pytest.approx(0.3 / len(x))
    assert row["falkon_penalty"] == pytest.approx(0.3 / len(x))
    assert row["falkon_penalty_identity"] == "penalty=absolute_ridge/n_train"
    assert instance.predict_calls == math.ceil(len(x_test) / 3)
    assert row["prediction_chunks"] == instance.predict_calls
    assert row["train_input_copied"] is False
    assert math.isfinite(row["test_rmse"])
    assert math.isfinite(row["test_mae"])
    assert math.isfinite(row["test_r2"])
