from __future__ import annotations

import math
import os

import numpy as np
import pytest
import scipy.sparse

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    structured_kernel_interpolation as ski,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.structured_kernel_interpolation import (
    BTTBMaternOperator2D,
    StructuredKernelInterpolationConfig,
    accumulate_interpolation_normal_equations,
    accumulate_interpolation_normal_equations_from_chunks,
    build_ski_grid_2d,
    fit_structured_kernel_interpolation,
    interpolation_rows,
    run_structured_kernel_interpolation,
)


def _explicit_interpolation_matrix(grid, x: np.ndarray, mode: str) -> np.ndarray:
    indices, weights, _cell_ids, _tx, _ty = interpolation_rows(
        grid, x, interpolation=mode
    )
    matrix = np.zeros((len(x), grid.size), dtype=np.float64)
    np.put_along_axis(matrix, indices, weights, axis=1)
    return matrix


def _small_problem(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(0.08, 0.92, size=(24, 2))
    x_test = rng.uniform(0.08, 0.92, size=(11, 2))
    y_train = np.sin(2.0 * np.pi * x_train[:, 0]) + 0.4 * np.cos(
        2.0 * np.pi * x_train[:, 1]
    )
    return x_train, y_train, x_test


def test_cg_can_stop_on_declared_original_system_residual() -> None:
    diagonal = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    rhs = np.ones(3, dtype=np.float64)

    _solution, diagnostics = ski._conjugate_gradient(
        lambda vector: diagonal * vector,
        rhs,
        tolerance=1e-15,
        maxiter=100,
        preconditioner=None,
        external_residual=lambda _candidate: 5e-4,
        external_tolerance=1e-3,
        external_check_interval=25,
    )

    assert diagnostics["iterations"] == 1
    assert diagnostics["stopped_by_external_criterion"] is True
    assert diagnostics["external_residual_checks"] == 1
    assert diagnostics["external_relative_residual"] == pytest.approx(5e-4)


def test_grid_covers_declared_domain_with_two_cubic_padding_points() -> None:
    grid = build_ski_grid_2d(
        ((0.0, 5.0 / 7.0), (0.0, 1.0)),
        spacing=1.0 / 128.0,
        padding_points=2,
    )

    assert grid.shape == (97, 133)
    assert grid.size == 12_901
    assert grid.x_start == pytest.approx(-2.0 / 128.0)
    assert grid.y_start == pytest.approx(-2.0 / 128.0)
    assert grid.x_stop >= 5.0 / 7.0 + 2.0 / 128.0
    assert grid.y_stop >= 1.0 + 2.0 / 128.0


def test_bttb_fft_matvec_matches_dense_isotropic_matern() -> None:
    grid = build_ski_grid_2d(((-0.1, 0.8), (0.2, 1.0)), spacing=0.3, padding_points=2)
    operator = BTTBMaternOperator2D.build(grid, lengthscale=0.27, nu=1.5, variance=1.3)
    rng = np.random.default_rng(41)
    vector = rng.normal(size=grid.size)

    expected = operator.dense_matrix() @ vector
    observed = operator.matvec(vector)

    np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=2e-13)
    # This is a genuinely two-dimensional radial kernel.  A product of the two
    # one-dimensional Matérn values would give a different off-axis entry.
    x_delta = grid.spacing
    radial = math.sqrt(2.0) * x_delta
    scaled_radial = math.sqrt(3.0) * radial / 0.27
    isotropic = 1.3 * (1.0 + scaled_radial) * math.exp(-scaled_radial)
    scaled_axis = math.sqrt(3.0) * x_delta / 0.27
    product = 1.3 * ((1.0 + scaled_axis) * math.exp(-scaled_axis)) ** 2
    assert isotropic != pytest.approx(product, rel=1e-3)


@pytest.mark.parametrize(
    ("mode", "expected_nonzeros", "expected_updates"),
    [("linear", 4, 13), ("cubic", 16, 65)],
)
def test_streamed_moment_statistics_match_explicit_w(
    mode: str, expected_nonzeros: int, expected_updates: int
) -> None:
    rng = np.random.default_rng(123)
    grid = build_ski_grid_2d(((0.0, 1.0), (0.0, 1.0)), spacing=0.25)
    x = rng.uniform(0.01, 0.99, size=(37, 2))
    y = rng.normal(size=len(x))
    explicit_w = _explicit_interpolation_matrix(grid, x, mode)

    normal = accumulate_interpolation_normal_equations(
        x, y, grid, interpolation=mode, chunk_size=6
    )

    np.testing.assert_allclose(
        normal.dense_matrix(), explicit_w.T @ explicit_w, rtol=2e-13, atol=3e-13
    )
    np.testing.assert_allclose(normal.rhs, explicit_w.T @ y, rtol=2e-13, atol=3e-13)
    assert normal.n_rows == len(x)
    assert normal.interpolation_nonzeros_per_row == expected_nonzeros
    assert normal.moment_updates_per_row == expected_updates


def test_cupy_bincount_reduction_logic_matches_numpy_reference() -> None:
    rng = np.random.default_rng(83)
    grid = build_ski_grid_2d(((0.0, 1.0), (0.0, 1.0)), spacing=0.25)
    x = rng.uniform(0.01, 0.99, size=(43, 2))
    y = rng.normal(size=len(x))
    expected = accumulate_interpolation_normal_equations(
        x, y, grid, interpolation="linear", chunk_size=8
    )

    # NumPy/SciPy implement the same array and sparse APIs used here.  This
    # executes the production reduction/CSR assembly deterministically without
    # requiring a CUDA runner; the opt-in test below covers real CuPy kernels.
    observed = ski._accumulate_cupy_linear_normal_equations(
        x,
        y,
        grid,
        chunk_size=7,
        array_module=np,
        sparse_module=scipy.sparse,
    )

    np.testing.assert_allclose(
        observed.sparse_matrix.toarray(), expected.dense_matrix(), atol=3e-13
    )
    np.testing.assert_allclose(observed.rhs, expected.rhs, atol=3e-13)
    assert observed.trace == pytest.approx(expected.trace, abs=3e-13)
    assert observed.n_rows == len(x)


def test_cupy_backend_control_flow_matches_numpy_array_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ski, "_load_cupy_backend", lambda: (np, scipy.sparse))
    monkeypatch.setattr(ski, "_synchronize_cupy", lambda _array_module: None)
    x_train, y_train, x_test = _small_problem(seed=89)
    common = {
        "interpolation": "linear",
        "grid_spacing": 0.25,
        "lengthscale": 0.3,
        "absolute_ridge": 0.4,
        "train_chunk_size": 5,
        "prediction_chunk_size": 4,
        "cg_tolerance": 1e-8,
        "cg_maxiter": 1_000,
        "cg_preconditioner": "circulant_density",
    }
    expected_model = fit_structured_kernel_interpolation(
        x_train, y_train, StructuredKernelInterpolationConfig(**common)
    )
    observed_model = fit_structured_kernel_interpolation(
        x_train,
        y_train,
        StructuredKernelInterpolationConfig(**common, backend="cupy"),
    )

    np.testing.assert_allclose(
        observed_model.predict(x_test),
        expected_model.predict(x_test),
        rtol=3e-8,
        atol=2e-9,
    )
    diagnostics = observed_model.fit_diagnostics
    assert diagnostics["backend"] == "cupy"
    assert diagnostics["moment_reduction"].endswith("cupy_bincount")
    assert diagnostics["converged"] is True


@pytest.mark.parametrize("mode", ["linear", "cubic"])
def test_chunk_iterator_statistics_are_chunk_boundary_invariant(mode: str) -> None:
    rng = np.random.default_rng(7)
    grid = build_ski_grid_2d(((0.0, 1.0), (0.0, 1.0)), spacing=0.2)
    x = rng.uniform(0.05, 0.95, size=(31, 2))
    y = rng.normal(size=len(x))
    chunks = (
        (x[start : start + 4], y[start : start + 4]) for start in range(0, len(x), 4)
    )

    streamed = accumulate_interpolation_normal_equations_from_chunks(
        chunks, grid, interpolation=mode
    )
    single = accumulate_interpolation_normal_equations(
        x, y, grid, interpolation=mode, chunk_size=len(x)
    )

    np.testing.assert_allclose(
        streamed.dense_matrix(), single.dense_matrix(), atol=2e-13
    )
    np.testing.assert_allclose(streamed.rhs, single.rhs, atol=2e-13)


def test_keys_cubic_rows_reproduce_grid_values_and_partition_unity() -> None:
    grid = build_ski_grid_2d(((0.0, 1.0), (0.0, 1.0)), spacing=0.25)
    points = np.asarray([[0.0, 0.0], [0.25, 0.5], [0.375, 0.625]])
    indices, weights, _cells, tx, ty = interpolation_rows(
        grid, points, interpolation="cubic"
    )

    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=2e-15)
    assert np.count_nonzero(np.abs(weights[0]) > 0.0) == 1
    assert np.count_nonzero(np.abs(weights[1]) > 0.0) == 1
    assert tx[2] == pytest.approx(0.5)
    assert ty[2] == pytest.approx(0.5)
    assert indices.shape == weights.shape == (3, 16)


@pytest.mark.parametrize("mode", ["linear", "cubic"])
def test_inducing_cg_prediction_matches_dense_data_space_ski_krr(mode: str) -> None:
    x_train, y_train, x_test = _small_problem(seed=19)
    ridge = 0.4
    config = StructuredKernelInterpolationConfig(
        interpolation=mode,
        grid_spacing=0.25,
        lengthscale=0.3,
        nu=1.5,
        kernel_variance=1.0,
        absolute_ridge=ridge,
        train_chunk_size=5,
        prediction_chunk_size=4,
        cg_tolerance=1e-9,
        cg_maxiter=1_000,
        cg_preconditioner="circulant_density",
    )

    model = fit_structured_kernel_interpolation(x_train, y_train, config)
    train_w = _explicit_interpolation_matrix(model.grid, x_train, mode)
    test_w = _explicit_interpolation_matrix(model.grid, x_test, mode)
    inducing_kernel = model.kernel_operator.dense_matrix()
    approximate_train_kernel = train_w @ inducing_kernel @ train_w.T
    alpha = np.linalg.solve(
        approximate_train_kernel + ridge * np.eye(len(x_train)), y_train
    )
    expected = test_w @ inducing_kernel @ train_w.T @ alpha

    observed = model.predict(x_test, chunk_size=3)

    np.testing.assert_allclose(observed, expected, rtol=3e-9, atol=7e-10)
    diagnostics = model.fit_diagnostics
    assert diagnostics["regularization_convention"] == "absolute"
    assert diagnostics["absolute_ridge"] == ridge
    assert diagnostics["kronecker_product_used"] is False
    assert diagnostics["stores_full_interpolation_matrix"] is False
    assert diagnostics["converged"] is True
    assert diagnostics["original_inducing_relative_residual"] <= config.cg_tolerance
    assert diagnostics["train_total_seconds"] == pytest.approx(
        diagnostics["setup_seconds"] + diagnostics["solving_phase_seconds"]
    )


def test_absolute_ridge_is_not_scaled_by_training_count() -> None:
    x_train, y_train, x_test = _small_problem(seed=29)
    ridge = 0.23
    config = StructuredKernelInterpolationConfig(
        interpolation="linear",
        grid_spacing=0.25,
        lengthscale=0.35,
        absolute_ridge=ridge,
        train_chunk_size=7,
        cg_tolerance=1e-9,
        cg_maxiter=1_000,
        cg_preconditioner="none",
    )
    model = fit_structured_kernel_interpolation(x_train, y_train, config)
    train_w = _explicit_interpolation_matrix(model.grid, x_train, "linear")
    test_w = _explicit_interpolation_matrix(model.grid, x_test, "linear")
    kernel = model.kernel_operator.dense_matrix()

    correct_alpha = np.linalg.solve(
        train_w @ kernel @ train_w.T + ridge * np.eye(len(x_train)), y_train
    )
    incorrectly_scaled_alpha = np.linalg.solve(
        train_w @ kernel @ train_w.T + len(x_train) * ridge * np.eye(len(x_train)),
        y_train,
    )
    observed = model.predict(x_test)
    correct = test_w @ kernel @ train_w.T @ correct_alpha
    incorrectly_scaled = test_w @ kernel @ train_w.T @ incorrectly_scaled_alpha

    np.testing.assert_allclose(observed, correct, rtol=3e-9, atol=7e-10)
    assert np.linalg.norm(observed - correct) < 1e-7 * np.linalg.norm(correct)
    assert np.linalg.norm(observed - incorrectly_scaled) > 1e-2


def test_run_reports_streamed_metrics_and_exact_time_identity() -> None:
    x_train, y_train, x_test = _small_problem(seed=33)
    y_test = np.sin(2.0 * np.pi * x_test[:, 0]) + 0.4 * np.cos(
        2.0 * np.pi * x_test[:, 1]
    )
    config = StructuredKernelInterpolationConfig(
        interpolation="linear",
        grid_spacing=0.25,
        lengthscale=0.3,
        absolute_ridge=0.4,
        train_chunk_size=5,
        prediction_chunk_size=3,
        cg_tolerance=1e-8,
        cg_maxiter=1_000,
    )

    row = run_structured_kernel_interpolation(x_train, y_train, x_test, y_test, config)
    prediction = row["model"].predict(x_test)
    residual = prediction - y_test
    expected_rmse = math.sqrt(float(np.mean(residual * residual)))
    expected_mae = float(np.mean(np.abs(residual)))
    expected_r2 = 1.0 - float(np.dot(residual, residual)) / float(
        np.sum((y_test - np.mean(y_test)) ** 2)
    )

    assert row["status"] == "ok"
    assert row["method_label"] == "ski-linear"
    assert row["test_rmse"] == pytest.approx(expected_rmse, rel=2e-14)
    assert row["test_mae"] == pytest.approx(expected_mae, rel=2e-14)
    assert row["test_r2"] == pytest.approx(expected_r2, rel=2e-14)
    assert row["train_total_seconds"] == pytest.approx(
        row["setup_seconds"] + row["solving_phase_seconds"]
    )
    assert row["prediction_seconds"] >= 0.0
    assert row["diagnostics"]["moment_updates_total"] == 13 * len(x_train)


@pytest.mark.skipif(
    os.environ.get("EFGP_RUN_CUDA_TESTS") != "1",
    reason="set EFGP_RUN_CUDA_TESTS=1 on a healthy CUDA runner",
)
def test_cupy_linear_path_matches_numpy_reference_when_cuda_is_available() -> None:
    cupy = pytest.importorskip("cupy")
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - environment-specific CUDA failure
        pytest.skip(f"CuPy has no usable CUDA runtime: {exc}")
    if device_count <= 0:  # pragma: no cover - environment-specific CUDA absence
        pytest.skip("CuPy has no CUDA device")

    x_train, y_train, x_test = _small_problem(seed=47)
    common = {
        "interpolation": "linear",
        "grid_spacing": 0.25,
        "lengthscale": 0.3,
        "absolute_ridge": 0.4,
        "train_chunk_size": 5,
        "prediction_chunk_size": 4,
        "cg_tolerance": 1e-8,
        "cg_maxiter": 1_000,
        "cg_preconditioner": "circulant_density",
    }
    numpy_model = fit_structured_kernel_interpolation(
        x_train, y_train, StructuredKernelInterpolationConfig(**common)
    )
    cupy_model = fit_structured_kernel_interpolation(
        x_train,
        y_train,
        StructuredKernelInterpolationConfig(**common, backend="cupy"),
    )

    expected = numpy_model.predict(x_test)
    observed = cupy.asnumpy(cupy_model.predict(x_test))

    np.testing.assert_allclose(observed, expected, rtol=3e-8, atol=2e-9)
    diagnostics = cupy_model.fit_diagnostics
    assert diagnostics["backend"] == "cupy"
    assert diagnostics["normal_operator_format"] == "cupyx.scipy.sparse.csr_matrix"
    assert diagnostics["moment_reduction"].endswith("cupy_bincount")
    assert diagnostics["stores_full_interpolation_matrix"] is False
    assert diagnostics["converged"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"interpolation": "nearest"}, "interpolation"),
        ({"interpolation": "cubic", "grid_padding_points": 1}, "padding"),
        ({"backend": "jax"}, "backend"),
        ({"backend": "cupy"}, "linear"),
        ({"absolute_ridge": 0.0}, "absolute_ridge"),
        ({"cg_preconditioner": "jacobi"}, "preconditioner"),
    ],
)
def test_invalid_configuration_fails_closed(
    kwargs: dict[str, object], message: str
) -> None:
    x_train, y_train, _x_test = _small_problem(seed=4)
    config = StructuredKernelInterpolationConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        fit_structured_kernel_interpolation(x_train, y_train, config)


def test_out_of_bounds_points_fail_instead_of_clipping() -> None:
    grid = build_ski_grid_2d(((0.0, 1.0), (0.0, 1.0)), spacing=0.25)
    points = np.asarray([[-1.0, 0.5], [0.5, 2.0]])

    with pytest.raises(ValueError, match="outside"):
        interpolation_rows(grid, points, interpolation="cubic")
