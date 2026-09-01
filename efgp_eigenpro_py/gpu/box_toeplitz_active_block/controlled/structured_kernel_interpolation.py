"""Streaming structured kernel interpolation for fixed 2-D Matérn KRR.

This module implements the posterior-mean/KRR part of SKI without storing the
``N x M`` interpolation matrix.  For a regular inducing grid ``U`` and sparse
interpolation matrix ``W``, the approximating kernel is

``K_ski = W K_UU W.T``.

With the repository's absolute ridge convention, the data-space equation is

``(W K_UU W.T + lambda_abs I) alpha = y``.

Writing ``S = W.T W``, ``b = W.T y``, and ``beta = W.T alpha`` gives the exact
inducing-space equation

``(S K_UU + lambda_abs I) beta = b``.

The implementation solves the equivalent symmetric positive-definite system

``(K_UU S K_UU + lambda_abs K_UU) beta = K_UU b``

with conjugate gradients.  Prediction is ``W_test K_UU beta``.  In particular,
``lambda_abs`` is never multiplied or divided by the number of observations.

The repository's two-dimensional Matérn kernel is isotropic in Euclidean
distance.  It is not a product of one-dimensional Matérn kernels, so ``K_UU``
must not be silently replaced by a Kronecker product.  On a Cartesian grid it
is instead block-Toeplitz with Toeplitz blocks (BTTB); its exact matrix-vector
product is evaluated through a two-dimensional circulant embedding and FFT.

The NumPy reference path supports both linear and Keys cubic interpolation.
The production CuPy path intentionally supports linear interpolation only and
uses ``cupy.bincount`` for its 13 per-row polynomial-moment reductions.  Cubic
uses 65 values per row and remains a strict small-scale NumPy reference.  No
full interpolation matrix or per-observation CG vector is retained.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import numpy as np


_INTERPOLATION_DEGREES = {"linear": 1, "cubic": 3}


@dataclass(frozen=True)
class SKIGrid2D:
    """Uniform two-dimensional grid with explicit data-domain provenance."""

    x_start: float
    y_start: float
    spacing: float
    nx: int
    ny: int
    data_bounds: tuple[tuple[float, float], tuple[float, float]]
    padding_points: int = 2

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.nx), int(self.ny))

    @property
    def size(self) -> int:
        return int(self.nx) * int(self.ny)

    @property
    def x_stop(self) -> float:
        return float(self.x_start) + (int(self.nx) - 1) * float(self.spacing)

    @property
    def y_stop(self) -> float:
        return float(self.y_start) + (int(self.ny) - 1) * float(self.spacing)

    def x_coordinates(self) -> np.ndarray:
        return float(self.x_start) + float(self.spacing) * np.arange(
            int(self.nx), dtype=np.float64
        )

    def y_coordinates(self) -> np.ndarray:
        return float(self.y_start) + float(self.spacing) * np.arange(
            int(self.ny), dtype=np.float64
        )


@dataclass(frozen=True)
class StructuredKernelInterpolationConfig:
    """Configuration for native streaming SKI/KISS-GP posterior-mean KRR.

    ``grid_bounds`` are the data-domain bounds.  Two inducing points are added
    outside each side by default, matching the support needed by local cubic
    interpolation.  NumPy supports both interpolation modes as a strict
    reference implementation.  The production CuPy path deliberately supports
    linear interpolation only: it reduces the 13 polynomial moments with
    ``cupy.bincount`` and assembles a GPU CSR normal operator, so no N-sized
    interpolation object survives a chunk.
    """

    interpolation: str = "cubic"
    grid_spacing: float = 1.0 / 128.0
    grid_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 1.0),
        (0.0, 1.0),
    )
    grid_padding_points: int = 2
    lengthscale: float = 0.1
    nu: float = 1.5
    kernel_variance: float = 1.0
    absolute_ridge: float = 0.1
    train_chunk_size: int = 250_000
    prediction_chunk_size: int = 250_000
    cg_tolerance: float = 1e-7
    cg_maxiter: int = 5_000
    cg_preconditioner: str = "circulant_density"
    circulant_spectral_floor_relative: float = 1e-10
    require_convergence: bool = True
    backend: str = "numpy"


@dataclass
class BTTBMaternOperator2D:
    """Exact isotropic Matérn grid MVM through a 2-D circulant embedding."""

    grid: SKIGrid2D
    lengthscale: float
    nu: float
    variance: float
    first_column: np.ndarray
    embedding_spectrum: np.ndarray
    embedding_shape: tuple[int, int]

    @classmethod
    def build(
        cls,
        grid: SKIGrid2D,
        *,
        lengthscale: float,
        nu: float,
        variance: float,
    ) -> "BTTBMaternOperator2D":
        _validate_kernel_parameters(lengthscale, nu, variance)
        dx = float(grid.spacing) * np.arange(int(grid.nx), dtype=np.float64)
        dy = float(grid.spacing) * np.arange(int(grid.ny), dtype=np.float64)
        radius = np.hypot(dx[:, None], dy[None, :])
        first_column = _matern_from_radius(
            radius,
            lengthscale=float(lengthscale),
            nu=float(nu),
            variance=float(variance),
        )
        x_mirror = np.concatenate(
            (
                np.arange(int(grid.nx), dtype=np.int64),
                np.arange(int(grid.nx) - 2, 0, -1, dtype=np.int64),
            )
        )
        y_mirror = np.concatenate(
            (
                np.arange(int(grid.ny), dtype=np.int64),
                np.arange(int(grid.ny) - 2, 0, -1, dtype=np.int64),
            )
        )
        embedding = first_column[np.ix_(x_mirror, y_mirror)]
        embedding_shape = (int(embedding.shape[0]), int(embedding.shape[1]))
        spectrum = np.fft.rfftn(embedding, s=embedding_shape)
        return cls(
            grid=grid,
            lengthscale=float(lengthscale),
            nu=float(nu),
            variance=float(variance),
            first_column=np.ascontiguousarray(first_column),
            embedding_spectrum=np.ascontiguousarray(spectrum),
            embedding_shape=embedding_shape,
        )

    @property
    def storage_bytes(self) -> int:
        return int(self.first_column.nbytes + self.embedding_spectrum.nbytes)

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if int(values.size) != int(self.grid.size):
            raise ValueError(
                f"BTTB matvec expected {self.grid.size} entries, got {values.size}."
            )
        padded = np.zeros(self.embedding_shape, dtype=np.float64)
        padded[: int(self.grid.nx), : int(self.grid.ny)] = values.reshape(
            self.grid.shape
        )
        transformed = np.fft.rfftn(padded, s=self.embedding_shape)
        convolved = np.fft.irfftn(
            transformed * self.embedding_spectrum, s=self.embedding_shape
        )
        result = convolved[: int(self.grid.nx), : int(self.grid.ny)]
        return np.ascontiguousarray(result.reshape(-1), dtype=np.float64)

    def dense_matrix(self, *, max_grid_size: int = 4_096) -> np.ndarray:
        """Materialize ``K_UU`` for tests and small audits only."""

        if int(self.grid.size) > int(max_grid_size):
            raise MemoryError(
                "dense inducing kernel is an audit-only path; increase "
                "max_grid_size explicitly for a known-small grid"
            )
        x_grid, y_grid = np.meshgrid(
            self.grid.x_coordinates(), self.grid.y_coordinates(), indexing="ij"
        )
        points = np.column_stack((x_grid.reshape(-1), y_grid.reshape(-1)))
        difference = points[:, None, :] - points[None, :, :]
        radius = np.linalg.norm(difference, axis=2)
        return _matern_from_radius(
            radius,
            lengthscale=self.lengthscale,
            nu=self.nu,
            variance=self.variance,
        )

    def make_circulant_density_preconditioner(
        self,
        *,
        density: float,
        absolute_ridge: float,
        spectral_floor_relative: float,
    ) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
        """Return an SPD clipped-BCCB preconditioner for the symmetric system.

        The circulant approximation is used only as a preconditioner.  The
        system operator itself always uses the exact cropped BTTB MVM.
        """

        spectrum_real = np.real(self.embedding_spectrum)
        largest = max(float(np.max(np.abs(spectrum_real))), np.finfo(float).tiny)
        floor = max(float(spectral_floor_relative) * largest, np.finfo(np.float64).tiny)
        clipped = np.maximum(spectrum_real, floor)
        density_value = max(float(density), 0.0)
        denominator = (
            density_value * clipped * clipped + float(absolute_ridge) * clipped
        )

        def apply(vector: np.ndarray) -> np.ndarray:
            values = np.asarray(vector, dtype=np.float64).reshape(-1)
            padded = np.zeros(self.embedding_shape, dtype=np.float64)
            padded[: int(self.grid.nx), : int(self.grid.ny)] = values.reshape(
                self.grid.shape
            )
            transformed = np.fft.rfftn(padded, s=self.embedding_shape)
            solved = np.fft.irfftn(transformed / denominator, s=self.embedding_shape)
            return np.ascontiguousarray(
                solved[: int(self.grid.nx), : int(self.grid.ny)].reshape(-1),
                dtype=np.float64,
            )

        diagnostics = {
            "kind": "clipped_bccb_density",
            "density": density_value,
            "spectral_floor": floor,
            "spectral_floor_relative": float(spectral_floor_relative),
            "nonpositive_embedding_eigenvalues": int(
                np.count_nonzero(spectrum_real <= 0.0)
            ),
            "clipped_embedding_eigenvalues": int(
                np.count_nonzero(spectrum_real < floor)
            ),
            "embedding_eigenvalues": int(spectrum_real.size),
        }
        return apply, diagnostics


@dataclass
class CuPyBTTBMaternOperator2D:
    """CuPy counterpart of :class:`BTTBMaternOperator2D` for A100 runs."""

    grid: SKIGrid2D
    lengthscale: float
    nu: float
    variance: float
    first_column: Any
    embedding_spectrum: Any
    embedding_shape: tuple[int, int]
    array_module: Any

    @classmethod
    def build(
        cls,
        grid: SKIGrid2D,
        *,
        lengthscale: float,
        nu: float,
        variance: float,
        array_module: Any,
    ) -> "CuPyBTTBMaternOperator2D":
        _validate_kernel_parameters(lengthscale, nu, variance)
        xp = array_module
        dx = float(grid.spacing) * xp.arange(int(grid.nx), dtype=xp.float64)
        dy = float(grid.spacing) * xp.arange(int(grid.ny), dtype=xp.float64)
        radius = xp.sqrt(dx[:, None] * dx[:, None] + dy[None, :] * dy[None, :])
        first_column = _matern_from_radius_cupy(
            radius,
            lengthscale=float(lengthscale),
            nu=float(nu),
            variance=float(variance),
            array_module=xp,
        )
        x_mirror = xp.concatenate(
            (
                xp.arange(int(grid.nx), dtype=xp.int64),
                xp.arange(int(grid.nx) - 2, 0, -1, dtype=xp.int64),
            )
        )
        y_mirror = xp.concatenate(
            (
                xp.arange(int(grid.ny), dtype=xp.int64),
                xp.arange(int(grid.ny) - 2, 0, -1, dtype=xp.int64),
            )
        )
        embedding = first_column[x_mirror[:, None], y_mirror[None, :]]
        embedding_shape = (int(embedding.shape[0]), int(embedding.shape[1]))
        spectrum = xp.fft.rfftn(embedding, s=embedding_shape)
        return cls(
            grid=grid,
            lengthscale=float(lengthscale),
            nu=float(nu),
            variance=float(variance),
            first_column=xp.ascontiguousarray(first_column),
            embedding_spectrum=xp.ascontiguousarray(spectrum),
            embedding_shape=embedding_shape,
            array_module=xp,
        )

    @property
    def storage_bytes(self) -> int:
        return int(self.first_column.nbytes + self.embedding_spectrum.nbytes)

    def matvec(self, vector: Any) -> Any:
        xp = self.array_module
        values = xp.asarray(vector, dtype=xp.float64).reshape(-1)
        if int(values.size) != int(self.grid.size):
            raise ValueError(
                f"BTTB matvec expected {self.grid.size} entries, got {values.size}."
            )
        padded = xp.zeros(self.embedding_shape, dtype=xp.float64)
        padded[: int(self.grid.nx), : int(self.grid.ny)] = values.reshape(
            self.grid.shape
        )
        transformed = xp.fft.rfftn(padded, s=self.embedding_shape)
        convolved = xp.fft.irfftn(
            transformed * self.embedding_spectrum, s=self.embedding_shape
        )
        return xp.ascontiguousarray(
            convolved[: int(self.grid.nx), : int(self.grid.ny)].reshape(-1),
            dtype=xp.float64,
        )

    def make_circulant_density_preconditioner(
        self,
        *,
        density: float,
        absolute_ridge: float,
        spectral_floor_relative: float,
    ) -> tuple[Callable[[Any], Any], dict[str, Any]]:
        xp = self.array_module
        spectrum_real = xp.real(self.embedding_spectrum)
        largest = max(_cupy_scalar(xp.max(xp.abs(spectrum_real))), np.finfo(float).tiny)
        floor = max(float(spectral_floor_relative) * largest, np.finfo(np.float64).tiny)
        clipped = xp.maximum(spectrum_real, floor)
        density_value = max(float(density), 0.0)
        denominator = (
            density_value * clipped * clipped + float(absolute_ridge) * clipped
        )

        def apply(vector: Any) -> Any:
            values = xp.asarray(vector, dtype=xp.float64).reshape(-1)
            padded = xp.zeros(self.embedding_shape, dtype=xp.float64)
            padded[: int(self.grid.nx), : int(self.grid.ny)] = values.reshape(
                self.grid.shape
            )
            transformed = xp.fft.rfftn(padded, s=self.embedding_shape)
            solved = xp.fft.irfftn(transformed / denominator, s=self.embedding_shape)
            return xp.ascontiguousarray(
                solved[: int(self.grid.nx), : int(self.grid.ny)].reshape(-1),
                dtype=xp.float64,
            )

        diagnostics = {
            "kind": "clipped_bccb_density",
            "density": density_value,
            "spectral_floor": floor,
            "spectral_floor_relative": float(spectral_floor_relative),
            "nonpositive_embedding_eigenvalues": int(
                _cupy_scalar(xp.count_nonzero(spectrum_real <= 0.0))
            ),
            "clipped_embedding_eigenvalues": int(
                _cupy_scalar(xp.count_nonzero(spectrum_real < floor))
            ),
            "embedding_eigenvalues": int(spectrum_real.size),
        }
        return apply, diagnostics


@dataclass
class CuPyInterpolationNormalEquations2D:
    """GPU CSR representation of linear-interpolation sufficient statistics."""

    grid: SKIGrid2D
    sparse_matrix: Any
    rhs: Any
    cell_ids: Any
    n_rows: int
    trace_value: float
    moment_updates_per_row: int = 13
    interpolation_nonzeros_per_row: int = 4

    @property
    def storage_bytes(self) -> int:
        matrix = self.sparse_matrix
        return int(
            matrix.data.nbytes
            + matrix.indices.nbytes
            + matrix.indptr.nbytes
            + self.rhs.nbytes
            + self.cell_ids.nbytes
        )

    @property
    def trace(self) -> float:
        return float(self.trace_value)

    def matvec(self, vector: Any) -> Any:
        return self.sparse_matrix @ vector


@dataclass
class InterpolationNormalEquations2D:
    """Compact representation of ``S=W.T W`` and ``b=W.T y``."""

    grid: SKIGrid2D
    interpolation: str
    cell_ids: np.ndarray
    cell_node_indices: np.ndarray
    gram_blocks: np.ndarray
    rhs: np.ndarray
    n_rows: int
    moment_updates_per_row: int
    interpolation_nonzeros_per_row: int

    @property
    def storage_bytes(self) -> int:
        return int(
            self.cell_ids.nbytes
            + self.cell_node_indices.nbytes
            + self.gram_blocks.nbytes
            + self.rhs.nbytes
        )

    @property
    def trace(self) -> float:
        return float(np.trace(self.gram_blocks, axis1=1, axis2=2).sum())

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if int(values.size) != int(self.grid.size):
            raise ValueError(
                f"interpolation normal operator expected {self.grid.size} entries, "
                f"got {values.size}."
            )
        if int(self.cell_ids.size) == 0:
            return np.zeros_like(values)
        local_values = values[self.cell_node_indices]
        local_results = np.einsum(
            "cij,cj->ci", self.gram_blocks, local_values, optimize=True
        )
        output = np.zeros(int(self.grid.size), dtype=np.float64)
        np.add.at(output, self.cell_node_indices.reshape(-1), local_results.reshape(-1))
        return output

    def dense_matrix(self, *, max_grid_size: int = 4_096) -> np.ndarray:
        """Materialize ``W.T W`` for tests and small audits only."""

        if int(self.grid.size) > int(max_grid_size):
            raise MemoryError(
                "dense interpolation normal matrix is an audit-only path; increase "
                "max_grid_size explicitly for a known-small grid"
            )
        dense = np.zeros((int(self.grid.size), int(self.grid.size)), dtype=np.float64)
        for nodes, block in zip(self.cell_node_indices, self.gram_blocks):
            dense[np.ix_(nodes, nodes)] += block
        return dense


@dataclass
class StructuredKernelInterpolationModel:
    """Fitted fixed-kernel SKI posterior mean."""

    config: StructuredKernelInterpolationConfig
    grid: SKIGrid2D
    kernel_operator: Any
    beta: Any
    inducing_prediction_values: Any
    fit_diagnostics: dict[str, Any]
    array_module: Any

    def predict_chunk(self, x: np.ndarray) -> np.ndarray:
        if self.array_module is not np:
            indices, weights, _cell_ids, _tx, _ty = _cupy_linear_interpolation_rows(
                self.grid, x, self.array_module
            )
            return self.array_module.sum(
                weights * self.inducing_prediction_values[indices],
                axis=1,
                dtype=self.array_module.float64,
            )
        indices, weights, _cell_ids, _tx, _ty = interpolation_rows(
            self.grid, x, interpolation=self.config.interpolation
        )
        return np.sum(
            weights * self.inducing_prediction_values[indices], axis=1, dtype=np.float64
        )

    def predict(self, x: np.ndarray, *, chunk_size: int | None = None) -> np.ndarray:
        if self.array_module is not np:
            return _predict_cupy_model(self, x, chunk_size=chunk_size)
        points = _validate_x(x)
        use_chunk = int(chunk_size or self.config.prediction_chunk_size)
        if use_chunk <= 0:
            raise ValueError("prediction chunk_size must be positive.")
        output = np.empty(int(points.shape[0]), dtype=np.float64)
        for start in range(0, int(points.shape[0]), use_chunk):
            stop = min(start + use_chunk, int(points.shape[0]))
            output[start:stop] = self.predict_chunk(points[start:stop])
        return output


class InterpolationMomentAccumulator2D:
    """Streaming per-cell polynomial moment accumulator."""

    def __init__(self, grid: SKIGrid2D, interpolation: str) -> None:
        mode = _normalize_interpolation(interpolation)
        degree = _INTERPOLATION_DEGREES[mode]
        self.grid = grid
        self.interpolation = mode
        self.degree = int(degree)
        self.n_cells = (int(grid.nx) - 1) * (int(grid.ny) - 1)
        self.gram_moments = np.zeros(
            (self.n_cells, 2 * degree + 1, 2 * degree + 1), dtype=np.float64
        )
        self.rhs_moments = np.zeros(
            (self.n_cells, degree + 1, degree + 1), dtype=np.float64
        )
        self.n_rows = 0

    @property
    def moment_updates_per_row(self) -> int:
        degree = int(self.degree)
        return int((2 * degree + 1) ** 2 + (degree + 1) ** 2)

    def update(self, x_chunk: np.ndarray, y_chunk: np.ndarray) -> None:
        points = _validate_x(x_chunk)
        targets = np.asarray(y_chunk, dtype=np.float64).reshape(-1)
        if int(points.shape[0]) != int(targets.size):
            raise ValueError("x_chunk and y_chunk row counts differ.")
        if int(targets.size) == 0:
            return
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(targets)):
            raise ValueError("SKI training chunks must contain only finite values.")
        _indices, _weights, cell_ids, tx, ty = interpolation_rows(
            self.grid, points, interpolation=self.interpolation
        )
        gram_degree = 2 * int(self.degree)
        tx_powers = [np.ones_like(tx)]
        ty_powers = [np.ones_like(ty)]
        for _ in range(gram_degree):
            tx_powers.append(tx_powers[-1] * tx)
            ty_powers.append(ty_powers[-1] * ty)
        for px in range(gram_degree + 1):
            for py in range(gram_degree + 1):
                np.add.at(
                    self.gram_moments[:, px, py],
                    cell_ids,
                    tx_powers[px] * ty_powers[py],
                )
        for px in range(int(self.degree) + 1):
            for py in range(int(self.degree) + 1):
                np.add.at(
                    self.rhs_moments[:, px, py],
                    cell_ids,
                    targets * tx_powers[px] * ty_powers[py],
                )
        self.n_rows += int(targets.size)

    def finalize(self) -> InterpolationNormalEquations2D:
        occupied = np.flatnonzero(self.gram_moments[:, 0, 0] > 0.0).astype(
            np.int64, copy=False
        )
        coefficients = _tensor_interpolation_coefficients(self.interpolation)
        gram_map = _gram_coefficient_map(coefficients)
        occupied_gram_moments = self.gram_moments[occupied]
        occupied_rhs_moments = self.rhs_moments[occupied]
        gram_blocks = np.einsum(
            "cpq,abpq->cab", occupied_gram_moments, gram_map, optimize=True
        )
        rhs_blocks = np.einsum(
            "cpq,apq->ca", occupied_rhs_moments, coefficients, optimize=True
        )
        cell_node_indices = _cell_node_indices(
            self.grid, occupied, interpolation=self.interpolation
        )
        rhs = np.zeros(int(self.grid.size), dtype=np.float64)
        if int(occupied.size):
            np.add.at(rhs, cell_node_indices.reshape(-1), rhs_blocks.reshape(-1))
        # Independent moment reductions can produce tiny asymmetric roundoff.
        gram_blocks = 0.5 * (gram_blocks + np.swapaxes(gram_blocks, 1, 2))
        return InterpolationNormalEquations2D(
            grid=self.grid,
            interpolation=self.interpolation,
            cell_ids=occupied,
            cell_node_indices=np.ascontiguousarray(cell_node_indices),
            gram_blocks=np.ascontiguousarray(gram_blocks, dtype=np.float64),
            rhs=rhs,
            n_rows=int(self.n_rows),
            moment_updates_per_row=self.moment_updates_per_row,
            interpolation_nonzeros_per_row=int(coefficients.shape[0]),
        )


def build_ski_grid_2d(
    data_bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    spacing: float,
    padding_points: int = 2,
) -> SKIGrid2D:
    """Build a uniform grid covering the declared rectangle plus padding."""

    if not math.isfinite(float(spacing)) or float(spacing) <= 0.0:
        raise ValueError("grid spacing must be finite and positive.")
    if int(padding_points) < 0:
        raise ValueError("padding_points must be nonnegative.")
    if len(data_bounds) != 2:
        raise ValueError("data_bounds must contain exactly two dimensions.")
    normalized: list[tuple[float, float]] = []
    for bound in data_bounds:
        if len(bound) != 2:
            raise ValueError("each data bound must be a (minimum, maximum) pair.")
        minimum, maximum = float(bound[0]), float(bound[1])
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum >= maximum
        ):
            raise ValueError("each data bound must be finite and strictly increasing.")
        normalized.append((minimum, maximum))
    h = float(spacing)
    padding = int(padding_points)

    def axis(minimum: float, maximum: float) -> tuple[float, int]:
        start = minimum - padding * h
        required_stop = maximum + padding * h
        intervals = int(math.ceil((required_stop - start) / h - 1e-12))
        return start, intervals + 1

    x_start, nx = axis(*normalized[0])
    y_start, ny = axis(*normalized[1])
    if nx < 2 or ny < 2:
        raise ValueError("SKI grid needs at least two points per dimension.")
    return SKIGrid2D(
        x_start=x_start,
        y_start=y_start,
        spacing=h,
        nx=nx,
        ny=ny,
        data_bounds=(normalized[0], normalized[1]),
        padding_points=padding,
    )


def interpolation_rows(
    grid: SKIGrid2D,
    x: np.ndarray,
    *,
    interpolation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse interpolation rows for one transient chunk.

    This helper is public for small dense validation.  The fit path consumes its
    output immediately and never retains rows across chunks.
    """

    points = _validate_x(x)
    mode = _normalize_interpolation(interpolation)
    scaled_x = (points[:, 0] - float(grid.x_start)) / float(grid.spacing)
    scaled_y = (points[:, 1] - float(grid.y_start)) / float(grid.spacing)
    left_x = np.floor(scaled_x).astype(np.int64)
    left_y = np.floor(scaled_y).astype(np.int64)
    tx = scaled_x - left_x
    ty = scaled_y - left_y
    # Protect exact grid points from harmless 1-epsilon/1+epsilon drift.
    near_one_x = np.isclose(tx, 1.0, rtol=0.0, atol=32.0 * np.finfo(float).eps)
    near_one_y = np.isclose(ty, 1.0, rtol=0.0, atol=32.0 * np.finfo(float).eps)
    left_x[near_one_x] += 1
    left_y[near_one_y] += 1
    tx[near_one_x] = 0.0
    ty[near_one_y] = 0.0
    if mode == "linear":
        offsets = np.asarray([0, 1], dtype=np.int64)
        wx = np.column_stack((1.0 - tx, tx))
        wy = np.column_stack((1.0 - ty, ty))
    else:
        offsets = np.asarray([-1, 0, 1, 2], dtype=np.int64)
        wx = _keys_cubic_weights(tx)
        wy = _keys_cubic_weights(ty)
    min_x = left_x + int(offsets[0])
    max_x = left_x + int(offsets[-1])
    min_y = left_y + int(offsets[0])
    max_y = left_y + int(offsets[-1])
    if (
        np.any(min_x < 0)
        or np.any(min_y < 0)
        or np.any(max_x >= int(grid.nx))
        or np.any(max_y >= int(grid.ny))
    ):
        raise ValueError(
            "input lies outside the interpolation-safe grid bounds; expand the "
            "declared data bounds or padding"
        )
    x_nodes = left_x[:, None] + offsets[None, :]
    y_nodes = left_y[:, None] + offsets[None, :]
    indices = (x_nodes[:, :, None] * int(grid.ny) + y_nodes[:, None, :]).reshape(
        int(points.shape[0]), -1
    )
    weights = (wx[:, :, None] * wy[:, None, :]).reshape(int(points.shape[0]), -1)
    if np.any(left_x < 0) or np.any(left_x >= int(grid.nx) - 1):
        raise ValueError("input does not map to a valid x grid cell.")
    if np.any(left_y < 0) or np.any(left_y >= int(grid.ny) - 1):
        raise ValueError("input does not map to a valid y grid cell.")
    cell_ids = left_x * (int(grid.ny) - 1) + left_y
    return (
        np.ascontiguousarray(indices),
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(cell_ids),
        np.ascontiguousarray(tx, dtype=np.float64),
        np.ascontiguousarray(ty, dtype=np.float64),
    )


def accumulate_interpolation_normal_equations(
    x: np.ndarray,
    y: np.ndarray,
    grid: SKIGrid2D,
    *,
    interpolation: str,
    chunk_size: int,
) -> InterpolationNormalEquations2D:
    """Stream arrays into moment-compressed ``W.T W`` and ``W.T y``."""

    points = _validate_x(x)
    targets = np.asarray(y).reshape(-1)
    if int(points.shape[0]) != int(targets.size):
        raise ValueError("x and y row counts differ.")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    accumulator = InterpolationMomentAccumulator2D(grid, interpolation)
    for start in range(0, int(points.shape[0]), int(chunk_size)):
        stop = min(start + int(chunk_size), int(points.shape[0]))
        accumulator.update(points[start:stop], targets[start:stop])
    return accumulator.finalize()


def accumulate_interpolation_normal_equations_from_chunks(
    chunks: Iterable[tuple[np.ndarray, np.ndarray]],
    grid: SKIGrid2D,
    *,
    interpolation: str,
) -> InterpolationNormalEquations2D:
    """Accumulate sufficient statistics from an arbitrary one-pass chunk source."""

    accumulator = InterpolationMomentAccumulator2D(grid, interpolation)
    for x_chunk, y_chunk in chunks:
        accumulator.update(x_chunk, y_chunk)
    return accumulator.finalize()


def fit_structured_kernel_interpolation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: StructuredKernelInterpolationConfig,
) -> StructuredKernelInterpolationModel:
    """Fit the fixed isotropic-Matérn SKI posterior mean/KRR model."""

    _validate_config(config)
    if str(config.backend).strip().lower() == "cupy":
        return _fit_structured_kernel_interpolation_cupy(x_train, y_train, config)
    points = _validate_x(x_train)
    targets = np.asarray(y_train).reshape(-1)
    if int(points.shape[0]) != int(targets.size):
        raise ValueError("x_train and y_train row counts differ.")
    if int(targets.size) <= 0:
        raise ValueError("SKI fit requires at least one training row.")

    setup_start = time.perf_counter()
    grid = build_ski_grid_2d(
        config.grid_bounds,
        spacing=float(config.grid_spacing),
        padding_points=int(config.grid_padding_points),
    )
    kernel_start = time.perf_counter()
    kernel_operator = BTTBMaternOperator2D.build(
        grid,
        lengthscale=float(config.lengthscale),
        nu=float(config.nu),
        variance=float(config.kernel_variance),
    )
    kernel_setup_seconds = float(time.perf_counter() - kernel_start)
    statistics_start = time.perf_counter()
    normal = accumulate_interpolation_normal_equations(
        points,
        targets,
        grid,
        interpolation=config.interpolation,
        chunk_size=int(config.train_chunk_size),
    )
    statistics_seconds = float(time.perf_counter() - statistics_start)
    system_setup_start = time.perf_counter()
    ridge = float(config.absolute_ridge)
    rhs = kernel_operator.matvec(normal.rhs)

    def symmetric_system_matvec(vector: np.ndarray) -> np.ndarray:
        kernel_vector = kernel_operator.matvec(vector)
        return kernel_operator.matvec(normal.matvec(kernel_vector) + ridge * vector)

    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None
    preconditioner_diagnostics: dict[str, Any] = {"kind": "none"}
    if str(config.cg_preconditioner).strip().lower() == "circulant_density":
        density = normal.trace / max(int(grid.size), 1)
        preconditioner, preconditioner_diagnostics = (
            kernel_operator.make_circulant_density_preconditioner(
                density=density,
                absolute_ridge=ridge,
                spectral_floor_relative=float(config.circulant_spectral_floor_relative),
            )
        )

    system_setup_seconds = float(time.perf_counter() - system_setup_start)
    setup_seconds = float(time.perf_counter() - setup_start)
    solve_start = time.perf_counter()
    beta, cg_diagnostics = _conjugate_gradient(
        symmetric_system_matvec,
        rhs,
        # The symmetric residual can modestly understate the residual of the
        # original inducing equation when K_UU is ill-conditioned.  Solve the
        # symmetric system two decimal orders tighter, then audit the equation
        # that actually defines the SKI posterior mean below.
        tolerance=max(float(config.cg_tolerance) * 0.01, np.finfo(float).eps),
        maxiter=int(config.cg_maxiter),
        preconditioner=preconditioner,
    )
    inducing_prediction_values = kernel_operator.matvec(beta)
    original_residual = normal.rhs - (
        normal.matvec(inducing_prediction_values) + ridge * beta
    )
    original_denominator = max(
        float(np.linalg.norm(normal.rhs)), np.finfo(np.float64).tiny
    )
    original_relative_residual = float(
        np.linalg.norm(original_residual) / original_denominator
    )
    solving_phase_seconds = float(time.perf_counter() - solve_start)
    converged = bool(
        cg_diagnostics["converged"]
        and math.isfinite(original_relative_residual)
        and original_relative_residual <= float(config.cg_tolerance)
    )
    if bool(config.require_convergence) and not converged:
        raise RuntimeError(
            "SKI inducing solve did not meet the declared original-system "
            f"tolerance: relative_residual={original_relative_residual:.6e}, "
            f"tolerance={config.cg_tolerance:.6e}, "
            f"iterations={cg_diagnostics['iterations']}"
        )

    train_total_seconds = float(setup_seconds + solving_phase_seconds)
    diagnostics: dict[str, Any] = {
        "implementation": "native_streamed_ski_krr",
        "method_label": (
            "kissgp-cubic"
            if _normalize_interpolation(config.interpolation) == "cubic"
            else "ski-linear"
        ),
        "kernel_family": "matern",
        "backend": "numpy",
        "precision": "fp64",
        "kernel_geometry": "isotropic_euclidean_2d",
        "kernel_structure": "bttb_exact_mvm_via_2d_circulant_embedding_fft",
        "kronecker_product_used": False,
        "lengthscale": float(config.lengthscale),
        "nu": float(config.nu),
        "kernel_variance": float(config.kernel_variance),
        "absolute_ridge": ridge,
        "regularization_convention": "absolute",
        "ridge_identity": "(W K_UU W.T + lambda_abs I) alpha = y",
        "inducing_identity": "(S K_UU + lambda_abs I) beta = W.T y",
        "symmetric_cg_identity": ("(K_UU S K_UU + lambda_abs K_UU) beta = K_UU W.T y"),
        "n_train": int(points.shape[0]),
        "input_dim": 2,
        "interpolation": _normalize_interpolation(config.interpolation),
        "interpolation_nonzeros_per_row": int(normal.interpolation_nonzeros_per_row),
        "moment_updates_per_row": int(normal.moment_updates_per_row),
        "moment_updates_total": int(normal.moment_updates_per_row * normal.n_rows),
        "stores_full_interpolation_matrix": False,
        "grid_shape": [int(grid.nx), int(grid.ny)],
        "grid_size": int(grid.size),
        "grid_spacing": float(grid.spacing),
        "grid_outer_bounds": [
            [float(grid.x_start), float(grid.x_stop)],
            [float(grid.y_start), float(grid.y_stop)],
        ],
        "occupied_interpolation_cells": int(normal.cell_ids.size),
        "normal_operator_storage_bytes": int(normal.storage_bytes),
        "kernel_operator_storage_bytes": int(kernel_operator.storage_bytes),
        "cg_tolerance": float(config.cg_tolerance),
        "cg_maxiter": int(config.cg_maxiter),
        "cg_iterations": int(cg_diagnostics["iterations"]),
        "cg_symmetric_relative_residual": float(cg_diagnostics["relative_residual"]),
        "original_inducing_relative_residual": original_relative_residual,
        "converged": converged,
        "preconditioner": preconditioner_diagnostics,
        "kernel_setup_seconds": kernel_setup_seconds,
        "statistics_seconds": statistics_seconds,
        "system_setup_seconds": system_setup_seconds,
        "setup_seconds": setup_seconds,
        "solving_phase_seconds": solving_phase_seconds,
        "train_total_seconds": train_total_seconds,
        "timing_scope": (
            "method-owned grid/kernel construction, streamed sufficient-statistic "
            "reduction, inducing-system/preconditioner setup, and CG solve"
        ),
        "config": asdict(config),
    }
    return StructuredKernelInterpolationModel(
        config=config,
        grid=grid,
        kernel_operator=kernel_operator,
        beta=np.ascontiguousarray(beta, dtype=np.float64),
        inducing_prediction_values=np.ascontiguousarray(
            inducing_prediction_values, dtype=np.float64
        ),
        fit_diagnostics=diagnostics,
        array_module=np,
    )


def _fit_structured_kernel_interpolation_cupy(
    x_train: Any,
    y_train: Any,
    config: StructuredKernelInterpolationConfig,
) -> StructuredKernelInterpolationModel:
    """Fit the production linear-SKI path entirely with CuPy/CuPyX operators."""

    cp, cupy_sparse = _load_cupy_backend()
    n_train = _validate_paired_row_shapes(x_train, y_train, label="train")
    if n_train <= 0:
        raise ValueError("SKI fit requires at least one training row.")

    _synchronize_cupy(cp)
    setup_start = time.perf_counter()
    grid = build_ski_grid_2d(
        config.grid_bounds,
        spacing=float(config.grid_spacing),
        padding_points=int(config.grid_padding_points),
    )
    kernel_start = time.perf_counter()
    kernel_operator = CuPyBTTBMaternOperator2D.build(
        grid,
        lengthscale=float(config.lengthscale),
        nu=float(config.nu),
        variance=float(config.kernel_variance),
        array_module=cp,
    )
    _synchronize_cupy(cp)
    kernel_setup_seconds = float(time.perf_counter() - kernel_start)
    statistics_start = time.perf_counter()
    normal = _accumulate_cupy_linear_normal_equations(
        x_train,
        y_train,
        grid,
        chunk_size=int(config.train_chunk_size),
        array_module=cp,
        sparse_module=cupy_sparse,
    )
    _synchronize_cupy(cp)
    statistics_seconds = float(time.perf_counter() - statistics_start)
    system_setup_start = time.perf_counter()
    ridge = float(config.absolute_ridge)
    rhs = kernel_operator.matvec(normal.rhs)

    def symmetric_system_matvec(vector: Any) -> Any:
        kernel_vector = kernel_operator.matvec(vector)
        return kernel_operator.matvec(normal.matvec(kernel_vector) + ridge * vector)

    preconditioner: Callable[[Any], Any] | None = None
    preconditioner_diagnostics: dict[str, Any] = {"kind": "none"}
    if str(config.cg_preconditioner).strip().lower() == "circulant_density":
        density = normal.trace / max(int(grid.size), 1)
        preconditioner, preconditioner_diagnostics = (
            kernel_operator.make_circulant_density_preconditioner(
                density=density,
                absolute_ridge=ridge,
                spectral_floor_relative=float(config.circulant_spectral_floor_relative),
            )
        )

    _synchronize_cupy(cp)
    system_setup_seconds = float(time.perf_counter() - system_setup_start)
    setup_seconds = float(time.perf_counter() - setup_start)
    solve_start = time.perf_counter()
    beta, cg_diagnostics = _conjugate_gradient_cupy(
        symmetric_system_matvec,
        rhs,
        tolerance=max(float(config.cg_tolerance) * 0.01, np.finfo(float).eps),
        maxiter=int(config.cg_maxiter),
        preconditioner=preconditioner,
        array_module=cp,
    )
    inducing_prediction_values = kernel_operator.matvec(beta)
    original_residual = normal.rhs - (
        normal.matvec(inducing_prediction_values) + ridge * beta
    )
    original_denominator = max(
        _cupy_scalar(cp.linalg.norm(normal.rhs)), np.finfo(np.float64).tiny
    )
    original_relative_residual = float(
        _cupy_scalar(cp.linalg.norm(original_residual)) / original_denominator
    )
    _synchronize_cupy(cp)
    solving_phase_seconds = float(time.perf_counter() - solve_start)
    converged = bool(
        cg_diagnostics["converged"]
        and math.isfinite(original_relative_residual)
        and original_relative_residual <= float(config.cg_tolerance)
    )
    if bool(config.require_convergence) and not converged:
        raise RuntimeError(
            "SKI inducing solve did not meet the declared original-system "
            f"tolerance: relative_residual={original_relative_residual:.6e}, "
            f"tolerance={config.cg_tolerance:.6e}, "
            f"iterations={cg_diagnostics['iterations']}"
        )

    train_total_seconds = float(setup_seconds + solving_phase_seconds)
    diagnostics: dict[str, Any] = {
        "implementation": "native_streamed_ski_krr_cupy",
        "method_label": "ski-linear",
        "backend": "cupy",
        "precision": "fp64",
        "kernel_family": "matern",
        "kernel_geometry": "isotropic_euclidean_2d",
        "kernel_structure": "bttb_exact_mvm_via_cupy_2d_circulant_embedding_fft",
        "kronecker_product_used": False,
        "lengthscale": float(config.lengthscale),
        "nu": float(config.nu),
        "kernel_variance": float(config.kernel_variance),
        "absolute_ridge": ridge,
        "regularization_convention": "absolute",
        "ridge_identity": "(W K_UU W.T + lambda_abs I) alpha = y",
        "inducing_identity": "(S K_UU + lambda_abs I) beta = W.T y",
        "symmetric_cg_identity": ("(K_UU S K_UU + lambda_abs K_UU) beta = K_UU W.T y"),
        "n_train": int(n_train),
        "input_dim": 2,
        "interpolation": "linear",
        "interpolation_nonzeros_per_row": 4,
        "moment_reduction": "13_per_row_polynomial_moments_via_cupy_bincount",
        "moment_updates_per_row": 13,
        "moment_updates_total": int(13 * n_train),
        "stores_full_interpolation_matrix": False,
        "stores_data_space_cg_vectors": False,
        "grid_shape": [int(grid.nx), int(grid.ny)],
        "grid_size": int(grid.size),
        "grid_spacing": float(grid.spacing),
        "grid_outer_bounds": [
            [float(grid.x_start), float(grid.x_stop)],
            [float(grid.y_start), float(grid.y_stop)],
        ],
        "occupied_interpolation_cells": int(normal.cell_ids.size),
        "normal_operator_format": "cupyx.scipy.sparse.csr_matrix",
        "normal_operator_nnz": int(normal.sparse_matrix.nnz),
        "normal_operator_storage_bytes": int(normal.storage_bytes),
        "kernel_operator_storage_bytes": int(kernel_operator.storage_bytes),
        "cg_tolerance": float(config.cg_tolerance),
        "cg_maxiter": int(config.cg_maxiter),
        "cg_iterations": int(cg_diagnostics["iterations"]),
        "cg_symmetric_relative_residual": float(cg_diagnostics["relative_residual"]),
        "original_inducing_relative_residual": original_relative_residual,
        "converged": converged,
        "preconditioner": preconditioner_diagnostics,
        "kernel_setup_seconds": kernel_setup_seconds,
        "statistics_seconds": statistics_seconds,
        "system_setup_seconds": system_setup_seconds,
        "setup_seconds": setup_seconds,
        "solving_phase_seconds": solving_phase_seconds,
        "train_total_seconds": train_total_seconds,
        "timing_scope": (
            "method-owned grid/kernel construction, streamed host-to-device "
            "training transfers, sufficient-statistic reduction, and CG solve"
        ),
        "config": asdict(config),
    }
    return StructuredKernelInterpolationModel(
        config=config,
        grid=grid,
        kernel_operator=kernel_operator,
        beta=cp.ascontiguousarray(beta, dtype=cp.float64),
        inducing_prediction_values=cp.ascontiguousarray(
            inducing_prediction_values, dtype=cp.float64
        ),
        fit_diagnostics=diagnostics,
        array_module=cp,
    )


def run_structured_kernel_interpolation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: StructuredKernelInterpolationConfig,
) -> dict[str, Any]:
    """Fit and evaluate SKI with streamed test metrics and explicit timing scope."""

    model = fit_structured_kernel_interpolation(x_train, y_train, config)
    if model.array_module is not np:
        return _evaluate_cupy_model(
            model, x_train=x_train, x_test=x_test, y_test=y_test
        )
    test_points = _validate_x(x_test)
    test_targets = np.asarray(y_test).reshape(-1)
    if int(test_points.shape[0]) != int(test_targets.size):
        raise ValueError("x_test and y_test row counts differ.")
    if int(test_targets.size) <= 0:
        raise ValueError("SKI evaluation requires at least one test row.")
    if not np.all(np.isfinite(test_targets)):
        raise ValueError("y_test must contain only finite values.")

    squared_error = 0.0
    absolute_error = 0.0
    target_sum = 0.0
    target_square_sum = 0.0
    prediction_start = time.perf_counter()
    chunk = int(config.prediction_chunk_size)
    for start in range(0, int(test_points.shape[0]), chunk):
        stop = min(start + chunk, int(test_points.shape[0]))
        prediction = model.predict_chunk(test_points[start:stop])
        truth = np.asarray(test_targets[start:stop], dtype=np.float64)
        residual = prediction - truth
        squared_error += float(np.dot(residual, residual))
        absolute_error += float(np.abs(residual).sum(dtype=np.float64))
        target_sum += float(truth.sum(dtype=np.float64))
        target_square_sum += float(np.dot(truth, truth))
    prediction_seconds = float(time.perf_counter() - prediction_start)
    metrics = _metrics_from_sums(
        n_rows=int(test_targets.size),
        squared_error=squared_error,
        absolute_error=absolute_error,
        target_sum=target_sum,
        target_square_sum=target_square_sum,
    )
    diagnostics = model.fit_diagnostics
    return {
        "status": "ok",
        "implementation": diagnostics["implementation"],
        "method_label": diagnostics["method_label"],
        "n_train": int(np.asarray(x_train).shape[0]),
        "n_test": int(test_targets.size),
        "input_dim": 2,
        "setup_seconds": float(diagnostics["setup_seconds"]),
        "solving_phase_seconds": float(diagnostics["solving_phase_seconds"]),
        "train_total_seconds": float(diagnostics["train_total_seconds"]),
        "prediction_seconds": prediction_seconds,
        **metrics,
        "diagnostics": diagnostics,
        "model": model,
    }


def _load_cupy_backend() -> tuple[Any, Any]:
    """Load CuPy lazily so CPU-only test and documentation environments work."""

    try:
        import cupy as cp
        from cupyx.scipy import sparse as cupy_sparse
    except ImportError as exc:  # pragma: no cover - depends on CUDA environment
        raise RuntimeError(
            "backend='cupy' requires a CUDA-compatible CuPy installation."
        ) from exc
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - depends on CUDA environment
        raise RuntimeError(
            "CuPy is installed but no usable CUDA runtime was found."
        ) from exc
    if device_count <= 0:  # pragma: no cover - depends on CUDA environment
        raise RuntimeError("backend='cupy' requires at least one CUDA device.")
    return cp, cupy_sparse


def _validate_paired_row_shapes(x: Any, y: Any, *, label: str) -> int:
    x_shape = getattr(x, "shape", None)
    y_shape = getattr(y, "shape", None)
    if x_shape is None or len(x_shape) != 2 or int(x_shape[1]) != 2:
        raise ValueError(f"x_{label} must have shape (n_rows, 2).")
    if y_shape is None or len(y_shape) not in {1, 2}:
        raise ValueError(f"y_{label} must be one- or two-dimensional.")
    y_rows = int(y_shape[0])
    if len(y_shape) == 2 and int(y_shape[1]) != 1:
        raise ValueError(f"y_{label} must have shape (n_rows,) or (n_rows, 1).")
    if int(x_shape[0]) != y_rows:
        raise ValueError(f"x_{label} and y_{label} row counts differ.")
    return int(x_shape[0])


def _cupy_linear_interpolation_rows(
    grid: SKIGrid2D, x: Any, array_module: Any
) -> tuple[Any, Any, Any, Any, Any]:
    """Transient four-weight interpolation rows for a CuPy chunk."""

    cp = array_module
    left_x, left_y, cell_ids, tx, ty = _cupy_linear_cell_coordinates(grid, x, cp)
    wx = cp.stack((1.0 - tx, tx), axis=1)
    wy = cp.stack((1.0 - ty, ty), axis=1)
    x_nodes = cp.stack((left_x, left_x + 1), axis=1)
    y_nodes = cp.stack((left_y, left_y + 1), axis=1)
    indices = (x_nodes[:, :, None] * int(grid.ny) + y_nodes[:, None, :]).reshape(
        int(tx.size), 4
    )
    weights = (wx[:, :, None] * wy[:, None, :]).reshape(int(tx.size), 4)
    return indices, weights, cell_ids, tx, ty


def _cupy_linear_cell_coordinates(
    grid: SKIGrid2D, x: Any, array_module: Any
) -> tuple[Any, Any, Any, Any, Any]:
    """Map one GPU chunk to safe bilinear cells without materializing ``W``."""

    cp = array_module
    points = cp.asarray(x, dtype=cp.float64)
    if points.ndim != 2 or int(points.shape[1]) != 2:
        raise ValueError("SKI inputs must have shape (n_rows, 2).")
    if _cupy_bool(cp.any(~cp.isfinite(points))):
        raise ValueError("SKI input chunks must contain only finite values.")
    scaled_x = (points[:, 0] - float(grid.x_start)) / float(grid.spacing)
    scaled_y = (points[:, 1] - float(grid.y_start)) / float(grid.spacing)
    left_x = cp.floor(scaled_x).astype(cp.int64)
    left_y = cp.floor(scaled_y).astype(cp.int64)
    tx = scaled_x - left_x
    ty = scaled_y - left_y
    tolerance = 32.0 * np.finfo(float).eps
    near_one_x = cp.isclose(tx, 1.0, rtol=0.0, atol=tolerance)
    near_one_y = cp.isclose(ty, 1.0, rtol=0.0, atol=tolerance)
    left_x = left_x + near_one_x.astype(cp.int64)
    left_y = left_y + near_one_y.astype(cp.int64)
    tx = cp.where(near_one_x, 0.0, tx)
    ty = cp.where(near_one_y, 0.0, ty)
    outside = (
        cp.any(left_x < 0)
        | cp.any(left_y < 0)
        | cp.any(left_x + 1 >= int(grid.nx))
        | cp.any(left_y + 1 >= int(grid.ny))
    )
    if _cupy_bool(outside):
        raise ValueError(
            "input lies outside the interpolation-safe grid bounds; expand the "
            "declared data bounds or padding"
        )
    cell_ids = left_x * (int(grid.ny) - 1) + left_y
    return left_x, left_y, cell_ids, tx, ty


def _accumulate_cupy_linear_normal_equations(
    x: Any,
    y: Any,
    grid: SKIGrid2D,
    *,
    chunk_size: int,
    array_module: Any,
    sparse_module: Any,
) -> CuPyInterpolationNormalEquations2D:
    """Reduce 13 exact bilinear moments per row with GPU bincount kernels."""

    cp = array_module
    n_rows = _validate_paired_row_shapes(x, y, label="train")
    n_cells = (int(grid.nx) - 1) * (int(grid.ny) - 1)
    gram_moments = cp.zeros((n_cells, 3, 3), dtype=cp.float64)
    rhs_moments = cp.zeros((n_cells, 2, 2), dtype=cp.float64)
    for start in range(0, n_rows, int(chunk_size)):
        stop = min(start + int(chunk_size), n_rows)
        _left_x, _left_y, cell_ids, tx, ty = _cupy_linear_cell_coordinates(
            grid, x[start:stop], cp
        )
        targets = cp.asarray(y[start:stop], dtype=cp.float64).reshape(-1)
        if _cupy_bool(cp.any(~cp.isfinite(targets))):
            raise ValueError("SKI target chunks must contain only finite values.")
        tx_powers = (cp.ones_like(tx), tx, tx * tx)
        ty_powers = (cp.ones_like(ty), ty, ty * ty)
        for px in range(3):
            for py in range(3):
                gram_moments[:, px, py] += cp.bincount(
                    cell_ids,
                    weights=tx_powers[px] * ty_powers[py],
                    minlength=n_cells,
                )
        for px in range(2):
            for py in range(2):
                rhs_moments[:, px, py] += cp.bincount(
                    cell_ids,
                    weights=targets * tx_powers[px] * ty_powers[py],
                    minlength=n_cells,
                )

    occupied = cp.flatnonzero(gram_moments[:, 0, 0] > 0.0).astype(cp.int64)
    coefficients_numpy = _tensor_interpolation_coefficients("linear")
    coefficients = cp.asarray(coefficients_numpy, dtype=cp.float64)
    gram_map = cp.asarray(_gram_coefficient_map(coefficients_numpy), dtype=cp.float64)
    gram_blocks = cp.einsum(
        "cpq,abpq->cab", gram_moments[occupied], gram_map, optimize=True
    )
    gram_blocks = 0.5 * (gram_blocks + cp.swapaxes(gram_blocks, 1, 2))
    rhs_blocks = cp.einsum(
        "cpq,apq->ca", rhs_moments[occupied], coefficients, optimize=True
    )
    cell_nodes = _cupy_linear_cell_node_indices(grid, occupied, cp)
    rhs = cp.zeros(int(grid.size), dtype=cp.float64)
    cp.add.at(rhs, cell_nodes.reshape(-1), rhs_blocks.reshape(-1))

    rows = cp.repeat(cell_nodes, 4, axis=1).reshape(-1)
    columns = cp.tile(cell_nodes, (1, 4)).reshape(-1)
    sparse_matrix = sparse_module.coo_matrix(
        (gram_blocks.reshape(-1), (rows, columns)),
        shape=(int(grid.size), int(grid.size)),
        dtype=cp.float64,
    ).tocsr()
    sparse_matrix.sum_duplicates()
    trace_value = _cupy_scalar(
        cp.trace(gram_blocks, axis1=1, axis2=2).sum(dtype=cp.float64)
    )
    return CuPyInterpolationNormalEquations2D(
        grid=grid,
        sparse_matrix=sparse_matrix,
        rhs=cp.ascontiguousarray(rhs),
        cell_ids=cp.ascontiguousarray(occupied),
        n_rows=n_rows,
        trace_value=trace_value,
    )


def _cupy_linear_cell_node_indices(
    grid: SKIGrid2D, cell_ids: Any, array_module: Any
) -> Any:
    cp = array_module
    ids = cp.asarray(cell_ids, dtype=cp.int64).reshape(-1)
    left_x = ids // (int(grid.ny) - 1)
    left_y = ids % (int(grid.ny) - 1)
    x_nodes = cp.stack((left_x, left_x + 1), axis=1)
    y_nodes = cp.stack((left_y, left_y + 1), axis=1)
    return cp.ascontiguousarray(
        (x_nodes[:, :, None] * int(grid.ny) + y_nodes[:, None, :]).reshape(-1, 4)
    )


def _conjugate_gradient_cupy(
    matvec: Callable[[Any], Any],
    rhs: Any,
    *,
    tolerance: float,
    maxiter: int,
    preconditioner: Callable[[Any], Any] | None,
    array_module: Any,
) -> tuple[Any, dict[str, Any]]:
    cp = array_module
    vector = cp.asarray(rhs, dtype=cp.float64).reshape(-1)
    solution = cp.zeros_like(vector)
    rhs_norm = _cupy_scalar(cp.linalg.norm(vector))
    if rhs_norm == 0.0:
        return solution, {
            "converged": True,
            "iterations": 0,
            "relative_residual": 0.0,
        }
    residual = vector.copy()
    preconditioned = (
        residual.copy() if preconditioner is None else preconditioner(residual)
    )
    rho = _cupy_scalar(cp.dot(residual, preconditioned))
    if not math.isfinite(rho) or rho <= 0.0:
        raise RuntimeError("SKI CG preconditioner is not positive definite.")
    direction = preconditioned.copy()
    relative_residual = 1.0
    iterations = 0
    for iteration in range(1, int(maxiter) + 1):
        product = matvec(direction)
        denominator = _cupy_scalar(cp.dot(direction, product))
        scale = max(
            _cupy_scalar(cp.linalg.norm(direction))
            * _cupy_scalar(cp.linalg.norm(product)),
            np.finfo(np.float64).tiny,
        )
        if not math.isfinite(denominator) or denominator <= np.finfo(float).eps * scale:
            raise RuntimeError("SKI CG encountered a non-positive system curvature.")
        alpha = rho / denominator
        solution += alpha * direction
        residual -= alpha * product
        relative_residual = _cupy_scalar(cp.linalg.norm(residual)) / rhs_norm
        iterations = iteration
        if relative_residual <= float(tolerance):
            break
        next_preconditioned = (
            residual.copy() if preconditioner is None else preconditioner(residual)
        )
        next_rho = _cupy_scalar(cp.dot(residual, next_preconditioned))
        if not math.isfinite(next_rho) or next_rho <= 0.0:
            raise RuntimeError("SKI CG preconditioner lost positive definiteness.")
        direction = next_preconditioned + (next_rho / rho) * direction
        rho = next_rho
    return solution, {
        "converged": bool(relative_residual <= float(tolerance)),
        "iterations": int(iterations),
        "relative_residual": float(relative_residual),
    }


def _predict_cupy_model(
    model: StructuredKernelInterpolationModel,
    x: Any,
    *,
    chunk_size: int | None,
) -> Any:
    cp = model.array_module
    shape = getattr(x, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[1]) != 2:
        raise ValueError("SKI inputs must have shape (n_rows, 2).")
    use_chunk = int(chunk_size or model.config.prediction_chunk_size)
    if use_chunk <= 0:
        raise ValueError("prediction chunk_size must be positive.")
    output = cp.empty(int(shape[0]), dtype=cp.float64)
    for start in range(0, int(shape[0]), use_chunk):
        stop = min(start + use_chunk, int(shape[0]))
        output[start:stop] = model.predict_chunk(x[start:stop])
    return output


def _evaluate_cupy_model(
    model: StructuredKernelInterpolationModel,
    *,
    x_train: Any,
    x_test: Any,
    y_test: Any,
) -> dict[str, Any]:
    cp = model.array_module
    n_test = _validate_paired_row_shapes(x_test, y_test, label="test")
    if n_test <= 0:
        raise ValueError("SKI evaluation requires at least one test row.")
    squared_error = cp.asarray(0.0, dtype=cp.float64)
    absolute_error = cp.asarray(0.0, dtype=cp.float64)
    target_sum = cp.asarray(0.0, dtype=cp.float64)
    target_square_sum = cp.asarray(0.0, dtype=cp.float64)
    _synchronize_cupy(cp)
    prediction_start = time.perf_counter()
    chunk = int(model.config.prediction_chunk_size)
    for start in range(0, n_test, chunk):
        stop = min(start + chunk, n_test)
        prediction = model.predict_chunk(x_test[start:stop])
        truth = cp.asarray(y_test[start:stop], dtype=cp.float64).reshape(-1)
        if _cupy_bool(cp.any(~cp.isfinite(truth))):
            raise ValueError("y_test must contain only finite values.")
        residual = prediction - truth
        squared_error += cp.dot(residual, residual)
        absolute_error += cp.abs(residual).sum(dtype=cp.float64)
        target_sum += truth.sum(dtype=cp.float64)
        target_square_sum += cp.dot(truth, truth)
    _synchronize_cupy(cp)
    prediction_seconds = float(time.perf_counter() - prediction_start)
    metrics = _metrics_from_sums(
        n_rows=n_test,
        squared_error=_cupy_scalar(squared_error),
        absolute_error=_cupy_scalar(absolute_error),
        target_sum=_cupy_scalar(target_sum),
        target_square_sum=_cupy_scalar(target_square_sum),
    )
    diagnostics = model.fit_diagnostics
    return {
        "status": "ok",
        "implementation": diagnostics["implementation"],
        "method_label": diagnostics["method_label"],
        "n_train": int(getattr(x_train, "shape")[0]),
        "n_test": n_test,
        "input_dim": 2,
        "setup_seconds": float(diagnostics["setup_seconds"]),
        "solving_phase_seconds": float(diagnostics["solving_phase_seconds"]),
        "train_total_seconds": float(diagnostics["train_total_seconds"]),
        "prediction_seconds": prediction_seconds,
        **metrics,
        "diagnostics": diagnostics,
        "model": model,
    }


def _matern_from_radius_cupy(
    radius: Any,
    *,
    lengthscale: float,
    nu: float,
    variance: float,
    array_module: Any,
) -> Any:
    cp = array_module
    value = cp.asarray(radius, dtype=cp.float64)
    if math.isclose(float(nu), 0.5):
        scaled = value / float(lengthscale)
        return float(variance) * cp.exp(-scaled)
    if math.isclose(float(nu), 1.5):
        scaled = math.sqrt(3.0) * value / float(lengthscale)
        return float(variance) * (1.0 + scaled) * cp.exp(-scaled)
    if math.isclose(float(nu), 2.5):
        scaled = math.sqrt(5.0) * value / float(lengthscale)
        return (
            float(variance) * (1.0 + scaled + scaled * scaled / 3.0) * cp.exp(-scaled)
        )
    raise ValueError("unsupported Matérn nu")


def _cupy_scalar(value: Any) -> float:
    item = getattr(value, "item", None)
    return float(item() if item is not None else value)


def _cupy_bool(value: Any) -> bool:
    item = getattr(value, "item", None)
    return bool(item() if item is not None else value)


def _synchronize_cupy(array_module: Any) -> None:
    array_module.cuda.Stream.null.synchronize()


def _validate_config(config: StructuredKernelInterpolationConfig) -> None:
    mode = _normalize_interpolation(config.interpolation)
    backend = str(config.backend).strip().lower()
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend must be 'numpy' or 'cupy'.")
    if backend == "cupy" and mode != "linear":
        raise ValueError(
            "CuPy production SKI currently supports interpolation='linear' only."
        )
    if mode == "cubic" and int(config.grid_padding_points) < 2:
        raise ValueError("cubic interpolation requires at least two padding points.")
    if int(config.train_chunk_size) <= 0 or int(config.prediction_chunk_size) <= 0:
        raise ValueError("train and prediction chunk sizes must be positive.")
    if (
        not math.isfinite(float(config.absolute_ridge))
        or float(config.absolute_ridge) <= 0.0
    ):
        raise ValueError("absolute_ridge must be finite and positive.")
    if (
        not math.isfinite(float(config.cg_tolerance))
        or float(config.cg_tolerance) <= 0.0
    ):
        raise ValueError("cg_tolerance must be finite and positive.")
    if int(config.cg_maxiter) <= 0:
        raise ValueError("cg_maxiter must be positive.")
    preconditioner = str(config.cg_preconditioner).strip().lower()
    if preconditioner not in {"none", "circulant_density"}:
        raise ValueError("cg_preconditioner must be 'none' or 'circulant_density'.")
    floor = float(config.circulant_spectral_floor_relative)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("circulant_spectral_floor_relative must be positive.")
    build_ski_grid_2d(
        config.grid_bounds,
        spacing=float(config.grid_spacing),
        padding_points=int(config.grid_padding_points),
    )
    _validate_kernel_parameters(config.lengthscale, config.nu, config.kernel_variance)


def _validate_kernel_parameters(lengthscale: float, nu: float, variance: float) -> None:
    if not math.isfinite(float(lengthscale)) or float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be finite and positive.")
    if float(nu) not in {0.5, 1.5, 2.5}:
        raise ValueError("native SKI Matérn supports nu in {0.5, 1.5, 2.5}.")
    if not math.isfinite(float(variance)) or float(variance) <= 0.0:
        raise ValueError("kernel_variance must be finite and positive.")


def _matern_from_radius(
    radius: np.ndarray,
    *,
    lengthscale: float,
    nu: float,
    variance: float,
) -> np.ndarray:
    value = np.asarray(radius, dtype=np.float64)
    if math.isclose(float(nu), 0.5):
        scaled = value / float(lengthscale)
        return float(variance) * np.exp(-scaled)
    if math.isclose(float(nu), 1.5):
        scaled = math.sqrt(3.0) * value / float(lengthscale)
        return float(variance) * (1.0 + scaled) * np.exp(-scaled)
    if math.isclose(float(nu), 2.5):
        scaled = math.sqrt(5.0) * value / float(lengthscale)
        return (
            float(variance) * (1.0 + scaled + scaled * scaled / 3.0) * np.exp(-scaled)
        )
    raise ValueError("unsupported Matérn nu")


def _normalize_interpolation(interpolation: str) -> str:
    mode = str(interpolation).strip().lower()
    if mode not in _INTERPOLATION_DEGREES:
        raise ValueError("interpolation must be 'linear' or 'cubic'.")
    return mode


def _validate_x(x: np.ndarray) -> np.ndarray:
    points = np.asarray(x)
    if points.ndim != 2 or int(points.shape[1]) != 2:
        raise ValueError("SKI inputs must have shape (n_rows, 2).")
    return points


def _keys_cubic_weights(t: np.ndarray) -> np.ndarray:
    value = np.asarray(t, dtype=np.float64)
    value2 = value * value
    value3 = value2 * value
    return np.column_stack(
        (
            -0.5 * value + value2 - 0.5 * value3,
            1.0 - 2.5 * value2 + 1.5 * value3,
            0.5 * value + 2.0 * value2 - 1.5 * value3,
            -0.5 * value2 + 0.5 * value3,
        )
    )


def _axis_interpolation_coefficients(interpolation: str) -> np.ndarray:
    if _normalize_interpolation(interpolation) == "linear":
        return np.asarray([[1.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    # Rows correspond to nodes left-1, left, left+1, left+2; columns to t^0..t^3.
    return np.asarray(
        [
            [0.0, -0.5, 1.0, -0.5],
            [1.0, 0.0, -2.5, 1.5],
            [0.0, 0.5, 2.0, -1.5],
            [0.0, 0.0, -0.5, 0.5],
        ],
        dtype=np.float64,
    )


def _tensor_interpolation_coefficients(interpolation: str) -> np.ndarray:
    axis = _axis_interpolation_coefficients(interpolation)
    tensor = np.einsum("ip,jq->ijpq", axis, axis, optimize=True)
    return np.ascontiguousarray(
        tensor.reshape(axis.shape[0] * axis.shape[0], axis.shape[1], axis.shape[1])
    )


def _gram_coefficient_map(coefficients: np.ndarray) -> np.ndarray:
    n_weights, degree_plus_one, _ = coefficients.shape
    degree = int(degree_plus_one) - 1
    output = np.zeros(
        (n_weights, n_weights, 2 * degree + 1, 2 * degree + 1),
        dtype=np.float64,
    )
    for first in range(n_weights):
        for second in range(n_weights):
            for px in range(degree + 1):
                for py in range(degree + 1):
                    first_value = coefficients[first, px, py]
                    if first_value == 0.0:
                        continue
                    for qx in range(degree + 1):
                        for qy in range(degree + 1):
                            output[first, second, px + qx, py + qy] += (
                                first_value * coefficients[second, qx, qy]
                            )
    return output


def _cell_node_indices(
    grid: SKIGrid2D,
    cell_ids: np.ndarray,
    *,
    interpolation: str,
) -> np.ndarray:
    ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    left_x = ids // (int(grid.ny) - 1)
    left_y = ids % (int(grid.ny) - 1)
    if _normalize_interpolation(interpolation) == "linear":
        offsets = np.asarray([0, 1], dtype=np.int64)
    else:
        offsets = np.asarray([-1, 0, 1, 2], dtype=np.int64)
    x_nodes = left_x[:, None] + offsets[None, :]
    y_nodes = left_y[:, None] + offsets[None, :]
    if (
        np.any(x_nodes < 0)
        or np.any(y_nodes < 0)
        or np.any(x_nodes >= int(grid.nx))
        or np.any(y_nodes >= int(grid.ny))
    ):
        raise ValueError("occupied interpolation cell lacks the required grid padding.")
    return np.ascontiguousarray(
        (x_nodes[:, :, None] * int(grid.ny) + y_nodes[:, None, :]).reshape(
            int(ids.size), -1
        )
    )


def _conjugate_gradient(
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    *,
    tolerance: float,
    maxiter: int,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    vector = np.asarray(rhs, dtype=np.float64).reshape(-1)
    solution = np.zeros_like(vector)
    rhs_norm = float(np.linalg.norm(vector))
    if rhs_norm == 0.0:
        return solution, {"converged": True, "iterations": 0, "relative_residual": 0.0}
    residual = vector.copy()
    preconditioned = (
        residual.copy() if preconditioner is None else preconditioner(residual)
    )
    rho = float(np.dot(residual, preconditioned))
    if not math.isfinite(rho) or rho <= 0.0:
        raise RuntimeError("SKI CG preconditioner is not positive definite.")
    direction = preconditioned.copy()
    relative_residual = 1.0
    iterations = 0
    for iteration in range(1, int(maxiter) + 1):
        product = matvec(direction)
        denominator = float(np.dot(direction, product))
        scale = max(
            float(np.linalg.norm(direction)) * float(np.linalg.norm(product)),
            np.finfo(np.float64).tiny,
        )
        if not math.isfinite(denominator) or denominator <= np.finfo(float).eps * scale:
            raise RuntimeError("SKI CG encountered a non-positive system curvature.")
        alpha = rho / denominator
        solution += alpha * direction
        residual -= alpha * product
        relative_residual = float(np.linalg.norm(residual) / rhs_norm)
        iterations = iteration
        if relative_residual <= float(tolerance):
            break
        next_preconditioned = (
            residual.copy() if preconditioner is None else preconditioner(residual)
        )
        next_rho = float(np.dot(residual, next_preconditioned))
        if not math.isfinite(next_rho) or next_rho <= 0.0:
            raise RuntimeError("SKI CG preconditioner lost positive definiteness.")
        direction = next_preconditioned + (next_rho / rho) * direction
        preconditioned = next_preconditioned
        rho = next_rho
    return solution, {
        "converged": bool(relative_residual <= float(tolerance)),
        "iterations": int(iterations),
        "relative_residual": float(relative_residual),
    }


def _metrics_from_sums(
    *,
    n_rows: int,
    squared_error: float,
    absolute_error: float,
    target_sum: float,
    target_square_sum: float,
) -> dict[str, float]:
    count = int(n_rows)
    if count <= 0:
        raise ValueError("n_rows must be positive.")
    sse = max(float(squared_error), 0.0)
    target_centered = max(
        float(target_square_sum) - float(target_sum) ** 2 / count, 0.0
    )
    denominator = max(target_centered, np.finfo(np.float64).tiny)
    rmse = math.sqrt(sse / count)
    mae = max(float(absolute_error), 0.0) / count
    r2 = 1.0 - sse / denominator
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
    }


__all__ = [
    "BTTBMaternOperator2D",
    "CuPyBTTBMaternOperator2D",
    "CuPyInterpolationNormalEquations2D",
    "InterpolationMomentAccumulator2D",
    "InterpolationNormalEquations2D",
    "SKIGrid2D",
    "StructuredKernelInterpolationConfig",
    "StructuredKernelInterpolationModel",
    "accumulate_interpolation_normal_equations",
    "accumulate_interpolation_normal_equations_from_chunks",
    "build_ski_grid_2d",
    "fit_structured_kernel_interpolation",
    "interpolation_rows",
    "run_structured_kernel_interpolation",
]
