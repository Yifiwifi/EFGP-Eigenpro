"""Column-randomized Nyström PCG for the original data-space KRR system.

This module deliberately solves the full system

    (K(X, X) + absolute_ridge * I) alpha = y.

The Nyström matrix is used only as a preconditioner.  In particular, every
Krylov product is an exact (up to floating-point evaluation) double-blocked
isotropic Matérn-3/2 kernel product; the system is never replaced by a
low-rank or Fourier approximation.

The KRR experiments in Frangella, Tropp, and Udell use random column sampling
because kernel columns are cheap to request relative to Gaussian sketches.
At the scales targeted by the surrounding project, exact data-space products
are nevertheless quadratic.  The prospective resource gate below therefore
runs before backend resolution or device staging and fails closed when either
one exact product or the ``N x rank`` Nyström factor exceeds its declared cap.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np


METHOD_NAME = "column-randomized-nystrom-pcg-original-krr"
_CITATIONS = ("frangella2023randomized",)


class OriginalKRRResourceLimit(MemoryError):
    """Prospective exact-KRR resource gate failure with auditable estimates."""

    def __init__(self, reason: str, audit: dict[str, Any]) -> None:
        self.reason = str(reason)
        self.audit = dict(audit)
        super().__init__(
            "Original data-space KRR resource preflight failed "
            f"({self.reason}): {self.audit}. No backend was resolved and no "
            "training data were staged."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "resource_limit",
            "method": METHOD_NAME,
            "resource_limit_reason": self.reason,
            **self.audit,
        }


@dataclass(frozen=True)
class OriginalKRRNystromConfig:
    """Configuration for one exact original-KRR Nyström-PCG run."""

    rank: int = 128
    seed: int = 17
    absolute_ridge: float = 0.1
    tolerance: float = 1e-3
    maxiter: int = 250
    lengthscale: float = 0.1
    kernel_variance: float = 1.0
    precision: str = "fp64"
    backend: str = "auto"
    matvec_row_chunk_size: int = 2_048
    matvec_column_chunk_size: int = 2_048
    nystrom_row_chunk_size: int = 32_768
    prediction_row_chunk_size: int = 2_048
    prediction_column_chunk_size: int = 2_048
    nystrom_rcond: float = 1e-12
    # One exact training matvec is N^2 pair evaluations.  This prospective
    # cap intentionally gates the formal 10M--300M cases before GPU staging.
    max_exact_matvec_pairs: int | None = 1_000_000_000
    max_prediction_pairs: int | None = 1_000_000_000
    # Persistent storage for the one in-place N x rank factor.  Temporary
    # row-block workspaces are reported separately in the resource audit.
    max_preconditioner_bytes: int | None = 4 * 1024**3


@dataclass
class ColumnNystromPreconditioner:
    """Factored column-Nyström approximation and FTU inverse action."""

    basis: Any
    eigenvalues: Any
    absolute_ridge: float
    sample_indices: np.ndarray
    requested_rank: int
    array_module: Any

    @property
    def effective_rank(self) -> int:
        return int(self.eigenvalues.shape[0])

    def apply_inverse(self, vector: Any) -> Any:
        """Apply equation (5.3)'s normalized Nyström preconditioner inverse."""

        xp = self.array_module
        value = xp.asarray(vector, dtype=xp.float64).reshape(-1)
        projected = self.basis.T @ value
        tail = self.eigenvalues[-1]
        scale = (tail + self.absolute_ridge) / (
            self.eigenvalues + self.absolute_ridge
        )
        return value + self.basis @ ((scale - 1.0) * projected)

    def approximation_matvec(self, vector: Any) -> Any:
        """Apply C K_SS^dagger C.T through its eigendecomposition."""

        xp = self.array_module
        value = xp.asarray(vector, dtype=xp.float64).reshape(-1)
        return self.basis @ (self.eigenvalues * (self.basis.T @ value))


@dataclass
class OriginalKRRNystromModel:
    """Fitted original-KRR model retained on its fit backend."""

    x_train: Any
    alpha: Any
    config: OriginalKRRNystromConfig
    array_module: Any
    preconditioner: ColumnNystromPreconditioner


@dataclass
class ExactMatern32Operator:
    """Exact double-blocked isotropic Matérn-3/2 training operator."""

    x_train: Any
    lengthscale: float
    variance: float
    row_chunk_size: int
    column_chunk_size: int
    array_module: Any
    matvec_calls: int = 0
    matvec_pair_evaluations: int = 0

    def matvec(self, vector: Any) -> Any:
        value = exact_matern32_cross_matvec(
            self.x_train,
            self.x_train,
            vector,
            lengthscale=self.lengthscale,
            variance=self.variance,
            row_chunk_size=self.row_chunk_size,
            column_chunk_size=self.column_chunk_size,
            array_module=self.array_module,
        )
        n_train = int(self.x_train.shape[0])
        self.matvec_calls += 1
        self.matvec_pair_evaluations += n_train * n_train
        return value


def _validate_config(cfg: OriginalKRRNystromConfig) -> None:
    if int(cfg.rank) <= 0:
        raise ValueError("rank must be positive.")
    if not math.isfinite(float(cfg.absolute_ridge)) or float(
        cfg.absolute_ridge
    ) <= 0.0:
        raise ValueError("absolute_ridge must be positive and finite.")
    if not math.isfinite(float(cfg.tolerance)) or not (
        0.0 < float(cfg.tolerance) < 1.0
    ):
        raise ValueError("tolerance must lie strictly between zero and one.")
    if int(cfg.maxiter) <= 0:
        raise ValueError("maxiter must be positive.")
    if not math.isfinite(float(cfg.lengthscale)) or float(cfg.lengthscale) <= 0.0:
        raise ValueError("lengthscale must be positive and finite.")
    if not math.isfinite(float(cfg.kernel_variance)) or float(
        cfg.kernel_variance
    ) <= 0.0:
        raise ValueError("kernel_variance must be positive and finite.")
    if str(cfg.precision).strip().lower() != "fp64":
        raise ValueError("original data-space KRR currently requires precision='fp64'.")
    if str(cfg.backend).strip().lower() not in {"auto", "numpy", "cupy"}:
        raise ValueError("backend must be one of: auto, numpy, cupy.")
    for field_name in (
        "matvec_row_chunk_size",
        "matvec_column_chunk_size",
        "nystrom_row_chunk_size",
        "prediction_row_chunk_size",
        "prediction_column_chunk_size",
    ):
        if int(getattr(cfg, field_name)) <= 0:
            raise ValueError(f"{field_name} must be positive.")
    if not math.isfinite(float(cfg.nystrom_rcond)) or not (
        0.0 < float(cfg.nystrom_rcond) < 1.0
    ):
        raise ValueError("nystrom_rcond must lie strictly between zero and one.")
    for field_name in (
        "max_exact_matvec_pairs",
        "max_prediction_pairs",
        "max_preconditioner_bytes",
    ):
        value = getattr(cfg, field_name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{field_name} must be positive or None.")


def _shape_xy(x: Any, y: Any, *, prefix: str) -> tuple[int, int]:
    x_shape = getattr(x, "shape", None)
    y_shape = getattr(y, "shape", None)
    if x_shape is None or len(x_shape) != 2:
        raise ValueError(f"{prefix} x must have shape (rows, features).")
    if y_shape is None or len(y_shape) not in {1, 2}:
        raise ValueError(f"{prefix} y must have shape (rows,) or (rows, 1).")
    n_rows = int(x_shape[0])
    n_features = int(x_shape[1])
    if n_rows <= 0 or n_features != 2:
        raise ValueError(f"{prefix} requires nonempty isotropic 2-D coordinates.")
    if int(y_shape[0]) != n_rows or (len(y_shape) == 2 and int(y_shape[1]) != 1):
        raise ValueError(f"{prefix} x/y row counts must match.")
    return n_rows, n_features


def preflight_original_krr_resources(
    n_train: int,
    n_test: int,
    cfg: OriginalKRRNystromConfig,
) -> dict[str, Any]:
    """Audit exact pair counts and factor memory before backend/device work."""

    _validate_config(cfg)
    n_train = int(n_train)
    n_test = int(n_test)
    if n_train <= 0 or n_test < 0:
        raise ValueError("n_train must be positive and n_test nonnegative.")
    if int(cfg.rank) > n_train:
        raise ValueError("rank cannot exceed n_train.")

    exact_matvec_pairs = n_train * n_train
    prediction_pairs = n_train * n_test
    dense_kernel_matrix_bytes = (
        exact_matvec_pairs * np.dtype(np.float64).itemsize
    )
    factor_bytes = n_train * int(cfg.rank) * np.dtype(np.float64).itemsize
    nystrom_workspace_bytes = (
        min(n_train, int(cfg.nystrom_row_chunk_size))
        * int(cfg.rank)
        * np.dtype(np.float64).itemsize
    )
    matvec_kernel_block_bytes = (
        min(n_train, int(cfg.matvec_row_chunk_size))
        * min(n_train, int(cfg.matvec_column_chunk_size))
        * np.dtype(np.float64).itemsize
    )
    audit = {
        "n_train": n_train,
        "n_test": n_test,
        "requested_rank": int(cfg.rank),
        "precision": "fp64",
        "exact_matvec_pairs": exact_matvec_pairs,
        "dense_kernel_matrix_bytes": dense_kernel_matrix_bytes,
        "max_exact_matvec_pairs": cfg.max_exact_matvec_pairs,
        "prediction_pairs": prediction_pairs,
        "max_prediction_pairs": cfg.max_prediction_pairs,
        "preconditioner_factor_bytes": factor_bytes,
        "max_preconditioner_bytes": cfg.max_preconditioner_bytes,
        "nystrom_row_workspace_bytes": nystrom_workspace_bytes,
        "matvec_kernel_block_bytes": matvec_kernel_block_bytes,
        "resource_preflight_before_dataset_load": True,
        "resource_preflight_before_backend": True,
    }
    if (
        cfg.max_exact_matvec_pairs is not None
        and exact_matvec_pairs > int(cfg.max_exact_matvec_pairs)
    ):
        raise OriginalKRRResourceLimit("exact_matvec_pair_cap", audit)
    if (
        cfg.max_preconditioner_bytes is not None
        and factor_bytes > int(cfg.max_preconditioner_bytes)
    ):
        raise OriginalKRRResourceLimit("preconditioner_factor_memory_cap", audit)
    if (
        n_test > 0
        and cfg.max_prediction_pairs is not None
        and prediction_pairs > int(cfg.max_prediction_pairs)
    ):
        raise OriginalKRRResourceLimit("prediction_pair_cap", audit)
    return audit


def _resolve_array_module(
    backend: str,
    array_module: Any | None = None,
) -> tuple[Any, str]:
    if array_module is not None:
        name = "numpy" if array_module is np else getattr(array_module, "__name__", "injected")
        return array_module, str(name)
    requested = str(backend).strip().lower()
    if requested == "numpy":
        return np, "numpy"
    if requested in {"auto", "cupy"}:
        try:
            import cupy as cp  # type: ignore

            return cp, "cupy"
        except Exception as exc:  # noqa: BLE001
            if requested == "cupy":
                raise RuntimeError("backend='cupy' requested but CuPy is unavailable.") from exc
    return np, "numpy"


def _sync(xp: Any) -> None:
    if xp is np:
        return
    cuda = getattr(xp, "cuda", None)
    if cuda is not None:
        cuda.get_current_stream().synchronize()


def _scalar(value: Any) -> float:
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def matern32_cross(
    x_left: Any,
    x_right: Any,
    *,
    lengthscale: float,
    variance: float,
    array_module: Any = np,
) -> Any:
    """Dense isotropic two-dimensional Matérn-3/2 cross-kernel block."""

    xp = array_module
    left = xp.asarray(x_left, dtype=xp.float64)
    right = xp.asarray(x_right, dtype=xp.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != 2 or right.shape[1] != 2:
        raise ValueError("matern32_cross requires two arrays with shape (rows, 2).")
    delta = left[:, None, :] - right[None, :, :]
    distance = xp.sqrt(xp.sum(delta * delta, axis=2))
    scaled = (math.sqrt(3.0) / float(lengthscale)) * distance
    return float(variance) * (1.0 + scaled) * xp.exp(-scaled)


def exact_matern32_cross_matvec(
    x_left: Any,
    x_right: Any,
    vector: Any,
    *,
    lengthscale: float,
    variance: float,
    row_chunk_size: int,
    column_chunk_size: int,
    array_module: Any = np,
) -> Any:
    """Apply a Matérn cross-kernel using bounded two-dimensional blocks."""

    xp = array_module
    left = xp.asarray(x_left, dtype=xp.float64)
    right = xp.asarray(x_right, dtype=xp.float64)
    value = xp.asarray(vector, dtype=xp.float64).reshape(-1)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != 2 or right.shape[1] != 2:
        raise ValueError("kernel product coordinates must have shape (rows, 2).")
    if int(right.shape[0]) != int(value.shape[0]):
        raise ValueError("kernel product vector length must match right coordinates.")
    row_chunk_size = int(row_chunk_size)
    column_chunk_size = int(column_chunk_size)
    if row_chunk_size <= 0 or column_chunk_size <= 0:
        raise ValueError("kernel product chunk sizes must be positive.")

    output = xp.empty(int(left.shape[0]), dtype=xp.float64)
    for row_start in range(0, int(left.shape[0]), row_chunk_size):
        row_stop = min(int(left.shape[0]), row_start + row_chunk_size)
        accumulated = xp.zeros(row_stop - row_start, dtype=xp.float64)
        left_block = left[row_start:row_stop]
        for column_start in range(0, int(right.shape[0]), column_chunk_size):
            column_stop = min(
                int(right.shape[0]), column_start + column_chunk_size
            )
            kernel_block = matern32_cross(
                left_block,
                right[column_start:column_stop],
                lengthscale=lengthscale,
                variance=variance,
                array_module=xp,
            )
            accumulated += kernel_block @ value[column_start:column_stop]
        output[row_start:row_stop] = accumulated
    return output


def build_column_nystrom_preconditioner(
    x_train: Any,
    cfg: OriginalKRRNystromConfig,
    *,
    array_module: Any = np,
) -> ColumnNystromPreconditioner:
    """Build ``C K_SS^dagger C.T`` and its FTU inverse action in-place."""

    _validate_config(cfg)
    xp = array_module
    x_train = xp.asarray(x_train, dtype=xp.float64)
    if x_train.ndim != 2 or int(x_train.shape[1]) != 2:
        raise ValueError("x_train must have shape (rows, 2).")
    n_train = int(x_train.shape[0])
    rank = int(cfg.rank)
    if rank > n_train:
        raise ValueError("rank cannot exceed n_train.")

    rng = np.random.default_rng(int(cfg.seed))
    sample_indices = np.asarray(
        rng.choice(n_train, size=rank, replace=False), dtype=np.int64
    )
    device_indices = xp.asarray(sample_indices)
    landmarks = x_train[device_indices]

    # C is transformed in-place first into F=C K_SS^{-1/2}, then into the
    # orthonormal eigenbasis U.  This retains one N x requested_rank factor.
    factor = xp.empty((n_train, rank), dtype=xp.float64)
    chunk_size = int(cfg.nystrom_row_chunk_size)
    for start in range(0, n_train, chunk_size):
        stop = min(n_train, start + chunk_size)
        factor[start:stop] = matern32_cross(
            x_train[start:stop],
            landmarks,
            lengthscale=float(cfg.lengthscale),
            variance=float(cfg.kernel_variance),
            array_module=xp,
        )

    center_kernel = factor[device_indices, :].copy()
    center_kernel = 0.5 * (center_kernel + center_kernel.T)
    center_values, center_vectors = xp.linalg.eigh(center_kernel)
    center_max = max(_scalar(xp.max(center_values)), 0.0)
    center_threshold = max(
        float(cfg.nystrom_rcond) * center_max,
        np.finfo(np.float64).tiny,
    )
    center_keep = center_values > center_threshold
    center_rank = int(_scalar(xp.sum(center_keep)))
    if center_rank <= 0:
        raise RuntimeError("sampled center kernel has no positive numerical rank.")
    center_transform = center_vectors[:, center_keep] / xp.sqrt(
        center_values[center_keep]
    )[None, :]
    for start in range(0, n_train, chunk_size):
        stop = min(n_train, start + chunk_size)
        transformed = factor[start:stop, :] @ center_transform
        factor[start:stop, :center_rank] = transformed
    feature_factor = factor[:, :center_rank]

    gram = feature_factor.T @ feature_factor
    gram = 0.5 * (gram + gram.T)
    eigenvalues, right_vectors = xp.linalg.eigh(gram)
    order = xp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    right_vectors = right_vectors[:, order]
    eigen_max = max(_scalar(eigenvalues[0]), 0.0)
    eigen_threshold = max(
        float(cfg.nystrom_rcond) * eigen_max,
        np.finfo(np.float64).tiny,
    )
    eigen_keep = eigenvalues > eigen_threshold
    effective_rank = int(_scalar(xp.sum(eigen_keep)))
    if effective_rank <= 0:
        raise RuntimeError("column Nyström approximation has zero numerical rank.")
    kept_values = eigenvalues[eigen_keep]
    basis_transform = right_vectors[:, eigen_keep] / xp.sqrt(kept_values)[None, :]
    for start in range(0, n_train, chunk_size):
        stop = min(n_train, start + chunk_size)
        basis_block = feature_factor[start:stop, :] @ basis_transform
        factor[start:stop, :effective_rank] = basis_block
    basis = factor[:, :effective_rank]

    return ColumnNystromPreconditioner(
        basis=basis,
        eigenvalues=kept_values,
        absolute_ridge=float(cfg.absolute_ridge),
        sample_indices=sample_indices,
        requested_rank=rank,
        array_module=xp,
    )


def _pcg_original_krr(
    operator: ExactMatern32Operator,
    y_train: Any,
    preconditioner: ColumnNystromPreconditioner,
    cfg: OriginalKRRNystromConfig,
) -> tuple[Any, dict[str, Any]]:
    xp = operator.array_module
    rhs = xp.asarray(y_train, dtype=xp.float64).reshape(-1)
    solution = xp.zeros_like(rhs)
    residual = rhs.copy()
    rhs_norm = _scalar(xp.linalg.norm(rhs))
    if rhs_norm == 0.0:
        true_residual = operator.matvec(solution) + float(cfg.absolute_ridge) * solution
        return solution, {
            "iterations": 0,
            "converged": True,
            "true_relative_residual": _scalar(xp.linalg.norm(true_residual)),
        }

    preconditioned = preconditioner.apply_inverse(residual)
    direction = preconditioned.copy()
    residual_dot = _scalar(xp.vdot(residual, preconditioned).real)
    converged = False
    iterations = 0
    recurrence_relative_residual = 1.0
    for iterations in range(1, int(cfg.maxiter) + 1):
        product = operator.matvec(direction) + float(cfg.absolute_ridge) * direction
        denominator = _scalar(xp.vdot(direction, product).real)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("PCG encountered a nonpositive search-direction product.")
        step = residual_dot / denominator
        solution = solution + step * direction
        residual = residual - step * product
        recurrence_relative_residual = _scalar(xp.linalg.norm(residual)) / rhs_norm
        if recurrence_relative_residual <= float(cfg.tolerance):
            converged = True
            break
        next_preconditioned = preconditioner.apply_inverse(residual)
        next_residual_dot = _scalar(xp.vdot(residual, next_preconditioned).real)
        if not math.isfinite(next_residual_dot) or next_residual_dot <= 0.0:
            raise RuntimeError("PCG preconditioned residual lost positive definiteness.")
        direction = next_preconditioned + (next_residual_dot / residual_dot) * direction
        preconditioned = next_preconditioned
        residual_dot = next_residual_dot

    true_residual = (
        operator.matvec(solution)
        + float(cfg.absolute_ridge) * solution
        - rhs
    )
    true_relative_residual = _scalar(xp.linalg.norm(true_residual)) / rhs_norm
    converged = bool(converged and true_relative_residual <= float(cfg.tolerance) * 1.05)
    return solution, {
        "iterations": int(iterations),
        "converged": converged,
        "recurrence_relative_residual": float(recurrence_relative_residual),
        "true_relative_residual": float(true_relative_residual),
    }


def fit_original_krr_nystrom_pcg(
    x_train: Any,
    y_train: Any,
    cfg: OriginalKRRNystromConfig,
    *,
    array_module: Any | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> tuple[OriginalKRRNystromModel, dict[str, Any]]:
    """Fit exact original data-space KRR with column-Nyström PCG."""

    n_train, _ = _shape_xy(x_train, y_train, prefix="train")
    resource_audit = preflight_original_krr_resources(n_train, 0, cfg)

    setup_start = timer()
    xp, backend_name = _resolve_array_module(cfg.backend, array_module)
    x_device = xp.asarray(x_train, dtype=xp.float64)
    y_device = xp.asarray(y_train, dtype=xp.float64).reshape(-1)
    _sync(xp)
    staging_seconds = float(timer() - setup_start)

    preconditioner_start = timer()
    preconditioner = build_column_nystrom_preconditioner(
        x_device,
        cfg,
        array_module=xp,
    )
    _sync(xp)
    preconditioner_seconds = float(timer() - preconditioner_start)
    setup_seconds = staging_seconds + preconditioner_seconds

    operator = ExactMatern32Operator(
        x_train=x_device,
        lengthscale=float(cfg.lengthscale),
        variance=float(cfg.kernel_variance),
        row_chunk_size=int(cfg.matvec_row_chunk_size),
        column_chunk_size=int(cfg.matvec_column_chunk_size),
        array_module=xp,
    )
    solve_start = timer()
    alpha, solve_diagnostics = _pcg_original_krr(
        operator,
        y_device,
        preconditioner,
        cfg,
    )
    _sync(xp)
    solve_seconds = float(timer() - solve_start)

    model = OriginalKRRNystromModel(
        x_train=x_device,
        alpha=alpha,
        config=cfg,
        array_module=xp,
        preconditioner=preconditioner,
    )
    diagnostics = {
        **solve_diagnostics,
        "backend": backend_name,
        "data_staging_seconds": staging_seconds,
        "preconditioner_setup_seconds": preconditioner_seconds,
        "setup_seconds": setup_seconds,
        "solve_seconds": solve_seconds,
        "train_total_seconds": setup_seconds + solve_seconds,
        "exact_matvec_count": int(operator.matvec_calls),
        "kernel_pair_evaluations": int(operator.matvec_pair_evaluations),
        "nystrom_kernel_pair_evaluations": n_train * int(cfg.rank),
        "requested_nystrom_rank": int(cfg.rank),
        "effective_nystrom_rank": int(preconditioner.effective_rank),
        "sample_indices": preconditioner.sample_indices.tolist(),
        "resource_audit": resource_audit,
    }
    return model, diagnostics


def _regression_metrics(xp: Any, prediction: Any, truth: Any) -> dict[str, float]:
    residual = prediction - truth
    squared_error = _scalar(xp.sum(residual * residual, dtype=xp.float64))
    absolute_error = _scalar(xp.sum(xp.abs(residual), dtype=xp.float64))
    target_sum = _scalar(xp.sum(truth, dtype=xp.float64))
    target_square_sum = _scalar(xp.sum(truth * truth, dtype=xp.float64))
    n_rows = int(truth.shape[0])
    total_square = target_square_sum - target_sum * target_sum / n_rows
    return {
        "rmse": math.sqrt(squared_error / n_rows),
        "mae": absolute_error / n_rows,
        "r2": float("nan") if total_square <= 0.0 else 1.0 - squared_error / total_square,
    }


def score_original_krr_nystrom_pcg(
    model: OriginalKRRNystromModel,
    x_test: Any,
    y_test: Any,
    *,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Predict with the exact train-test kernel and compute regression metrics."""

    n_test, _ = _shape_xy(x_test, y_test, prefix="test")
    n_train = int(model.x_train.shape[0])
    cfg = model.config
    # Direct score calls retain the same prospective prediction gate.  The
    # all-in-one runner performs this check before fitting/device staging.
    preflight_original_krr_resources(n_train, n_test, cfg)
    xp = model.array_module
    start = timer()
    x_test_device = xp.asarray(x_test, dtype=xp.float64)
    y_test_device = xp.asarray(y_test, dtype=xp.float64).reshape(-1)
    prediction = exact_matern32_cross_matvec(
        x_test_device,
        model.x_train,
        model.alpha,
        lengthscale=float(cfg.lengthscale),
        variance=float(cfg.kernel_variance),
        row_chunk_size=int(cfg.prediction_row_chunk_size),
        column_chunk_size=int(cfg.prediction_column_chunk_size),
        array_module=xp,
    )
    _sync(xp)
    prediction_seconds = float(timer() - start)
    metrics = _regression_metrics(xp, prediction, y_test_device)
    return {
        **metrics,
        "prediction_seconds": prediction_seconds,
        "prediction_kernel_pair_evaluations": n_train * n_test,
    }


def run_original_krr_nystrom_pcg(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    cfg: OriginalKRRNystromConfig,
    *,
    array_module: Any | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one complete, auditable original-KRR Nyström-PCG pipeline."""

    n_train, dim = _shape_xy(x_train, y_train, prefix="train")
    n_test, test_dim = _shape_xy(x_test, y_test, prefix="test")
    if dim != test_dim:
        raise ValueError("training and test feature counts differ.")
    # This must remain before fit/backend resolution to make the gate useful
    # for shape-only 10M--300M dataset handles.
    resource_audit = preflight_original_krr_resources(n_train, n_test, cfg)
    model, fit_diagnostics = fit_original_krr_nystrom_pcg(
        x_train,
        y_train,
        cfg,
        array_module=array_module,
        timer=timer,
    )
    score = score_original_krr_nystrom_pcg(
        model,
        x_test,
        y_test,
        timer=timer,
    )
    result = {
        "status": "converged" if bool(fit_diagnostics["converged"]) else "maxiter",
        "method": METHOD_NAME,
        "implementation": "exact_blocked_data_space_krr_column_nystrom_pcg",
        "pipeline_family": "literature_original_data_space_krr",
        "citations": list(_CITATIONS),
        "n_train": n_train,
        "n_test": n_test,
        "input_dim": dim,
        "kernel_family": "matern",
        "nu": 1.5,
        "lengthscale": float(cfg.lengthscale),
        "kernel_variance": float(cfg.kernel_variance),
        "absolute_ridge": float(cfg.absolute_ridge),
        "regularization_convention": "absolute",
        "solved_system": "original_data_space_K_plus_absolute_ridge_I",
        "operator_approximation": False,
        "kernel_matvec": "exact_double_blocked_isotropic_matern32",
        "preconditioner_sketch": "uniform_random_column_sampling_without_replacement",
        "precision": "fp64",
        **fit_diagnostics,
        **score,
        "resource_audit": resource_audit,
        "timing_scope": (
            "setup includes backend data staging and column-Nystrom construction; "
            "solve includes exact double-blocked K PCG and a true residual product; "
            "prediction, including test staging, is separate"
        ),
        "config": asdict(cfg),
    }
    # The public result remains row/JSON friendly; coefficients are available
    # from ``fit_original_krr_nystrom_pcg`` for numerical validation or reuse.
    result.pop("sample_indices", None)
    return result


__all__ = [
    "METHOD_NAME",
    "ColumnNystromPreconditioner",
    "ExactMatern32Operator",
    "OriginalKRRNystromConfig",
    "OriginalKRRNystromModel",
    "OriginalKRRResourceLimit",
    "build_column_nystrom_preconditioner",
    "exact_matern32_cross_matvec",
    "fit_original_krr_nystrom_pcg",
    "matern32_cross",
    "preflight_original_krr_resources",
    "run_original_krr_nystrom_pcg",
    "score_original_krr_nystrom_pcg",
]
