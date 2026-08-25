from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class RandomizedPivotedCholeskyData:
    """Inverse operation induced by a randomized pivoted-Cholesky factor.

    The builder approximates the unregularized positive-semidefinite part of
    the fixed Fourier system as ``K approximately L L*``.  The stored operation
    is ``(L L* + reg_lambda I)^{-1}``, applied with the Woodbury identity.
    """

    L: Any
    LH: Any
    middle_inverse: Any
    pivots: tuple[int, ...]
    requested_rank: int
    effective_rank: int
    sampling_seed: int
    reg_lambda: float
    diagnostics: dict[str, Any]


def _device_float(value: Any) -> float:
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def _device_int(value: Any) -> int:
    item = getattr(value, "item", None)
    return int(item() if callable(item) else value)


def make_weighted_toeplitz_column_accessor(
    xp: Any,
    xtxcol: Any,
    weights: Any,
    *,
    mtot: int,
    dim: int,
    dtype: Any | None = None,
) -> Callable[[int], Any]:
    """Return direct columns of ``D G D`` from the stored Toeplitz generator."""

    side = int(mtot)
    ndim = int(dim)
    if side <= 0 or ndim <= 0:
        raise ValueError("mtot and dim must be positive.")
    size = side**ndim
    weights_array = xp.asarray(weights, dtype=dtype).reshape(-1)
    if int(weights_array.size) != size:
        raise ValueError(f"weights has {int(weights_array.size)} entries; expected {size}.")
    generator = xp.asarray(xtxcol)
    expected_shape = (2 * side - 1,) * ndim
    if tuple(generator.shape) != expected_shape:
        raise ValueError(
            f"xtxcol has shape {tuple(generator.shape)}; expected {expected_shape}."
        )
    flat_indices = np.arange(size, dtype=np.int64)
    multi_host = np.unravel_index(flat_indices, (side,) * ndim)
    multi_device = tuple(xp.asarray(axis, dtype=xp.int64) for axis in multi_host)
    shift = side - 1

    def column(index: int) -> Any:
        pivot = int(index)
        if pivot < 0 or pivot >= size:
            raise IndexError(f"column index {pivot} is outside [0, {size}).")
        pivot_multi = np.unravel_index(pivot, (side,) * ndim)
        lag_index = tuple(
            multi_device[axis] - int(pivot_multi[axis]) + shift
            for axis in range(ndim)
        )
        return weights_array * generator[lag_index] * weights_array[pivot]

    return column


def build_randomized_pivoted_cholesky_preconditioner(
    backend: Any,
    apply_psd_column: Callable[[int], Any],
    psd_diagonal: Any,
    *,
    rank: int,
    reg_lambda: float,
    seed: int = 0,
    dtype: Any | None = None,
    relative_trace_tolerance: float = 0.0,
    column_access_model: str = "supplied_psd_columns",
) -> RandomizedPivotedCholeskyData:
    """Build a Fourier-system randomized pivoted-Cholesky preconditioner.

    ``apply_psd_column(j)`` must return column ``j`` of the same Hermitian PSD
    matrix ``K = A - reg_lambda I`` used by the solver.  Pivots are sampled in
    proportion to the current residual diagonal.  This is the natural
    complex-Hermitian, operator-column adaptation of RPCholesky; it is not the
    published data-space KRR solver.
    """

    xp = backend.xp
    reg = float(reg_lambda)
    requested_rank = int(rank)
    rel_tol = float(relative_trace_tolerance)
    if reg <= 0.0:
        raise ValueError("reg_lambda must be positive.")
    if requested_rank <= 0:
        raise ValueError("rank must be positive.")
    if not np.isfinite(rel_tol) or rel_tol < 0.0:
        raise ValueError("relative_trace_tolerance must be finite and nonnegative.")
    if dtype is None:
        dtype = xp.complex128
    dtype = xp.dtype(dtype)
    if dtype.kind != "c":
        real_dtype = xp.float32 if dtype == xp.dtype(xp.float32) else xp.float64
    else:
        real_dtype = xp.float32 if dtype == xp.dtype(xp.complex64) else xp.float64

    diagonal = xp.asarray(psd_diagonal, dtype=real_dtype).reshape(-1)
    n = int(diagonal.size)
    if n <= 0:
        raise ValueError("psd_diagonal must be nonempty.")
    if requested_rank > n:
        raise ValueError(
            f"rank must not exceed the system size; got rank={requested_rank}, size={n}."
        )
    if not bool(_device_int(xp.all(xp.isfinite(diagonal)))):
        raise ValueError("psd_diagonal contains non-finite values.")

    # Small negative entries can arise from roundoff in an otherwise PSD
    # diagonal.  Materially negative entries indicate a mismatched operator.
    eps = float(xp.finfo(real_dtype).eps)
    diagonal_scale = max(
        _device_float(xp.max(xp.abs(diagonal))),
        float(xp.finfo(real_dtype).tiny),
    )
    negative_tolerance = 100.0 * eps * diagonal_scale
    if _device_float(xp.min(diagonal)) < -negative_tolerance:
        raise ValueError("psd_diagonal is materially negative; the supplied K is not PSD.")
    residual = xp.maximum(diagonal, xp.asarray(0.0, dtype=real_dtype)).copy()
    initial_trace = _device_float(xp.sum(residual))
    if not np.isfinite(initial_trace) or initial_trace <= 0.0:
        raise ValueError("psd_diagonal must have positive finite trace.")

    rng = np.random.default_rng(int(seed))
    factor = xp.empty((n, requested_rank), dtype=dtype)
    pivots: list[int] = []
    pivot_values: list[float] = []
    trace_history: list[float] = [initial_trace]
    absolute_trace_tolerance = rel_tol * initial_trace
    pivot_floor = 100.0 * eps * max(
        diagonal_scale,
        initial_trace / n,
        float(xp.finfo(real_dtype).tiny),
    )

    # A rejected near-zero pivot is removed from the sampling distribution and
    # resampled.  In exact arithmetic this branch is unnecessary, but it avoids
    # dividing by a roundoff-sized residual after many updates.
    rejected_pivots = 0
    maximum_pivot_diagonal_mismatch = 0.0
    while len(pivots) < requested_rank:
        residual_trace = _device_float(xp.sum(residual))
        if residual_trace <= max(absolute_trace_tolerance, 0.0):
            break
        cdf = xp.cumsum(residual)
        target = float(rng.random()) * residual_trace
        target_array = xp.asarray([target], dtype=real_dtype)
        pivot = _device_int(xp.searchsorted(cdf, target_array, side="right")[0])
        if pivot >= n:
            pivot = _device_int(xp.argmax(residual))

        column = xp.asarray(apply_psd_column(int(pivot)), dtype=dtype).reshape(-1).copy()
        if int(column.size) != n:
            raise ValueError(
                f"apply_psd_column returned {int(column.size)} entries; expected {n}."
            )
        k = len(pivots)
        if k:
            column -= factor[:, :k] @ xp.conj(factor[pivot, :k])
        pivot_entry = column[pivot]
        pivot_value = _device_float(xp.real(pivot_entry))
        pivot_imag = abs(_device_float(xp.imag(pivot_entry)))
        sampled_residual = _device_float(residual[pivot])
        entry_scale = max(abs(pivot_value), diagonal_scale, float(xp.finfo(real_dtype).tiny))
        if not np.isfinite(pivot_value) or not np.isfinite(pivot_imag):
            raise ValueError("apply_psd_column returned a non-finite pivot entry.")
        if pivot_imag > 100.0 * eps * entry_scale:
            raise ValueError(
                "apply_psd_column is not Hermitian: a pivot diagonal has material imaginary part."
            )
        if pivot_value < -pivot_floor:
            raise ValueError(
                "apply_psd_column is inconsistent with a PSD matrix: a residual pivot is negative."
            )
        pivot_mismatch = abs(pivot_value - sampled_residual)
        mismatch_tolerance = 1000.0 * eps * max(
            abs(pivot_value),
            abs(sampled_residual),
            diagonal_scale,
            float(xp.finfo(real_dtype).tiny),
        )
        if pivot_mismatch > mismatch_tolerance:
            raise ValueError(
                "apply_psd_column is inconsistent with psd_diagonal at a sampled pivot."
            )
        maximum_pivot_diagonal_mismatch = max(
            maximum_pivot_diagonal_mismatch, pivot_mismatch
        )
        if pivot_value <= pivot_floor:
            residual[pivot] = 0.0
            rejected_pivots += 1
            if rejected_pivots > n:
                break
            continue

        new_column = column / xp.sqrt(xp.asarray(pivot_value, dtype=real_dtype))
        factor[:, k] = new_column
        residual = xp.maximum(
            residual - xp.asarray(xp.abs(new_column) ** 2, dtype=real_dtype),
            xp.asarray(0.0, dtype=real_dtype),
        )
        residual[pivot] = 0.0
        pivots.append(int(pivot))
        pivot_values.append(float(pivot_value))
        trace_history.append(_device_float(xp.sum(residual)))

    effective_rank = len(pivots)
    if effective_rank == 0:
        raise RuntimeError("RPCholesky did not obtain a numerically positive pivot.")
    L = xp.asfortranarray(factor[:, :effective_rank])
    LH = xp.asfortranarray(L.conj().T)
    middle = LH @ L
    middle += reg * xp.eye(effective_rank, dtype=dtype)
    middle = 0.5 * (middle + middle.conj().T)
    middle_inverse = xp.ascontiguousarray(xp.linalg.inv(middle))

    final_trace = trace_history[-1]
    diagnostics = {
        "rank": int(effective_rank),
        "requested_rank": int(requested_rank),
        "effective_rank": int(effective_rank),
        "sampling_seed": int(seed),
        "regularization": reg,
        "relative_trace_tolerance": rel_tol,
        "initial_residual_trace": float(initial_trace),
        "final_residual_trace": float(final_trace),
        "relative_residual_trace": float(final_trace / initial_trace),
        "pivot_values": pivot_values,
        "pivots": pivots,
        "rejected_near_zero_pivots": int(rejected_pivots),
        "maximum_pivot_diagonal_mismatch": float(maximum_pivot_diagonal_mismatch),
        "psd_column_products": int(effective_rank + rejected_pivots),
        "dtype": str(dtype),
        "storage_bytes": int(L.nbytes + LH.nbytes + middle_inverse.nbytes),
        "access_model": str(column_access_model),
        "method_scope": "complex_Hermitian_Fourier_system_adaptation",
        "block_size": 1,
    }
    return RandomizedPivotedCholeskyData(
        L=L,
        LH=LH,
        middle_inverse=middle_inverse,
        pivots=tuple(pivots),
        requested_rank=int(requested_rank),
        effective_rank=int(effective_rank),
        sampling_seed=int(seed),
        reg_lambda=reg,
        diagnostics=diagnostics,
    )


def apply_randomized_pivoted_cholesky_preconditioner(
    backend: Any,
    data: RandomizedPivotedCholeskyData,
    v: Any,
    *,
    out: Any | None = None,
) -> Any:
    """Apply ``(L L* + lambda I)^{-1}`` to a vector or column block."""

    xp = backend.xp
    vv = xp.asarray(v, dtype=data.L.dtype)
    if vv.ndim not in (1, 2):
        raise ValueError("v must be a vector or a column block.")
    result = xp.empty_like(vv) if out is None else out
    projected = data.LH @ vv
    correction = data.L @ (data.middle_inverse @ projected)
    result[...] = (vv - correction) / float(data.reg_lambda)
    return result
