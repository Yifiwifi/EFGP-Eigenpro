"""End-to-end KRR pipeline benchmark.

This module answers a different question from :mod:`controlled.benchmark`.
Here every row owns its model construction, setup, solve, and predictor.  The
fixed-system runner instead freezes one Fourier ``A beta = b`` and compares
linear solvers/preconditioners only.  Keeping the two schemas separate is a
scientific requirement: a data-space Nyström or RPCholesky KRR pipeline is not
a preconditioner for the already-constructed Fourier system.

The scalable low-rank baselines implement restricted KRR

    (C.T C + ridge * W) alpha = C.T y,

where ``C = K(X, landmarks)`` and ``W = K(landmarks, landmarks)``.  Uniform
landmarks give the Nyström baseline; randomly pivoted Cholesky selects the
RPCholesky landmarks.  Normal equations are accumulated in row chunks, so the
Nyström path never materializes an ``N x rank`` feature matrix.  Exact simple
RPCholesky does require its ``rank x N`` partial-Cholesky factor; a preflight
gate records ``resource_limit`` instead of silently substituting a different
algorithm when that factor does not fit.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ....efgp_solver import EFGPSolver
from ....kernels import make_matern, make_squared_exponential
from ...backends import BackendConfig, build_gpu_backend_bundle
from ...benchmark_dataset.stored_npz import (
    StoredNpzError,
    inspect_stored_npy_member,
    load_stored_npz_prefix,
)
from ...v1_ops import predict_v1
from . import benchmark as fixed_ab


END_TO_END_METHODS = (
    "nystrom-krr",
    "rpcholesky-krr",
    "efgp-standard-cg",
    "efgp-standard-jacobi",
    "efgp-standard-full-eig",
    "ours-binned-default",
)

# Configuration carried from the selected Stage-1 KRR regime into the one
# fixed Fourier system used by Stage 2. This mirrors benchmark.py's
# _SYSTEM_CONFIG_FIELDS so a different subset, variance, precision, or NUFFT
# route cannot be substituted after target selection.
STAGE2_SYSTEM_CONFIG_FIELDS = (
    "dataset_stem",
    "n_train",
    "subset_seed",
    "subset_mode",
    "kernel_family",
    "lengthscale",
    "nu",
    "variance",
    "reg_lambda",
    "fourier_eps",
    "nufft_tol",
    "l2_scaled",
    "precision",
    "nufft_backend",
    "precompute_chunk_size",
)

PROTOCOL_FAMILY = "end_to_end_krr"
TIMING_SCOPE = (
    "Each method owns algorithmic model setup and solve. train_total_seconds = "
    "setup_seconds + solving_phase_seconds. This is a method-owned algorithmic "
    "training total, not process wall clock: common dataset I/O, backend creation, "
    "and host-to-device staging are excluded, and prediction is separate."
)


class RPCholeskyResourceLimit(MemoryError):
    """Exact RPCholesky cannot fit its declared rank-by-N factor."""

    def __init__(
        self,
        *,
        required_bytes: int,
        effective_cap_bytes: int,
        declared_cap_bytes: int,
        available_device_bytes: int | None,
    ) -> None:
        self.required_bytes = int(required_bytes)
        self.effective_cap_bytes = int(effective_cap_bytes)
        self.declared_cap_bytes = int(declared_cap_bytes)
        self.available_device_bytes = (
            int(available_device_bytes) if available_device_bytes is not None else None
        )
        super().__init__(
            "Exact RPCholesky factor preflight failed: "
            f"rank*N*dtype requires {self.required_bytes} bytes, cap is "
            f"{self.effective_cap_bytes}. The benchmark will record "
            "resource_limit; it will not substitute a pilot-set method."
        )


@dataclass(frozen=True)
class EndToEndConfig:
    dataset_stem: str = "synthetic_true_func_2d_ntrain10000000"
    dataset_dir: str = ""
    n_train: int | None = None
    subset_seed: int = 0
    subset_mode: str = "prefix"
    max_test_rows: int = 2_500_000
    kernel_family: str = "matern"
    lengthscale: float = 0.1
    nu: float = 1.5
    variance: float = 1.0
    reg_lambda: float = 0.1
    # ``absolute`` matches the current EFGP system Phi*Phi + lambda I.
    # ``mean_loss`` uses N*lambda in restricted KRR and must only be used when
    # the EFGP regularization is converted to the same convention upstream.
    regularization_convention: str = "absolute"
    fourier_eps: float = 1e-5
    nufft_tol: float = 1e-10
    l2_scaled: bool = True
    tol: float = 1e-7
    maxiter: int = 80_000
    precision: str = "fp64"
    methods: tuple[str, ...] = END_TO_END_METHODS
    # Active/ours and full-grid EigenPro may use distinct parameters selected
    # by a predeclared pilot or transferred from the archived experiment.
    rank: int = 256
    full_eig_rank: int | None = None
    active_topk: int | None = None
    expected_active_box_size: int | None = None
    allow_frozen_topk_capacity_adaptation: bool = False
    parameter_selection_policy: str = "deployable_score_rule"
    parameter_source: str = ""
    nystrom_rank: int = 256
    rpcholesky_rank: int = 256
    eig_tol: float = 1e-3
    eig_maxiter: int | None = 1280
    score_tau: float = 1.0
    box_budget: int = 8192
    inverse_max_size: int = 1024
    nystrom_seed: int = 17
    rpcholesky_seed: int = 23
    eig_seed: int = 0
    method_order_seed: int = 20260826
    warmup_repeats: int = 0
    measured_repeats: int = 1
    low_rank_chunk_size: int = 250_000
    low_rank_dtype: str = "fp64"
    # Exact RPCholesky is never replaced by a pilot/subsampled variant.  If the
    # declared factor does not fit this cap (or current device memory), the row
    # is retained with status=resource_limit.
    rpcholesky_max_factor_bytes: int = 48 * 2**30
    accuracy_reference_method: str = "efgp-standard-full-eig"
    accuracy_relative_tolerance: float = 0.01
    # Relative equivalence is descriptive: it identifies nearly equal-accuracy
    # runs, but never suppresses an otherwise valid time/accuracy trade-off.
    # Formal suite cases set a broad, dataset-specific usable-quality range with
    # these absolute thresholds.  That range, not relative equivalence, controls
    # eligibility for quality-qualified headline comparisons and target selection.
    accuracy_max_rmse: float | None = None
    accuracy_min_r2: float | None = None
    nufft_backend: str = "cufinufft"
    precompute_chunk_size: int | None = 1_000_000
    output_dir: str = ""


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _sync(xp: Any) -> None:
    cuda = getattr(xp, "cuda", None)
    if cuda is not None:
        cuda.get_current_stream().synchronize()


def _release_gpu_allocator_cache() -> None:
    """Make pipeline ordering unable to consume another method's cached blocks."""
    gc.collect()
    try:
        import cupy as cp  # type: ignore

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except (ImportError, AttributeError, RuntimeError):
        return


def _device_float(value: Any) -> float:
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


def _to_numpy(value: Any) -> np.ndarray:
    getter = getattr(value, "get", None)
    return np.asarray(getter() if callable(getter) else value)


def _dtype_for_name(xp: Any, name: str) -> Any:
    key = str(name).strip().lower()
    if key == "fp64":
        return xp.float64
    if key == "fp32":
        return xp.float32
    raise ValueError("low_rank_dtype must be 'fp32' or 'fp64'.")


def _validate_config(cfg: EndToEndConfig) -> None:
    unknown = [method for method in cfg.methods if method not in END_TO_END_METHODS]
    if unknown:
        raise ValueError(
            f"unknown end-to-end methods {unknown}; choices are {list(END_TO_END_METHODS)}"
        )
    if not cfg.methods:
        raise ValueError("methods must be nonempty.")
    if int(cfg.measured_repeats) <= 0 or int(cfg.warmup_repeats) < 0:
        raise ValueError(
            "measured_repeats must be positive and warmup_repeats nonnegative."
        )
    if int(cfg.low_rank_chunk_size) <= 0:
        raise ValueError("low_rank_chunk_size must be positive.")
    if any(
        int(rank) <= 0 for rank in (cfg.rank, cfg.nystrom_rank, cfg.rpcholesky_rank)
    ):
        raise ValueError("all ranks must be positive.")
    if cfg.full_eig_rank is not None and int(cfg.full_eig_rank) <= 0:
        raise ValueError("full_eig_rank must be positive or None.")
    if cfg.active_topk is not None and int(cfg.active_topk) <= 0:
        raise ValueError("active_topk must be positive or None.")
    if (
        cfg.expected_active_box_size is not None
        and int(cfg.expected_active_box_size) <= 0
    ):
        raise ValueError("expected_active_box_size must be positive or None.")
    if float(cfg.reg_lambda) <= 0.0:
        raise ValueError("reg_lambda must be positive.")
    convention = str(cfg.regularization_convention).strip().lower()
    if convention not in {"absolute", "mean_loss"}:
        raise ValueError("regularization_convention must be 'absolute' or 'mean_loss'.")
    if convention != "absolute" and any(
        method.startswith("efgp-") or method.startswith("ours-")
        for method in cfg.methods
    ):
        raise ValueError(
            "Current EFGP uses Phi*Phi + reg_lambda*I. Use convention='absolute' "
            "for mixed pipeline comparisons; do not compare mismatched ridge scaling."
        )
    if int(cfg.max_test_rows) <= 0:
        raise ValueError("max_test_rows must be positive.")
    if float(cfg.accuracy_relative_tolerance) < 0.0:
        raise ValueError("accuracy_relative_tolerance must be nonnegative.")
    if cfg.accuracy_max_rmse is not None and float(cfg.accuracy_max_rmse) <= 0.0:
        raise ValueError("accuracy_max_rmse must be positive or None.")
    if cfg.accuracy_min_r2 is not None and not math.isfinite(
        float(cfg.accuracy_min_r2)
    ):
        raise ValueError("accuracy_min_r2 must be finite or None.")


def _make_kernel(cfg: EndToEndConfig, dim: int) -> Any:
    family = str(cfg.kernel_family).strip().lower()
    if family in {"matern", "mat", "mat32"}:
        return make_matern(
            lengthscale=float(cfg.lengthscale),
            nu=float(cfg.nu),
            dim=int(dim),
            variance=float(cfg.variance),
        )
    if family in {"se", "rbf", "gaussian", "squared_exponential"}:
        return make_squared_exponential(
            lengthscale=float(cfg.lengthscale),
            dim=int(dim),
            variance=float(cfg.variance),
        )
    raise ValueError(f"unsupported kernel_family={cfg.kernel_family!r}")


def kernel_cross(
    xp: Any,
    x: Any,
    z: Any,
    *,
    family: str,
    lengthscale: float,
    nu: float,
    variance: float,
    dtype: Any,
) -> Any:
    """Evaluate a supported stationary kernel without an ``N x r x d`` tensor."""
    xx = xp.asarray(x, dtype=dtype)
    zz = xp.asarray(z, dtype=dtype)
    if xx.ndim != 2 or zz.ndim != 2 or int(xx.shape[1]) != int(zz.shape[1]):
        raise ValueError(
            "x and z must be two-dimensional with matching feature counts."
        )
    squared = (
        xp.sum(xx * xx, axis=1).reshape(-1, 1)
        + xp.sum(zz * zz, axis=1).reshape(1, -1)
        - 2.0 * (xx @ zz.T)
    )
    squared = xp.maximum(squared, xp.asarray(0.0, dtype=dtype))
    fam = str(family).strip().lower()
    ell = float(lengthscale)
    var = float(variance)
    if fam in {"se", "rbf", "gaussian", "squared_exponential"}:
        return var * xp.exp(-0.5 * squared / (ell * ell))
    if fam not in {"matern", "mat", "mat32"}:
        raise ValueError(f"unsupported spatial kernel family {family!r}")
    radius = xp.sqrt(squared)
    nu_value = float(nu)
    if math.isclose(nu_value, 0.5):
        scaled = radius / ell
        return var * xp.exp(-scaled)
    if math.isclose(nu_value, 1.5):
        scaled = math.sqrt(3.0) * radius / ell
        return var * (1.0 + scaled) * xp.exp(-scaled)
    if math.isclose(nu_value, 2.5):
        scaled = math.sqrt(5.0) * radius / ell
        return var * (1.0 + scaled + (scaled * scaled) / 3.0) * xp.exp(-scaled)
    raise ValueError(
        "GPU/streaming restricted KRR currently supports Matern nu in {0.5,1.5,2.5}."
    )


def _kernel_kwargs(cfg: EndToEndConfig, dtype: Any) -> dict[str, Any]:
    return {
        "family": cfg.kernel_family,
        "lengthscale": float(cfg.lengthscale),
        "nu": float(cfg.nu),
        "variance": float(cfg.variance),
        "dtype": dtype,
    }


def _ridge_diagonal(cfg: EndToEndConfig, n_train: int) -> float:
    if str(cfg.regularization_convention).strip().lower() == "mean_loss":
        return float(cfg.reg_lambda) * int(n_train)
    return float(cfg.reg_lambda)


def choose_uniform_landmarks(n: int, rank: int, seed: int) -> np.ndarray:
    if int(rank) > int(n):
        raise ValueError("rank cannot exceed n_train.")
    return np.sort(
        np.random.default_rng(int(seed)).choice(int(n), size=int(rank), replace=False)
    ).astype(np.int64, copy=False)


def _available_device_bytes(xp: Any) -> int | None:
    cuda = getattr(xp, "cuda", None)
    if cuda is None:
        return None
    try:
        return int(cuda.runtime.memGetInfo()[0])
    except Exception:
        return None


def choose_rpcholesky_landmarks(
    xp: Any,
    x: Any,
    cfg: EndToEndConfig,
    *,
    rank: int,
    dtype: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run exact simple RPCholesky, or fail closed at the declared memory gate."""
    n = int(x.shape[0])
    requested_rank = min(int(rank), n)
    itemsize = int(np.dtype(str(xp.dtype(dtype))).itemsize)
    factor_bytes = int(requested_rank * n * itemsize)
    available = _available_device_bytes(xp)
    declared_cap = int(cfg.rpcholesky_max_factor_bytes)
    effective_cap = declared_cap
    if available is not None:
        # Preserve headroom for X, y, diagonal, kernel rows, and BLAS workspace.
        effective_cap = min(effective_cap, int(0.70 * available))
    if factor_bytes > effective_cap:
        raise RPCholeskyResourceLimit(
            required_bytes=factor_bytes,
            effective_cap_bytes=effective_cap,
            declared_cap_bytes=declared_cap,
            available_device_bytes=available,
        )

    xx = xp.asarray(x, dtype=dtype)
    diagonal = xp.full(n, float(cfg.variance), dtype=dtype)
    factor = xp.empty((requested_rank, n), dtype=dtype)
    rng = np.random.default_rng(int(cfg.rpcholesky_seed))
    pivots: list[int] = []
    trace_history: list[float] = [_device_float(xp.sum(diagonal))]
    eps = float(np.finfo(np.dtype(str(xp.dtype(dtype)))).eps)
    pivot_floor = max(100.0 * eps * float(cfg.variance), np.finfo(np.float64).tiny)
    chunk = int(cfg.low_rank_chunk_size)
    kernel_kwargs = _kernel_kwargs(cfg, dtype)

    for iteration in range(requested_rank):
        total = _device_float(xp.sum(diagonal))
        if not math.isfinite(total) or total <= pivot_floor:
            break
        target = float(rng.random()) * total
        cdf = xp.cumsum(diagonal)
        pivot = int(_device_float(xp.searchsorted(cdf, target, side="right")))
        if pivot >= n:
            pivot = int(_device_float(xp.argmax(diagonal)))
        denom2 = _device_float(diagonal[pivot])
        if denom2 <= pivot_floor:
            # Never leave an uninitialized factor row and then include it in a
            # later projection.  A numerically exhausted residual terminates the
            # exact factorization; it is not a skipped iteration.
            break
        pivot_point = xx[pivot : pivot + 1]
        pivot_coeff = factor[:iteration, pivot].copy() if iteration else None
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            row = kernel_cross(
                xp,
                xx[start:stop],
                pivot_point,
                **kernel_kwargs,
            ).reshape(-1)
            if iteration:
                row -= pivot_coeff @ factor[:iteration, start:stop]
            factor[iteration, start:stop] = row / math.sqrt(denom2)
        diagonal -= factor[iteration] * factor[iteration]
        diagonal = xp.maximum(diagonal, xp.asarray(0.0, dtype=dtype))
        diagonal[pivot] = 0.0
        pivots.append(pivot)
        trace_history.append(_device_float(xp.sum(diagonal)))
    _sync(xp)
    del factor, diagonal
    if not pivots:
        raise RuntimeError("RPCholesky did not obtain a positive pivot.")
    return np.asarray(pivots, dtype=np.int64), {
        "requested_rank": requested_rank,
        "effective_rank": len(pivots),
        "factor_storage_bytes": factor_bytes,
        "pivot_seed": int(cfg.rpcholesky_seed),
        "trace_initial": trace_history[0],
        "trace_final": trace_history[-1],
        "relative_trace_final": trace_history[-1] / trace_history[0],
        "selection_algorithm": "exact_simple_rpcholesky",
    }


def fit_restricted_krr(
    xp: Any,
    x: Any,
    y: Any,
    landmark_indices: np.ndarray,
    cfg: EndToEndConfig,
) -> tuple[Any, Any, dict[str, Any]]:
    """Fit restricted KRR using streaming normal-equation accumulation."""
    dtype = _dtype_for_name(xp, cfg.low_rank_dtype)
    xx = xp.asarray(x, dtype=dtype)
    yy = xp.asarray(y, dtype=dtype).reshape(-1)
    indices = xp.asarray(np.asarray(landmark_indices, dtype=np.int64))
    landmarks = xx[indices].copy()
    n = int(xx.shape[0])
    rank = int(landmarks.shape[0])
    if int(yy.size) != n:
        raise ValueError("x and y row counts differ.")
    kernel_kwargs = _kernel_kwargs(cfg, dtype)
    gram = xp.zeros((rank, rank), dtype=dtype)
    rhs = xp.zeros(rank, dtype=dtype)
    chunk = int(cfg.low_rank_chunk_size)
    t0 = time.perf_counter()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        cross = kernel_cross(xp, xx[start:stop], landmarks, **kernel_kwargs)
        gram += cross.T @ cross
        rhs += cross.T @ yy[start:stop]
    W = kernel_cross(xp, landmarks, landmarks, **kernel_kwargs)
    ridge = _ridge_diagonal(cfg, n)
    system = gram + float(ridge) * W
    scale = max(_device_float(xp.max(xp.abs(system))), 1.0)
    jitter = 100.0 * float(np.finfo(np.dtype(str(xp.dtype(dtype)))).eps) * scale
    system += jitter * xp.eye(rank, dtype=dtype)
    _sync(xp)
    system_build_seconds = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    coefficients = xp.linalg.solve(system, rhs)
    _sync(xp)
    solve_seconds = float(time.perf_counter() - t1)
    diagnostics = {
        "rank": rank,
        "ridge_diagonal": float(ridge),
        "regularization_convention": str(cfg.regularization_convention),
        "system_build_seconds": system_build_seconds,
        "dense_solve_seconds": solve_seconds,
        "normal_matrix_storage_bytes": int(system.nbytes),
        "landmark_storage_bytes": int(landmarks.nbytes),
        "low_rank_dtype": str(xp.dtype(dtype)),
        "jitter": float(jitter),
    }
    return coefficients, landmarks, diagnostics


def predict_restricted_krr(
    xp: Any,
    x_test: Any,
    landmarks: Any,
    coefficients: Any,
    cfg: EndToEndConfig,
) -> Any:
    dtype = _dtype_for_name(xp, cfg.low_rank_dtype)
    xx = xp.asarray(x_test, dtype=dtype)
    output = xp.empty(int(xx.shape[0]), dtype=dtype)
    kernel_kwargs = _kernel_kwargs(cfg, dtype)
    chunk = int(cfg.low_rank_chunk_size)
    for start in range(0, int(xx.shape[0]), chunk):
        stop = min(start + chunk, int(xx.shape[0]))
        output[start:stop] = (
            kernel_cross(xp, xx[start:stop], landmarks, **kernel_kwargs) @ coefficients
        )
    _sync(xp)
    return output


def _metrics(y_true: np.ndarray, y_pred: Any) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = _to_numpy(y_pred).astype(np.float64, copy=False).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("prediction and target shapes differ.")
    residual = pred - truth
    mse = float(np.mean(residual * residual))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((truth - float(np.mean(truth))) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / max(denom, np.finfo(float).tiny)
    return {"test_rmse": math.sqrt(max(mse, 0.0)), "test_mae": mae, "test_r2": r2}


def _load_member_prefix(path: Path, name: str, rows: int, *, dtype: Any) -> np.ndarray:
    try:
        info = inspect_stored_npy_member(path, name)
        use_rows = min(int(rows), int(info.shape[0]))
        return load_stored_npz_prefix(path, name, use_rows, dtype=dtype)
    except StoredNpzError:
        with np.load(path) as loaded:
            array = np.asarray(loaded[name], dtype=dtype)
        return np.ascontiguousarray(array[: int(rows)])


def load_end_to_end_dataset(cfg: EndToEndConfig) -> dict[str, Any]:
    train = fixed_ab._load_dataset(
        cfg.dataset_stem,
        cfg.n_train,
        cfg.subset_seed,
        cfg.dataset_dir,
        cfg.subset_mode,
    )
    path = Path(train["path"])
    try:
        x_info = inspect_stored_npy_member(path, "x_test")
        test_rows = min(int(cfg.max_test_rows), int(x_info.shape[0]))
    except StoredNpzError:
        with np.load(path) as loaded:
            test_rows = min(int(cfg.max_test_rows), int(loaded["x_test"].shape[0]))
    x_test = _load_member_prefix(path, "x_test", test_rows, dtype=np.float64)
    y_test = _load_member_prefix(path, "y_test", test_rows, dtype=np.float64).reshape(
        -1
    )
    if int(x_test.shape[0]) != int(y_test.size):
        raise ValueError("x_test and y_test row counts differ.")
    return {**train, "x_test": x_test, "y_test": y_test, "n_test": test_rows}


def _fixed_config(
    cfg: EndToEndConfig, methods: tuple[str, ...]
) -> fixed_ab.ControlledConfig:
    uses_active_rule = any(
        method in {"default", "active-eig", "active-inverse"} for method in methods
    )
    return fixed_ab.ControlledConfig(
        dataset_stem=cfg.dataset_stem,
        dataset_dir=cfg.dataset_dir,
        n_train=cfg.n_train,
        subset_seed=cfg.subset_seed,
        subset_mode=cfg.subset_mode,
        kernel_family=cfg.kernel_family,
        lengthscale=cfg.lengthscale,
        nu=cfg.nu,
        variance=cfg.variance,
        reg_lambda=cfg.reg_lambda,
        fourier_eps=cfg.fourier_eps,
        nufft_tol=cfg.nufft_tol,
        l2_scaled=cfg.l2_scaled,
        tol=cfg.tol,
        maxiter=cfg.maxiter,
        precision=cfg.precision,
        methods=methods,
        score_tau=cfg.score_tau,
        box_budget=cfg.box_budget,
        inverse_max_size=cfg.inverse_max_size,
        active_topk=cfg.active_topk if uses_active_rule else None,
        rank=cfg.rank,
        full_eig_rank=cfg.full_eig_rank,
        expected_active_box_size=(
            cfg.expected_active_box_size if uses_active_rule else None
        ),
        allow_frozen_topk_capacity_adaptation=(
            bool(cfg.allow_frozen_topk_capacity_adaptation)
            if uses_active_rule
            else False
        ),
        parameter_selection_policy=cfg.parameter_selection_policy,
        parameter_source=cfg.parameter_source,
        eig_tol=cfg.eig_tol,
        eig_maxiter=cfg.eig_maxiter,
        measured_repeats=max(5, cfg.measured_repeats),
        warmup_repeats=cfg.warmup_repeats,
        method_order_seed=cfg.method_order_seed,
        eig_seed=cfg.eig_seed,
        nufft_backend=cfg.nufft_backend,
        precompute_chunk_size=cfg.precompute_chunk_size,
        post_diagnostic_mode="none",
    )


def _prepare_binned_system(
    cfg: fixed_ab.ControlledConfig,
    dataset: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Use the existing C1 implementation while confining its global patch."""
    from ...benchmark_dataset import accuracy_based_eigenpro_tables as accuracy

    patch_cfg = accuracy.AccuracyBenchmarkConfig(
        precompute_methods={"default": "c1"},
        binned_quality="balanced",
        binned_use_sparse_bins=False,
        binned_use_gpu_dense_bins=True,
        binned_allow_exact_nufft_fallback=False,
        binned_nufft_allow_cpu_fallback=False,
    )
    accuracy.install_gpu_precompute_patch(patch_cfg)
    wrapped = accuracy._gpu_v1_ops_bm.gpu_precompute_v1
    original = fixed_ab.gpu_precompute_v1
    previous_mode = accuracy._BENCHMARK_PC_METHOD_ACTIVE
    try:
        fixed_ab.gpu_precompute_v1 = wrapped
        accuracy._BENCHMARK_PC_METHOD_ACTIVE = "c1"
        accuracy._LAST_PC_PATCH_EXTRA.clear()
        system = fixed_ab.prepare_shared_system(cfg, dataset_payload=dataset)
        extra = dict(accuracy._LAST_PC_PATCH_EXTRA)
        system.manifest["setup_route"] = "binned_c1"
        system.manifest["setup_route_diagnostics"] = extra
        return system, extra
    finally:
        fixed_ab.gpu_precompute_v1 = original
        accuracy._BENCHMARK_PC_METHOD_ACTIVE = previous_mode
        accuracy._LAST_PC_PATCH_EXTRA.clear()


def _run_efgp_method(
    method: str,
    cfg: EndToEndConfig,
    dataset: dict[str, Any],
    *,
    repeat_idx: int,
    is_warmup: bool,
) -> dict[str, Any]:
    if method == "efgp-standard-cg":
        controlled_cfg = _fixed_config(cfg, ("cg",))
        system = fixed_ab.prepare_shared_system(controlled_cfg, dataset_payload=dataset)
        specs, _ = fixed_ab.resolve_method_specs(system, controlled_cfg)
        spec = next(spec for spec in specs if spec.label == "cg")
        setup_route = "standard_nufft"
        setup_extra: dict[str, Any] = {}
    elif method == "efgp-standard-jacobi":
        controlled_cfg = _fixed_config(cfg, ("cg", "jacobi"))
        system = fixed_ab.prepare_shared_system(controlled_cfg, dataset_payload=dataset)
        specs, _ = fixed_ab.resolve_method_specs(system, controlled_cfg)
        spec = next(spec for spec in specs if spec.label == "jacobi")
        setup_route = "standard_nufft"
        setup_extra: dict[str, Any] = {}
    elif method == "efgp-standard-full-eig":
        controlled_cfg = _fixed_config(cfg, ("cg", "full-eig"))
        system = fixed_ab.prepare_shared_system(controlled_cfg, dataset_payload=dataset)
        specs, _ = fixed_ab.resolve_method_specs(system, controlled_cfg)
        spec = next(spec for spec in specs if spec.label == "full-eig")
        setup_route = "standard_nufft"
        setup_extra = {}
    elif method == "ours-binned-default":
        controlled_cfg = _fixed_config(cfg, ("cg", "default"))
        system, setup_extra = _prepare_binned_system(controlled_cfg, dataset)
        specs, _ = fixed_ab.resolve_method_specs(system, controlled_cfg)
        spec = next(spec for spec in specs if spec.label == "default")
        setup_route = "binned_c1"
    else:  # pragma: no cover - dispatch is validated before this helper
        raise ValueError(f"unsupported EFGP pipeline {method!r}")

    method_row, beta = fixed_ab.run_one_method(
        system,
        controlled_cfg,
        spec,
        repeat_idx=int(repeat_idx),
        order_position=0,
        is_warmup=bool(is_warmup),
    )
    if beta is None:
        raise RuntimeError(
            f"{method} failed in fixed Fourier solve: {method_row.get('error_message', method_row.get('status'))}"
        )
    xp = system.backend.xp
    beta_gpu = xp.asarray(beta, dtype=system.rhs_gpu.dtype)
    t_pred = time.perf_counter()
    prediction = predict_v1(
        system.backend,
        system.data_ctx,
        dataset["x_test"],
        beta_gpu,
    )
    _sync(xp)
    prediction_seconds = float(time.perf_counter() - t_pred)
    selection = float(method_row.get("selection_seconds", 0.0))
    preconditioner_build = float(method_row["preconditioner_build_seconds"])
    solve = float(method_row["solve_seconds"])
    setup = float(system.setup_seconds)
    solving = selection + preconditioner_build + solve
    effective_active_topk = method_row.get("active_topk")
    effective_active_box_size = method_row.get("box_size")
    if spec.active_set is not None:
        effective_active_topk = int(spec.active_set.active_idx.size)
        effective_active_box_size = int(spec.active_set.box_idx.size)
    elif spec.btab_config is not None and spec.btab_config.active_topk is not None:
        effective_active_topk = int(spec.btab_config.active_topk)
    return {
        **method_row,
        **_metrics(dataset["y_test"], prediction),
        "method": method,
        "pipeline_family": "efgp",
        "setup_route": setup_route,
        "setup_seconds": setup,
        "solver_build_seconds": preconditioner_build,
        "iterative_solve_seconds": solve,
        "solving_phase_seconds": solving,
        "train_total_seconds": setup + solving,
        "prediction_seconds": prediction_seconds,
        "setup_route_diagnostics": setup_extra,
        "M": int(system.manifest["M"]),
        "mtot": int(system.manifest["mtot"]),
        "system_id": system.system_id,
        "active_selection_rule": str(method_row.get("selection_rule", "")),
        "effective_active_topk": effective_active_topk,
        "effective_active_box_size": effective_active_box_size,
        "effective_active_rank": method_row.get("rank"),
        "capacity_adapted": "clamped_to_" in str(
            method_row.get("selection_rule", "")
        ),
    }


def _run_low_rank_method(
    method: str,
    cfg: EndToEndConfig,
    dataset: dict[str, Any],
    *,
    repeat_idx: int,
    is_warmup: bool,
) -> dict[str, Any]:
    backend = build_gpu_backend_bundle(BackendConfig(nufft=str(cfg.nufft_backend)))
    xp = backend.xp
    dtype = _dtype_for_name(xp, cfg.low_rank_dtype)
    # Common data staging is deliberately outside the method timer, matching
    # the EFGP runner's ensure_gpu_data_context call.
    x_gpu = xp.asarray(dataset["x"], dtype=dtype)
    y_gpu = xp.asarray(dataset["y"], dtype=dtype).reshape(-1)
    x_test_gpu = xp.asarray(dataset["x_test"], dtype=dtype)
    _sync(xp)

    t_setup = time.perf_counter()
    if method == "nystrom-krr":
        rank = min(int(cfg.nystrom_rank), int(x_gpu.shape[0]))
        # Timing repeats rebuild the complete method but keep its randomized
        # algorithm fixed.  Seed sensitivity is a separate experiment; varying
        # landmarks here would mix algorithmic variance into the paired timer.
        indices = choose_uniform_landmarks(
            int(x_gpu.shape[0]), rank, int(cfg.nystrom_seed)
        )
        selection_diag = {
            "selection_algorithm": "uniform_without_replacement",
            "requested_rank": int(cfg.nystrom_rank),
            "effective_rank": rank,
            "pivot_seed": int(cfg.nystrom_seed),
        }
    elif method == "rpcholesky-krr":
        indices, selection_diag = choose_rpcholesky_landmarks(
            xp,
            x_gpu,
            cfg,
            rank=int(cfg.rpcholesky_rank),
            dtype=dtype,
        )
    else:  # pragma: no cover
        raise ValueError(f"unsupported low-rank pipeline {method!r}")
    _sync(xp)
    selection_seconds = float(time.perf_counter() - t_setup)

    coefficients, landmarks, fit_diag = fit_restricted_krr(
        xp, x_gpu, y_gpu, indices, cfg
    )
    setup_seconds = selection_seconds + float(fit_diag["system_build_seconds"])
    solve_seconds = float(fit_diag["dense_solve_seconds"])
    t_pred = time.perf_counter()
    prediction = predict_restricted_krr(xp, x_test_gpu, landmarks, coefficients, cfg)
    prediction_seconds = float(time.perf_counter() - t_pred)
    return {
        "status": "converged",
        "method": method,
        "method_kind": method,
        "pipeline_family": "restricted_krr",
        "repeat_idx": int(repeat_idx),
        "is_warmup": bool(is_warmup),
        "selection_seconds": selection_seconds,
        "setup_seconds": setup_seconds,
        "solver_build_seconds": 0.0,
        "iterative_solve_seconds": solve_seconds,
        "solving_phase_seconds": solve_seconds,
        "train_total_seconds": setup_seconds + solve_seconds,
        "prediction_seconds": prediction_seconds,
        **_metrics(dataset["y_test"], prediction),
        **selection_diag,
        **fit_diag,
    }


def _base_row(
    cfg: EndToEndConfig, dataset: dict[str, Any], method: str
) -> dict[str, Any]:
    return {
        "protocol_family": PROTOCOL_FAMILY,
        "timing_scope": TIMING_SCOPE,
        "method": method,
        "dataset_stem": cfg.dataset_stem,
        "dataset_family": dataset.get("metadata", {}).get(
            "dataset_name", cfg.dataset_stem
        ),
        "n_train": int(dataset["x"].shape[0]),
        "n_test": int(dataset["n_test"]),
        "dim": int(dataset["x"].shape[1]),
        "subset_seed": int(cfg.subset_seed),
        "subset_mode": str(cfg.subset_mode),
        "kernel_family": str(cfg.kernel_family),
        "lengthscale": float(cfg.lengthscale),
        "nu": float(cfg.nu),
        "variance": float(cfg.variance),
        "reg_lambda": float(cfg.reg_lambda),
        "regularization_convention": str(cfg.regularization_convention),
        "fourier_eps": float(cfg.fourier_eps),
        "nufft_tol": float(cfg.nufft_tol),
        "l2_scaled": bool(cfg.l2_scaled),
        "precision": str(cfg.precision),
        "nufft_backend": str(cfg.nufft_backend),
        "precompute_chunk_size": cfg.precompute_chunk_size,
        "tol": float(cfg.tol),
        "box_budget": int(cfg.box_budget),
        "configured_active_rank": int(cfg.rank),
        "configured_full_eig_rank": int(cfg.full_eig_rank or cfg.rank),
        "configured_active_topk": (
            None if cfg.active_topk is None else int(cfg.active_topk)
        ),
        "configured_expected_active_box_size": (
            None
            if cfg.expected_active_box_size is None
            else int(cfg.expected_active_box_size)
        ),
        "configured_allow_frozen_topk_capacity_adaptation": bool(
            cfg.allow_frozen_topk_capacity_adaptation
        ),
        "parameter_selection_policy": str(cfg.parameter_selection_policy),
        "parameter_source": str(cfg.parameter_source),
        "accuracy_max_rmse": cfg.accuracy_max_rmse,
        "accuracy_min_r2": cfg.accuracy_min_r2,
        "gpu_allocator_cache_reset_between_pipelines": True,
    }


def run_pipeline_once(
    method: str,
    cfg: EndToEndConfig,
    dataset: dict[str, Any],
    *,
    repeat_idx: int,
    is_warmup: bool,
) -> dict[str, Any]:
    base = _base_row(cfg, dataset, method)
    try:
        if method in {"nystrom-krr", "rpcholesky-krr"}:
            row = _run_low_rank_method(
                method, cfg, dataset, repeat_idx=repeat_idx, is_warmup=is_warmup
            )
        else:
            row = _run_efgp_method(
                method, cfg, dataset, repeat_idx=repeat_idx, is_warmup=is_warmup
            )
        return {**base, **row}
    except MemoryError as exc:
        return {
            **base,
            "repeat_idx": int(repeat_idx),
            "is_warmup": bool(is_warmup),
            "status": "resource_limit",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "resource_required_bytes": getattr(exc, "required_bytes", None),
            "resource_effective_cap_bytes": getattr(exc, "effective_cap_bytes", None),
            "resource_declared_cap_bytes": getattr(exc, "declared_cap_bytes", None),
            "resource_available_device_bytes": getattr(
                exc, "available_device_bytes", None
            ),
            "setup_seconds": math.nan,
            "solving_phase_seconds": math.nan,
            "train_total_seconds": math.nan,
            "test_rmse": math.nan,
        }
    except Exception as exc:
        return {
            **base,
            "repeat_idx": int(repeat_idx),
            "is_warmup": bool(is_warmup),
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "setup_seconds": math.nan,
            "solving_phase_seconds": math.nan,
            "train_total_seconds": math.nan,
            "test_rmse": math.nan,
        }


def _finite(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, math.nan))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def summarize_pipeline_rows(
    rows: Iterable[dict[str, Any]], cfg: EndToEndConfig
) -> list[dict[str, Any]]:
    measured = [row for row in rows if not bool(row.get("is_warmup", False))]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in measured:
        grouped.setdefault(str(row["method"]), []).append(row)
    for method_rows in grouped.values():
        method_rows.sort(key=lambda row: int(row.get("repeat_idx", -1)))

    def successful(row: dict[str, Any]) -> bool:
        return bool(
            str(row.get("status", "")).lower() in {"ok", "converged"}
            and _finite(row, "setup_seconds") is not None
            and _finite(row, "solving_phase_seconds") is not None
            and _finite(row, "train_total_seconds") is not None
            and _finite(row, "test_rmse") is not None
            and _finite(row, "test_r2") is not None
        )

    def diagnostic_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def aggregate_status(method_rows: list[dict[str, Any]]) -> str:
        if not method_rows:
            return "missing"
        if len(method_rows) == int(cfg.measured_repeats) and all(
            successful(row) for row in method_rows
        ):
            return "ok"
        statuses = {str(row.get("status", "missing")).lower() for row in method_rows}
        if "error" in statuses:
            return "error"
        if "resource_limit" in statuses:
            return "resource_limit"
        return "incomplete"

    summaries: list[dict[str, Any]] = []
    for method in cfg.methods:
        method_rows = grouped.get(method, [])
        ok = [row for row in method_rows if successful(row)]
        first = method_rows[0] if method_rows else {}
        diagnostic_source = ok[0] if ok else first
        for diagnostic_field in (
            "active_selection_rule",
            "effective_active_topk",
            "effective_active_box_size",
            "effective_active_rank",
            "capacity_adapted",
        ):
            diagnostic_values = [row.get(diagnostic_field) for row in ok]
            if diagnostic_values and any(
                value != diagnostic_values[0] for value in diagnostic_values[1:]
            ):
                raise RuntimeError(
                    f"{method} changed {diagnostic_field} across measured repeats"
                )
        summary: dict[str, Any] = {
            **{
                key: first.get(key)
                for key in (
                    "protocol_family",
                    "timing_scope",
                    "dataset_stem",
                    "dataset_family",
                    "n_train",
                    "n_test",
                    "dim",
                    "subset_seed",
                    "subset_mode",
                    "kernel_family",
                    "lengthscale",
                    "nu",
                    "variance",
                    "reg_lambda",
                    "regularization_convention",
                    "fourier_eps",
                    "nufft_tol",
                    "l2_scaled",
                    "precision",
                    "nufft_backend",
                    "precompute_chunk_size",
                    "box_budget",
                    "configured_active_rank",
                    "configured_full_eig_rank",
                    "configured_active_topk",
                    "configured_expected_active_box_size",
                    "configured_allow_frozen_topk_capacity_adaptation",
                    "parameter_selection_policy",
                    "parameter_source",
                    "accuracy_max_rmse",
                    "accuracy_min_r2",
                    "resource_required_bytes",
                    "resource_effective_cap_bytes",
                    "resource_declared_cap_bytes",
                    "resource_available_device_bytes",
                )
            },
            "method": method,
            "measured_repeats": len(method_rows),
            "expected_measured_repeats": int(cfg.measured_repeats),
            "successful_repeats": len(ok),
            "status": aggregate_status(method_rows),
            "active_selection_rule": diagnostic_source.get(
                "active_selection_rule"
            ),
            "effective_active_topk": diagnostic_source.get(
                "effective_active_topk"
            ),
            "effective_active_box_size": diagnostic_source.get(
                "effective_active_box_size"
            ),
            "effective_active_rank": diagnostic_source.get(
                "effective_active_rank"
            ),
            "capacity_adapted": diagnostic_bool(
                diagnostic_source.get("capacity_adapted")
            ),
        }
        for config_key in (
            "dataset_stem",
            "subset_seed",
            "subset_mode",
            "kernel_family",
            "lengthscale",
            "nu",
            "variance",
            "reg_lambda",
            "regularization_convention",
            "fourier_eps",
            "nufft_tol",
            "l2_scaled",
            "precision",
            "nufft_backend",
            "precompute_chunk_size",
            "box_budget",
            "parameter_selection_policy",
            "parameter_source",
            "accuracy_max_rmse",
            "accuracy_min_r2",
        ):
            if summary.get(config_key) is None:
                summary[config_key] = getattr(cfg, config_key)
        for key in (
            "setup_seconds",
            "solver_build_seconds",
            "iterative_solve_seconds",
            "solving_phase_seconds",
            "train_total_seconds",
            "prediction_seconds",
            "test_rmse",
            "test_mae",
            "test_r2",
            "iterations",
        ):
            values = [value for row in ok if (value := _finite(row, key)) is not None]
            summary[f"{key}_median"] = float(np.median(values)) if values else math.nan
            summary[f"{key}_min"] = float(np.min(values)) if values else math.nan
            summary[f"{key}_max"] = float(np.max(values)) if values else math.nan

        # For the stacked setup/solve figure, use the middle total-time row (or
        # the average of the two middle rows).  These paired components add
        # exactly to the reported total median, unlike two independent medians.
        if ok:
            ordered = sorted(ok, key=lambda row: float(row["train_total_seconds"]))
            middle = len(ordered) // 2
            representatives = (
                [ordered[middle]]
                if len(ordered) % 2
                else [ordered[middle - 1], ordered[middle]]
            )
            summary["setup_seconds_at_median_total"] = float(
                np.mean([float(row["setup_seconds"]) for row in representatives])
            )
            summary["solving_phase_seconds_at_median_total"] = float(
                np.mean(
                    [float(row["solving_phase_seconds"]) for row in representatives]
                )
            )
            summary["median_total_component_repeat_indices"] = [
                int(row["repeat_idx"]) for row in representatives
            ]
        else:
            summary["setup_seconds_at_median_total"] = math.nan
            summary["solving_phase_seconds_at_median_total"] = math.nan
            summary["median_total_component_repeat_indices"] = []
        summaries.append(summary)

    reference_rows = {
        int(row["repeat_idx"]): row
        for row in grouped.get(cfg.accuracy_reference_method, [])
        if successful(row)
    }
    expected_repeat_ids = set(range(int(cfg.measured_repeats)))
    for summary in summaries:
        method = str(summary["method"])
        method_rows = {
            int(row["repeat_idx"]): row
            for row in grouped.get(method, [])
            if successful(row)
        }
        usability_evaluated: list[dict[str, float | bool]] = []
        reference_evaluated: list[dict[str, float | bool]] = []
        for repeat_idx in sorted(expected_repeat_ids):
            method_row = method_rows.get(repeat_idx)
            if method_row is None:
                continue
            rmse = float(method_row["test_rmse"])
            r2 = float(method_row["test_r2"])
            absolute_pass = bool(
                cfg.accuracy_max_rmse is None or rmse <= float(cfg.accuracy_max_rmse)
            )
            r2_pass = bool(
                cfg.accuracy_min_r2 is None or r2 >= float(cfg.accuracy_min_r2)
            )
            usability_evaluated.append(
                {"rmse": rmse, "r2": r2, "passed": absolute_pass and r2_pass}
            )

            reference_row = reference_rows.get(repeat_idx)
            if reference_row is None:
                continue
            reference_rmse = float(reference_row["test_rmse"])
            relative_ratio = rmse / max(reference_rmse, np.finfo(float).tiny)
            reference_evaluated.append(
                {
                    "rmse": rmse,
                    "reference_rmse": reference_rmse,
                    "relative_ratio": relative_ratio,
                    "relative_delta": relative_ratio - 1.0,
                    "absolute_delta": rmse - reference_rmse,
                    "passed": bool(
                        rmse
                        <= (1.0 + float(cfg.accuracy_relative_tolerance))
                        * reference_rmse
                    ),
                }
            )

        execution_eligible = bool(
            summary["status"] == "ok"
            and summary["successful_repeats"] == int(cfg.measured_repeats)
            and summary["measured_repeats"] == int(cfg.measured_repeats)
        )
        usability_passed = sum(
            bool(item["passed"]) for item in usability_evaluated
        )
        reference_equivalent_repeats = sum(
            bool(item["passed"]) for item in reference_evaluated
        )
        summary["accuracy_reference_method"] = cfg.accuracy_reference_method
        summary["accuracy_relative_tolerance"] = float(cfg.accuracy_relative_tolerance)
        summary["accuracy_max_rmse"] = cfg.accuracy_max_rmse
        summary["accuracy_min_r2"] = cfg.accuracy_min_r2
        summary["execution_eligible"] = execution_eligible
        summary["usability_range_declared"] = bool(
            cfg.accuracy_max_rmse is not None or cfg.accuracy_min_r2 is not None
        )
        summary["usability_definition"] = (
            "Every measured repeat must execute successfully and satisfy the "
            "prospectively declared absolute RMSE ceiling and R2 floor when present; "
            "reference-relative equivalence is not part of this usable-quality range."
        )
        summary["usability_evaluated_repeats"] = len(usability_evaluated)
        summary["usability_passed_repeats"] = usability_passed
        summary["usability_eligible"] = bool(
            execution_eligible
            and len(usability_evaluated) == int(cfg.measured_repeats)
            and usability_passed == int(cfg.measured_repeats)
        )
        summary["reference_equivalence_definition"] = (
            "Descriptive near-equal-accuracy label: every measured repeat must be "
            "paired with the declared reference and have RMSE <= "
            "(1 + relative_tolerance) * reference RMSE. It does not suppress raw "
            "timing or time/accuracy trade-off results."
        )
        summary["reference_evaluated_repeats"] = len(reference_evaluated)
        summary["reference_equivalent_repeats"] = reference_equivalent_repeats
        summary["reference_equivalent"] = bool(
            execution_eligible
            and len(reference_evaluated) == int(cfg.measured_repeats)
            and reference_equivalent_repeats == int(cfg.measured_repeats)
        )
        reference_ratios = [
            float(item["relative_ratio"]) for item in reference_evaluated
        ]
        reference_relative_deltas = [
            float(item["relative_delta"]) for item in reference_evaluated
        ]
        reference_absolute_deltas = [
            float(item["absolute_delta"]) for item in reference_evaluated
        ]
        summary["rmse_ratio_to_reference_median"] = (
            float(np.median(reference_ratios)) if reference_ratios else math.nan
        )
        summary["rmse_ratio_to_reference_min"] = (
            float(np.min(reference_ratios)) if reference_ratios else math.nan
        )
        summary["rmse_ratio_to_reference_max"] = (
            float(np.max(reference_ratios)) if reference_ratios else math.nan
        )
        summary["rmse_relative_delta_to_reference_median"] = (
            float(np.median(reference_relative_deltas))
            if reference_relative_deltas
            else math.nan
        )
        summary["rmse_delta_from_reference_median"] = (
            float(np.median(reference_absolute_deltas))
            if reference_absolute_deltas
            else math.nan
        )

        # Backward-compatible aliases for older report readers. Their semantics
        # are deliberately the broad absolute usable-quality range, not the
        # reference-equivalence label. New code should consume usability_*.
        summary["accuracy_gate_definition"] = (
            "Deprecated compatibility alias: accuracy_eligible equals "
            "usability_eligible; see usability_definition and "
            "reference_equivalence_definition for the separated semantics."
        )
        summary["accuracy_eligible_legacy_alias_for"] = "usability_eligible"
        summary["accuracy_evaluated_repeats"] = len(usability_evaluated)
        summary["accuracy_passed_repeats"] = usability_passed
        summary["accuracy_eligible"] = summary["usability_eligible"]
        summary["quality_qualified_performance_eligible"] = bool(
            execution_eligible and summary["usability_eligible"]
        )
        summary["performance_claim_eligible_definition"] = (
            "Execution-complete and inside the broad absolute usable-quality range; "
            "reference equivalence is descriptive and is not required."
        )
        # Compatibility alias used by existing campaign/reporting code.
        summary["performance_claim_eligible"] = summary[
            "quality_qualified_performance_eligible"
        ]

    summaries_by_method = {str(row["method"]): row for row in summaries}
    ours = summaries_by_method.get("ours-binned-default")
    ours_rows = {
        int(row["repeat_idx"]): row
        for row in grouped.get("ours-binned-default", [])
        if successful(row)
    }
    for summary in summaries:
        method = str(summary["method"])
        method_rows = {
            int(row["repeat_idx"]): row
            for row in grouped.get(method, [])
            if successful(row)
        }
        paired_ids = sorted(set(method_rows).intersection(ours_rows))

        def paired_ratios(key: str) -> list[float]:
            ratios: list[float] = []
            for repeat_idx in paired_ids:
                denominator = _finite(ours_rows[repeat_idx], key)
                numerator = _finite(method_rows[repeat_idx], key)
                if denominator is None or denominator <= 0.0 or numerator is None:
                    return []
                ratios.append(numerator / denominator)
            return ratios

        def paired_differences(key: str) -> list[float]:
            differences: list[float] = []
            for repeat_idx in paired_ids:
                ours_value = _finite(ours_rows[repeat_idx], key)
                comparison_value = _finite(method_rows[repeat_idx], key)
                if ours_value is None or comparison_value is None:
                    return []
                differences.append(comparison_value - ours_value)
            return differences

        total_ratios = paired_ratios("train_total_seconds")
        setup_ratios = paired_ratios("setup_seconds")
        solve_ratios = paired_ratios("solving_phase_seconds")
        rmse_ratios = paired_ratios("test_rmse")
        rmse_deltas = paired_differences("test_rmse")
        summary["ours_speedup_definition"] = (
            "Raw matched-repeat comparison-method time / ours-binned-default time; "
            "values above one favor ours. These descriptive timings are retained "
            "whenever both runs succeeded, independent of usability or reference "
            "equivalence; use ours_speedup_claim_eligible for a quality-qualified "
            "headline comparison."
        )
        summary["ours_speedup_paired_repeats"] = len(total_ratios)
        summary["ours_speedup_complete_pairing"] = bool(
            set(paired_ids) == expected_repeat_ids
            and len(total_ratios) == int(cfg.measured_repeats)
        )
        summary["ours_speedup_claim_eligible"] = bool(
            summary["ours_speedup_complete_pairing"]
            and summary.get("quality_qualified_performance_eligible")
            and ours is not None
            and ours.get("quality_qualified_performance_eligible")
        )
        for label, values in (
            ("ours_total_speedup", total_ratios),
            ("ours_setup_speedup", setup_ratios),
            ("ours_solving_speedup", solve_ratios),
            ("comparison_rmse_ratio_to_ours", rmse_ratios),
            ("comparison_rmse_delta_from_ours", rmse_deltas),
        ):
            summary[label] = float(np.median(values)) if values else math.nan
            summary[f"{label}_min"] = float(np.min(values)) if values else math.nan
            summary[f"{label}_max"] = float(np.max(values)) if values else math.nan
    return summaries


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_end_to_end_experiment(cfg: EndToEndConfig) -> dict[str, Any]:
    _validate_config(cfg)
    dataset = load_end_to_end_dataset(cfg)
    output = (
        Path(cfg.output_dir).expanduser().resolve()
        if cfg.output_dir
        else (Path.cwd() / "end_to_end_krr_results" / cfg.dataset_stem).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total_repeats = int(cfg.warmup_repeats) + int(cfg.measured_repeats)
    for repeat in range(total_repeats):
        is_warmup = repeat < int(cfg.warmup_repeats)
        repeat_idx = repeat - int(cfg.warmup_repeats) if not is_warmup else repeat
        order = list(cfg.methods)
        np.random.default_rng(int(cfg.method_order_seed) + repeat).shuffle(order)
        for method in order:
            _release_gpu_allocator_cache()
            row = run_pipeline_once(
                method,
                cfg,
                dataset,
                repeat_idx=repeat_idx,
                is_warmup=is_warmup,
            )
            rows.append(row)
            _write_rows(output / "pipeline_runs.json", rows)
            _write_rows(output / "pipeline_runs.csv", rows)
            _release_gpu_allocator_cache()
    summary = summarize_pipeline_rows(rows, cfg)
    _write_rows(output / "pipeline_summary.json", summary)
    _write_rows(output / "pipeline_summary.csv", summary)
    (output / "experiment_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    expected_row_keys = {
        (method, True, repeat_idx)
        for repeat_idx in range(int(cfg.warmup_repeats))
        for method in cfg.methods
    }.union(
        {
            (method, False, repeat_idx)
            for repeat_idx in range(int(cfg.measured_repeats))
            for method in cfg.methods
        }
    )
    observed_row_keys = [
        (
            str(row.get("method")),
            bool(row.get("is_warmup", False)),
            int(row.get("repeat_idx", -1)),
        )
        for row in rows
    ]
    summary_by_method = {str(row.get("method")): row for row in summary}
    resource_limit_methods = sorted(
        method
        for method, row in summary_by_method.items()
        if str(row.get("status")) == "resource_limit"
    )
    error_methods = sorted(
        method
        for method, row in summary_by_method.items()
        if str(row.get("status")) in {"error", "incomplete", "missing"}
    )
    ineligible_methods = sorted(
        method
        for method, row in summary_by_method.items()
        if not bool(row.get("performance_claim_eligible"))
    )
    all_rows_present = bool(
        len(observed_row_keys) == len(expected_row_keys)
        and len(set(observed_row_keys)) == len(observed_row_keys)
        and set(observed_row_keys) == expected_row_keys
        and set(summary_by_method) == set(cfg.methods)
    )
    proposed = summary_by_method.get("ours-binned-default")
    proposed_claim_eligible = bool(
        proposed is not None and proposed.get("performance_claim_eligible") is True
    )
    if error_methods:
        formal_result_status = "execution_error"
    elif resource_limit_methods:
        formal_result_status = "complete_with_resource_limits"
    elif ineligible_methods:
        formal_result_status = "complete_with_usability_ineligible_methods"
    else:
        formal_result_status = "claim_eligible_complete"
    completion = {
        "protocol_family": PROTOCOL_FAMILY,
        "timing_scope": TIMING_SCOPE,
        "methods": list(cfg.methods),
        "n_train": int(dataset["x"].shape[0]),
        "n_test": int(dataset["n_test"]),
        "expected_row_count": len(expected_row_keys),
        "observed_row_count": len(rows),
        "all_rows_present": all_rows_present,
        "artifact_complete": all_rows_present,
        "formal_result_status": formal_result_status,
        "resource_limit_methods": resource_limit_methods,
        "error_methods": error_methods,
        "performance_ineligible_methods": ineligible_methods,
        "all_methods_executed_successfully": bool(
            all_rows_present
            and not resource_limit_methods
            and not error_methods
            and all(str(row.get("status")) == "ok" for row in summary)
        ),
        "all_methods_performance_claim_eligible": bool(
            all_rows_present
            and len(summary) == len(cfg.methods)
            and all(bool(row.get("performance_claim_eligible")) for row in summary)
        ),
        "proposed_performance_claim_eligible": proposed_claim_eligible,
        # Backward-compatible alias.  It is deliberately non-vacuous.
        "all_claimed_rows_eligible": proposed_claim_eligible,
    }
    (output / "run_complete.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "output_dir": str(output),
        "rows": rows,
        "summary": summary,
        "completion": completion,
    }


def _parse_methods(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(raw).split(",") if token.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = EndToEndConfig()
    parser = argparse.ArgumentParser(
        description="Run end-to-end KRR pipeline comparisons."
    )
    parser.add_argument("--dataset-stem", default=defaults.dataset_stem)
    parser.add_argument("--dataset-dir", default=defaults.dataset_dir)
    parser.add_argument("--n-train", type=int, default=0, help="0 means all rows")
    parser.add_argument(
        "--subset-mode", choices=("prefix", "random"), default=defaults.subset_mode
    )
    parser.add_argument("--max-test-rows", type=int, default=defaults.max_test_rows)
    parser.add_argument("--kernel-family", default=defaults.kernel_family)
    parser.add_argument("--lengthscale", type=float, default=defaults.lengthscale)
    parser.add_argument("--nu", type=float, default=defaults.nu)
    parser.add_argument("--variance", type=float, default=defaults.variance)
    parser.add_argument("--reg-lambda", type=float, default=defaults.reg_lambda)
    parser.add_argument("--fourier-eps", type=float, default=defaults.fourier_eps)
    parser.add_argument("--tol", type=float, default=defaults.tol)
    parser.add_argument("--maxiter", type=int, default=defaults.maxiter)
    parser.add_argument("--methods", default=",".join(defaults.methods))
    parser.add_argument("--rank", type=int, default=defaults.rank)
    parser.add_argument("--full-eig-rank", type=int, default=None)
    parser.add_argument("--active-topk", type=int, default=None)
    parser.add_argument("--expected-active-box-size", type=int, default=None)
    parser.add_argument(
        "--allow-frozen-topk-capacity-adaptation",
        action="store_true",
        help="Allow deterministic top-k shortening in declared robustness systems.",
    )
    parser.add_argument(
        "--parameter-selection-policy",
        default=defaults.parameter_selection_policy,
    )
    parser.add_argument("--parameter-source", default=defaults.parameter_source)
    parser.add_argument("--nystrom-rank", type=int, default=defaults.nystrom_rank)
    parser.add_argument("--rpcholesky-rank", type=int, default=defaults.rpcholesky_rank)
    parser.add_argument("--box-budget", type=int, default=defaults.box_budget)
    parser.add_argument(
        "--inverse-max-size", type=int, default=defaults.inverse_max_size
    )
    parser.add_argument(
        "--low-rank-chunk-size", type=int, default=defaults.low_rank_chunk_size
    )
    parser.add_argument(
        "--low-rank-dtype", choices=("fp32", "fp64"), default=defaults.low_rank_dtype
    )
    parser.add_argument("--warmup-repeats", type=int, default=defaults.warmup_repeats)
    parser.add_argument(
        "--measured-repeats", type=int, default=defaults.measured_repeats
    )
    parser.add_argument(
        "--accuracy-max-rmse",
        type=float,
        default=defaults.accuracy_max_rmse,
        help="Prospective absolute RMSE ceiling; omit only for exploratory runs.",
    )
    parser.add_argument(
        "--accuracy-min-r2",
        type=float,
        default=defaults.accuracy_min_r2,
        help="Prospective absolute R2 floor; omit only for exploratory runs.",
    )
    parser.add_argument(
        "--accuracy-relative-tolerance",
        type=float,
        default=defaults.accuracy_relative_tolerance,
    )
    parser.add_argument("--nufft-backend", default=defaults.nufft_backend)
    parser.add_argument("--output-dir", required=True)
    return parser


def config_from_args(args: argparse.Namespace) -> EndToEndConfig:
    return EndToEndConfig(
        dataset_stem=str(args.dataset_stem),
        dataset_dir=str(args.dataset_dir),
        n_train=None if int(args.n_train) == 0 else int(args.n_train),
        subset_mode=str(args.subset_mode),
        max_test_rows=int(args.max_test_rows),
        kernel_family=str(args.kernel_family),
        lengthscale=float(args.lengthscale),
        nu=float(args.nu),
        variance=float(args.variance),
        reg_lambda=float(args.reg_lambda),
        fourier_eps=float(args.fourier_eps),
        tol=float(args.tol),
        maxiter=int(args.maxiter),
        methods=_parse_methods(args.methods),
        rank=int(args.rank),
        full_eig_rank=(
            None if args.full_eig_rank is None else int(args.full_eig_rank)
        ),
        active_topk=None if args.active_topk is None else int(args.active_topk),
        expected_active_box_size=(
            None
            if args.expected_active_box_size is None
            else int(args.expected_active_box_size)
        ),
        allow_frozen_topk_capacity_adaptation=bool(
            args.allow_frozen_topk_capacity_adaptation
        ),
        parameter_selection_policy=str(args.parameter_selection_policy),
        parameter_source=str(args.parameter_source),
        nystrom_rank=int(args.nystrom_rank),
        rpcholesky_rank=int(args.rpcholesky_rank),
        box_budget=int(args.box_budget),
        inverse_max_size=int(args.inverse_max_size),
        low_rank_chunk_size=int(args.low_rank_chunk_size),
        low_rank_dtype=str(args.low_rank_dtype),
        warmup_repeats=int(args.warmup_repeats),
        measured_repeats=int(args.measured_repeats),
        accuracy_max_rmse=(
            None if args.accuracy_max_rmse is None else float(args.accuracy_max_rmse)
        ),
        accuracy_min_r2=(
            None if args.accuracy_min_r2 is None else float(args.accuracy_min_r2)
        ),
        accuracy_relative_tolerance=float(args.accuracy_relative_tolerance),
        nufft_backend=str(args.nufft_backend),
        output_dir=str(args.output_dir),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_end_to_end_experiment(config_from_args(args))
    print(f"wrote end-to-end KRR results to {result['output_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
