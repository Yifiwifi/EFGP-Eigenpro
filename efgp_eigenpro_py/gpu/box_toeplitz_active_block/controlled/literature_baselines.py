"""Isolated adapters for literature KRR baselines.

The adapters in this module deliberately do not share the Fourier system used
by the controlled preconditioner experiments.  They own their model setup,
training, and prediction, which makes their wall-clock rows suitable for an
end-to-end comparison.

Two normalization details are important:

* The repository's restricted KRR convention uses an *absolute* ridge,

  ``(K_nm.T @ K_nm + lambda_abs * K_mm) alpha = K_nm.T @ y``.

  FALKON calls its public regularization argument ``penalty`` and solves the
  same system with ``n * penalty * K_mm``.  Therefore the matched value is
  exactly ``penalty = lambda_abs / n``.  Passing ``lambda_abs`` directly would
  over-regularize by a factor of ``n``.

* The random Fourier feature (RFF) path solves

  ``(Phi.T @ Phi + lambda_abs * I) beta = Phi.T @ y``

  and accumulates both sufficient statistics in row chunks.  It never stores
  the full ``n x num_features`` feature matrix.

The FALKON import is lazy because official FALKON wheels are platform- and
PyTorch/CUDA-specific.  A separately labeled native FALKON implementation
provides the published two-Cholesky preconditioner and streamed kernel
matrix-vector products on NumPy/CuPy when the official package is unavailable.
RFF likewise has a NumPy backend and an optional CuPy backend.

References
----------
Rudi, Carratino, and Rosasco, "FALKON: An Optimal Large Scale Kernel
Method", NeurIPS 2017.

Meanti, Carratino, Rosasco, and Rudi, "Kernel Methods Through the Roof:
Handling Billions of Points Efficiently", NeurIPS 2020.

Rahimi and Recht, "Random Features for Large-Scale Kernel Machines",
NeurIPS 2007.
"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np


_FALKON_CITATIONS = ("rudi2017falkon", "meanti2020kernelroof")
_RFF_CITATIONS = ("rahimi2007random",)


@dataclass(frozen=True)
class FalkonKRRConfig:
    """Configuration for the official FALKON KRR adapter.

    ``kernel_variance`` is currently required to be one.  Official FALKON's
    built-in Matérn and Gaussian kernels have unit amplitude; failing closed
    avoids silently benchmarking a different kernel.
    """

    nystrom_centers: int = 256
    maxiter: int = 20
    seed: int = 0
    kernel_family: str = "matern"
    lengthscale: float = 0.1
    nu: float = 1.5
    kernel_variance: float = 1.0
    absolute_ridge: float = 0.1
    precision: str = "fp64"
    prediction_chunk_size: int = 250_000
    use_cpu: bool = False
    require_cuda: bool = False
    keops_active: str = "auto"
    never_store_kernel: bool = True
    max_gpu_mem_bytes: int | None = None
    max_cpu_mem_bytes: int | None = None
    cg_tolerance: float = 1e-7
    debug: bool = False
    allow_input_copy: bool = False


@dataclass(frozen=True)
class NativeFalkonKRRConfig:
    """Configuration for the dependency-free native FALKON algorithm.

    This is an independent implementation of the FALKON linear algebra, not
    the official ``falkon`` package.  It uses the two-Cholesky preconditioner
    of Rudi et al. and streamed ``K_nm`` matrix-vector products.
    """

    nystrom_centers: int = 256
    maxiter: int = 20
    tolerance: float = 1e-7
    seed: int = 0
    kernel_family: str = "matern"
    lengthscale: float = 0.1
    nu: float = 1.5
    kernel_variance: float = 1.0
    absolute_ridge: float = 0.1
    train_chunk_size: int = 100_000
    prediction_chunk_size: int = 250_000
    precision: str = "fp64"
    backend: str = "auto"
    preconditioner_jitter: float | None = None
    time_budget_seconds: float | None = None


@dataclass(frozen=True)
class MaternRFFRidgeConfig:
    """Configuration for streaming Matérn random-feature ridge regression."""

    num_features: int = 256
    seed: int = 0
    lengthscale: float = 0.1
    nu: float = 1.5
    kernel_variance: float = 1.0
    absolute_ridge: float = 0.1
    train_chunk_size: int = 100_000
    prediction_chunk_size: int = 250_000
    precision: str = "fp64"
    backend: str = "auto"
    time_budget_seconds: float | None = None


@dataclass
class MaternRFFModel:
    """A fitted RFF model whose arrays live on ``array_module``."""

    frequencies: Any
    phases: Any
    coefficients: Any
    feature_scale: float
    input_dim: int
    array_module: Any
    backend_name: str
    precision: str


@dataclass
class NativeFalkonModel:
    """Fitted coefficients and centers from the native FALKON algorithm."""

    centers: Any
    coefficients: Any
    input_dim: int
    array_module: Any
    backend_name: str
    precision: str
    kernel_family: str
    lengthscale: float
    nu: float
    kernel_variance: float


def _regression_metrics_from_sums(
    *,
    n_rows: int,
    squared_error: float,
    absolute_error: float,
    target_sum: float,
    target_square_sum: float,
) -> dict[str, float]:
    """Finalize RMSE, MAE, and R2 from streamable scalar statistics."""

    count = int(n_rows)
    if count <= 0:
        raise ValueError("n_rows must be positive.")
    sse = max(0.0, float(squared_error))
    sae = max(0.0, float(absolute_error))
    centered_target_sum = max(
        0.0,
        float(target_square_sum) - (float(target_sum) * float(target_sum)) / count,
    )
    scale = max(1.0, abs(float(target_square_sum)))
    if centered_target_sum <= np.finfo(np.float64).eps * scale:
        r2 = 1.0 if sse <= np.finfo(np.float64).eps * scale else 0.0
    else:
        r2 = 1.0 - sse / centered_target_sum
    rmse = math.sqrt(sse / count)
    mae = sae / count
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
    }


def falkon_penalty_from_absolute_ridge(absolute_ridge: float, n_train: int) -> float:
    """Convert this repository's absolute KRR ridge to FALKON ``penalty``.

    FALKON's sketched system is

    ``K_nm.T K_nm + n_train * penalty * K_mm``.

    Matching ``K_nm.T K_nm + absolute_ridge * K_mm`` therefore requires
    ``penalty = absolute_ridge / n_train``.
    """

    ridge = float(absolute_ridge)
    n_rows = int(n_train)
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("absolute_ridge must be finite and positive.")
    if n_rows <= 0:
        raise ValueError("n_train must be positive.")
    return ridge / n_rows


def _precision_dtypes(precision: str) -> tuple[np.dtype[Any], str]:
    key = str(precision).strip().lower()
    if key in {"fp64", "float64", "double"}:
        return np.dtype(np.float64), "float64"
    if key in {"fp32", "float32", "single", "mixed32"}:
        return np.dtype(np.float32), "float32"
    raise ValueError("precision must be 'fp64' or 'fp32' (alias 'mixed32').")


def _validate_xy_shapes(x: Any, y: Any, *, prefix: str) -> tuple[int, int]:
    if not hasattr(x, "shape") or len(x.shape) != 2:
        raise ValueError(f"{prefix}_x must be a two-dimensional array.")
    if not hasattr(y, "shape") or len(y.shape) not in {1, 2}:
        raise ValueError(f"{prefix}_y must have shape (n,) or (n, 1).")
    n_rows = int(x.shape[0])
    dim = int(x.shape[1])
    y_rows = int(y.shape[0])
    if n_rows <= 0 or dim <= 0:
        raise ValueError(f"{prefix}_x must have positive dimensions.")
    if y_rows != n_rows:
        raise ValueError(f"{prefix}_x and {prefix}_y row counts differ.")
    if len(y.shape) == 2 and int(y.shape[1]) != 1:
        raise ValueError(f"{prefix}_y must contain exactly one regression target.")
    return n_rows, dim


def _validate_falkon_config(cfg: FalkonKRRConfig) -> None:
    if int(cfg.nystrom_centers) <= 0:
        raise ValueError("nystrom_centers must be positive.")
    if int(cfg.maxiter) <= 0:
        raise ValueError("maxiter must be positive.")
    if int(cfg.prediction_chunk_size) <= 0:
        raise ValueError("prediction_chunk_size must be positive.")
    if not math.isfinite(float(cfg.lengthscale)) or float(cfg.lengthscale) <= 0.0:
        raise ValueError("lengthscale must be finite and positive.")
    if not math.isfinite(float(cfg.kernel_variance)) or not math.isclose(
        float(cfg.kernel_variance), 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(
            "the official FALKON built-in kernel adapter requires "
            "kernel_variance=1.0"
        )
    falkon_penalty_from_absolute_ridge(cfg.absolute_ridge, 1)
    _precision_dtypes(cfg.precision)
    if not math.isfinite(float(cfg.cg_tolerance)) or float(cfg.cg_tolerance) <= 0.0:
        raise ValueError("cg_tolerance must be finite and positive.")
    for label, value in (
        ("max_gpu_mem_bytes", cfg.max_gpu_mem_bytes),
        ("max_cpu_mem_bytes", cfg.max_cpu_mem_bytes),
    ):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{label} must be positive or None.")
    family = str(cfg.kernel_family).strip().lower()
    if family in {"matern", "mat", "mat32"}:
        if float(cfg.nu) not in {0.5, 1.5, 2.5}:
            raise ValueError("FALKON Matérn supports nu in {0.5, 1.5, 2.5}.")
    elif family not in {"se", "rbf", "gaussian", "squared_exponential"}:
        raise ValueError(f"unsupported FALKON kernel_family={cfg.kernel_family!r}")


def _load_torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("the FALKON baseline requires PyTorch") from exc


def _load_falkon_module() -> Any:
    try:
        return importlib.import_module("falkon")
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "the FALKON baseline requires the official 'falkon' package; "
            "install a wheel matching the host PyTorch/CUDA versions"
        ) from exc


def _torch_cuda_available(torch_module: Any) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available"):
        return False
    return bool(cuda.is_available())


def _sync_torch_cuda(torch_module: Any) -> None:
    if _torch_cuda_available(torch_module):
        torch_module.cuda.synchronize()


def _as_cpu_torch_tensor(
    value: Any,
    *,
    name: str,
    np_dtype: np.dtype[Any],
    torch_dtype: Any,
    torch_module: Any,
    allow_copy: bool,
) -> tuple[Any, bool]:
    """Wrap a NumPy/memmap input without copying when dtype/layout permits."""

    if torch_module.is_tensor(value):
        tensor = value
        device_type = getattr(getattr(tensor, "device", None), "type", None)
        if device_type != "cpu":
            raise ValueError(
                f"{name} must be a CPU tensor; FALKON streams CPU input blocks "
                "to its selected compute devices"
            )
        copied = False
        if tensor.dtype != torch_dtype:
            if not allow_copy:
                raise ValueError(
                    f"{name} has dtype {tensor.dtype}; expected {torch_dtype}. "
                    "Set allow_input_copy=True to permit a full-size conversion."
                )
            tensor = tensor.to(dtype=torch_dtype)
            copied = True
        return tensor, copied

    array = np.asarray(value)
    copied = False
    if array.dtype != np_dtype:
        if not allow_copy:
            raise ValueError(
                f"{name} has dtype {array.dtype}; expected {np_dtype}. "
                "Set allow_input_copy=True to permit a full-size conversion."
            )
        array = np.asarray(array, dtype=np_dtype)
        copied = True
    if not (array.flags.c_contiguous or array.flags.f_contiguous):
        if not allow_copy:
            raise ValueError(
                f"{name} is neither C- nor Fortran-contiguous. Set "
                "allow_input_copy=True to permit a contiguous copy."
            )
        array = np.ascontiguousarray(array)
        copied = True
    return torch_module.from_numpy(array), copied


def _prediction_to_numpy(prediction: Any) -> np.ndarray:
    out = prediction
    if hasattr(out, "detach"):
        out = out.detach()
    if hasattr(out, "cpu"):
        out = out.cpu()
    if hasattr(out, "numpy"):
        out = out.numpy()
    return np.asarray(out).reshape(-1)


def run_falkon_krr(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    cfg: FalkonKRRConfig,
    *,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Fit official FALKON and score it with chunked, non-materialized output.

    Matching-dtype NumPy arrays (including arrays backed by ``np.memmap``) are
    exposed to PyTorch as zero-copy CPU tensors.  FALKON performs its kernel
    matrix-vector products in blocks; ``never_store_kernel=True`` additionally
    prevents materialization of the full ``K_nm`` matrix.  Test prediction and
    RMSE accumulation are explicitly row-streamed here.
    """

    _validate_falkon_config(cfg)
    n_train, dim = _validate_xy_shapes(x_train, y_train, prefix="train")
    n_test, test_dim = _validate_xy_shapes(x_test, y_test, prefix="test")
    if test_dim != dim:
        raise ValueError("training and test feature counts differ.")
    if n_train < 2:
        raise ValueError("FALKON requires at least two training rows.")

    torch_module = _load_torch_module()
    falkon_module = _load_falkon_module()
    cuda_available = _torch_cuda_available(torch_module)
    if cfg.require_cuda and not cuda_available:
        raise RuntimeError("require_cuda=True but PyTorch reports no CUDA device.")

    np_dtype, torch_dtype_name = _precision_dtypes(cfg.precision)
    torch_dtype = getattr(torch_module, torch_dtype_name)
    train_x, copied_x = _as_cpu_torch_tensor(
        x_train,
        name="train_x",
        np_dtype=np_dtype,
        torch_dtype=torch_dtype,
        torch_module=torch_module,
        allow_copy=bool(cfg.allow_input_copy),
    )
    train_y, copied_y = _as_cpu_torch_tensor(
        y_train,
        name="train_y",
        np_dtype=np_dtype,
        torch_dtype=torch_dtype,
        torch_module=torch_module,
        allow_copy=bool(cfg.allow_input_copy),
    )
    train_y = train_y.reshape(-1, 1)

    penalty = falkon_penalty_from_absolute_ridge(cfg.absolute_ridge, n_train)
    option_kwargs: dict[str, Any] = {
        "use_cpu": bool(cfg.use_cpu),
        "keops_active": str(cfg.keops_active),
        "never_store_kernel": bool(cfg.never_store_kernel),
        "cg_tolerance": float(cfg.cg_tolerance),
        "debug": bool(cfg.debug),
    }
    if cfg.max_gpu_mem_bytes is not None:
        option_kwargs["max_gpu_mem"] = float(cfg.max_gpu_mem_bytes)
    if cfg.max_cpu_mem_bytes is not None:
        option_kwargs["max_cpu_mem"] = float(cfg.max_cpu_mem_bytes)

    _sync_torch_cuda(torch_module)
    setup_start = timer()
    options = falkon_module.FalkonOptions(**option_kwargs)
    family = str(cfg.kernel_family).strip().lower()
    if family in {"matern", "mat", "mat32"}:
        kernel = falkon_module.kernels.MaternKernel(
            sigma=float(cfg.lengthscale), nu=float(cfg.nu), opt=options
        )
        canonical_family = "matern"
    else:
        kernel = falkon_module.kernels.GaussianKernel(
            sigma=float(cfg.lengthscale), opt=options
        )
        canonical_family = "squared_exponential"
    effective_centers = min(int(cfg.nystrom_centers), n_train - 1)
    model = falkon_module.Falkon(
        kernel=kernel,
        penalty=penalty,
        M=effective_centers,
        center_selection="uniform",
        maxiter=int(cfg.maxiter),
        seed=int(cfg.seed),
        error_every=None,
        options=options,
    )
    _sync_torch_cuda(torch_module)
    setup_seconds = float(timer() - setup_start)

    fit_start = timer()
    model.fit(train_x, train_y)
    _sync_torch_cuda(torch_module)
    fit_seconds = float(timer() - fit_start)

    prediction_start = timer()
    squared_error = 0.0
    absolute_error = 0.0
    target_sum = 0.0
    target_square_sum = 0.0
    prediction_chunks = 0
    copied_test_chunk = False
    chunk_size = int(cfg.prediction_chunk_size)
    for start in range(0, n_test, chunk_size):
        stop = min(n_test, start + chunk_size)
        test_x_chunk, chunk_copied = _as_cpu_torch_tensor(
            x_test[start:stop],
            name="test_x_chunk",
            np_dtype=np_dtype,
            torch_dtype=torch_dtype,
            torch_module=torch_module,
            allow_copy=bool(cfg.allow_input_copy),
        )
        copied_test_chunk = copied_test_chunk or chunk_copied
        predicted = _prediction_to_numpy(model.predict(test_x_chunk))
        truth = np.asarray(y_test[start:stop], dtype=np.float64).reshape(-1)
        if predicted.size != truth.size:
            raise RuntimeError("FALKON prediction row count differs from y_test.")
        residual = predicted.astype(np.float64, copy=False) - truth
        squared_error += float(np.dot(residual, residual))
        absolute_error += float(np.abs(residual).sum(dtype=np.float64))
        target_sum += float(truth.sum(dtype=np.float64))
        target_square_sum += float(np.dot(truth, truth))
        prediction_chunks += 1
    _sync_torch_cuda(torch_module)
    prediction_seconds = float(timer() - prediction_start)

    train_total = setup_seconds + fit_seconds
    metrics = _regression_metrics_from_sums(
        n_rows=n_test,
        squared_error=squared_error,
        absolute_error=absolute_error,
        target_sum=target_sum,
        target_square_sum=target_square_sum,
    )
    return {
        "status": "ok",
        "method": "falkon-krr",
        "implementation": "official_falkon_package",
        "official_falkon_package": True,
        "pipeline_family": "literature_data_space_krr",
        "citations": list(_FALKON_CITATIONS),
        "n_train": n_train,
        "n_test": n_test,
        "input_dim": dim,
        "kernel_family": canonical_family,
        "lengthscale": float(cfg.lengthscale),
        "nu": float(cfg.nu) if canonical_family == "matern" else None,
        "kernel_variance": float(cfg.kernel_variance),
        "absolute_ridge": float(cfg.absolute_ridge),
        "regularization_convention": "absolute",
        "falkon_penalty": penalty,
        "falkon_penalty_identity": "penalty=absolute_ridge/n_train",
        "nystrom_centers": effective_centers,
        "requested_nystrom_centers": int(cfg.nystrom_centers),
        "maxiter": int(cfg.maxiter),
        "seed": int(cfg.seed),
        "precision": str(cfg.precision),
        "never_store_kernel": bool(cfg.never_store_kernel),
        "cuda_available": cuda_available,
        "use_cpu": bool(cfg.use_cpu),
        "setup_seconds": setup_seconds,
        "fit_seconds": fit_seconds,
        "train_total_seconds": train_total,
        "prediction_seconds": prediction_seconds,
        **metrics,
        "prediction_chunks": prediction_chunks,
        "train_input_copied": bool(copied_x or copied_y),
        "test_input_copied": copied_test_chunk,
        "timing_scope": (
            "setup_seconds times FALKON options/kernel/model construction; "
            "fit_seconds times model.fit including center selection, "
            "preconditioner construction, and PCG; prediction is separate"
        ),
        "config": asdict(cfg),
        "falkon_version": str(getattr(falkon_module, "__version__", "unknown")),
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
    }


def _validate_rff_config(cfg: MaternRFFRidgeConfig) -> None:
    if int(cfg.num_features) <= 0:
        raise ValueError("num_features must be positive.")
    if int(cfg.train_chunk_size) <= 0:
        raise ValueError("train_chunk_size must be positive.")
    if int(cfg.prediction_chunk_size) <= 0:
        raise ValueError("prediction_chunk_size must be positive.")
    for name, value in (
        ("lengthscale", cfg.lengthscale),
        ("nu", cfg.nu),
        ("kernel_variance", cfg.kernel_variance),
        ("absolute_ridge", cfg.absolute_ridge),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    _precision_dtypes(cfg.precision)
    if str(cfg.backend).strip().lower() not in {"auto", "numpy", "cupy"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'cupy'.")
    _validate_time_budget(cfg.time_budget_seconds)


def sample_matern_rff_parameters(
    input_dim: int, cfg: MaternRFFRidgeConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Sample angular frequencies and phases for the configured Matérn kernel.

    For

    ``k(r) = variance * Matérn_nu(sqrt(2*nu) * r / lengthscale)``,

    the normalized angular-frequency spectrum is a multivariate Student t
    distribution with ``df=2*nu`` and scale matrix
    ``lengthscale**-2 * I``.  Sampling ``g ~ N(0, I)`` and
    ``u ~ chi2(2*nu)`` gives

    ``omega = (g / lengthscale) * sqrt(2*nu / u)``.
    """

    _validate_rff_config(cfg)
    dim = int(input_dim)
    if dim <= 0:
        raise ValueError("input_dim must be positive.")
    rng = np.random.default_rng(int(cfg.seed))
    count = int(cfg.num_features)
    degrees_of_freedom = 2.0 * float(cfg.nu)
    normals = rng.standard_normal(size=(count, dim))
    chi_square = rng.chisquare(degrees_of_freedom, size=(count, 1))
    frequencies = (normals / float(cfg.lengthscale)) * np.sqrt(
        degrees_of_freedom / chi_square
    )
    phases = rng.uniform(0.0, 2.0 * np.pi, size=count)
    np_dtype, _ = _precision_dtypes(cfg.precision)
    return (
        np.asarray(frequencies, dtype=np_dtype),
        np.asarray(phases, dtype=np_dtype),
    )


def _resolve_backend(backend: str, array_module: Any | None) -> Any:
    if array_module is not None:
        return array_module
    key = str(backend).strip().lower()
    if key == "numpy":
        return np
    if key == "cupy":
        try:
            return importlib.import_module("cupy")
        except Exception as exc:  # pragma: no cover - optional GPU runtime
            raise RuntimeError(
                "backend='cupy' requires a working CuPy install"
            ) from exc
    if key != "auto":
        raise ValueError("backend must be 'auto', 'numpy', or 'cupy'.")
    try:  # auto: prefer a usable CUDA backend, otherwise remain portable.
        cupy = importlib.import_module("cupy")
        if int(cupy.cuda.runtime.getDeviceCount()) > 0:
            return cupy
    except Exception:
        pass
    return np


def _resolve_array_module(cfg: MaternRFFRidgeConfig, array_module: Any | None) -> Any:
    return _resolve_backend(cfg.backend, array_module)


def _array_module_name(xp: Any) -> str:
    name = str(getattr(xp, "__name__", type(xp).__name__))
    return "cupy" if name.startswith("cupy") else "numpy" if name == "numpy" else name


def _sync_array_module(xp: Any) -> None:
    cuda = getattr(xp, "cuda", None)
    if cuda is not None:
        cuda.Stream.null.synchronize()


def _validate_time_budget(value: float | None) -> None:
    if value is not None and (
        not math.isfinite(float(value)) or float(value) <= 0.0
    ):
        raise ValueError("time_budget_seconds must be positive or None.")


def _check_deadline(
    deadline: float | None,
    timer: Callable[[], float],
    *,
    stage: str,
) -> None:
    if deadline is not None and timer() >= float(deadline):
        raise TimeoutError(f"literature baseline time budget exhausted during {stage}")


def _xp_dtype(xp: Any, precision: str) -> Any:
    _, dtype_name = _precision_dtypes(precision)
    return getattr(xp, dtype_name)


def _validate_native_falkon_config(cfg: NativeFalkonKRRConfig) -> None:
    if int(cfg.nystrom_centers) <= 0:
        raise ValueError("nystrom_centers must be positive.")
    if int(cfg.maxiter) <= 0:
        raise ValueError("maxiter must be positive.")
    if int(cfg.train_chunk_size) <= 0:
        raise ValueError("train_chunk_size must be positive.")
    if int(cfg.prediction_chunk_size) <= 0:
        raise ValueError("prediction_chunk_size must be positive.")
    for name, value in (
        ("tolerance", cfg.tolerance),
        ("lengthscale", cfg.lengthscale),
        ("kernel_variance", cfg.kernel_variance),
        ("absolute_ridge", cfg.absolute_ridge),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    family = str(cfg.kernel_family).strip().lower()
    if family in {"matern", "mat", "mat32"}:
        if float(cfg.nu) not in {0.5, 1.5, 2.5}:
            raise ValueError("native FALKON Matérn supports nu in {0.5,1.5,2.5}.")
    elif family not in {"se", "rbf", "gaussian", "squared_exponential"}:
        raise ValueError(
            f"unsupported native FALKON kernel_family={cfg.kernel_family!r}"
        )
    _precision_dtypes(cfg.precision)
    if str(cfg.backend).strip().lower() not in {"auto", "numpy", "cupy"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'cupy'.")
    if cfg.preconditioner_jitter is not None and (
        not math.isfinite(float(cfg.preconditioner_jitter))
        or float(cfg.preconditioner_jitter) <= 0.0
    ):
        raise ValueError("preconditioner_jitter must be positive or None.")
    _validate_time_budget(cfg.time_budget_seconds)


def _native_kernel_cross(
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
    """Stationary kernel block used by the native streamed FALKON path."""

    xx = xp.asarray(x, dtype=dtype)
    zz = xp.asarray(z, dtype=dtype)
    if xx.ndim != 2 or zz.ndim != 2 or int(xx.shape[1]) != int(zz.shape[1]):
        raise ValueError("x and z must be 2D arrays with matching feature counts.")
    squared = (
        xp.sum(xx * xx, axis=1).reshape(-1, 1)
        + xp.sum(zz * zz, axis=1).reshape(1, -1)
        - 2.0 * (xx @ zz.T)
    )
    squared = xp.maximum(squared, xp.asarray(0.0, dtype=dtype))
    canonical_family = str(family).strip().lower()
    ell = float(lengthscale)
    var = float(variance)
    if canonical_family in {"se", "rbf", "gaussian", "squared_exponential"}:
        return var * xp.exp(-0.5 * squared / (ell * ell))
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
        return var * (1.0 + scaled + scaled * scaled / 3.0) * xp.exp(-scaled)
    raise ValueError("native FALKON Matérn supports nu in {0.5,1.5,2.5}.")


def _triangular_solve(
    xp: Any,
    matrix: Any,
    rhs: Any,
    *,
    lower: bool,
    transpose: bool = False,
) -> Any:
    """Use SciPy/CuPy triangular solve with a portable dense fallback."""

    backend_name = _array_module_name(xp)
    try:
        if backend_name == "cupy":
            solver = importlib.import_module("cupyx.scipy.linalg").solve_triangular
        elif backend_name == "numpy":
            solver = importlib.import_module("scipy.linalg").solve_triangular
        else:
            raise ImportError
        return solver(
            matrix,
            rhs,
            lower=bool(lower),
            trans=1 if transpose else 0,
            check_finite=False,
        )
    except (ImportError, AttributeError):
        system = matrix.T if transpose else matrix
        return xp.linalg.solve(system, rhs)


def _device_scalar(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def fit_native_falkon_krr(
    x_train: Any,
    y_train: Any,
    cfg: NativeFalkonKRRConfig,
    *,
    array_module: Any | None = None,
    timer: Callable[[], float] = time.perf_counter,
    _deadline: float | None = None,
) -> tuple[NativeFalkonModel, dict[str, Any]]:
    """Fit restricted KRR with the native two-Cholesky FALKON solver.

    If ``C = K(X, centers)`` and ``W = K(centers, centers)``, the target
    system is exactly

    ``(C.T C + absolute_ridge W) alpha = C.T y``.

    The implementation works with the system divided by ``n`` and therefore
    sets ``penalty = absolute_ridge / n``.  Its FALKON preconditioner is
    ``P = T^-1 A^-1``, where ``T.T T`` approximates ``W`` and
    ``A.T A = T T.T / M + penalty I``.  CG solves ``P.T H P beta = P.T b``;
    every multiplication by ``C.T C`` is accumulated over row chunks.
    """

    _validate_native_falkon_config(cfg)
    deadline = (
        _deadline
        if _deadline is not None
        else (
            None
            if cfg.time_budget_seconds is None
            else timer() + float(cfg.time_budget_seconds)
        )
    )
    _check_deadline(deadline, timer, stage="native FALKON setup")
    n_train, dim = _validate_xy_shapes(x_train, y_train, prefix="train")
    xp = _resolve_backend(cfg.backend, array_module)
    dtype = _xp_dtype(xp, cfg.precision)
    backend_name = _array_module_name(xp)
    center_count = min(int(cfg.nystrom_centers), n_train)
    penalty = falkon_penalty_from_absolute_ridge(cfg.absolute_ridge, n_train)

    _sync_array_module(xp)
    setup_start = timer()
    rng = np.random.default_rng(int(cfg.seed))
    center_indices = np.sort(
        rng.choice(n_train, size=center_count, replace=False).astype(np.int64)
    )
    centers = xp.asarray(x_train[center_indices], dtype=dtype)
    _sync_array_module(xp)
    setup_seconds = float(timer() - setup_start)

    _sync_array_module(xp)
    build_start = timer()
    kernel_kwargs = {
        "family": cfg.kernel_family,
        "lengthscale": float(cfg.lengthscale),
        "nu": float(cfg.nu),
        "variance": float(cfg.kernel_variance),
        "dtype": dtype,
    }
    center_kernel = _native_kernel_cross(xp, centers, centers, **kernel_kwargs)
    center_kernel = 0.5 * (center_kernel + center_kernel.T)
    if cfg.preconditioner_jitter is None:
        # Match the official FALKON defaults (pc_epsilon_64/32).
        jitter = 1e-13 if _precision_dtypes(cfg.precision)[1] == "float64" else 1e-5
    else:
        jitter = float(cfg.preconditioner_jitter)
    preconditioner_kernel = center_kernel.copy()
    diagonal = xp.arange(center_count)
    preconditioner_kernel[diagonal, diagonal] += jitter * center_count

    # NumPy/CuPy Cholesky returns L with L L.T = matrix.  FALKON's notation
    # uses the corresponding upper factors T=L.T and A=L_A.T.
    t_factor = xp.linalg.cholesky(preconditioner_kernel).T
    a_matrix = (t_factor @ t_factor.T) / center_count
    a_matrix[diagonal, diagonal] += penalty
    a_factor = xp.linalg.cholesky(0.5 * (a_matrix + a_matrix.T)).T

    def inv_t(vector: Any) -> Any:
        return _triangular_solve(xp, t_factor, vector, lower=False, transpose=False)

    def inv_tt(vector: Any) -> Any:
        return _triangular_solve(xp, t_factor, vector, lower=False, transpose=True)

    def inv_a(vector: Any) -> Any:
        return _triangular_solve(xp, a_factor, vector, lower=False, transpose=False)

    def inv_at(vector: Any) -> Any:
        return _triangular_solve(xp, a_factor, vector, lower=False, transpose=True)

    rhs_unpreconditioned = xp.zeros(center_count, dtype=dtype)
    chunk_size = int(cfg.train_chunk_size)
    chunks_per_pass = 0
    for start in range(0, n_train, chunk_size):
        _check_deadline(deadline, timer, stage="native FALKON rhs accumulation")
        stop = min(n_train, start + chunk_size)
        kernel_block = _native_kernel_cross(
            xp, x_train[start:stop], centers, **kernel_kwargs
        )
        targets = xp.asarray(y_train[start:stop], dtype=dtype).reshape(-1)
        rhs_unpreconditioned += kernel_block.T @ targets
        chunks_per_pass += 1
    rhs_unpreconditioned /= n_train
    rhs = inv_at(inv_tt(rhs_unpreconditioned))
    _sync_array_module(xp)
    solver_build_seconds = float(timer() - build_start)

    matvec_passes = 0

    def streamed_normal_matvec(vector: Any) -> Any:
        nonlocal matvec_passes
        out = xp.zeros(center_count, dtype=dtype)
        for start in range(0, n_train, chunk_size):
            _check_deadline(deadline, timer, stage="native FALKON streamed matvec")
            stop = min(n_train, start + chunk_size)
            kernel_block = _native_kernel_cross(
                xp, x_train[start:stop], centers, **kernel_kwargs
            )
            out += kernel_block.T @ (kernel_block @ vector)
        matvec_passes += 1
        return out / n_train

    def preconditioned_operator(beta: Any) -> Any:
        coefficients = inv_t(inv_a(beta))
        # Use the unjittered W here.  Jitter stabilizes only P; it does not
        # perturb the restricted KRR system being solved.
        system_product = streamed_normal_matvec(coefficients)
        system_product += penalty * (center_kernel @ coefficients)
        return inv_at(inv_tt(system_product))

    def vector_norm(vector: Any) -> float:
        squared_norm = _device_scalar(xp.real(xp.vdot(vector, vector)))
        return math.sqrt(max(0.0, squared_norm))

    rhs_norm = vector_norm(rhs_unpreconditioned)

    def original_relative_residual(preconditioned_residual: Any) -> float:
        if rhs_norm == 0.0:
            return 0.0 if vector_norm(preconditioned_residual) == 0.0 else math.inf
        original_residual = t_factor.T @ (a_factor.T @ preconditioned_residual)
        return vector_norm(original_residual) / rhs_norm

    _sync_array_module(xp)
    solve_start = timer()
    beta = xp.zeros(center_count, dtype=dtype)
    residual = rhs.copy()
    direction = residual.copy()
    residual_inner = _device_scalar(xp.real(xp.vdot(residual, residual)))
    initial_preconditioned_norm = math.sqrt(max(0.0, residual_inner))
    relative_residual = original_relative_residual(residual)
    iterations = 0
    converged = relative_residual <= float(cfg.tolerance)
    while iterations < int(cfg.maxiter) and not converged:
        _check_deadline(deadline, timer, stage="native FALKON CG")
        operator_direction = preconditioned_operator(direction)
        denominator = _device_scalar(xp.real(xp.vdot(direction, operator_direction)))
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError(
                "native FALKON CG encountered a non-positive search curvature"
            )
        step = residual_inner / denominator
        beta += step * direction
        residual -= step * operator_direction
        new_inner = _device_scalar(xp.real(xp.vdot(residual, residual)))
        iterations += 1
        relative_residual = original_relative_residual(residual)
        converged = relative_residual <= float(cfg.tolerance)
        if converged:
            residual_inner = new_inner
            break
        if residual_inner <= 0.0 or not math.isfinite(new_inner):
            raise RuntimeError("native FALKON CG residual became non-finite")
        direction = residual + (new_inner / residual_inner) * direction
        residual_inner = new_inner
    coefficients = inv_t(inv_a(beta))
    _sync_array_module(xp)
    iterative_solve_seconds = float(timer() - solve_start)
    final_preconditioned_norm = math.sqrt(max(0.0, residual_inner))
    preconditioned_relative_residual = (
        final_preconditioned_norm / initial_preconditioned_norm
        if initial_preconditioned_norm > 0.0
        else 0.0
    )

    model = NativeFalkonModel(
        centers=centers,
        coefficients=coefficients,
        input_dim=dim,
        array_module=xp,
        backend_name=backend_name,
        precision=str(cfg.precision),
        kernel_family=str(cfg.kernel_family),
        lengthscale=float(cfg.lengthscale),
        nu=float(cfg.nu),
        kernel_variance=float(cfg.kernel_variance),
    )
    diagnostics = {
        "setup_seconds": setup_seconds,
        "solver_build_seconds": solver_build_seconds,
        "iterative_solve_seconds": iterative_solve_seconds,
        "train_total_seconds": setup_seconds
        + solver_build_seconds
        + iterative_solve_seconds,
        "iterations": iterations,
        "relative_residual": relative_residual,
        "preconditioned_relative_residual": preconditioned_relative_residual,
        "converged": converged,
        "matvec_passes": matvec_passes,
        "chunks_per_data_pass": chunks_per_pass,
        "streamed_kernel_chunks": chunks_per_pass * (1 + matvec_passes),
        "preconditioner_jitter": jitter,
        "backend": backend_name,
        "falkon_penalty": penalty,
    }
    return model, diagnostics


def score_native_falkon_krr(
    model: NativeFalkonModel,
    x_test: Any,
    y_test: Any,
    *,
    prediction_chunk_size: int,
    timer: Callable[[], float] = time.perf_counter,
    _deadline: float | None = None,
) -> dict[str, Any]:
    """Stream native FALKON prediction and regression metrics."""

    n_test, dim = _validate_xy_shapes(x_test, y_test, prefix="test")
    if dim != int(model.input_dim):
        raise ValueError("training and test feature counts differ.")
    chunk_size = int(prediction_chunk_size)
    if chunk_size <= 0:
        raise ValueError("prediction_chunk_size must be positive.")
    xp = model.array_module
    dtype = _xp_dtype(xp, model.precision)
    kernel_kwargs = {
        "family": model.kernel_family,
        "lengthscale": model.lengthscale,
        "nu": model.nu,
        "variance": model.kernel_variance,
        "dtype": dtype,
    }
    squared_error = xp.asarray(0.0, dtype=xp.float64)
    absolute_error = xp.asarray(0.0, dtype=xp.float64)
    target_sum = xp.asarray(0.0, dtype=xp.float64)
    target_square_sum = xp.asarray(0.0, dtype=xp.float64)
    prediction_chunks = 0
    _sync_array_module(xp)
    prediction_start = timer()
    for start in range(0, n_test, chunk_size):
        _check_deadline(_deadline, timer, stage="native FALKON prediction")
        stop = min(n_test, start + chunk_size)
        kernel_block = _native_kernel_cross(
            xp, x_test[start:stop], model.centers, **kernel_kwargs
        )
        prediction = kernel_block @ model.coefficients
        truth = xp.asarray(y_test[start:stop], dtype=dtype).reshape(-1)
        residual = prediction - truth
        squared_error += xp.sum(residual * residual, dtype=xp.float64)
        absolute_error += xp.sum(xp.abs(residual), dtype=xp.float64)
        target_sum += xp.sum(truth, dtype=xp.float64)
        target_square_sum += xp.sum(truth * truth, dtype=xp.float64)
        prediction_chunks += 1
    _sync_array_module(xp)
    prediction_seconds = float(timer() - prediction_start)
    metrics = _regression_metrics_from_sums(
        n_rows=n_test,
        squared_error=_device_scalar(squared_error),
        absolute_error=_device_scalar(absolute_error),
        target_sum=_device_scalar(target_sum),
        target_square_sum=_device_scalar(target_square_sum),
    )
    return {
        **metrics,
        "prediction_seconds": prediction_seconds,
        "prediction_chunks": prediction_chunks,
    }


def run_native_falkon_krr(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    cfg: NativeFalkonKRRConfig,
    *,
    array_module: Any | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Fit and score the explicitly labeled native FALKON algorithm."""

    n_train, dim = _validate_xy_shapes(x_train, y_train, prefix="train")
    n_test, test_dim = _validate_xy_shapes(x_test, y_test, prefix="test")
    if dim != test_dim:
        raise ValueError("training and test feature counts differ.")
    _validate_time_budget(cfg.time_budget_seconds)
    deadline = (
        None
        if cfg.time_budget_seconds is None
        else timer() + float(cfg.time_budget_seconds)
    )
    model, fit_diagnostics = fit_native_falkon_krr(
        x_train,
        y_train,
        cfg,
        array_module=array_module,
        timer=timer,
        _deadline=deadline,
    )
    score = score_native_falkon_krr(
        model,
        x_test,
        y_test,
        prediction_chunk_size=int(cfg.prediction_chunk_size),
        timer=timer,
        _deadline=deadline,
    )
    canonical_family = (
        "matern"
        if str(cfg.kernel_family).strip().lower() in {"matern", "mat", "mat32"}
        else "squared_exponential"
    )
    return {
        "status": "converged" if bool(fit_diagnostics["converged"]) else "maxiter",
        "method": "native-falkon-krr",
        "implementation": "native_falkon_algorithm",
        "official_falkon_package": False,
        "pipeline_family": "literature_data_space_krr",
        "citations": list(_FALKON_CITATIONS),
        "n_train": n_train,
        "n_test": n_test,
        "input_dim": dim,
        "kernel_family": canonical_family,
        "lengthscale": float(cfg.lengthscale),
        "nu": float(cfg.nu) if canonical_family == "matern" else None,
        "kernel_variance": float(cfg.kernel_variance),
        "absolute_ridge": float(cfg.absolute_ridge),
        "regularization_convention": "absolute",
        "falkon_penalty": falkon_penalty_from_absolute_ridge(
            cfg.absolute_ridge, n_train
        ),
        "falkon_penalty_identity": "penalty=absolute_ridge/n_train",
        "nystrom_centers": min(int(cfg.nystrom_centers), n_train),
        "requested_nystrom_centers": int(cfg.nystrom_centers),
        "maxiter": int(cfg.maxiter),
        "tolerance": float(cfg.tolerance),
        "seed": int(cfg.seed),
        "precision": str(cfg.precision),
        **fit_diagnostics,
        **score,
        "timing_scope": (
            "setup selects and stages centers; solver_build constructs the "
            "two-Cholesky FALKON preconditioner and streamed RHS; "
            "iterative_solve times streamed preconditioned CG; prediction is separate"
        ),
        "config": asdict(cfg),
    }


def matern_rff_features(
    x: Any,
    frequencies: Any,
    phases: Any,
    *,
    kernel_variance: float,
    array_module: Any = np,
    dtype: Any | None = None,
) -> Any:
    """Construct one row chunk of random phase Fourier features."""

    xp = array_module
    feature_dtype = dtype if dtype is not None else frequencies.dtype
    xx = xp.asarray(x, dtype=feature_dtype)
    if xx.ndim != 2:
        raise ValueError("x must be two-dimensional.")
    if int(xx.shape[1]) != int(frequencies.shape[1]):
        raise ValueError("x and frequencies have different feature counts.")
    angles = xx @ frequencies.T
    angles += phases.reshape(1, -1)
    xp.cos(angles, out=angles)
    scale = math.sqrt(2.0 * float(kernel_variance) / int(frequencies.shape[0]))
    angles *= scale
    return angles


def fit_matern_rff_ridge(
    x_train: Any,
    y_train: Any,
    cfg: MaternRFFRidgeConfig,
    *,
    array_module: Any | None = None,
    timer: Callable[[], float] = time.perf_counter,
    _deadline: float | None = None,
) -> tuple[MaternRFFModel, dict[str, Any]]:
    """Fit RFF ridge while retaining only ``D x D`` sufficient statistics."""

    _validate_rff_config(cfg)
    deadline = (
        _deadline
        if _deadline is not None
        else (
            None
            if cfg.time_budget_seconds is None
            else timer() + float(cfg.time_budget_seconds)
        )
    )
    _check_deadline(deadline, timer, stage="Matérn RFF setup")
    n_train, dim = _validate_xy_shapes(x_train, y_train, prefix="train")
    xp = _resolve_array_module(cfg, array_module)
    dtype = _xp_dtype(xp, cfg.precision)
    backend_name = _array_module_name(xp)

    _sync_array_module(xp)
    setup_start = timer()
    frequencies_np, phases_np = sample_matern_rff_parameters(dim, cfg)
    frequencies = xp.asarray(frequencies_np, dtype=dtype)
    phases = xp.asarray(phases_np, dtype=dtype)
    count = int(cfg.num_features)
    gram = xp.zeros((count, count), dtype=dtype)
    rhs = xp.zeros(count, dtype=dtype)
    _sync_array_module(xp)
    setup_seconds = float(timer() - setup_start)

    _sync_array_module(xp)
    accumulation_start = timer()
    train_chunks = 0
    chunk_size = int(cfg.train_chunk_size)
    for start in range(0, n_train, chunk_size):
        _check_deadline(deadline, timer, stage="Matérn RFF accumulation")
        stop = min(n_train, start + chunk_size)
        features = matern_rff_features(
            x_train[start:stop],
            frequencies,
            phases,
            kernel_variance=float(cfg.kernel_variance),
            array_module=xp,
            dtype=dtype,
        )
        targets = xp.asarray(y_train[start:stop], dtype=dtype).reshape(-1)
        gram += features.T @ features
        rhs += features.T @ targets
        train_chunks += 1
    _sync_array_module(xp)
    accumulation_seconds = float(timer() - accumulation_start)

    _check_deadline(deadline, timer, stage="Matérn RFF dense solve")
    solve_start = timer()
    diagonal = xp.arange(count)
    gram[diagonal, diagonal] += float(cfg.absolute_ridge)
    coefficients = xp.linalg.solve(gram, rhs)
    _sync_array_module(xp)
    solve_seconds = float(timer() - solve_start)

    model = MaternRFFModel(
        frequencies=frequencies,
        phases=phases,
        coefficients=coefficients,
        feature_scale=math.sqrt(
            2.0 * float(cfg.kernel_variance) / int(cfg.num_features)
        ),
        input_dim=dim,
        array_module=xp,
        backend_name=backend_name,
        precision=str(cfg.precision),
    )
    diagnostics = {
        "setup_seconds": setup_seconds,
        "feature_accumulation_seconds": accumulation_seconds,
        "solve_seconds": solve_seconds,
        "train_total_seconds": setup_seconds + accumulation_seconds + solve_seconds,
        "train_chunks": train_chunks,
        "backend": backend_name,
    }
    return model, diagnostics


def score_matern_rff_ridge(
    model: MaternRFFModel,
    x_test: Any,
    y_test: Any,
    *,
    prediction_chunk_size: int,
    kernel_variance: float,
    timer: Callable[[], float] = time.perf_counter,
    _deadline: float | None = None,
) -> dict[str, Any]:
    """Compute RMSE without materializing all RFF predictions or features."""

    n_test, dim = _validate_xy_shapes(x_test, y_test, prefix="test")
    if dim != int(model.input_dim):
        raise ValueError("training and test feature counts differ.")
    chunk_size = int(prediction_chunk_size)
    if chunk_size <= 0:
        raise ValueError("prediction_chunk_size must be positive.")
    xp = model.array_module
    dtype = _xp_dtype(xp, model.precision)
    squared_error = xp.asarray(0.0, dtype=xp.float64)
    absolute_error = xp.asarray(0.0, dtype=xp.float64)
    target_sum = xp.asarray(0.0, dtype=xp.float64)
    target_square_sum = xp.asarray(0.0, dtype=xp.float64)
    prediction_chunks = 0
    _sync_array_module(xp)
    prediction_start = timer()
    for start in range(0, n_test, chunk_size):
        _check_deadline(_deadline, timer, stage="Matérn RFF prediction")
        stop = min(n_test, start + chunk_size)
        features = matern_rff_features(
            x_test[start:stop],
            model.frequencies,
            model.phases,
            kernel_variance=kernel_variance,
            array_module=xp,
            dtype=dtype,
        )
        prediction = features @ model.coefficients
        truth = xp.asarray(y_test[start:stop], dtype=dtype).reshape(-1)
        residual = prediction - truth
        squared_error += xp.sum(residual * residual, dtype=xp.float64)
        absolute_error += xp.sum(xp.abs(residual), dtype=xp.float64)
        target_sum += xp.sum(truth, dtype=xp.float64)
        target_square_sum += xp.sum(truth * truth, dtype=xp.float64)
        prediction_chunks += 1
    _sync_array_module(xp)
    prediction_seconds = float(timer() - prediction_start)

    def scalar(value: Any) -> float:
        return float(value.item()) if hasattr(value, "item") else float(value)

    metrics = _regression_metrics_from_sums(
        n_rows=n_test,
        squared_error=scalar(squared_error),
        absolute_error=scalar(absolute_error),
        target_sum=scalar(target_sum),
        target_square_sum=scalar(target_square_sum),
    )
    return {
        **metrics,
        "prediction_seconds": prediction_seconds,
        "prediction_chunks": prediction_chunks,
    }


def run_matern_rff_ridge(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    cfg: MaternRFFRidgeConfig,
    *,
    array_module: Any | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Fit and score the streaming Rahimi--Recht Matérn RFF baseline."""

    n_train, dim = _validate_xy_shapes(x_train, y_train, prefix="train")
    n_test, test_dim = _validate_xy_shapes(x_test, y_test, prefix="test")
    if dim != test_dim:
        raise ValueError("training and test feature counts differ.")
    _validate_time_budget(cfg.time_budget_seconds)
    deadline = (
        None
        if cfg.time_budget_seconds is None
        else timer() + float(cfg.time_budget_seconds)
    )
    model, fit_diagnostics = fit_matern_rff_ridge(
        x_train,
        y_train,
        cfg,
        array_module=array_module,
        timer=timer,
        _deadline=deadline,
    )
    score = score_matern_rff_ridge(
        model,
        x_test,
        y_test,
        prediction_chunk_size=int(cfg.prediction_chunk_size),
        kernel_variance=float(cfg.kernel_variance),
        timer=timer,
        _deadline=deadline,
    )
    return {
        "status": "ok",
        "method": "matern-rff-ridge",
        "implementation": "native_streaming_rff",
        "pipeline_family": "literature_random_features_krr",
        "citations": list(_RFF_CITATIONS),
        "n_train": n_train,
        "n_test": n_test,
        "input_dim": dim,
        "kernel_family": "matern",
        "lengthscale": float(cfg.lengthscale),
        "nu": float(cfg.nu),
        "kernel_variance": float(cfg.kernel_variance),
        "num_features": int(cfg.num_features),
        "seed": int(cfg.seed),
        "precision": str(cfg.precision),
        "absolute_ridge": float(cfg.absolute_ridge),
        "regularization_convention": "absolute",
        "feature_distribution": "multivariate_student_t_df_2nu",
        "feature_system": "Phi.T@Phi + absolute_ridge*I",
        **fit_diagnostics,
        **score,
        "timing_scope": (
            "train_total_seconds includes deterministic feature sampling and "
            "device setup, streamed Phi.T@Phi/Phi.T@y accumulation, and the "
            "D-by-D ridge solve; prediction is separately streamed"
        ),
        "config": asdict(cfg),
    }


__all__ = [
    "FalkonKRRConfig",
    "MaternRFFModel",
    "MaternRFFRidgeConfig",
    "NativeFalkonKRRConfig",
    "NativeFalkonModel",
    "falkon_penalty_from_absolute_ridge",
    "fit_matern_rff_ridge",
    "fit_native_falkon_krr",
    "matern_rff_features",
    "run_falkon_krr",
    "run_matern_rff_ridge",
    "run_native_falkon_krr",
    "sample_matern_rff_parameters",
    "score_matern_rff_ridge",
    "score_native_falkon_krr",
]
