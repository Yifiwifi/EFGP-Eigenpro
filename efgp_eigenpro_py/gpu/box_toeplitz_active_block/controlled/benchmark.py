from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ....efgp_solver import EFGPSolver
from ....kernels import make_matern, make_squared_exponential
from ...benchmark_dataset.stored_npz import (
    StoredNpzError,
    inspect_stored_npy_member,
    load_stored_npz_prefix,
)
from ...backends import BackendConfig, build_gpu_backend_bundle
from ...contexts import GPUDataContext, GPUOperatorContext, ensure_gpu_data_context
from ...deflation_core import make_jacobi_precond
from ...iterative_solvers import cg_solve_gpu, pcg_solve_gpu
from ...v1_ops import apply_A_block_v1, apply_A_v1, gpu_precompute_v1
from ..active_set import BoxActiveSet, build_box_active_set
from ..box_eigenpro import (
    apply_box_eigenpro_preconditioner,
    build_box_eigenpro_preconditioner,
)
from ..config import BTABConfig
from ..diagnostics import run_btab_post_diagnostics
from ..preconditioner import (
    _diag_A_gpu,
    _gamma_from_xtxcol,
    apply_box_toeplitz_preconditioner,
    build_box_toeplitz_preconditioner,
)
from .randomized_nystrom import (
    apply_randomized_nystrom_preconditioner,
    build_randomized_nystrom_preconditioner,
)
from .randomized_pivoted_cholesky import (
    apply_randomized_pivoted_cholesky_preconditioner,
    build_randomized_pivoted_cholesky_preconditioner,
    make_weighted_toeplitz_column_accessor,
)


_HERE = Path(__file__).resolve().parent
_PROCESSED_DIR = _HERE.parents[1] / "benchmark_dataset" / "processed"
_VALID_METHODS = {
    "cg",
    "jacobi",
    "default",
    "active-inverse",
    "full-inverse",
    "active-eig",
    "full-eig",
    "fourier-nystrom-precond",
    "fourier-rpcholesky-precond",
}
_LEGACY_AMBIGUOUS_METHODS = {"nystrom", "rpcholesky"}
_SOLVER_TOTAL_DEFINITION = (
    "score selection + preconditioner construction + CG/PCG solve"
)


# These are exactly the ControlledConfig fields that can change the arrays
# defining the Fourier system.  Method, timing, diagnostic, and active-box
# settings are deliberately absent so a suite can compare them on one A,b.
_SYSTEM_CONFIG_FIELDS = (
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

_SYSTEM_ARTIFACT_SCHEMA_VERSION = 1
TIMING_SYSTEM_ARTIFACT_FILENAME = "timing_system_arrays.npz"
TIMING_SOLUTIONS_ARTIFACT_FILENAME = "timing_prediction_solutions.npz"
TIMING_SOLUTIONS_MANIFEST_FILENAME = "timing_prediction_solutions.json"


@dataclass(frozen=True)
class ControlledConfig:
    dataset_stem: str = "MUR_JPL_SST_North_Atlantic_20230101_n100000"
    dataset_dir: str = ""
    n_train: int | None = 20_000
    subset_seed: int = 0
    subset_mode: str = "random"
    kernel_family: str = "matern"
    lengthscale: float = 0.1
    nu: float = 1.5
    variance: float = 1.0
    reg_lambda: float = 0.1
    fourier_eps: float = 1e-3
    nufft_tol: float = 1e-10
    l2_scaled: bool = True
    tol: float = 1e-7
    maxiter: int = 6000
    zero_initial_vector: bool = True
    precision: str = "fp64"
    allow_near_epsilon_tol: bool = False
    methods: tuple[str, ...] = (
        "cg",
        "jacobi",
        "default",
        "full-eig",
    )
    score_tau: float = 1.0
    box_budget: int = 1024
    inverse_max_size: int = 1024
    rank: int = 32
    nystrom_rank: int = 32
    rpcholesky_rank: int = 32
    eig_tol: float = 1e-3
    eig_maxiter: int | None = None
    measured_repeats: int = 5
    warmup_repeats: int = 1
    method_order_seed: int = 20260823
    eig_seed: int = 0
    nystrom_seed: int = 17
    rpcholesky_seed: int = 23
    nufft_backend: str = "auto"
    precompute_chunk_size: int | None = None
    post_diagnostic_mode: str = "none"
    diagnostic_tol: float = 1e-2
    diagnostic_power_iter: int = 30
    diagnostic_topk: tuple[int, ...] = ()
    strict_gpu_eig: bool = False
    output_dir: str = ""


@dataclass
class PreparedSystem:
    backend: Any
    data_ctx: Any
    rhs_gpu: Any
    reg_lambda: float
    setup_seconds: float
    system_id: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ResolvedBoxRule:
    config: BTABConfig
    active: BoxActiveSet
    raw_tau_box_size: int
    requested_rank: int
    effective_rank: int
    selection_rule: str
    selection_seconds: float = 0.0


@dataclass(frozen=True)
class MethodSpec:
    label: str
    kind: str
    btab_config: BTABConfig | None = None
    rank: int | None = None
    selection_rule: str = ""
    result_role: str = ""
    selection_seconds: float = 0.0


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _sanitize_json(value.tolist())
    return value


def _asnumpy(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


def _sync_device(backend: Any) -> None:
    cuda = getattr(backend.xp, "cuda", None)
    if cuda is not None:
        cuda.runtime.deviceSynchronize()


_RUNTIME_MANIFEST_FIELDS = (
    "device_name",
    "device_id",
    "supports_fp64",
    "supports_complex128",
    "cupy_version",
    "cuda_runtime_version",
    "cuda_driver_version",
    "compute_capability",
    "nufft_backend_resolved",
)


def _backend_runtime_manifest(backend: Any) -> dict[str, Any]:
    """Describe the backend that will perform the current method timings."""
    xp = backend.xp
    runtime_version = None
    driver_version = None
    compute_capability = None
    try:
        runtime_version = int(xp.cuda.runtime.runtimeGetVersion())
        driver_version = int(xp.cuda.runtime.driverGetVersion())
        props = xp.cuda.runtime.getDeviceProperties(int(backend.device_id))
        major = (
            props.get("major")
            if isinstance(props, dict)
            else getattr(props, "major", None)
        )
        minor = (
            props.get("minor")
            if isinstance(props, dict)
            else getattr(props, "minor", None)
        )
        if major is not None and minor is not None:
            compute_capability = f"{int(major)}.{int(minor)}"
    except Exception:
        pass
    manifest = {
        "device_name": str(getattr(backend, "device_name", "unknown")),
        "device_id": int(getattr(backend, "device_id", 0)),
        "supports_fp64": bool(getattr(backend, "supports_fp64", True)),
        "supports_complex128": bool(getattr(backend, "supports_complex128", True)),
        "cupy_version": str(getattr(xp, "__version__", "unknown")),
        "cuda_runtime_version": runtime_version,
        "cuda_driver_version": driver_version,
        "compute_capability": compute_capability,
        "nufft_backend_resolved": str(getattr(backend, "nufft_name", "unknown")),
    }
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    manifest["timing_runtime_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def probe_timing_runtime(nufft_backend: str) -> dict[str, Any]:
    """Resolve the current GPU/NUFFT runtime without constructing a data system."""
    _ensure_writable_runtime_temp()
    backend = build_gpu_backend_bundle(BackendConfig(nufft=str(nufft_backend)))
    return _backend_runtime_manifest(backend)


def _runtime_subset(manifest: dict[str, Any]) -> dict[str, Any]:
    subset = {field: manifest.get(field) for field in _RUNTIME_MANIFEST_FIELDS}
    if manifest.get("timing_runtime_sha256"):
        subset["timing_runtime_sha256"] = manifest["timing_runtime_sha256"]
    return subset


def _ensure_writable_runtime_temp() -> str:
    """Give CUDA compilers a known writable temp directory.

    Windows security layers can allow a simple probe in the user temp folder
    while denying the named temporary files created by NVRTC.  A workspace-local
    directory avoids the resulting 10,000-name retry loop.  Users may override it
    with ``BTAB_RUNTIME_TMP``.
    """
    configured = os.environ.get("BTAB_RUNTIME_TMP", "").strip()
    runtime_dir = (
        Path(configured).expanduser().resolve()
        if configured
        else (_HERE.parents[3] / "tmp" / "controlled_runtime_temp").resolve()
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cupy_cache = runtime_dir / "cupy_kernel_cache"
    cupy_cache.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(runtime_dir)
    os.environ["TEMP"] = str(runtime_dir)
    os.environ["TMP"] = str(runtime_dir)
    os.environ.setdefault("CUPY_CACHE_DIR", str(cupy_cache))
    return str(runtime_dir)


def _timed(backend: Any, operation: Callable[[], Any]) -> tuple[Any, float]:
    _sync_device(backend)
    start = time.perf_counter_ns()
    result = operation()
    _sync_device(backend)
    elapsed = (time.perf_counter_ns() - start) * 1e-9
    return result, float(elapsed)


def _hash_array(hasher: Any, name: str, value: Any) -> None:
    arr = np.ascontiguousarray(_asnumpy(value))
    hasher.update(name.encode("utf-8"))
    hasher.update(str(arr.dtype).encode("ascii"))
    hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    hasher.update(arr.view(np.uint8))


def system_fingerprint(
    data_ctx: Any,
    reg_lambda: float,
    *,
    solve_rhs_gpu: Any | None = None,
) -> str:
    """Hash exactly the arrays used by the fixed Fourier solve.

    ``data_ctx.rhs_gpu`` is the precompute/storage RHS.  In mixed precision the
    solver consumes a cast copy, so callers that own a :class:`PreparedSystem`
    must pass its ``rhs_gpu`` as ``solve_rhs_gpu``.  Keeping the optional
    fallback preserves the small standalone helpers that operate directly on a
    data context while ensuring the formal runner hashes the actual ``b``.
    """
    solve_rhs = data_ctx.rhs_gpu if solve_rhs_gpu is None else solve_rhs_gpu
    hasher = hashlib.sha256()
    _hash_array(hasher, "weights", data_ctx.weights_gpu_flat)
    _hash_array(hasher, "gf", data_ctx.gf_gpu)
    _hash_array(hasher, "rhs", solve_rhs)
    hasher.update(np.asarray([float(reg_lambda)], dtype=np.float64).tobytes())
    return hasher.hexdigest()


def system_component_fingerprints(
    data_ctx: Any,
    *,
    solve_rhs_gpu: Any | None = None,
) -> dict[str, str]:
    """Hash fixed-system components, including the RHS actually solved."""
    solve_rhs = data_ctx.rhs_gpu if solve_rhs_gpu is None else solve_rhs_gpu
    components = {
        "weights_sha256": ("weights", data_ctx.weights_gpu_flat),
        "gf_sha256": ("gf", data_ctx.gf_gpu),
        "rhs_sha256": ("rhs", solve_rhs),
        "rhs_storage_sha256": ("rhs_storage", data_ctx.rhs_gpu),
    }
    fingerprints: dict[str, str] = {}
    for field, (name, value) in components.items():
        hasher = hashlib.sha256()
        _hash_array(hasher, name, value)
        fingerprints[field] = hasher.hexdigest()
    return fingerprints


def system_config_payload(cfg: ControlledConfig) -> dict[str, Any]:
    """Return the canonical config subset that determines the Fourier system."""
    payload: dict[str, Any] = {}
    for field_name in _SYSTEM_CONFIG_FIELDS:
        value = getattr(cfg, field_name)
        if field_name == "n_train":
            value = _normalize_n_train(value)
        elif field_name == "precompute_chunk_size":
            value = None if value is None else int(value)
        elif field_name in {"subset_seed"}:
            value = int(value)
        elif field_name in {
            "lengthscale",
            "nu",
            "variance",
            "reg_lambda",
            "fourier_eps",
            "nufft_tol",
        }:
            value = float(value)
        elif field_name == "l2_scaled":
            value = bool(value)
        elif field_name in {
            "subset_mode",
            "kernel_family",
            "precision",
            "nufft_backend",
        }:
            value = value.strip().lower()
        payload[field_name] = value
    return payload


def system_config_fingerprint(cfg: ControlledConfig) -> str:
    """Hash the canonical system-building config, excluding method settings."""
    encoded = json.dumps(
        system_config_payload(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_artifact_fingerprint(name: str, value: Any) -> str:
    hasher = hashlib.sha256()
    _hash_array(hasher, name, value)
    return hasher.hexdigest()


def _artifact_array_descriptor(name: str, value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": [int(size) for size in array.shape],
        "sha256": _array_artifact_fingerprint(name, array),
    }


def _system_artifact_arrays(system: PreparedSystem) -> dict[str, np.ndarray]:
    """Copy the frozen solve/prediction context to exact host arrays."""
    ctx = system.data_ctx
    arrays = {
        "weights_flat": np.ascontiguousarray(_asnumpy(ctx.weights_gpu_flat)),
        "weights_np_flat": np.ascontiguousarray(
            np.asarray(ctx.weights_np_flat, dtype=np.float64).reshape(-1)
        ),
        "gf": np.ascontiguousarray(_asnumpy(ctx.gf_gpu)),
        "rhs_storage": np.ascontiguousarray(_asnumpy(ctx.rhs_gpu)),
        "rhs_solve": np.ascontiguousarray(_asnumpy(system.rhs_gpu)),
    }
    for name, value in (
        ("xtxcol", getattr(ctx, "xtxcol_gpu", None)),
        ("x_center", getattr(ctx, "x_center_gpu", None)),
    ):
        if value is not None:
            arrays[name] = np.ascontiguousarray(_asnumpy(value))
    return arrays


def save_prepared_system_artifact(
    system: PreparedSystem,
    cfg: ControlledConfig,
    path: str | Path,
) -> Path:
    """Atomically persist one exact PreparedSystem as a portable NPZ artifact.

    The artifact stores the byte-exact weights, Gf, storage/solve rhs, spatial
    Toeplitz column, prediction center, and context metadata.  Large training
    arrays are intentionally omitted because they are not needed after Fourier
    precomputation.
    """
    _validate_prepared_system_for_config(system, cfg)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = _system_artifact_arrays(system)
    component_hashes = system_component_fingerprints(
        system.data_ctx,
        solve_rhs_gpu=system.rhs_gpu,
    )
    clean_system_manifest = {
        key: value
        for key, value in system.manifest.items()
        if not str(key).startswith(("system_artifact_", "timing_solution_"))
    }
    artifact_manifest = {
        "schema_version": _SYSTEM_ARTIFACT_SCHEMA_VERSION,
        "system_id": system.system_id,
        **component_hashes,
        "system_config": system_config_payload(cfg),
        "system_config_sha256": system_config_fingerprint(cfg),
        "reg_lambda": float(system.reg_lambda),
        "setup_seconds": float(system.setup_seconds),
        "source_bundle_sha256": system.manifest.get("source_bundle_sha256"),
        "dataset_content_index_sha256": system.manifest.get(
            "dataset_content_index_sha256"
        ),
        "dataset_metadata_sha256": system.manifest.get("dataset_metadata_sha256"),
        "data_context_meta": _sanitize_json(dict(system.data_ctx.meta)),
        "system_manifest": _sanitize_json(clean_system_manifest),
        "arrays": {
            name: _artifact_array_descriptor(name, value)
            for name, value in arrays.items()
        },
    }
    metadata_bytes = json.dumps(
        _sanitize_json(artifact_manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    payload = dict(arrays)
    payload["artifact_manifest_json"] = np.frombuffer(metadata_bytes, dtype=np.uint8)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(target)
    artifact_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    loaded_from_artifact = bool(
        system.manifest.get("prepared_system_loaded_from_artifact", False)
    )
    system.manifest.update(
        {
            "system_artifact_schema_version": _SYSTEM_ARTIFACT_SCHEMA_VERSION,
            "system_artifact_path": str(target),
            "system_artifact_sha256": artifact_sha256,
            "system_artifact_export_path": str(target),
            "system_artifact_export_sha256": artifact_sha256,
            # Kept for compatibility, but now describes the origin of the
            # PreparedSystem rather than the most recent export operation.
            "system_artifact_loaded": loaded_from_artifact,
            "prepared_system_loaded_from_artifact": loaded_from_artifact,
        }
    )
    return target


def _load_system_artifact_payload(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if "artifact_manifest_json" not in loaded.files:
                raise ValueError("artifact_manifest_json is missing")
            metadata_bytes = np.asarray(
                loaded["artifact_manifest_json"], dtype=np.uint8
            ).tobytes()
            manifest = json.loads(metadata_bytes.decode("utf-8"))
            descriptors = manifest.get("arrays", {})
            if not isinstance(descriptors, dict) or not descriptors:
                raise ValueError("array descriptors are missing")
            arrays: dict[str, np.ndarray] = {}
            for name, descriptor in descriptors.items():
                if name not in loaded.files:
                    raise ValueError(f"artifact array {name!r} is missing")
                array = np.ascontiguousarray(loaded[name])
                if str(array.dtype) != str(descriptor.get("dtype")):
                    raise ValueError(f"artifact array {name!r} dtype changed")
                if [int(size) for size in array.shape] != list(
                    descriptor.get("shape", [])
                ):
                    raise ValueError(f"artifact array {name!r} shape changed")
                if _array_artifact_fingerprint(name, array) != descriptor.get("sha256"):
                    raise ValueError(f"artifact array {name!r} checksum changed")
                arrays[name] = array
    except (
        OSError,
        ValueError,
        KeyError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        raise ValueError(f"invalid prepared-system artifact {path}: {exc}") from exc
    return manifest, arrays


def load_prepared_system_artifact(
    cfg: ControlledConfig,
    path: str | Path,
    *,
    expected_source_sha256: str | None = None,
    expected_dataset_content_index_sha256: str | None = None,
    expected_dataset_metadata_sha256: str | None = None,
) -> PreparedSystem:
    """Restore a PreparedSystem and reject any provenance/config/hash mismatch."""
    # Historical timing/prediction artifacts can still be audited under the
    # labels they were written with.  New runs reject those ambiguous labels.
    _validate_config(cfg, allow_legacy_method_names=True)
    artifact_path = Path(path).expanduser().resolve()
    artifact_manifest, arrays = _load_system_artifact_payload(artifact_path)
    if int(artifact_manifest.get("schema_version", -1)) != int(
        _SYSTEM_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported prepared-system artifact schema "
            f"{artifact_manifest.get('schema_version')!r}"
        )
    expected_config_sha256 = system_config_fingerprint(cfg)
    if artifact_manifest.get("system_config_sha256") != expected_config_sha256:
        raise ValueError("prepared-system artifact config does not match this case")
    for field, expected in (
        ("source_bundle_sha256", expected_source_sha256),
        (
            "dataset_content_index_sha256",
            expected_dataset_content_index_sha256,
        ),
        ("dataset_metadata_sha256", expected_dataset_metadata_sha256),
    ):
        if expected is not None and artifact_manifest.get(field) != expected:
            raise ValueError(
                f"prepared-system artifact {field}={artifact_manifest.get(field)!r}, "
                f"expected {expected!r}"
            )
    required_arrays = {
        "weights_flat",
        "weights_np_flat",
        "gf",
        "rhs_storage",
        "rhs_solve",
    }
    missing = sorted(required_arrays - set(arrays))
    if missing:
        raise ValueError(
            f"prepared-system artifact is missing required arrays {missing}"
        )

    backend = build_gpu_backend_bundle(BackendConfig(nufft=str(cfg.nufft_backend)))
    xp = backend.xp
    data_meta = dict(artifact_manifest.get("data_context_meta", {}))
    dim = int(data_meta.get("dim", 0))
    weight_shape = tuple(int(size) for size in data_meta.get("weight_shape", ()))
    if not weight_shape or int(np.prod(weight_shape)) != int(
        arrays["weights_flat"].size
    ):
        raise ValueError("prepared-system artifact has an invalid weight_shape")
    weights_flat_gpu = xp.ascontiguousarray(xp.asarray(arrays["weights_flat"]))
    data_ctx = GPUDataContext(
        x_gpu=xp.empty((0, dim), dtype=xp.float64),
        y_gpu=xp.empty((0,), dtype=xp.float64),
        weights_gpu_nd=weights_flat_gpu.reshape(weight_shape),
        weights_gpu_flat=weights_flat_gpu,
        weights_np_flat=np.ascontiguousarray(arrays["weights_np_flat"]),
        rhs_gpu=xp.ascontiguousarray(xp.asarray(arrays["rhs_storage"])),
        gf_gpu=xp.ascontiguousarray(xp.asarray(arrays["gf"])),
        xtxcol_gpu=(
            xp.ascontiguousarray(xp.asarray(arrays["xtxcol"]))
            if "xtxcol" in arrays
            else None
        ),
        x_center_gpu=(
            xp.ascontiguousarray(xp.asarray(arrays["x_center"]))
            if "x_center" in arrays
            else None
        ),
        meta=data_meta,
    )
    reg_lambda = float(artifact_manifest["reg_lambda"])
    restored_rhs_gpu = xp.ascontiguousarray(xp.asarray(arrays["rhs_solve"]))
    restored_system_id = system_fingerprint(
        data_ctx,
        reg_lambda,
        solve_rhs_gpu=restored_rhs_gpu,
    )
    if restored_system_id != artifact_manifest.get("system_id"):
        raise ValueError(
            "prepared-system artifact arrays do not reproduce the recorded system_id"
        )
    nested_manifest = artifact_manifest.get("system_manifest")
    if not isinstance(nested_manifest, dict):
        raise ValueError("prepared-system artifact has no valid nested system manifest")
    manifest = dict(nested_manifest)
    for field in (
        "source_bundle_sha256",
        "dataset_content_index_sha256",
        "dataset_metadata_sha256",
    ):
        if manifest.get(field) != artifact_manifest.get(field):
            raise ValueError(
                f"prepared-system nested manifest {field} differs from its "
                "artifact provenance envelope"
            )
    actual_components = system_component_fingerprints(
        data_ctx,
        solve_rhs_gpu=restored_rhs_gpu,
    )
    if manifest.get("system_id") != restored_system_id:
        raise ValueError("prepared-system nested manifest system_id is inconsistent")
    if manifest.get("system_config_sha256") != expected_config_sha256:
        raise ValueError("prepared-system nested manifest config hash is inconsistent")
    if (
        manifest.get("reg_lambda") is None
        or float(manifest["reg_lambda"]) != reg_lambda
    ):
        raise ValueError(
            "prepared-system nested manifest regularization is inconsistent"
        )
    for field, actual in actual_components.items():
        if manifest.get(field) != actual:
            raise ValueError(
                f"prepared-system nested manifest {field} is missing or inconsistent"
            )
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    build_runtime = manifest.get("system_build_runtime")
    if not isinstance(build_runtime, dict):
        build_runtime = _runtime_subset(manifest)
    current_runtime = _backend_runtime_manifest(backend)
    manifest.update(
        {
            "system_id": restored_system_id,
            "system_config": system_config_payload(cfg),
            "system_config_sha256": expected_config_sha256,
            "system_artifact_schema_version": _SYSTEM_ARTIFACT_SCHEMA_VERSION,
            "system_artifact_path": str(artifact_path),
            "system_artifact_sha256": artifact_sha256,
            "system_artifact_loaded": True,
            "prepared_system_loaded_from_artifact": True,
            "prepared_system_origin_artifact_path": str(artifact_path),
            "prepared_system_origin_artifact_sha256": artifact_sha256,
            "system_build_runtime": build_runtime,
            "current_timing_runtime": current_runtime,
            "setup_timing_source": "reused_from_prepared_system_artifact",
            "setup_inclusive_timing_eligible": False,
            **current_runtime,
        }
    )
    system = PreparedSystem(
        backend=backend,
        data_ctx=data_ctx,
        rhs_gpu=restored_rhs_gpu,
        reg_lambda=reg_lambda,
        setup_seconds=float(artifact_manifest["setup_seconds"]),
        system_id=restored_system_id,
        manifest=manifest,
    )
    _validate_prepared_system_for_config(
        system,
        cfg,
        allow_legacy_method_names=True,
    )
    return system


def _validate_prepared_system_for_config(
    system: PreparedSystem,
    cfg: ControlledConfig,
    *,
    allow_legacy_method_names: bool = False,
) -> None:
    """Fail closed unless a supplied PreparedSystem is exact for this config."""
    _validate_config(
        cfg,
        allow_legacy_method_names=allow_legacy_method_names,
    )
    expected_config_sha256 = system_config_fingerprint(cfg)
    recorded_config_sha256 = system.manifest.get("system_config_sha256")
    if recorded_config_sha256 != expected_config_sha256:
        raise ValueError(
            "PreparedSystem lacks the exact dataset/kernel/Fourier system config hash "
            "required by this case"
        )
    if float(system.reg_lambda) != float(cfg.reg_lambda):
        raise ValueError("PreparedSystem regularization does not match the case config")
    if system.manifest.get("reg_lambda") is None or float(
        system.manifest["reg_lambda"]
    ) != float(system.reg_lambda):
        raise ValueError(
            "PreparedSystem manifest regularization is missing or inconsistent"
        )
    actual_system_id = system_fingerprint(
        system.data_ctx,
        float(system.reg_lambda),
        solve_rhs_gpu=system.rhs_gpu,
    )
    if (
        actual_system_id != system.system_id
        or system.manifest.get("system_id") != system.system_id
    ):
        raise ValueError(
            "PreparedSystem weights/Gf/rhs no longer match its recorded system_id"
        )
    actual_components = system_component_fingerprints(
        system.data_ctx,
        solve_rhs_gpu=system.rhs_gpu,
    )
    mismatched_components = {
        field: {
            "recorded": system.manifest.get(field),
            "actual": actual,
        }
        for field, actual in actual_components.items()
        if system.manifest.get(field) != actual
    }
    if mismatched_components:
        raise ValueError(
            "PreparedSystem component hashes are missing or inconsistent: "
            f"{mismatched_components}"
        )


def _box_fingerprint(active: BoxActiveSet) -> str:
    arr = np.ascontiguousarray(np.asarray(active.box_idx, dtype=np.int64))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def _git_revision() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_HERE),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _source_manifest() -> dict[str, Any]:
    repo_root = _HERE.parents[3]
    paths = [
        Path(__file__).resolve(),
        (_HERE / "randomized_nystrom.py").resolve(),
        (_HERE / "randomized_pivoted_cholesky.py").resolve(),
        (_HERE / "suite.py").resolve(),
        (_HERE / "sweep.py").resolve(),
        (_HERE.parent / "active_set.py").resolve(),
        (_HERE.parent / "box_eigenpro.py").resolve(),
        (_HERE.parent / "config.py").resolve(),
        (_HERE.parent / "diagnostics.py").resolve(),
        (_HERE.parent / "preconditioner.py").resolve(),
        (_HERE.parents[1] / "backends.py").resolve(),
        (_HERE.parents[1] / "contexts.py").resolve(),
        (_HERE.parents[1] / "deflation_core.py").resolve(),
        (_HERE.parents[1] / "iterative_solvers.py").resolve(),
        (_HERE.parents[1] / "v1_ops.py").resolve(),
        (_HERE.parents[2] / "efgp_solver.py").resolve(),
        (_HERE.parents[2] / "kernels.py").resolve(),
    ]
    combined = hashlib.sha256()
    hashes: dict[str, str] = {}
    relative_paths: list[str] = []
    for path in paths:
        relative = str(path.relative_to(repo_root)).replace("\\", "/")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative] = digest
        relative_paths.append(relative)
        combined.update(relative.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    status_lines: list[str] = []
    try:
        proc = subprocess.run(
            [
                "git",
                "status",
                "--short",
                "--untracked-files=all",
                "--",
                *relative_paths,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        status_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        status_lines = ["git status unavailable"]
    return {
        "source_bundle_sha256": combined.hexdigest(),
        "source_files_sha256": hashes,
        "source_git_status": status_lines,
        "source_worktree_clean": not status_lines,
    }


def _normalize_n_train(n_train: int | None) -> int | None:
    if n_train is None or int(n_train) == 0:
        return None
    if int(n_train) < 0:
        raise ValueError(
            "n_train must be positive, zero, or None; zero means all rows."
        )
    return int(n_train)


def _resolve_dataset_dir(dataset_dir: str = "") -> Path:
    requested = (
        str(dataset_dir).strip() or os.environ.get("BTAB_PROCESSED_DIR", "").strip()
    )
    directory = (
        Path(requested).expanduser().resolve()
        if requested
        else _PROCESSED_DIR.resolve()
    )
    if not directory.is_dir():
        raise FileNotFoundError(
            f"processed dataset directory was not found: {directory}"
        )
    return directory


def _npz_content_index_sha256(path: str | Path) -> str:
    """Hash ZIP member names, sizes, and content CRCs without rereading large arrays."""
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        entries = [
            {
                "name": info.filename,
                "file_size": int(info.file_size),
                "compress_size": int(info.compress_size),
                "crc32": f"{int(info.CRC):08x}",
            }
            for info in sorted(archive.infolist(), key=lambda item: item.filename)
            if not info.is_dir()
        ]
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_dataset(
    stem: str,
    n_train: int | None,
    subset_seed: int,
    dataset_dir: str = "",
    subset_mode: str = "random",
) -> dict[str, Any]:
    processed_dir = _resolve_dataset_dir(dataset_dir)
    path = processed_dir / f"{stem}.npz"
    if not path.exists():
        available = ", ".join(p.stem for p in sorted(processed_dir.glob("*.npz")))
        raise FileNotFoundError(
            f"processed dataset {stem!r} was not found; available: {available}"
        )
    meta_path = path.with_suffix(".json")
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    requested_n = _normalize_n_train(n_train)
    mode = str(subset_mode).strip().lower()
    if mode not in {"random", "prefix"}:
        raise ValueError("subset_mode must be 'random' or 'prefix'.")

    if mode == "prefix":
        try:
            x_info = inspect_stored_npy_member(path, "x_train")
            y_info = inspect_stored_npy_member(path, "y_train")
            if not x_info.shape or not y_info.shape:
                raise StoredNpzError("x_train and y_train must have a row dimension")
            source_n = int(x_info.shape[0])
            if int(y_info.shape[0]) != source_n:
                raise ValueError("x_train and y_train row counts differ")
            rows = source_n if requested_n is None else int(requested_n)
            if rows > source_n:
                raise ValueError(
                    f"requested n_train={rows} exceeds the {source_n} rows in {stem!r}; "
                    "use an exact larger processed stem rather than duplicating observations."
                )
            x = load_stored_npz_prefix(path, "x_train", rows, dtype=np.float64)
            y = load_stored_npz_prefix(path, "y_train", rows, dtype=np.float64).reshape(
                -1
            )
        except StoredNpzError:
            # Compressed legacy artifacts cannot be memory-mapped.  Preserve the
            # exact prefix semantics, but fall back to NumPy's ordinary loader.
            with np.load(path) as loaded:
                x_full = np.asarray(loaded["x_train"], dtype=np.float64)
                y_full = np.asarray(loaded["y_train"], dtype=np.float64).reshape(-1)
            source_n = int(x_full.shape[0])
            rows = source_n if requested_n is None else int(requested_n)
            if rows > source_n:
                raise ValueError(
                    f"requested n_train={rows} exceeds the {source_n} rows in {stem!r}; "
                    "use an exact larger processed stem rather than duplicating observations."
                )
            x = np.ascontiguousarray(x_full[:rows])
            y = np.ascontiguousarray(y_full[:rows])
    else:
        with np.load(path) as loaded:
            x = np.asarray(loaded["x_train"], dtype=np.float64)
            y = np.asarray(loaded["y_train"], dtype=np.float64).reshape(-1)
        source_n = int(x.shape[0])
    if requested_n is not None and requested_n > source_n:
        raise ValueError(
            f"requested n_train={requested_n} exceeds the {source_n} rows in {stem!r}; "
            "use an exact larger processed stem rather than duplicating observations."
        )
    if mode == "random" and requested_n is not None and requested_n < source_n:
        rng = np.random.default_rng(int(subset_seed))
        idx = np.sort(rng.choice(source_n, size=requested_n, replace=False))
        x = np.ascontiguousarray(x[idx])
        y = np.ascontiguousarray(y[idx])
    return {
        "stem": stem,
        "path": str(path),
        "metadata": meta,
        "source_n_train": source_n,
        "subset_mode": mode,
        "file_size_bytes": int(path.stat().st_size),
        "content_index_sha256": _npz_content_index_sha256(path),
        "metadata_sha256": (
            hashlib.sha256(meta_path.read_bytes()).hexdigest()
            if meta_path.exists()
            else None
        ),
        "x": x,
        "y": y,
    }


def _make_kernel(cfg: ControlledConfig, dim: int) -> Any:
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


def _precision_dtypes(xp: Any, precision: str) -> tuple[Any, Any, float]:
    key = str(precision).strip().lower()
    if key == "fp64":
        return xp.complex128, xp.float64, float(np.finfo(np.float64).eps)
    if key == "mixed32":
        return xp.complex64, xp.float32, float(np.finfo(np.float32).eps)
    raise ValueError("precision must be 'fp64' or 'mixed32'.")


def _validate_config(
    cfg: ControlledConfig,
    *,
    allow_legacy_method_names: bool = False,
) -> None:
    _normalize_n_train(cfg.n_train)
    if str(cfg.subset_mode).strip().lower() not in {"random", "prefix"}:
        raise ValueError("subset_mode must be 'random' or 'prefix'.")
    allowed_methods = set(_VALID_METHODS)
    if allow_legacy_method_names:
        allowed_methods.update(_LEGACY_AMBIGUOUS_METHODS)
    ambiguous = [
        method for method in cfg.methods if method in _LEGACY_AMBIGUOUS_METHODS
    ]
    if ambiguous and not allow_legacy_method_names:
        replacements = {
            "nystrom": "fourier-nystrom-precond",
            "rpcholesky": "fourier-rpcholesky-precond",
        }
        suggestions = ", ".join(
            f"{method!r} -> {replacements[method]!r}"
            for method in dict.fromkeys(ambiguous)
        )
        raise ValueError(
            "ambiguous fixed-system method name(s) are not accepted for new runs: "
            f"{ambiguous}. Use the explicit Fourier-preconditioner names "
            f"({suggestions}); data-space KRR pipelines belong in the end-to-end "
            "KRR comparison."
        )
    unknown = [method for method in cfg.methods if method not in allowed_methods]
    if unknown:
        raise ValueError(
            f"unknown methods: {unknown}; choices are {sorted(_VALID_METHODS)}"
        )
    if "cg" not in cfg.methods:
        raise ValueError("controlled comparisons require method 'cg' as the reference.")
    if int(cfg.measured_repeats) < 5:
        raise ValueError(
            "measured_repeats must be at least five for the controlled experiment."
        )
    if int(cfg.warmup_repeats) < 0:
        raise ValueError("warmup_repeats must be nonnegative.")
    if float(cfg.tol) <= 0.0 or int(cfg.maxiter) <= 0:
        raise ValueError("tol and maxiter must be positive.")
    if cfg.zero_initial_vector is not True:
        raise ValueError(
            "controlled comparisons require zero_initial_vector=true for every method."
        )
    if float(cfg.reg_lambda) <= 0.0:
        raise ValueError("reg_lambda must be positive.")
    if any(
        int(value) <= 0 for value in (cfg.rank, cfg.nystrom_rank, cfg.rpcholesky_rank)
    ):
        raise ValueError("rank, nystrom_rank, and rpcholesky_rank must be positive.")
    if cfg.precompute_chunk_size is not None and int(cfg.precompute_chunk_size) <= 0:
        raise ValueError("precompute_chunk_size must be positive or None.")
    if int(cfg.box_budget) <= 0 or int(cfg.inverse_max_size) <= 0:
        raise ValueError("box_budget and inverse_max_size must be positive.")
    if any(int(value) <= 0 for value in cfg.diagnostic_topk):
        raise ValueError("diagnostic_topk values must be positive.")
    _, _, machine_eps = _precision_dtypes(np, cfg.precision)
    if float(cfg.tol) < 100.0 * machine_eps:
        message = (
            f"tol={cfg.tol:.3e} is below 100 machine eps for {cfg.precision} "
            f"(eps={machine_eps:.3e}); recursive residuals may not be reliable."
        )
        if not bool(cfg.allow_near_epsilon_tol):
            raise ValueError(
                message + " Use fp64 or explicitly set allow_near_epsilon_tol."
            )
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def prepare_shared_system(
    cfg: ControlledConfig,
    *,
    dataset_payload: dict[str, Any] | None = None,
) -> PreparedSystem:
    """Construct one Fourier system once and return the shared GPU context."""
    _validate_config(cfg)
    runtime_temp = _ensure_writable_runtime_temp()
    dataset = (
        dataset_payload
        if dataset_payload is not None
        else _load_dataset(
            cfg.dataset_stem,
            cfg.n_train,
            cfg.subset_seed,
            cfg.dataset_dir,
            cfg.subset_mode,
        )
    )
    if str(dataset.get("stem")) != str(cfg.dataset_stem):
        raise ValueError(
            "preloaded dataset stem does not match the controlled configuration: "
            f"{dataset.get('stem')!r} != {cfg.dataset_stem!r}"
        )
    x = dataset["x"]
    y = dataset["y"]
    dim = int(x.shape[1])
    kernel = _make_kernel(cfg, dim)
    solver = EFGPSolver(
        kernel,
        reg_lambda=float(cfg.reg_lambda),
        eps=float(cfg.fourier_eps),
        nufft_tol=float(cfg.nufft_tol),
        l2scaled=bool(cfg.l2_scaled),
    )
    backend = build_gpu_backend_bundle(BackendConfig(nufft=str(cfg.nufft_backend)))
    data_ctx = ensure_gpu_data_context(backend, x, y, state=None)
    data_ctx.meta["debug_finite_checks"] = False
    setup_ctx = GPUOperatorContext()

    def _setup() -> Any:
        return gpu_precompute_v1(
            backend,
            solver.kernel,
            solver.eps,
            solver.nufft_tol,
            data_ctx,
            setup_ctx,
            l2scaled=solver.l2scaled,
            chunk_size=cfg.precompute_chunk_size,
        )

    data_ctx, setup_seconds = _timed(backend, _setup)
    solve_dtype, real_dtype, machine_eps = _precision_dtypes(backend.xp, cfg.precision)
    if str(cfg.precision).lower() == "mixed32":
        data_ctx.meta["complex_dtype"] = "complex64"
    rhs_gpu = backend.xp.asarray(data_ctx.rhs_gpu, dtype=solve_dtype)
    system_id = system_fingerprint(
        data_ctx,
        float(cfg.reg_lambda),
        solve_rhs_gpu=rhs_gpu,
    )
    component_fingerprints = system_component_fingerprints(
        data_ctx,
        solve_rhs_gpu=rhs_gpu,
    )

    runtime_manifest = _backend_runtime_manifest(backend)

    manifest = {
        "system_id": system_id,
        **component_fingerprints,
        "system_config": system_config_payload(cfg),
        "system_config_sha256": system_config_fingerprint(cfg),
        "dataset_stem": cfg.dataset_stem,
        "dataset_dir": str(Path(dataset["path"]).parent),
        "dataset_path": dataset["path"],
        "dataset_file_size_bytes": int(dataset["file_size_bytes"]),
        "dataset_content_index_sha256": dataset["content_index_sha256"],
        "dataset_metadata_sha256": dataset["metadata_sha256"],
        "dataset_task_type": dataset["metadata"].get("task_type"),
        "dataset_target_definition": dataset["metadata"].get("target_definition"),
        "dataset_generation": dataset["metadata"].get("generation"),
        "source_n_train": int(dataset["source_n_train"]),
        "n_train": int(x.shape[0]),
        "dim": dim,
        "subset_seed": int(cfg.subset_seed),
        "subset_mode": str(cfg.subset_mode).strip().lower(),
        "kernel_family": str(cfg.kernel_family),
        "kernel_lengthscale": float(cfg.lengthscale),
        "kernel_nu": float(cfg.nu),
        "kernel_variance": float(cfg.variance),
        "reg_lambda": float(cfg.reg_lambda),
        "fourier_eps": float(cfg.fourier_eps),
        "nufft_tol": float(cfg.nufft_tol),
        "precompute_chunk_size": (
            int(cfg.precompute_chunk_size)
            if cfg.precompute_chunk_size is not None
            else None
        ),
        "l2_scaled": bool(cfg.l2_scaled),
        "mtot": int(data_ctx.meta["mtot"]),
        "M": int(data_ctx.meta["mtot"]) ** dim,
        "h": float(data_ctx.meta["h"]),
        "setup_seconds": float(setup_seconds),
        "nufft_backend_requested": str(cfg.nufft_backend),
        "nufft_backend_resolved": str(backend.nufft_name),
        "nufft_stage": data_ctx.meta.get("nufft_stage"),
        "precision_mode": str(cfg.precision),
        "x_host_dtype": str(x.dtype),
        "y_host_dtype": str(y.dtype),
        "weights_dtype": str(data_ctx.weights_gpu_flat.dtype),
        "gf_dtype": str(data_ctx.gf_gpu.dtype),
        "rhs_storage_dtype": str(data_ctx.rhs_gpu.dtype),
        "rhs_solve_dtype": str(rhs_gpu.dtype),
        "matvec_requested_dtype": str(backend.xp.dtype(solve_dtype)),
        "real_component_dtype": str(backend.xp.dtype(real_dtype)),
        "machine_epsilon": float(machine_eps),
        "tolerance": float(cfg.tol),
        "tol_over_machine_epsilon": float(cfg.tol / machine_eps),
        **runtime_manifest,
        "system_build_runtime": runtime_manifest,
        "current_timing_runtime": runtime_manifest,
        "prepared_system_loaded_from_artifact": False,
        "setup_timing_source": "measured_in_current_process",
        "setup_inclusive_timing_eligible": True,
        "git_revision": _git_revision(),
        **_source_manifest(),
        "runtime_temp_dir": runtime_temp,
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "fixed_system_statement": (
            "Every timed method shares these exact weights, Gf, rhs, and regularization arrays."
        ),
    }
    return PreparedSystem(
        backend=backend,
        data_ctx=data_ctx,
        rhs_gpu=rhs_gpu,
        reg_lambda=float(cfg.reg_lambda),
        setup_seconds=float(setup_seconds),
        system_id=system_id,
        manifest=manifest,
    )


def _active_from_cfg(system: PreparedSystem, btab_cfg: BTABConfig) -> BoxActiveSet:
    xp = system.backend.xp
    ctx = system.data_ctx
    if ctx.xtxcol_gpu is None:
        ctx.xtxcol_gpu = xp.ascontiguousarray(system.backend.fft.ifftn(ctx.gf_gpu))
    gamma = _gamma_from_xtxcol(
        xp,
        ctx.xtxcol_gpu,
        int(ctx.meta["mtot"]),
        int(ctx.meta["dim"]),
    )
    return build_box_active_set(
        gamma=float(gamma),
        weights=np.asarray(ctx.weights_np_flat, dtype=np.float64),
        reg_lambda=float(system.reg_lambda),
        mtot=int(ctx.meta["mtot"]),
        dim=int(ctx.meta["dim"]),
        active_mode=str(btab_cfg.active_mode),
        active_topk=btab_cfg.active_topk,
        active_tau=btab_cfg.active_tau,
        box_budget=btab_cfg.box_budget,
    )


def resolve_score_box_rule(
    system: PreparedSystem, cfg: ControlledConfig
) -> ResolvedBoxRule:
    """Apply the predeclared score threshold, then only a memory-cap fallback."""
    M = int(system.manifest["M"])
    base = BTABConfig(
        active_mode="tau",
        active_topk=None,
        active_tau=float(cfg.score_tau),
        box_budget=None,
        solve_mode="exact",
        exact_box_max_size=int(cfg.inverse_max_size),
        exact_apply_mode="inverse",
        eig_q=int(cfg.rank),
        eig_tol=float(cfg.eig_tol),
        eig_maxiter=cfg.eig_maxiter,
        diagnostic_mode="none",
    )
    raw = _active_from_cfg(system, base)
    budget = min(int(cfg.box_budget), M)
    selected_cfg = replace(base, box_budget=budget)
    selection_rule = "score_tau"
    if int(raw.box_idx.size) > budget:
        # Sort once, then find the largest score prefix whose centered
        # enclosing box respects the declared memory cap.  This uses no
        # timing, iterations, rhs, or labels.
        order = np.argsort(np.asarray(raw.rho, dtype=np.float64))[::-1]
        shape = (int(system.manifest["mtot"]),) * int(system.manifest["dim"])
        multi = np.stack(np.unravel_index(order, shape), axis=1).astype(
            np.int64,
            copy=False,
        )
        center = np.asarray(raw.center_multi, dtype=np.int64)
        cumulative_radii = np.maximum.accumulate(
            np.abs(multi - center[None, :]), axis=0
        )
        box_sizes = np.prod(2 * cumulative_radii + 1, axis=1, dtype=np.int64)
        feasible = np.flatnonzero(box_sizes <= budget)
        if feasible.size == 0:
            raise ValueError(
                "box_budget is too small even for the highest-score mode's centered box."
            )
        best_k = int(feasible[-1]) + 1
        selected_cfg = replace(
            base,
            active_mode="topk",
            active_topk=int(best_k),
            active_tau=None,
            box_budget=budget,
        )
        selection_rule = "score_ranked_memory_capped_box"
    active = _active_from_cfg(system, selected_cfg)
    box_size = int(active.box_idx.size)
    q = min(int(cfg.rank), max(box_size - 1, 0))
    selected_cfg = replace(selected_cfg, eig_q=int(q))
    return ResolvedBoxRule(
        config=selected_cfg,
        active=active,
        raw_tau_box_size=int(raw.box_idx.size),
        requested_rank=int(cfg.rank),
        effective_rank=int(q),
        selection_rule=selection_rule,
    )


def resolve_method_specs(
    system: PreparedSystem,
    cfg: ControlledConfig,
) -> tuple[list[MethodSpec], ResolvedBoxRule]:
    rule, selection_seconds = _timed(
        system.backend,
        lambda: resolve_score_box_rule(system, cfg),
    )
    rule = replace(rule, selection_seconds=float(selection_seconds))
    M = int(system.manifest["M"])
    full_rank = min(int(cfg.rank), M - 1)
    specs: list[MethodSpec] = []
    for method in cfg.methods:
        if method in {"cg", "jacobi"}:
            specs.append(MethodSpec(label=method, kind=method, result_role="baseline"))
        elif method == "default":
            if int(rule.active.box_idx.size) == 0:
                specs.append(
                    MethodSpec(
                        label="default",
                        kind="jacobi",
                        selection_rule=rule.selection_rule + "_empty_fallback_jacobi",
                        result_role="deployable_default",
                        selection_seconds=rule.selection_seconds,
                    )
                )
            elif int(rule.active.box_idx.size) <= int(cfg.inverse_max_size):
                specs.append(
                    MethodSpec(
                        label="default",
                        kind="active-inverse",
                        btab_config=replace(
                            rule.config,
                            solve_mode="exact",
                            exact_box_max_size=int(cfg.inverse_max_size),
                            exact_apply_mode="inverse",
                        ),
                        selection_rule=rule.selection_rule + "_inverse_if_fits",
                        result_role="deployable_default",
                        selection_seconds=rule.selection_seconds,
                    )
                )
            else:
                specs.append(
                    MethodSpec(
                        label="default",
                        kind="active-eig",
                        btab_config=replace(rule.config, eig_q=rule.effective_rank),
                        rank=rule.effective_rank,
                        selection_rule=rule.selection_rule + "_eigenpro_if_not",
                        result_role="deployable_default",
                        selection_seconds=rule.selection_seconds,
                    )
                )
        elif method == "active-inverse":
            if int(rule.active.box_idx.size) == 0:
                raise ValueError(
                    "active-inverse requires a nonempty score-selected box."
                )
            if int(rule.active.box_idx.size) > int(cfg.inverse_max_size):
                raise ValueError(
                    "active-inverse was requested but the fixed score box exceeds inverse_max_size; "
                    "increase the declared cap or use method 'default'."
                )
            specs.append(
                MethodSpec(
                    label=method,
                    kind=method,
                    btab_config=replace(
                        rule.config,
                        solve_mode="exact",
                        exact_box_max_size=int(cfg.inverse_max_size),
                        exact_apply_mode="inverse",
                    ),
                    selection_rule=rule.selection_rule,
                    result_role="sensitivity_candidate",
                    selection_seconds=rule.selection_seconds,
                )
            )
        elif method == "full-inverse":
            full_inverse_cfg = BTABConfig(
                active_mode="topk",
                active_topk=M,
                active_tau=None,
                box_budget=M,
                solve_mode="exact",
                exact_box_max_size=M,
                exact_apply_mode="inverse",
                diagnostic_mode="none",
            )
            specs.append(
                MethodSpec(
                    label=method,
                    kind="active-inverse",
                    btab_config=full_inverse_cfg,
                    selection_rule="full_grid",
                    result_role="baseline",
                )
            )
        elif method == "active-eig":
            if int(rule.active.box_idx.size) <= 1:
                raise ValueError(
                    "active-eig requires at least two modes in the score-selected box."
                )
            specs.append(
                MethodSpec(
                    label=method,
                    kind=method,
                    btab_config=replace(rule.config, eig_q=rule.effective_rank),
                    rank=rule.effective_rank,
                    selection_rule=rule.selection_rule,
                    result_role="sensitivity_candidate",
                    selection_seconds=rule.selection_seconds,
                )
            )
        elif method == "full-eig":
            full_cfg = BTABConfig(
                active_mode="topk",
                active_topk=M,
                active_tau=None,
                box_budget=M,
                eig_q=full_rank,
                eig_tol=float(cfg.eig_tol),
                eig_maxiter=cfg.eig_maxiter,
                diagnostic_mode="none",
            )
            specs.append(
                MethodSpec(
                    label=method,
                    kind=method,
                    btab_config=full_cfg,
                    rank=full_rank,
                    selection_rule="full_grid",
                    result_role="baseline",
                )
            )
        elif method == "fourier-nystrom-precond":
            specs.append(
                MethodSpec(
                    label=method,
                    kind=method,
                    rank=min(int(cfg.nystrom_rank), M),
                    selection_rule="fourier_global_gaussian_sketch",
                    result_role="exploratory_fourier_adaptation",
                )
            )
        elif method == "fourier-rpcholesky-precond":
            specs.append(
                MethodSpec(
                    label=method,
                    kind=method,
                    rank=min(int(cfg.rpcholesky_rank), M),
                    selection_rule="fourier_weighted_toeplitz_residual_pivots",
                    result_role="exploratory_fourier_adaptation",
                )
            )
    return specs, rule


def _seed_eigensolver(system: PreparedSystem, seed: int) -> None:
    np.random.seed(int(seed))
    try:
        system.backend.xp.random.seed(int(seed))
    except Exception:
        pass


def _stored_array_bytes(*arrays: Any) -> int:
    """Sum explicitly stored arrays, deduplicating views that share a pointer."""
    total = 0
    seen: set[tuple[int, int]] = set()
    for array in arrays:
        if array is None:
            continue
        nbytes = int(getattr(array, "nbytes", 0))
        if nbytes <= 0:
            continue
        pointer = None
        data = getattr(array, "data", None)
        if data is not None:
            pointer = getattr(data, "ptr", None)
        if pointer is None:
            try:
                pointer = int(np.asarray(array).__array_interface__["data"][0])
            except Exception:
                pointer = id(array)
        key = (int(pointer), nbytes)
        if key in seen:
            continue
        seen.add(key)
        total += nbytes
    return int(total)


def _solve_dtype(system: PreparedSystem, cfg: ControlledConfig) -> Any:
    solve_dtype, _, _ = _precision_dtypes(system.backend.xp, cfg.precision)
    return solve_dtype


def _fresh_operator_context(
    system: PreparedSystem, cfg: ControlledConfig
) -> GPUOperatorContext:
    ctx = GPUOperatorContext()
    ctx.solve_dtype = _solve_dtype(system, cfg)
    return ctx


def _matvec_closure(
    system: PreparedSystem, op_ctx: GPUOperatorContext
) -> Callable[[Any, Any], None]:
    def matvec(v: Any, out: Any) -> None:
        apply_A_v1(
            system.backend,
            system.data_ctx,
            v,
            float(system.reg_lambda),
            op_ctx,
            out=out,
        )

    return matvec


def _build_preconditioner(
    system: PreparedSystem,
    cfg: ControlledConfig,
    spec: MethodSpec,
) -> tuple[Callable[[Any, Any], None] | None, Any | None, dict[str, Any]]:
    backend = system.backend
    xp = backend.xp
    kind = spec.kind
    if kind == "cg":
        return None, None, {}
    if kind == "jacobi":
        if system.data_ctx.xtxcol_gpu is None:
            system.data_ctx.xtxcol_gpu = xp.ascontiguousarray(
                backend.fft.ifftn(system.data_ctx.gf_gpu)
            )
        gamma = _gamma_from_xtxcol(
            xp,
            system.data_ctx.xtxcol_gpu,
            int(system.data_ctx.meta["mtot"]),
            int(system.data_ctx.meta["dim"]),
        )
        diag = _diag_A_gpu(
            xp,
            float(gamma),
            system.data_ctx.weights_gpu_flat,
            float(system.reg_lambda),
        )
        diag_inv = 1.0 / diag
        return (
            make_jacobi_precond(backend, diag_inv),
            diag_inv,
            {
                "gamma": float(gamma),
                "storage_bytes": int(diag_inv.nbytes),
                "preconditioner_storage_bytes": int(diag_inv.nbytes),
                "preconditioner_dtype": str(diag_inv.dtype),
            },
        )
    if kind == "active-inverse":
        if spec.btab_config is None:
            raise RuntimeError("active-inverse requires btab_config")
        pre = build_box_toeplitz_preconditioner(
            backend,
            system.data_ctx,
            float(system.reg_lambda),
            spec.btab_config,
            profile_apply_components=False,
        )

        def apply(v: Any, out: Any) -> None:
            apply_box_toeplitz_preconditioner(backend, pre, v, out=out)

        diag = dict(pre.diagnostics)
        stored_bytes = _stored_array_bytes(
            pre.box_idx_gpu,
            pre.tail_idx_gpu,
            pre.box_inverse_gpu,
            pre.diag_inv_full_gpu,
            pre.diag_inv_tail_gpu,
            pre.diag_inv_box_gpu,
            pre.chol_factor_gpu,
            pre.box_matrix_gpu,
            pre.box_weights_gpu,
            pre.local_gf_gpu,
        )
        diag.update(
            {
                "box_hash": _box_fingerprint(pre.active),
                "preconditioner_dtype": str(pre.box_inverse_gpu.dtype),
                "preconditioner_storage_bytes": stored_bytes,
            }
        )
        return apply, pre, diag
    if kind in {"active-eig", "full-eig"}:
        if spec.btab_config is None or spec.rank is None:
            raise RuntimeError(f"{kind} requires btab_config and rank")
        _seed_eigensolver(system, cfg.eig_seed)
        pre = build_box_eigenpro_preconditioner(
            backend,
            system.data_ctx,
            float(system.reg_lambda),
            spec.btab_config,
            q=int(spec.rank),
            profile_apply_components=False,
        )
        eig_backend = str(pre.diagnostics.get("btab_eig_backend", ""))
        if bool(cfg.strict_gpu_eig) and eig_backend.lower() != "cupy":
            raise RuntimeError(
                f"strict_gpu_eig requested, but eigensolver backend was {eig_backend!r}."
            )

        def apply(v: Any, out: Any) -> None:
            apply_box_eigenpro_preconditioner(backend, pre, v, out=out)

        diag = dict(pre.diagnostics)
        stored_bytes = _stored_array_bytes(
            pre.box_idx_gpu,
            pre.tail_idx_gpu,
            pre.diag_inv_full_gpu,
            pre.diag_inv_tail_gpu,
            pre.diag_inv_box_gpu,
            pre.local_gf_gpu,
            pre.box_weights_gpu,
            pre.eig_U_gpu,
            pre.eig_UH_gpu,
            pre.eig_coeff_gpu,
            pre.eig_theta_top_gpu,
        )
        diag.update(
            {
                "box_hash": _box_fingerprint(pre.active),
                "preconditioner_dtype": str(pre.eig_U_gpu.dtype),
                "preconditioner_coeff_dtype": str(pre.eig_coeff_gpu.dtype),
                "spectral_factor_storage_bytes": int(
                    pre.diagnostics.get("btab_eig_storage_bytes", 0)
                ),
                "preconditioner_storage_bytes": stored_bytes,
            }
        )
        return apply, pre, diag
    if kind == "fourier-nystrom-precond":
        if spec.rank is None:
            raise RuntimeError("fourier-nystrom-precond requires rank")
        block_ctx = _fresh_operator_context(system, cfg)

        def apply_psd_block(V: Any) -> Any:
            AV = apply_A_block_v1(
                backend,
                system.data_ctx,
                V,
                float(system.reg_lambda),
                block_ctx,
            )
            return AV - float(system.reg_lambda) * xp.asarray(V, dtype=AV.dtype)

        pre = build_randomized_nystrom_preconditioner(
            backend,
            apply_psd_block,
            size=int(system.manifest["M"]),
            rank=int(spec.rank),
            reg_lambda=float(system.reg_lambda),
            seed=int(cfg.nystrom_seed),
            dtype=_solve_dtype(system, cfg),
        )

        def apply(v: Any, out: Any) -> None:
            apply_randomized_nystrom_preconditioner(backend, pre, v, out=out)

        diag = dict(pre.diagnostics)
        diag.update(
            {
                "preconditioner_dtype": str(pre.U.dtype),
                "preconditioner_coeff_dtype": str(pre.coeff.dtype),
                "preconditioner_storage_bytes": int(pre.diagnostics["storage_bytes"]),
            }
        )
        return apply, pre, diag
    if kind == "fourier-rpcholesky-precond":
        if spec.rank is None:
            raise RuntimeError("fourier-rpcholesky-precond requires rank")
        if system.data_ctx.xtxcol_gpu is None:
            system.data_ctx.xtxcol_gpu = xp.ascontiguousarray(
                backend.fft.ifftn(system.data_ctx.gf_gpu)
            )
        gamma = _gamma_from_xtxcol(
            xp,
            system.data_ctx.xtxcol_gpu,
            int(system.data_ctx.meta["mtot"]),
            int(system.data_ctx.meta["dim"]),
        )
        diag_A = _diag_A_gpu(
            xp,
            float(gamma),
            system.data_ctx.weights_gpu_flat,
            float(system.reg_lambda),
        )
        psd_diagonal = xp.maximum(
            xp.real(diag_A) - float(system.reg_lambda),
            xp.asarray(0.0, dtype=xp.real(diag_A).dtype),
        )
        mtot = int(system.data_ctx.meta["mtot"])
        dim = int(system.data_ctx.meta["dim"])
        apply_psd_column = make_weighted_toeplitz_column_accessor(
            xp,
            system.data_ctx.xtxcol_gpu,
            system.data_ctx.weights_gpu_flat,
            mtot=mtot,
            dim=dim,
            dtype=_solve_dtype(system, cfg),
        )

        pre = build_randomized_pivoted_cholesky_preconditioner(
            backend,
            apply_psd_column,
            psd_diagonal,
            rank=int(spec.rank),
            reg_lambda=float(system.reg_lambda),
            seed=int(cfg.rpcholesky_seed),
            dtype=_solve_dtype(system, cfg),
            column_access_model="direct_weighted_toeplitz_columns_of_A_minus_lambda_I",
        )

        def apply(v: Any, out: Any) -> None:
            apply_randomized_pivoted_cholesky_preconditioner(backend, pre, v, out=out)

        diag = dict(pre.diagnostics)
        diag.update(
            {
                "rank": int(pre.effective_rank),
                "preconditioner_dtype": str(pre.L.dtype),
                "preconditioner_coeff_dtype": str(pre.middle_inverse.dtype),
                "preconditioner_storage_bytes": int(pre.diagnostics["storage_bytes"]),
            }
        )
        return apply, pre, diag
    raise ValueError(f"unsupported method kind={kind!r}")


def _true_residual(
    system: PreparedSystem,
    cfg: ControlledConfig,
    beta_gpu: Any,
) -> float:
    backend = system.backend
    xp = backend.xp
    audit_dtype = xp.complex128
    check_ctx = GPUOperatorContext()
    check_ctx.solve_dtype = audit_dtype
    rhs = xp.asarray(system.rhs_gpu, dtype=audit_dtype)
    beta_audit = xp.asarray(beta_gpu, dtype=audit_dtype)
    out = xp.empty_like(rhs)
    meta = system.data_ctx.meta
    had_dtype = "complex_dtype" in meta
    previous_dtype = meta.get("complex_dtype")
    try:
        meta["complex_dtype"] = "complex128"
        apply_A_v1(
            backend,
            system.data_ctx,
            beta_audit,
            float(system.reg_lambda),
            check_ctx,
            out=out,
        )
    finally:
        if had_dtype:
            meta["complex_dtype"] = previous_dtype
        else:
            meta.pop("complex_dtype", None)
    _sync_device(backend)
    return float(xp.linalg.norm(rhs - out) / max(float(xp.linalg.norm(rhs)), 1e-300))


def run_one_method(
    system: PreparedSystem,
    cfg: ControlledConfig,
    spec: MethodSpec,
    *,
    repeat_idx: int,
    order_position: int,
    is_warmup: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    backend = system.backend
    op_ctx = _fresh_operator_context(system, cfg)
    setup_inclusive_eligible = bool(
        system.manifest.get("setup_inclusive_timing_eligible", True)
    )
    row: dict[str, Any] = {
        "system_id": system.system_id,
        "method": spec.label,
        "method_kind": spec.kind,
        "selection_rule": spec.selection_rule,
        "result_role": spec.result_role,
        "repeat_idx": int(repeat_idx),
        "order_position": int(order_position),
        "is_warmup": bool(is_warmup),
        "tol": float(cfg.tol),
        "maxiter": int(cfg.maxiter),
        "zero_initial_vector": bool(cfg.zero_initial_vector),
        "precision_mode": str(cfg.precision),
        "solve_dtype": str(backend.xp.dtype(_solve_dtype(system, cfg))),
        "true_residual_audit_dtype": "complex128",
        "setup_inclusive_timing_eligible": setup_inclusive_eligible,
        "setup_timing_source": system.manifest.get("setup_timing_source"),
        "rank": spec.rank,
        "selection_seconds": float(spec.selection_seconds),
        "solver_total_definition": _SOLVER_TOTAL_DEFINITION,
        "status": "running",
    }
    method_stage = "preconditioner_build"
    try:
        if spec.kind == "cg":
            precond, precond_data, build_diag = None, None, {}
            preconditioner_build_seconds = 0.0
        else:
            (precond, precond_data, build_diag), preconditioner_build_seconds = _timed(
                backend,
                lambda: _build_preconditioner(system, cfg, spec),
            )
        selection_seconds = float(spec.selection_seconds)
        build_seconds = float(preconditioner_build_seconds + selection_seconds)
        row.update(build_diag)
        row["selection_seconds"] = selection_seconds
        row["preconditioner_build_seconds"] = float(preconditioner_build_seconds)
        row["build_seconds"] = float(build_seconds)
        matvec = _matvec_closure(system, op_ctx)

        def _solve() -> tuple[Any, int, float, dict[str, Any]]:
            if precond is None:
                return cg_solve_gpu(
                    backend,
                    matvec,
                    system.rhs_gpu,
                    op_ctx,
                    float(cfg.tol),
                    int(cfg.maxiter),
                    return_stats=True,
                    work_prefix=f"controlled_{spec.label.replace('-', '_')}",
                    profile_components=False,
                )
            return pcg_solve_gpu(
                backend,
                matvec,
                precond,
                system.rhs_gpu,
                op_ctx,
                float(cfg.tol),
                int(cfg.maxiter),
                return_stats=True,
                work_prefix=f"controlled_{spec.label.replace('-', '_')}",
                profile_components=False,
            )

        method_stage = "solve"
        (beta_gpu, iterations, recursive_relres, solve_stats), solve_seconds = _timed(
            backend,
            _solve,
        )
        beta_saved = backend.xp.asarray(beta_gpu).copy()
        _sync_device(backend)
        method_stage = "true_residual"
        true_relres = _true_residual(system, cfg, beta_saved)
        beta_host = np.ascontiguousarray(_asnumpy(beta_saved))
        solver_total_seconds = float(build_seconds + solve_seconds)
        row.update(
            {
                "status": str(solve_stats.get("status", "ok")),
                "iterations": int(iterations),
                "recursive_relres": float(recursive_relres),
                "true_relres": float(true_relres),
                "solve_seconds": float(solve_seconds),
                "solver_total_seconds": solver_total_seconds,
                # Backward-compatible alias retained for archived consumers.
                "build_plus_solve_seconds": solver_total_seconds,
                "shared_setup_plus_method_seconds": (
                    float(system.setup_seconds + solver_total_seconds)
                    if setup_inclusive_eligible
                    else math.nan
                ),
                "n_matvec": int(solve_stats.get("n_matvec", -1)),
                "beta_dtype": str(beta_saved.dtype),
                "beta_norm": float(np.linalg.norm(beta_host)),
            }
        )
        del precond_data
        return row, beta_host
    except Exception as exc:
        failure_traceback = traceback.format_exc()
        row.update(
            {
                "status": "error",
                "error_stage": method_stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": failure_traceback,
                "build_seconds": float(row.get("build_seconds", math.nan)),
                "solve_seconds": math.nan,
                "solver_total_seconds": math.nan,
                "build_plus_solve_seconds": math.nan,
                "true_relres": math.nan,
                "recursive_relres": math.nan,
                "iterations": -1,
            }
        )
        solver_breakdown = getattr(exc, "diagnostics", None)
        if isinstance(solver_breakdown, dict):
            row["solver_breakdown"] = dict(solver_breakdown)
        print(
            f"[controlled-method] ERROR method={spec.label!r} "
            f"repeat={repeat_idx} stage={method_stage}: "
            f"{type(exc).__name__}: {exc}\n{failure_traceback}",
            file=sys.stderr,
            flush=True,
        )
        return row, None


def _is_converged(row: dict[str, Any], tol: float) -> bool:
    status = str(row.get("status", "")).lower()
    residual = float(row.get("true_relres", math.nan))
    return (
        status.startswith("converged")
        and math.isfinite(residual)
        and residual <= float(tol)
    )


def _solver_total_seconds(row: dict[str, Any]) -> float:
    """Return the canonical fixed-A,b method total, with legacy-row fallback."""
    for key in ("solver_total_seconds", "build_plus_solve_seconds"):
        try:
            value = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return math.nan


def _finite_values(rows: Iterable[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = (
                _solver_total_seconds(row)
                if key == "solver_total_seconds"
                else float(row.get(key, math.nan))
            )
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = _finite_values(rows, key)
    if values.size == 0:
        return {
            f"{key}_median": math.nan,
            f"{key}_min": math.nan,
            f"{key}_max": math.nan,
        }
    return {
        f"{key}_median": float(np.median(values)),
        f"{key}_min": float(np.min(values)),
        f"{key}_max": float(np.max(values)),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    setup_seconds: float,
    tol: float,
    method_order: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    measured = [row for row in rows if not bool(row.get("is_warmup", False))]
    setup_inclusive_eligible = bool(
        measured
        and all(
            bool(row.get("setup_inclusive_timing_eligible", True)) for row in measured
        )
    )
    cg_by_repeat = {
        int(row["repeat_idx"]): row
        for row in measured
        if row.get("method") == "cg"
        and _is_converged(row, tol)
        and math.isfinite(_solver_total_seconds(row))
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in measured:
        grouped.setdefault(str(row.get("method")), []).append(row)

    summaries: list[dict[str, Any]] = []
    for method, method_rows in grouped.items():
        paired_cold = []
        paired_reuse = []
        paired_pipeline = []
        paired_wins = 0
        for row in method_rows:
            if not _is_converged(row, tol):
                continue
            baseline = cg_by_repeat.get(int(row["repeat_idx"]))
            if baseline is None:
                continue
            cg_total = _solver_total_seconds(baseline)
            cg_solve = float(baseline.get("solve_seconds", math.nan))
            method_solve = float(row.get("solve_seconds", math.nan))
            method_total = _solver_total_seconds(row)
            if all(math.isfinite(v) and v > 0.0 for v in (cg_total, method_total)):
                cold = cg_total / method_total
                paired_cold.append(cold)
                paired_wins += int(cold > 1.0)
                if setup_inclusive_eligible:
                    paired_pipeline.append(
                        (float(setup_seconds) + cg_total)
                        / (float(setup_seconds) + method_total)
                    )
            if all(math.isfinite(v) and v > 0.0 for v in (cg_solve, method_solve)):
                paired_reuse.append(cg_solve / method_solve)
        converged_rows = [row for row in method_rows if _is_converged(row, tol)]
        build_values = _finite_values(converged_rows, "build_seconds")
        solve_values = _finite_values(converged_rows, "solve_seconds")
        cg_values = np.asarray(
            [float(row["solve_seconds"]) for row in cg_by_repeat.values()],
            dtype=np.float64,
        )
        build_median = float(np.median(build_values)) if build_values.size else math.nan
        solve_median = float(np.median(solve_values)) if solve_values.size else math.nan
        cg_median = float(np.median(cg_values)) if cg_values.size else math.nan
        if method == "cg":
            break_even_rhs = 0.0
        elif cg_median > solve_median:
            ratio = build_median / (cg_median - solve_median)
            break_even_rhs = float(math.floor(ratio) + 1)
        else:
            break_even_rhs = math.inf
        first = method_rows[0]
        summary: dict[str, Any] = {
            "method": method,
            "method_kind": first.get("method_kind"),
            "selection_rule": first.get("selection_rule"),
            "result_role": first.get("result_role"),
            "measured_repeats": len(method_rows),
            "converged_repeats": sum(_is_converged(row, tol) for row in method_rows),
            "performance_claim_eligible": bool(
                len(method_rows) > 0
                and all(_is_converged(row, tol) for row in method_rows)
                and len(paired_cold) == len(method_rows)
            ),
            "setup_inclusive_timing_eligible": setup_inclusive_eligible,
            "setup_timing_source": first.get("setup_timing_source"),
            "paired_wins_over_cg": int(paired_wins),
            "paired_comparisons": len(paired_cold),
            "cold_speedup_median": (
                float(np.median(paired_cold)) if paired_cold else math.nan
            ),
            "cold_speedup_min": float(np.min(paired_cold)) if paired_cold else math.nan,
            "cold_speedup_max": float(np.max(paired_cold)) if paired_cold else math.nan,
            "solver_total_speedup_over_cg_median": (
                float(np.median(paired_cold)) if paired_cold else math.nan
            ),
            "solver_total_speedup_over_cg_min": (
                float(np.min(paired_cold)) if paired_cold else math.nan
            ),
            "solver_total_speedup_over_cg_max": (
                float(np.max(paired_cold)) if paired_cold else math.nan
            ),
            "solver_total_definition": _SOLVER_TOTAL_DEFINITION,
            "reuse_speedup_median": (
                float(np.median(paired_reuse)) if paired_reuse else math.nan
            ),
            "shared_fourier_setup_plus_method_speedup_median": (
                float(np.median(paired_pipeline)) if paired_pipeline else math.nan
            ),
            "break_even_rhs": break_even_rhs,
            "rank": first.get("rank"),
            "box_size": first.get("box_size"),
            "box_radii": first.get("box_radii"),
            "box_hash": first.get("box_hash"),
            "preconditioner_storage_bytes": first.get("preconditioner_storage_bytes"),
        }
        for metric in (
            "selection_seconds",
            "preconditioner_build_seconds",
            "build_seconds",
            "solve_seconds",
            "solver_total_seconds",
            "build_plus_solve_seconds",
            "iterations",
            "recursive_relres",
            "true_relres",
            "relative_beta_error_vs_cg",
        ):
            summary.update(_metric_summary(method_rows, metric))
        finite_total_rows = [
            row
            for row in method_rows
            if _is_converged(row, tol)
            and math.isfinite(_solver_total_seconds(row))
            and all(
                math.isfinite(float(row.get(key, math.nan)))
                for key in (
                    "selection_seconds",
                    "preconditioner_build_seconds",
                    "solve_seconds",
                )
            )
        ]
        finite_total_rows.sort(key=_solver_total_seconds)
        if finite_total_rows:
            middle = len(finite_total_rows) // 2
            representatives = (
                [finite_total_rows[middle]]
                if len(finite_total_rows) % 2
                else [finite_total_rows[middle - 1], finite_total_rows[middle]]
            )
            for key in (
                "selection_seconds",
                "preconditioner_build_seconds",
                "solve_seconds",
            ):
                summary[f"{key}_at_median_total"] = float(
                    np.mean([float(row[key]) for row in representatives])
                )
            summary["solver_total_component_repeat_indices"] = [
                int(row["repeat_idx"]) for row in representatives
            ]
        else:
            for key in (
                "selection_seconds",
                "preconditioner_build_seconds",
                "solve_seconds",
            ):
                summary[f"{key}_at_median_total"] = math.nan
            summary["solver_total_component_repeat_indices"] = []
        summaries.append(summary)
    ordered_names = list(method_order) if method_order is not None else list(grouped)
    order = {method: index for index, method in enumerate(ordered_names)}
    summaries.sort(key=lambda row: order[str(row["method"])])
    return summaries


def pairwise_comparisons(
    rows: list[dict[str, Any]],
    tol: float | None = None,
) -> list[dict[str, Any]]:
    """Pair method times by repeat; a speedup above one favors the candidate."""

    def eligible(row: dict[str, Any]) -> bool:
        row_tol = float(tol) if tol is not None else float(row.get("tol", 1e-7))
        return _is_converged(row, row_tol)

    measured = [row for row in rows if not bool(row.get("is_warmup", False))]
    by_method_repeat = {
        (str(row["method"]), int(row["repeat_idx"])): row for row in measured
    }
    methods = list(dict.fromkeys(str(row["method"]) for row in measured))
    pairs = [("cg", method) for method in methods if method != "cg"]
    if "default" in methods and "full-eig" in methods:
        pairs.append(("full-eig", "default"))
    comparisons: list[dict[str, Any]] = []
    for reference, candidate in pairs:
        total_speedups = []
        solve_speedups = []
        reference_builds = []
        candidate_builds = []
        reference_solves = []
        candidate_solves = []
        wins = 0
        available_repeat_ids = sorted(
            repeat
            for method, repeat in by_method_repeat
            if method == reference and (candidate, repeat) in by_method_repeat
        )

        def has_finite_positive_times(method: str, repeat: int) -> bool:
            row = by_method_repeat[(method, repeat)]
            total = _solver_total_seconds(row)
            try:
                solve = float(row.get("solve_seconds", math.nan))
            except (TypeError, ValueError):
                return False
            return all(math.isfinite(value) and value > 0.0 for value in (total, solve))

        repeat_ids = [
            repeat
            for repeat in available_repeat_ids
            if eligible(by_method_repeat[(reference, repeat)])
            and eligible(by_method_repeat[(candidate, repeat)])
            and has_finite_positive_times(reference, repeat)
            and has_finite_positive_times(candidate, repeat)
        ]
        for repeat in repeat_ids:
            ref = by_method_repeat[(reference, repeat)]
            cand = by_method_repeat[(candidate, repeat)]
            ref_total = _solver_total_seconds(ref)
            cand_total = _solver_total_seconds(cand)
            ref_solve = float(ref["solve_seconds"])
            cand_solve = float(cand["solve_seconds"])
            reference_builds.append(float(ref.get("build_seconds", 0.0)))
            candidate_builds.append(float(cand.get("build_seconds", 0.0)))
            reference_solves.append(ref_solve)
            candidate_solves.append(cand_solve)
            total_speedups.append(ref_total / cand_total)
            solve_speedups.append(ref_solve / cand_solve)
            wins += int(cand_total < ref_total)
        crossover = math.nan
        candidate_faster_through_rhs: float = math.nan
        crossover_status = "unavailable"
        if repeat_ids:
            ref_build_median = float(np.median(reference_builds))
            cand_build_median = float(np.median(candidate_builds))
            ref_solve_median = float(np.median(reference_solves))
            cand_solve_median = float(np.median(candidate_solves))
            if (
                cand_build_median < ref_build_median
                and cand_solve_median > ref_solve_median
            ):
                crossover = (ref_build_median - cand_build_median) / (
                    cand_solve_median - ref_solve_median
                )
                candidate_faster_through_rhs = float(
                    max(int(math.ceil(crossover)) - 1, 0)
                )
                crossover_status = "candidate_lower_build_higher_solve"
            elif (
                cand_build_median <= ref_build_median
                and cand_solve_median <= ref_solve_median
            ):
                candidate_faster_through_rhs = math.inf
                crossover_status = "candidate_dominates_build_and_solve"
            elif (
                cand_build_median >= ref_build_median
                and cand_solve_median >= ref_solve_median
            ):
                candidate_faster_through_rhs = 0.0
                crossover_status = "reference_dominates_build_and_solve"
            else:
                crossover = (cand_build_median - ref_build_median) / (
                    ref_solve_median - cand_solve_median
                )
                crossover_status = "candidate_higher_build_lower_solve"
        comparisons.append(
            {
                "reference_method": reference,
                "candidate_method": candidate,
                "available_paired_repeats": len(available_repeat_ids),
                "paired_repeats": len(repeat_ids),
                "candidate_wins": int(wins),
                "performance_claim_eligible": bool(
                    len(available_repeat_ids) > 0
                    and len(repeat_ids) == len(available_repeat_ids)
                ),
                "total_speedup_median": (
                    float(np.median(total_speedups)) if total_speedups else math.nan
                ),
                "total_speedup_min": (
                    float(np.min(total_speedups)) if total_speedups else math.nan
                ),
                "total_speedup_max": (
                    float(np.max(total_speedups)) if total_speedups else math.nan
                ),
                "solver_total_speedup_median": (
                    float(np.median(total_speedups)) if total_speedups else math.nan
                ),
                "solver_total_speedup_min": (
                    float(np.min(total_speedups)) if total_speedups else math.nan
                ),
                "solver_total_speedup_max": (
                    float(np.max(total_speedups)) if total_speedups else math.nan
                ),
                "solve_speedup_median": (
                    float(np.median(solve_speedups)) if solve_speedups else math.nan
                ),
                "cold_to_reuse_crossover_rhs": crossover,
                "candidate_faster_through_rhs": candidate_faster_through_rhs,
                "crossover_status": crossover_status,
                "speedup_definition": (
                    "reference solver_total_seconds / candidate solver_total_seconds; "
                    "values above one favor candidate"
                ),
                "solver_total_definition": _SOLVER_TOTAL_DEFINITION,
            }
        )
    return comparisons


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            cooked = {
                key: (
                    json.dumps(value, ensure_ascii=False, default=_json_default)
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(cooked)


def _post_diagnostics(
    system: PreparedSystem,
    cfg: ControlledConfig,
    specs: list[MethodSpec],
    rule: ResolvedBoxRule,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    mode = str(cfg.post_diagnostic_mode).strip().lower()
    if mode == "none":
        return [], {}
    diagnostics: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    diagnostic_specs = list(specs)
    existing_labels = {spec.label for spec in diagnostic_specs}
    for topk in cfg.diagnostic_topk:
        label = f"diagnostic-inverse-topk-{int(topk)}"
        if label in existing_labels:
            continue
        diagnostic_specs.append(
            MethodSpec(
                label=label,
                kind="active-inverse",
                btab_config=BTABConfig(
                    active_mode="topk",
                    active_topk=int(topk),
                    active_tau=None,
                    box_budget=int(cfg.inverse_max_size),
                    solve_mode="exact",
                    exact_box_max_size=int(cfg.inverse_max_size),
                    exact_apply_mode="inverse",
                    diagnostic_mode="none",
                ),
                selection_rule="nested_score_topk_diagnostic",
                result_role="diagnostic_only",
            )
        )

    for spec in diagnostic_specs:
        if spec.kind not in {"active-inverse", "active-eig", "full-eig"}:
            continue
        record: dict[str, Any] = {
            "system_id": system.system_id,
            "method": spec.label,
            "method_kind": spec.kind,
            "result_role": spec.result_role,
        }
        diagnostic_stage = "preconditioner_build"
        try:
            (apply_preconditioner, pre, build_diag), build_seconds = _timed(
                system.backend,
                lambda spec=spec: _build_preconditioner(system, cfg, spec),
            )
            record.update(
                {
                    "diagnostic_build_seconds": float(build_seconds),
                    **build_diag,
                }
            )
            route = "inverse" if spec.kind == "active-inverse" else "boxeig"
            diagnostic_stage = "operator_diagnostics"
            diag = run_btab_post_diagnostics(
                system.backend,
                system.data_ctx,
                float(system.reg_lambda),
                pre,
                route=route,
                mode=mode,
                tol=float(cfg.diagnostic_tol),
                power_iter=int(cfg.diagnostic_power_iter),
            )
            record.update(diag)
            solve_ctx = _fresh_operator_context(system, cfg)
            matvec = _matvec_closure(system, solve_ctx)

            def _solve_diagnostic() -> tuple[Any, int, float, dict[str, Any]]:
                return pcg_solve_gpu(
                    system.backend,
                    matvec,
                    apply_preconditioner,
                    system.rhs_gpu,
                    solve_ctx,
                    float(cfg.tol),
                    int(cfg.maxiter),
                    return_stats=True,
                    work_prefix=f"diagnostic_{spec.label.replace('-', '_')}",
                    profile_components=False,
                )

            diagnostic_stage = "diagnostic_pcg"
            (
                (beta_diag, diag_iterations, diag_relres, diag_solve_stats),
                diag_solve_seconds,
            ) = _timed(
                system.backend,
                _solve_diagnostic,
            )
            record.update(
                {
                    "diagnostic_pcg_status": str(diag_solve_stats.get("status", "ok")),
                    "diagnostic_pcg_iterations": int(diag_iterations),
                    "diagnostic_pcg_recursive_relres": float(diag_relres),
                    "diagnostic_pcg_solve_seconds": float(diag_solve_seconds),
                }
            )
            diagnostic_stage = "diagnostic_true_residual"
            diag_true_relres = _true_residual(system, cfg, beta_diag)
            record["diagnostic_pcg_true_relres"] = float(diag_true_relres)
            diagnostic_stage = "diagnostic_postprocess"
            delta = float(record.get("epsilon_T", math.nan))
            gamma_key = "eta_inv" if route == "inverse" else "eta_eig"
            gamma = float(record.get(gamma_key, math.nan))
            epsilon_ok = bool(record.get("epsilon_T_norm_stabilized", False))
            gamma_ok = bool(record.get("eta_inv_sq_norm_stabilized", False))
            if route == "inverse" and not (epsilon_ok and gamma_ok):
                record["bound_status"] = "heuristic_norm_estimator_unstabilized"
            elif route == "inverse" and math.isfinite(delta) and math.isfinite(gamma):
                lower = 1.0 - delta - gamma
                upper = 1.0 + delta + gamma
                record["bound_lower"] = lower
                record["bound_upper"] = upper
                record["condition_bound"] = upper / lower if lower > 0.0 else math.nan
                record["bound_status"] = (
                    "estimated_informative_not_certified"
                    if lower > 0.0
                    else "estimated_not_informative"
                )
            elif route == "boxeig":
                record["bound_status"] = "mu_B_minus_not_estimated"

            if spec.kind == "full-eig":
                leverage = _asnumpy(
                    system.backend.xp.sum(
                        system.backend.xp.abs(pre.eig_U_gpu) ** 2,
                        axis=1,
                    )
                ).astype(np.float64, copy=False)
                score_box = np.asarray(rule.active.box_idx, dtype=np.int64)
                capture = float(
                    np.sum(leverage[score_box]) / max(int(pre.eig_U_gpu.shape[1]), 1)
                )
                rho = np.asarray(rule.active.rho, dtype=np.float64)
                record.update(
                    {
                        "score_box_size": int(score_box.size),
                        "score_box_fraction": float(score_box.size / leverage.size),
                        "score_box_leverage_capture": capture,
                        "score_mass_capture": float(
                            np.sum(rho[score_box]) / max(np.sum(rho), 1e-300)
                        ),
                    }
                )
                arrays["dominant_subspace_leverage"] = leverage.reshape(
                    (int(system.manifest["mtot"]),) * int(system.manifest["dim"])
                )
                arrays["score_rho"] = rho.reshape(
                    (int(system.manifest["mtot"]),) * int(system.manifest["dim"])
                )
                arrays["score_box_indices"] = score_box
            record["diagnostic_status"] = "ok"
        except Exception as exc:
            failure_traceback = traceback.format_exc()
            record.update(
                {
                    "diagnostic_status": "error",
                    "diagnostic_error_stage": diagnostic_stage,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": failure_traceback,
                }
            )
            if diagnostic_stage == "diagnostic_pcg":
                record["diagnostic_pcg_status"] = "error"
            solver_breakdown = getattr(exc, "diagnostics", None)
            if isinstance(solver_breakdown, dict):
                record["solver_breakdown"] = dict(solver_breakdown)
            print(
                f"[post-diagnostics] ERROR method={spec.label!r} "
                f"stage={diagnostic_stage}: {type(exc).__name__}: {exc}\n"
                f"{failure_traceback}",
                file=sys.stderr,
                flush=True,
            )
        diagnostics.append(record)
    return diagnostics, arrays


def save_timing_prediction_solutions(
    system: PreparedSystem,
    cfg: ControlledConfig,
    rows: list[dict[str, Any]],
    saved_solutions: list[tuple[dict[str, Any], np.ndarray]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Persist one canonical *measured* beta per method for prediction audits.

    The lowest-index converged measured repeat is selected.  If no repeat
    converged, the lowest-index available measured beta is retained and marked
    ineligible so a downstream audit can fail with evidence instead of silently
    rebuilding or re-solving the system.
    """
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / TIMING_SOLUTIONS_ARTIFACT_FILENAME
    manifest_path = destination / TIMING_SOLUTIONS_MANIFEST_FILENAME

    arrays: dict[str, np.ndarray] = {}
    solution_records: list[dict[str, Any]] = []
    method_order = [str(method) for method in cfg.methods]
    for method_index, method in enumerate(method_order):
        candidates = [
            (row, np.ascontiguousarray(np.asarray(beta)))
            for row, beta in saved_solutions
            if str(row.get("method")) == method and not bool(row.get("is_warmup"))
        ]
        candidates.sort(
            key=lambda item: (
                int(item[0].get("repeat_idx", 10**9)),
                int(item[0].get("order_position", 10**9)),
            )
        )
        converged = [item for item in candidates if _is_converged(item[0], cfg.tol)]
        chosen = converged[0] if converged else (candidates[0] if candidates else None)
        if chosen is None:
            measured_rows = sorted(
                (
                    row
                    for row in rows
                    if str(row.get("method")) == method
                    and not bool(row.get("is_warmup"))
                ),
                key=lambda row: (
                    int(row.get("repeat_idx", 10**9)),
                    int(row.get("order_position", 10**9)),
                ),
            )
            solution_records.append(
                {
                    "method": method,
                    "available": False,
                    "selection_eligible": False,
                    "reason": "no measured beta was returned",
                    "timing_row": (
                        _sanitize_json(measured_rows[0]) if measured_rows else None
                    ),
                }
            )
            continue

        timing_row, beta = chosen
        array_key = f"beta_{method_index:03d}"
        arrays[array_key] = beta
        descriptor = _artifact_array_descriptor(array_key, beta)
        solution_records.append(
            {
                "method": method,
                "method_kind": timing_row.get("method_kind"),
                "available": True,
                "selection_eligible": bool(_is_converged(timing_row, cfg.tol)),
                "array_key": array_key,
                "beta": descriptor,
                "timing_repeat_idx": int(timing_row.get("repeat_idx", -1)),
                "timing_order_position": int(timing_row.get("order_position", -1)),
                "timing_row": _sanitize_json(timing_row),
            }
        )

    temporary = artifact_path.with_name(f".{artifact_path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(artifact_path)
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    component_hashes = system_component_fingerprints(
        system.data_ctx,
        solve_rhs_gpu=system.rhs_gpu,
    )
    payload = {
        "schema_version": 1,
        "artifact_role": (
            "canonical measured timing solutions for out-of-timing prediction only"
        ),
        "system_id": system.system_id,
        **component_hashes,
        "system_config_sha256": system_config_fingerprint(cfg),
        "source_bundle_sha256": system.manifest.get("source_bundle_sha256"),
        "dataset_content_index_sha256": system.manifest.get(
            "dataset_content_index_sha256"
        ),
        "dataset_metadata_sha256": system.manifest.get("dataset_metadata_sha256"),
        "timing_system_artifact": TIMING_SYSTEM_ARTIFACT_FILENAME,
        "timing_system_artifact_sha256": system.manifest.get("system_artifact_sha256"),
        "timing_solution_artifact": TIMING_SOLUTIONS_ARTIFACT_FILENAME,
        "timing_solution_artifact_sha256": artifact_sha256,
        "selection_policy": (
            "lowest repeat_idx converged measured beta; otherwise lowest available "
            "measured beta marked ineligible"
        ),
        "solution_count": int(
            sum(bool(record.get("available")) for record in solution_records)
        ),
        "solutions": solution_records,
    }
    manifest_path.write_text(
        json.dumps(
            _sanitize_json(payload),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    system.manifest.update(
        {
            "timing_solution_artifact": TIMING_SOLUTIONS_ARTIFACT_FILENAME,
            "timing_solution_artifact_sha256": artifact_sha256,
            "timing_solution_manifest": TIMING_SOLUTIONS_MANIFEST_FILENAME,
            "timing_solution_manifest_sha256": manifest_sha256,
            "timing_solution_count": payload["solution_count"],
            "timing_solution_selection_policy": payload["selection_policy"],
        }
    )
    payload["timing_solution_manifest_sha256"] = manifest_sha256
    return payload


def run_controlled_experiment(
    cfg: ControlledConfig,
    *,
    prepared_system: PreparedSystem | None = None,
) -> Path:
    system = prepare_shared_system(cfg) if prepared_system is None else prepared_system
    _validate_prepared_system_for_config(system, cfg)
    specs, rule = resolve_method_specs(system, cfg)
    if len({spec.label for spec in specs}) != len(specs):
        raise ValueError(
            "method labels must be unique; do not request duplicate methods."
        )
    print(
        f"Prepared system {system.system_id[:12]}: N={system.manifest['n_train']}, "
        f"M={system.manifest['M']}, setup={system.setup_seconds:.3f}s, "
        f"stage={system.manifest.get('nufft_stage')}."
    )
    print(
        f"Frozen score box: |B|={rule.active.box_idx.size}/{system.manifest['M']}, "
        f"radii={rule.active.radii.tolist()}, rule={rule.selection_rule}, q={rule.effective_rank}."
    )
    frozen_box_hash = _box_fingerprint(rule.active)

    def with_repeated_selection_timing(spec: MethodSpec) -> MethodSpec:
        """Rerun and time active-set selection for each cold method repeat."""
        if float(spec.selection_seconds) <= 0.0:
            return spec
        fresh_rule, selection_seconds = _timed(
            system.backend,
            lambda: resolve_score_box_rule(system, cfg),
        )
        fresh_hash = _box_fingerprint(fresh_rule.active)
        if (
            fresh_hash != frozen_box_hash
            or fresh_rule.selection_rule != rule.selection_rule
            or int(fresh_rule.effective_rank) != int(rule.effective_rank)
            or int(fresh_rule.active.box_idx.size) != int(rule.active.box_idx.size)
        ):
            raise RuntimeError(
                "score-box selection changed during repeated timing: "
                f"frozen={frozen_box_hash}, fresh={fresh_hash}"
            )
        return replace(spec, selection_seconds=float(selection_seconds))

    order_rng = np.random.default_rng(int(cfg.method_order_seed))
    rows: list[dict[str, Any]] = []
    saved_solutions: list[tuple[dict[str, Any], np.ndarray]] = []
    total_rounds = int(cfg.warmup_repeats) + int(cfg.measured_repeats)
    for round_idx in range(total_rounds):
        is_warmup = round_idx < int(cfg.warmup_repeats)
        repeat_idx = round_idx - int(cfg.warmup_repeats)
        order = order_rng.permutation(len(specs))
        for position, spec_index in enumerate(order):
            frozen_spec = specs[int(spec_index)]
            spec = with_repeated_selection_timing(frozen_spec)
            print(
                f"{'warmup' if is_warmup else f'repeat {repeat_idx + 1}'} "
                f"[{position + 1}/{len(specs)}] {spec.label}",
                flush=True,
            )
            row, beta = run_one_method(
                system,
                cfg,
                spec,
                repeat_idx=int(repeat_idx),
                order_position=int(position),
                is_warmup=bool(is_warmup),
            )
            rows.append(row)
            if beta is not None and not is_warmup:
                saved_solutions.append((row, beta))

    cg_reference = next(
        (
            beta
            for row, beta in saved_solutions
            if row.get("method") == "cg" and _is_converged(row, cfg.tol)
        ),
        None,
    )
    if cg_reference is None:
        raise RuntimeError(
            "no converged measured CG solution is available as the fixed-system reference."
        )
    reference_norm = max(float(np.linalg.norm(cg_reference)), 1e-300)
    for row, beta in saved_solutions:
        row["relative_beta_error_vs_cg"] = float(
            np.linalg.norm(beta - cg_reference) / reference_norm
        )

    final_system_id = system_fingerprint(
        system.data_ctx,
        float(system.reg_lambda),
        solve_rhs_gpu=system.rhs_gpu,
    )
    if final_system_id != system.system_id:
        raise RuntimeError(
            "the arrays defining A,b changed during the controlled experiment: "
            f"{system.system_id} -> {final_system_id}"
        )
    for label in {row.get("method") for row in rows if row.get("box_hash")}:
        hashes = {
            str(row["box_hash"])
            for row in rows
            if row.get("method") == label and row.get("box_hash")
        }
        if len(hashes) != 1:
            raise RuntimeError(
                f"active box changed across repeats for method {label!r}: {hashes}"
            )

    summaries = summarize_rows(
        rows,
        system.setup_seconds,
        cfg.tol,
        method_order=[spec.label for spec in specs],
    )
    comparisons = pairwise_comparisons(rows, cfg.tol)
    diagnostic_rows, diagnostic_arrays = _post_diagnostics(system, cfg, specs, rule)
    if (
        system_fingerprint(
            system.data_ctx,
            float(system.reg_lambda),
            solve_rhs_gpu=system.rhs_gpu,
        )
        != system.system_id
    ):
        raise RuntimeError(
            "post diagnostics modified the arrays defining the fixed system."
        )

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir).expanduser().resolve()
    else:
        tag = datetime.now().strftime("matched_%Y%m%d_%H%M%S")
        output_dir = (_HERE / "outputs" / tag).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "run_complete.json"
    completion_path.unlink(missing_ok=True)
    save_prepared_system_artifact(
        system,
        cfg,
        output_dir / TIMING_SYSTEM_ARTIFACT_FILENAME,
    )
    timing_solution_payload = save_timing_prediction_solutions(
        system,
        cfg,
        rows,
        saved_solutions,
        output_dir,
    )
    config_payload = asdict(cfg)
    config_payload["methods"] = list(cfg.methods)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            _sanitize_json(config_payload),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    manifest = dict(system.manifest)
    score_tail = np.asarray(rule.active.tail_idx, dtype=np.int64)
    score_rho = np.asarray(rule.active.rho, dtype=np.float64)
    induced_tail_threshold = (
        float(np.max(score_rho[score_tail])) if score_tail.size else 0.0
    )
    measured_selection_seconds = [
        float(row["selection_seconds"])
        for row in rows
        if not bool(row.get("is_warmup", False))
        and float(row.get("selection_seconds", 0.0)) > 0.0
    ]
    default_selection_seconds = [
        float(row["selection_seconds"])
        for row in rows
        if not bool(row.get("is_warmup", False))
        and str(row.get("method")) == "default"
        and float(row.get("selection_seconds", 0.0)) > 0.0
    ]
    selection_seconds_median_by_method = {
        str(method): float(np.median(values))
        for method in {str(row.get("method")) for row in rows}
        if (
            values := [
                float(row["selection_seconds"])
                for row in rows
                if not bool(row.get("is_warmup", False))
                and str(row.get("method")) == str(method)
                and float(row.get("selection_seconds", 0.0)) > 0.0
            ]
        )
    }
    manifest.update(
        {
            "final_system_id": final_system_id,
            "system_unchanged": True,
            "score_rule": rule.selection_rule,
            "score_tau": float(cfg.score_tau),
            "score_tau_raw_box_size": int(rule.raw_tau_box_size),
            "score_box_size": int(rule.active.box_idx.size),
            "score_box_radii": [int(v) for v in rule.active.radii],
            "score_box_hash": _box_fingerprint(rule.active),
            "score_requested_rank": int(rule.requested_rank),
            "score_effective_rank": int(rule.effective_rank),
            "score_induced_tail_threshold": induced_tail_threshold,
            "score_cap_excludes_requested_threshold_modes": bool(
                induced_tail_threshold > float(cfg.score_tau)
            ),
            "score_protocol_freeze_selection_seconds": float(rule.selection_seconds),
            "score_selection_seconds": (
                float(np.median(measured_selection_seconds))
                if measured_selection_seconds
                else 0.0
            ),
            "selection_seconds_median_by_method": selection_seconds_median_by_method,
            # Backward-compatible manifest alias, now the median of the actual
            # per-repeat default selections rather than the protocol-freeze pass.
            "default_selection_seconds": (
                float(np.median(default_selection_seconds))
                if default_selection_seconds
                else 0.0
            ),
            "selection_timing_protocol": (
                "score-box selection is rerun and synchronized for every warmup and "
                "measured active-method invocation; the frozen box hash/rule/rank "
                "must remain unchanged"
            ),
            "timing_scope": (
                "solver_total_seconds is score selection plus preconditioner "
                "construction plus CG/PCG solve. Score-selected default and explicit "
                "active methods each include their own per-repeat score-box selection "
                "cost. The one shared Fourier setup is reported separately."
            ),
            "method_order": "independently shuffled in every warmup/measured round",
        }
    )
    (output_dir / "system_manifest.json").write_text(
        json.dumps(
            _sanitize_json(manifest),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "matched_runs.json").write_text(
        json.dumps(
            _sanitize_json(rows),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "matched_runs.csv", rows)
    (output_dir / "matched_summary.json").write_text(
        json.dumps(
            _sanitize_json(summaries),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "matched_summary.csv", summaries)
    (output_dir / "matched_comparisons.json").write_text(
        json.dumps(
            _sanitize_json(comparisons),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "matched_comparisons.csv", comparisons)
    if diagnostic_rows:
        (output_dir / "post_diagnostics.json").write_text(
            json.dumps(
                _sanitize_json(diagnostic_rows),
                indent=2,
                ensure_ascii=False,
                default=_json_default,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        _write_csv(output_dir / "post_diagnostics.csv", diagnostic_rows)
    if diagnostic_arrays:
        np.savez_compressed(
            output_dir / "score_leverage_arrays.npz", **diagnostic_arrays
        )
    completion_payload = {
        "schema_version": 1,
        "system_id": system.system_id,
        "source_bundle_sha256": manifest["source_bundle_sha256"],
        "dataset_content_index_sha256": manifest["dataset_content_index_sha256"],
        "dataset_metadata_sha256": manifest["dataset_metadata_sha256"],
        "methods": [spec.label for spec in specs],
        "warmup_repeats": int(cfg.warmup_repeats),
        "measured_repeats": int(cfg.measured_repeats),
        "tol": float(cfg.tol),
        "maxiter": int(cfg.maxiter),
        "zero_initial_vector": bool(cfg.zero_initial_vector),
        "run_row_count": len(rows),
        "summary_row_count": len(summaries),
        "comparison_row_count": len(comparisons),
        "timing_system_artifact": TIMING_SYSTEM_ARTIFACT_FILENAME,
        "timing_system_artifact_sha256": manifest["system_artifact_sha256"],
        "timing_solution_artifact": TIMING_SOLUTIONS_ARTIFACT_FILENAME,
        "timing_solution_artifact_sha256": timing_solution_payload[
            "timing_solution_artifact_sha256"
        ],
        "timing_solution_manifest": TIMING_SOLUTIONS_MANIFEST_FILENAME,
        "timing_solution_manifest_sha256": timing_solution_payload[
            "timing_solution_manifest_sha256"
        ],
        "timing_solution_count": int(timing_solution_payload["solution_count"]),
    }
    completion_temp = output_dir / ".run_complete.json.tmp"
    completion_temp.write_text(
        json.dumps(
            _sanitize_json(completion_payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    completion_temp.replace(completion_path)
    print(f"Wrote controlled experiment to {output_dir}")
    return output_dir


def _parse_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(
        part.strip().lower() for part in str(raw).split(",") if part.strip()
    )
    if not methods:
        raise argparse.ArgumentTypeError("methods must contain at least one name")
    return methods


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    if not str(raw).strip():
        return ()
    try:
        values = tuple(
            int(part.strip()) for part in str(raw).split(",") if part.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CG and preconditioned CG on one immutable Fourier system. "
            "Every measured repeat starts at zero and uses randomized method order."
        )
    )
    parser.add_argument("--dataset-stem", default=ControlledConfig.dataset_stem)
    parser.add_argument(
        "--dataset-dir",
        default=ControlledConfig.dataset_dir,
        help=(
            "Directory containing processed NPZ/JSON files. If omitted, use "
            "BTAB_PROCESSED_DIR and then the repository default."
        ),
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=ControlledConfig.n_train,
        help="Seeded training subset size; use 0 for all available rows.",
    )
    parser.add_argument("--subset-seed", type=int, default=ControlledConfig.subset_seed)
    parser.add_argument(
        "--subset-mode",
        choices=("random", "prefix"),
        default=ControlledConfig.subset_mode,
        help=(
            "random preserves the historical seeded subset; prefix reads an exact nested "
            "row prefix and memory-maps uncompressed NPZ masters."
        ),
    )
    parser.add_argument(
        "--kernel", dest="kernel_family", default=ControlledConfig.kernel_family
    )
    parser.add_argument(
        "--lengthscale", type=float, default=ControlledConfig.lengthscale
    )
    parser.add_argument("--nu", type=float, default=ControlledConfig.nu)
    parser.add_argument("--variance", type=float, default=ControlledConfig.variance)
    parser.add_argument(
        "--lambda", dest="reg_lambda", type=float, default=ControlledConfig.reg_lambda
    )
    parser.add_argument(
        "--fourier-eps", type=float, default=ControlledConfig.fourier_eps
    )
    parser.add_argument("--nufft-tol", type=float, default=ControlledConfig.nufft_tol)
    parser.add_argument(
        "--l2-scaled",
        action=argparse.BooleanOptionalAction,
        default=ControlledConfig.l2_scaled,
    )
    parser.add_argument("--tol", type=float, default=ControlledConfig.tol)
    parser.add_argument("--maxiter", type=int, default=ControlledConfig.maxiter)
    parser.add_argument(
        "--precision",
        choices=("fp64", "mixed32"),
        default=ControlledConfig.precision,
        help=(
            "fp64 is the publication path. mixed32 lowers Krylov vectors/matvec outputs "
            "but retains fp64 Fourier storage and is reported as mixed precision."
        ),
    )
    parser.add_argument("--allow-near-epsilon-tol", action="store_true")
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=ControlledConfig.methods,
        help=(
            "Comma-separated fixed-A,b methods: cg,jacobi,default,active-inverse,"
            "full-inverse,active-eig,full-eig. Optional exploratory Fourier-space "
            "adaptations are fourier-nystrom-precond and "
            "fourier-rpcholesky-precond; they are not data-space KRR pipelines."
        ),
    )
    parser.add_argument("--score-tau", type=float, default=ControlledConfig.score_tau)
    parser.add_argument("--box-budget", type=int, default=ControlledConfig.box_budget)
    parser.add_argument(
        "--inverse-max-size", type=int, default=ControlledConfig.inverse_max_size
    )
    parser.add_argument("--rank", type=int, default=ControlledConfig.rank)
    parser.add_argument(
        "--fourier-nystrom-rank",
        "--nystrom-rank",
        dest="nystrom_rank",
        type=int,
        default=ControlledConfig.nystrom_rank,
        help=(
            "Rank for fourier-nystrom-precond. --nystrom-rank is retained as a "
            "backward-compatible option alias."
        ),
    )
    parser.add_argument(
        "--fourier-rpcholesky-rank",
        "--rpcholesky-rank",
        dest="rpcholesky_rank",
        type=int,
        default=ControlledConfig.rpcholesky_rank,
        help=(
            "Rank for fourier-rpcholesky-precond. --rpcholesky-rank is retained "
            "as a backward-compatible option alias."
        ),
    )
    parser.add_argument("--eig-tol", type=float, default=ControlledConfig.eig_tol)
    parser.add_argument("--eig-maxiter", type=int, default=ControlledConfig.eig_maxiter)
    parser.add_argument(
        "--measured-repeats", type=int, default=ControlledConfig.measured_repeats
    )
    parser.add_argument(
        "--warmup-repeats", type=int, default=ControlledConfig.warmup_repeats
    )
    parser.add_argument(
        "--method-order-seed", type=int, default=ControlledConfig.method_order_seed
    )
    parser.add_argument("--eig-seed", type=int, default=ControlledConfig.eig_seed)
    parser.add_argument(
        "--fourier-nystrom-seed",
        "--nystrom-seed",
        dest="nystrom_seed",
        type=int,
        default=ControlledConfig.nystrom_seed,
    )
    parser.add_argument(
        "--fourier-rpcholesky-seed",
        "--rpcholesky-seed",
        dest="rpcholesky_seed",
        type=int,
        default=ControlledConfig.rpcholesky_seed,
    )
    parser.add_argument(
        "--nufft-backend",
        choices=("auto", "cufinufft", "none"),
        default=ControlledConfig.nufft_backend,
    )
    parser.add_argument(
        "--precompute-chunk-size",
        type=int,
        default=0,
        help="NUFFT setup chunk size; zero disables chunking.",
    )
    parser.add_argument(
        "--post-diagnostic-mode",
        choices=("none", "cheap", "full"),
        default=ControlledConfig.post_diagnostic_mode,
    )
    parser.add_argument(
        "--diagnostic-tol", type=float, default=ControlledConfig.diagnostic_tol
    )
    parser.add_argument(
        "--diagnostic-power-iter",
        type=int,
        default=ControlledConfig.diagnostic_power_iter,
    )
    parser.add_argument(
        "--diagnostic-topk",
        type=_parse_int_tuple,
        default=ControlledConfig.diagnostic_topk,
        help=(
            "Optional comma-separated nested score top-k values. In full diagnostic mode, "
            "each feasible enclosing box is inverted and measured for Section 4.2."
        ),
    )
    parser.add_argument("--strict-gpu-eig", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser


def config_from_args(args: argparse.Namespace) -> ControlledConfig:
    values = vars(args).copy()
    values["n_train"] = None if int(values["n_train"]) == 0 else int(values["n_train"])
    values["precompute_chunk_size"] = (
        None
        if int(values["precompute_chunk_size"]) == 0
        else int(values["precompute_chunk_size"])
    )
    values["methods"] = tuple(values["methods"])
    values["diagnostic_topk"] = tuple(values["diagnostic_topk"])
    return ControlledConfig(**values)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    cfg = config_from_args(parser.parse_args(argv))
    run_controlled_experiment(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
