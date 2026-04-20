from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _normalize_backend_token(field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"BackendConfig.{field} must be str, got {type(value).__name__}")
    s = value.strip().lower()
    if not s:
        raise ValueError(f"BackendConfig.{field} must be a non-empty string.")
    return s


@dataclass(frozen=True)
class BackendConfig:
    """
    Backend selection for staged GPU migration.
    All string fields are normalized to lowercase non-empty tokens at build time.
    """

    xp: str = "cupy"
    fft: str = "cupy"
    nufft: str = "auto"
    linalg: str = "cupy"


class CuPyFFTOps:
    """
    Thin FFT surface so callers do not depend on the raw ``cupy.fft`` module object.
    Later: plan cache, alternate backends, timing hooks.
    """

    __slots__ = ("_cp",)

    def __init__(self, cp: Any) -> None:
        self._cp = cp

    def fftn(self, a: Any, *args: Any, **kwargs: Any) -> Any:
        return self._cp.fft.fftn(a, *args, **kwargs)

    def ifftn(self, a: Any, *args: Any, **kwargs: Any) -> Any:
        return self._cp.fft.ifftn(a, *args, **kwargs)


class CuPyLinalgOps:
    """
    Thin linalg surface (expand as CG / eigenspace code needs more ops).
    """

    __slots__ = ("_cp",)

    def __init__(self, cp: Any) -> None:
        self._cp = cp

    def norm(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        return self._cp.linalg.norm(x, *args, **kwargs)

    def vdot(self, a: Any, b: Any) -> Any:
        return self._cp.vdot(a, b)


def _import_cupy() -> Any:
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CuPy is required for GPU backend.") from exc
    return cp


def _device_name(cp: Any, device_id: int) -> str:
    try:
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        raw = getattr(props, "name", None)
        if raw is None and isinstance(props, dict):
            raw = props.get("name")
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        if raw is not None:
            return str(raw)
    except Exception:
        pass
    return f"cuda_device_{device_id}"


def _ensure_cuda_device_available(cp: Any, device_id: int) -> None:
    """
    Fail fast if CUDA is not usable (import succeeds but runtime/device is broken).
    """
    try:
        n = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError("CuPy is importable but CUDA runtime device count is unavailable.") from exc
    if n <= 0:
        raise RuntimeError("No CUDA devices are visible to the runtime (getDeviceCount() == 0).")
    if device_id < 0 or device_id >= n:
        raise RuntimeError(f"device_id={device_id} is out of range for {n} visible device(s).")

    try:
        with cp.cuda.Device(device_id):
            cp.cuda.Stream.null.synchronize()
            _ = cp.zeros(8, dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
    except Exception as exc:
        raise RuntimeError(
            f"CUDA device {device_id} failed a minimal allocation/sync smoke test "
            "(driver/runtime mismatch or no functional GPU)."
        ) from exc


def _probe_dtype_support(cp: Any, device_id: int) -> tuple[bool, bool]:
    """
    Returns (supports_fp64, supports_complex128) on the chosen device.
    """
    supports_fp64 = False
    supports_complex128 = False
    with cp.cuda.Device(device_id):
        try:
            a = cp.zeros(1, dtype=cp.float64)
            a += 1.0
            cp.cuda.Stream.null.synchronize()
            supports_fp64 = True
        except Exception:
            supports_fp64 = False
        try:
            b = cp.zeros(1, dtype=cp.complex128)
            b += 1.0 + 0.0j
            cp.cuda.Stream.null.synchronize()
            supports_complex128 = True
        except Exception:
            supports_complex128 = False
    return supports_fp64, supports_complex128


def _select_nufft_backend(preference: str) -> tuple[Optional[Any], str]:
    pref = preference.lower()
    if pref not in {"auto", "cufinufft", "none"}:
        raise ValueError("nufft must be one of: auto, cufinufft, none")

    if pref in {"auto", "cufinufft"}:
        try:
            import cufinufft as cuf  # type: ignore

            return cuf, "cufinufft"
        except Exception as exc:
            if pref == "cufinufft":
                raise RuntimeError("Requested cufinufft but it is not available.") from exc
    return None, "none"


@dataclass
class GPUBackendBundle:
    """
    Runtime backend bundle.

    ``nufft`` holds the *raw* discovered Python module (e.g. ``cufinufft``) when present.
    Do not assume a stable function surface across backends; add a dedicated NUFFT
    adapter module (type-1/type-2) before calling into GPU NUFFT from solvers.
    """

    xp: Any
    fft: CuPyFFTOps
    linalg: CuPyLinalgOps
    nufft: Optional[Any]
    nufft_name: str
    device_id: int
    device_name: str
    has_nufft: bool
    supports_fp64: bool
    supports_complex128: bool


def build_gpu_backend_bundle(cfg: Optional[BackendConfig] = None) -> GPUBackendBundle:
    """
    Build a backend bundle for GPU orchestration.

    Performs early CUDA smoke tests so failures happen here, not on first large allocation.
    """
    cfg = cfg or BackendConfig()
    xp_tok = _normalize_backend_token("xp", cfg.xp)
    fft_tok = _normalize_backend_token("fft", cfg.fft)
    linalg_tok = _normalize_backend_token("linalg", cfg.linalg)
    nufft_tok = _normalize_backend_token("nufft", cfg.nufft)

    if xp_tok != "cupy":
        raise ValueError("Current scaffold supports xp='cupy' only.")
    if fft_tok != "cupy":
        raise ValueError("Current scaffold supports fft='cupy' only.")
    if linalg_tok != "cupy":
        raise ValueError("Current scaffold supports linalg='cupy' only.")

    cp = _import_cupy()
    device_id = 0
    _ensure_cuda_device_available(cp, device_id)
    device_name = _device_name(cp, device_id)
    supports_fp64, supports_complex128 = _probe_dtype_support(cp, device_id)

    nufft_backend, nufft_name = _select_nufft_backend(nufft_tok)
    has_nufft = nufft_backend is not None

    return GPUBackendBundle(
        xp=cp,
        fft=CuPyFFTOps(cp),
        linalg=CuPyLinalgOps(cp),
        nufft=nufft_backend,
        nufft_name=nufft_name,
        device_id=device_id,
        device_name=device_name,
        has_nufft=has_nufft,
        supports_fp64=supports_fp64,
        supports_complex128=supports_complex128,
    )
