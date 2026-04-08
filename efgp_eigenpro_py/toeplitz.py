import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from scipy import fft as _fft  # type: ignore
    _HAS_SCIPY_FFT = True
except Exception:  # pragma: no cover
    import numpy.fft as _fft  # type: ignore
    _HAS_SCIPY_FFT = False


def _fftn(
    a: np.ndarray,
    s: Optional[Tuple[int, ...]] = None,
    axes: Optional[Tuple[int, ...]] = None,
    overwrite_x: bool = False,
) -> np.ndarray:
    workers_env = os.environ.get("EFGP_FFT_WORKERS")
    if _HAS_SCIPY_FFT:
        if workers_env is not None:
            return _fft.fftn(
                a, s=s, axes=axes, workers=int(workers_env), overwrite_x=overwrite_x
            )
        return _fft.fftn(a, s=s, axes=axes, overwrite_x=overwrite_x)
    return _fft.fftn(a, s=s, axes=axes)


def _ifftn(
    a: np.ndarray,
    axes: Optional[Tuple[int, ...]] = None,
    overwrite_x: bool = False,
) -> np.ndarray:
    workers_env = os.environ.get("EFGP_FFT_WORKERS")
    if _HAS_SCIPY_FFT:
        if workers_env is not None:
            return _fft.ifftn(
                a, axes=axes, workers=int(workers_env), overwrite_x=overwrite_x
            )
        return _fft.ifftn(a, axes=axes, overwrite_x=overwrite_x)
    return _fft.ifftn(a, axes=axes)


@dataclass
class ToeplitzWorkspace:
    pad: np.ndarray
    out: np.ndarray
    pad_slicer: Tuple[slice, ...]
    out_slicer: Tuple[slice, ...]


@dataclass
class ToeplitzBlockWorkspace:
    pad: np.ndarray
    out: np.ndarray
    pad_slicer: Tuple[slice, ...]
    out_slicer: Tuple[slice, ...]
    block_size: int


def make_toeplitz_workspace(
    Gf: np.ndarray,
    mtot: int,
    dim: int,
    dtype: Optional[np.dtype] = None,
) -> ToeplitzWorkspace:
    if dim < 1:
        raise ValueError("dim must be >= 1.")
    if mtot <= 0:
        raise ValueError("mtot must be positive.")
    if Gf.ndim != dim:
        raise ValueError("Gf must have ndim == dim.")
    use_dtype = dtype or Gf.dtype
    pad = np.empty(Gf.shape, dtype=use_dtype)
    out = np.empty((mtot,) * dim, dtype=use_dtype).reshape(-1)
    pad_slicer = tuple(slice(0, mtot) for _ in range(dim))
    out_slicer = tuple(slice(mtot - 1, 2 * mtot - 1) for _ in range(dim))
    return ToeplitzWorkspace(pad=pad, out=out, pad_slicer=pad_slicer, out_slicer=out_slicer)


def make_toeplitz_block_workspace(
    Gf: np.ndarray,
    mtot: int,
    dim: int,
    block_size: int,
    dtype: Optional[np.dtype] = None,
) -> ToeplitzBlockWorkspace:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    use_dtype = dtype or Gf.dtype
    pad = np.empty(Gf.shape + (block_size,), dtype=use_dtype)
    out = np.empty(((mtot,) * dim + (block_size,)), dtype=use_dtype)
    pad_slicer = tuple(slice(0, mtot) for _ in range(dim))
    out_slicer = tuple(slice(mtot - 1, 2 * mtot - 1) for _ in range(dim))
    return ToeplitzBlockWorkspace(
        pad=pad,
        out=out,
        pad_slicer=pad_slicer,
        out_slicer=out_slicer,
        block_size=block_size,
    )


def apply_toeplitz_fft_1d(
    Gf: np.ndarray,
    a: np.ndarray,
    mtot: int,
    workspace: Optional[ToeplitzWorkspace] = None,
) -> np.ndarray:
    """
    1D Toeplitz multiply using FFT embedding.
    TODO: verify slice convention matches your NUFFT construction.
    """
    if workspace is None:
        af = _fftn(a, s=Gf.shape)
        np.multiply(af, Gf, out=af)
        v = _ifftn(af)
        return v[mtot - 1 :]

    pad = workspace.pad
    pad.fill(0)
    pad[:mtot] = a
    af = _fftn(pad, overwrite_x=_HAS_SCIPY_FFT)
    np.multiply(af, Gf, out=af)
    v = _ifftn(af, overwrite_x=_HAS_SCIPY_FFT)
    np.copyto(workspace.out, v[mtot - 1 :])
    return workspace.out


def apply_toeplitz_fft_nd(
    Gf: np.ndarray,
    a: np.ndarray,
    mtot: int,
    dim: int,
    workspace: Optional[ToeplitzWorkspace] = None,
) -> np.ndarray:
    """
    ND Toeplitz multiply using FFT embedding.
    """
    shape = (mtot,) * dim
    a_nd = np.reshape(a, shape)
    if workspace is None:
        af = _fftn(a_nd, s=Gf.shape)
        np.multiply(af, Gf, out=af)
        v = _ifftn(af)
        slicer = tuple(slice(mtot - 1, 2 * mtot - 1) for _ in range(dim))
        v = v[slicer]
        return v.reshape(-1)

    pad = workspace.pad
    pad.fill(0)
    pad[workspace.pad_slicer] = a_nd
    af = _fftn(pad, overwrite_x=_HAS_SCIPY_FFT)
    np.multiply(af, Gf, out=af)
    v = _ifftn(af, overwrite_x=_HAS_SCIPY_FFT)
    v = v[workspace.out_slicer]
    np.copyto(workspace.out, v.reshape(-1))
    return workspace.out


def apply_toeplitz_fft_nd_block(
    Gf: np.ndarray,
    a_block: np.ndarray,
    mtot: int,
    dim: int,
    workspace: Optional[ToeplitzBlockWorkspace] = None,
) -> np.ndarray:
    """
    ND Toeplitz multiply on a block of vectors.
    a_block has shape (M, b) where M = mtot**dim.
    """
    if a_block.ndim != 2:
        raise ValueError("a_block must be a 2D array of shape (M, b).")
    block_size = int(a_block.shape[1])
    shape = (mtot,) * dim
    a_nd = np.reshape(a_block, shape + (block_size,))
    axes = tuple(range(dim))
    if workspace is None:
        pad = np.zeros(Gf.shape + (block_size,), dtype=np.result_type(Gf.dtype, a_nd.dtype))
        pad_slicer = tuple(slice(0, mtot) for _ in range(dim))
        pad[pad_slicer + (slice(None),)] = a_nd
        af = _fftn(pad, s=Gf.shape, axes=axes)
        np.multiply(af, Gf[(...,) + (None,)], out=af)
        v = _ifftn(af, axes=axes)
        out_slicer = tuple(slice(mtot - 1, 2 * mtot - 1) for _ in range(dim))
        v = v[out_slicer + (slice(None),)]
        return v.reshape((mtot**dim, block_size))

    if workspace.block_size != block_size:
        raise ValueError("workspace block_size does not match a_block.")
    pad = workspace.pad
    pad.fill(0)
    pad[workspace.pad_slicer + (slice(None),)] = a_nd
    af = _fftn(pad, axes=axes, overwrite_x=_HAS_SCIPY_FFT)
    np.multiply(af, Gf[(...,) + (None,)], out=af)
    v = _ifftn(af, axes=axes, overwrite_x=_HAS_SCIPY_FFT)
    v = v[workspace.out_slicer + (slice(None),)]
    np.copyto(workspace.out, v)
    return workspace.out.reshape((mtot**dim, block_size))
