"""
Unified NUFFT surface for V1: prefer cuFINUFFT on GPU when ``backend.has_nufft``,
otherwise CPU FINUFFT (``nufft_ops``) with explicit transfers.

Sign / grid conventions follow ``efgp_eigenpro_py/nufft_ops.py`` and ``EFGPSolver``.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..nufft_ops import (
    nufft1d1,
    nufft1d2,
    nufft2d1,
    nufft2d2,
    nufft3d1,
    nufft3d2,
    nufftnd1,
    nufftnd2,
)
from .backends import GPUBackendBundle


def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "get"):
        return np.asarray(x.get())
    return np.asarray(x)


def _as_gpu(xp: Any, x: np.ndarray, dtype: Any = None) -> Any:
    arr = xp.asarray(np.ascontiguousarray(x))
    if dtype is not None and arr.dtype != dtype:
        return arr.astype(dtype)
    return arr


def type1_ones_xtx(
    backend: GPUBackendBundle,
    tphx: Any,
    dim: int,
    mtot: int,
    eps: float,
    isign: int,
) -> Tuple[Any, str]:
    """
    Type-1 NUFFT with unit strengths (Toeplitz first stage), output grid shape ``(2*mtot-1,)*dim``.
    """
    xp = backend.xp
    ms_xtx = 2 * int(mtot) - 1
    n = int(tphx.shape[0])

    if backend.has_nufft and backend.nufft is not None:
        cuf = backend.nufft
        c = xp.ones(n, dtype=xp.complex128)
        try:
            if dim == 1:
                x0 = xp.ascontiguousarray(tphx[:, 0])
                out = cuf.nufft1d1(x0, c, (ms_xtx,), eps=eps, isign=isign)
            elif dim == 2:
                x0 = xp.ascontiguousarray(tphx[:, 0])
                x1 = xp.ascontiguousarray(tphx[:, 1])
                out = cuf.nufft2d1(x0, x1, c, (ms_xtx, ms_xtx), eps=eps, isign=isign)
            elif dim == 3:
                x0 = xp.ascontiguousarray(tphx[:, 0])
                x1 = xp.ascontiguousarray(tphx[:, 1])
                x2 = xp.ascontiguousarray(tphx[:, 2])
                out = cuf.nufft3d1(
                    x0,
                    x1,
                    x2,
                    c,
                    (ms_xtx, ms_xtx, ms_xtx),
                    eps=eps,
                    isign=isign,
                )
            else:
                raise NotImplementedError("cufinufft path supports dim<=3 only.")
            return xp.ascontiguousarray(out), "cufinufft"
        except Exception:
            pass

    tphx_np = np.ascontiguousarray(_as_numpy(tphx), dtype=np.float64)
    c_np = np.ones(n, dtype=np.complex128)
    if dim == 1:
        Xt = nufft1d1(tphx_np[:, 0], c_np, ms_xtx, eps, isign)
    elif dim == 2:
        Xt = nufft2d1(tphx_np[:, 0], tphx_np[:, 1], c_np, ms_xtx, ms_xtx, eps, isign)
    elif dim == 3:
        Xt = nufft3d1(
            tphx_np[:, 0],
            tphx_np[:, 1],
            tphx_np[:, 2],
            c_np,
            ms_xtx,
            ms_xtx,
            ms_xtx,
            eps,
            isign,
        )
    else:
        Xt = nufftnd1(tphx_np, c_np, ms_xtx, eps, isign)
        Xt = np.asarray(Xt).reshape((ms_xtx,) * dim)
    return _as_gpu(xp, Xt, dtype=xp.complex128), "cpu_finufft"


def type1_coeffs_rhs(
    backend: GPUBackendBundle,
    tphx: Any,
    coeffs: Any,
    dim: int,
    mtot: int,
    eps: float,
    isign: int,
) -> Tuple[Any, str]:
    """
    Type-1 NUFFT with per-point complex coefficients (rhs stage), grid ``(mtot,)*dim`` flattened later.
    """
    xp = backend.xp
    n = int(tphx.shape[0])
    coeffs = xp.asarray(coeffs, dtype=xp.complex128).reshape(n)

    if backend.has_nufft and backend.nufft is not None:
        cuf = backend.nufft
        try:
            if dim == 1:
                x0 = xp.ascontiguousarray(tphx[:, 0])
                out = cuf.nufft1d1(x0, coeffs, (int(mtot),), eps=eps, isign=isign)
            elif dim == 2:
                x0 = xp.ascontiguousarray(tphx[:, 0])
                x1 = xp.ascontiguousarray(tphx[:, 1])
                out = cuf.nufft2d1(x0, x1, coeffs, (int(mtot), int(mtot)), eps=eps, isign=isign)
            elif dim == 3:
                m = int(mtot)
                x0 = xp.ascontiguousarray(tphx[:, 0])
                x1 = xp.ascontiguousarray(tphx[:, 1])
                x2 = xp.ascontiguousarray(tphx[:, 2])
                out = cuf.nufft3d1(
                    x0,
                    x1,
                    x2,
                    coeffs,
                    (m, m, m),
                    eps=eps,
                    isign=isign,
                )
            else:
                raise NotImplementedError("cufinufft path supports dim<=3 only.")
            return xp.ascontiguousarray(out.reshape(-1)), "cufinufft"
        except Exception:
            pass

    tphx_np = np.ascontiguousarray(_as_numpy(tphx), dtype=np.float64)
    c_np = np.ascontiguousarray(_as_numpy(coeffs), dtype=np.complex128)
    if dim == 1:
        rhs = nufft1d1(tphx_np[:, 0], c_np, int(mtot), eps, isign)
    elif dim == 2:
        rhs = nufft2d1(tphx_np[:, 0], tphx_np[:, 1], c_np, int(mtot), int(mtot), eps, isign)
    elif dim == 3:
        m = int(mtot)
        rhs = nufft3d1(tphx_np[:, 0], tphx_np[:, 1], tphx_np[:, 2], c_np, m, m, m, eps, isign)
    else:
        rhs = nufftnd1(tphx_np, c_np, int(mtot), eps, isign)
        rhs = np.asarray(rhs).reshape(-1)
    return _as_gpu(xp, rhs.reshape(-1), dtype=xp.complex128), "cpu_finufft"


def type2_eval(
    backend: GPUBackendBundle,
    tphx: Any,
    wbeta_flat: Any,
    dim: int,
    mtot: int,
    eps: float,
    isign: int,
) -> Tuple[Any, str]:
    """
    Type-2 NUFFT: uniform Fourier coeffs ``wbeta`` on ``(mtot,)*dim`` to nonuniform ``tphx``.
    Returns real part on GPU as float64 (caller may cast).
    """
    xp = backend.xp
    w = xp.asarray(wbeta_flat, dtype=xp.complex128).reshape(-1)

    if backend.has_nufft and backend.nufft is not None:
        cuf = backend.nufft
        m = int(mtot)
        try:
            if dim == 1:
                f1 = w.reshape(m)
                x0 = xp.ascontiguousarray(tphx[:, 0])
                yhat = cuf.nufft1d2(x0, f1, eps=eps, isign=isign)
            elif dim == 2:
                f2 = w.reshape(m, m)
                x0 = xp.ascontiguousarray(tphx[:, 0])
                x1 = xp.ascontiguousarray(tphx[:, 1])
                yhat = cuf.nufft2d2(x0, x1, f2, eps=eps, isign=isign)
            elif dim == 3:
                f3 = w.reshape(m, m, m)
                x0 = xp.ascontiguousarray(tphx[:, 0])
                x1 = xp.ascontiguousarray(tphx[:, 1])
                x2 = xp.ascontiguousarray(tphx[:, 2])
                yhat = cuf.nufft3d2(x0, x1, x2, f3, eps=eps, isign=isign)
            else:
                raise NotImplementedError("cufinufft path supports dim<=3 only.")
            return xp.real(yhat).astype(xp.float64), "cufinufft"
        except Exception:
            pass

    tphx_np = np.ascontiguousarray(_as_numpy(tphx), dtype=np.float64)
    w_np = np.ascontiguousarray(_as_numpy(w), dtype=np.complex128)
    m = int(mtot)
    if dim == 1:
        yhat = nufft1d2(tphx_np[:, 0], w_np.reshape(m), eps, isign)
    elif dim == 2:
        yhat = nufft2d2(tphx_np[:, 0], tphx_np[:, 1], w_np.reshape(m, m), eps, isign)
    elif dim == 3:
        yhat = nufft3d2(tphx_np[:, 0], tphx_np[:, 1], tphx_np[:, 2], w_np.reshape(m, m, m), eps, isign)
    else:
        yhat = nufftnd2(
            tphx_np,
            w_np.reshape(-1),
            eps,
            isign,
            n_modes=(m,) * dim,
        )
    yhat_r = np.real(yhat).astype(np.float64, copy=False)
    return _as_gpu(xp, yhat_r, dtype=xp.float64), "cpu_finufft"
