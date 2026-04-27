from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..efgp_solver import PrecomputeState
from .backends import GPUBackendBundle


@dataclass
class GPUDataContext:
    """
    Persistent GPU data for V1-V3 runtime.

    ``weights_gpu_nd`` / ``weights_gpu_flat`` avoid repeated reshape/ravel on hot paths.
    """

    x_gpu: Any
    y_gpu: Any
    weights_gpu_nd: Optional[Any] = None
    weights_gpu_flat: Optional[Any] = None
    # CPU copy of flat weights (for Nyström sampling, etc.); filled in precompute to avoid D2H each call.
    weights_np_flat: Optional[Any] = None
    rhs_gpu: Optional[Any] = None
    gf_gpu: Optional[Any] = None
    # First column of F^*F in the extended XTX grid (Toeplitz kernel); for Nyström and hot matvec.
    # Preferred source: precompute stores ``fftn`` of this as ``gf_gpu``; this is the spatial form.
    xtxcol_gpu: Optional[Any] = None
    x_center_gpu: Optional[Any] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUOperatorContext:
    """
    Workspaces for apply_A and iterative solvers (expand as implementations land).
    """

    work_vec: Optional[Any] = None
    pad: Optional[Any] = None
    fft_buf: Optional[Any] = None
    ifft_buf: Optional[Any] = None
    cg_x: Optional[Any] = None
    cg_r: Optional[Any] = None
    cg_p: Optional[Any] = None
    cg_ap: Optional[Any] = None
    cg_tmp: Optional[Any] = None
    cg_z: Optional[Any] = None
    precond_proj: Optional[Any] = None
    precond_scaled_proj: Optional[Any] = None
    precond_tmp: Optional[Any] = None


def ensure_gpu_data_context(
    backend: GPUBackendBundle,
    x: np.ndarray,
    y: np.ndarray,
    state: Optional[PrecomputeState] = None,
) -> GPUDataContext:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2:
        raise ValueError("x must have shape (N, d).")
    if y.ndim != 1:
        raise ValueError("y must have shape (N,).")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same leading dimension.")

    xp = backend.xp
    x_gpu = xp.asarray(x)
    y_gpu = xp.asarray(y)

    ctx = GPUDataContext(x_gpu=x_gpu, y_gpu=y_gpu)

    if state is not None:
        if int(state.weights.ndim) != int(x.shape[1]):
            raise ValueError("state.weights.ndim must match spatial dimension d = x.shape[1].")

        w = xp.asarray(state.weights)
        ctx.weights_gpu_nd = w
        ctx.weights_gpu_flat = w.reshape(-1)
        ctx.weights_np_flat = np.ascontiguousarray(
            np.asarray(state.weights, dtype=np.float64).reshape(-1)
        )
        ctx.rhs_gpu = xp.asarray(state.rhs)
        ctx.gf_gpu = xp.asarray(state.Gf)
        try:
            ctx.xtxcol_gpu = xp.ascontiguousarray(backend.fft.ifftn(ctx.gf_gpu))
        except Exception:
            ctx.xtxcol_gpu = None
        ctx.x_center_gpu = xp.asarray(state.x_center, dtype=float)

        ctx.meta.update(
            {
                "mtot": int(state.grid.mtot),
                "dim": int(state.weights.ndim),
                "h": float(state.grid.h),
                "weight_shape": tuple(int(s) for s in state.weights.shape),
                "gf_shape": tuple(int(s) for s in state.Gf.shape),
                "rhs_shape": tuple(int(s) for s in np.shape(state.rhs)),
            }
        )

    return ctx
