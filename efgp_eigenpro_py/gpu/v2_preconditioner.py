from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .backends import GPUBackendBundle
from .contexts import GPUOperatorContext


@dataclass
class GPUPreconditionerData:
    """
    GPU resident preconditioner data for V2.
    """

    U_gpu: Any
    UH_gpu: Any
    scale_gpu: Any
    scale_col_gpu: Any


def build_gpu_preconditioner_data(
    backend: GPUBackendBundle,
    U_cpu: Any,
    scale_cpu: Any,
) -> GPUPreconditionerData:
    xp = backend.xp
    U = xp.ascontiguousarray(xp.asarray(U_cpu))
    UH = xp.ascontiguousarray(U.conj().T)
    scale = xp.ascontiguousarray(xp.asarray(scale_cpu).reshape(-1))
    return GPUPreconditionerData(
        U_gpu=U,
        UH_gpu=UH,
        scale_gpu=scale,
        scale_col_gpu=scale.reshape(-1, 1),
    )


def apply_preconditioner_v2(
    backend: GPUBackendBundle,
    precond_data: GPUPreconditionerData,
    v_gpu: Any,
    op_ctx: Optional[GPUOperatorContext] = None,
    out: Optional[Any] = None,
) -> Any:
    """
    V2 hook for P(v) = v - U((scale) .* (U^* v)).
    """

    xp = backend.xp
    U = precond_data.U_gpu
    UH = precond_data.UH_gpu
    scale = precond_data.scale_gpu
    scale_col = precond_data.scale_col_gpu
    v = xp.asarray(v_gpu, dtype=U.dtype)

    if U.ndim != 2:
        raise ValueError("U_gpu must be 2D.")
    if scale.ndim != 1 or scale.shape[0] != U.shape[1]:
        raise ValueError("scale shape is incompatible with U.")
    if v.ndim not in (1, 2):
        raise ValueError("v_gpu must be 1D or 2D.")
    if v.shape[0] != U.shape[0]:
        raise ValueError("Leading dimension of v must match U.shape[0].")

    proj = None
    scaled = None
    tmp = None
    if op_ctx is not None:
        proj = op_ctx.precond_proj
        scaled = op_ctx.precond_scaled_proj
        tmp = op_ctx.precond_tmp

    proj_shape = (U.shape[1],) if v.ndim == 1 else (U.shape[1], v.shape[1])
    if proj is None or proj.shape != proj_shape or proj.dtype != U.dtype:
        proj = xp.empty(proj_shape, dtype=U.dtype)
        if op_ctx is not None:
            op_ctx.precond_proj = proj
    if scaled is None or scaled.shape != proj_shape or scaled.dtype != U.dtype:
        scaled = xp.empty(proj_shape, dtype=U.dtype)
        if op_ctx is not None:
            op_ctx.precond_scaled_proj = scaled

    xp.dot(UH, v, out=proj)
    if proj.ndim == 2:
        xp.multiply(scale_col, proj, out=scaled)
    else:
        xp.multiply(scale, proj, out=scaled)

    if tmp is None or tmp.shape != v.shape or tmp.dtype != U.dtype:
        tmp = xp.empty_like(v)
        if op_ctx is not None:
            op_ctx.precond_tmp = tmp
    xp.dot(U, scaled, out=tmp)

    out_buf = out
    may_share = getattr(xp, "may_share_memory", None)
    shares_mem = False
    if out_buf is not None and callable(may_share):
        try:
            shares_mem = bool(may_share(out_buf, v))
        except Exception:
            shares_mem = False
    if out_buf is None or out_buf.shape != v.shape or out_buf.dtype != U.dtype or shares_mem:
        out_buf = xp.empty_like(v, dtype=U.dtype)
    xp.subtract(v, tmp, out=out_buf)
    return out_buf
