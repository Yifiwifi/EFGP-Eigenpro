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


@dataclass
class CoordinateNystromPreconditionerData:
    """
    Compact coordinate Nyström preconditioner data.

    P(z) = z - I_S V diag(1 - mu / theta) V^* z_S.
    """

    S_gpu: Any
    V_gpu: Any
    VH_gpu: Any
    alpha_gpu: Any
    alpha_col_gpu: Any
    gamma: float = 1.0
    # Optional Jacobi/diagonal scaling: apply as D^{-1/2} P(D^{-1/2} v)
    # where diag_inv_sqrt_gpu stores D^{-1/2} (shape (M,)).
    diag_inv_sqrt_gpu: Any = None


@dataclass
class EnsembleCoordinateNystromPreconditionerData:
    """
    Ensemble compact coordinate Nyström preconditioner data.

    Applies a block-Jacobi style averaged correction

        P(z) = z - sum_l gamma_l I_{S_l} V_l diag(1 - mu_l / theta_l) V_l^* z[S_l]

    where each correction is evaluated against the original input ``z`` (not
    sequentially composed).
    """

    entries: list[CoordinateNystromPreconditionerData]


@dataclass
class HybridToprCoordinatePreconditionerData:
    """
    Hybrid preconditioner: apply a dense top-r dominant-subspace correction,
    then a compact coordinate Nyström correction for the remaining tail.
    """

    dense: GPUPreconditionerData
    tail: CoordinateNystromPreconditionerData


@dataclass
class GPUDominantSubspacePreconditionerData:
    """
    GPU resident dominant-subspace preconditioner data.
    """

    U_gpu: Any
    UH_gpu: Any
    inv_shift_gpu: Any
    inv_shift_col_gpu: Any


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


def build_coordinate_nystrom_preconditioner_data(
    backend: GPUBackendBundle,
    S: Any,
    V: Any,
    theta: Any,
    mu: float,
    *,
    gamma: float = 1.0,
    diag_inv_sqrt_gpu: Any = None,
    theta_floor: Optional[float] = None,
    theta_floor_ratio: float = 1e-12,
) -> CoordinateNystromPreconditionerData:
    xp = backend.xp
    S_gpu = xp.ascontiguousarray(xp.asarray(S, dtype=xp.int64).reshape(-1))
    V_gpu = xp.ascontiguousarray(xp.asarray(V, dtype=xp.complex128))
    theta_gpu = xp.ascontiguousarray(xp.real(xp.asarray(theta, dtype=xp.float64).reshape(-1)))
    if V_gpu.ndim != 2:
        raise ValueError("V must be 2D.")
    if theta_gpu.ndim != 1:
        raise ValueError("theta must be 1D.")
    if int(V_gpu.shape[0]) != int(S_gpu.size):
        raise ValueError("V rows must match S length.")
    if int(V_gpu.shape[1]) != int(theta_gpu.size):
        raise ValueError("theta length must match V columns.")
    theta_max = float(xp.max(theta_gpu)) if theta_gpu.size else 0.0
    floor = float(theta_floor) if theta_floor is not None else 0.0
    floor = max(floor, float(theta_floor_ratio) * max(theta_max, 0.0))
    theta_safe = xp.maximum(theta_gpu, floor)
    alpha = xp.ascontiguousarray(1.0 - (float(mu) / theta_safe))
    gg = float(gamma)
    if not (0.0 < gg <= 1.0):
        raise ValueError(f"gamma must satisfy 0 < gamma <= 1, got {gamma!r}")
    return CoordinateNystromPreconditionerData(
        S_gpu=S_gpu,
        V_gpu=V_gpu,
        VH_gpu=xp.ascontiguousarray(V_gpu.conj().T),
        alpha_gpu=alpha,
        alpha_col_gpu=alpha.reshape(-1, 1),
        gamma=gg,
        diag_inv_sqrt_gpu=None if diag_inv_sqrt_gpu is None else xp.asarray(diag_inv_sqrt_gpu),
    )


def build_ensemble_coordinate_nystrom_preconditioner_data(
    backend: GPUBackendBundle,
    entries: list[dict[str, Any]],
    *,
    gamma: float = 1.0,
) -> EnsembleCoordinateNystromPreconditionerData:
    if len(entries) == 0:
        raise ValueError("entries must be non-empty for ensemble_coordinate_nystrom.")
    gg = float(gamma)
    if not (0.0 < gg <= 1.0):
        raise ValueError(f"gamma must satisfy 0 < gamma <= 1, got {gamma!r}")
    per_entry_gamma = gg / float(len(entries))
    built = [
        build_coordinate_nystrom_preconditioner_data(
            backend,
            entry["S_gpu"],
            entry["V_gpu"],
            entry["theta_gpu"],
            float(entry["mu"]),
            gamma=per_entry_gamma,
        )
        for entry in entries
    ]
    return EnsembleCoordinateNystromPreconditionerData(entries=built)


def build_gpu_dominant_subspace_data(
    backend: GPUBackendBundle,
    U_cpu: Any,
    theta_cpu: Any,
    *,
    theta_floor: Optional[float] = None,
    theta_floor_ratio: float = 1e-12,
) -> GPUDominantSubspacePreconditionerData:
    xp = backend.xp
    U = xp.ascontiguousarray(xp.asarray(U_cpu))
    UH = xp.ascontiguousarray(U.conj().T)
    theta = xp.ascontiguousarray(xp.asarray(theta_cpu).reshape(-1))
    if U.ndim != 2:
        raise ValueError("U_cpu must be 2D.")
    if theta.ndim != 1:
        raise ValueError("theta_cpu must be 1D.")
    if theta.shape[0] != U.shape[1]:
        raise ValueError("theta_cpu length must match U columns.")
    theta_real = xp.real(theta)
    theta_max = float(xp.max(theta_real)) if theta_real.size else 0.0
    floor = float(theta_floor) if theta_floor is not None else 0.0
    floor = max(floor, float(theta_floor_ratio) * max(theta_max, 0.0))
    theta_safe = xp.maximum(theta_real, floor)
    inv_shift = xp.ascontiguousarray((1.0 / theta_safe) - 1.0)
    return GPUDominantSubspacePreconditionerData(
        U_gpu=U,
        UH_gpu=UH,
        inv_shift_gpu=inv_shift,
        inv_shift_col_gpu=inv_shift.reshape(-1, 1),
    )


def build_dominant_subspace_preconditioner(
    backend: GPUBackendBundle,
    apply_A_block: Any,
    size: int,
    sigma2: float,
    q: int,
    *,
    s: int = 8,
    kmax: int = 2,
    keep_factor: float = 5.0,
    keep_mode: str = "shifted",
    keep_ratio: float = 0.02,
    dtype: Optional[Any] = None,
    seed: Optional[int] = 0,
    theta_floor: Optional[float] = None,
    theta_floor_ratio: float = 1e-12,
) -> tuple[GPUDominantSubspacePreconditionerData, dict[str, Any]]:
    xp = backend.xp
    size = int(size)
    q = int(q)
    s = int(s)
    kmax = int(kmax)
    if size <= 0:
        raise ValueError("size must be positive.")
    if q < 1 or q >= size:
        raise ValueError("q must satisfy 1 <= q < size.")
    if s < 0:
        raise ValueError("s must be >= 0.")
    if kmax < 0:
        raise ValueError("kmax must be >= 0.")
    p = q + s
    if p >= size:
        raise ValueError("q + s must be < size.")
    sigma2 = float(sigma2)
    keep_factor = float(keep_factor)

    if keep_mode not in ("absolute", "shifted", "relative"):
        raise ValueError("keep_mode must be one of: absolute, shifted, relative.")
    dtype = xp.dtype(dtype) if dtype is not None else None
    if dtype is not None and not (
        xp.issubdtype(dtype, xp.floating) or xp.issubdtype(dtype, xp.complexfloating)
    ):
        raise ValueError("dtype must be a float or complex dtype.")

    rng = None
    randn = None
    if seed is not None:
        try:
            rng = xp.random.RandomState(int(seed))
        except Exception:
            rng = None
    if rng is not None:
        randn = rng.standard_normal
    else:
        randn = xp.random.standard_normal

    rand_dtype = xp.float64
    is_complex = False
    if dtype is not None:
        is_complex = bool(xp.issubdtype(dtype, xp.complexfloating))
        if is_complex:
            rand_dtype = xp.float32 if dtype == xp.dtype(xp.complex64) else xp.float64
        else:
            rand_dtype = dtype

    Omega = randn((size, p)).astype(rand_dtype)
    if is_complex:
        Omega = Omega + 1j * randn((size, p)).astype(rand_dtype)
        Omega = Omega.astype(dtype)
    elif dtype is not None:
        Omega = Omega.astype(dtype)
    Q, _ = xp.linalg.qr(Omega)
    Q = xp.ascontiguousarray(Q)

    def _validate_block(block: Any, ref: Any) -> None:
        if block.shape != ref.shape:
            raise ValueError("apply_A_block returned shape mismatch.")

    def _resolve_dtype(block: Any) -> Any:
        nonlocal dtype
        if dtype is None:
            dtype = block.dtype
        if block.dtype != dtype:
            block = block.astype(dtype)
        return xp.ascontiguousarray(block)

    for _ in range(kmax):
        Y = apply_A_block(Q)
        _validate_block(Y, Q)
        Y = _resolve_dtype(Y)
        Q = _resolve_dtype(Q)
        Q, _ = xp.linalg.qr(Y)
        Q = xp.ascontiguousarray(Q)

    Y = apply_A_block(Q)
    _validate_block(Y, Q)
    Y = _resolve_dtype(Y)
    Q = _resolve_dtype(Q)
    B = Q.conj().T @ Y
    B = 0.5 * (B + B.conj().T)
    theta, S = xp.linalg.eigh(B)
    order = xp.argsort(theta)[::-1]
    theta = theta[order]
    S = S[:, order]
    U = xp.ascontiguousarray(Q @ S)

    theta = xp.real(theta)
    theta_full = theta
    theta_max = float(xp.max(theta_full)) if theta_full.size else float("nan")
    theta_shift = theta_full - sigma2
    if keep_mode == "relative":
        denom = float(theta_shift[0]) if theta_shift.size else 0.0
        denom = max(denom, 1e-30)
        keep = (theta_shift > 0) & ((theta_shift / denom) > keep_ratio)
    elif keep_mode == "shifted":
        keep = theta_shift > keep_factor * sigma2
    else:
        keep = theta > keep_factor * sigma2
    kept_rank = int(keep.sum()) if getattr(keep, "size", 0) else 0
    if kept_rank > 0:
        U = xp.ascontiguousarray(U[:, keep])
        theta = theta_full[keep]
    else:
        U = xp.zeros((size, 0), dtype=U.dtype)
        theta = xp.asarray([], dtype=theta_full.dtype)

    theta_min_kept = float(theta[-1]) if theta.size else float("nan")
    theta_first_dropped = float(theta_full[kept_rank]) if kept_rank < theta_full.size else float("nan")
    theta_floor_base = sigma2 if theta_floor is None else float(theta_floor)
    theta_max_used = float(theta[0]) if theta.size else 0.0
    theta_floor_used = max(theta_floor_base, float(theta_floor_ratio) * max(theta_max_used, 0.0))
    data = build_gpu_dominant_subspace_data(
        backend,
        U,
        theta,
        theta_floor=theta_floor_base,
        theta_floor_ratio=theta_floor_ratio,
    )
    diag = {
        "p": int(p),
        "q": int(q),
        "s": int(s),
        "kmax": int(kmax),
        "keep_factor": float(keep_factor),
        "keep_mode": keep_mode,
        "keep_ratio": float(keep_ratio),
        "kept_rank": int(kept_rank),
        "theta_max": theta_max,
        "theta_min_kept": theta_min_kept,
        "theta_first_dropped": theta_first_dropped,
        "theta_floor": float(theta_floor_used),
        "theta_floor_ratio": float(theta_floor_ratio),
        "dtype": str(dtype),
        "seed": seed,
    }
    return data, diag


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
    v = xp.asarray(v_gpu)

    if U.ndim != 2:
        raise ValueError("U_gpu must be 2D.")
    if scale.ndim != 1 or scale.shape[0] != U.shape[1]:
        raise ValueError("scale shape is incompatible with U.")
    if v.ndim not in (1, 2):
        raise ValueError("v_gpu must be 1D or 2D.")
    if v.shape[0] != U.shape[0]:
        # Compatibility: some call sites may accidentally store U as (q, n) instead of (n, q).
        # If dimensions are unambiguous, transpose once here instead of hard-failing.
        if (
            v.shape[0] == U.shape[1]
            and scale.shape[0] == U.shape[0]
            and UH.shape == (U.shape[1], U.shape[0])
        ):
            U = xp.ascontiguousarray(U.conj().T)
            UH = xp.ascontiguousarray(U.conj().T)
            scale_col = scale.reshape(-1, 1)
        else:
            raise ValueError("Leading dimension of v must match U.shape[0].")
    v = xp.asarray(v, dtype=U.dtype)

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


def apply_preconditioner_coordinate_nystrom(
    backend: GPUBackendBundle,
    precond_data: CoordinateNystromPreconditionerData,
    v_gpu: Any,
    op_ctx: Optional[GPUOperatorContext] = None,
    out: Optional[Any] = None,
) -> Any:
    """
    Compact coordinate Nyström preconditioner:
    P(v) = v - I_S V((alpha) .* (V^* v[S])).
    """

    del op_ctx
    xp = backend.xp
    S = precond_data.S_gpu
    V = precond_data.V_gpu
    VH = precond_data.VH_gpu
    alpha = precond_data.alpha_gpu
    alpha_col = precond_data.alpha_col_gpu
    gamma = float(getattr(precond_data, "gamma", 1.0))
    diag_inv_sqrt = getattr(precond_data, "diag_inv_sqrt_gpu", None)
    v = xp.asarray(v_gpu, dtype=V.dtype)

    if V.ndim != 2:
        raise ValueError("V_gpu must be 2D.")
    if alpha.ndim != 1 or alpha.shape[0] != V.shape[1]:
        raise ValueError("alpha shape is incompatible with V.")
    if v.ndim not in (1, 2):
        raise ValueError("v_gpu must be 1D or 2D.")
    if int(S.size) != int(V.shape[0]):
        raise ValueError("S length must match V rows.")

    def _core_apply(v_in: Any, out_in: Optional[Any]) -> Any:
        out_buf = out_in
        may_share = getattr(xp, "may_share_memory", None)
        shares_mem = False
        if out_buf is not None and callable(may_share):
            try:
                shares_mem = bool(may_share(out_buf, v_in))
            except Exception:
                shares_mem = False
        if (
            out_buf is None
            or out_buf.shape != v_in.shape
            or out_buf.dtype != V.dtype
            or shares_mem
        ):
            out_buf = xp.empty_like(v_in, dtype=V.dtype)
        xp.copyto(out_buf, v_in)

        z_s = v_in[S] if v_in.ndim == 1 else v_in[S, :]
        coeff = VH @ z_s
        coeff = alpha * coeff if coeff.ndim == 1 else alpha_col * coeff
        corr_s = V @ coeff
        if gamma != 1.0:
            corr_s = gamma * corr_s
        if out_buf.ndim == 1:
            out_buf[S] -= corr_s
        else:
            out_buf[S, :] -= corr_s
        return out_buf

    # Optional Jacobi-style scaling wrapper: D^{-1/2} P(D^{-1/2} v).
    if diag_inv_sqrt is not None:
        d = xp.asarray(diag_inv_sqrt)
        if d.ndim != 1 or int(d.shape[0]) != int(v.shape[0]):
            raise ValueError("diag_inv_sqrt_gpu must have shape (M,) matching v.")
        d = d.astype(v.dtype, copy=False)
        v_scaled = d * v if v.ndim == 1 else d.reshape(-1, 1) * v
        tmp = _core_apply(v_scaled, None)
        out_buf = out
        may_share = getattr(xp, "may_share_memory", None)
        shares_mem = False
        if out_buf is not None and callable(may_share):
            try:
                shares_mem = bool(may_share(out_buf, tmp))
            except Exception:
                shares_mem = False
        if (
            out_buf is None
            or out_buf.shape != tmp.shape
            or out_buf.dtype != tmp.dtype
            or shares_mem
        ):
            out_buf = xp.empty_like(tmp, dtype=tmp.dtype)
        if tmp.ndim == 1:
            xp.multiply(d, tmp, out=out_buf)
        else:
            xp.multiply(d.reshape(-1, 1), tmp, out=out_buf)
        return out_buf

    return _core_apply(v, out)


def apply_preconditioner_ensemble_coordinate_nystrom(
    backend: GPUBackendBundle,
    precond_data: EnsembleCoordinateNystromPreconditionerData,
    v_gpu: Any,
    op_ctx: Optional[GPUOperatorContext] = None,
    out: Optional[Any] = None,
) -> Any:
    """
    Ensemble compact coordinate Nyström preconditioner.

    Each ensemble correction is computed from the same original input ``v`` and
    accumulated additively, matching the intended averaged block-Jacobi update.
    """

    del op_ctx
    xp = backend.xp
    entries = list(precond_data.entries or [])
    if len(entries) == 0:
        raise ValueError("ensemble preconditioner must contain at least one entry.")

    first = entries[0]
    v = xp.asarray(v_gpu, dtype=first.V_gpu.dtype)
    out_buf = out
    may_share = getattr(xp, "may_share_memory", None)
    shares_mem = False
    if out_buf is not None and callable(may_share):
        try:
            shares_mem = bool(may_share(out_buf, v))
        except Exception:
            shares_mem = False
    if out_buf is None or out_buf.shape != v.shape or out_buf.dtype != v.dtype or shares_mem:
        out_buf = xp.empty_like(v, dtype=v.dtype)
    xp.copyto(out_buf, v)

    for entry in entries:
        S = entry.S_gpu
        V = entry.V_gpu
        VH = entry.VH_gpu
        alpha = entry.alpha_gpu
        alpha_col = entry.alpha_col_gpu
        gamma = float(getattr(entry, "gamma", 1.0))

        if v.ndim == 1:
            z_s = v[S]
            coeff = VH @ z_s
            coeff = alpha * coeff
            corr_s = V @ coeff
            if gamma != 1.0:
                corr_s = gamma * corr_s
            out_buf[S] -= corr_s
        else:
            z_s = v[S, :]
            coeff = VH @ z_s
            coeff = alpha_col * coeff
            corr_s = V @ coeff
            if gamma != 1.0:
                corr_s = gamma * corr_s
            out_buf[S, :] -= corr_s
    return out_buf


def apply_preconditioner_hybrid_topr_coordinate(
    backend: GPUBackendBundle,
    precond_data: HybridToprCoordinatePreconditionerData,
    v_gpu: Any,
    op_ctx: Optional[GPUOperatorContext] = None,
    out: Optional[Any] = None,
) -> Any:
    """
    Hybrid apply: first dense dominant-subspace correction, then compact coordinate correction.
    """
    tmp = apply_preconditioner_v2(backend, precond_data.dense, v_gpu, op_ctx=op_ctx, out=None)
    return apply_preconditioner_coordinate_nystrom(
        backend, precond_data.tail, tmp, op_ctx=op_ctx, out=out
    )


def apply_preconditioner_dominant_subspace(
    backend: GPUBackendBundle,
    precond_data: GPUDominantSubspacePreconditionerData,
    v_gpu: Any,
    op_ctx: Optional[GPUOperatorContext] = None,
    out: Optional[Any] = None,
) -> Any:
    """
    Dominant-subspace preconditioner: P^{-1}(v) = v + U((inv_shift) .* (U^* v)).
    """

    xp = backend.xp
    U = precond_data.U_gpu
    UH = precond_data.UH_gpu
    inv_shift = precond_data.inv_shift_gpu
    inv_shift_col = precond_data.inv_shift_col_gpu
    v = xp.asarray(v_gpu, dtype=U.dtype)

    if U.ndim != 2:
        raise ValueError("U_gpu must be 2D.")
    if inv_shift.ndim != 1 or inv_shift.shape[0] != U.shape[1]:
        raise ValueError("inv_shift shape is incompatible with U.")
    if v.ndim not in (1, 2):
        raise ValueError("v_gpu must be 1D or 2D.")
    if v.shape[0] != U.shape[0]:
        raise ValueError("Leading dimension of v must match U.shape[0].")

    if U.shape[1] == 0:
        out_buf = out
        if out_buf is None or out_buf.shape != v.shape or out_buf.dtype != U.dtype:
            out_buf = xp.empty_like(v, dtype=U.dtype)
        xp.copyto(out_buf, v)
        return out_buf

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
        xp.multiply(inv_shift_col, proj, out=scaled)
    else:
        xp.multiply(inv_shift, proj, out=scaled)

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
    xp.add(v, tmp, out=out_buf)
    return out_buf
