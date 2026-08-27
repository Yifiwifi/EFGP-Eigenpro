from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .active_set import (
    BoxActiveSet,
    build_box_active_set,
    validate_precomputed_active_set,
)
from .config import BTABConfig
from ..contexts import GPUOperatorContext
from ..iterative_solvers import pcg_solve_gpu
from ..v3_eigenspace import _toeplitz_submatrix_gpu


@dataclass
class BoxToeplitzPreconditionerData:
    active: BoxActiveSet
    solve_mode: str
    exact_apply_mode: str
    box_idx_gpu: Any
    tail_idx_gpu: Any
    box_inverse_gpu: Any | None
    diag_inv_full_gpu: Any
    diag_inv_tail_gpu: Any
    diag_inv_box_gpu: Any | None
    chol_factor_gpu: Any | None
    box_matrix_gpu: Any | None
    box_weights_gpu: Any | None
    global_shape: tuple[int, ...]
    box_slices: tuple[slice, ...]
    local_gf_gpu: Any | None
    box_shape: tuple[int, ...]
    lag_shape: tuple[int, ...]
    local_op_ctx: Any | None
    runtime_stats: dict[str, Any]
    diag_A_gpu: Any
    diagnostics: dict[str, Any]


def _gamma_from_xtxcol(xp: Any, xtxcol_gpu: Any, mtot: int, dim: int) -> float:
    center = (int(mtot) - 1,) * int(dim)
    val = xp.real(xtxcol_gpu[center])
    return float(val.item() if hasattr(val, "item") else val)


def _diag_A_gpu(xp: Any, gamma: float, weights_gpu: Any, reg_lambda: float) -> Any:
    diag = (xp.asarray(float(gamma), dtype=xp.float64) * (xp.abs(weights_gpu) ** 2)) + float(reg_lambda)
    return xp.maximum(diag, xp.asarray(1e-30, dtype=xp.float64))


def _triangular_solve(backend: Any, a: Any, b: Any, *, lower: bool) -> Any:
    xp = backend.xp
    if getattr(xp, "__name__", "") == "cupy":
        from cupyx.scipy.linalg import solve_triangular

        return solve_triangular(a, b, lower=lower, check_finite=False)

    from scipy.linalg import solve_triangular

    a_np = np.asarray(a)
    b_np = np.asarray(b)
    x_np = solve_triangular(a_np, b_np, lower=lower, check_finite=False)
    return xp.asarray(x_np)


def _sync_backend(backend: Any) -> None:
    xp = backend.xp
    cuda = getattr(xp, "cuda", None)
    if cuda is not None:
        cuda.Stream.null.synchronize()


def _now_synced(backend: Any) -> float:
    _sync_backend(backend)
    return time.perf_counter()


def _resolve_exact_apply_mode(cfg: BTABConfig) -> str:
    mode = str(getattr(cfg, "exact_apply_mode", "inverse")).strip().lower()
    if mode not in ("inverse", "chol_solve"):
        raise ValueError(
            f"unknown exact_apply_mode={cfg.exact_apply_mode!r}; expected 'inverse' or 'chol_solve'."
        )
    return mode


def _resolve_solve_mode(cfg: BTABConfig, box_size: int) -> str:
    mode = str(cfg.solve_mode).strip().lower()
    if mode not in ("auto", "exact", "inner_pcg"):
        raise ValueError(f"unknown solve_mode={cfg.solve_mode!r}; expected 'auto', 'exact', or 'inner_pcg'.")
    if mode == "auto":
        if cfg.exact_box_max_size is None:
            return "exact"
        return "exact" if int(box_size) <= int(cfg.exact_box_max_size) else "inner_pcg"
    if mode == "exact" and cfg.exact_box_max_size is not None and int(box_size) > int(cfg.exact_box_max_size):
        raise ValueError(
            "box size exceeds exact_box_max_size for solve_mode='exact': "
            f"|S_g|={int(box_size)} > {int(cfg.exact_box_max_size)}"
        )
    return mode


def _build_local_fft_tensor(
    backend: Any,
    xtxcol_gpu: Any,
    active: BoxActiveSet,
    mtot: int,
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
    xp = backend.xp
    box_shape = tuple(int(2 * r + 1) for r in np.asarray(active.radii).reshape(-1))
    lag_shape = tuple(int(2 * q - 1) for q in box_shape)
    center = int(mtot) - 1
    lag_slices = tuple(
        slice(center - (q - 1), center + (q - 1) + 1)
        for q in box_shape
    )
    local_xtxcol_gpu = xp.ascontiguousarray(xtxcol_gpu[lag_slices])
    local_gf_gpu = xp.ascontiguousarray(backend.fft.fftn(local_xtxcol_gpu))
    return local_gf_gpu, box_shape, lag_shape


def _ensure_local_buffer(local_op_ctx: Any, xp: Any, name: str, shape: tuple[int, ...], dtype: Any) -> Any:
    buf = getattr(local_op_ctx, name, None)
    if buf is None or tuple(getattr(buf, "shape", ())) != tuple(shape) or buf.dtype != dtype:
        buf = xp.empty(shape, dtype=dtype)
        setattr(local_op_ctx, name, buf)
    return buf


def _apply_local_box_operator(
    backend: Any,
    precond_data: BoxToeplitzPreconditionerData,
    reg_lambda: float,
    v_box: Any,
    *,
    out: Any | None = None,
) -> Any:
    xp = backend.xp
    box_shape = tuple(int(s) for s in precond_data.box_shape)
    lag_shape = tuple(int(s) for s in precond_data.lag_shape)
    n_box = int(np.prod(box_shape, dtype=np.int64))
    vin = xp.asarray(v_box, dtype=xp.complex128).reshape(-1)
    if int(vin.size) != n_box:
        raise ValueError(f"local box vector has size {int(vin.size)}; expected {n_box}.")
    out_arr = xp.empty_like(vin) if out is None else out
    weights_flat = xp.asarray(precond_data.box_weights_gpu, dtype=xp.float64).reshape(-1)
    local_op_ctx = precond_data.local_op_ctx

    weighted = _ensure_local_buffer(local_op_ctx, xp, "box_weighted", (n_box,), xp.complex128)
    pad = _ensure_local_buffer(local_op_ctx, xp, "box_pad", lag_shape, precond_data.local_gf_gpu.dtype)
    tmp = _ensure_local_buffer(local_op_ctx, xp, "box_tmp", (n_box,), xp.complex128)

    xp.multiply(weights_flat, vin, out=weighted)
    pad.fill(0)
    pad[tuple(slice(0, q) for q in box_shape)] = weighted.reshape(box_shape)
    af = backend.fft.fftn(pad)
    ypad = backend.fft.ifftn(af * precond_data.local_gf_gpu)
    slicer = tuple(slice(q - 1, 2 * q - 1) for q in box_shape)
    xp.copyto(tmp, ypad[slicer].reshape(-1))
    xp.multiply(weights_flat, tmp, out=out_arr)
    out_arr += float(reg_lambda) * vin
    return out_arr


def _apply_inner_precond(
    backend: Any,
    precond_data: BoxToeplitzPreconditionerData,
    v_box: Any,
    *,
    out: Any,
) -> Any:
    xp = backend.xp
    mode = str(precond_data.diagnostics.get("inner_precond", "diag")).lower()
    if mode == "identity":
        xp.copyto(out, xp.asarray(v_box, dtype=out.dtype).reshape(-1))
        return out
    diag_inv = xp.asarray(precond_data.diag_inv_box_gpu, dtype=out.dtype).reshape(-1)
    xp.multiply(diag_inv, xp.asarray(v_box, dtype=out.dtype).reshape(-1), out=out)
    return out


def build_box_toeplitz_preconditioner(
    backend: Any,
    data_ctx: Any,
    reg_lambda: float,
    cfg: BTABConfig,
    *,
    profile_apply_components: bool = True,
    precomputed_active_set: BoxActiveSet | None = None,
) -> BoxToeplitzPreconditionerData:
    xp = backend.xp
    if data_ctx.xtxcol_gpu is None:
        data_ctx.xtxcol_gpu = xp.ascontiguousarray(backend.fft.ifftn(data_ctx.gf_gpu))
    weights_gpu = xp.asarray(data_ctx.weights_gpu_flat, dtype=xp.float64).reshape(-1)
    weights_np = getattr(data_ctx, "weights_np_flat", None)
    if weights_np is None:
        weights_np = np.asarray(weights_gpu.get() if hasattr(weights_gpu, "get") else weights_gpu, dtype=np.float64)
        data_ctx.weights_np_flat = weights_np
    mtot = int(data_ctx.meta["mtot"])
    dim = int(data_ctx.meta["dim"])
    global_shape = (int(mtot),) * int(dim)

    t0 = _now_synced(backend)
    if precomputed_active_set is None:
        gamma = _gamma_from_xtxcol(xp, data_ctx.xtxcol_gpu, mtot, dim)
        active = build_box_active_set(
            gamma=gamma,
            weights=weights_np,
            reg_lambda=reg_lambda,
            mtot=mtot,
            dim=dim,
            active_mode=cfg.active_mode,
            active_topk=cfg.active_topk,
            active_tau=cfg.active_tau,
            box_budget=cfg.box_budget,
        )
        t1 = _now_synced(backend)
    else:
        active = precomputed_active_set
        gamma = _gamma_from_xtxcol(xp, data_ctx.xtxcol_gpu, mtot, dim)
        validate_precomputed_active_set(
            active,
            gamma=gamma,
            weights=weights_np,
            reg_lambda=reg_lambda,
            mtot=mtot,
            dim=dim,
            active_mode=cfg.active_mode,
            active_topk=cfg.active_topk,
            active_tau=cfg.active_tau,
            box_budget=cfg.box_budget,
        )
        # Selection was timed by the controlled runner and its result is being
        # consumed directly; no second score sort/box construction occurs here.
        t1 = t0

    diag_A = _diag_A_gpu(xp, gamma, weights_gpu, reg_lambda)
    diag_floor = xp.asarray(float(cfg.diag_floor), dtype=xp.float64)
    diag_inv_full_gpu = 1.0 / xp.maximum(diag_A, diag_floor)
    box_idx_gpu = xp.asarray(active.box_idx, dtype=xp.int64)
    tail_idx_gpu = xp.asarray(active.tail_idx, dtype=xp.int64)
    solve_mode = _resolve_solve_mode(cfg, int(active.box_idx.size))
    exact_apply_mode = _resolve_exact_apply_mode(cfg)
    box_diag = diag_A[box_idx_gpu] if int(active.box_idx.size) else xp.empty((0,), dtype=xp.float64)
    diag_inv_box_gpu = (
        1.0 / xp.maximum(box_diag, diag_floor)
        if int(active.box_idx.size)
        else xp.empty((0,), dtype=xp.float64)
    )
    center_multi = np.asarray(active.center_multi, dtype=np.int64).reshape(-1)
    radii = np.asarray(active.radii, dtype=np.int64).reshape(-1)
    box_slices = tuple(
        slice(int(center_multi[ax] - radii[ax]), int(center_multi[ax] + radii[ax] + 1))
        for ax in range(int(dim))
    )

    t2 = _now_synced(backend)
    box_matrix_gpu = None
    box_inverse_gpu = None
    chol_factor_gpu = None
    local_gf_gpu = None
    box_weights_gpu = None
    box_shape = tuple(int(2 * r + 1) for r in np.asarray(active.radii).reshape(-1))
    lag_shape = tuple(int(2 * q - 1) for q in box_shape)
    local_op_ctx = GPUOperatorContext()
    t_mat = t2
    t_sym = t2
    t_chol = t2
    t_inv = t2
    t_local = t2
    if int(active.box_idx.size) > 0 and solve_mode == "exact":
        box_matrix_gpu = _toeplitz_submatrix_gpu(
            xp,
            data_ctx.xtxcol_gpu,
            weights_gpu,
            active.box_idx,
            mtot=mtot,
            dim=dim,
        )
        t_mat = _now_synced(backend)
        eye = xp.eye(int(active.box_idx.size), dtype=box_matrix_gpu.dtype)
        box_matrix_gpu = 0.5 * (box_matrix_gpu + box_matrix_gpu.conj().T) + float(reg_lambda) * eye
        if float(cfg.chol_jitter) > 0.0:
            box_matrix_gpu = box_matrix_gpu + float(cfg.chol_jitter) * eye
        t_sym = _now_synced(backend)
        if exact_apply_mode == "chol_solve":
            chol_factor_gpu = xp.linalg.cholesky(box_matrix_gpu)
            t_chol = _now_synced(backend)
        else:
            chol_factor_gpu = xp.linalg.cholesky(box_matrix_gpu)
            t_chol = _now_synced(backend)
            ident = xp.eye(int(active.box_idx.size), dtype=box_matrix_gpu.dtype)
            y_inv = _triangular_solve(backend, chol_factor_gpu, ident, lower=True)
            box_inverse_gpu = _triangular_solve(backend, chol_factor_gpu.conj().T, y_inv, lower=False)
            t_inv = _now_synced(backend)
            chol_factor_gpu = None
        if not bool(cfg.keep_box_matrix) and exact_apply_mode != "chol_solve":
            box_matrix_gpu = None
        elif not bool(cfg.keep_box_matrix) and exact_apply_mode == "chol_solve":
            box_matrix_gpu = None
        t3 = max(t_chol, t_inv)
    elif int(active.box_idx.size) > 0 and solve_mode == "inner_pcg":
        local_gf_gpu, box_shape, lag_shape = _build_local_fft_tensor(
            backend,
            data_ctx.xtxcol_gpu,
            active,
            mtot,
        )
        box_weights_gpu = xp.ascontiguousarray(weights_gpu[box_idx_gpu].reshape(box_shape))
        t_local = _now_synced(backend)
        t3 = t_local
    else:
        t3 = _now_synced(backend)

    diag_tail = diag_A[tail_idx_gpu] if int(active.tail_idx.size) else xp.empty((0,), dtype=xp.float64)
    diag_inv_tail_gpu = 1.0 / xp.maximum(diag_tail, diag_floor)
    runtime_stats = {
        "precond_apply_calls": 0,
        "inner_total_iters": 0,
        "inner_total_matvec": 0,
        "inner_total_precond": 0,
        "inner_last_relres": float("nan"),
        "inner_max_iters": 0,
        "time_diag_total": 0.0,
        "time_box_gather_total": 0.0,
        "time_box_solve_total": 0.0,
        "time_box_scatter_total": 0.0,
        "profile_apply_components": bool(profile_apply_components),
    }

    diagnostics = {
        "reg_lambda": float(reg_lambda),
        "gamma": float(gamma),
        "solve_mode": solve_mode,
        "exact_apply_mode": exact_apply_mode,
        "inner_tol": float(cfg.inner_tol),
        "inner_maxiter": int(cfg.inner_maxiter),
        "inner_precond": str(cfg.inner_precond),
        "active_mode": str(active.active_mode),
        "active_topk": None if active.active_topk is None else int(active.active_topk),
        "active_tau": None if active.active_tau is None else float(active.active_tau),
        "active_size": int(active.active_idx.size),
        "box_size": int(active.box_idx.size),
        "tail_size": int(active.tail_idx.size),
        "box_radii": [int(v) for v in np.asarray(active.radii).reshape(-1)],
        "box_shape": [int(v) for v in box_shape],
        "time_active_set": float(t1 - t0),
        "time_build_Ag": float(t3 - t2),
        "time_build_box_matrix": float(t_mat - t2),
        "time_symmetrize_regularize": float(t_sym - t_mat),
        "time_cholesky": float(t_chol - t_sym),
        "time_build_inverse": float(t_inv - t_chol),
        "time_build_local_fft": float(t_local - t2),
        "box_memory_bytes": int(0 if box_matrix_gpu is None else box_matrix_gpu.nbytes),
        "chol_memory_bytes": int(0 if chol_factor_gpu is None else chol_factor_gpu.nbytes),
        "inverse_memory_bytes": int(0 if box_inverse_gpu is None else box_inverse_gpu.nbytes),
        "local_fft_memory_bytes": int(0 if local_gf_gpu is None else local_gf_gpu.nbytes),
    }
    return BoxToeplitzPreconditionerData(
        active=active,
        solve_mode=solve_mode,
        exact_apply_mode=exact_apply_mode,
        box_idx_gpu=box_idx_gpu,
        tail_idx_gpu=tail_idx_gpu,
        box_inverse_gpu=box_inverse_gpu,
        diag_inv_full_gpu=diag_inv_full_gpu,
        diag_inv_tail_gpu=diag_inv_tail_gpu,
        diag_inv_box_gpu=diag_inv_box_gpu,
        chol_factor_gpu=chol_factor_gpu,
        box_matrix_gpu=box_matrix_gpu,
        box_weights_gpu=box_weights_gpu,
        global_shape=global_shape,
        box_slices=box_slices,
        local_gf_gpu=local_gf_gpu,
        box_shape=box_shape,
        lag_shape=lag_shape,
        local_op_ctx=local_op_ctx,
        runtime_stats=runtime_stats,
        diag_A_gpu=diag_A,
        diagnostics=diagnostics,
    )


def apply_box_toeplitz_preconditioner(
    backend: Any,
    precond_data: BoxToeplitzPreconditionerData,
    v: Any,
    *,
    out: Any | None = None,
) -> Any:
    xp = backend.xp
    vin = xp.asarray(v, dtype=xp.complex128).reshape(-1)
    out_arr = xp.empty_like(vin) if out is None else out
    stats = precond_data.runtime_stats
    profile_apply = bool(stats.get("profile_apply_components", False))

    if profile_apply:
        t0 = _now_synced(backend)
    xp.multiply(precond_data.diag_inv_full_gpu.astype(xp.complex128, copy=False), vin, out=out_arr)
    if profile_apply:
        t1 = _now_synced(backend)
        stats["time_diag_total"] += float(t1 - t0)

    if int(precond_data.box_idx_gpu.size) > 0:
        vin_nd = vin.reshape(precond_data.global_shape)
        out_nd = out_arr.reshape(precond_data.global_shape)

        if profile_apply:
            tg0 = _now_synced(backend)
        rhs_box = xp.ascontiguousarray(vin_nd[precond_data.box_slices].reshape(-1))
        if profile_apply:
            tg1 = _now_synced(backend)
            stats["time_box_gather_total"] += float(tg1 - tg0)

        if profile_apply:
            ts0 = _now_synced(backend)
        if str(precond_data.solve_mode).lower() == "exact":
            if str(precond_data.exact_apply_mode).lower() == "inverse":
                z_box = precond_data.box_inverse_gpu @ rhs_box
            else:
                y = _triangular_solve(
                    backend,
                    precond_data.chol_factor_gpu,
                    rhs_box,
                    lower=True,
                )
                z_box = _triangular_solve(
                    backend,
                    precond_data.chol_factor_gpu.conj().T,
                    y,
                    lower=False,
                )
        else:
            local_op_ctx = precond_data.local_op_ctx

            def _matvec_local(vloc: Any, outloc: Any) -> None:
                _apply_local_box_operator(
                    backend,
                    precond_data,
                    float(precond_data.diagnostics.get("reg_lambda", 0.0)),
                    vloc,
                    out=outloc,
                )

            def _precond_local(vloc: Any, outloc: Any) -> None:
                _apply_inner_precond(backend, precond_data, vloc, out=outloc)

            z_box, it_in, rel_in, inner_stats = pcg_solve_gpu(
                backend,
                _matvec_local,
                _precond_local,
                rhs_box,
                local_op_ctx,
                float(precond_data.diagnostics.get("inner_tol", 1e-3)),
                int(precond_data.diagnostics.get("inner_maxiter", 50)),
                return_stats=True,
                work_prefix="btab_inner",
                profile_components=False,
            )
            stats["inner_total_iters"] += int(it_in)
            stats["inner_total_matvec"] += int(inner_stats.get("n_matvec", 0))
            stats["inner_total_precond"] += int(inner_stats.get("n_precond", 0))
            stats["inner_last_relres"] = float(rel_in)
            stats["inner_max_iters"] = max(
                int(stats["inner_max_iters"]),
                int(it_in),
            )
        if profile_apply:
            ts1 = _now_synced(backend)
            stats["time_box_solve_total"] += float(ts1 - ts0)

        if profile_apply:
            tw0 = _now_synced(backend)
        out_nd[precond_data.box_slices] = xp.asarray(z_box, dtype=out_arr.dtype).reshape(precond_data.box_shape)
        if profile_apply:
            tw1 = _now_synced(backend)
            stats["time_box_scatter_total"] += float(tw1 - tw0)

    stats["precond_apply_calls"] += 1
    return out_arr
