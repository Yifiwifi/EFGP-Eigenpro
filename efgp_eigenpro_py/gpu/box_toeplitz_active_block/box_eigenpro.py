from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .active_set import (
    BoxActiveSet,
    build_box_active_set,
    validate_precomputed_active_set,
)
from .config import BTABConfig
from .preconditioner import (
    _apply_local_box_operator,
    _build_local_fft_tensor,
    _diag_A_gpu,
    _gamma_from_xtxcol,
    _now_synced,
)
from ..contexts import GPUOperatorContext


@dataclass
class BoxEigenProPreconditionerData:
    active: BoxActiveSet
    solve_mode: str
    exact_apply_mode: str
    box_idx_gpu: Any
    tail_idx_gpu: Any
    diag_inv_full_gpu: Any
    diag_inv_tail_gpu: Any
    diag_inv_box_gpu: Any | None
    global_shape: tuple[int, ...]
    box_slices: tuple[slice, ...]
    local_gf_gpu: Any | None
    box_weights_gpu: Any | None
    box_shape: tuple[int, ...]
    lag_shape: tuple[int, ...]
    local_op_ctx: Any | None
    runtime_stats: dict[str, Any]
    diag_A_gpu: Any
    diagnostics: dict[str, Any]
    eig_U_gpu: Any
    eig_UH_gpu: Any
    eig_coeff_gpu: Any
    eig_coeff_col_gpu: Any
    eig_alpha: float
    eig_theta_top_gpu: Any
    eig_theta_q1: float


def _asnumpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "get"):
        return np.asarray(arr.get())
    return np.asarray(arr)


def _resolve_eig_maxiter(q: int, value: int | None) -> int:
    if value is None:
        return max(1, 5 * int(q))
    return max(1, int(value))


def _local_box_apply_block(
    backend: Any,
    precond_data: BoxEigenProPreconditionerData,
    reg_lambda: float,
    V: Any,
    *,
    counter: dict[str, int] | None = None,
) -> Any:
    xp = backend.xp
    V = xp.asarray(V, dtype=xp.complex128)
    was_1d = V.ndim == 1
    if was_1d:
        V = V.reshape(-1, 1)
    if V.ndim != 2:
        raise ValueError("V must be 1D or 2D.")
    out = xp.empty_like(V)
    for j in range(int(V.shape[1])):
        _apply_local_box_operator(
            backend,
            precond_data,
            float(reg_lambda),
            V[:, j],
            out=out[:, j],
        )
    if counter is not None:
        counter["n_eig_matvec"] = int(counter.get("n_eig_matvec", 0)) + int(V.shape[1])
    return out[:, 0] if was_1d else out


def _dense_local_eigh(
    backend: Any,
    precond_data: BoxEigenProPreconditionerData,
    reg_lambda: float,
    n_need: int,
    counter: dict[str, int],
) -> tuple[Any, Any, dict[str, Any]]:
    xp = backend.xp
    n = int(np.prod(precond_data.box_shape, dtype=np.int64))
    eye = xp.eye(n, dtype=xp.complex128)
    A = _local_box_apply_block(
        backend,
        precond_data,
        float(reg_lambda),
        eye,
        counter=counter,
    )
    A = 0.5 * (A + A.conj().T)
    vals, vecs = xp.linalg.eigh(A)
    order = xp.argsort(xp.real(vals))[::-1]
    vals = xp.real(vals[order])[:n_need]
    vecs = xp.ascontiguousarray(vecs[:, order][:, :n_need])
    return vals, vecs, {"backend": "dense"}


def _sort_and_check_eigenpairs(
    xp: Any,
    vals: Any,
    vecs: Any,
    n_need: int,
    *,
    source: str,
) -> tuple[Any, Any]:
    vals = xp.real(xp.asarray(vals, dtype=xp.float64).reshape(-1))
    vecs = xp.asarray(vecs, dtype=xp.complex128)
    if int(vals.size) < int(n_need):
        raise RuntimeError(
            f"{source} returned {int(vals.size)} eigenvalues, expected {int(n_need)}."
        )
    if vecs.ndim != 2 or int(vecs.shape[1]) < int(n_need):
        got = 0 if vecs.ndim != 2 else int(vecs.shape[1])
        raise RuntimeError(
            f"{source} returned {got} eigenvectors, expected {int(n_need)}."
        )
    order = xp.argsort(vals)[::-1]
    vals = vals[order][:n_need]
    vecs = xp.ascontiguousarray(vecs[:, order][:, :n_need])
    return vals, vecs


def _compute_local_eigenpairs(
    backend: Any,
    precond_data: BoxEigenProPreconditionerData,
    reg_lambda: float,
    q: int,
    cfg: BTABConfig,
    counter: dict[str, int],
) -> tuple[Any, Any, dict[str, Any]]:
    xp = backend.xp
    n = int(np.prod(precond_data.box_shape, dtype=np.int64))
    q = int(q)
    n_need = q + 1
    if q < 1:
        raise ValueError("btab_eig_q must be >= 1.")
    if n_need > n:
        raise ValueError(f"btab_eig_q+1={n_need} exceeds box size {n}.")
    if n_need >= n - 1:
        return _dense_local_eigh(backend, precond_data, reg_lambda, n_need, counter)

    eig_tol = float(getattr(cfg, "eig_tol", 1e-3))
    eig_maxiter = _resolve_eig_maxiter(q, getattr(cfg, "eig_maxiter", None))
    eig_ncv = getattr(cfg, "eig_ncv", None)
    eig_backend = "cupy"
    try:
        from ..cupy_eigenspace_methods import cupy_eigsh

        if xp is np or getattr(xp, "__name__", "") != "cupy":
            raise RuntimeError("cupy backend is not active")

        eig_cfg = {
            "tol": eig_tol,
            "maxiter": eig_maxiter,
            "ncv": eig_ncv,
            "which": "LA",
        }

        def _matvec_block(V: Any) -> Any:
            return _local_box_apply_block(
                backend,
                precond_data,
                float(reg_lambda),
                V,
                counter=counter,
            )

        # cupy_eigsh uses the repository convention top_q -> returns top_q + 1
        # eigenpairs.  We need nev=q+1, so top_q_for_nev=nev-1=q.
        top_q_for_nev = n_need - 1
        res = cupy_eigsh(
            None,
            n,
            top_q_for_nev,
            matvec_block=_matvec_block,
            block_size=eig_ncv,
            tol=eig_tol,
            maxiter=eig_maxiter,
            xp=xp,
            cfg=eig_cfg,
        )
        vals, vecs = _sort_and_check_eigenpairs(
            xp,
            res.values,
            res.vectors,
            n_need,
            source="cupy_eigsh",
        )
        ncv_actual = eig_ncv
        if ncv_actual is None:
            ncv_actual = min(n - 1, max(2 * n_need + 32, n_need + 2))
        return vals, vecs, {
            "backend": eig_backend,
            "ncv_actual": int(ncv_actual),
            "maxiter": int(eig_maxiter),
            "tol": float(eig_tol),
        }
    except Exception as exc:
        if (
            getattr(xp, "__name__", "") == "cupy"
            and bool(getattr(cfg, "strict_gpu_eig", False))
        ):
            raise RuntimeError(
                "strict_gpu_eig is enabled: the CuPy eigensolver failed and "
                "CPU/SciPy fallback is disabled to prevent an unbounded host "
                "computation."
            ) from exc
        eig_backend = "scipy"
        from scipy.sparse.linalg import LinearOperator, eigsh

        def _mv(v: Any) -> np.ndarray:
            vg = xp.asarray(np.asarray(v, dtype=np.complex128))
            yg = _local_box_apply_block(
                backend,
                precond_data,
                float(reg_lambda),
                vg,
                counter=counter,
            )
            return np.asarray(_asnumpy(yg), dtype=np.complex128).reshape(-1)

        op = LinearOperator((n, n), matvec=_mv, dtype=np.complex128)
        ncv_actual = eig_ncv
        vals_np, vecs_np = eigsh(
            op,
            k=n_need,
            which="LA",
            tol=eig_tol,
            maxiter=eig_maxiter,
            ncv=None if eig_ncv is None else int(eig_ncv),
            return_eigenvectors=True,
        )
        vals_np, vecs_np = _sort_and_check_eigenpairs(
            np,
            vals_np,
            vecs_np,
            n_need,
            source="scipy.eigsh",
        )
        if ncv_actual is None:
            ncv_actual = min(n - 1, max(2 * n_need + 32, n_need + 2))
        return xp.asarray(vals_np), xp.asarray(vecs_np), {
            "backend": eig_backend,
            "ncv_actual": int(ncv_actual),
            "maxiter": int(eig_maxiter),
            "tol": float(eig_tol),
        }


def build_box_eigenpro_preconditioner(
    backend: Any,
    data_ctx: Any,
    reg_lambda: float,
    cfg: BTABConfig,
    *,
    q: int,
    profile_apply_components: bool = True,
    precomputed_active_set: BoxActiveSet | None = None,
) -> BoxEigenProPreconditionerData:
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

    box_size = int(active.box_idx.size)
    if box_size <= 0:
        raise ValueError("Box-EigenPro route requires a non-empty expanded box.")
    q = int(q)
    if q + 1 > box_size:
        raise ValueError(f"btab_eig_q+1={q + 1} exceeds expanded box size {box_size}.")

    diag_A = _diag_A_gpu(xp, gamma, weights_gpu, reg_lambda)
    diag_floor = xp.asarray(float(cfg.diag_floor), dtype=xp.float64)
    diag_inv_full_gpu = 1.0 / xp.maximum(diag_A, diag_floor)
    box_idx_gpu = xp.asarray(active.box_idx, dtype=xp.int64)
    tail_idx_gpu = xp.asarray(active.tail_idx, dtype=xp.int64)
    box_diag = diag_A[box_idx_gpu]
    diag_inv_box_gpu = 1.0 / xp.maximum(box_diag, diag_floor)
    diag_tail = diag_A[tail_idx_gpu] if int(active.tail_idx.size) else xp.empty((0,), dtype=xp.float64)
    diag_inv_tail_gpu = 1.0 / xp.maximum(diag_tail, diag_floor)

    center_multi = np.asarray(active.center_multi, dtype=np.int64).reshape(-1)
    radii = np.asarray(active.radii, dtype=np.int64).reshape(-1)
    box_slices = tuple(
        slice(int(center_multi[ax] - radii[ax]), int(center_multi[ax] + radii[ax] + 1))
        for ax in range(int(dim))
    )
    local_op_ctx = GPUOperatorContext()
    local_gf_gpu, box_shape, lag_shape = _build_local_fft_tensor(
        backend,
        data_ctx.xtxcol_gpu,
        active,
        mtot,
    )
    box_weights_gpu = xp.ascontiguousarray(weights_gpu[box_idx_gpu].reshape(box_shape))
    t_local = _now_synced(backend)

    runtime_stats = {
        "precond_apply_calls": 0,
        "time_diag_total": 0.0,
        "time_box_gather_total": 0.0,
        "time_box_solve_total": 0.0,
        "time_box_scatter_total": 0.0,
        "profile_apply_components": bool(profile_apply_components),
    }

    placeholder = BoxEigenProPreconditionerData(
        active=active,
        solve_mode="boxeig",
        exact_apply_mode="boxeig",
        box_idx_gpu=box_idx_gpu,
        tail_idx_gpu=tail_idx_gpu,
        diag_inv_full_gpu=diag_inv_full_gpu,
        diag_inv_tail_gpu=diag_inv_tail_gpu,
        diag_inv_box_gpu=diag_inv_box_gpu,
        global_shape=global_shape,
        box_slices=box_slices,
        local_gf_gpu=local_gf_gpu,
        box_weights_gpu=box_weights_gpu,
        box_shape=box_shape,
        lag_shape=lag_shape,
        local_op_ctx=local_op_ctx,
        runtime_stats=runtime_stats,
        diag_A_gpu=diag_A,
        diagnostics={},
        eig_U_gpu=None,
        eig_UH_gpu=None,
        eig_coeff_gpu=None,
        eig_coeff_col_gpu=None,
        eig_alpha=float("nan"),
        eig_theta_top_gpu=None,
        eig_theta_q1=float("nan"),
    )

    eig_counter: dict[str, int] = {}
    t_eig0 = _now_synced(backend)
    vals, vecs, eig_info = _compute_local_eigenpairs(
        backend,
        placeholder,
        float(reg_lambda),
        q,
        cfg,
        eig_counter,
    )
    t_eig1 = _now_synced(backend)
    vals = xp.real(xp.asarray(vals, dtype=xp.float64).reshape(-1))
    vecs = xp.asarray(vecs, dtype=xp.complex128)
    theta_top = xp.ascontiguousarray(vals[:q])
    Uq = xp.asfortranarray(vecs[:, :q])
    theta_q1 = float(vals[q])
    theta_safe = xp.maximum(theta_top, xp.asarray(1e-300, dtype=xp.float64))
    alpha = 1.0 / max(theta_q1, 1e-300)
    coeff = xp.ascontiguousarray((1.0 / theta_safe) - float(alpha))

    placeholder.eig_U_gpu = Uq
    placeholder.eig_UH_gpu = xp.asfortranarray(Uq.conj().T)
    placeholder.eig_coeff_gpu = coeff
    placeholder.eig_coeff_col_gpu = coeff.reshape(-1, 1)
    placeholder.eig_alpha = float(alpha)
    placeholder.eig_theta_top_gpu = theta_top
    placeholder.eig_theta_q1 = float(theta_q1)

    diagnostics = {
        "reg_lambda": float(reg_lambda),
        "sigma2_equiv_reg_lambda": float(reg_lambda),
        "gamma": float(gamma),
        "S_kind": "expanded_box",
        "solve_mode": "boxeig",
        "exact_apply_mode": "boxeig",
        "active_mode": str(active.active_mode),
        "active_topk": None if active.active_topk is None else int(active.active_topk),
        "active_tau": None if active.active_tau is None else float(active.active_tau),
        "active_size": int(active.active_idx.size),
        "box_size": int(active.box_idx.size),
        "tail_size": int(active.tail_idx.size),
        "btab_active_size_raw": int(active.active_idx.size),
        "btab_box_size": int(active.box_idx.size),
        "btab_eig_q": int(q),
        "btab_eig_size_S": int(active.box_idx.size),
        "btab_eig_size_T": int(active.tail_idx.size),
        "btab_eig_theta_q1": float(theta_q1),
        "btab_eig_theta_q1_over_sigma2": float(theta_q1 / max(float(reg_lambda), 1e-300)),
        "theta_q1_over_sigma2": float(theta_q1 / max(float(reg_lambda), 1e-300)),
        "btab_eig_storage_bytes": int(Uq.nbytes + theta_top.nbytes + coeff.nbytes),
        "btab_eig_n_eig_matvec": int(eig_counter.get("n_eig_matvec", 0)),
        "btab_eig_n_ABB_matvec_cols": int(eig_counter.get("n_eig_matvec", 0)),
        "btab_eig_ncv_actual": int(eig_info.get("ncv_actual", 0)),
        "btab_eig_backend": str(eig_info.get("backend", "")),
        "btab_eig_tol": float(eig_info.get("tol", getattr(cfg, "eig_tol", float("nan")))),
        "btab_eig_maxiter": int(eig_info.get("maxiter", _resolve_eig_maxiter(q, getattr(cfg, "eig_maxiter", None)))),
        "btab_eig_apply_batch_cols": getattr(cfg, "eig_apply_batch_cols", None),
        "box_radii": [int(v) for v in np.asarray(active.radii).reshape(-1)],
        "box_shape": [int(v) for v in box_shape],
        "time_active_set": float(t1 - t0),
        "time_build_local_fft": float(t_local - t1),
        "time_eig_setup": float(t_eig1 - t_eig0),
        "time_build_Ag": float((t_local - t1) + (t_eig1 - t_eig0)),
        "box_memory_bytes": 0,
        "chol_memory_bytes": 0,
        "inverse_memory_bytes": 0,
        "local_fft_memory_bytes": int(0 if local_gf_gpu is None else local_gf_gpu.nbytes),
    }
    placeholder.diagnostics = diagnostics
    return placeholder


def _apply_P_eig_box(
    backend: Any,
    precond_data: BoxEigenProPreconditionerData,
    rhs_box: Any,
    *,
    out: Any | None = None,
) -> Any:
    xp = backend.xp
    U = precond_data.eig_U_gpu
    UH = precond_data.eig_UH_gpu
    coeff = precond_data.eig_coeff_gpu
    coeff_col = precond_data.eig_coeff_col_gpu
    rhs = xp.asarray(rhs_box, dtype=xp.complex128)
    out_arr = xp.empty_like(rhs) if out is None else out
    proj = UH @ rhs
    if proj.ndim == 2:
        proj = coeff_col * proj
    else:
        proj = coeff * proj
    out_arr[...] = float(precond_data.eig_alpha) * rhs + (U @ proj)
    return out_arr


def apply_box_eigenpro_preconditioner(
    backend: Any,
    precond_data: BoxEigenProPreconditionerData,
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
        z_box = _apply_P_eig_box(backend, precond_data, rhs_box)
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


def apply_box_eigenpro_local(
    backend: Any,
    precond_data: BoxEigenProPreconditionerData,
    rhs_box: Any,
    *,
    out: Any | None = None,
) -> Any:
    return _apply_P_eig_box(backend, precond_data, rhs_box, out=out)
