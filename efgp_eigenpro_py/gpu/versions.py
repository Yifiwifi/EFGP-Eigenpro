from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Optional

import numpy as np

from ..efgp_solver import EFGPSolver
from ..eigenspace import estimate_top_eigenspace
from .backends import BackendConfig, build_gpu_backend_bundle
from .contexts import GPUOperatorContext, ensure_gpu_data_context
from .v1_ops import (
    V1Outputs,
    gpu_precompute_v1,
    predict_v1,
    solve_beta_plain_cg_v1,
)
from .v2_preconditioner import (
    CoordinateNystromPreconditionerData,
    EnsembleCoordinateNystromPreconditionerData,
    GPUPreconditionerData,
    HybridToprCoordinatePreconditionerData,
    apply_preconditioner_dominant_subspace,
    apply_preconditioner_coordinate_nystrom,
    apply_preconditioner_ensemble_coordinate_nystrom,
    apply_preconditioner_hybrid_topr_coordinate,
    apply_preconditioner_v2,
    build_dominant_subspace_preconditioner,
    build_coordinate_nystrom_preconditioner_data,
    build_ensemble_coordinate_nystrom_preconditioner_data,
    build_gpu_preconditioner_data,
)
from .iterative_solvers import pcg_solve_gpu
from .v3_eigenspace import (
    EigenspaceConfig,
    estimate_top_eigenspace_eigenpro_nystrom,
    estimate_top_eigenspace_v3,
    mu_for_precond_from_eig,
)
from .deflation_core import (
    DeflationData,
    build_deflation_data,
    deflation_memory_estimate,
    make_jacobi_precond,
    run_deflated_cg,
    run_deflated_pcg,
)
from .deflation_subspace import build_deflation_subspace
from .box_toeplitz_active_block.config import BTABConfig
from .box_toeplitz_active_block.runner import solve_box_toeplitz_active_block


@dataclass
class GPURunConfig:
    reg_lambda: float
    tol: float = 1e-8
    maxiter: int = 2000
    chunk_size: Optional[int] = None
    debug_finite_checks: bool = False
    profile_components: bool = True
    backend: BackendConfig = BackendConfig()


def _resolve_precond_storage_dtype(backend: Any, cfg: Any) -> Optional[Any]:
    """
    Resolve the storage dtype for a dense eigenspace preconditioner from ``cfg``.

    Looks for an explicit ``precond_storage_dtype`` / ``precond_dtype`` attribute on
    ``cfg`` (string like ``"complex64"`` / ``"complex128"`` or a numpy/cupy dtype)
    and returns the corresponding ``backend.xp`` dtype; returns ``None`` when nothing
    is configured so callers can fall back to their default (``complex128``).
    """
    xp = backend.xp
    raw = getattr(cfg, "precond_storage_dtype", None)
    if raw is None:
        raw = getattr(cfg, "precond_dtype", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in ("", "same_as_a", "default", "auto"):
            return None
        if key in ("complex64", "single", "c8"):
            return xp.complex64
        if key in ("complex128", "double", "c16"):
            return xp.complex128
        return xp.dtype(raw)
    return xp.dtype(raw)


def _dense_preconditioner_from_gpu_eigenspace(
    backend: Any,
    vecs_gpu: Any,
    vals_gpu: Any,
    q: int,
    mu: float,
    *,
    dtype: Optional[Any] = None,
) -> GPUPreconditionerData:
    xp = backend.xp
    q = int(q)
    Uq = xp.asarray(vecs_gpu[:, :q])
    if dtype is not None:
        Uq = Uq.astype(xp.dtype(dtype), copy=False)
    Uq = xp.asfortranarray(Uq)
    UHq = xp.asfortranarray(Uq.conj().T)
    scale = xp.ascontiguousarray(
        xp.asarray(1.0 - (float(mu) / vals_gpu[:q]), dtype=Uq.dtype).reshape(-1)
    )
    USq = xp.asfortranarray(Uq * scale.reshape(1, -1))
    return GPUPreconditionerData(
        U_gpu=Uq,
        UH_gpu=UHq,
        scale_gpu=scale,
        scale_col_gpu=scale.reshape(-1, 1),
        US_gpu=USq,
    )


def run_v1_pure_efgp(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
) -> V1Outputs:
    """
    V1: ``top_q=0`` — GPU FFT Toeplitz matvec + GPU plain CG; NUFFT precompute/predict
    use CPU FINUFFT (same as ``nufft_ops``) then transfer, for numerical parity with
    ``EFGPSolver``.
    """
    backend = build_gpu_backend_bundle(cfg.backend)
    data_ctx = ensure_gpu_data_context(backend, x, y, state=None)
    data_ctx.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    data_ctx.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    op_ctx = GPUOperatorContext()
    t0 = time.perf_counter()
    data_ctx = gpu_precompute_v1(
        backend,
        solver.kernel,
        solver.eps,
        solver.nufft_tol,
        data_ctx,
        op_ctx,
        l2scaled=solver.l2scaled,
        chunk_size=cfg.chunk_size,
    )
    t1 = time.perf_counter()

    reg = float(cfg.reg_lambda)
    t2 = time.perf_counter()
    beta_gpu, it, relres, cg_stats = solve_beta_plain_cg_v1(
        backend,
        data_ctx,
        reg,
        op_ctx,
        cfg.tol,
        cfg.maxiter,
        return_stats=True,
        profile_components=cfg.profile_components,
    )
    t3 = time.perf_counter()
    _ = predict_v1(backend, data_ctx, x, beta_gpu)
    t4 = time.perf_counter()

    return V1Outputs(
        beta_gpu=beta_gpu,
        diagnostics={
            "version": "v1",
            "status": "ok",
            "nufft_backend": backend.nufft_name,
            "nufft_stage": data_ctx.meta.get("nufft_stage"),
            "cg_iters": it,
            "cg_relres": relres,
            "time_precompute": float(t1 - t0),
            "time_solve": float(t3 - t2),
            "time_predict": float(t4 - t3),
            "time_total": float(t4 - t0),
            "t_matvec_avg": float(cg_stats["t_matvec_avg"]),
            "t_matvec_total": float(cg_stats["t_matvec_total"]),
            "n_matvec": int(cg_stats["n_matvec"]),
            # Explicit naming for clarity in benchmark tables: these matvecs are from CG solve.
            "cg_n_matvec": int(cg_stats["n_matvec"]),
            # V1 has no eigenspace / preconditioner estimation stage.
            "eigen_n_matvec": 0,
            "eigen_apply_A_block_calls": 0,
            "device_name": backend.device_name,
            "has_nufft": backend.has_nufft,
            "chunk_size": cfg.chunk_size,
            "debug_finite_checks": bool(cfg.debug_finite_checks),
            "profile_components": bool(cfg.profile_components),
        },
        backend=backend,
        data_ctx=data_ctx,
    )


def run_v2_with_preconditioner_apply(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
    top_q: int,
    *,
    U_cpu: Optional[np.ndarray] = None,
    scale_cpu: Optional[np.ndarray] = None,
    eig_method: str = "subspace_iter",
    eig_tol: float = 1e-2,
    eig_maxiter: int = 20,
    eig_block_size: Optional[int] = None,
    eig_oversample: int = 2,
) -> V1Outputs:
    """
    V2: CPU eigenspace + GPU preconditioner apply + GPU PCG.
    """
    backend = build_gpu_backend_bundle(cfg.backend)
    data_ctx = ensure_gpu_data_context(backend, x, y, state=None)
    data_ctx.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    op_ctx = GPUOperatorContext()

    t0 = time.perf_counter()
    data_ctx = gpu_precompute_v1(
        backend,
        solver.kernel,
        solver.eps,
        solver.nufft_tol,
        data_ctx,
        op_ctx,
        l2scaled=solver.l2scaled,
        chunk_size=cfg.chunk_size,
    )
    t1 = time.perf_counter()

    if top_q <= 0:
        raise ValueError("top_q must be > 0 for V2 preconditioning.")

    if U_cpu is None or scale_cpu is None:
        # CPU eigenspace estimation on the same kernel/params.
        state_cpu = solver.precompute(x, y)

        eigpairs = estimate_top_eigenspace(
            lambda v: solver._apply_A(state_cpu, v),
            size=state_cpu.rhs.size,
            top_q=top_q,
            method=eig_method,
            tol=eig_tol,
            maxiter=eig_maxiter,
            matvec_block=lambda V: solver._apply_A_block(state_cpu, V),
            block_size=eig_block_size,
            oversample=eig_oversample,
        )

        if eigpairs.values.size > top_q:
            mu = float(eigpairs.values[top_q])
        else:
            mu = float(eigpairs.values[-1])
        U_cpu = eigpairs.vectors[:, :top_q]
        scale_cpu = 1.0 - (mu / eigpairs.values[:top_q])

    t2 = time.perf_counter()
    precond_data = build_gpu_preconditioner_data(backend, U_cpu, scale_cpu)
    op_ctx.solve_dtype = precond_data.U_gpu.dtype
    t3 = time.perf_counter()

    def _matvec(v: Any, out: Any) -> None:
        from .v1_ops import apply_A_v1

        apply_A_v1(backend, data_ctx, v, float(cfg.reg_lambda), op_ctx, out=out)

    def _precond(v: Any, out: Any) -> None:
        apply_preconditioner_v2(backend, precond_data, v, op_ctx=op_ctx, out=out)

    beta_gpu, it, relres, stats = pcg_solve_gpu(
        backend,
        _matvec,
        _precond,
        data_ctx.rhs_gpu,
        op_ctx,
        cfg.tol,
        cfg.maxiter,
        return_stats=True,
        profile_components=cfg.profile_components,
    )
    t4 = time.perf_counter()
    _ = predict_v1(backend, data_ctx, x, beta_gpu)
    t5 = time.perf_counter()

    return V1Outputs(
        beta_gpu=beta_gpu,
        diagnostics={
            "version": "v2",
            "status": "ok",
            "top_q": int(top_q),
            "nufft_backend": backend.nufft_name,
            "nufft_stage": data_ctx.meta.get("nufft_stage"),
            "cg_iters": int(it),
            "cg_relres": float(relres),
            "time_precompute": float(t1 - t0),
            "time_eigenspace": float(t2 - t1),
            "time_precond_build": float(t3 - t2),
            "time_solve": float(t4 - t3),
            "time_predict": float(t5 - t4),
            "time_total": float(t5 - t0),
            "t_matvec_avg": float(stats["t_matvec_avg"]),
            "t_matvec_total": float(stats["t_matvec_total"]),
            "n_matvec": int(stats["n_matvec"]),
            "t_precond_total": float(stats["t_precond_total"]),
            "t_precond_avg": float(stats["t_precond_avg"]),
            "n_precond": int(stats["n_precond"]),
            "device_name": backend.device_name,
            "has_nufft": backend.has_nufft,
            "chunk_size": cfg.chunk_size,
            "debug_finite_checks": bool(cfg.debug_finite_checks),
            "profile_components": bool(cfg.profile_components),
        },
        backend=backend,
        data_ctx=data_ctx,
    )


def run_v3_full_gpu_eigenspace(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
    eig_cfg: Optional[EigenspaceConfig] = None,
) -> V1Outputs:
    """
    V3: GPU eigenspace estimation + GPU preconditioner + GPU PCG.
    """
    backend = build_gpu_backend_bundle(cfg.backend)
    data_ctx = ensure_gpu_data_context(backend, x, y, state=None)
    data_ctx.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    op_ctx = GPUOperatorContext()

    t0 = time.perf_counter()
    data_ctx = gpu_precompute_v1(
        backend,
        solver.kernel,
        solver.eps,
        solver.nufft_tol,
        data_ctx,
        op_ctx,
        l2scaled=solver.l2scaled,
        chunk_size=cfg.chunk_size,
    )
    t1 = time.perf_counter()

    eig_cfg = eig_cfg or EigenspaceConfig(q_max=32, block_size=40)
    method_name = str((eig_cfg.eig_method if eig_cfg.eig_method is not None else eig_cfg.method) or "subspace_iter").lower()
    if method_name in (
        "eigenpro_nystrom",
        "nystrom",
        "ep_nystrom",
        "coordinate_nystrom",
        "coord_nystrom",
        "ensemble_coordinate_nystrom",
        "random_support_lift",
        "support_lift",
    ):
        eig_cfg.method_cfg = dict(eig_cfg.method_cfg or {})
        eig_cfg.method_cfg.setdefault("data_ctx", data_ctx)
        eig_cfg.method_cfg.setdefault("reg_lambda", float(cfg.reg_lambda))

    # Count matvecs used during eigenspace / preconditioner estimation separately from CG/PCG solve.
    # Our `_apply_A_block` implementation calls `apply_A_v1` once per column, so we count scalar matvecs
    # as the number of columns processed in each block call.
    eigen_apply_A_block_calls = 0
    eigen_n_matvec = 0
    eigen_t_matvec_total = 0.0
    _eigen_event_pairs: list[tuple[Any, Any]] = []

    def _apply_A_block(v_block: Any) -> Any:
        from .v1_ops import apply_A_v1

        xp = backend.xp
        v_block = xp.asarray(v_block, dtype=xp.complex128)
        if v_block.ndim == 1:
            v_block = v_block.reshape(-1, 1)
        out_block = xp.empty_like(v_block)
        for i in range(v_block.shape[1]):
            apply_A_v1(
                backend,
                data_ctx,
                v_block[:, i],
                float(cfg.reg_lambda),
                op_ctx,
                out=out_block[:, i],
            )
        return out_block

    def _apply_A_block_counted(v_block: Any) -> Any:
        nonlocal eigen_apply_A_block_calls, eigen_n_matvec, eigen_t_matvec_total, _eigen_event_pairs
        try:
            ndim = int(getattr(v_block, "ndim", 1))
            cols = 1 if ndim == 1 else int(getattr(v_block, "shape")[1])
        except Exception:
            cols = 1
        eigen_apply_A_block_calls += 1
        eigen_n_matvec += int(cols)

        xp = backend.xp
        cuda = getattr(xp, "cuda", None)
        if cuda is not None:
            # Use CUDA events to avoid synchronizing on every matvec call.
            start = cuda.Event()
            end = cuda.Event()
            start.record()
            out = _apply_A_block(v_block)
            end.record()
            _eigen_event_pairs.append((start, end))
            return out

        t0_local = time.perf_counter()
        out = _apply_A_block(v_block)
        eigen_t_matvec_total += time.perf_counter() - t0_local
        return out

    vals_gpu, vecs_gpu, eig_diag = estimate_top_eigenspace_v3(
        backend=backend,
        apply_A_block_gpu=_apply_A_block_counted,
        size=int(data_ctx.rhs_gpu.size),
        cfg=eig_cfg,
    )
    # Finalize eigen-estimation matvec timing if we used CUDA events.
    if _eigen_event_pairs:
        try:
            _eigen_event_pairs[-1][1].synchronize()
            cuda = getattr(backend.xp, "cuda", None)
            if cuda is not None:
                eigen_t_matvec_total = 1e-3 * float(
                    sum(cuda.get_elapsed_time(s, e) for (s, e) in _eigen_event_pairs)
                )
        except Exception:
            pass
    t2 = time.perf_counter()

    q = int(eig_cfg.q_max)
    precond_kind = str(eig_diag.get("precond_kind", "full_eigenpro")).lower()
    if precond_kind in ("coordinate_nystrom", "diag_coordinate_nystrom"):
        coord_gamma = float(eig_diag.get("coord_nystrom_gamma", 1.0))
        precond_data = build_coordinate_nystrom_preconditioner_data(
            backend,
            eig_diag["S_gpu"],
            eig_diag["V_gpu"],
            eig_diag["theta_gpu"],
            float(eig_diag["mu"]),
            gamma=coord_gamma,
            diag_inv_sqrt_gpu=eig_diag.get("diag_inv_sqrt_gpu", None),
        )
    elif precond_kind == "ensemble_coordinate_nystrom":
        precond_data = build_ensemble_coordinate_nystrom_preconditioner_data(
            backend,
            list(eig_diag.get("ensemble_entries", []) or []),
            gamma=float(eig_diag.get("ensemble_gamma", 1.0)),
        )
    elif precond_kind in ("hybrid_topr_coordinate", "hybrid_top_r_coordinate", "topr_hybrid_coordinate"):
        # Dense top-r block + compact coordinate tail.
        xp = backend.xp
        coord_gamma = float(eig_diag.get("coord_nystrom_gamma", 1.0))
        mu_dense = float(eig_diag.get("mu"))
        evals_r = xp.asarray(eig_diag.get("hybrid_dense_eigvals_gpu"))
        U_r = xp.asarray(eig_diag.get("hybrid_dense_eigvecs_gpu"))
        if evals_r.ndim != 1 or U_r.ndim != 2 or int(U_r.shape[1]) != int(evals_r.shape[0]):
            raise ValueError("hybrid_topr_coordinate requires hybrid_dense_eigvals_gpu and hybrid_dense_eigvecs_gpu.")
        scale_dense = xp.ascontiguousarray(1.0 - (mu_dense / xp.asarray(evals_r)))
        U_dense = xp.asfortranarray(U_r)
        US_dense = xp.asfortranarray(U_dense * scale_dense.reshape(1, -1))
        dense = GPUPreconditionerData(
            U_gpu=U_dense,
            UH_gpu=xp.asfortranarray(U_dense.conj().T),
            scale_gpu=scale_dense,
            scale_col_gpu=scale_dense.reshape(-1, 1),
            US_gpu=US_dense,
        )
        tail = build_coordinate_nystrom_preconditioner_data(
            backend,
            eig_diag["hybrid_tail_S_gpu"],
            eig_diag["hybrid_tail_V_gpu"],
            eig_diag["hybrid_tail_theta_gpu"],
            float(eig_diag.get("hybrid_tail_mu", eig_diag.get("mu"))),
            gamma=coord_gamma,
            diag_inv_sqrt_gpu=eig_diag.get("diag_inv_sqrt_gpu", None),
        )
        precond_data = HybridToprCoordinatePreconditionerData(dense=dense, tail=tail)
    else:
        mu = mu_for_precond_from_eig(vals_gpu, q, eig_diag)
        precond_data = _dense_preconditioner_from_gpu_eigenspace(
            backend,
            vecs_gpu,
            vals_gpu,
            q,
            mu,
            dtype=getattr(data_ctx.rhs_gpu, "dtype", None),
        )
        op_ctx.solve_dtype = precond_data.U_gpu.dtype
    t3 = time.perf_counter()

    def _matvec(v: Any, out: Any) -> None:
        from .v1_ops import apply_A_v1

        apply_A_v1(backend, data_ctx, v, float(cfg.reg_lambda), op_ctx, out=out)

    def _precond(v: Any, out: Any) -> None:
        if hasattr(precond_data, "tail"):
            apply_preconditioner_hybrid_topr_coordinate(
                backend, precond_data, v, op_ctx=op_ctx, out=out
            )
        elif precond_kind == "ensemble_coordinate_nystrom" or isinstance(
            precond_data, EnsembleCoordinateNystromPreconditionerData
        ):
            apply_preconditioner_ensemble_coordinate_nystrom(
                backend, precond_data, v, op_ctx=op_ctx, out=out
            )
        elif precond_kind in ("coordinate_nystrom", "diag_coordinate_nystrom") or all(
            hasattr(precond_data, k) for k in ("S_gpu", "V_gpu", "VH_gpu")
        ):
            apply_preconditioner_coordinate_nystrom(
                backend, precond_data, v, op_ctx=op_ctx, out=out
            )
        else:
            apply_preconditioner_v2(backend, precond_data, v, op_ctx=op_ctx, out=out)

    beta_gpu, it, relres, stats = pcg_solve_gpu(
        backend,
        _matvec,
        _precond,
        data_ctx.rhs_gpu,
        op_ctx,
        cfg.tol,
        cfg.maxiter,
        return_stats=True,
        profile_components=cfg.profile_components,
    )
    t4 = time.perf_counter()
    _ = predict_v1(backend, data_ctx, x, beta_gpu)
    t5 = time.perf_counter()

    return V1Outputs(
        beta_gpu=beta_gpu,
        diagnostics={
            "version": "v3",
            "status": "ok",
            "top_q": int(q),
            "precond_kind": precond_kind,
            "nufft_backend": backend.nufft_name,
            "nufft_stage": data_ctx.meta.get("nufft_stage"),
            "cg_iters": int(it),
            "cg_relres": float(relres),
            "time_precompute": float(t1 - t0),
            "time_eigenspace": float(t2 - t1),
            "time_precond_build": float(t3 - t2),
            "time_solve": float(t4 - t3),
            "time_predict": float(t5 - t4),
            "time_total": float(t5 - t0),
            "eig_n_iter": int(eig_diag.get("n_iter", 0)),
            "eig_block_size": int(eig_diag.get("block_size", 0)),
            "eig_residual_fro": float(eig_diag.get("residual_fro", float("nan"))),
            "eig_residual_fro_rel": float(eig_diag.get("residual_fro_rel", float("nan"))),
            "eig_residual_cols_rel": eig_diag.get("residual_cols_rel"),
            "surrogate_tag": str(eig_diag.get("surrogate_tag", "")),
            "eig_nystrom_kernel_s": float(eig_diag.get("eig_nystrom_kernel_s", float("nan"))),
            "coord_nystrom_gamma": float(eig_diag.get("coord_nystrom_gamma", float("nan"))),
            "lambda1_coord_nystrom": float(eig_diag.get("lambda1_coord_nystrom", float("nan"))),
            "theta_coord_topq": eig_diag.get("theta_coord_topq", []),
            "injected_eps_coord_topq": eig_diag.get("injected_eps_coord_topq", []),
            # Estimation-stage matvec accounting (eigenspace / preconditioner construction).
            "eigen_n_matvec": int(eigen_n_matvec),
            "eigen_apply_A_block_calls": int(eigen_apply_A_block_calls),
            "eigen_t_matvec_total": float(eigen_t_matvec_total),
            "eigen_t_matvec_avg": float(eigen_t_matvec_total / max(int(eigen_n_matvec), 1)),
            "t_matvec_avg": float(stats["t_matvec_avg"]),
            "t_matvec_total": float(stats["t_matvec_total"]),
            "n_matvec": int(stats["n_matvec"]),
            # Explicit alias for clarity: these matvecs are from CG/PCG solve.
            "cg_n_matvec": int(stats["n_matvec"]),
            "t_precond_total": float(stats["t_precond_total"]),
            "t_precond_avg": float(stats["t_precond_avg"]),
            "n_precond": int(stats["n_precond"]),
            "device_name": backend.device_name,
            "has_nufft": backend.has_nufft,
            "chunk_size": cfg.chunk_size,
            "debug_finite_checks": bool(cfg.debug_finite_checks),
            "profile_components": bool(cfg.profile_components),
        },
        backend=backend,
        data_ctx=data_ctx,
    )


def run_v6_box_toeplitz_active_block(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
    *,
    btab_cfg: Optional[BTABConfig] = None,
) -> V1Outputs:
    """
    V6: GPU FFT Toeplitz matvec + Box-Toeplitz Active Block preconditioner
    (exact dense box or matrix-free inner-PCG box solve).
    """
    backend = build_gpu_backend_bundle(cfg.backend)
    data_ctx = ensure_gpu_data_context(backend, x, y, state=None)
    data_ctx.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    op_ctx = GPUOperatorContext()

    t0 = time.perf_counter()
    data_ctx = gpu_precompute_v1(
        backend,
        solver.kernel,
        solver.eps,
        solver.nufft_tol,
        data_ctx,
        op_ctx,
        l2scaled=solver.l2scaled,
        chunk_size=cfg.chunk_size,
    )
    t1 = time.perf_counter()

    btab_cfg = btab_cfg or BTABConfig()
    beta_gpu, it, relres, stats, setup_diag = solve_box_toeplitz_active_block(
        backend,
        data_ctx,
        float(cfg.reg_lambda),
        data_ctx.rhs_gpu,
        op_ctx,
        tol=cfg.tol,
        maxiter=cfg.maxiter,
        btab_cfg=btab_cfg,
        profile_components=cfg.profile_components,
    )
    t2 = time.perf_counter()
    _ = predict_v1(backend, data_ctx, x, beta_gpu)
    t3 = time.perf_counter()

    diagnostics = {
        "version": "v6_btab",
        "status": "ok",
        "nufft_backend": backend.nufft_name,
        "nufft_stage": data_ctx.meta.get("nufft_stage"),
        "cg_iters": int(it),
        "cg_relres": float(relres),
        "outer_iters": int(it),
        "outer_relres": float(relres),
        "time_precompute": float(t1 - t0),
        "time_precond_build": float(setup_diag.get("time_precond_build", float("nan"))),
        "time_solve": float(t2 - t1),
        "time_predict": float(t3 - t2),
        "time_total": float(t3 - t0),
        "active_mode": setup_diag.get("active_mode"),
        "active_topk": setup_diag.get("active_topk"),
        "active_tau": setup_diag.get("active_tau"),
        "solve_mode": setup_diag.get("solve_mode"),
        "exact_apply_mode": setup_diag.get("exact_apply_mode"),
        "outer_solver": setup_diag.get("outer_solver"),
        "outer_gmres_restart": int(setup_diag.get("outer_gmres_restart", 0)),
        "inner_tol": float(setup_diag.get("inner_tol", float("nan"))),
        "inner_maxiter": int(setup_diag.get("inner_maxiter", 0)),
        "inner_precond": setup_diag.get("inner_precond"),
        "active_size": int(setup_diag.get("active_size", 0)),
        "box_size": int(setup_diag.get("box_size", 0)),
        "tail_size": int(setup_diag.get("tail_size", 0)),
        "box_radii": setup_diag.get("box_radii", []),
        "box_shape": setup_diag.get("box_shape", []),
        "gamma": float(setup_diag.get("gamma", float("nan"))),
        "time_active_set": float(setup_diag.get("time_active_set", float("nan"))),
        "time_build_Ag": float(setup_diag.get("time_build_Ag", float("nan"))),
        "time_build_box_matrix": float(setup_diag.get("time_build_box_matrix", float("nan"))),
        "time_symmetrize_regularize": float(setup_diag.get("time_symmetrize_regularize", float("nan"))),
        "time_cholesky": float(setup_diag.get("time_cholesky", float("nan"))),
        "time_build_inverse": float(setup_diag.get("time_build_inverse", float("nan"))),
        "time_build_local_fft": float(setup_diag.get("time_build_local_fft", float("nan"))),
        "box_memory_bytes": int(setup_diag.get("box_memory_bytes", 0)),
        "chol_memory_bytes": int(setup_diag.get("chol_memory_bytes", 0)),
        "inverse_memory_bytes": int(setup_diag.get("inverse_memory_bytes", 0)),
        "local_fft_memory_bytes": int(setup_diag.get("local_fft_memory_bytes", 0)),
        "precond_apply_calls": int(setup_diag.get("precond_apply_calls", 0)),
        "inner_total_iters": int(setup_diag.get("inner_total_iters", 0)),
        "inner_total_matvec": int(setup_diag.get("inner_total_matvec", 0)),
        "inner_total_precond": int(setup_diag.get("inner_total_precond", 0)),
        "inner_last_relres": float(setup_diag.get("inner_last_relres", float("nan"))),
        "inner_max_iters": int(setup_diag.get("inner_max_iters", 0)),
        "time_diag_total": float(setup_diag.get("time_diag_total", float("nan"))),
        "time_box_gather_total": float(setup_diag.get("time_box_gather_total", float("nan"))),
        "time_box_solve_total": float(setup_diag.get("time_box_solve_total", float("nan"))),
        "time_box_scatter_total": float(setup_diag.get("time_box_scatter_total", float("nan"))),
        "time_diag_avg": float(setup_diag.get("time_diag_avg", float("nan"))),
        "time_box_gather_avg": float(setup_diag.get("time_box_gather_avg", float("nan"))),
        "time_box_solve_avg": float(setup_diag.get("time_box_solve_avg", float("nan"))),
        "time_box_scatter_avg": float(setup_diag.get("time_box_scatter_avg", float("nan"))),
        "t_matvec_avg": float(stats["t_matvec_avg"]),
        "t_matvec_total": float(stats["t_matvec_total"]),
        "n_matvec": int(stats["n_matvec"]),
        "cg_n_matvec": int(stats["n_matvec"]),
        "t_precond_total": float(stats["t_precond_total"]),
        "t_precond_avg": float(stats["t_precond_avg"]),
        "n_precond": int(stats["n_precond"]),
        "outer_status": setup_diag.get("outer_status", stats.get("status")),
        "device_name": backend.device_name,
        "has_nufft": backend.has_nufft,
        "chunk_size": cfg.chunk_size,
        "debug_finite_checks": bool(cfg.debug_finite_checks),
        "profile_components": bool(cfg.profile_components),
    }
    return V1Outputs(
        beta_gpu=beta_gpu,
        diagnostics=diagnostics,
        backend=backend,
        data_ctx=data_ctx,
    )


def build_v3_pcg_left_precond_matvec(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
    eig_cfg: EigenspaceConfig,
) -> tuple[Any, Callable[[Any, Any], None], int, dict[str, Any]]:
    """
    Build the linear map used in ``pcg_solve_gpu``-style analysis: ``v -> P(A v)``,
    where ``A`` is the original SPD EFGP operator and ``P`` is ``apply_preconditioner_v2``,
    the same as applying the preconditioner to a residual after an ``A``-apply in the
    left-preconditioned view ``M^{-1} A`` (PCG: ``z = P(r)``, ``A`` on search directions).
    """
    from .v1_ops import apply_A_v1

    backend = build_gpu_backend_bundle(cfg.backend)
    data_ctx = ensure_gpu_data_context(backend, x, y, state=None)
    data_ctx.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    op_ctx = GPUOperatorContext()
    data_ctx = gpu_precompute_v1(
        backend,
        solver.kernel,
        solver.eps,
        solver.nufft_tol,
        data_ctx,
        op_ctx,
        l2scaled=solver.l2scaled,
        chunk_size=cfg.chunk_size,
    )

    def _apply_A_block(v_block: Any) -> Any:
        xp = backend.xp
        v_block = xp.asarray(v_block, dtype=xp.complex128)
        if v_block.ndim == 1:
            v_block = v_block.reshape(-1, 1)
        out_block = xp.empty_like(v_block)
        for i in range(v_block.shape[1]):
            apply_A_v1(
                backend,
                data_ctx,
                v_block[:, i],
                float(cfg.reg_lambda),
                op_ctx,
                out=out_block[:, i],
            )
        return out_block

    method_name = str((eig_cfg.eig_method if eig_cfg.eig_method is not None else eig_cfg.method) or "subspace_iter").lower()
    if method_name in (
        "eigenpro_nystrom",
        "nystrom",
        "ep_nystrom",
        "coordinate_nystrom",
        "coord_nystrom",
        "ensemble_coordinate_nystrom",
    ):
        eig_cfg.method_cfg = dict(eig_cfg.method_cfg or {})
        eig_cfg.method_cfg.setdefault("data_ctx", data_ctx)
        eig_cfg.method_cfg.setdefault("reg_lambda", float(cfg.reg_lambda))

    vals_gpu, vecs_gpu, eig_diag = estimate_top_eigenspace_v3(
        backend=backend,
        apply_A_block_gpu=_apply_A_block,
        size=int(data_ctx.rhs_gpu.size),
        cfg=eig_cfg,
    )
    q = int(eig_cfg.q_max)
    precond_kind = str(eig_diag.get("precond_kind", "full_eigenpro")).lower()
    if precond_kind == "coordinate_nystrom":
        precond_data = build_coordinate_nystrom_preconditioner_data(
            backend,
            eig_diag["S_gpu"],
            eig_diag["V_gpu"],
            eig_diag["theta_gpu"],
            float(eig_diag["mu"]),
        )
    elif precond_kind == "ensemble_coordinate_nystrom":
        precond_data = build_ensemble_coordinate_nystrom_preconditioner_data(
            backend,
            list(eig_diag.get("ensemble_entries", []) or []),
            gamma=float(eig_diag.get("ensemble_gamma", 1.0)),
        )
    else:
        mu = mu_for_precond_from_eig(vals_gpu, q, eig_diag)
        precond_data = _dense_preconditioner_from_gpu_eigenspace(
            backend,
            vecs_gpu,
            vals_gpu,
            q,
            mu,
            dtype=backend.xp.complex128,
        )
    n = int(data_ctx.rhs_gpu.size)
    av_buf: list[Any] = [None]

    def matvec(v: Any, out: Any) -> None:
        xp = backend.xp
        va = xp.asarray(v, dtype=xp.complex128).reshape(-1)
        if av_buf[0] is None or int(av_buf[0].size) != int(va.size):
            av_buf[0] = xp.empty((int(va.size),), dtype=xp.complex128)
        oa = xp.asarray(out, dtype=xp.complex128).reshape(-1)
        apply_A_v1(backend, data_ctx, va, float(cfg.reg_lambda), op_ctx, out=av_buf[0])
        if precond_kind == "ensemble_coordinate_nystrom" or isinstance(
            precond_data, EnsembleCoordinateNystromPreconditionerData
        ):
            apply_preconditioner_ensemble_coordinate_nystrom(
                backend, precond_data, av_buf[0], op_ctx=op_ctx, out=oa
            )
        elif precond_kind == "coordinate_nystrom" or all(
            hasattr(precond_data, k) for k in ("S_gpu", "V_gpu", "VH_gpu")
        ):
            apply_preconditioner_coordinate_nystrom(
                backend, precond_data, av_buf[0], op_ctx=op_ctx, out=oa
            )
        else:
            apply_preconditioner_v2(backend, precond_data, av_buf[0], op_ctx=op_ctx, out=oa)

    meta: dict[str, Any] = {
        "slq_spectrum": "M_inv_A",
        "slq_spectrum_desc": "P(A v); same P as apply_preconditioner_v2 in PCG.",
        "top_q": int(q),
        "precond_kind": precond_kind,
        "eig_residual_fro_rel": float(eig_diag.get("residual_fro_rel", float("nan"))),
    }
    return backend, matvec, n, meta


def run_v4_dominant_subspace_preconditioner(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
    q: int,
    *,
    s: int = 8,
    kmax: int = 2,
    keep_factor: float = 5.0,
) -> V1Outputs:
    """
    V4: GPU dominant-subspace preconditioner + GPU PCG.
    """
    backend = build_gpu_backend_bundle(cfg.backend)
    data_ctx = ensure_gpu_data_context(backend, x, y, state=None)
    data_ctx.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    op_ctx = GPUOperatorContext()

    t0 = time.perf_counter()
    data_ctx = gpu_precompute_v1(
        backend,
        solver.kernel,
        solver.eps,
        solver.nufft_tol,
        data_ctx,
        op_ctx,
        l2scaled=solver.l2scaled,
        chunk_size=cfg.chunk_size,
    )
    t1 = time.perf_counter()

    if q <= 0:
        raise ValueError("q must be > 0 for dominant-subspace preconditioning.")

    sigma2 = float(cfg.reg_lambda)

    def _apply_A_block(v_block: Any) -> Any:
        from .v1_ops import apply_A_v1

        xp = backend.xp
        v_block = xp.asarray(v_block, dtype=xp.complex128)
        if v_block.ndim == 1:
            v_block = v_block.reshape(-1, 1)
        out_block = xp.empty_like(v_block)
        for i in range(v_block.shape[1]):
            apply_A_v1(
                backend,
                data_ctx,
                v_block[:, i],
                sigma2,
                op_ctx,
                out=out_block[:, i],
            )
        return out_block

    precond_data, precond_diag = build_dominant_subspace_preconditioner(
        backend=backend,
        apply_A_block=_apply_A_block,
        size=int(data_ctx.rhs_gpu.size),
        sigma2=sigma2,
        q=q,
        s=s,
        kmax=kmax,
        keep_factor=keep_factor,
    )
    t2 = time.perf_counter()

    def _matvec(v: Any, out: Any) -> None:
        from .v1_ops import apply_A_v1

        apply_A_v1(backend, data_ctx, v, sigma2, op_ctx, out=out)

    def _precond(v: Any, out: Any) -> None:
        apply_preconditioner_dominant_subspace(backend, precond_data, v, op_ctx=op_ctx, out=out)

    beta_gpu, it, relres, stats = pcg_solve_gpu(
        backend,
        _matvec,
        _precond,
        data_ctx.rhs_gpu,
        op_ctx,
        cfg.tol,
        cfg.maxiter,
        return_stats=True,
        profile_components=cfg.profile_components,
    )
    t3 = time.perf_counter()
    _ = predict_v1(backend, data_ctx, x, beta_gpu)
    t4 = time.perf_counter()

    return V1Outputs(
        beta_gpu=beta_gpu,
        diagnostics={
            "version": "v4_dominant_subspace",
            "status": "ok",
            "precond_q": int(q),
            "precond_s": int(precond_diag["s"]),
            "precond_p": int(precond_diag["p"]),
            "precond_kmax": int(precond_diag["kmax"]),
            "precond_keep_factor": float(precond_diag["keep_factor"]),
            "precond_rank": int(precond_diag["kept_rank"]),
            "nufft_backend": backend.nufft_name,
            "nufft_stage": data_ctx.meta.get("nufft_stage"),
            "cg_iters": int(it),
            "cg_relres": float(relres),
            "time_precompute": float(t1 - t0),
            "time_precond_build": float(t2 - t1),
            "time_solve": float(t3 - t2),
            "time_predict": float(t4 - t3),
            "time_total": float(t4 - t0),
            "t_matvec_avg": float(stats["t_matvec_avg"]),
            "t_matvec_total": float(stats["t_matvec_total"]),
            "n_matvec": int(stats["n_matvec"]),
            "t_precond_total": float(stats["t_precond_total"]),
            "t_precond_avg": float(stats["t_precond_avg"]),
            "n_precond": int(stats["n_precond"]),
            "device_name": backend.device_name,
            "has_nufft": backend.has_nufft,
            "chunk_size": cfg.chunk_size,
            "debug_finite_checks": bool(cfg.debug_finite_checks),
            "profile_components": bool(cfg.profile_components),
        },
        backend=backend,
        data_ctx=data_ctx,
    )


def run_v5_deflated_cg(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
    *,
    m: int = 64,
    method: str = "coord_nystrom",
    which: str = "LM",
    eig_solver: str = "subspace_iter",
    lowfi_tol: float = 1e-3,
    lowfi_n_iter: int = 10,
    oversample: int = 12,
    hm_t: Optional[int] = None,
    coarse_ratio: float = 0.5,
    freq_box_mode: str = "center",
    rank_tol: float = 1e-12,
    jitter_ratio: float = 1e-12,
    block_cols: int = 8,
    z_storage_dtype: str = "same_as_A",
    w_mode: str = "dense",
    matvec_form: str = "structured_ZH",
    precond: str = "none",
    seed: int = 0,
) -> V1Outputs:
    """
    V5: multi-fidelity Frank-Vuik deflated CG.

    A cheap low-fidelity operator ``A_tilde`` (``method`` selects coord_nystrom /
    freq_trunc / float32) only finds the deflation subspace ``Z``.  The deflation
    projector ``P_D = I - A Z (Z* A Z)^{-1} Z*`` is built with the true
    high-fidelity ``A`` (W = A Z, G = Z* A Z), then ``P_D A x_hat = P_D b`` is
    solved by CG and the true solution is recovered.

    ``precond`` selects the inner Krylov solver:
      * ``"none"`` (default): plain deflated CG on ``P_D A`` (Phase 1).
      * ``"jacobi"``: Phase-2 deflated *preconditioned* CG (Frank-Vuik DPCG) with a
        diagonal SPD preconditioner ``M^{-1} = diag(A)^{-1}``.

    The headline cost metric is ``hi_n_matvec = m_eff (calibration) + N_CG + 1
    (recovery)``; low-fidelity matvecs used to build ``Z`` are reported separately.
    """
    from .v1_ops import apply_A_block_v1, apply_A_v1

    backend = build_gpu_backend_bundle(cfg.backend)
    data_ctx = ensure_gpu_data_context(backend, x, y, state=None)
    data_ctx.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    op_ctx = GPUOperatorContext()
    xp = backend.xp
    reg = float(cfg.reg_lambda)

    t0 = time.perf_counter()
    data_ctx = gpu_precompute_v1(
        backend,
        solver.kernel,
        solver.eps,
        solver.nufft_tol,
        data_ctx,
        op_ctx,
        l2scaled=solver.l2scaled,
        chunk_size=cfg.chunk_size,
    )
    t1 = time.perf_counter()

    if int(m) <= 0:
        raise ValueError("m must be > 0 for deflated CG.")

    # ----- Stage 1: cheap low-fidelity subspace Z -----
    Z0, subspace_diag = build_deflation_subspace(
        backend,
        data_ctx,
        reg,
        op_ctx,
        method=method,
        m=int(m),
        which=which,
        eig_solver=eig_solver,
        tol=float(lowfi_tol),
        n_iter=int(lowfi_n_iter),
        oversample=int(oversample),
        hm_t=hm_t,
        coarse_ratio=float(coarse_ratio),
        return_basis=True,
        freq_box_mode=str(freq_box_mode),
        seed=int(seed),
    )
    t2 = time.perf_counter()

    # ----- High-fidelity matvec wrappers (counted) -----
    hi_calls = {"vec": 0}

    def _apply_A_hi(v: Any, out: Any) -> None:
        apply_A_v1(backend, data_ctx, v, reg, op_ctx, out=out)
        hi_calls["vec"] += 1

    def _apply_A_hi_block(V_block: Any) -> Any:
        Vv = xp.asarray(V_block, dtype=xp.complex128)
        if Vv.ndim == 1:
            Vv = Vv.reshape(-1, 1)
        out_block = apply_A_block_v1(
            backend,
            data_ctx,
            Vv,
            reg,
            op_ctx,
            block_cols=block_cols,
        )
        hi_calls["vec"] += int(Vv.shape[1])
        return out_block

    # ----- Stage 2: high-fidelity calibration W = A Z, G = Z* A Z -----
    defl_data = build_deflation_data(
        backend,
        _apply_A_hi_block,
        Z0,
        rank_tol=float(rank_tol),
        jitter_ratio=float(jitter_ratio),
        block_cols=int(block_cols),
        z_storage_dtype=str(z_storage_dtype),
        compute_diagnostics=True,
        w_mode=str(w_mode),
        matvec_form=str(matvec_form),
    )
    hi_matvec_calibration = int(defl_data.m_eff)
    t3 = time.perf_counter()

    # ----- Stage 3: deflated CG (or DPCG) + recovery -----
    precond_kind = str(precond).lower()
    if precond_kind in ("none", "", "off"):
        beta_gpu, cg_it, def_relres, solve_diag = run_deflated_cg(
            backend,
            defl_data,
            _apply_A_hi,
            data_ctx.rhs_gpu,
            op_ctx,
            tol=cfg.tol,
            maxiter=cfg.maxiter,
            profile_components=cfg.profile_components,
        )
    elif precond_kind == "jacobi":
        mtot = int(data_ctx.meta["mtot"])
        dim = int(data_ctx.meta["dim"])
        weights_gpu = xp.asarray(data_ctx.weights_gpu_flat, dtype=xp.float64).reshape(-1)
        xtxcol_gpu = getattr(data_ctx, "xtxcol_gpu", None)
        if xtxcol_gpu is None:
            xtxcol_gpu = xp.ascontiguousarray(backend.fft.ifftn(data_ctx.gf_gpu))
            data_ctx.xtxcol_gpu = xtxcol_gpu
        t0_diag = xp.real(xtxcol_gpu[(int(mtot) - 1,) * dim])
        diag_A = (weights_gpu * weights_gpu) * xp.asarray(t0_diag, dtype=xp.float64) + reg
        diag_A = xp.maximum(diag_A, xp.asarray(1e-30, dtype=xp.float64))
        precond_fn = make_jacobi_precond(backend, 1.0 / diag_A)
        beta_gpu, cg_it, def_relres, solve_diag = run_deflated_pcg(
            backend,
            defl_data,
            _apply_A_hi,
            precond_fn,
            data_ctx.rhs_gpu,
            op_ctx,
            tol=cfg.tol,
            maxiter=cfg.maxiter,
            profile_components=cfg.profile_components,
        )
    else:
        raise ValueError(
            f"unknown precond={precond!r}; expected 'none' or 'jacobi'."
        )
    t4 = time.perf_counter()

    _ = predict_v1(backend, data_ctx, x, beta_gpu)
    t5 = time.perf_counter()

    n_cg = int(solve_diag.get("n_matvec", cg_it))
    hi_n_matvec = int(hi_calls["vec"])
    effective_formula = (
        "m_eff + 2*cg + 2"
        if str(getattr(defl_data, "w_mode", "dense")).lower() == "implicit"
        else "m_eff + cg + 1"
    )
    mem = deflation_memory_estimate(defl_data)
    defl_diag = dict(defl_data.diagnostics)

    diagnostics: dict[str, Any] = {
        "version": "v5_deflated_cg",
        "status": "ok",
        "deflation_method": str(method),
        "precond": precond_kind,
        "basis_kind": str(defl_diag.get("basis_kind", getattr(defl_data.basis, "kind", ""))),
        "w_mode": str(getattr(defl_data, "w_mode", "dense")),
        "matvec_form": str(getattr(defl_data, "matvec_form", "structured_ZH")),
        "effective_himatvec_formula": effective_formula,
        "which": str(which).upper(),
        "m_requested": int(m),
        "m_eff": int(defl_data.m_eff),
        "rank_dropped": int(defl_data.rank_dropped),
        "nufft_backend": backend.nufft_name,
        "nufft_stage": data_ctx.meta.get("nufft_stage"),
        # Convergence.
        "cg_iters": int(cg_it),
        "deflated_relres": float(def_relres),
        "true_relres": float(solve_diag.get("true_relres_from_recovery", float("nan"))),
        "cg_status": str(solve_diag.get("status", "")),
        "max_imag_ratio": float(solve_diag.get("max_imag_ratio", float("nan"))),
        # Matvec accounting (headline = hi_n_matvec).
        "lowfi_n_matvec": int(subspace_diag.get("lowfi_n_matvec", 0)),
        "lowfi_kind": subspace_diag.get("lowfi_kind"),
        "hi_n_matvec": hi_n_matvec,
        "hi_n_matvec_calibration": hi_matvec_calibration,
        "hi_n_matvec_cg": int(n_cg),
        "hi_n_matvec_actual": hi_n_matvec,
        "hi_n_matvec_unattributed": int(hi_n_matvec - hi_matvec_calibration),
        "hi_n_matvec_recovery": 1 if str(getattr(defl_data, "w_mode", "dense")).lower() == "dense" else 2,
        "cg_n_matvec": int(n_cg),
        # Deflation quality diagnostics.
        "cond_G": float(defl_data.cond_G),
        "jitter": float(defl_data.jitter),
        "hermitian_error_G": float(defl_diag.get("hermitian_error_G", float("nan"))),
        "invariance_leakage": float(defl_diag.get("invariance_leakage", float("nan"))),
        "deflation_exactness": float(defl_diag.get("deflation_exactness", float("nan"))),
        "orthonormality_error": float(defl_diag.get("orthonormality_error", float("nan"))),
        "diagnostics_mode": str(defl_diag.get("diagnostics_mode", "")),
        "b_def_ratio": float(solve_diag.get("b_def_ratio", float("nan"))),
        # Memory estimate.
        **mem,
        # Timings.
        "time_precompute": float(t1 - t0),
        "time_subspace": float(t2 - t1),
        "time_calibration": float(t3 - t2),
        "time_solve": float(t4 - t3),
        "time_predict": float(t5 - t4),
        "time_total": float(t5 - t0),
        # Solve-stage matvec timing.
        "t_matvec_avg": float(solve_diag.get("t_matvec_avg", float("nan"))),
        "t_matvec_total": float(solve_diag.get("t_matvec_total", float("nan"))),
        "n_matvec": int(n_cg),
        # Subspace estimation passthrough.
        "subspace_diag": subspace_diag,
        "device_name": backend.device_name,
        "has_nufft": backend.has_nufft,
        "chunk_size": cfg.chunk_size,
        "debug_finite_checks": bool(cfg.debug_finite_checks),
        "profile_components": bool(cfg.profile_components),
    }
    return V1Outputs(
        beta_gpu=beta_gpu,
        diagnostics=diagnostics,
        backend=backend,
        data_ctx=data_ctx,
    )


def run_v5_oracle_deflated_cg(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
    *,
    m: int = 64,
    side: str = "top",
    m_bottom: Optional[int] = None,
    eig_solver: str = "subspace_iter",
    eig_n_iter: int = 8,
    oversample: int = 16,
    rank_tol: float = 1e-12,
    jitter_ratio: float = 1e-12,
    block_cols: int = 8,
    z_storage_dtype: str = "same_as_A",
    w_mode: str = "dense",
    matvec_form: str = "structured_ZH",
) -> V1Outputs:
    """
    Oracle V5 experiment: use high-fidelity V3 Ritz vectors as ``Z`` and apply
    Frank-Vuik deflation only, with no EigenPro eigenvalue scaling.

    ``side`` selects the Ritz subspace:
      * ``"top"``: dominant Ritz vectors, matching the usual V3 top eigenspace.
      * ``"bottom"``: smallest Ritz vectors via eigsh ``SA``.
      * ``"two_sided"``: concatenate top ``m`` and bottom ``m_bottom`` vectors.
    """
    from .v1_ops import apply_A_block_v1, apply_A_v1

    backend = build_gpu_backend_bundle(cfg.backend)
    data_ctx = ensure_gpu_data_context(backend, x, y, state=None)
    data_ctx.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    op_ctx = GPUOperatorContext()
    xp = backend.xp
    reg = float(cfg.reg_lambda)

    t0 = time.perf_counter()
    data_ctx = gpu_precompute_v1(
        backend,
        solver.kernel,
        solver.eps,
        solver.nufft_tol,
        data_ctx,
        op_ctx,
        l2scaled=solver.l2scaled,
        chunk_size=cfg.chunk_size,
    )
    t1 = time.perf_counter()

    if int(m) <= 0:
        raise ValueError("m must be > 0 for oracle deflation.")

    hi_calls = {"vec": 0}

    def _apply_A_hi(v: Any, out: Any) -> None:
        apply_A_v1(backend, data_ctx, v, reg, op_ctx, out=out)
        hi_calls["vec"] += 1

    def _apply_A_hi_block(V_block: Any) -> Any:
        Vv = xp.asarray(V_block, dtype=xp.complex128)
        if Vv.ndim == 1:
            Vv = Vv.reshape(-1, 1)
        out_block = apply_A_block_v1(
            backend,
            data_ctx,
            Vv,
            reg,
            op_ctx,
            block_cols=block_cols,
        )
        hi_calls["vec"] += int(Vv.shape[1])
        return out_block

    def _estimate_ritz(q: int, which_code: str, solver_name: str) -> tuple[Any, Any, dict[str, Any]]:
        q = int(q)
        method_name = str(solver_name).lower()
        if which_code == "SA" and method_name in ("subspace_iter", "subspace"):
            method_name = "cupy_eigsh"
        eig_cfg = EigenspaceConfig(
            q_max=q,
            block_size=int(q + max(1, oversample)),
            n_iter=int(max(1, eig_n_iter)),
            method=method_name,
            eig_method=method_name,
        )
        # cupyx eigsh reads cfg.which directly.
        setattr(eig_cfg, "which", which_code)
        vals, vecs, diag = estimate_top_eigenspace_v3(
            backend=backend,
            apply_A_block_gpu=_apply_A_hi_block,
            size=int(data_ctx.rhs_gpu.size),
            cfg=eig_cfg,
        )
        diag = dict(diag)
        diag["which"] = which_code
        diag["eig_solver"] = method_name
        return vals, vecs, diag

    side_key = str(side).lower()
    if side_key in ("top", "lm", "largest"):
        vals, Z0, eig_diag = _estimate_ritz(int(m), "LA", eig_solver)
        oracle_side = "top"
    elif side_key in ("bottom", "sm", "smallest"):
        vals, Z0, eig_diag = _estimate_ritz(int(m), "SA", "cupy_eigsh")
        oracle_side = "bottom"
    elif side_key in ("two_sided", "twosided", "both"):
        mb = int(m_bottom if m_bottom is not None else max(1, int(m) // 2))
        mt = int(m)
        vals_t, Zt, diag_t = _estimate_ritz(mt, "LA", eig_solver)
        vals_b, Zb, diag_b = _estimate_ritz(mb, "SA", "cupy_eigsh")
        vals = xp.concatenate([xp.asarray(vals_t).reshape(-1), xp.asarray(vals_b).reshape(-1)])
        Z0 = xp.ascontiguousarray(xp.concatenate([Zt, Zb], axis=1))
        eig_diag = {
            "top": diag_t,
            "bottom": diag_b,
            "top_m": mt,
            "bottom_m": mb,
            "eig_solver": f"{diag_t.get('eig_solver')}/{diag_b.get('eig_solver')}",
            "which": "LA+SA",
        }
        oracle_side = "two_sided"
    else:
        raise ValueError("side must be 'top', 'bottom', or 'two_sided'.")
    t2 = time.perf_counter()
    oracle_eigen_n_matvec = int(hi_calls["vec"])

    defl_data = build_deflation_data(
        backend,
        _apply_A_hi_block,
        Z0,
        rank_tol=float(rank_tol),
        jitter_ratio=float(jitter_ratio),
        block_cols=int(block_cols),
        z_storage_dtype=str(z_storage_dtype),
        compute_diagnostics=True,
        w_mode=str(w_mode),
        matvec_form=str(matvec_form),
    )
    hi_matvec_calibration = int(defl_data.m_eff)
    t3 = time.perf_counter()

    beta_gpu, cg_it, def_relres, solve_diag = run_deflated_cg(
        backend,
        defl_data,
        _apply_A_hi,
        data_ctx.rhs_gpu,
        op_ctx,
        tol=cfg.tol,
        maxiter=cfg.maxiter,
        profile_components=cfg.profile_components,
        work_prefix=f"oracle_{oracle_side}",
    )
    t4 = time.perf_counter()

    _ = predict_v1(backend, data_ctx, x, beta_gpu)
    t5 = time.perf_counter()

    n_cg = int(solve_diag.get("n_matvec", cg_it))
    hi_n_matvec = int(hi_calls["vec"])
    mem = deflation_memory_estimate(defl_data)
    defl_diag = dict(defl_data.diagnostics)
    vals_real = xp.real(xp.asarray(vals).reshape(-1))

    diagnostics: dict[str, Any] = {
        "version": "v5_oracle_deflated_cg",
        "status": "ok",
        "deflation_method": "oracle_v3_ritz",
        "oracle_side": oracle_side,
        "precond": "none",
        "basis_kind": str(defl_diag.get("basis_kind", getattr(defl_data.basis, "kind", ""))),
        "w_mode": str(getattr(defl_data, "w_mode", "dense")),
        "matvec_form": str(getattr(defl_data, "matvec_form", "structured_ZH")),
        "effective_himatvec_formula": "oracle_eigen + m_eff + cg + 1",
        "which": str(eig_diag.get("which", "")),
        "m_requested": int(m),
        "m_bottom": int(m_bottom or 0),
        "m_eff": int(defl_data.m_eff),
        "rank_dropped": int(defl_data.rank_dropped),
        "oracle_lambda_min": float(xp.min(vals_real)) if int(vals_real.size) else float("nan"),
        "oracle_lambda_max": float(xp.max(vals_real)) if int(vals_real.size) else float("nan"),
        "oracle_eig_diag": eig_diag,
        "nufft_backend": backend.nufft_name,
        "nufft_stage": data_ctx.meta.get("nufft_stage"),
        "cg_iters": int(cg_it),
        "deflated_relres": float(def_relres),
        "true_relres": float(solve_diag.get("true_relres_from_recovery", float("nan"))),
        "cg_status": str(solve_diag.get("status", "")),
        "max_imag_ratio": float(solve_diag.get("max_imag_ratio", float("nan"))),
        "lowfi_n_matvec": 0,
        "lowfi_kind": "none_oracle_high_fidelity_ritz",
        "hi_n_matvec": hi_n_matvec,
        "hi_n_matvec_oracle_eigen": oracle_eigen_n_matvec,
        "hi_n_matvec_calibration": hi_matvec_calibration,
        "hi_n_matvec_cg": int(n_cg),
        "hi_n_matvec_recovery": 1,
        "hi_n_matvec_actual": hi_n_matvec,
        "cg_n_matvec": int(n_cg),
        "cond_G": float(defl_data.cond_G),
        "jitter": float(defl_data.jitter),
        "hermitian_error_G": float(defl_diag.get("hermitian_error_G", float("nan"))),
        "invariance_leakage": float(defl_diag.get("invariance_leakage", float("nan"))),
        "deflation_exactness": float(defl_diag.get("deflation_exactness", float("nan"))),
        "orthonormality_error": float(defl_diag.get("orthonormality_error", float("nan"))),
        "diagnostics_mode": str(defl_diag.get("diagnostics_mode", "")),
        "b_def_ratio": float(solve_diag.get("b_def_ratio", float("nan"))),
        **mem,
        "time_precompute": float(t1 - t0),
        "time_subspace": float(t2 - t1),
        "time_calibration": float(t3 - t2),
        "time_solve": float(t4 - t3),
        "time_predict": float(t5 - t4),
        "time_total": float(t5 - t0),
        "t_matvec_avg": float(solve_diag.get("t_matvec_avg", float("nan"))),
        "t_matvec_total": float(solve_diag.get("t_matvec_total", float("nan"))),
        "n_matvec": int(n_cg),
        "device_name": backend.device_name,
        "has_nufft": backend.has_nufft,
        "chunk_size": cfg.chunk_size,
        "debug_finite_checks": bool(cfg.debug_finite_checks),
        "profile_components": bool(cfg.profile_components),
    }
    return V1Outputs(
        beta_gpu=beta_gpu,
        diagnostics=diagnostics,
        backend=backend,
        data_ctx=data_ctx,
    )
