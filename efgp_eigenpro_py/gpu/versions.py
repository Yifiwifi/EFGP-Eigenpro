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
    GPUPreconditionerData,
    apply_preconditioner_dominant_subspace,
    apply_preconditioner_coordinate_nystrom,
    apply_preconditioner_v2,
    build_dominant_subspace_preconditioner,
    build_coordinate_nystrom_preconditioner_data,
    build_gpu_preconditioner_data,
)
from .iterative_solvers import pcg_solve_gpu
from .v3_eigenspace import (
    EigenspaceConfig,
    estimate_top_eigenspace_eigenpro_nystrom,
    estimate_top_eigenspace_v3,
    mu_for_precond_from_eig,
)


@dataclass
class GPURunConfig:
    reg_lambda: float
    tol: float = 1e-8
    maxiter: int = 2000
    chunk_size: Optional[int] = None
    debug_finite_checks: bool = False
    backend: BackendConfig = BackendConfig()


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
            "device_name": backend.device_name,
            "has_nufft": backend.has_nufft,
            "chunk_size": cfg.chunk_size,
            "debug_finite_checks": bool(cfg.debug_finite_checks),
        },
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
        },
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
    ):
        eig_cfg.method_cfg = dict(eig_cfg.method_cfg or {})
        eig_cfg.method_cfg.setdefault("data_ctx", data_ctx)
        eig_cfg.method_cfg.setdefault("reg_lambda", float(cfg.reg_lambda))

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

    vals_gpu, vecs_gpu, eig_diag = estimate_top_eigenspace_v3(
        backend=backend,
        apply_A_block_gpu=_apply_A_block,
        size=int(data_ctx.rhs_gpu.size),
        cfg=eig_cfg,
    )
    t2 = time.perf_counter()

    q = int(eig_cfg.q_max)
    precond_kind = str(eig_diag.get("precond_kind", "full_eigenpro")).lower()
    if precond_kind == "coordinate_nystrom":
        precond_data: GPUPreconditionerData | CoordinateNystromPreconditionerData = (
            build_coordinate_nystrom_preconditioner_data(
                backend,
                eig_diag["S_gpu"],
                eig_diag["V_gpu"],
                eig_diag["theta_gpu"],
                float(eig_diag["mu"]),
            )
        )
    else:
        mu = mu_for_precond_from_eig(vals_gpu, q, eig_diag)
        scale_gpu = backend.xp.asarray(1.0 - (mu / vals_gpu[:q]))
        precond_data = GPUPreconditionerData(
            U_gpu=vecs_gpu[:, :q],
            UH_gpu=vecs_gpu[:, :q].conj().T,
            scale_gpu=scale_gpu,
            scale_col_gpu=scale_gpu.reshape(-1, 1),
        )
    t3 = time.perf_counter()

    def _matvec(v: Any, out: Any) -> None:
        from .v1_ops import apply_A_v1

        apply_A_v1(backend, data_ctx, v, float(cfg.reg_lambda), op_ctx, out=out)

    def _precond(v: Any, out: Any) -> None:
        if precond_kind == "coordinate_nystrom" or all(
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
        },
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
    else:
        mu = mu_for_precond_from_eig(vals_gpu, q, eig_diag)
        scale_gpu = backend.xp.asarray(1.0 - (mu / vals_gpu[:q]))
        precond_data = GPUPreconditionerData(
            U_gpu=vecs_gpu[:, :q],
            UH_gpu=vecs_gpu[:, :q].conj().T,
            scale_gpu=scale_gpu,
            scale_col_gpu=scale_gpu.reshape(-1, 1),
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
        if precond_kind == "coordinate_nystrom" or all(
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
        },
    )
