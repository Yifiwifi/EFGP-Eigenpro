"""
SLQ / Lanczos spectrum on the *same* matvec that PCG uses in each benchmark mode.

- ``gpu_v1_topq0``: original SPD operator ``A`` (``apply_A_v1``).
- ``gpu_v3_topq`` / ``gpu_v3_topq_combo``: left-preconditioned map ``v -> P(A v)``
  (``P`` = ``apply_preconditioner_v2``), matching ``pcg_solve_gpu``'s
  matvec+precond composition for residual updates.

Note: ``P(A)`` is generally not Hermitian; SLQ will track ``Im(alpha)`` in diagnostics.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np

from ..efgp_solver import EFGPSolver
from ..discretization import GridSpec
from . import surrogate_ops as sur_ops
from .backends import build_gpu_backend_bundle
from .contexts import GPUOperatorContext, ensure_gpu_data_context
from .v1_ops import apply_A_v1, gpu_precompute_v1
from .v2_preconditioner import GPUPreconditionerData, apply_preconditioner_v2
from .v3_eigenspace import EigenspaceConfig, estimate_top_eigenspace_v3
from .versions import GPURunConfig


def _build_surrogate_grid(fine_m: int, fine_h: float, grid_scale: Any = None, grid_m: Any = None) -> GridSpec:
    if grid_m is not None:
        m_coarse = int(grid_m)
    else:
        scale = 1.0 if grid_scale is None else float(grid_scale)
        m_coarse = max(1, int(fine_m * scale))
    xis = np.arange(-m_coarse, m_coarse + 1) * float(fine_h)
    return GridSpec(xis=xis, h=float(fine_h), mtot=xis.size, hm=m_coarse)


def _embed_coarse_to_fine(xp: Any, vecs_coarse: Any, coarse_m: int, fine_m: int, dim: int = 2) -> Any:
    if coarse_m == fine_m:
        return xp.asarray(vecs_coarse)
    cols = []
    for j in range(vecs_coarse.shape[1]):
        v = xp.asarray(vecs_coarse[:, j]).reshape((2 * coarse_m + 1,) * dim)
        vf = xp.zeros((2 * fine_m + 1,) * dim, dtype=v.dtype)
        s = fine_m - coarse_m
        e = s + (2 * coarse_m + 1)
        if dim == 2:
            vf[s:e, s:e] = v
        else:
            vf[(slice(s, e),) * dim] = v
        cols.append(vf.reshape(-1))
    return xp.stack(cols, axis=1)


def build_unprecond_A_matvec(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
) -> Tuple[Any, Callable[[Any, Any], None], int, dict[str, Any]]:
    """``v |-> A v`` (unpreconditioned EFGP normal operator)."""
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
    n = int(data_ctx.rhs_gpu.size)

    def matvec(v: Any, out: Any) -> None:
        xp = backend.xp
        va = xp.asarray(v, dtype=xp.complex128).reshape(-1)
        oa = xp.asarray(out, dtype=xp.complex128).reshape(-1)
        apply_A_v1(backend, data_ctx, va, float(cfg.reg_lambda), op_ctx, out=oa)

    meta: dict[str, Any] = {
        "slq_spectrum": "A",
        "slq_spectrum_desc": "Unpreconditioned A: apply_A_v1 only (same as plain CG on A).",
    }
    return backend, matvec, n, meta


def _build_v3_pcg_left_precond_matvec_local(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
    eig_cfg: EigenspaceConfig,
) -> Tuple[Any, Callable[[Any, Any], None], int, dict[str, Any]]:
    """
    Local fallback for ``v -> P(A v)`` to avoid hard dependency on
    ``versions.build_v3_pcg_left_precond_matvec`` during notebook hot-reload.
    """
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
        vb = xp.asarray(v_block, dtype=xp.complex128)
        if vb.ndim == 1:
            vb = vb.reshape(-1, 1)
        out_block = xp.empty_like(vb)
        for i in range(vb.shape[1]):
            apply_A_v1(
                backend,
                data_ctx,
                vb[:, i],
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
    q = int(eig_cfg.q_max)
    if vals_gpu.size <= q:
        mu = float(vals_gpu[-1])
    else:
        mu = float(vals_gpu[q])
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
        apply_preconditioner_v2(backend, precond_data, av_buf[0], op_ctx=op_ctx, out=oa)

    meta: dict[str, Any] = {
        "slq_spectrum": "M_inv_A",
        "slq_spectrum_desc": "P(A v); same P as apply_preconditioner_v2 in PCG.",
        "top_q": int(q),
        "eig_residual_fro_rel": float(eig_diag.get("residual_fro_rel", float("nan"))),
    }
    return backend, matvec, n, meta


def build_v3_combo_pcg_left_precond_matvec(
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
    top_q: int,
    combo_cfg: dict,
    v3_oversample: int,
    dim: int = 2,
) -> Tuple[Any, Callable[[Any, Any], None], int, dict[str, Any]]:
    """
    Same as ``_run_v3_combo_case`` in the benchmark notebook up to the PCG call:
    return ``P(A v)`` for the *fine* grid operator and the combo-built preconditioner.
    """
    backend = build_gpu_backend_bundle(cfg.backend)
    xp = backend.xp
    q_max = int(top_q)

    data_ctx = ensure_gpu_data_context(backend, x, y, state=None)
    data_ctx.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    op_ctx = GPUOperatorContext()

    data_ctx = sur_ops.gpu_precompute_v1(
        backend,
        solver.kernel,
        solver.eps,
        solver.nufft_tol,
        data_ctx,
        op_ctx,
        l2scaled=solver.l2scaled,
        chunk_size=cfg.chunk_size,
    )

    def _apply_block(local_ctx: Any, local_op_ctx: Any):
        def _fn(V: Any) -> Any:
            Vv = xp.asarray(V, dtype=xp.complex128)
            if Vv.ndim == 1:
                Vv = Vv.reshape(-1, 1)
            out = xp.empty_like(Vv)
            for i in range(Vv.shape[1]):
                apply_A_v1(
                    backend,
                    local_ctx,
                    Vv[:, i],
                    float(cfg.reg_lambda),
                    local_op_ctx,
                    out=out[:, i],
                )
            return out

        return _fn

    apply_A_fine = _apply_block(data_ctx, op_ctx)
    fine_mtot = int(data_ctx.meta.get("mtot", 0))
    fine_m = max(1, (fine_mtot - 1) // 2)
    fine_h = float(data_ctx.meta.get("h", 1.0))

    frac = float(combo_cfg.get("subsample_frac", 1.0))
    frac = min(max(frac, 1e-6), 1.0)
    n_sub = max(q_max + 2, int(len(x) * frac))
    n_sub = min(n_sub, len(x))
    seed = int(combo_cfg.get("subsample_seed", 0))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n_sub, replace=False)
    x_use = np.asarray(x[idx])
    y_use = np.asarray(y[idx])

    eps_factor = float(combo_cfg.get("eps_factor", 1.0))
    eps_sur = float(solver.eps) * eps_factor
    grid_override = _build_surrogate_grid(
        fine_m,
        fine_h,
        grid_scale=combo_cfg.get("grid_scale", 1.0),
        grid_m=combo_cfg.get("grid_m", None),
    )
    data_ctx_sur = ensure_gpu_data_context(backend, x_use, y_use, state=None)
    data_ctx_sur.meta["debug_finite_checks"] = bool(cfg.debug_finite_checks)
    op_ctx_sur = GPUOperatorContext()
    data_ctx_sur = sur_ops.gpu_precompute_v1(
        backend,
        solver.kernel,
        eps_sur,
        solver.nufft_tol,
        data_ctx_sur,
        op_ctx_sur,
        l2scaled=solver.l2scaled,
        chunk_size=cfg.chunk_size,
        grid_override=grid_override,
    )
    size_sur = int(data_ctx_sur.rhs_gpu.size)
    if size_sur <= q_max:
        raise ValueError(f"surrogate size ({size_sur}) must be > top_q ({q_max})")

    sur_oversample = int(combo_cfg.get("oversample", v3_oversample))
    sur_block = combo_cfg.get("block_size")
    sur_block = int(q_max + sur_oversample) if sur_block is None else int(sur_block)
    sur_block = min(sur_block, size_sur)
    if sur_block <= q_max:
        sur_block = q_max + 1

    apply_A_sur = _apply_block(data_ctx_sur, op_ctx_sur)
    eig_cfg_sur = sur_ops.EigenspaceConfig(
        q_max=q_max,
        block_size=sur_block,
        n_iter=int(combo_cfg.get("sur_iter", 1)),
    )
    vals_sur, vecs_sur, _ = sur_ops.estimate_top_eigenspace_v3(
        backend=backend,
        apply_A_block_gpu=apply_A_sur,
        size=size_sur,
        cfg=eig_cfg_sur,
    )
    init_Q = _embed_coarse_to_fine(xp, vecs_sur, int(grid_override.hm), fine_m, dim=dim)
    sur_block2 = int(max(sur_block, q_max + 1))
    eig_cfg_fine = sur_ops.EigenspaceConfig(
        q_max=q_max,
        block_size=sur_block2,
        n_iter=int(combo_cfg.get("refine_iter", 1)),
    )
    vals_gpu, vecs_gpu, eig_diag = sur_ops.estimate_top_eigenspace_v3(
        backend=backend,
        apply_A_block_gpu=apply_A_fine,
        size=int(data_ctx.rhs_gpu.size),
        cfg=eig_cfg_fine,
        init_Q=init_Q,
    )
    if vals_gpu.size <= q_max:
        mu = float(vals_gpu[-1])
    else:
        mu = float(vals_gpu[q_max])
    scale_gpu = xp.asarray(1.0 - (mu / vals_gpu[:q_max]))
    precond_data = GPUPreconditionerData(
        U_gpu=vecs_gpu[:, :q_max],
        UH_gpu=vecs_gpu[:, :q_max].conj().T,
        scale_gpu=scale_gpu,
        scale_col_gpu=scale_gpu.reshape(-1, 1),
    )
    n = int(data_ctx.rhs_gpu.size)
    av_buf: list[Any] = [None]

    def matvec(v: Any, out: Any) -> None:
        va = xp.asarray(v, dtype=xp.complex128).reshape(-1)
        if av_buf[0] is None or int(av_buf[0].size) != int(va.size):
            av_buf[0] = xp.empty((int(va.size),), dtype=xp.complex128)
        oa = xp.asarray(out, dtype=xp.complex128).reshape(-1)
        apply_A_v1(backend, data_ctx, va, float(cfg.reg_lambda), op_ctx, out=av_buf[0])
        apply_preconditioner_v2(backend, precond_data, av_buf[0], op_ctx=op_ctx, out=oa)

    meta: dict[str, Any] = {
        "slq_spectrum": "M_inv_A",
        "slq_spectrum_desc": "gpu_v3_topq_combo: P(A v) on fine grid with combo eigenspace precond; matches PCG in benchmark.",
        "top_q": int(q_max),
        "eig_residual_fro_rel": float(eig_diag.get("residual_fro_rel", float("nan"))),
        "surrogate_tag": str(combo_cfg.get("name", "combo")),
    }
    return backend, matvec, n, meta


def build_slq_matvec_for_benchmark_mode(
    mode: str,
    solver: EFGPSolver,
    x: np.ndarray,
    y: np.ndarray,
    cfg: GPURunConfig,
    *,
    top_q: int = 0,
    combo_cfg: Optional[dict] = None,
    v3_oversample: int = 16,
    v3_n_iter: int = 3,
    dim: int = 2,
) -> Tuple[Any, Callable[[Any, Any], None], int, dict[str, Any]]:
    """
    Dispatch SLQ matvec to match the benchmark mode's PCG/ CG operator.
    """
    m = str(mode).strip()
    if m == "gpu_v1_topq0":
        return build_unprecond_A_matvec(solver, x, y, cfg)
    if m == "gpu_v3_topq":
        if int(top_q) <= 0:
            raise ValueError("top_q must be > 0 for gpu_v3_topq")
        eig_cfg = EigenspaceConfig(
            q_max=int(top_q),
            block_size=int(top_q + v3_oversample),
            n_iter=int(v3_n_iter),
        )
        return _build_v3_pcg_left_precond_matvec_local(solver, x, y, cfg, eig_cfg)
    if m == "gpu_v3_topq_combo":
        if int(top_q) <= 0:
            raise ValueError("top_q must be > 0 for gpu_v3_topq_combo")
        if not isinstance(combo_cfg, dict):
            raise TypeError("combo_cfg must be a dict for gpu_v3_topq_combo")
        return build_v3_combo_pcg_left_precond_matvec(
            solver, x, y, cfg, int(top_q), combo_cfg, v3_oversample, dim=dim
        )
    raise ValueError(f"Unsupported SLQ benchmark mode: {mode}")
