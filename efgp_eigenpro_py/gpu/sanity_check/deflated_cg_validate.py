from __future__ import annotations

"""
deflated_cg_validate.py
=======================

Minimal validation / comparison harness for multi-fidelity deflated CG (V5).

It logs everything to a timestamped ``.txt`` (no reliance on an interactive
terminal, friendly to SSH/remote runs):

1. Correctness: V5 ``true_relres`` aligns with baseline V1 plain CG to ``cg_tol``.
2. Four-way comparison (headline = high-fidelity matvec count, not iters):
     - baseline CG                 (run_v1_pure_efgp)
     - eigenvalue PCG              (run_v3_full_gpu_eigenspace)
     - low-A deflation             (run_v5_deflated_cg: coord_nystrom / freq_trunc / float32)
3. freq_trunc indexing sanity tests:
     (a) hm_t == hm  -> coarse central slice equals the fine arrays (identity).
     (b) A_fine(embed v_c) central restriction ~= A_coarse v_c  (convention check).

Run (on the GPU machine):
    python -m efgp_eigenpro_py.gpu.sanity_check.deflated_cg_validate
"""

import os
import time
import traceback
from datetime import datetime
from types import SimpleNamespace

import numpy as np


def _log_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(here, f"deflated_cg_validate_{stamp}.txt")


class _Logger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        line = str(msg)
        print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _fmt(d: dict, keys: list[str]) -> str:
    parts = []
    for k in keys:
        v = d.get(k)
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def main() -> None:
    log = _Logger(_log_path())
    log(f"# deflated CG validation  {datetime.now().isoformat()}")
    log(f"# log file: {log.path}")

    try:
        from ...kernels import make_squared_exponential
        from ...efgp_solver import EFGPSolver
        from ...toy_data import generate_toy_data
        from ..versions import (
            GPURunConfig,
            run_v1_pure_efgp,
            run_v3_full_gpu_eigenspace,
            run_v5_deflated_cg,
        )
        from ..v3_eigenspace import EigenspaceConfig
        from ..backends import build_gpu_backend_bundle
        from ..contexts import ensure_gpu_data_context, GPUOperatorContext
        from ..v1_ops import gpu_precompute_v1, apply_A_v1
        from ..deflation_subspace import make_coarse_ctx, embed_freq
        from ..deflation_core import build_deflation_data, run_deflated_cg
        from ..iterative_solvers import cg_solve_gpu
    except Exception:
        log("IMPORT FAILED:")
        log(traceback.format_exc())
        log.close()
        return

    # ---- problem setup ----
    n_samples = 6000
    lengthscale = 0.02
    reg_lambda = 1e-4
    eps = 1e-6
    nufft_tol = 1e-10
    cg_tol = 1e-6
    maxiter = 1000
    m = 16

    xi, yi = generate_toy_data(n_samples=n_samples, nu=3.0, sigma_eps=0.1, seed=7)
    x = np.asarray(xi, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(yi, dtype=np.float64).reshape(-1)
    kernel = make_squared_exponential(lengthscale=lengthscale, dim=1, variance=1.0)
    solver = EFGPSolver(kernel, reg_lambda=reg_lambda, eps=eps, nufft_tol=nufft_tol)
    cfg = GPURunConfig(reg_lambda=reg_lambda, tol=cg_tol, maxiter=maxiter)

    log("")
    log(f"problem: n_samples={n_samples}, l={lengthscale}, reg_lambda={reg_lambda}, "
        f"eps={eps}, cg_tol={cg_tol}, m={m}")

    # ---- baseline V1 ----
    base_relres = float("nan")
    try:
        t0 = time.perf_counter()
        out1 = run_v1_pure_efgp(solver, x, y, cfg)
        dt = time.perf_counter() - t0
        d1 = out1.diagnostics
        base_relres = float(d1.get("cg_relres", float("nan")))
        log("")
        log("[V1 baseline CG]")
        log("  " + _fmt(d1, ["cg_iters", "cg_relres", "n_matvec", "time_total"]))
        log(f"  wall={dt:.3f}s, mtot={out1.data_ctx.meta.get('mtot')}")
    except Exception:
        log("[V1] FAILED:")
        log(traceback.format_exc())
        log.close()
        return

    # ---- eigenvalue PCG V3 ----
    try:
        eig_cfg = EigenspaceConfig(q_max=m, block_size=m + 8, n_iter=8, method="subspace_iter")
        t0 = time.perf_counter()
        out3 = run_v3_full_gpu_eigenspace(solver, x, y, cfg, eig_cfg=eig_cfg)
        dt = time.perf_counter() - t0
        d3 = out3.diagnostics
        hi3 = int(d3.get("eigen_n_matvec", 0)) + int(d3.get("cg_n_matvec", 0))
        log("")
        log("[V3 eigenvalue PCG]")
        log("  " + _fmt(d3, ["cg_iters", "cg_relres", "eigen_n_matvec", "cg_n_matvec", "time_total"]))
        log(f"  hi_n_matvec(eigen+cg)={hi3}")
    except Exception:
        log("[V3] FAILED (continuing):")
        log(traceback.format_exc())

    # ---- V5 deflated CG: three methods ----
    methods = [
        ("freq_box", {"freq_box_mode": "center"}),
        ("coord_nystrom", {}),
        ("freq_trunc", {"coarse_ratio": 0.5}),
        ("float32", {}),
        # Phase 2: deflated preconditioned CG (DPCG) with Jacobi preconditioner.
        ("coord_nystrom", {"precond": "jacobi"}),
    ]
    for method, kw in methods:
        try:
            t0 = time.perf_counter()
            out5 = run_v5_deflated_cg(solver, x, y, cfg, m=m, method=method, **kw)
            dt = time.perf_counter() - t0
            d5 = out5.diagnostics
            tag = method + (f"+{kw['precond']}" if kw.get("precond") else "")
            log("")
            log(f"[V5 deflated CG: {tag}]")
            log("  " + _fmt(d5, [
                "m_eff", "rank_dropped", "cg_iters", "deflated_relres", "true_relres",
                "cg_status",
            ]))
            log("  " + _fmt(d5, [
                "lowfi_n_matvec", "hi_n_matvec", "hi_n_matvec_calibration",
                "hi_n_matvec_cg", "hi_n_matvec_recovery", "hi_n_matvec_actual",
            ]))
            log("  " + _fmt(d5, [
                "basis_kind", "w_mode", "matvec_form", "effective_himatvec_formula",
            ]))
            log("  " + _fmt(d5, [
                "cond_G", "hermitian_error_G", "invariance_leakage",
                "deflation_exactness", "orthonormality_error", "b_def_ratio", "max_imag_ratio",
            ]))
            log("  " + _fmt(d5, [
                "estimated_total_deflation_GB", "time_subspace", "time_calibration",
                "time_solve", "time_total",
            ]))
            ok = (
                np.isfinite(d5.get("true_relres", np.nan))
                and d5.get("true_relres", 1.0) <= max(10 * cg_tol, base_relres * 10 + cg_tol)
            )
            log(f"  CORRECTNESS true_relres<=~10*cg_tol: {'PASS' if ok else 'CHECK'}")
        except Exception:
            log(f"[V5 {method} {kw}] FAILED (continuing):")
            log(traceback.format_exc())

    # ---- freq_trunc indexing sanity tests ----
    try:
        log("")
        log("[freq_trunc indexing sanity]")
        backend = build_gpu_backend_bundle(cfg.backend)
        xp = backend.xp
        dctx = ensure_gpu_data_context(backend, x, y, state=None)
        opx = GPUOperatorContext()
        dctx = gpu_precompute_v1(
            backend, solver.kernel, solver.eps, solver.nufft_tol, dctx, opx,
            l2scaled=solver.l2scaled,
        )
        mtot = int(dctx.meta["mtot"])
        dim = int(dctx.meta["dim"])
        hm = (mtot - 1) // 2

        # (a) hm_t == hm identity: central slice formulas must reproduce the fine arrays.
        weights_flat = xp.asarray(dctx.weights_gpu_flat).reshape(-1)
        w_nd = weights_flat.reshape((mtot,) * dim)
        sl_w = tuple(slice(hm - hm, hm + hm + 1) for _ in range(dim))
        w_id = w_nd[sl_w].reshape(-1)
        xtx = dctx.xtxcol_gpu
        if xtx is None:
            xtx = xp.ascontiguousarray(backend.fft.ifftn(dctx.gf_gpu))
        start = mtot - mtot
        stop = start + (2 * mtot - 1)
        sl_x = tuple(slice(start, stop) for _ in range(dim))
        xtx_id = xtx[sl_x]
        w_err = float(xp.max(xp.abs(w_id - weights_flat)))
        x_err = float(xp.max(xp.abs(xtx_id - xtx)))
        log(f"  (a) identity hm_t==hm: weights_max_abs_err={w_err:.3e}, xtxcol_max_abs_err={x_err:.3e}  "
            f"-> {'PASS' if (w_err < 1e-12 and x_err < 1e-12) else 'FAIL'}")

        # (b) coarse vs fine central restriction.
        hm_t = max(1, hm // 2)
        coarse, mtot_t = make_coarse_ctx(backend, dctx, hm_t)
        cop = GPUOperatorContext()
        rng = np.random.default_rng(0)
        vc = xp.asarray(
            rng.standard_normal(mtot_t ** dim) + 1j * rng.standard_normal(mtot_t ** dim),
            dtype=xp.complex128,
        )
        a_coarse = apply_A_v1(backend, coarse, vc, reg_lambda, cop)
        vc_embed = embed_freq(xp, vc.reshape(-1, 1), mtot, mtot_t, dim)[:, 0]
        a_fine_full = apply_A_v1(backend, dctx, vc_embed, reg_lambda, opx)
        a_fine_nd = a_fine_full.reshape((mtot,) * dim)
        sl_c = tuple(slice(hm - hm_t, hm + hm_t + 1) for _ in range(dim))
        a_fine_restr = a_fine_nd[sl_c].reshape(-1)
        rel = float(xp.linalg.norm(a_fine_restr - a_coarse) / max(float(xp.linalg.norm(a_coarse)), 1e-30))
        log(f"  (b) hm_t={hm_t}, mtot_t={mtot_t}: ||restrict(A_fine embed v) - A_coarse v||/||.|| = {rel:.3e}  "
            f"-> {'PASS' if rel < 1e-2 else 'CHECK (truncation diff expected to be small but nonzero)'}")
    except Exception:
        log("[freq_trunc sanity] FAILED (continuing):")
        log(traceback.format_exc())

    # ---- toy dense SPD sanity: exact eigenvectors as Z ----
    try:
        log("")
        log("[toy dense SPD sanity]")

        class _NpLinalg:
            @staticmethod
            def norm(x, *args, **kwargs):
                return np.linalg.norm(x, *args, **kwargs)

            @staticmethod
            def vdot(a, b):
                return np.vdot(a, b)

        backend_np = SimpleNamespace(xp=np, linalg=_NpLinalg())
        op_np = SimpleNamespace()
        rng = np.random.default_rng(123)
        n_toy = 96
        m_toy = 12

        # Strong top outliers: deflating the largest eigenmodes should sharply reduce kappa.
        evals_desc = np.geomspace(1e6, 1.0, num=n_toy).astype(np.float64)
        G = rng.standard_normal((n_toy, n_toy))
        Q, _ = np.linalg.qr(G)
        A_toy = (Q * evals_desc.reshape(1, -1)) @ Q.T
        b_toy = rng.standard_normal(n_toy).astype(np.float64)
        norm_b_toy = max(float(np.linalg.norm(b_toy)), 1e-300)

        def _apply_A_dense(v, out):
            vv = np.asarray(v, dtype=np.complex128).reshape(-1)
            np.copyto(out, A_toy @ vv)

        def _apply_A_dense_block(V):
            VV = np.asarray(V, dtype=np.complex128)
            if VV.ndim == 1:
                VV = VV.reshape(-1, 1)
            return np.ascontiguousarray(A_toy @ VV)

        def _run_plain_dense():
            def _mv(v, out):
                _apply_A_dense(v, out)

            x, it, relres, stats = cg_solve_gpu(
                backend_np,
                _mv,
                np.asarray(b_toy, dtype=np.complex128),
                op_np,
                1e-10,
                2000,
                return_stats=True,
                work_prefix="toy_plain",
                profile_components=False,
            )
            x = np.asarray(x)
            true_rel = float(np.linalg.norm(b_toy - A_toy @ x) / norm_b_toy)
            return it, relres, true_rel, stats

        def _run_deflated_dense(Z_exact, tag):
            data = build_deflation_data(
                backend_np,
                _apply_A_dense_block,
                np.asarray(Z_exact, dtype=np.complex128),
                rank_tol=1e-14,
                jitter_ratio=1e-14,
                block_cols=8,
                compute_diagnostics=True,
            )
            x, it, def_rel, diag = run_deflated_cg(
                backend_np,
                data,
                _apply_A_dense,
                np.asarray(b_toy, dtype=np.complex128),
                op_np,
                tol=1e-10,
                maxiter=2000,
                profile_components=False,
                work_prefix=f"toy_{tag}",
            )
            x = np.asarray(x)
            true_rel = float(np.linalg.norm(b_toy - A_toy @ x) / norm_b_toy)
            return data, it, def_rel, true_rel, diag

        def _actual_deflated_kappa(data):
            Z = np.asarray(data.basis.to_dense(), dtype=np.complex128)
            W = np.asarray(data.W, dtype=np.complex128)
            Ginv = np.linalg.inv(np.asarray(data.G, dtype=np.complex128))
            # H = P_D A = A - A Z (Z* A Z)^{-1} Z* A
            H = np.asarray(A_toy, dtype=np.complex128) - W @ Ginv @ (W.conj().T)
            H = 0.5 * (H + H.conj().T)
            evals_H = np.real(np.linalg.eigvalsh(H))
            pos = evals_H[evals_H > max(1e-12, 1e-10 * np.max(np.abs(evals_H)))]
            if pos.size == 0:
                return float("nan"), evals_H
            return float(np.max(pos) / np.min(pos)), evals_H

        plain_it, plain_rel, plain_true_rel, _plain_stats = _run_plain_dense()
        kappa_plain = float(evals_desc[0] / evals_desc[-1])
        log("  plain: " + _fmt({
            "n": n_toy,
            "m": m_toy,
            "kappa_plain": kappa_plain,
            "cg_iters": plain_it,
            "cg_relres": plain_rel,
            "true_relres": plain_true_rel,
        }, ["n", "m", "kappa_plain", "cg_iters", "cg_relres", "true_relres"]))

        # Top deflation: remove dominant eigenmodes.
        Z_top = Q[:, :m_toy]
        data_top, it_top, rel_top, true_top, diag_top = _run_deflated_dense(Z_top, "top")
        retained_top = evals_desc[m_toy:]
        kappa_top_pred = float(retained_top[0] / retained_top[-1])
        kappa_top_actual, evals_top = _actual_deflated_kappa(data_top)
        log("  top exact-Z deflation: " + _fmt({
            "m_eff": data_top.m_eff,
            "kappa_pred": kappa_top_pred,
            "kappa_eff_actual": kappa_top_actual,
            "cg_iters": it_top,
            "deflated_relres": rel_top,
            "true_relres": true_top,
            "orthonormality_error": data_top.diagnostics.get("orthonormality_error", np.nan),
        }, ["m_eff", "kappa_pred", "kappa_eff_actual", "cg_iters", "deflated_relres", "true_relres", "orthonormality_error"]))

        # Bottom deflation: remove the smallest eigenmodes; useful as a contrast sanity check.
        Z_bot = Q[:, -m_toy:]
        data_bot, it_bot, rel_bot, true_bot, diag_bot = _run_deflated_dense(Z_bot, "bottom")
        retained_bot = evals_desc[:-m_toy]
        kappa_bot_pred = float(retained_bot[0] / retained_bot[-1])
        kappa_bot_actual, evals_bot = _actual_deflated_kappa(data_bot)
        log("  bottom exact-Z deflation: " + _fmt({
            "m_eff": data_bot.m_eff,
            "kappa_pred": kappa_bot_pred,
            "kappa_eff_actual": kappa_bot_actual,
            "cg_iters": it_bot,
            "deflated_relres": rel_bot,
            "true_relres": true_bot,
            "orthonormality_error": data_bot.diagnostics.get("orthonormality_error", np.nan),
        }, ["m_eff", "kappa_pred", "kappa_eff_actual", "cg_iters", "deflated_relres", "true_relres", "orthonormality_error"]))

        log("  top positive eig range after deflation: "
            + _fmt({
                "lambda_max_pos": float(np.max(np.real(evals_top[evals_top > 1e-12]))),
                "lambda_min_pos": float(np.min(np.real(evals_top[evals_top > 1e-12]))),
            }, ["lambda_max_pos", "lambda_min_pos"]))
        log("  bottom positive eig range after deflation: "
            + _fmt({
                "lambda_max_pos": float(np.max(np.real(evals_bot[evals_bot > 1e-12]))),
                "lambda_min_pos": float(np.min(np.real(evals_bot[evals_bot > 1e-12]))),
            }, ["lambda_max_pos", "lambda_min_pos"]))

        top_ok = (
            np.isfinite(true_top)
            and true_top < 1e-8
            and int(it_top) < int(plain_it)
        )
        bot_ok = (
            np.isfinite(true_bot)
            and true_bot < 1e-8
        )
        log(f"  CHECK top exact deflation improves CG: {'PASS' if top_ok else 'CHECK'}")
        log(f"  CHECK bottom exact deflation remains correct: {'PASS' if bot_ok else 'CHECK'}")
    except Exception:
        log("[toy dense SPD sanity] FAILED (continuing):")
        log(traceback.format_exc())

    log("")
    log("# done")
    log.close()


if __name__ == "__main__":
    main()
