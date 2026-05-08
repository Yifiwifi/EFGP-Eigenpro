from __future__ import annotations

from typing import Any, Callable, Optional
import time


# -----------------------------------------------------------------------------
# Basic dense/full-space utilities used by the existing refinement modes.
# -----------------------------------------------------------------------------

def embed_coordinate_basis(xp: Any, size: int, s_gpu: Any, V_gpu: Any, *, dtype: Any = None) -> Any:
    """Dense embedding U0 = I_S V. This is only for full-space refinement modes."""
    dtype = dtype or xp.complex128
    V = xp.asarray(V_gpu, dtype=dtype)
    U0 = xp.zeros((int(size), int(V.shape[1])), dtype=dtype)
    U0[xp.asarray(s_gpu, dtype=xp.int64), :] = V
    return xp.ascontiguousarray(U0)


def normalize_columns(xp: Any, U: Any, *, eps: float = 1e-30) -> Any:
    nrm = xp.linalg.norm(U, axis=0)
    return xp.ascontiguousarray(U / xp.maximum(nrm[None, :], eps))


def orthonormalize(xp: Any, U: Any, mode: str) -> tuple[Any, str]:
    """Return an orthonormalized or normalized basis and the effective mode."""
    mode = str(mode or "auto").lower()
    if mode in ("auto", "qr"):
        Q, _ = xp.linalg.qr(U, mode="reduced")
        return xp.ascontiguousarray(Q), "qr"
    if mode == "normalize":
        return normalize_columns(xp, U), "normalize"
    if mode == "none":
        return xp.ascontiguousarray(U), "none"
    raise ValueError("orthogonalize must be one of 'auto', 'qr', 'normalize', or 'none'.")


def apply_matvec_columns(
    xp: Any,
    matvec: Callable[[Any], Any],
    U: Any,
    *,
    block_cols: int = 8,
) -> Any:
    """Apply a block matvec to matrix columns; optional batching limits peak memory."""
    m, q = int(U.shape[0]), int(U.shape[1])
    AU = xp.empty((m, q), dtype=U.dtype)
    step = int(max(1, block_cols))
    for lo in range(0, q, step):
        hi = min(lo + step, q)
        X = U[:, lo:hi]
        try:
            Y = matvec(X)
            if getattr(Y, "shape", None) == X.shape:
                AU[:, lo:hi] = Y
                continue
        except Exception:
            pass
        for j in range(lo, hi):
            y = matvec(U[:, j])
            if getattr(y, "ndim", 1) == 2:
                y = y[:, 0]
            AU[:, j] = y
    return AU


def rayleigh_ritz(
    xp: Any,
    apply_A_block_gpu: Callable[[Any], Any],
    Q0: Any,
    *,
    q_out: int,
    block_cols: int = 8,
    orthogonalize: str = "qr",
) -> tuple[Any, Any, Optional[float], dict[str, Any]]:
    """Rayleigh--Ritz projection in span(Q0), returning top q_out pairs and q+1 threshold if available."""
    t0 = time.perf_counter()
    Q, orth_eff = orthonormalize(xp, Q0, orthogonalize)
    AQ = apply_matvec_columns(xp, apply_A_block_gpu, Q, block_cols=block_cols)
    H = Q.conj().T @ AQ
    H = 0.5 * (H + H.conj().T)
    vals, R = xp.linalg.eigh(H)
    order = xp.argsort(vals)[::-1]
    vals = xp.real(vals[order])
    R = R[:, order]
    Uall = xp.ascontiguousarray(Q @ R)
    AUall = AQ @ R
    q = int(q_out)
    eigvals = vals[:q]
    eigvecs = xp.ascontiguousarray(Uall[:, :q])
    res = AUall[:, :q] - eigvecs * eigvals.reshape(1, -1)
    residual_fro = float(xp.linalg.norm(res))
    denom = max(float(xp.linalg.norm(AUall[:, :q])), 1e-30)
    mu = float(vals[q]) if int(vals.size) > q else float(eigvals[-1])
    info = {
        "refine_rr_time_s": float(time.perf_counter() - t0),
        "refine_orthogonalize_effective": orth_eff,
        "residual_fro": residual_fro,
        "residual_fro_rel": residual_fro / denom,
        "surrogate_mu": mu,
        "mu": mu,
        "refine_basis_dim": int(Q.shape[1]),
    }
    return eigvals, eigvecs, mu, info


# -----------------------------------------------------------------------------
# New compact/low-M-dependence outputs.
# These keep the same return convention as refine_nystrom_basis:
#     (eigvals_like, basis_like, mu, info)
# but info["precond_kind"] tells the caller that basis_like is compact data,
# not a dense M x q full-space eigenvector block.
# -----------------------------------------------------------------------------

def _maybe_call_support_refine_fn(
    *,
    xp: Any,
    s_gpu: Any,
    V: Any,
    theta: Any,
    q_out: int,
    support_refine_fn: Optional[Callable[..., tuple[Any, Any, Any, dict[str, Any]]]],
    support_refine_kwargs: Optional[dict[str, Any]],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """
    Optional driver hook for adaptive support expansion.

    This module deliberately does not know how to build A[S,S] or A[C,S], because
    that is EFGP/Toeplitz-specific and lives naturally in v3_eigenspace.py.  If a
    caller wants true adaptive support expansion, pass a callback with signature
    approximately

        fn(xp=xp, S_gpu=s_gpu, V_gpu=V, theta_gpu=theta, q_out=q_out, **kwargs)
          -> (S_new_gpu, V_new_gpu, theta_new_gpu, info_dict)

    If no callback is supplied, the current support is used unchanged.  This keeps
    the old interface valid and makes the new modes safe as drop-in diagnostics.
    """
    if support_refine_fn is None:
        return s_gpu, V, theta, {"support_refine_effective": "none"}
    kwargs = dict(support_refine_kwargs or {})
    S2, V2, th2, extra = support_refine_fn(
        xp=xp,
        S_gpu=s_gpu,
        V_gpu=V,
        theta_gpu=theta,
        q_out=int(q_out),
        **kwargs,
    )
    return S2, V2, th2, dict(extra or {})


def compact_coordinate_result(
    *,
    xp: Any,
    s_gpu: Any,
    V_lift: Any,
    theta_lift: Any,
    q_out: int,
    precond_kind: str = "coordinate_nystrom",
    coord_gamma: float = 1.0,
    diag_inv_sqrt_gpu: Any = None,
    support_info: Optional[dict[str, Any]] = None,
    t0: Optional[float] = None,
) -> tuple[Any, Any, Optional[float], dict[str, Any]]:
    """
    Package compact coordinate preconditioner data.

    Returns V_q as the second object instead of an M x q dense basis.  The caller
    must dispatch on info['precond_kind'] and build/apply the compact preconditioner.
    """
    q = int(q_out)
    theta = xp.asarray(theta_lift, dtype=xp.float64).reshape(-1)
    V = xp.asarray(V_lift)
    if int(V.shape[1]) < q or int(theta.size) < q:
        raise ValueError(f"Need at least q_out={q} compact vectors/eigenvalues.")
    Vq = xp.ascontiguousarray(V[:, :q])
    theta_q = xp.ascontiguousarray(theta[:q])
    mu = float(theta[q]) if int(theta.size) > q else float(theta_q[-1])
    info: dict[str, Any] = {
        "refine_mode": precond_kind,
        "precond_kind": precond_kind,
        "compact_basis": True,
        "S_gpu": xp.ascontiguousarray(xp.asarray(s_gpu, dtype=xp.int64)),
        "V_gpu": Vq,
        "theta_gpu": theta_q,
        "coord_nystrom_gamma": float(coord_gamma),
        "surrogate_mu": mu,
        "mu": mu,
        "residual_fro": float("nan"),
        "residual_fro_rel": float("nan"),
        "refine_basis_dim": int(q),
        "refine_matvec_count_blocks": 0,
        "refine_build_time_s": float(time.perf_counter() - t0) if t0 is not None else 0.0,
    }
    if diag_inv_sqrt_gpu is not None:
        info["diag_inv_sqrt_gpu"] = xp.asarray(diag_inv_sqrt_gpu)
        info["uses_jacobi_scaling"] = True
    else:
        info["uses_jacobi_scaling"] = False
    if support_info:
        info.update({f"support_{k}": v for k, v in support_info.items()})
    return theta_q, Vq, mu, info


def _hybrid_topr_result(
    *,
    xp: Any,
    apply_A_block_gpu: Callable[[Any], Any],
    size: int,
    s_gpu: Any,
    V_lift: Any,
    theta_lift: Any,
    q_out: int,
    hybrid_top_r: int,
    block_cols: int,
    orthogonalize: str,
    eig_floor: float,
    coord_gamma: float,
    diag_inv_sqrt_gpu: Any,
    t0: float,
) -> tuple[Any, Any, Optional[float], dict[str, Any]]:
    """
    Level 2: refine only the first r directions in full M-space; keep the tail as compact data.

    The returned eigvecs/eigvals are the dense full-space part (r columns).  The compact tail is in
    info['hybrid_tail_*']; downstream preconditioner code can use both.  If downstream code does not
    support hybrid preconditioners yet, it can still use the returned top-r dense basis as a normal
    lower-rank full preconditioner.
    """
    q = int(q_out)
    r = int(max(0, min(int(hybrid_top_r), q)))
    if r < 1:
        return compact_coordinate_result(
            xp=xp,
            s_gpu=s_gpu,
            V_lift=V_lift,
            theta_lift=theta_lift,
            q_out=q,
            precond_kind="coordinate_nystrom",
            coord_gamma=coord_gamma,
            diag_inv_sqrt_gpu=diag_inv_sqrt_gpu,
            t0=t0,
        )

    theta = xp.asarray(theta_lift, dtype=xp.float64).reshape(-1)
    V = xp.asarray(V_lift)
    ncols = int(min(int(theta.size), int(V.shape[1])))
    if ncols < q:
        raise ValueError(f"Need at least q_out={q} surrogate columns for hybrid_topr; got {ncols}.")

    # Full refine the first r directions by one matvec lift + RR.
    U0r = embed_coordinate_basis(xp, int(size), s_gpu, V[:, :r], dtype=xp.complex128)
    th_safe = xp.maximum(theta[:r], xp.asarray(float(eig_floor), dtype=theta.dtype))
    AU0r = apply_matvec_columns(xp, apply_A_block_gpu, U0r, block_cols=block_cols)
    U1r = AU0r / th_safe.reshape(1, -1)
    U1r, orth_eff = orthonormalize(xp, U1r, orthogonalize)
    evals_r, U_r, mu_r, rr_info = rayleigh_ritz(
        xp,
        apply_A_block_gpu,
        U1r,
        q_out=r,
        block_cols=block_cols,
        orthogonalize="none" if orth_eff in ("qr", "normalize") else "qr",
    )

    # Compact tail remains only on S. It starts at r and ends at q.
    V_tail = xp.ascontiguousarray(V[:, r:q])
    theta_tail = xp.ascontiguousarray(theta[r:q])
    mu_tail = float(theta[q]) if int(theta.size) > q else float(theta[q - 1])

    info: dict[str, Any] = {
        "refine_mode": "hybrid_topr",
        "precond_kind": "hybrid_topr_coordinate",
        "compact_basis": False,
        "hybrid_top_r": int(r),
        "hybrid_total_q": int(q),
        "hybrid_dense_eigvals_gpu": evals_r,
        "hybrid_dense_eigvecs_gpu": U_r,
        "hybrid_tail_S_gpu": xp.ascontiguousarray(xp.asarray(s_gpu, dtype=xp.int64)),
        "hybrid_tail_V_gpu": V_tail,
        "hybrid_tail_theta_gpu": theta_tail,
        "hybrid_tail_mu": mu_tail,
        "coord_nystrom_gamma": float(coord_gamma),
        "surrogate_mu": mu_r,
        "mu": mu_r,
        "refine_orthogonalize_effective_initial": orth_eff,
        "refine_matvec_count_blocks": 2,
        "refine_build_time_s": float(time.perf_counter() - t0),
        **rr_info,
    }
    if diag_inv_sqrt_gpu is not None:
        info["diag_inv_sqrt_gpu"] = xp.asarray(diag_inv_sqrt_gpu)
        info["uses_jacobi_scaling"] = True
    else:
        info["uses_jacobi_scaling"] = False
    return evals_r, U_r, mu_r, info


# -----------------------------------------------------------------------------
# Main dispatcher. Existing modes are unchanged; new low-M modes are added.
# -----------------------------------------------------------------------------

def refine_nystrom_basis(
    *,
    xp: Any,
    apply_A_block_gpu: Callable[[Any], Any],
    size: int,
    s_gpu: Any,
    V_lift: Any,
    theta_lift: Any,
    q_out: int,
    mode: str,
    block_cols: int = 8,
    orthogonalize: str = "auto",
    polish_iters: int = 1,
    eig_floor: float = 1e-12,
    # New optional parameters.  They are all optional, so old callers are still valid.
    coord_gamma: float = 1.0,
    diag_inv_sqrt_gpu: Any = None,
    hybrid_top_r: Optional[int] = None,
    support_refine_fn: Optional[Callable[..., tuple[Any, Any, Any, dict[str, Any]]]] = None,
    support_refine_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[Any, Any, Optional[float], dict[str, Any]]:
    """
    Build/refine a Nyström surrogate basis/preconditioner representation.

    Existing full-space modes
    -------------------------
    inject
        Dense U0 = I_S V baseline. It returns an M x q dense basis.
    matvec_lift
        Nyström lift by one full operator application: U1 = A U0 Theta^{-1}, then RR.
    krylov_ritz
        One-step block Krylov RR in span{U0, A U0 Theta^{-1}}.
    subspace_polish
        Start from U1 = A U0 Theta^{-1}, run a few block power/subspace steps, then RR.

    New low-M-dependence modes
    --------------------------
    adaptive_support
        Level 0.  Use/optionally refine support S, then return compact coordinate data only.
        Heavy iterations are expected to happen inside support_refine_fn on s/candidate spaces.
    diag_adaptive_support
        Level 1.  Same as adaptive_support, but carries diag_inv_sqrt_gpu for Jacobi-scaled
        compact preconditioning.  The M-dependence is only elementwise scaling in the apply step.
    hybrid_topr
        Level 2.  Refine only the first r directions in full M-space and keep the remaining
        q-r directions as compact coordinate tail.  This limits full matvec cost to r << q.
    """
    t0 = time.perf_counter()
    mode = str(mode or "inject").lower()
    q = int(q_out)
    theta = xp.asarray(theta_lift, dtype=xp.float64).reshape(-1)
    V = xp.asarray(V_lift)
    ncols = int(min(int(V.shape[1]), int(theta.size)))
    if ncols < q:
        raise ValueError(f"Need at least q_out={q} surrogate vectors; got {ncols}.")
    V = V[:, :ncols]
    theta = theta[:ncols]

    # ------------------------------------------------------------------
    # Level 0: adaptive support compact coordinate preconditioner.
    # ------------------------------------------------------------------
    if mode in ("adaptive_support", "compact_adaptive", "support_adaptive"):
        S2, V2, th2, s_info = _maybe_call_support_refine_fn(
            xp=xp,
            s_gpu=s_gpu,
            V=V,
            theta=theta,
            q_out=q,
            support_refine_fn=support_refine_fn,
            support_refine_kwargs=support_refine_kwargs,
        )
        return compact_coordinate_result(
            xp=xp,
            s_gpu=S2,
            V_lift=V2,
            theta_lift=th2,
            q_out=q,
            precond_kind="coordinate_nystrom",
            coord_gamma=coord_gamma,
            support_info=s_info,
            t0=t0,
        )

    # ------------------------------------------------------------------
    # Level 1: Jacobi/diagonal-scaled adaptive support compact preconditioner.
    # ------------------------------------------------------------------
    if mode in ("diag_adaptive_support", "jacobi_adaptive_support", "diag_support_adaptive"):
        if diag_inv_sqrt_gpu is None:
            raise ValueError(
                "diag_adaptive_support requires diag_inv_sqrt_gpu. "
                "Pass the elementwise inverse square root of diag(A), or use adaptive_support."
            )
        S2, V2, th2, s_info = _maybe_call_support_refine_fn(
            xp=xp,
            s_gpu=s_gpu,
            V=V,
            theta=theta,
            q_out=q,
            support_refine_fn=support_refine_fn,
            support_refine_kwargs=support_refine_kwargs,
        )
        return compact_coordinate_result(
            xp=xp,
            s_gpu=S2,
            V_lift=V2,
            theta_lift=th2,
            q_out=q,
            precond_kind="diag_coordinate_nystrom",
            coord_gamma=coord_gamma,
            diag_inv_sqrt_gpu=diag_inv_sqrt_gpu,
            support_info=s_info,
            t0=t0,
        )

    # ------------------------------------------------------------------
    # Level 2: M-budgeted hybrid: only top-r directions get full-space matvec refinement.
    # ------------------------------------------------------------------
    if mode in ("hybrid_topr", "hybrid_top_r", "topr_hybrid"):
        r = int(hybrid_top_r if hybrid_top_r is not None else min(q, max(1, q // 4)))
        return _hybrid_topr_result(
            xp=xp,
            apply_A_block_gpu=apply_A_block_gpu,
            size=int(size),
            s_gpu=s_gpu,
            V_lift=V,
            theta_lift=theta,
            q_out=q,
            hybrid_top_r=r,
            block_cols=block_cols,
            orthogonalize=orthogonalize,
            eig_floor=eig_floor,
            coord_gamma=coord_gamma,
            diag_inv_sqrt_gpu=diag_inv_sqrt_gpu,
            t0=t0,
        )

    # ------------------------------------------------------------------
    # Existing dense/full-space modes below.  Behavior preserved.
    # ------------------------------------------------------------------
    U0 = embed_coordinate_basis(xp, int(size), s_gpu, V, dtype=xp.complex128)
    info: dict[str, Any] = {
        "refine_mode": mode,
        "refine_input_cols": int(ncols),
        "refine_build_time_s": None,
        "refine_matvec_count_blocks": 0,
    }

    if mode in ("inject", "coord_inject", "coordinate", "none"):
        # U0 columns are orthonormal if V columns are orthonormal. RR is optional outside this function;
        # here we return the coordinate-injected basis with surrogate theta values.
        eigvals = theta[:q]
        mu = float(theta[q]) if int(theta.size) > q else float(eigvals[-1])
        info.update({
            "refine_build_time_s": float(time.perf_counter() - t0),
            "surrogate_mu": mu,
            "mu": mu,
            "residual_fro": float("nan"),
            "residual_fro_rel": float("nan"),
            "refine_basis_dim": int(q),
        })
        return eigvals, xp.ascontiguousarray(U0[:, :q]), mu, info

    theta_safe = xp.maximum(theta, xp.asarray(float(eig_floor), dtype=theta.dtype))
    AU0 = apply_matvec_columns(xp, apply_A_block_gpu, U0, block_cols=block_cols)
    info["refine_matvec_count_blocks"] = 1
    U1 = AU0 / theta_safe.reshape(1, -1)

    if mode in ("matvec_lift", "au_lift", "nystrom_matvec_lift"):
        U1, orth_eff = orthonormalize(xp, U1, orthogonalize)
        eigvals, eigvecs, mu, rr_info = rayleigh_ritz(
            xp,
            apply_A_block_gpu,
            U1,
            q_out=q,
            block_cols=block_cols,
            orthogonalize="none" if orth_eff in ("qr", "normalize") else "qr",
        )
        info.update(rr_info)
        info["refine_orthogonalize_effective_initial"] = orth_eff
        info["refine_matvec_count_blocks"] = 2
        info["refine_build_time_s"] = float(time.perf_counter() - t0)
        return eigvals, eigvecs, mu, info

    if mode in ("krylov_ritz", "krylov2", "block_krylov"):
        Z = xp.concatenate([U0, U1], axis=1)
        eigvals, eigvecs, mu, rr_info = rayleigh_ritz(
            xp,
            apply_A_block_gpu,
            Z,
            q_out=q,
            block_cols=block_cols,
            orthogonalize="qr",
        )
        info.update(rr_info)
        info["refine_matvec_count_blocks"] = 2
        info["refine_build_time_s"] = float(time.perf_counter() - t0)
        return eigvals, eigvecs, mu, info

    if mode in ("subspace_polish", "power_polish", "polish"):
        Q, orth_eff = orthonormalize(xp, U1, "qr" if orthogonalize == "auto" else orthogonalize)
        n_iter = int(max(0, polish_iters))
        for _ in range(n_iter):
            Y = apply_matvec_columns(xp, apply_A_block_gpu, Q, block_cols=block_cols)
            Q, _ = xp.linalg.qr(Y, mode="reduced")
            Q = xp.ascontiguousarray(Q)
        eigvals, eigvecs, mu, rr_info = rayleigh_ritz(
            xp,
            apply_A_block_gpu,
            Q,
            q_out=q,
            block_cols=block_cols,
            orthogonalize="none",
        )
        info.update(rr_info)
        info["refine_orthogonalize_effective_initial"] = orth_eff
        info["refine_polish_iters"] = int(n_iter)
        info["refine_matvec_count_blocks"] = 2 + int(n_iter)
        info["refine_build_time_s"] = float(time.perf_counter() - t0)
        return eigvals, eigvecs, mu, info

    raise ValueError(
        "Unknown nystrom refine mode. Expected one of: inject, matvec_lift, "
        "krylov_ritz, subspace_polish, adaptive_support, diag_adaptive_support, "
        "hybrid_topr. Got %r" % mode
    )
