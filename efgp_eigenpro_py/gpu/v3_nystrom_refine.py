from __future__ import annotations

from typing import Any, Callable, Optional
import time


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
) -> tuple[Any, Any, Optional[float], dict[str, Any]]:
    """
    Build/refine full-space eigenvectors from a coordinate Nyström surrogate.

    Modes
    -----
    inject
        No refinement: U0 = I_S V. Cheap baseline, usually bad if S--S^c coupling is strong.
    matvec_lift
        Nyström lift by one full operator application: U1 = A U0 Theta^{-1}, then optional RR.
    krylov_ritz
        One-step block Krylov RR in span{U0, A U0 Theta^{-1}}.
    subspace_polish
        Start from U1 = A U0 Theta^{-1}, run a few block power/subspace steps, then RR.
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
        "krylov_ritz, subspace_polish. Got %r" % mode
    )
