from __future__ import annotations

"""
deflation_core.py
=================

Frank-Vuik style deflation for the EFGP/EigenPro high-fidelity operator

    A = D T D* + lambda I,   A in C^{n x n}.

The deflation subspace ``Z`` (n x m) may come from a cheap low-fidelity
operator ``A_tilde`` (see ``deflation_subspace.py``), but the deflation
projector must be built with the *true* high-fidelity ``A``:

    P_D = I - A Z (Z* A Z)^{-1} Z*.

This module owns the pure-algebra pieces that do not depend on how ``Z`` is
generated:

- :class:`DeflationData`             : Z, W = A Z, G = Z* A Z, Cholesky factor.
- :func:`build_deflation_data`       : Gram-eigh orthonormalize + rank truncation,
                                       then W = A_hi Z, G, Cholesky (jitter only as
                                       a last resort).
- :func:`solve_G`                    : apply G^{-1} via the Cholesky factor.
- :func:`make_deflated_matvec`       : H v = A v - W G^{-1} Z* (A v) with op_ctx buffers.
- :func:`project_left`               : P_D v = v - W G^{-1} Z* v (for b_def = P_D b).
- :func:`recover_solution`           : x = x_hat + Z G^{-1} Z* (b - A x_hat), reusing
                                       A x_hat to obtain the true residual with no extra
                                       high-fidelity matvec.
- :func:`run_deflated_cg`            : project_left -> CG (spd_guard=False) -> recover.

All array ops go through ``backend.xp`` so the module works with CuPy (and NumPy
in tests).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import math

from .deflation_basis import DeflationBasis, as_deflation_basis
from .iterative_solvers import cg_solve_gpu, pcg_solve_gpu


# ---------------------------------------------------------------------------
# Triangular-solve resolution (Cholesky based G^{-1}).
# ---------------------------------------------------------------------------

_TRISOLVE_CACHE: dict[int, Any] = {}


def _resolve_triangular_solver(xp: Any) -> Optional[Callable[..., Any]]:
    """Return a ``solve_triangular(a, b, lower, trans)`` for the active backend, or None."""
    key = id(xp)
    if key in _TRISOLVE_CACHE:
        return _TRISOLVE_CACHE[key]
    fn: Optional[Callable[..., Any]] = None
    try:
        import cupy  # type: ignore

        if xp is cupy:
            from cupyx.scipy.linalg import solve_triangular as st  # type: ignore

            fn = st
    except Exception:
        fn = None
    if fn is None:
        try:
            from scipy.linalg import solve_triangular as st  # type: ignore

            fn = st
        except Exception:
            fn = None
    _TRISOLVE_CACHE[key] = fn
    return fn


def _cholesky_with_jitter(
    xp: Any, G: Any, *, jitter_ratio: float, m_eff: int
) -> tuple[Any, float]:
    """Cholesky of Hermitian ``G``; add a tiny diagonal jitter only if it fails."""
    G = 0.5 * (G + G.conj().T)
    try:
        L = xp.linalg.cholesky(G)
        return L, 0.0
    except Exception:
        pass
    tr = float(xp.real(xp.trace(G)))
    base = abs(tr) / max(int(m_eff), 1)
    jitter = float(jitter_ratio) * (base if base > 0.0 else 1.0)
    eye = xp.eye(int(G.shape[0]), dtype=G.dtype)
    for scale in (1.0, 10.0, 100.0, 1e3, 1e4, 1e6):
        jit = jitter * scale
        try:
            L = xp.linalg.cholesky(G + jit * eye)
            return L, float(jit)
        except Exception:
            continue
    # Last resort: symmetric eigenvalue floor.
    w, V = xp.linalg.eigh(G)
    w = xp.real(w)
    floor = jitter if jitter > 0.0 else 1e-30
    w = xp.maximum(w, floor)
    Gpsd = (V * w[None, :]) @ V.conj().T
    L = xp.linalg.cholesky(0.5 * (Gpsd + Gpsd.conj().T))
    return L, floor


# ---------------------------------------------------------------------------
# Deflation data container.
# ---------------------------------------------------------------------------

@dataclass
class DeflationData:
    """High-fidelity deflation operands and diagnostics."""

    basis: DeflationBasis        # structured orthonormalized Z basis
    W: Any                       # A_hi @ Z correction handle; dense/fp32 array or None for implicit
    G: Any                       # Z* A_hi Z, (m_eff, m_eff), Hermitian
    chol: Any                    # Cholesky factor L of G (lower, G = L L^H)
    m: int                       # requested rank
    m_eff: int                   # effective rank after truncation
    z_dtype: str = "complex128"
    w_dtype: str = "complex128"
    jitter: float = 0.0
    cond_G: float = float("nan")
    rank_dropped: int = 0
    diagnostics: dict = field(default_factory=dict)
    Z: Any = None                # optional dense fallback/debug copy
    ZH: Any = None               # optional dense fallback/debug copy
    w_mode: str = "dense"
    matvec_form: str = "structured_ZH"
    W_fro_norm: float = float("nan")


def solve_G(backend: Any, data: DeflationData, y: Any) -> Any:
    """Solve ``G c = y`` using the stored Cholesky factor (1D or 2D ``y``)."""
    xp = backend.xp
    L = data.chol
    st = _resolve_triangular_solver(xp)
    if st is not None:
        try:
            z = st(L, y, lower=True, trans=0)
            c = st(L, z, lower=True, trans=2)
            return c
        except Exception:
            pass
    # Fallback: solve against the full Hermitian G.
    return xp.linalg.solve(data.G, y)


# ---------------------------------------------------------------------------
# Build: orthonormalize Z (Gram-eigh + rank truncation), W = A Z, G, Cholesky.
# ---------------------------------------------------------------------------

def _gram_eigh_orthonormalize(
    xp: Any, Z: Any, *, rank_tol: float
) -> tuple[Any, int, int]:
    """
    CholeskyQR2-style orthonormalization via the Gram matrix eigendecomposition,
    with rank truncation.  Returns (Z_orth, m_eff, rank_dropped).

    C = Z* Z;  C = U S U*;  keep S_i > rank_tol * S_max;  Z <- Z U_keep S_keep^{-1/2}.
    """
    m = int(Z.shape[1])
    C = Z.conj().T @ Z
    C = 0.5 * (C + C.conj().T)
    evals, evecs = xp.linalg.eigh(C)
    evals = xp.real(evals)
    smax = float(xp.max(evals)) if int(evals.size) else 0.0
    if smax <= 0.0:
        raise RuntimeError("deflation subspace Z has non-positive Gram spectrum.")
    keep = evals > (float(rank_tol) * smax)
    m_eff = int(keep.sum())
    if m_eff < 1:
        raise RuntimeError("deflation subspace Z collapsed to rank 0 after truncation.")
    evecs_k = evecs[:, keep]
    evals_k = evals[keep]
    Z_orth = Z @ (evecs_k / xp.sqrt(evals_k)[None, :])
    # One refinement pass (CholeskyQR2) to tighten orthonormality.
    C2 = Z_orth.conj().T @ Z_orth
    C2 = 0.5 * (C2 + C2.conj().T)
    ev2, U2 = xp.linalg.eigh(C2)
    ev2 = xp.maximum(xp.real(ev2), float(rank_tol) * float(xp.max(xp.real(ev2))))
    Z_orth = Z_orth @ (U2 / xp.sqrt(ev2)[None, :])
    return xp.ascontiguousarray(Z_orth), m_eff, int(m - m_eff)


def _apply_block_in_chunks(
    xp: Any,
    apply_A_hi_block: Callable[[Any], Any],
    basis: DeflationBasis,
    *,
    block_cols: int,
) -> Any:
    """W = A Z column-block by column-block to cap peak memory."""
    n, m = int(basis.n), int(basis.m)
    W = xp.empty((n, m), dtype=xp.complex128)
    step = int(max(1, block_cols))
    for lo in range(0, m, step):
        hi = min(lo + step, m)
        Zb = basis.columns(lo, hi)
        Yb = apply_A_hi_block(Zb)
        if getattr(Yb, "shape", None) != (n, hi - lo):
            raise ValueError("apply_A_hi_block returned shape mismatch.")
        W[:, lo:hi] = xp.asarray(Yb, dtype=xp.complex128)
    return W


def build_deflation_data(
    backend: Any,
    apply_A_hi_block: Callable[[Any], Any],
    Z: Any,
    *,
    rank_tol: float = 1e-12,
    jitter_ratio: float = 1e-12,
    block_cols: int = 8,
    z_storage_dtype: str = "same_as_A",
    compute_diagnostics: bool = True,
    w_mode: str = "dense",
    matvec_form: str = "structured_ZH",
) -> DeflationData:
    """
    Build :class:`DeflationData` from a raw subspace ``Z`` or a structured
    :class:`DeflationBasis`.

    ``z_storage_dtype`` controls the stored dtype of ``Z`` only: ``"same_as_A"``
    (default, complex128) or ``"complex64"`` for memory savings.  ``W`` is always
    complex128.  Orthonormalization and ``G`` are computed in complex128 for
    numerical stability regardless of the storage choice.
    """
    xp = backend.xp
    basis_raw = as_deflation_basis(xp, Z)
    m_req = int(basis_raw.m)
    basis, basis_diag = basis_raw.orthonormalized(rank_tol=rank_tol)
    m_eff = int(basis.m)
    rank_dropped = int(m_req - m_eff)

    w_mode = str(w_mode).lower()
    if w_mode == "auto":
        w_mode = "dense"
    if w_mode not in ("dense", "dense_fp32", "implicit"):
        raise ValueError("w_mode must be 'dense', 'dense_fp32', 'implicit', or 'auto'.")
    matvec_form = str(matvec_form).lower()
    if matvec_form not in ("structured_zh", "symmetric_w"):
        raise ValueError("matvec_form must be 'structured_ZH' or 'symmetric_W'.")
    if w_mode == "implicit" and matvec_form == "symmetric_w":
        raise ValueError("matvec_form='symmetric_W' requires a stored dense W.")

    W_hi = _apply_block_in_chunks(xp, apply_A_hi_block, basis, block_cols=block_cols)
    W_fro_norm = float(xp.linalg.norm(W_hi))

    G_raw = basis.apply_H(W_hi)
    herm_err = float(
        xp.linalg.norm(G_raw - G_raw.conj().T) / max(float(xp.linalg.norm(G_raw)), 1e-300)
    )
    G = 0.5 * (G_raw + G_raw.conj().T)

    chol, jitter = _cholesky_with_jitter(
        xp, G, jitter_ratio=jitter_ratio, m_eff=m_eff
    )

    cond_G = float("nan")
    try:
        evg = xp.real(xp.linalg.eigvalsh(G))
        gmin = float(xp.min(evg))
        gmax = float(xp.max(evg))
        if gmin > 0.0:
            cond_G = gmax / gmin
    except Exception:
        pass

    # Optional dense debug copy only for DenseBasis; structured bases do not
    # materialize fine-space Z unless explicitly requested elsewhere.
    z_dtype = "structured"
    Z_store = None
    ZH_store = None
    if getattr(basis, "kind", "") == "dense":
        Z_dense = basis.to_dense()
        z_dtype = str(Z_dense.dtype)
        if str(z_storage_dtype).lower() in ("complex64", "single"):
            Z_dense = Z_dense.astype(xp.complex64, copy=False)
            z_dtype = "complex64"
        Z_store = xp.ascontiguousarray(Z_dense)
        ZH_store = xp.ascontiguousarray(Z_store.conj().T)

    if w_mode == "dense":
        W_store = xp.ascontiguousarray(W_hi)
        w_dtype = "complex128"
    elif w_mode == "dense_fp32":
        W_store = xp.ascontiguousarray(W_hi.astype(xp.complex64, copy=False))
        w_dtype = "complex64"
    else:
        W_store = None
        w_dtype = "implicit"

    data = DeflationData(
        basis=basis,
        W=W_store,
        G=xp.ascontiguousarray(G),
        chol=chol,
        m=m_req,
        m_eff=int(m_eff),
        z_dtype=z_dtype,
        w_dtype=w_dtype,
        jitter=float(jitter),
        cond_G=cond_G,
        rank_dropped=int(rank_dropped),
        diagnostics={},
        Z=Z_store,
        ZH=ZH_store,
        w_mode=w_mode,
        matvec_form=matvec_form,
        W_fro_norm=W_fro_norm,
    )

    diag: dict[str, Any] = {
        "m_requested": m_req,
        "m_eff": int(m_eff),
        "rank_dropped": int(rank_dropped),
        "cond_G": cond_G,
        "jitter": float(jitter),
        "hermitian_error_G": herm_err,
        "z_dtype": z_dtype,
        "w_dtype": w_dtype,
        "w_mode": w_mode,
        "matvec_form": matvec_form,
        "basis_kind": getattr(basis, "kind", "unknown"),
        **basis_diag,
    }
    if compute_diagnostics:
        diag.update(_static_deflation_diagnostics(backend, data))
    data.diagnostics = diag
    return data


# ---------------------------------------------------------------------------
# Diagnostics that depend only on (Z, W, G).
# ---------------------------------------------------------------------------

def _static_deflation_diagnostics(backend: Any, data: DeflationData) -> dict[str, Any]:
    """Fast diagnostics using small matrices and Frobenius norms."""
    xp = backend.xp
    Wn = float(data.W_fro_norm)
    Wn2 = max(Wn * Wn, 1e-300)
    Gram = data.basis.gram()
    eye = xp.eye(int(data.m_eff), dtype=xp.complex128)
    orth_err = float(xp.linalg.norm(Gram - eye) / math.sqrt(max(int(data.m_eff), 1)))
    # With Euclidean-orthonormal Z, ||(I-ZZ*)W||_F^2 = ||W||_F^2 - ||Z*W||_F^2.
    Gf2 = float(xp.linalg.norm(data.G) ** 2)
    leak = float(math.sqrt(max(1.0 - (Gf2 / Wn2), 0.0)))
    # Deflation exactness is measured in small space to avoid W @ (G^{-1}G).
    GiG = solve_G(backend, data, data.G)
    exact = float(xp.linalg.norm(GiG - eye) / math.sqrt(max(int(data.m_eff), 1)))
    return {
        "invariance_leakage": leak,
        "deflation_exactness": exact,
        "orthonormality_error": orth_err,
        "diagnostics_mode": "fast",
        "W_fro_norm": Wn,
    }


def deflation_memory_estimate(data: DeflationData) -> dict[str, Any]:
    """Estimate Z/W memory footprints in GB for logging."""
    n = int(data.basis.n)
    m = int(data.m_eff)
    if data.w_mode == "implicit":
        w_bytes = 0
    elif data.w_dtype == "complex64":
        w_bytes = 8
    else:
        w_bytes = 16
    bmem = data.basis.memory_estimate()
    gb_basis = float(bmem.get("estimated_basis_GB", 0.0))
    gb_w = n * m * w_bytes / (1024.0**3)
    return {
        "n": n,
        "m_eff": m,
        "z_dtype": data.z_dtype,
        "w_dtype": data.w_dtype,
        "basis_kind": bmem.get("basis_kind", getattr(data.basis, "kind", "unknown")),
        "estimated_basis_GB": gb_basis,
        "estimated_Z_GB": gb_basis,
        "estimated_W_GB": gb_w,
        "estimated_total_deflation_GB": gb_basis + gb_w,
        **{k: v for k, v in bmem.items() if k not in ("basis_kind", "estimated_basis_GB")},
    }


# ---------------------------------------------------------------------------
# Projector applications.
# ---------------------------------------------------------------------------

def make_jacobi_precond(
    backend: Any, diag_inv: Any
) -> Callable[[Any, Any], None]:
    """
    Build a diagonal (Jacobi) SPD preconditioner ``M^{-1} v = diag_inv * v`` as a
    ``precond(v, out)`` closure for deflated-PCG.  ``diag_inv`` is the elementwise
    inverse of ``diag(A)`` (shape (n,)).
    """
    xp = backend.xp
    d = xp.asarray(diag_inv).reshape(-1)

    def precond(v: Any, out: Any) -> None:
        vv = xp.asarray(v).reshape(-1)
        dd = d.astype(out.dtype, copy=False) if d.dtype != out.dtype else d
        xp.multiply(dd, vv, out=out)

    return precond


def project_left(
    backend: Any,
    data: DeflationData,
    v: Any,
    out: Optional[Any] = None,
    apply_A_hi: Optional[Callable[[Any, Any], None]] = None,
    op_ctx: Any = None,
) -> Any:
    """P_D v = v - W G^{-1} (Z* v)."""
    xp = backend.xp
    v = xp.asarray(v).reshape(-1)
    coeff = solve_G(backend, data, data.basis.apply_H(v))
    corr = _apply_W_correction(backend, data, coeff, apply_A_hi, op_ctx)
    if out is None:
        return v - corr
    xp.subtract(v, corr, out=out)
    return out


def _apply_W_correction(
    backend: Any,
    data: DeflationData,
    coeff: Any,
    apply_A_hi: Optional[Callable[[Any, Any], None]],
    op_ctx: Any,
) -> Any:
    xp = backend.xp
    if data.W is not None:
        return xp.asarray(data.W, dtype=xp.complex128) @ coeff
    if apply_A_hi is None:
        raise RuntimeError("implicit W mode requires apply_A_hi.")
    zc = data.basis.apply(coeff)
    out = xp.empty((int(data.basis.n),), dtype=xp.complex128)
    apply_A_hi(zc, out)
    return out


def _ensure_buf(op_ctx: Any, xp: Any, name: str, size: int, dtype: Any) -> Any:
    buf = getattr(op_ctx, name, None)
    if buf is None or int(getattr(buf, "size", -1)) != int(size) or buf.dtype != dtype:
        buf = xp.empty((int(size),), dtype=dtype)
        setattr(op_ctx, name, buf)
    return buf


def make_deflated_matvec(
    backend: Any,
    data: DeflationData,
    apply_A_hi: Callable[[Any, Any], None],
    op_ctx: Any,
) -> Callable[[Any, Any], None]:
    """
    Return a ``matvec(v, out)`` computing H v = A v - W G^{-1} Z* (A v),
    reusing op_ctx workspaces (defl_av, defl_wc) to limit allocation.
    """
    xp = backend.xp
    n = int(data.basis.n)

    def matvec(v: Any, out: Any) -> None:
        av = _ensure_buf(op_ctx, xp, "defl_av", n, xp.complex128)
        apply_A_hi(v, av)
        if data.matvec_form == "symmetric_w":
            if data.W is None:
                raise RuntimeError("symmetric_W requires stored W.")
            coeff = solve_G(backend, data, xp.asarray(data.W, dtype=xp.complex128).conj().T @ xp.asarray(v).reshape(-1))
        else:
            coeff = solve_G(backend, data, data.basis.apply_H(av))
        wc = _ensure_buf(op_ctx, xp, "defl_wc", n, xp.complex128)
        corr = _apply_W_correction(backend, data, coeff, apply_A_hi, op_ctx)
        xp.copyto(wc, corr)
        xp.subtract(av, wc, out=out)

    return matvec


def recover_solution(
    backend: Any,
    data: DeflationData,
    apply_A_hi: Callable[[Any, Any], None],
    x_hat: Any,
    b: Any,
    op_ctx: Any,
) -> tuple[Any, dict[str, Any]]:
    """
    x = x_hat + Z G^{-1} Z* (b - A x_hat).

    Reuse ``A x_hat`` to compute the true residual without an extra matvec:
        A x = A x_hat + W c,   c = G^{-1} Z* (b - A x_hat).
    Returns (x, recover_diag).
    """
    xp = backend.xp
    n = int(data.basis.n)
    b = xp.asarray(b).reshape(-1)
    x_hat = xp.asarray(x_hat).reshape(-1)

    Ax_hat = _ensure_buf(op_ctx, xp, "defl_axhat", n, xp.complex128)
    apply_A_hi(x_hat, Ax_hat)
    resid0 = b - Ax_hat
    c = solve_G(backend, data, data.basis.apply_H(resid0))
    Zc = data.basis.apply(c)
    x = x_hat + Zc
    if data.w_mode == "dense":
        Ax = Ax_hat + xp.asarray(data.W, dtype=xp.complex128) @ c
    else:
        Azc = xp.empty((n,), dtype=xp.complex128)
        apply_A_hi(Zc, Azc)
        Ax = Ax_hat + Azc
    true_res_vec = b - Ax
    norm_b = max(float(xp.linalg.norm(b)), 1e-300)
    true_relres = float(xp.linalg.norm(true_res_vec) / norm_b)
    recover_diag = {
        "true_relres_from_recovery": true_relres,
        "recover_correction_norm": float(xp.linalg.norm(Zc)),
    }
    return x, recover_diag


# ---------------------------------------------------------------------------
# End-to-end deflated CG.
# ---------------------------------------------------------------------------

def run_deflated_cg(
    backend: Any,
    data: DeflationData,
    apply_A_hi: Callable[[Any, Any], None],
    b: Any,
    op_ctx: Any,
    *,
    tol: float,
    maxiter: int,
    profile_components: bool = True,
    work_prefix: str = "dcg",
) -> tuple[Any, int, float, dict[str, Any]]:
    """
    Solve A x = b by deflated CG:
        b_def = P_D b;  H x_hat = b_def  (H = P_D A, SPSD, consistent);  recover x.

    Returns (x, cg_iters, deflated_relres, diagnostics).
    """
    xp = backend.xp
    b = xp.asarray(b).reshape(-1).astype(xp.complex128, copy=False)
    norm_b = max(float(xp.linalg.norm(b)), 1e-300)

    b_def = project_left(backend, data, b, apply_A_hi=apply_A_hi, op_ctx=op_ctx)
    b_def_ratio = float(xp.linalg.norm(b_def) / norm_b)

    matvec = make_deflated_matvec(backend, data, apply_A_hi, op_ctx)
    x_hat, it, def_relres, cg_stats = cg_solve_gpu(
        backend,
        matvec,
        b_def,
        op_ctx,
        tol,
        maxiter,
        return_stats=True,
        work_prefix=work_prefix,
        profile_components=profile_components,
        spd_guard=False,
    )

    x, recover_diag = recover_solution(backend, data, apply_A_hi, x_hat, b, op_ctx)

    diag: dict[str, Any] = {
        "deflated_relres": float(def_relres),
        "b_def_ratio": b_def_ratio,
        "cg_iters": int(it),
        **{k: v for k, v in cg_stats.items()},
        **recover_diag,
    }
    return x, int(it), float(def_relres), diag


def run_deflated_pcg(
    backend: Any,
    data: DeflationData,
    apply_A_hi: Callable[[Any, Any], None],
    precond: Callable[[Any, Any], None],
    b: Any,
    op_ctx: Any,
    *,
    tol: float,
    maxiter: int,
    profile_components: bool = True,
    work_prefix: str = "dpcg",
) -> tuple[Any, int, float, dict[str, Any]]:
    """
    Phase 2: deflated *preconditioned* CG (Frank-Vuik DPCG).

    Solve A x = b with deflation projector ``P_D`` and an SPD preconditioner whose
    application is ``precond(v, out)`` (i.e. ``M^{-1} v``).  DPCG runs ordinary PCG
    on the deflated operator ``H = P_D A`` with the same preconditioner; this is
    valid because ``H`` is symmetric and PCG works in the ``M``-inner-product
    (``(r, M^{-1} r)``), so no extra symmetrization is needed.  The solution is then
    recovered as in :func:`run_deflated_cg`.

    Returns (x, cg_iters, deflated_relres, diagnostics).
    """
    xp = backend.xp
    b = xp.asarray(b).reshape(-1).astype(xp.complex128, copy=False)
    norm_b = max(float(xp.linalg.norm(b)), 1e-300)

    b_def = project_left(backend, data, b, apply_A_hi=apply_A_hi, op_ctx=op_ctx)
    b_def_ratio = float(xp.linalg.norm(b_def) / norm_b)

    matvec = make_deflated_matvec(backend, data, apply_A_hi, op_ctx)
    x_hat, it, def_relres, pcg_stats = pcg_solve_gpu(
        backend,
        matvec,
        precond,
        b_def,
        op_ctx,
        tol,
        maxiter,
        return_stats=True,
        work_prefix=work_prefix,
        profile_components=profile_components,
        spd_guard=False,
    )

    x, recover_diag = recover_solution(backend, data, apply_A_hi, x_hat, b, op_ctx)

    diag: dict[str, Any] = {
        "deflated_relres": float(def_relres),
        "b_def_ratio": b_def_ratio,
        "cg_iters": int(it),
        **{k: v for k, v in pcg_stats.items()},
        **recover_diag,
    }
    return x, int(it), float(def_relres), diag
