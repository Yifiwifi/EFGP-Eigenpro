from __future__ import annotations

"""
v3_eigenspace_extra_algorithms.py
==================================

Experimental eigenspace/preconditioner builders for EFGP/EigenPro-style
weight-space systems.

The intended matrix is a Hermitian positive definite operator

    A = B^* B + lambda I,   A in C^{M x M},   M << N but M may be ~1e5.

The caller supplies two black-box operations:

1. apply_A_block_gpu(V): apply the full M-space operator A to a dense block
   V with shape (M, r).  In your EFGP implementation this costs roughly
   O(r M log M) using FFT/Toeplitz structure.

2. build_submatrix_gpu(S): build a principal coordinate submatrix A[S,S]
   for a 1D integer index array S.  In your EFGP implementation this can be
   done by Toeplitz lookup and diagonal weights at roughly O(|S|^2).

The design philosophy is:

    Heavy randomized search should happen in small s-dimensional coordinate
    sketches.  Full M-space matvecs are spent only when they are expected to
    save more PCG iterations than they cost.

This module is intentionally callback-based.  It does not know the details of
EFGP Toeplitz indexing, NUFFT precompute, or your preconditioner builder.  It
returns either full M x q Ritz vectors or compact preconditioner data stored in
``info``.  This keeps it compatible with the existing v3_eigenspace.py / 
v3_nystrom_refine.py style.

Implemented ideas
-----------------
A. compact_coordinate_nystrom
   Single coordinate Nyström sketch: W=A[S,S], exact spectral flattening on W.

B. ensemble_coordinate_nystrom
   Multiple small coordinate sketches, averaged as a randomized block-Jacobi
   spectral smoother.  No full A-matvec is used.

C. random_support_lift
   Many cheap s-sketches select candidate directions; only the best r directions
   get one full A-matvec lift and an optional Rayleigh--Ritz projection.

D. rand_range_onepass
   GPU-friendly randomized range finder / randomized SVD.  Uses dense block
   matvecs and optional power steps with a hard pass budget.

E. chebyshev_filtered_subspace
   Chebyshev polynomial filtered subspace iteration when a target spectral
   interval is available.  It is a more aggressive version of power iteration.

F. sample_side_nystrom_range
   A generic hook for sample-side Nyström / stochastic empirical operator
   approximation.  The caller provides a builder that constructs an approximate
   matvec from sampled data indices.  This reduces N-dependence, not necessarily
   M-dependence.

All functions accept ``xp`` so they work with cupy or numpy.  They avoid direct
imports from the rest of the project, except for the optional expectation that
``apply_A_block_gpu`` and ``build_submatrix_gpu`` come from your EFGP code.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence
import math
import time

import numpy as np

Array = Any
Matvec = Callable[[Array], Array]
SubmatrixBuilder = Callable[[np.ndarray], Array]
SampleMatvecBuilder = Callable[[np.ndarray], Matvec]


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------

@dataclass
class ExtraEigResult:
    """Common return container.

    values:
        Approximate eigenvalues, usually length q or q+1 depending on the mode.
    vectors:
        Either a dense M x q full-space basis, a compact basis, or None.
    mu:
        EigenPro threshold, typically lambda_{q+1} or a sketch-based proxy.
    info:
        Diagnostics and compact data needed by downstream preconditioner builders.
    """

    values: Array
    vectors: Optional[Array]
    mu: Optional[float]
    info: dict[str, Any]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _to_np(x: Any) -> np.ndarray:
    if hasattr(x, "get"):
        return np.asarray(x.get())
    return np.asarray(x)


def _choice_without_replacement(
    *,
    M: int,
    s: int,
    seed: int,
    weights: Optional[np.ndarray] = None,
    lowfreq_indices: Optional[np.ndarray] = None,
    lowfreq_ratio: float = 0.0,
) -> np.ndarray:
    """Choose s coordinate indices.

    If lowfreq_indices is provided, a fraction ``lowfreq_ratio`` is taken from
    that deterministic pool.  The remainder is sampled either uniformly or using
    probability proportional to ``weights``.
    """
    M = int(M)
    s = int(max(1, min(int(s), M)))
    rng = _rng(seed)
    chosen: list[int] = []

    if lowfreq_indices is not None and lowfreq_ratio > 0.0:
        lf = np.asarray(lowfreq_indices, dtype=np.int64).reshape(-1)
        lf = lf[(lf >= 0) & (lf < M)]
        lf = np.unique(lf)
        n_lf = min(int(round(float(lowfreq_ratio) * s)), int(lf.size), s)
        if n_lf > 0:
            # Deterministic high-priority low-frequency prefix if already sorted;
            # otherwise random subset from the low-frequency candidate pool.
            if int(lf.size) <= n_lf:
                det = lf
            else:
                det = rng.choice(lf, size=n_lf, replace=False)
            chosen.extend([int(x) for x in det])

    remaining = s - len(set(chosen))
    if remaining <= 0:
        return np.asarray(np.unique(chosen)[:s], dtype=np.int64)

    mask = np.ones(M, dtype=bool)
    if chosen:
        mask[np.asarray(chosen, dtype=np.int64)] = False
    pool = np.flatnonzero(mask)
    if int(pool.size) < remaining:
        remaining = int(pool.size)

    if weights is None:
        extra = rng.choice(pool, size=remaining, replace=False)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if int(w.size) != M:
            raise ValueError(f"weights.size={w.size} does not match M={M}.")
        p = np.maximum(w[pool], 0.0)
        psum = float(np.sum(p))
        if (not math.isfinite(psum)) or psum <= 0.0:
            extra = rng.choice(pool, size=remaining, replace=False)
        else:
            extra = rng.choice(pool, size=remaining, replace=False, p=p / psum)
    out = np.unique(np.concatenate([np.asarray(chosen, dtype=np.int64), extra]))
    if int(out.size) < s:
        missing = s - int(out.size)
        pool2 = np.setdiff1d(np.arange(M, dtype=np.int64), out, assume_unique=False)
        out = np.concatenate([out, rng.choice(pool2, size=missing, replace=False)])
    return np.asarray(out[:s], dtype=np.int64)


def _top_eigh(xp: Any, W: Array, n_eigs: int) -> tuple[Array, Array]:
    """Dense Hermitian top eigensolve for small coordinate sketches.

    This is intentionally simple and reliable.  If s is large, the caller can
    pass a smaller s, or replace this module's function with a small eigsh.
    """
    W = xp.asarray(W, dtype=xp.complex128)
    W = 0.5 * (W + W.conj().T)
    vals, vecs = xp.linalg.eigh(W)
    order = xp.argsort(xp.real(vals))[::-1]
    n = int(min(int(n_eigs), int(vals.size)))
    idx = order[:n]
    return xp.real(vals[idx]), xp.ascontiguousarray(vecs[:, idx])


def apply_matvec_columns(
    xp: Any,
    matvec: Matvec,
    U: Array,
    *,
    block_cols: int = 16,
) -> Array:
    """Apply matvec to a tall dense block with optional column batching."""
    U = xp.asarray(U, dtype=xp.complex128)
    if U.ndim == 1:
        y = matvec(U.reshape(-1, 1))
        if getattr(y, "ndim", 1) == 2:
            return xp.asarray(y[:, 0], dtype=xp.complex128)
        return xp.asarray(y, dtype=xp.complex128)
    M, r = int(U.shape[0]), int(U.shape[1])
    out = xp.empty((M, r), dtype=xp.complex128)
    step = int(max(1, block_cols))
    for lo in range(0, r, step):
        hi = min(lo + step, r)
        X = U[:, lo:hi]
        Y = matvec(X)
        if getattr(Y, "shape", None) != X.shape:
            # Fall back to single-column calls for matvecs that do not accept blocks.
            for j in range(lo, hi):
                yj = matvec(U[:, j].reshape(-1, 1))
                if getattr(yj, "ndim", 1) == 2:
                    yj = yj[:, 0]
                out[:, j] = xp.asarray(yj, dtype=xp.complex128)
        else:
            out[:, lo:hi] = xp.asarray(Y, dtype=xp.complex128)
    return out


def orthonormalize(xp: Any, U: Array, *, mode: str = "qr", eps: float = 1e-30) -> tuple[Array, str]:
    """QR or column normalization."""
    mode = str(mode or "qr").lower()
    U = xp.asarray(U, dtype=xp.complex128)
    if mode in ("qr", "auto"):
        Q, _ = xp.linalg.qr(U, mode="reduced")
        return xp.ascontiguousarray(Q), "qr"
    if mode == "normalize":
        nrm = xp.maximum(xp.linalg.norm(U, axis=0, keepdims=True), eps)
        return xp.ascontiguousarray(U / nrm), "normalize"
    if mode == "none":
        return xp.ascontiguousarray(U), "none"
    raise ValueError("mode must be 'qr', 'auto', 'normalize', or 'none'.")


def rayleigh_ritz(
    *,
    xp: Any,
    apply_A_block_gpu: Matvec,
    Q0: Array,
    q_out: int,
    block_cols: int = 16,
    orthogonalize: str = "qr",
) -> ExtraEigResult:
    """Rayleigh--Ritz in span(Q0).

    Cost is one full block A-matvec plus a small dense eigensolve.  This should
    be used only after the candidate space is small enough.
    """
    t0 = time.perf_counter()
    Q, orth_eff = orthonormalize(xp, Q0, mode=orthogonalize)
    AQ = apply_matvec_columns(xp, apply_A_block_gpu, Q, block_cols=block_cols)
    H = Q.conj().T @ AQ
    H = 0.5 * (H + H.conj().T)
    vals, R = xp.linalg.eigh(H)
    order = xp.argsort(xp.real(vals))[::-1]
    vals = xp.real(vals[order])
    R = R[:, order]
    Uall = xp.ascontiguousarray(Q @ R)
    AUall = AQ @ R
    q = int(q_out)
    eigvals = vals[:q]
    eigvecs = xp.ascontiguousarray(Uall[:, :q])
    res = AUall[:, :q] - eigvecs * eigvals.reshape(1, -1)
    res_fro = float(xp.linalg.norm(res))
    denom = max(float(xp.linalg.norm(AUall[:, :q])), 1e-30)
    mu = float(vals[q]) if int(vals.size) > q else float(eigvals[-1])
    return ExtraEigResult(
        values=eigvals,
        vectors=eigvecs,
        mu=mu,
        info={
            "method": "rayleigh_ritz",
            "rr_basis_dim": int(Q.shape[1]),
            "rr_orthogonalize": orth_eff,
            "rr_time_s": float(time.perf_counter() - t0),
            "residual_fro": res_fro,
            "residual_fro_rel": res_fro / denom,
            "surrogate_mu": mu,
            "mu": mu,
            "full_matvec_passes": 1,
        },
    )


def embed_coordinate_basis(xp: Any, M: int, S: Array, V: Array) -> Array:
    """Dense embedding U0 = I_S V."""
    S_gpu = xp.asarray(S, dtype=xp.int64).reshape(-1)
    V = xp.asarray(V, dtype=xp.complex128)
    U = xp.zeros((int(M), int(V.shape[1])), dtype=xp.complex128)
    U[S_gpu, :] = V
    return xp.ascontiguousarray(U)


# -----------------------------------------------------------------------------
# Algorithm A. Compact coordinate Nyström.
# -----------------------------------------------------------------------------

def compact_coordinate_nystrom(
    *,
    xp: Any,
    M: int,
    q_out: int,
    build_submatrix_gpu: SubmatrixBuilder,
    s: int,
    seed: int = 0,
    weights: Optional[np.ndarray] = None,
    lowfreq_indices: Optional[np.ndarray] = None,
    lowfreq_ratio: float = 0.25,
    reg_lambda: float = 0.0,
) -> ExtraEigResult:
    """Single support compact coordinate Nyström preconditioner.

    Algorithm idea
    --------------
    1. Draw a coordinate support S with |S|=s.
    2. Form W=A[S,S] by the caller's fast Toeplitz/submatrix builder.
    3. Compute the top q+1 eigenpairs of W.
    4. Return compact data (S, V_q, theta_q, mu) for the preconditioner

           P z = z - I_S V diag(1 - mu/theta_i) V^* I_S^* z.

    This has no full A-matvec.  It is therefore extremely cheap, but it only
    deflates the restricted coordinate surrogate W, not the full A.
    """
    t0 = time.perf_counter()
    q = int(q_out)
    S = _choice_without_replacement(
        M=int(M), s=int(s), seed=int(seed), weights=weights,
        lowfreq_indices=lowfreq_indices, lowfreq_ratio=float(lowfreq_ratio),
    )
    W = build_submatrix_gpu(S)
    vals, V = _top_eigh(xp, W, q + 1)
    vals = xp.maximum(xp.real(vals), 0.0) + float(reg_lambda)
    theta_q = xp.ascontiguousarray(vals[:q])
    Vq = xp.ascontiguousarray(V[:, :q])
    mu = float(vals[q]) if int(vals.size) > q else float(theta_q[-1])
    S_gpu = xp.ascontiguousarray(xp.asarray(S, dtype=xp.int64))
    info = {
        "method": "compact_coordinate_nystrom",
        "precond_kind": "coordinate_nystrom",
        "compact_basis": True,
        "S_gpu": S_gpu,
        "V_gpu": Vq,
        "theta_gpu": theta_q,
        "surrogate_mu": mu,
        "mu": mu,
        "support_size": int(s),
        "full_matvec_passes": 0,
        "time_s": float(time.perf_counter() - t0),
    }
    return ExtraEigResult(theta_q, Vq, mu, info)


# -----------------------------------------------------------------------------
# Algorithm B. Multi-sketch coordinate ensemble.
# -----------------------------------------------------------------------------

def ensemble_coordinate_nystrom(
    *,
    xp: Any,
    M: int,
    q_each: int,
    build_submatrix_gpu: SubmatrixBuilder,
    s: int,
    n_sketches: int = 8,
    seed: int = 0,
    weights: Optional[np.ndarray] = None,
    lowfreq_indices: Optional[np.ndarray] = None,
    lowfreq_ratio: float = 0.25,
    reg_lambda: float = 0.0,
    gamma: float = 0.25,
) -> ExtraEigResult:
    """Randomized ensemble of compact coordinate sketches.

    Algorithm idea
    --------------
    A single coordinate support can be biased.  Here we sample L supports and
    average L cheap spectral corrections.  Downstream apply should perform

        z <- z - (gamma/L) sum_l I_{S_l} V_l diag(1 - mu_l/theta_l) V_l^* z[S_l].

    There is still no full A-matvec.  The price is a larger apply cost
    O(L s q_each), but all operations are gather/small GEMM/scatter.
    """
    t0 = time.perf_counter()
    entries: list[dict[str, Any]] = []
    all_theta = []
    for ell in range(int(n_sketches)):
        res = compact_coordinate_nystrom(
            xp=xp,
            M=M,
            q_out=q_each,
            build_submatrix_gpu=build_submatrix_gpu,
            s=s,
            seed=int(seed) + 104729 * ell,
            weights=weights,
            lowfreq_indices=lowfreq_indices,
            lowfreq_ratio=lowfreq_ratio,
            reg_lambda=reg_lambda,
        )
        entries.append({
            "S_gpu": res.info["S_gpu"],
            "V_gpu": res.info["V_gpu"],
            "theta_gpu": res.info["theta_gpu"],
            "mu": res.mu,
        })
        all_theta.append(res.values)
    theta_stack = xp.concatenate([xp.asarray(t) for t in all_theta]) if all_theta else xp.asarray([])
    mu = float(xp.median(theta_stack)) if int(theta_stack.size) else None
    info = {
        "method": "ensemble_coordinate_nystrom",
        "precond_kind": "ensemble_coordinate_nystrom",
        "compact_basis": True,
        "ensemble_entries": entries,
        "ensemble_gamma": float(gamma),
        "ensemble_size": int(n_sketches),
        "q_each": int(q_each),
        "support_size": int(s),
        "full_matvec_passes": 0,
        "time_s": float(time.perf_counter() - t0),
        "surrogate_mu": mu,
        "mu": mu,
    }
    return ExtraEigResult(theta_stack, None, mu, info)


# -----------------------------------------------------------------------------
# Algorithm C. Random support search + one full matvec lift.
# -----------------------------------------------------------------------------

def random_support_lift(
    *,
    xp: Any,
    M: int,
    q_out: int,
    build_submatrix_gpu: SubmatrixBuilder,
    apply_A_block_gpu: Matvec,
    s: int,
    q_each: int = 8,
    n_sketches: int = 16,
    r_full: int = 24,
    seed: int = 0,
    weights: Optional[np.ndarray] = None,
    lowfreq_indices: Optional[np.ndarray] = None,
    lowfreq_ratio: float = 0.25,
    reg_lambda: float = 0.0,
    block_cols: int = 16,
    final_ritz: bool = True,
    orthogonalize: str = "qr",
    eig_floor: float = 1e-12,
) -> ExtraEigResult:
    """Many cheap support sketches, then one full-space lift on selected directions.

    Algorithm idea
    --------------
    1. Draw L supports S_l and solve W_l=A[S_l,S_l].
    2. Score every small eigenvector by its estimated flattening gain

           gain = max(theta_i - mu_l, 0).

    3. Keep the top r_full directions across all sketches.
    4. Embed them as sparse vectors U0=I_S V.
    5. Use one full operator call Y=A U0 to recover leaked S^c components.
    6. Optionally run Rayleigh--Ritz in span(Y theta^{-1}).

    This is designed for your desired regime: most trials are O(s^2) sketches;
    only the selected r_full directions pay O(r_full M log M).
    """
    t0 = time.perf_counter()
    candidates: list[tuple[float, float, Array, np.ndarray, int]] = []
    q_each_eff = int(max(1, q_each))
    for ell in range(int(n_sketches)):
        S = _choice_without_replacement(
            M=int(M), s=int(s), seed=int(seed) + 13007 * ell,
            weights=weights, lowfreq_indices=lowfreq_indices,
            lowfreq_ratio=lowfreq_ratio,
        )
        W = build_submatrix_gpu(S)
        vals, V = _top_eigh(xp, W, q_each_eff + 1)
        vals = xp.maximum(xp.real(vals), 0.0) + float(reg_lambda)
        mu_l = float(vals[q_each_eff]) if int(vals.size) > q_each_eff else float(vals[-1])
        vals_np = _to_np(vals)
        for i in range(min(q_each_eff, int(V.shape[1]))):
            gain = max(float(vals_np[i]) - mu_l, 0.0)
            candidates.append((gain, float(vals_np[i]), V[:, i], S, ell))

    if not candidates:
        raise RuntimeError("random_support_lift found no candidate directions.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    r = int(max(1, min(int(r_full), len(candidates))))
    chosen = candidates[:r]

    U0 = xp.zeros((int(M), r), dtype=xp.complex128)
    theta = xp.empty((r,), dtype=xp.float64)
    for j, (_gain, th, vj, S, _ell) in enumerate(chosen):
        S_gpu = xp.asarray(S, dtype=xp.int64)
        U0[S_gpu, j] = xp.asarray(vj, dtype=xp.complex128)
        theta[j] = float(th)

    AU0 = apply_matvec_columns(xp, apply_A_block_gpu, U0, block_cols=block_cols)
    theta_safe = xp.maximum(theta, xp.asarray(float(eig_floor), dtype=xp.float64))
    U1 = AU0 / theta_safe.reshape(1, -1)
    Q, orth_eff = orthonormalize(xp, U1, mode=orthogonalize)

    if final_ritz:
        rr = rayleigh_ritz(
            xp=xp,
            apply_A_block_gpu=apply_A_block_gpu,
            Q0=Q,
            q_out=min(int(q_out), r),
            block_cols=block_cols,
            orthogonalize="none",
        )
        rr.info.update({
            "method": "random_support_lift",
            "support_search_sketches": int(n_sketches),
            "support_size": int(s),
            "q_each": int(q_each),
            "r_full": int(r),
            "initial_orthogonalize": orth_eff,
            "full_matvec_passes": 2,  # AU0 plus AQ in RR
            "time_s": float(time.perf_counter() - t0),
        })
        return rr

    # One-pass version: returns normalized lifted basis with sketch eigenvalues.
    q = int(min(int(q_out), r))
    vals = theta[:q]
    U = xp.ascontiguousarray(Q[:, :q])
    mu = float(theta[q]) if int(theta.size) > q else float(vals[-1])
    return ExtraEigResult(
        vals,
        U,
        mu,
        {
            "method": "random_support_lift_onepass",
            "support_search_sketches": int(n_sketches),
            "support_size": int(s),
            "q_each": int(q_each),
            "r_full": int(r),
            "full_matvec_passes": 1,
            "time_s": float(time.perf_counter() - t0),
            "surrogate_mu": mu,
            "mu": mu,
        },
    )


# -----------------------------------------------------------------------------
# Algorithm D. GPU-friendly randomized range finder / randomized power iteration.
# -----------------------------------------------------------------------------

def _structured_random_matrix(
    xp: Any,
    M: int,
    r: int,
    *,
    seed: int,
    kind: str = "gaussian",
    weights: Optional[np.ndarray] = None,
    sparsity: int = 8,
) -> Array:
    """Construct a GPU-friendly random test matrix Omega."""
    kind = str(kind or "gaussian").lower()
    rng = _rng(seed)
    if kind == "gaussian":
        re = rng.standard_normal((int(M), int(r)))
        im = rng.standard_normal((int(M), int(r)))
        return xp.asarray(re + 1j * im, dtype=xp.complex128) / math.sqrt(max(1, int(r)))
    if kind in ("rademacher", "sign"):
        re = rng.choice([-1.0, 1.0], size=(int(M), int(r)))
        im = rng.choice([-1.0, 1.0], size=(int(M), int(r)))
        return xp.asarray(re + 1j * im, dtype=xp.complex128) / math.sqrt(2.0 * max(1, int(M)))
    if kind in ("sparse", "countsketch"):
        Omega = xp.zeros((int(M), int(r)), dtype=xp.complex128)
        p = None
        if weights is not None:
            w = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0)
            if int(w.size) == int(M) and float(w.sum()) > 0.0:
                p = w / float(w.sum())
        nnz = int(max(1, sparsity))
        for j in range(int(r)):
            idx = rng.choice(int(M), size=min(nnz, int(M)), replace=False, p=p)
            signs = rng.choice([-1.0, 1.0], size=len(idx))
            Omega[xp.asarray(idx, dtype=xp.int64), j] = xp.asarray(signs, dtype=xp.complex128) / math.sqrt(nnz)
        return Omega
    raise ValueError(f"unknown random matrix kind: {kind}")


def rand_range_onepass(
    *,
    xp: Any,
    M: int,
    q_out: int,
    apply_A_block_gpu: Matvec,
    oversample: int = 16,
    power_iters: int = 0,
    seed: int = 0,
    omega_kind: str = "gaussian",
    weights: Optional[np.ndarray] = None,
    block_cols: int = 16,
    final_ritz: bool = True,
    sparsity: int = 8,
) -> ExtraEigResult:
    """Randomized range finder / randomized power iteration.

    Algorithm idea
    --------------
    Draw Omega with r=q+oversample columns and compute

        Y = A^{power_iters+1} Omega.

    QR gives a basis Q for the approximate dominant range.  If final_ritz=True,
    run one extra A Q pass and solve the small projected eigenproblem.

    This is the GPU-friendly block version of randomized SVD.  It replaces many
    sequential Lanczos matvecs by a small number of block matvec passes.
    """
    t0 = time.perf_counter()
    r = int(q_out) + int(max(0, oversample))
    Omega = _structured_random_matrix(
        xp, int(M), r, seed=int(seed), kind=omega_kind,
        weights=weights, sparsity=sparsity,
    )
    Y = apply_matvec_columns(xp, apply_A_block_gpu, Omega, block_cols=block_cols)
    passes = 1
    for _ in range(int(max(0, power_iters))):
        Q, _ = orthonormalize(xp, Y, mode="qr")
        Y = apply_matvec_columns(xp, apply_A_block_gpu, Q, block_cols=block_cols)
        passes += 1
    Q, _ = orthonormalize(xp, Y, mode="qr")
    if final_ritz:
        rr = rayleigh_ritz(
            xp=xp,
            apply_A_block_gpu=apply_A_block_gpu,
            Q0=Q,
            q_out=int(q_out),
            block_cols=block_cols,
            orthogonalize="none",
        )
        rr.info.update({
            "method": "rand_range_onepass",
            "omega_kind": omega_kind,
            "oversample": int(oversample),
            "power_iters": int(power_iters),
            "full_matvec_passes": int(passes + 1),
            "time_s": float(time.perf_counter() - t0),
        })
        return rr
    # Basis-only fast mode.
    vals = xp.full((int(q_out),), xp.nan, dtype=xp.float64)
    return ExtraEigResult(
        vals,
        xp.ascontiguousarray(Q[:, : int(q_out)]),
        None,
        {
            "method": "rand_range_onepass_basis_only",
            "omega_kind": omega_kind,
            "oversample": int(oversample),
            "power_iters": int(power_iters),
            "full_matvec_passes": int(passes),
            "time_s": float(time.perf_counter() - t0),
        },
    )


# -----------------------------------------------------------------------------
# Algorithm E. Chebyshev filtered subspace iteration.
# -----------------------------------------------------------------------------

def chebyshev_filtered_subspace(
    *,
    xp: Any,
    M: int,
    q_out: int,
    apply_A_block_gpu: Matvec,
    lambda_low: float,
    lambda_cut: float,
    degree: int = 4,
    oversample: int = 16,
    seed: int = 0,
    omega_kind: str = "gaussian",
    weights: Optional[np.ndarray] = None,
    block_cols: int = 16,
    final_ritz: bool = True,
) -> ExtraEigResult:
    """Chebyshev polynomial filtered subspace iteration.

    Algorithm idea
    --------------
    A polynomial p_m(A) is applied to a random block Omega.  The Chebyshev
    recurrence suppresses eigencomponents in the unwanted interval

        [lambda_low, lambda_cut]

    while components above lambda_cut grow rapidly.  This can separate a cluster
    of top eigenvalues faster than plain power iteration if the interval is
    specified well.

    Mapping:
        T_j((A-cI)/e), c=(lambda_cut+lambda_low)/2, e=(lambda_cut-lambda_low)/2.

    For top-eigen filtering, eigenvalues larger than lambda_cut map outside
    [-1, 1] and are amplified by T_j.
    """
    if not (float(lambda_cut) > float(lambda_low)):
        raise ValueError("Require lambda_cut > lambda_low for Chebyshev filtering.")
    t0 = time.perf_counter()
    r = int(q_out) + int(max(0, oversample))
    Omega = _structured_random_matrix(
        xp, int(M), r, seed=int(seed), kind=omega_kind, weights=weights,
    )
    c = 0.5 * (float(lambda_cut) + float(lambda_low))
    e = 0.5 * (float(lambda_cut) - float(lambda_low))

    def scaled_A(V: Array) -> Array:
        return (apply_matvec_columns(xp, apply_A_block_gpu, V, block_cols=block_cols) - c * V) / e

    deg = int(max(0, degree))
    if deg == 0:
        Y = Omega
        passes = 0
    elif deg == 1:
        Y = scaled_A(Omega)
        passes = 1
    else:
        T0 = Omega
        T1 = scaled_A(Omega)
        passes = 1
        for _ in range(2, deg + 1):
            T2 = 2.0 * scaled_A(T1) - T0
            passes += 1
            T0, T1 = T1, T2
        Y = T1
    Q, _ = orthonormalize(xp, Y, mode="qr")
    if final_ritz:
        rr = rayleigh_ritz(
            xp=xp,
            apply_A_block_gpu=apply_A_block_gpu,
            Q0=Q,
            q_out=int(q_out),
            block_cols=block_cols,
            orthogonalize="none",
        )
        rr.info.update({
            "method": "chebyshev_filtered_subspace",
            "degree": int(deg),
            "lambda_low": float(lambda_low),
            "lambda_cut": float(lambda_cut),
            "oversample": int(oversample),
            "full_matvec_passes": int(passes + 1),
            "time_s": float(time.perf_counter() - t0),
        })
        return rr
    vals = xp.full((int(q_out),), xp.nan, dtype=xp.float64)
    return ExtraEigResult(
        vals,
        xp.ascontiguousarray(Q[:, : int(q_out)]),
        None,
        {
            "method": "chebyshev_filtered_subspace_basis_only",
            "degree": int(deg),
            "lambda_low": float(lambda_low),
            "lambda_cut": float(lambda_cut),
            "full_matvec_passes": int(passes),
            "time_s": float(time.perf_counter() - t0),
        },
    )


# -----------------------------------------------------------------------------
# Algorithm F. Sample-side Nyström / stochastic empirical operator hook.
# -----------------------------------------------------------------------------

def sample_side_nystrom_range(
    *,
    xp: Any,
    M: int,
    q_out: int,
    n_samples: int,
    sample_size: int,
    build_sample_matvec: SampleMatvecBuilder,
    seed: int = 0,
    sample_prob: Optional[np.ndarray] = None,
    apply_A_block_gpu_full: Optional[Matvec] = None,
    oversample: int = 16,
    power_iters: int = 0,
    block_cols: int = 16,
    final_full_ritz: bool = False,
) -> ExtraEigResult:
    """Sample-side Nyström / stochastic empirical operator approximation.

    Algorithm idea
    --------------
    This method attacks the N-dependence rather than the M-dependence.  The
    caller samples m << N training examples and builds an approximate operator

        A_hat = (N/m) B_sample^* B_sample + lambda I.

    The actual construction is problem-specific, so the caller supplies
    build_sample_matvec(sample_indices).  We then run randomized range finding
    on A_hat.  Optionally, one final Rayleigh--Ritz can be performed using the
    full operator A to recalibrate the vectors.

    This is useful when the O(N) NUFFT/precompute or empirical sum is the
    bottleneck.  It is not meant to replace full-space refinement when M-matvecs
    dominate.
    """
    t0 = time.perf_counter()
    N = int(n_samples)
    m = int(max(1, min(int(sample_size), N)))
    rng = _rng(seed)
    if sample_prob is None:
        idx = rng.choice(N, size=m, replace=False)
    else:
        p = np.asarray(sample_prob, dtype=np.float64).reshape(-1)
        if int(p.size) != N:
            raise ValueError(f"sample_prob.size={p.size} != n_samples={N}")
        p = np.maximum(p, 0.0)
        p = p / float(p.sum())
        idx = rng.choice(N, size=m, replace=False, p=p)
    Ahat = build_sample_matvec(np.asarray(idx, dtype=np.int64))
    res = rand_range_onepass(
        xp=xp,
        M=int(M),
        q_out=int(q_out),
        apply_A_block_gpu=Ahat,
        oversample=int(oversample),
        power_iters=int(power_iters),
        seed=int(seed) + 17,
        omega_kind="gaussian",
        block_cols=block_cols,
        final_ritz=True,
    )
    if final_full_ritz:
        if apply_A_block_gpu_full is None:
            raise ValueError("final_full_ritz=True requires apply_A_block_gpu_full.")
        # Recalibrate approximate vectors under the true full operator.
        res = rayleigh_ritz(
            xp=xp,
            apply_A_block_gpu=apply_A_block_gpu_full,
            Q0=res.vectors,
            q_out=int(q_out),
            block_cols=block_cols,
            orthogonalize="qr",
        )
        res.info["full_recalibration"] = True
    res.info.update({
        "method": "sample_side_nystrom_range",
        "sample_size": int(m),
        "n_samples": int(N),
        "sample_indices": np.asarray(idx, dtype=np.int64),
        "time_s_total": float(time.perf_counter() - t0),
    })
    return res


__all__ = [
    "ExtraEigResult",
    "compact_coordinate_nystrom",
    "ensemble_coordinate_nystrom",
    "random_support_lift",
    "rand_range_onepass",
    "chebyshev_filtered_subspace",
    "sample_side_nystrom_range",
    "rayleigh_ritz",
    "apply_matvec_columns",
]
