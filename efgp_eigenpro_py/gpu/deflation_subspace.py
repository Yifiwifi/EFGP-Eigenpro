from __future__ import annotations

"""
deflation_subspace.py
=====================

Cheap "direction detector" subspace generation for multi-fidelity deflated CG.

A low-fidelity operator ``A_tilde`` is used *only* to find the deflation subspace
``Z`` (n x m); the deflation projector itself is later built with the true
high-fidelity ``A`` in :mod:`deflation_core`.

Three low-fidelity backends (``method``):

- ``"coord_nystrom"`` : coordinate Nystrom A[S,S] (frequency-axis subsampling).
  Reuses the existing Toeplitz submatrix machinery; the heavy work is a small
  s x s dense eigensolve, with **no full-space matvec**.  Z = I_S V (embedded).

- ``"freq_trunc"``    : frequency-truncated nested coarse grid.  Keeping h fixed and
  shrinking hm -> hm_t makes the coarse frequencies a centered subset of the fine
  grid, so the coarse Toeplitz column is the central sub-block of the fine
  ``xtxcol`` (no new NUFFT).  Subspace iteration runs on the (cheaper) coarse
  operator; Z is zero-padded back to the fine grid.

- ``"float32"``       : same grid, complex64 matvec (lower arithmetic/bandwidth).

For deflation the subspace *angle* matters more than eigenvalue accuracy, so the
default solver is randomized subspace iteration with loose tolerance; ``eigsh`` is
available as a benchmark.  ``which="LM"`` (dominant top-m modes) is the default for
this EigenPro-like spectrum (a few large outliers dominate kappa); ``which="SM"``
is kept for experiments.
"""

from typing import Any, Callable, Optional
import time

import numpy as np

from .contexts import GPUDataContext, GPUOperatorContext
from .deflation_basis import (
    CanonicalFreqBasis,
    CoordNystromBasis,
    DenseBasis,
    FreqTruncBasis,
)
from .v1_ops import apply_A_block_v1
from .v3_eigenspace import (
    EigenspaceConfig,
    estimate_top_eigenspace_v3,
    _sample_frequency_indices,
    _toeplitz_submatrix_gpu,
)
from .v3_nystrom_refine import embed_coordinate_basis


# ---------------------------------------------------------------------------
# Public entry.
# ---------------------------------------------------------------------------

def build_deflation_subspace(
    backend: Any,
    data_ctx: GPUDataContext,
    reg_lambda: float,
    op_ctx: GPUOperatorContext,
    *,
    method: str = "coord_nystrom",
    m: int = 64,
    which: str = "LM",
    eig_solver: str = "subspace_iter",
    tol: float = 1e-3,
    n_iter: int = 10,
    oversample: int = 12,
    hm_t: Optional[int] = None,
    coarse_ratio: float = 0.5,
    return_basis: bool = False,
    freq_box_mode: str = "center",
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """
    Build a raw deflation subspace ``Z`` (n x m) from a low-fidelity operator.

    Returns ``(Z_fine, diag)``.  ``Z_fine`` is NOT yet orthonormalized for the
    projector; :func:`deflation_core.build_deflation_data` does the Gram-eigh
    orthonormalization and rank truncation.
    """
    method = str(method).lower()
    if int(m) < 1:
        raise ValueError("m must be >= 1.")
    if method in ("coord_nystrom", "coordinate_nystrom", "coord", "nystrom"):
        return _subspace_coord_nystrom(
            backend, data_ctx, float(reg_lambda),
            m=int(m), which=which, oversample=int(oversample), seed=int(seed),
            return_basis=bool(return_basis),
        )
    if method in ("freq_trunc", "frequency_truncation", "coarse_grid", "freq"):
        return _subspace_freq_trunc(
            backend, data_ctx, float(reg_lambda), op_ctx,
            m=int(m), which=which, eig_solver=eig_solver, tol=float(tol),
            n_iter=int(n_iter), oversample=int(oversample),
            hm_t=hm_t, coarse_ratio=float(coarse_ratio), seed=int(seed),
            return_basis=bool(return_basis),
        )
    if method in ("float32", "complex64", "lowprec", "single"):
        return _subspace_float32(
            backend, data_ctx, float(reg_lambda),
            m=int(m), which=which, eig_solver=eig_solver, tol=float(tol),
            n_iter=int(n_iter), oversample=int(oversample), seed=int(seed),
            return_basis=bool(return_basis),
        )
    if method in ("freq_box", "frequency_box", "canonical_freq", "canonical_frequency"):
        return _subspace_freq_box(
            backend, data_ctx,
            m=int(m), mode=str(freq_box_mode), return_basis=bool(return_basis),
        )
    raise ValueError(
        f"Unknown deflation subspace method {method!r}. Expected one of: "
        "coord_nystrom, freq_trunc, float32, freq_box."
    )


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------

def _asnumpy(x: Any) -> np.ndarray:
    if hasattr(x, "get"):
        return np.asarray(x.get())
    return np.asarray(x)


def _make_block_matvec(
    backend: Any,
    ctx: GPUDataContext,
    reg_lambda: float,
    op_ctx: GPUOperatorContext,
    *,
    cast_to_complex128: bool,
) -> tuple[Callable[[Any], Any], list[int]]:
    """
    Block matvec ``V -> A_ctx V`` (column loop), counting low-fidelity matvecs.
    If ``cast_to_complex128`` the output is upcast so callers expecting a fixed
    dtype (e.g. estimate_top_eigenspace_v3) stay happy while arithmetic still runs
    in the ctx dtype.
    """
    xp = backend.xp
    counter = [0]

    def block(V: Any) -> Any:
        Vv = xp.asarray(V)
        if Vv.ndim == 1:
            Vv = Vv.reshape(-1, 1)
        out = apply_A_block_v1(
            backend,
            ctx,
            Vv,
            float(reg_lambda),
            op_ctx,
            block_cols="auto",
        )
        out = xp.asarray(out, dtype=xp.complex128)
        counter[0] += int(Vv.shape[1])
        if cast_to_complex128:
            return out
        return out

    return block, counter


def _estimate_subspace(
    backend: Any,
    block: Callable[[Any], Any],
    size: int,
    *,
    m: int,
    which: str,
    eig_solver: str,
    tol: float,
    n_iter: int,
    oversample: int,
) -> tuple[Any, dict[str, Any]]:
    """Run subspace_iter / eigsh through estimate_top_eigenspace_v3 and return vecs (size x m)."""
    which = str(which).upper()
    block_size = int(min(size - 1, m + max(1, oversample)))
    if block_size <= m:
        block_size = int(min(size - 1, m + 1))
    solver = str(eig_solver).lower()
    if which == "SM" and solver in ("subspace_iter", "subspace"):
        # Plain subspace iteration converges to dominant modes; SM needs eigsh.
        solver = "cupy_eigsh"
    method_cfg: dict[str, Any] = {"tol": float(tol)}
    if solver in ("cupy_eigsh", "eigsh"):
        method_cfg["which"] = "SA" if which == "SM" else "LA"
        eig_method = "cupy_eigsh"
    elif solver in ("subspace_iter", "subspace"):
        eig_method = "subspace_iter"
    else:
        raise ValueError("eig_solver must be 'subspace_iter' or 'eigsh'.")
    cfg = EigenspaceConfig(
        q_max=int(m),
        block_size=int(block_size),
        n_iter=int(max(1, n_iter)),
        method=eig_method,
        eig_method=eig_method,
        method_cfg=method_cfg,
    )
    vals, vecs, diag = estimate_top_eigenspace_v3(
        backend=backend,
        apply_A_block_gpu=block,
        size=int(size),
        cfg=cfg,
    )
    return vecs, {"eig_solver": eig_method, "which": which, **{k: diag.get(k) for k in ("residual_fro_rel", "n_iter", "block_size")}}


# ---------------------------------------------------------------------------
# Method A reuse: coordinate Nystrom (no full-space matvec).
# ---------------------------------------------------------------------------

def _subspace_coord_nystrom(
    backend: Any,
    data_ctx: GPUDataContext,
    reg_lambda: float,
    *,
    m: int,
    which: str,
    oversample: int,
    seed: int,
    return_basis: bool,
) -> tuple[Any, dict[str, Any]]:
    t0 = time.perf_counter()
    xp = backend.xp
    if data_ctx.weights_gpu_flat is None or data_ctx.gf_gpu is None:
        raise RuntimeError("data_ctx must be precomputed before coord_nystrom subspace.")
    mtot = int(data_ctx.meta["mtot"])
    dim = int(data_ctx.meta["dim"])
    n = int(data_ctx.weights_gpu_flat.size)
    weights_gpu = xp.asarray(data_ctx.weights_gpu_flat, dtype=xp.float64).reshape(-1)
    weights_np = getattr(data_ctx, "weights_np_flat", None)
    if weights_np is None:
        weights_np = _asnumpy(weights_gpu)
    xtxcol = getattr(data_ctx, "xtxcol_gpu", None)
    if xtxcol is None:
        xtxcol = xp.ascontiguousarray(backend.fft.ifftn(data_ctx.gf_gpu))

    s_idx = _sample_frequency_indices(
        weights_np,
        q_max=int(m),
        surrogate_size=None,
        oversample=int(max(2, oversample)),
        lowfreq_ratio=0.25,
        seed=int(seed),
    )
    s = int(s_idx.size)
    if s < m + 1:
        raise ValueError(f"coordinate support s={s} too small for m={m}.")

    w0 = _toeplitz_submatrix_gpu(xp, xtxcol, weights_gpu, s_idx, mtot=mtot, dim=dim)
    if reg_lambda != 0.0:
        w0 = w0 + float(reg_lambda) * xp.eye(s, dtype=w0.dtype)
    w0 = 0.5 * (w0 + w0.conj().T)
    ew, ev = xp.linalg.eigh(w0)
    ew = xp.real(ew)
    order = xp.argsort(ew)
    if str(which).upper() == "SM":
        pick = order[:m]
    else:
        pick = order[::-1][:m]
    V = xp.ascontiguousarray(ev[:, pick])
    s_gpu = xp.ascontiguousarray(xp.asarray(s_idx, dtype=xp.int64))
    if return_basis:
        Z = CoordNystromBasis(xp, n, s_gpu, V)
    else:
        Z = embed_coordinate_basis(xp, n, s_gpu, V, dtype=xp.complex128)
    diag = {
        "subspace_method": "coord_nystrom",
        "basis_kind": "coord_nystrom" if return_basis else "dense",
        "which": str(which).upper(),
        "support_size": s,
        "lowfi_n_matvec": 0,
        "lowfi_kind": "coordinate_submatrix_eigh",
        "n_tilde": s,
        "time_subspace": float(time.perf_counter() - t0),
    }
    return Z, diag


# ---------------------------------------------------------------------------
# Method B: frequency-truncated nested coarse grid.
# ---------------------------------------------------------------------------

def make_coarse_ctx(
    backend: Any, data_ctx: GPUDataContext, hm_t: int
) -> tuple[GPUDataContext, int]:
    """
    Build a lightweight coarse :class:`GPUDataContext` by central-slicing the fine
    ``xtxcol`` (Toeplitz column) and ``weights``.  Same h, smaller hm_t.
    """
    xp = backend.xp
    mtot = int(data_ctx.meta["mtot"])
    dim = int(data_ctx.meta["dim"])
    hm = (mtot - 1) // 2
    hm_t = int(hm_t)
    if hm_t < 1 or hm_t >= hm:
        raise ValueError(f"hm_t must satisfy 1 <= hm_t < hm={hm}; got {hm_t}.")
    mtot_t = 2 * hm_t + 1

    weights_flat = xp.asarray(data_ctx.weights_gpu_flat).reshape(-1)
    w_nd = weights_flat.reshape((mtot,) * dim)
    sl_w = tuple(slice(hm - hm_t, hm + hm_t + 1) for _ in range(dim))
    w_t = xp.ascontiguousarray(w_nd[sl_w].reshape(-1))

    xtxcol = getattr(data_ctx, "xtxcol_gpu", None)
    if xtxcol is None:
        xtxcol = xp.ascontiguousarray(backend.fft.ifftn(data_ctx.gf_gpu))
    start = mtot - mtot_t
    stop = start + (2 * mtot_t - 1)
    sl_x = tuple(slice(start, stop) for _ in range(dim))
    xtx_t = xp.ascontiguousarray(xtxcol[sl_x])
    gf_t = xp.ascontiguousarray(backend.fft.fftn(xtx_t))

    coarse = GPUDataContext(x_gpu=data_ctx.x_gpu, y_gpu=data_ctx.y_gpu)
    coarse.weights_gpu_flat = w_t
    coarse.weights_gpu_nd = w_t.reshape((mtot_t,) * dim)
    coarse.gf_gpu = gf_t
    coarse.xtxcol_gpu = xtx_t
    coarse.meta = {
        "mtot": mtot_t,
        "dim": dim,
        "gf_absmax": float(xp.max(xp.abs(gf_t))),
    }
    return coarse, mtot_t


def embed_freq(xp: Any, Z_coarse: Any, mtot: int, mtot_t: int, dim: int) -> Any:
    """Zero-pad coarse-grid columns into the central block of the fine grid."""
    hm = (mtot - 1) // 2
    hm_t = (mtot_t - 1) // 2
    m = int(Z_coarse.shape[1])
    Zf_nd = xp.zeros((mtot,) * dim + (m,), dtype=Z_coarse.dtype)
    sl = tuple(slice(hm - hm_t, hm + hm_t + 1) for _ in range(dim)) + (slice(None),)
    Zf_nd[sl] = Z_coarse.reshape((mtot_t,) * dim + (m,))
    return xp.ascontiguousarray(Zf_nd.reshape(int(mtot) ** int(dim), m))


def _subspace_freq_trunc(
    backend: Any,
    data_ctx: GPUDataContext,
    reg_lambda: float,
    op_ctx: GPUOperatorContext,
    *,
    m: int,
    which: str,
    eig_solver: str,
    tol: float,
    n_iter: int,
    oversample: int,
    hm_t: Optional[int],
    coarse_ratio: float,
    seed: int,
    return_basis: bool,
) -> tuple[Any, dict[str, Any]]:
    t0 = time.perf_counter()
    xp = backend.xp
    mtot = int(data_ctx.meta["mtot"])
    dim = int(data_ctx.meta["dim"])
    hm = (mtot - 1) // 2
    if hm_t is None:
        hm_t = max(1, int(round(float(coarse_ratio) * hm)))
    hm_t = int(min(max(1, hm_t), hm - 1))
    coarse, mtot_t = make_coarse_ctx(backend, data_ctx, hm_t)
    n_tilde = int(mtot_t) ** int(dim)
    if n_tilde <= m + max(1, oversample):
        raise ValueError(
            f"coarse grid n_tilde={n_tilde} too small for m={m}; increase hm_t/coarse_ratio."
        )

    coarse_op = GPUOperatorContext()
    block, counter = _make_block_matvec(
        backend, coarse, reg_lambda, coarse_op, cast_to_complex128=True
    )
    vecs, eig_diag = _estimate_subspace(
        backend, block, n_tilde,
        m=m, which=which, eig_solver=eig_solver, tol=tol,
        n_iter=n_iter, oversample=oversample,
    )
    Zc = xp.ascontiguousarray(xp.asarray(vecs, dtype=xp.complex128))
    if return_basis:
        Z = FreqTruncBasis(xp, Zc, mtot, mtot_t, dim)
    else:
        Z = embed_freq(xp, Zc, mtot, mtot_t, dim)
    diag = {
        "subspace_method": "freq_trunc",
        "basis_kind": "freq_trunc" if return_basis else "dense",
        "which": str(which).upper(),
        "hm": int(hm),
        "hm_t": int(hm_t),
        "mtot": int(mtot),
        "mtot_t": int(mtot_t),
        "n_tilde": int(n_tilde),
        "coarse_ratio": float(hm_t) / float(hm),
        "lowfi_n_matvec": int(counter[0]),
        "lowfi_kind": "coarse_grid_matvec",
        "time_subspace": float(time.perf_counter() - t0),
        **eig_diag,
    }
    return Z, diag


# ---------------------------------------------------------------------------
# Method C: float32 (complex64) low-precision operator on the same grid.
# ---------------------------------------------------------------------------

def make_lowprec_ctx(backend: Any, data_ctx: GPUDataContext) -> GPUDataContext:
    """Build a same-grid complex64 low-precision context (shares fine grid size)."""
    xp = backend.xp
    mtot = int(data_ctx.meta["mtot"])
    dim = int(data_ctx.meta["dim"])
    gf32 = xp.ascontiguousarray(xp.asarray(data_ctx.gf_gpu).astype(xp.complex64))
    w32 = xp.ascontiguousarray(xp.asarray(data_ctx.weights_gpu_flat, dtype=xp.float64).astype(xp.float32))
    low = GPUDataContext(x_gpu=data_ctx.x_gpu, y_gpu=data_ctx.y_gpu)
    low.weights_gpu_flat = w32
    low.weights_gpu_nd = w32.reshape((mtot,) * dim)
    low.gf_gpu = gf32
    low.xtxcol_gpu = None
    low.meta = {
        "mtot": mtot,
        "dim": dim,
        "complex_dtype": "complex64",
        "gf_absmax": float(xp.max(xp.abs(gf32))),
    }
    return low


def _subspace_float32(
    backend: Any,
    data_ctx: GPUDataContext,
    reg_lambda: float,
    *,
    m: int,
    which: str,
    eig_solver: str,
    tol: float,
    n_iter: int,
    oversample: int,
    seed: int,
    return_basis: bool,
) -> tuple[Any, dict[str, Any]]:
    t0 = time.perf_counter()
    xp = backend.xp
    n = int(data_ctx.weights_gpu_flat.size)
    low = make_lowprec_ctx(backend, data_ctx)
    low_op = GPUOperatorContext()
    block, counter = _make_block_matvec(
        backend, low, reg_lambda, low_op, cast_to_complex128=True
    )
    vecs, eig_diag = _estimate_subspace(
        backend, block, n,
        m=m, which=which, eig_solver=eig_solver, tol=tol,
        n_iter=n_iter, oversample=oversample,
    )
    Z_dense = xp.ascontiguousarray(xp.asarray(vecs, dtype=xp.complex128))
    Z = DenseBasis(xp, Z_dense, kind="float32_dense") if return_basis else Z_dense
    diag = {
        "subspace_method": "float32",
        "basis_kind": "float32_dense" if return_basis else "dense",
        "which": str(which).upper(),
        "n_tilde": int(n),
        "lowfi_n_matvec": int(counter[0]),
        "lowfi_kind": "complex64_matvec",
        "time_subspace": float(time.perf_counter() - t0),
        **eig_diag,
    }
    return Z, diag


# ---------------------------------------------------------------------------
# Method D: pure structured low-frequency coordinate box.
# ---------------------------------------------------------------------------

def _center_frequency_order(mtot: int, dim: int) -> np.ndarray:
    hm = (int(mtot) - 1) // 2
    grids = np.meshgrid(*[np.arange(int(mtot)) for _ in range(int(dim))], indexing="ij")
    radius2 = np.zeros((int(mtot),) * int(dim), dtype=np.int64)
    for g in grids:
        radius2 += (g - hm) ** 2
    return np.argsort(radius2.reshape(-1), kind="stable")


def _subspace_freq_box(
    backend: Any,
    data_ctx: GPUDataContext,
    *,
    m: int,
    mode: str,
    return_basis: bool,
) -> tuple[Any, dict[str, Any]]:
    t0 = time.perf_counter()
    xp = backend.xp
    mtot = int(data_ctx.meta["mtot"])
    dim = int(data_ctx.meta["dim"])
    n = int(data_ctx.weights_gpu_flat.size)
    mode = str(mode).lower()
    if mode == "weight":
        weights = _asnumpy(xp.asarray(data_ctx.weights_gpu_flat).reshape(-1))
        order = np.argsort(-np.abs(weights), kind="stable")
    elif mode == "center":
        order = _center_frequency_order(mtot, dim)
    else:
        raise ValueError("freq_box_mode must be 'center' or 'weight'.")
    if int(m) > int(order.size):
        raise ValueError(f"m={m} exceeds grid size n={order.size}.")
    indices_np = np.ascontiguousarray(order[:int(m)].astype(np.int64))
    indices = xp.ascontiguousarray(xp.asarray(indices_np, dtype=xp.int64))
    if return_basis:
        Z = CanonicalFreqBasis(xp, n, indices)
    else:
        Z = CanonicalFreqBasis(xp, n, indices).to_dense()
    diag = {
        "subspace_method": "freq_box",
        "basis_kind": "freq_box" if return_basis else "dense",
        "freq_box_mode": mode,
        "support_size": int(m),
        "lowfi_n_matvec": 0,
        "lowfi_kind": "canonical_frequency_box",
        "n_tilde": int(m),
        "time_subspace": float(time.perf_counter() - t0),
    }
    return Z, diag
