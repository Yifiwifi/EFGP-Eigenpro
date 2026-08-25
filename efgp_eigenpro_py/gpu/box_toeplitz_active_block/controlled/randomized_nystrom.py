from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RandomizedNystromData:
    """Inverse-like form of a randomized Nyström preconditioner.

    If ``K`` is the unregularized positive-semidefinite part of the system and
    ``K_hat = U diag(eigenvalues) U*``, the stored operation is

        alpha I + U diag(1 / (eigenvalues + reg) - alpha) U*,

    where ``alpha = 1 / (eigenvalues[-1] + reg)``.  This differs from the
    normalized form in Frangella--Tropp--Udell only by a positive scalar, which
    does not change PCG iterates in exact arithmetic.
    """

    U: Any
    UH: Any
    coeff: Any
    coeff_col: Any
    eigenvalues: Any
    alpha: float
    rank: int
    sketch_seed: int
    diagnostics: dict[str, Any]


def _complex_standard_normal(xp: Any, shape: tuple[int, int], seed: int, dtype: Any) -> Any:
    """Return a reproducible real or complex Gaussian sketch on CPU/GPU."""
    dtype = xp.dtype(dtype)
    random_mod = xp.random
    if hasattr(random_mod, "default_rng"):
        rng = random_mod.default_rng(int(seed))
        normal = rng.standard_normal
    else:  # older CuPy
        rng = random_mod.RandomState(int(seed))
        normal = rng.standard_normal
    if dtype.kind == "c":
        real_dtype = xp.float32 if dtype == xp.dtype(xp.complex64) else xp.float64
        scale = 2.0 ** -0.5
        try:
            real = normal(shape, dtype=real_dtype)
            imag = normal(shape, dtype=real_dtype)
        except TypeError:  # NumPy's Generator.standard_normal before dtype-aware backends
            real = xp.asarray(normal(shape), dtype=real_dtype)
            imag = xp.asarray(normal(shape), dtype=real_dtype)
        z = scale * (real + 1j * imag)
        return xp.asarray(z, dtype=dtype)
    return xp.asarray(normal(shape), dtype=dtype)


def _right_solve_lower_adjoint(xp: Any, Y: Any, L: Any) -> Any:
    """Compute ``Y @ inv(L.conj().T)`` without forming an inverse."""
    return xp.linalg.solve(L, Y.conj().T).conj().T


def build_randomized_nystrom_preconditioner(
    backend: Any,
    apply_psd_block: Callable[[Any], Any],
    *,
    size: int,
    rank: int,
    reg_lambda: float,
    seed: int = 0,
    dtype: Any | None = None,
) -> RandomizedNystromData:
    """Build a randomized Nyström preconditioner from block products with ``K``.

    ``apply_psd_block(V)`` must return ``K @ V`` for the unregularized Hermitian
    PSD matrix ``K``.  Regularization is supplied separately so the method acts
    on exactly ``K + reg_lambda I``.
    """
    xp = backend.xp
    n = int(size)
    ell = int(rank)
    reg = float(reg_lambda)
    if n <= 0:
        raise ValueError("size must be positive.")
    if ell <= 0 or ell > n:
        raise ValueError(f"rank must satisfy 1 <= rank <= size; got rank={ell}, size={n}.")
    if reg <= 0.0:
        raise ValueError("reg_lambda must be positive for the Nyström preconditioner.")
    if dtype is None:
        dtype = xp.complex128
    dtype = xp.dtype(dtype)

    omega = _complex_standard_normal(xp, (n, ell), int(seed), dtype)
    omega, _ = xp.linalg.qr(omega, mode="reduced")
    Y = xp.asarray(apply_psd_block(omega), dtype=dtype)
    if tuple(Y.shape) != (n, ell):
        raise ValueError(
            f"apply_psd_block returned shape {tuple(Y.shape)}; expected {(n, ell)}."
        )

    # Stabilized single-pass Nyström construction.  The shift is tied to the
    # working precision and removed from the recovered eigenvalues.
    real_dtype = xp.float32 if dtype == xp.dtype(xp.complex64) else xp.float64
    eps = float(xp.finfo(real_dtype).eps)
    y_fro = float(xp.linalg.norm(Y))
    shift = max(eps * y_fro, eps)
    base_shift = float(shift)
    stabilization_shift = float(base_shift)
    L = None
    Y_shift = None
    for attempt in range(6):
        Y_shift = Y + stabilization_shift * omega
        gram = omega.conj().T @ Y_shift
        gram = 0.5 * (gram + gram.conj().T)
        try:
            L = xp.linalg.cholesky(gram)
            break
        except Exception:
            stabilization_shift = base_shift * (10.0 ** (attempt + 1))
    if L is None or Y_shift is None:
        raise RuntimeError("Nyström sketch Gram matrix remained non-SPD after stabilization.")

    B = _right_solve_lower_adjoint(xp, Y_shift, L)
    U, singular_values, _ = xp.linalg.svd(B, full_matrices=False)
    eigenvalues = xp.maximum(
        xp.real(singular_values) ** 2 - stabilization_shift,
        xp.asarray(0.0, dtype=real_dtype),
    )
    order = xp.argsort(eigenvalues)[::-1]
    eigenvalues = xp.ascontiguousarray(eigenvalues[order])
    U = xp.asfortranarray(U[:, order])

    tail_scale = float(eigenvalues[-1]) + reg
    alpha = 1.0 / max(tail_scale, 1e-300)
    inv_top = 1.0 / xp.maximum(eigenvalues + reg, xp.asarray(1e-300, dtype=real_dtype))
    coeff = xp.ascontiguousarray(inv_top - alpha)
    UH = xp.asfortranarray(U.conj().T)
    diagnostics = {
        "rank": ell,
        "sketch_seed": int(seed),
        "dtype": str(dtype),
        "shift": float(stabilization_shift),
        "base_shift": float(base_shift),
        "cholesky_extra_shift": float(stabilization_shift - base_shift),
        "cholesky_jitter": 0.0,
        "lambda_max_hat": float(eigenvalues[0]),
        "lambda_min_retained_hat": float(eigenvalues[-1]),
        "regularization": reg,
        "storage_bytes": int(U.nbytes + UH.nbytes + coeff.nbytes + eigenvalues.nbytes),
        "psd_block_matvec_columns": ell,
    }
    return RandomizedNystromData(
        U=U,
        UH=UH,
        coeff=coeff,
        coeff_col=coeff.reshape(-1, 1),
        eigenvalues=eigenvalues,
        alpha=float(alpha),
        rank=ell,
        sketch_seed=int(seed),
        diagnostics=diagnostics,
    )


def apply_randomized_nystrom_preconditioner(
    backend: Any,
    data: RandomizedNystromData,
    v: Any,
    *,
    out: Any | None = None,
) -> Any:
    """Apply the stored inverse-like Nyström operation to one or more vectors."""
    xp = backend.xp
    vv = xp.asarray(v, dtype=data.U.dtype)
    if vv.ndim not in (1, 2):
        raise ValueError("v must be a vector or a column block.")
    result = xp.empty_like(vv) if out is None else out
    proj = data.UH @ vv
    if proj.ndim == 1:
        correction = data.U @ (data.coeff * proj)
    else:
        correction = data.U @ (data.coeff_col * proj)
    result[...] = float(data.alpha) * vv + correction
    return result
