from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _as2d_col(xp: Any, x: Any) -> tuple[Any, bool]:
    arr = xp.asarray(x)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("expected a vector or a 2D matrix.")
    return arr, was_1d


def _gram_eigh_transform(xp: Any, C: Any, *, rank_tol: float) -> tuple[Any, int, int, dict[str, Any]]:
    C = 0.5 * (C + C.conj().T)
    m = int(C.shape[0])
    evals, evecs = xp.linalg.eigh(C)
    evals = xp.real(evals)
    smax = float(xp.max(evals)) if int(evals.size) else 0.0
    if smax <= 0.0:
        raise RuntimeError("deflation basis has non-positive Gram spectrum.")
    keep = evals > (float(rank_tol) * smax)
    m_eff = int(keep.sum())
    if m_eff < 1:
        raise RuntimeError("deflation basis collapsed to rank 0 after truncation.")
    T = evecs[:, keep] / xp.sqrt(evals[keep])[None, :]
    rank_dropped = int(m - m_eff)
    diag = {
        "m_eff": m_eff,
        "rank_dropped": rank_dropped,
        "gram_eig_min": float(xp.min(evals)),
        "gram_eig_max": float(xp.max(evals)),
    }
    return xp.ascontiguousarray(T), m_eff, rank_dropped, diag


class DeflationBasis:
    """Structured representation of a fine-space deflation basis Z."""

    n: int
    m: int
    kind: str

    def apply(self, coeff: Any, out: Optional[Any] = None) -> Any:
        """Return Z @ coeff. Supports coeff shapes (m,) and (m, b)."""
        raise NotImplementedError

    def apply_H(self, v: Any) -> Any:
        """Return Z* @ v. Supports v shapes (n,) and (n, b)."""
        raise NotImplementedError

    def gram(self) -> Any:
        """Return Z* Z using the smallest available representation."""
        raise NotImplementedError

    def orthonormalized(self, rank_tol: float = 1e-12) -> tuple["DeflationBasis", dict[str, Any]]:
        raise NotImplementedError

    def columns(self, lo: int, hi: int) -> Any:
        xp = self.xp
        lo = int(lo)
        hi = int(hi)
        eye = xp.eye(self.m, dtype=xp.complex128)[:, lo:hi]
        cols = self.apply(eye)
        if cols.ndim == 1:
            cols = cols.reshape(self.n, 1)
        return xp.ascontiguousarray(cols)

    def to_dense(self) -> Any:
        return self.columns(0, self.m)

    def memory_estimate(self) -> dict[str, Any]:
        return {
            "basis_kind": self.kind,
            "estimated_basis_GB": 0.0,
        }


@dataclass
class DenseBasis(DeflationBasis):
    xp: Any
    Z: Any
    kind: str = "dense"

    def __post_init__(self) -> None:
        Z = self.xp.asarray(self.Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if Z.ndim != 2:
            raise ValueError("DenseBasis Z must be 2D.")
        self.Z = self.xp.ascontiguousarray(Z)
        self.n = int(Z.shape[0])
        self.m = int(Z.shape[1])

    def apply(self, coeff: Any, out: Optional[Any] = None) -> Any:
        C, was_1d = _as2d_col(self.xp, coeff)
        Y = self.Z @ C
        if was_1d:
            Y = Y.reshape(-1)
        if out is not None:
            self.xp.copyto(out, Y)
            return out
        return Y

    def apply_H(self, v: Any) -> Any:
        V, was_1d = _as2d_col(self.xp, v)
        Y = self.Z.conj().T @ V
        return Y.reshape(-1) if was_1d else Y

    def gram(self) -> Any:
        return self.Z.conj().T @ self.Z

    def orthonormalized(self, rank_tol: float = 1e-12) -> tuple[DeflationBasis, dict[str, Any]]:
        xp = self.xp
        T, _m_eff, _rank_dropped, diag = _gram_eigh_transform(
            xp, self.gram(), rank_tol=rank_tol
        )
        Z1 = self.Z @ T
        # Refinement pass tightens orthonormality without materializing new structure elsewhere.
        T2, _m2, _rd2, diag2 = _gram_eigh_transform(
            xp, Z1.conj().T @ Z1, rank_tol=rank_tol
        )
        Z2 = xp.ascontiguousarray(Z1 @ T2)
        out = DenseBasis(xp, Z2, kind=self.kind)
        diag.update({
            "m_eff": out.m,
            "rank_dropped": int(self.m - out.m),
            "basis_kind": out.kind,
            "orth_refine_rank_dropped": int(diag2.get("rank_dropped", 0)),
        })
        return out, diag

    def columns(self, lo: int, hi: int) -> Any:
        return self.xp.ascontiguousarray(self.Z[:, int(lo):int(hi)])

    def to_dense(self) -> Any:
        return self.Z

    def memory_estimate(self) -> dict[str, Any]:
        bytes_per = int(self.Z.dtype.itemsize)
        gb = int(self.n) * int(self.m) * bytes_per / (1024.0**3)
        return {
            "basis_kind": self.kind,
            "estimated_basis_GB": gb,
            "basis_storage_dtype": str(self.Z.dtype),
        }


@dataclass
class CoordNystromBasis(DeflationBasis):
    xp: Any
    n: int
    indices: Any
    V: Any
    kind: str = "coord_nystrom"

    def __post_init__(self) -> None:
        self.indices = self.xp.ascontiguousarray(self.xp.asarray(self.indices, dtype=self.xp.int64).reshape(-1))
        V = self.xp.asarray(self.V)
        if V.ndim == 1:
            V = V.reshape(-1, 1)
        self.V = self.xp.ascontiguousarray(V)
        self.n = int(self.n)
        self.m = int(self.V.shape[1])
        if int(self.V.shape[0]) != int(self.indices.size):
            raise ValueError("CoordNystromBasis V rows must match indices.")

    def apply(self, coeff: Any, out: Optional[Any] = None) -> Any:
        C, was_1d = _as2d_col(self.xp, coeff)
        Ys = self.V @ C
        shape = (self.n,) if was_1d else (self.n, int(C.shape[1]))
        Y = self.xp.zeros(shape, dtype=Ys.dtype)
        if was_1d:
            Y[self.indices] = Ys.reshape(-1)
        else:
            Y[self.indices, :] = Ys
        if out is not None:
            self.xp.copyto(out, Y)
            return out
        return Y

    def apply_H(self, v: Any) -> Any:
        Vfine, was_1d = _as2d_col(self.xp, v)
        Vs = Vfine[self.indices, :]
        Y = self.V.conj().T @ Vs
        return Y.reshape(-1) if was_1d else Y

    def gram(self) -> Any:
        return self.V.conj().T @ self.V

    def orthonormalized(self, rank_tol: float = 1e-12) -> tuple[DeflationBasis, dict[str, Any]]:
        T, _m_eff, _rank_dropped, diag = _gram_eigh_transform(
            self.xp, self.gram(), rank_tol=rank_tol
        )
        V1 = self.V @ T
        T2, _m2, _rd2, diag2 = _gram_eigh_transform(
            self.xp, V1.conj().T @ V1, rank_tol=rank_tol
        )
        out = CoordNystromBasis(
            self.xp, self.n, self.indices, self.xp.ascontiguousarray(V1 @ T2), kind=self.kind
        )
        diag.update({
            "m_eff": out.m,
            "rank_dropped": int(self.m - out.m),
            "basis_kind": out.kind,
            "orth_refine_rank_dropped": int(diag2.get("rank_dropped", 0)),
        })
        return out, diag

    def memory_estimate(self) -> dict[str, Any]:
        gb = (int(self.indices.size) * int(self.indices.dtype.itemsize)
              + int(self.V.size) * int(self.V.dtype.itemsize)) / (1024.0**3)
        return {
            "basis_kind": self.kind,
            "estimated_basis_GB": gb,
            "support_size": int(self.indices.size),
            "basis_storage_dtype": str(self.V.dtype),
        }


@dataclass
class FreqTruncBasis(DeflationBasis):
    xp: Any
    Zc: Any
    mtot: int
    mtot_t: int
    dim: int
    kind: str = "freq_trunc"

    def __post_init__(self) -> None:
        Zc = self.xp.asarray(self.Zc)
        if Zc.ndim == 1:
            Zc = Zc.reshape(-1, 1)
        self.Zc = self.xp.ascontiguousarray(Zc)
        self.mtot = int(self.mtot)
        self.mtot_t = int(self.mtot_t)
        self.dim = int(self.dim)
        self.n = int(self.mtot) ** int(self.dim)
        self.n_tilde = int(self.mtot_t) ** int(self.dim)
        self.m = int(self.Zc.shape[1])
        if int(self.Zc.shape[0]) != self.n_tilde:
            raise ValueError("FreqTruncBasis Zc rows must match mtot_t**dim.")

    def _slices(self) -> tuple[slice, ...]:
        hm = (self.mtot - 1) // 2
        hm_t = (self.mtot_t - 1) // 2
        return tuple(slice(hm - hm_t, hm + hm_t + 1) for _ in range(self.dim))

    def _restrict(self, v: Any) -> Any:
        V, was_1d = _as2d_col(self.xp, v)
        Vnd = V.reshape((self.mtot,) * self.dim + (int(V.shape[1]),))
        Vc = Vnd[self._slices() + (slice(None),)].reshape(self.n_tilde, int(V.shape[1]))
        return Vc, was_1d

    def apply(self, coeff: Any, out: Optional[Any] = None) -> Any:
        C, was_1d = _as2d_col(self.xp, coeff)
        Yc = self.Zc @ C
        Ynd = self.xp.zeros((self.mtot,) * self.dim + (int(C.shape[1]),), dtype=Yc.dtype)
        Ynd[self._slices() + (slice(None),)] = Yc.reshape((self.mtot_t,) * self.dim + (int(C.shape[1]),))
        Y = Ynd.reshape(self.n, int(C.shape[1]))
        if was_1d:
            Y = Y.reshape(-1)
        if out is not None:
            self.xp.copyto(out, Y)
            return out
        return Y

    def apply_H(self, v: Any) -> Any:
        Vc, was_1d = self._restrict(v)
        Y = self.Zc.conj().T @ Vc
        return Y.reshape(-1) if was_1d else Y

    def gram(self) -> Any:
        return self.Zc.conj().T @ self.Zc

    def orthonormalized(self, rank_tol: float = 1e-12) -> tuple[DeflationBasis, dict[str, Any]]:
        T, _m_eff, _rank_dropped, diag = _gram_eigh_transform(
            self.xp, self.gram(), rank_tol=rank_tol
        )
        Z1 = self.Zc @ T
        T2, _m2, _rd2, diag2 = _gram_eigh_transform(
            self.xp, Z1.conj().T @ Z1, rank_tol=rank_tol
        )
        out = FreqTruncBasis(
            self.xp, self.xp.ascontiguousarray(Z1 @ T2), self.mtot, self.mtot_t, self.dim, kind=self.kind
        )
        diag.update({
            "m_eff": out.m,
            "rank_dropped": int(self.m - out.m),
            "basis_kind": out.kind,
            "orth_refine_rank_dropped": int(diag2.get("rank_dropped", 0)),
        })
        return out, diag

    def memory_estimate(self) -> dict[str, Any]:
        gb = int(self.Zc.size) * int(self.Zc.dtype.itemsize) / (1024.0**3)
        return {
            "basis_kind": self.kind,
            "estimated_basis_GB": gb,
            "n_tilde": int(self.n_tilde),
            "mtot_t": int(self.mtot_t),
            "basis_storage_dtype": str(self.Zc.dtype),
        }


@dataclass
class CanonicalFreqBasis(DeflationBasis):
    xp: Any
    n: int
    indices: Any
    kind: str = "freq_box"

    def __post_init__(self) -> None:
        self.indices = self.xp.ascontiguousarray(self.xp.asarray(self.indices, dtype=self.xp.int64).reshape(-1))
        self.n = int(self.n)
        self.m = int(self.indices.size)
        if self.m != int(self.xp.unique(self.indices).size):
            raise ValueError("CanonicalFreqBasis indices must be unique.")

    def apply(self, coeff: Any, out: Optional[Any] = None) -> Any:
        C, was_1d = _as2d_col(self.xp, coeff)
        shape = (self.n,) if was_1d else (self.n, int(C.shape[1]))
        Y = self.xp.zeros(shape, dtype=C.dtype)
        if was_1d:
            Y[self.indices] = C.reshape(-1)
        else:
            Y[self.indices, :] = C
        if out is not None:
            self.xp.copyto(out, Y)
            return out
        return Y

    def apply_H(self, v: Any) -> Any:
        V, was_1d = _as2d_col(self.xp, v)
        Y = V[self.indices, :]
        return Y.reshape(-1) if was_1d else Y

    def gram(self) -> Any:
        return self.xp.eye(self.m, dtype=self.xp.complex128)

    def orthonormalized(self, rank_tol: float = 1e-12) -> tuple[DeflationBasis, dict[str, Any]]:
        del rank_tol
        return self, {
            "m_eff": self.m,
            "rank_dropped": 0,
            "basis_kind": self.kind,
            "gram_eig_min": 1.0,
            "gram_eig_max": 1.0,
        }

    def columns(self, lo: int, hi: int) -> Any:
        lo = int(lo)
        hi = int(hi)
        idx = self.indices[lo:hi]
        b = int(hi - lo)
        cols = self.xp.zeros((self.n, b), dtype=self.xp.complex128)
        if b > 0:
            cols[idx, self.xp.arange(b)] = 1.0
        return self.xp.ascontiguousarray(cols)

    def to_dense(self) -> Any:
        return self.columns(0, self.m)

    def memory_estimate(self) -> dict[str, Any]:
        gb = int(self.indices.size) * int(self.indices.dtype.itemsize) / (1024.0**3)
        return {
            "basis_kind": self.kind,
            "estimated_basis_GB": gb,
            "support_size": int(self.indices.size),
            "basis_storage_dtype": str(self.indices.dtype),
        }


def as_deflation_basis(xp: Any, obj: Any) -> DeflationBasis:
    if isinstance(obj, DeflationBasis):
        return obj
    return DenseBasis(xp, obj)
