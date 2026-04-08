from typing import Any

import numpy as np

from .discretization import generate_multi_index


def _require_finufft() -> Any:
    try:
        import finufft  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("finufft is required for NUFFT ops.") from exc
    return finufft


def nufft1d1(x: np.ndarray, c: np.ndarray, ms: int, eps: float, isign: int):
    """
    Type-1 NUFFT in 1D. x should be in [-pi, pi].
    """
    finufft = _require_finufft()
    x = np.ascontiguousarray(x, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.complex128)
    return finufft.nufft1d1(x, c, ms, eps=eps, isign=isign)


def nufft1d2(x: np.ndarray, f: np.ndarray, eps: float, isign: int):
    """
    Type-2 NUFFT in 1D. x should be in [-pi, pi].
    """
    finufft = _require_finufft()
    x = np.ascontiguousarray(x, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.complex128)
    return finufft.nufft1d2(x, f, eps=eps, isign=isign)


def nufft2d1(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    ms: int,
    mt: int,
    eps: float,
    isign: int,
):
    finufft = _require_finufft()
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.complex128)
    return finufft.nufft2d1(x, y, c, n_modes=(ms, mt), eps=eps, isign=isign)


def nufft2d2(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    eps: float,
    isign: int,
):
    finufft = _require_finufft()
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.complex128)
    return finufft.nufft2d2(x, y, f, eps=eps, isign=isign)


def nufft3d1(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    c: np.ndarray,
    ms: int,
    mt: int,
    mu: int,
    eps: float,
    isign: int,
):
    finufft = _require_finufft()
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    z = np.ascontiguousarray(z, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.complex128)
    return finufft.nufft3d1(x, y, z, c, n_modes=(ms, mt, mu), eps=eps, isign=isign)


def nufft3d2(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    f: np.ndarray,
    eps: float,
    isign: int,
):
    finufft = _require_finufft()
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    z = np.ascontiguousarray(z, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.complex128)
    return finufft.nufft3d2(x, y, z, f, eps=eps, isign=isign)


def _parse_n_modes(n_modes: int | tuple[int, ...] | list[int], dim: int) -> tuple[int, ...]:
    if n_modes is None:
        raise ValueError("n_modes must be provided.")
    if isinstance(n_modes, int):
        return (int(n_modes),) * dim
    if isinstance(n_modes, (tuple, list)) and len(n_modes) == dim:
        return tuple(int(n) for n in n_modes)
    raise ValueError("n_modes must be int or tuple/list with length dim.")


def _multi_index_from_n_modes(n_modes: tuple[int, ...]) -> np.ndarray:
    m_list = [(m - 1) // 2 for m in n_modes]
    if all(m == m_list[0] for m in m_list):
        return generate_multi_index(m_list[0], len(n_modes))
    grid = [np.arange(-m, m + 1) for m in m_list]
    mesh = np.meshgrid(*grid, indexing="ij")
    return np.stack([g.reshape(-1) for g in mesh], axis=1)


def nufftnd1(
    x: np.ndarray,
    c: np.ndarray,
    n_modes: int | tuple[int, ...] | list[int],
    eps: float,
    isign: int,
):
    """
    ND Type-1 NUFFT. For dim<=3 dispatches to FINUFFT. For dim>3 uses direct sum.
    x should have shape (n, dim) in [-pi, pi].
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must have shape (n, dim).")
    dim = x.shape[1]
    c = np.ascontiguousarray(c, dtype=np.complex128)
    n_modes_tuple = _parse_n_modes(n_modes, dim)

    if dim == 1:
        return nufft1d1(x[:, 0], c, n_modes_tuple[0], eps, isign)
    if dim == 2:
        return nufft2d1(x[:, 0], x[:, 1], c, n_modes_tuple[0], n_modes_tuple[1], eps, isign)
    if dim == 3:
        return nufft3d1(
            x[:, 0], x[:, 1], x[:, 2], c, n_modes_tuple[0], n_modes_tuple[1], n_modes_tuple[2], eps, isign
        )

    J = _multi_index_from_n_modes(n_modes_tuple)
    phase = np.exp(1j * isign * (x @ J.T))
    return phase.T @ c


def nufftnd2(
    x: np.ndarray,
    f: np.ndarray,
    eps: float,
    isign: int,
    n_modes: int | tuple[int, ...] | list[int] | None = None,
):
    """
    ND Type-2 NUFFT. For dim<=3 dispatches to FINUFFT. For dim>3 uses direct sum.
    x should have shape (n, dim) in [-pi, pi].
    If f is flat, n_modes must be provided for dim>1.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must have shape (n, dim).")
    dim = x.shape[1]
    f = np.asarray(f)

    if dim == 1:
        f1 = np.ascontiguousarray(f.reshape(-1), dtype=np.complex128)
        return nufft1d2(x[:, 0], f1, eps, isign)
    if dim == 2:
        n_modes_tuple = _parse_n_modes(n_modes or f.shape, dim)
        f2 = np.ascontiguousarray(f.reshape(n_modes_tuple), dtype=np.complex128)
        return nufft2d2(x[:, 0], x[:, 1], f2, eps, isign)
    if dim == 3:
        n_modes_tuple = _parse_n_modes(n_modes or f.shape, dim)
        f3 = np.ascontiguousarray(f.reshape(n_modes_tuple), dtype=np.complex128)
        return nufft3d2(x[:, 0], x[:, 1], x[:, 2], f3, eps, isign)

    if f.ndim > 1:
        n_modes_tuple = _parse_n_modes(f.shape, dim)
        f_flat = np.asarray(f.reshape(-1), dtype=np.complex128)
    else:
        n_modes_tuple = _parse_n_modes(n_modes, dim)
        f_flat = np.ascontiguousarray(f, dtype=np.complex128)

    J = _multi_index_from_n_modes(n_modes_tuple)
    phase = np.exp(1j * isign * (x @ J.T))
    return phase @ f_flat
