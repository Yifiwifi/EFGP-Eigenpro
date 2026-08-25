from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.randomized_pivoted_cholesky import (
    apply_randomized_pivoted_cholesky_preconditioner,
    build_randomized_pivoted_cholesky_preconditioner,
    make_weighted_toeplitz_column_accessor,
)


def _backend():
    return SimpleNamespace(xp=np)


def test_full_rank_complex_rpcholesky_recovers_fixed_system_inverse() -> None:
    rng = np.random.default_rng(8)
    Z = rng.standard_normal((7, 7)) + 1j * rng.standard_normal((7, 7))
    H = Z @ Z.conj().T
    reg = 0.3
    pre = build_randomized_pivoted_cholesky_preconditioner(
        _backend(),
        lambda j: H[:, j],
        np.real(np.diag(H)),
        rank=7,
        reg_lambda=reg,
        seed=4,
        dtype=np.complex128,
    )
    v = rng.standard_normal(7) + 1j * rng.standard_normal(7)
    got = apply_randomized_pivoted_cholesky_preconditioner(_backend(), pre, v)
    expected = np.linalg.solve(H + reg * np.eye(7), v)
    np.testing.assert_allclose(got, expected, rtol=2e-11, atol=2e-11)
    assert pre.effective_rank == 7
    assert pre.diagnostics["block_size"] == 1


def test_rpcholesky_is_reproducible_positive_and_scale_aware() -> None:
    scale = 1e-20
    diagonal = scale * np.geomspace(10.0, 0.2, 8)
    H = np.diag(diagonal)
    kwargs = dict(
        psd_diagonal=diagonal,
        rank=4,
        reg_lambda=1e-21,
        seed=12,
        dtype=np.complex128,
    )
    p1 = build_randomized_pivoted_cholesky_preconditioner(
        _backend(), lambda j: H[:, j], **kwargs
    )
    p2 = build_randomized_pivoted_cholesky_preconditioner(
        _backend(), lambda j: H[:, j], **kwargs
    )
    assert p1.pivots == p2.pivots
    np.testing.assert_allclose(p1.L, p2.L)
    v = np.arange(1.0, 9.0).astype(np.complex128)
    Pv = apply_randomized_pivoted_cholesky_preconditioner(_backend(), p1, v)
    assert float(np.real(np.vdot(v, Pv))) > 0.0


def test_low_rank_rpcholesky_stops_without_nan() -> None:
    u = np.array([1.0, 2.0, -0.5, 3.0, 0.25])
    H = np.outer(u, u)
    pre = build_randomized_pivoted_cholesky_preconditioner(
        _backend(),
        lambda j: H[:, j],
        np.diag(H),
        rank=5,
        reg_lambda=0.2,
        seed=2,
        dtype=np.complex128,
    )
    assert pre.effective_rank == 1
    assert np.all(np.isfinite(pre.L))
    assert np.all(np.isfinite(pre.middle_inverse))


def test_rpcholesky_rejects_nonhermitian_or_negative_residual_pivot() -> None:
    diagonal = np.ones(4)

    def nonhermitian(j: int) -> np.ndarray:
        column = np.eye(4, dtype=np.complex128)[:, j]
        column[j] += 1j
        return column

    with pytest.raises(ValueError, match="Hermitian"):
        build_randomized_pivoted_cholesky_preconditioner(
            _backend(), nonhermitian, diagonal, rank=2, reg_lambda=0.1
        )

    with pytest.raises(ValueError, match="negative"):
        build_randomized_pivoted_cholesky_preconditioner(
            _backend(), lambda j: -np.eye(4)[:, j], diagonal, rank=2, reg_lambda=0.1
        )

    with pytest.raises(ValueError, match="psd_diagonal"):
        build_randomized_pivoted_cholesky_preconditioner(
            _backend(),
            lambda j: 2.0 * np.eye(4)[:, j],
            diagonal,
            rank=2,
            reg_lambda=0.1,
        )


def test_weighted_toeplitz_accessor_matches_dense_complex_hermitian_matrix() -> None:
    mtot = 3
    dim = 2
    shape = (mtot,) * dim
    lag_shape = (2 * mtot - 1,) * dim
    center = mtot - 1
    angles = np.array([[0.2, 0.8], [-0.4, 0.3], [0.9, -0.5]])
    generator = np.empty(lag_shape, dtype=np.complex128)
    for index in np.ndindex(lag_shape):
        lag = np.asarray(index) - center
        generator[index] = np.sum(np.exp(1j * (angles @ lag)))
    weights = np.linspace(0.2, 1.0, mtot**dim)
    access = make_weighted_toeplitz_column_accessor(
        np,
        generator,
        weights,
        mtot=mtot,
        dim=dim,
        dtype=np.complex128,
    )
    multi = np.asarray(np.unravel_index(np.arange(mtot**dim), shape)).T
    dense = np.empty((mtot**dim, mtot**dim), dtype=np.complex128)
    for i in range(mtot**dim):
        for j in range(mtot**dim):
            lag = tuple(multi[i] - multi[j] + center)
            dense[i, j] = weights[i] * generator[lag] * weights[j]
    for j in range(mtot**dim):
        np.testing.assert_allclose(access(j), dense[:, j], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(dense, dense.conj().T, rtol=1e-13, atol=1e-13)
