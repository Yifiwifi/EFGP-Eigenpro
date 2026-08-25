from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.randomized_nystrom import (
    apply_randomized_nystrom_preconditioner,
    build_randomized_nystrom_preconditioner,
)


def _backend():
    return SimpleNamespace(xp=np)


def test_full_rank_randomized_nystrom_recovers_inverse_up_to_roundoff() -> None:
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    eigenvalues = np.geomspace(50.0, 0.2, 8)
    K = Q @ np.diag(eigenvalues) @ Q.T
    reg = 0.3

    pre = build_randomized_nystrom_preconditioner(
        _backend(),
        lambda V: K @ V,
        size=8,
        rank=8,
        reg_lambda=reg,
        seed=9,
        dtype=np.float64,
    )
    v = rng.standard_normal(8)
    got = apply_randomized_nystrom_preconditioner(_backend(), pre, v)
    expected = np.linalg.solve(K + reg * np.eye(8), v)

    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_randomized_nystrom_is_reproducible_and_positive() -> None:
    diagonal = np.geomspace(100.0, 0.01, 20)
    K = np.diag(diagonal)
    kwargs = dict(size=20, rank=5, reg_lambda=0.1, seed=12, dtype=np.float64)
    p1 = build_randomized_nystrom_preconditioner(_backend(), lambda V: K @ V, **kwargs)
    p2 = build_randomized_nystrom_preconditioner(_backend(), lambda V: K @ V, **kwargs)

    np.testing.assert_allclose(p1.U @ p1.U.T, p2.U @ p2.U.T, rtol=1e-12, atol=1e-12)
    v = np.arange(1.0, 21.0)
    Pv = apply_randomized_nystrom_preconditioner(_backend(), p1, v)
    assert float(v @ Pv) > 0.0
    assert p1.diagnostics["psd_block_matvec_columns"] == 5


def test_complex_hermitian_full_rank_case() -> None:
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    K = Z @ Z.conj().T
    reg = 0.2
    pre = build_randomized_nystrom_preconditioner(
        _backend(),
        lambda V: K @ V,
        size=6,
        rank=6,
        reg_lambda=reg,
        seed=2,
        dtype=np.complex128,
    )
    v = rng.standard_normal(6) + 1j * rng.standard_normal(6)
    got = apply_randomized_nystrom_preconditioner(_backend(), pre, v)
    expected = np.linalg.solve(K + reg * np.eye(6), v)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)
