from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.diagnostics import (
    DiagnosticCounter,
    _estimate_hermitian_norm_power,
)


def test_squared_power_estimates_indefinite_hermitian_norm() -> None:
    backend = SimpleNamespace(xp=np)
    matrix = np.diag(np.array([-3.0, 3.0, 0.5], dtype=np.complex128))
    calls = 0

    def matvec(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return matrix @ x

    counter = DiagnosticCounter()
    estimate = _estimate_hermitian_norm_power(
        backend,
        matvec,
        3,
        tol=1e-10,
        maxiter=100,
        counter=counter,
    )

    assert estimate.stabilized
    assert estimate.value == pytest.approx(3.0, rel=1e-10)
    assert estimate.iterations == counter.n_power_iter
    assert calls == 2 * estimate.iterations
    assert estimate.starts == 4


def test_squared_power_reports_zero_operator_without_certifying_it() -> None:
    backend = SimpleNamespace(xp=np)
    counter = DiagnosticCounter()
    estimate = _estimate_hermitian_norm_power(
        backend,
        lambda x: np.zeros_like(x),
        4,
        tol=1e-8,
        maxiter=10,
        counter=counter,
    )

    assert not estimate.stabilized
    assert estimate.value == 0.0
    assert estimate.relative_change == 0.0


def test_squared_power_handles_complex_nondiagonal_hermitian() -> None:
    backend = SimpleNamespace(xp=np)
    rng = np.random.default_rng(8)
    q, _ = np.linalg.qr(
        rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    )
    matrix = q @ np.diag([-7.0, 2.0, 0.25, 0.1]) @ q.conj().T
    estimate = _estimate_hermitian_norm_power(
        backend,
        lambda x: matrix @ x,
        4,
        tol=1e-10,
        maxiter=100,
        counter=DiagnosticCounter(),
    )

    assert estimate.stabilized
    assert estimate.value == pytest.approx(7.0, rel=1e-9)


def test_squared_power_marks_exhausted_iteration_budget_unstabilized() -> None:
    backend = SimpleNamespace(xp=np)
    matrix = np.diag([10.0, 1.0]).astype(np.complex128)
    estimate = _estimate_hermitian_norm_power(
        backend,
        lambda x: matrix @ x,
        2,
        tol=1e-12,
        maxiter=1,
        counter=DiagnosticCounter(),
    )

    assert not estimate.stabilized


def test_multistart_status_requires_every_start_to_stabilize() -> None:
    backend = SimpleNamespace(xp=np)
    matrix = np.diag([10.0, 9.0, 1.0, 0.1]).astype(np.complex128)
    estimate = _estimate_hermitian_norm_power(
        backend,
        lambda x: matrix @ x,
        4,
        tol=1e-2,
        maxiter=6,
        counter=DiagnosticCounter(),
    )

    assert estimate.stabilized == (estimate.stabilized_starts == estimate.starts)


@pytest.mark.parametrize("tol,maxiter", [(0.0, 10), (np.nan, 10), (1e-3, 0)])
def test_squared_power_rejects_invalid_controls(tol: float, maxiter: int) -> None:
    with pytest.raises(ValueError):
        _estimate_hermitian_norm_power(
            SimpleNamespace(xp=np),
            lambda x: x,
            2,
            tol=tol,
            maxiter=maxiter,
            counter=DiagnosticCounter(),
        )
