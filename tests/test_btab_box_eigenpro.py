from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block import (
    box_eigenpro as box_eigenpro_module,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block import (
    preconditioner as preconditioner_module,
)

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.box_eigenpro import (
    apply_box_eigenpro_local,
    build_box_eigenpro_preconditioner,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.config import BTABConfig
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.preconditioner import (
    _apply_local_box_operator,
    apply_box_toeplitz_preconditioner,
    build_box_toeplitz_preconditioner,
)
from efgp_eigenpro_py.gpu.contexts import GPUOperatorContext
from efgp_eigenpro_py.gpu.v1_ops import apply_A_v1
from efgp_eigenpro_py.gpu.v3_eigenspace import _toeplitz_submatrix_gpu


def _fake_problem(mtot: int = 7):
    backend = SimpleNamespace(
        xp=np,
        fft=np.fft,
        linalg=np.linalg,
        nufft_name="none",
        device_name="cpu",
        has_nufft=False,
    )
    lags = np.arange(2 * mtot - 1) - (mtot - 1)
    phases = np.array([0.2, 1.1, 2.4, 4.0], dtype=np.float64)
    xtxcol = np.sum(np.exp(-1j * lags[:, None] * phases[None, :]), axis=1).astype(np.complex128)
    weights = np.array([0.1, 0.2, 1.0, 2.0, 1.0, 0.2, 0.1], dtype=np.float64)
    data_ctx = SimpleNamespace(
        xtxcol_gpu=xtxcol,
        gf_gpu=np.fft.fftn(xtxcol),
        weights_gpu_flat=weights,
        weights_np_flat=weights.copy(),
        rhs_gpu=np.ones(mtot, dtype=np.complex128),
        meta={"mtot": mtot, "dim": 1},
    )
    cfg = BTABConfig(
        active_mode="topk",
        active_topk=3,
        box_budget=mtot,
        eig_q=1,
        eig_tol=1e-10,
        diagnostic_mode="cheap",
    )
    return backend, data_ctx, cfg


def test_strict_gpu_eig_disables_scipy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from efgp_eigenpro_py.gpu import cupy_eigenspace_methods

    def fail_gpu_eigensolver(*args, **kwargs):
        raise RuntimeError("synthetic GPU eigensolver failure")

    monkeypatch.setattr(
        cupy_eigenspace_methods,
        "cupy_eigsh",
        fail_gpu_eigensolver,
    )

    backend = SimpleNamespace(xp=SimpleNamespace(__name__="cupy"))
    preconditioner = SimpleNamespace(box_shape=(10,))
    cfg = SimpleNamespace(
        eig_tol=1e-3,
        eig_maxiter=10,
        eig_ncv=None,
        strict_gpu_eig=True,
    )

    with pytest.raises(RuntimeError, match="CPU/SciPy fallback is disabled"):
        box_eigenpro_module._compute_local_eigenpairs(
            backend,
            preconditioner,
            reg_lambda=0.1,
            q=2,
            cfg=cfg,
            counter={},
        )


def test_toeplitz_submatrix_matches_apply_A_v1_full_grid():
    backend, data_ctx, _ = _fake_problem()
    reg_lambda = 0.1
    mtot = int(data_ctx.meta["mtot"])
    dense = _toeplitz_submatrix_gpu(
        np,
        data_ctx.xtxcol_gpu,
        data_ctx.weights_gpu_flat,
        np.arange(mtot, dtype=np.int64),
        mtot=mtot,
        dim=data_ctx.meta["dim"],
    )
    dense = dense + reg_lambda * np.eye(mtot, dtype=np.complex128)

    cols = []
    for j in range(mtot):
        ej = np.zeros(mtot, dtype=np.complex128)
        ej[j] = 1.0
        cols.append(apply_A_v1(backend, data_ctx, ej, reg_lambda, GPUOperatorContext()).copy())
    from_matvec = np.column_stack(cols)

    np.testing.assert_allclose(dense, from_matvec, rtol=1e-10, atol=1e-10)


def test_local_box_matvec_matches_dense_toeplitz_submatrix():
    backend, data_ctx, cfg = _fake_problem()
    reg_lambda = 0.1
    pre = build_box_eigenpro_preconditioner(
        backend,
        data_ctx,
        reg_lambda,
        cfg,
        q=1,
        profile_apply_components=False,
    )
    dense = _toeplitz_submatrix_gpu(
        np,
        data_ctx.xtxcol_gpu,
        data_ctx.weights_gpu_flat,
        pre.active.box_idx,
        mtot=data_ctx.meta["mtot"],
        dim=data_ctx.meta["dim"],
    )
    dense = dense + reg_lambda * np.eye(dense.shape[0], dtype=np.complex128)
    v = np.array([1.0, -0.5, 0.25], dtype=np.complex128)
    got = _apply_local_box_operator(backend, pre, reg_lambda, v)
    np.testing.assert_allclose(got, dense @ v, rtol=1e-10, atol=1e-10)


def test_full_box_inverse_preconditions_apply_A_v1_to_identity():
    backend, data_ctx, _ = _fake_problem()
    reg_lambda = 0.1
    mtot = int(data_ctx.meta["mtot"])
    cfg = BTABConfig(
        active_mode="topk",
        active_topk=mtot,
        box_budget=mtot,
        solve_mode="exact",
        exact_apply_mode="inverse",
        diagnostic_mode="none",
    )
    pre = build_box_toeplitz_preconditioner(
        backend,
        data_ctx,
        reg_lambda,
        cfg,
        profile_apply_components=False,
    )
    assert int(pre.active.box_idx.size) == mtot
    assert int(pre.active.tail_idx.size) == 0

    v = np.array([0.3, -0.2, 0.8, 1.1, -0.7, 0.4, 0.05], dtype=np.complex128)
    Av = apply_A_v1(backend, data_ctx, v, reg_lambda, GPUOperatorContext())
    got = apply_box_toeplitz_preconditioner(backend, pre, Av)
    np.testing.assert_allclose(got, v, rtol=1e-10, atol=1e-10)


def test_box_eigenpro_apply_matches_dense_formula():
    backend, data_ctx, cfg = _fake_problem()
    pre = build_box_eigenpro_preconditioner(
        backend,
        data_ctx,
        0.1,
        cfg,
        q=1,
        profile_apply_components=False,
    )
    U = pre.eig_U_gpu
    coeff = pre.eig_coeff_gpu
    dense_P = pre.eig_alpha * np.eye(U.shape[0], dtype=np.complex128) + U @ np.diag(coeff) @ U.conj().T
    v = np.array([0.25, -1.0, 0.75], dtype=np.complex128)
    got = apply_box_eigenpro_local(backend, pre, v)
    np.testing.assert_allclose(got, dense_P @ v, rtol=1e-10, atol=1e-10)


def test_box_eigenpro_spectral_action_uses_theta_q_plus_one():
    backend, data_ctx, cfg = _fake_problem()
    reg_lambda = 0.1
    pre = build_box_eigenpro_preconditioner(
        backend,
        data_ctx,
        reg_lambda,
        cfg,
        q=1,
        profile_apply_components=False,
    )
    A = _toeplitz_submatrix_gpu(
        np,
        data_ctx.xtxcol_gpu,
        data_ctx.weights_gpu_flat,
        pre.active.box_idx,
        mtot=data_ctx.meta["mtot"],
        dim=data_ctx.meta["dim"],
    )
    A = A + reg_lambda * np.eye(A.shape[0], dtype=np.complex128)
    theta = np.linalg.eigvalsh(A)[::-1]
    U = pre.eig_U_gpu
    P = pre.eig_alpha * np.eye(A.shape[0], dtype=np.complex128) + U @ np.diag(pre.eig_coeff_gpu) @ U.conj().T
    got = np.sort(np.real(np.linalg.eigvals(P @ A)))[::-1]
    expected = np.sort(np.array([1.0, theta[1] / theta[1], theta[2] / theta[1]], dtype=np.float64))[::-1]
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("builder", "module", "extra"),
    [
        (build_box_toeplitz_preconditioner, preconditioner_module, {}),
        (build_box_eigenpro_preconditioner, box_eigenpro_module, {"q": 1}),
    ],
)
def test_precomputed_active_set_skips_second_score_box_selection(
    monkeypatch: pytest.MonkeyPatch,
    builder,
    module,
    extra,
) -> None:
    backend, data_ctx, cfg = _fake_problem()
    first = builder(
        backend,
        data_ctx,
        0.1,
        cfg,
        profile_apply_components=False,
        **extra,
    )

    def fail_if_reselected(*args, **kwargs):
        raise AssertionError("score-box selection ran twice")

    monkeypatch.setattr(module, "build_box_active_set", fail_if_reselected)
    second = builder(
        backend,
        data_ctx,
        0.1,
        cfg,
        profile_apply_components=False,
        precomputed_active_set=first.active,
        **extra,
    )

    assert second.active is first.active
    assert second.diagnostics["time_active_set"] == 0.0

    _, different_data_ctx, _ = _fake_problem()
    different_data_ctx.weights_np_flat = different_data_ctx.weights_np_flat.copy()
    different_data_ctx.weights_np_flat[0] *= 1.5
    different_data_ctx.weights_gpu_flat = different_data_ctx.weights_np_flat.copy()
    with pytest.raises(ValueError, match="current system"):
        builder(
            backend,
            different_data_ctx,
            0.1,
            cfg,
            profile_apply_components=False,
            precomputed_active_set=first.active,
            **extra,
        )
