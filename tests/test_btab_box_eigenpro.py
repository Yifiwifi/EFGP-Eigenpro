from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.box_eigenpro import (
    apply_box_eigenpro_local,
    build_box_eigenpro_preconditioner,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.config import BTABConfig
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.preconditioner import _apply_local_box_operator
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
    xtxcol = np.exp(-0.35 * np.abs(lags)).astype(np.complex128)
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
