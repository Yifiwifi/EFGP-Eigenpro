from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..contexts import GPUOperatorContext
from ..v1_ops import apply_A_v1
from .box_eigenpro import (
    BoxEigenProPreconditionerData,
    apply_box_eigenpro_local,
)
from .preconditioner import (
    BoxToeplitzPreconditionerData,
    _apply_local_box_operator,
    _now_synced,
    _triangular_solve,
)


@dataclass
class DiagnosticCounter:
    n_A_matvec: int = 0
    n_ASS_matvec: int = 0
    n_precond: int = 0
    n_eig_matvec: int = 0
    n_power_iter: int = 0

    @property
    def n_matvec(self) -> int:
        return int(self.n_A_matvec + self.n_ASS_matvec)

    def as_dict(self) -> dict[str, int]:
        return {
            "diagnostic_n_matvec": int(self.n_matvec),
            "diagnostic_n_A_matvec": int(self.n_A_matvec),
            "diagnostic_n_ASS_matvec": int(self.n_ASS_matvec),
            "diagnostic_n_precond": int(self.n_precond),
            "diagnostic_n_eig_matvec": int(self.n_eig_matvec),
            "diagnostic_n_power_iter": int(self.n_power_iter),
        }


@dataclass(frozen=True)
class HermitianNormEstimate:
    value: float
    stabilized: bool
    iterations: int
    relative_change: float
    starts: int
    stabilized_starts: int
    estimate_min: float
    estimate_max: float

    def as_dict(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_norm_estimator": "multistart_power_on_squared_hermitian",
            f"{prefix}_norm_stabilized": bool(self.stabilized),
            f"{prefix}_norm_iterations": int(self.iterations),
            f"{prefix}_norm_relative_change": float(self.relative_change),
            f"{prefix}_norm_starts": int(self.starts),
            f"{prefix}_norm_stabilized_starts": int(self.stabilized_starts),
            f"{prefix}_norm_estimate_min": float(self.estimate_min),
            f"{prefix}_norm_estimate_max": float(self.estimate_max),
        }


def _xp_asnumpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "get"):
        return np.asarray(arr.get())
    return np.asarray(arr)


def _tail_arrays(precond_data: Any) -> tuple[Any, Any, Any]:
    return precond_data.tail_idx_gpu, precond_data.diag_A_gpu[precond_data.tail_idx_gpu], precond_data.active.rho


def _common_cheap_diagnostics(precond_data: Any, reg_lambda: float) -> dict[str, Any]:
    tail_idx_np = np.asarray(precond_data.active.tail_idx, dtype=np.int64).reshape(-1)
    rho = np.asarray(precond_data.active.rho, dtype=np.float64).reshape(-1)
    if tail_idx_np.size:
        rho_max_T = float(np.max(rho[tail_idx_np]))
        tail_energy = float(np.sum(rho[tail_idx_np]))
    else:
        rho_max_T = 0.0
        tail_energy = 0.0
    return {
        "S_kind": "expanded_box",
        "btab_active_size_raw": int(precond_data.active.active_idx.size),
        "btab_box_size": int(precond_data.active.box_idx.size),
        "rho_max_T": rho_max_T,
        "tail_energy": tail_energy,
        "sigma2_equiv_reg_lambda": float(reg_lambda),
    }


def _make_counted_A_apply(
    backend: Any,
    data_ctx: Any,
    reg_lambda: float,
    counter: DiagnosticCounter,
) -> Callable[[Any], Any]:
    op_ctx = GPUOperatorContext()

    def _apply(v: Any) -> Any:
        xp = backend.xp
        vv = xp.asarray(v, dtype=xp.complex128)
        cols = 1 if vv.ndim == 1 else int(vv.shape[1])
        counter.n_A_matvec += int(cols)
        out = xp.empty_like(vv)
        if vv.ndim == 1:
            apply_A_v1(backend, data_ctx, vv, float(reg_lambda), op_ctx, out=out)
        else:
            for j in range(cols):
                apply_A_v1(backend, data_ctx, vv[:, j], float(reg_lambda), op_ctx, out=out[:, j])
        return out

    return _apply


def _make_counted_ASS_apply(
    backend: Any,
    precond_data: Any,
    reg_lambda: float,
    counter: DiagnosticCounter,
) -> Callable[[Any], Any] | None:
    if getattr(precond_data, "local_gf_gpu", None) is None:
        return None

    def _apply(v: Any) -> Any:
        xp = backend.xp
        vv = xp.asarray(v, dtype=xp.complex128)
        was_1d = vv.ndim == 1
        if was_1d:
            vv = vv.reshape(-1, 1)
        out = xp.empty_like(vv)
        for j in range(int(vv.shape[1])):
            _apply_local_box_operator(
                backend,
                precond_data,
                float(reg_lambda),
                vv[:, j],
                out=out[:, j],
            )
        counter.n_ASS_matvec += int(vv.shape[1])
        return out[:, 0] if was_1d else out

    return _apply


def _estimate_hermitian_norm_power(
    backend: Any,
    matvec: Callable[[Any], Any],
    n: int,
    *,
    tol: float,
    maxiter: int,
    counter: DiagnosticCounter,
) -> HermitianNormEstimate:
    """Heuristically estimate ``||H||_2`` by multistart power iteration on ``H^2``.

    Using the Rayleigh quotient of ``H`` can cancel when the two spectral ends
    have similar magnitude and opposite signs.  Iterating with ``H^2`` instead
    targets the largest squared eigenvalue and uses ``||H x||`` as the norm
    estimate.  Multiple complex-Gaussian starts reduce the chance of missing a
    dominant eigendirection.  Stabilization only describes the observed iterate
    sequence; it is not a certificate that the global spectral end was found.
    """
    n = int(n)
    if not math.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("Hermitian norm tolerance must be finite and positive.")
    if int(maxiter) <= 0:
        raise ValueError("Hermitian norm maxiter must be positive.")
    if n <= 0:
        return HermitianNormEstimate(0.0, True, 0, 0.0, 0, 0, 0.0, 0.0)
    xp = backend.xp
    n_starts = 4
    min_iterations = min(5, int(maxiter))
    target_tol = max(float(tol), float(np.finfo(np.float64).eps))
    estimates: list[float] = []
    changes: list[float] = []
    stabilized: list[bool] = []
    iterations: list[int] = []
    for start_idx in range(n_starts):
        seed = 17 + 104729 * start_idx
        try:
            rng = xp.random.RandomState(int(seed))
            real = rng.standard_normal((n,)).astype(xp.float64)
            imag = rng.standard_normal((n,)).astype(xp.float64)
        except Exception:
            rng_np = np.random.default_rng(int(seed))
            real = xp.asarray(rng_np.standard_normal(n), dtype=xp.float64)
            imag = xp.asarray(rng_np.standard_normal(n), dtype=xp.float64)
        x = xp.asarray((real + 1j * imag) / math.sqrt(2.0), dtype=xp.complex128)
        norm = float(xp.linalg.norm(x))
        if not math.isfinite(norm) or norm == 0.0:
            estimates.append(0.0)
            changes.append(math.inf)
            stabilized.append(False)
            iterations.append(0)
            continue
        x = x / norm
        last = math.nan
        relative_change = math.inf
        stable_count = 0
        value = 0.0
        completed = 0
        is_stable = False
        for iteration in range(int(maxiter)):
            y = matvec(x)
            value = float(xp.linalg.norm(y))
            z = matvec(y)
            zn = float(xp.linalg.norm(z))
            counter.n_power_iter += 1
            completed = iteration + 1
            if not math.isfinite(value) or not math.isfinite(zn):
                value = math.nan
                break
            if value == 0.0 or zn == 0.0:
                value = 0.0
                relative_change = 0.0
                break
            if math.isfinite(last):
                relative_change = abs(value - last) / max(abs(value), 1e-300)
                if relative_change <= target_tol:
                    stable_count += 1
                else:
                    stable_count = 0
            x = z / zn
            last = value
            if completed >= min_iterations and stable_count >= 2:
                is_stable = True
                break
        estimates.append(float(value))
        changes.append(float(relative_change))
        stabilized.append(bool(is_stable))
        iterations.append(int(completed))

    finite_indices = [idx for idx, value in enumerate(estimates) if math.isfinite(value)]
    if not finite_indices:
        return HermitianNormEstimate(
            math.nan,
            False,
            int(sum(iterations)),
            math.inf,
            n_starts,
            int(sum(stabilized)),
            math.nan,
            math.nan,
        )
    winner = max(finite_indices, key=lambda idx: estimates[idx])
    finite_estimates = [estimates[idx] for idx in finite_indices]
    return HermitianNormEstimate(
        float(estimates[winner]),
        bool(all(stabilized)),
        int(sum(iterations)),
        float(changes[winner]),
        int(n_starts),
        int(sum(stabilized)),
        float(min(finite_estimates)),
        float(max(finite_estimates)),
    )


def _apply_A_ST(
    backend: Any,
    precond_data: Any,
    apply_A: Callable[[Any], Any],
    wT: Any,
) -> Any:
    xp = backend.xp
    x = xp.zeros((int(precond_data.diag_A_gpu.size),), dtype=xp.complex128)
    x[precond_data.tail_idx_gpu] = xp.asarray(wT, dtype=xp.complex128).reshape(-1)
    y = apply_A(x)
    return y[precond_data.box_idx_gpu]


def _apply_A_TS(
    backend: Any,
    precond_data: Any,
    apply_A: Callable[[Any], Any],
    xS: Any,
) -> Any:
    xp = backend.xp
    x = xp.zeros((int(precond_data.diag_A_gpu.size),), dtype=xp.complex128)
    x[precond_data.box_idx_gpu] = xp.asarray(xS, dtype=xp.complex128).reshape(-1)
    y = apply_A(x)
    return y[precond_data.tail_idx_gpu]


def _make_A_SS_inv_apply(
    backend: Any,
    precond_data: BoxToeplitzPreconditionerData,
    counter: DiagnosticCounter,
) -> Callable[[Any], Any] | None:
    if getattr(precond_data, "box_inverse_gpu", None) is not None:
        def _apply(rhs: Any) -> Any:
            counter.n_precond += 1
            return precond_data.box_inverse_gpu @ rhs

        return _apply
    if getattr(precond_data, "chol_factor_gpu", None) is not None:
        def _apply(rhs: Any) -> Any:
            counter.n_precond += 1
            y = _triangular_solve(backend, precond_data.chol_factor_gpu, rhs, lower=True)
            return _triangular_solve(backend, precond_data.chol_factor_gpu.conj().T, y, lower=False)

        return _apply
    return None


def _eig_residuals(
    backend: Any,
    precond_data: BoxEigenProPreconditionerData,
    ASS_apply: Callable[[Any], Any],
    counter: DiagnosticCounter,
) -> tuple[float, float]:
    xp = backend.xp
    U = precond_data.eig_U_gpu
    theta = xp.asarray(precond_data.eig_theta_top_gpu, dtype=xp.float64).reshape(-1)
    if U is None or int(U.shape[1]) == 0:
        return float("nan"), float("nan")
    AU = ASS_apply(U)
    counter.n_eig_matvec += int(U.shape[1])
    R = AU - U * theta.reshape(1, -1)
    nr = xp.linalg.norm(R, axis=0)
    nu = xp.maximum(xp.linalg.norm(U, axis=0), 1e-30)
    rel = nr / xp.maximum(xp.abs(theta) * nu, 1e-30)
    rel_np = np.asarray(_xp_asnumpy(rel), dtype=np.float64)
    return float(np.max(rel_np)), float(np.median(rel_np))


def _epsilon_T(
    backend: Any,
    precond_data: Any,
    apply_A: Callable[[Any], Any],
    counter: DiagnosticCounter,
    *,
    tol: float,
    maxiter: int,
) -> HermitianNormEstimate:
    xp = backend.xp
    tail_idx = precond_data.tail_idx_gpu
    Lambda_T = precond_data.diag_A_gpu[tail_idx]
    nT = int(tail_idx.size)
    if nT <= 0:
        return HermitianNormEstimate(0.0, True, 0, 0.0, 0, 0, 0.0, 0.0)
    inv_sqrt = 1.0 / xp.sqrt(Lambda_T)
    M = int(precond_data.diag_A_gpu.size)

    def _mv(zT: Any) -> Any:
        zT = xp.asarray(zT, dtype=xp.complex128).reshape(-1)
        wT = inv_sqrt * zT
        w = xp.zeros((M,), dtype=xp.complex128)
        w[tail_idx] = wT
        Aw = apply_A(w)
        yT = Aw[tail_idx] - Lambda_T * wT
        return inv_sqrt * yT

    return _estimate_hermitian_norm_power(
        backend,
        _mv,
        nT,
        tol=tol,
        maxiter=maxiter,
        counter=counter,
    )


def _eta_inv(
    backend: Any,
    precond_data: BoxToeplitzPreconditionerData,
    apply_A: Callable[[Any], Any],
    A_SS_inv_apply: Callable[[Any], Any],
    counter: DiagnosticCounter,
    *,
    tol: float,
    maxiter: int,
) -> tuple[float, float, HermitianNormEstimate]:
    xp = backend.xp
    Lambda_T = precond_data.diag_A_gpu[precond_data.tail_idx_gpu]
    nT = int(precond_data.tail_idx_gpu.size)
    if nT <= 0:
        estimate = HermitianNormEstimate(0.0, True, 0, 0.0, 0, 0, 0.0, 0.0)
        return 0.0, 0.0, estimate
    inv_sqrt = 1.0 / xp.sqrt(Lambda_T)

    def _mv(zT: Any) -> Any:
        zT = xp.asarray(zT, dtype=xp.complex128).reshape(-1)
        wT = inv_sqrt * zT
        yS = _apply_A_ST(backend, precond_data, apply_A, wT)
        xS = A_SS_inv_apply(yS)
        uT = _apply_A_TS(backend, precond_data, apply_A, xS)
        return inv_sqrt * uT

    estimate = _estimate_hermitian_norm_power(
        backend,
        _mv,
        nT,
        tol=tol,
        maxiter=maxiter,
        counter=counter,
    )
    eta_sq = float(estimate.value)
    return float(math.sqrt(max(eta_sq, 0.0))), eta_sq, estimate


def _eta_eig(
    backend: Any,
    precond_data: BoxEigenProPreconditionerData,
    apply_A: Callable[[Any], Any],
    counter: DiagnosticCounter,
    *,
    tol: float,
    maxiter: int,
) -> tuple[float, float, HermitianNormEstimate]:
    xp = backend.xp
    Lambda_T = precond_data.diag_A_gpu[precond_data.tail_idx_gpu]
    nT = int(precond_data.tail_idx_gpu.size)
    if nT <= 0:
        estimate = HermitianNormEstimate(0.0, True, 0, 0.0, 0, 0, 0.0, 0.0)
        return 0.0, 0.0, estimate
    inv_sqrt = 1.0 / xp.sqrt(Lambda_T)

    def _mv(zT: Any) -> Any:
        zT = xp.asarray(zT, dtype=xp.complex128).reshape(-1)
        wT = inv_sqrt * zT
        yS = _apply_A_ST(backend, precond_data, apply_A, wT)
        counter.n_precond += 1
        xS = apply_box_eigenpro_local(backend, precond_data, yS)
        uT = _apply_A_TS(backend, precond_data, apply_A, xS)
        return inv_sqrt * uT

    estimate = _estimate_hermitian_norm_power(
        backend,
        _mv,
        nT,
        tol=tol,
        maxiter=maxiter,
        counter=counter,
    )
    eta_sq = float(estimate.value)
    return float(math.sqrt(max(eta_sq, 0.0))), eta_sq, estimate


def run_btab_post_diagnostics(
    backend: Any,
    data_ctx: Any,
    reg_lambda: float,
    precond_data: Any,
    *,
    route: str,
    mode: str,
    tol: float = 1e-2,
    power_iter: int = 30,
) -> dict[str, Any]:
    mode_key = str(mode or "cheap").strip().lower()
    if mode_key not in ("none", "cheap", "full"):
        raise ValueError("btab_diagnostic_mode must be one of 'none', 'cheap', or 'full'.")
    counter = DiagnosticCounter()
    out: dict[str, Any] = {
        "btab_diagnostic_mode": mode_key,
        "diagnostic_mode": mode_key,
    }
    t0 = _now_synced(backend)
    if mode_key == "none":
        out.update(counter.as_dict())
        t1 = _now_synced(backend)
        out["time_post_diagnostics"] = float(t1 - t0)
        return out

    out.update(_common_cheap_diagnostics(precond_data, float(reg_lambda)))

    ASS_apply = _make_counted_ASS_apply(backend, precond_data, float(reg_lambda), counter)
    if route == "boxeig" and ASS_apply is not None:
        eig_max, eig_med = _eig_residuals(backend, precond_data, ASS_apply, counter)
        theta_q1 = float(getattr(precond_data, "eig_theta_q1", float("nan")))
        theta_ratio = theta_q1 / max(float(reg_lambda), 1e-300)
        out.update(
            {
                "btab_eig_theta_q1": theta_q1,
                "btab_eig_theta_q1_over_sigma2": float(theta_ratio),
                "theta_q1_over_sigma2": float(theta_ratio),
                "btab_eig_eig_residual_max": float(eig_max),
                "btab_eig_eig_residual_median": float(eig_med),
                "eig_residual_max": float(eig_max),
                "eig_residual_median": float(eig_med),
            }
        )

    if mode_key == "full":
        apply_A = _make_counted_A_apply(backend, data_ctx, float(reg_lambda), counter)
        epsilon_estimate = _epsilon_T(
            backend,
            precond_data,
            apply_A,
            counter,
            tol=float(tol),
            maxiter=int(power_iter),
        )
        out["epsilon_T"] = float(epsilon_estimate.value)
        out.update(epsilon_estimate.as_dict("epsilon_T"))
        if route == "inverse":
            inv_apply = _make_A_SS_inv_apply(backend, precond_data, counter)
            if inv_apply is None:
                out["eta_inv"] = float("nan")
                out["eta_inv_sq"] = float("nan")
                out["eta_inv_status"] = "skipped_no_exact_inverse"
            else:
                eta, eta_sq, eta_estimate = _eta_inv(
                    backend,
                    precond_data,
                    apply_A,
                    inv_apply,
                    counter,
                    tol=float(tol),
                    maxiter=int(power_iter),
                )
                out["eta_inv"] = float(eta)
                out["eta_inv_sq"] = float(eta_sq)
                out["eta_inv_status"] = "stabilized" if eta_estimate.stabilized else "unstabilized"
                out.update(eta_estimate.as_dict("eta_inv_sq"))
        elif route == "boxeig":
            eta, eta_sq, eta_estimate = _eta_eig(
                backend,
                precond_data,
                apply_A,
                counter,
                tol=float(tol),
                maxiter=int(power_iter),
            )
            out["eta_eig"] = float(eta)
            out["eta_eig_sq"] = float(eta_sq)
            out["eta_eig_status"] = "stabilized" if eta_estimate.stabilized else "unstabilized"
            out.update(eta_estimate.as_dict("eta_eig_sq"))

    out.update(counter.as_dict())
    t1 = _now_synced(backend)
    out["time_post_diagnostics"] = float(t1 - t0)
    return out
