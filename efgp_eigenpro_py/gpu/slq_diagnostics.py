from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np


def _sync_device(xp: Any) -> None:
    cuda = getattr(xp, "cuda", None)
    if cuda is not None:
        cuda.Stream.null.synchronize()


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    qq = float(np.clip(q, 0.0, 1.0))
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    total = float(cdf[-1])
    if total <= 0.0 or not math.isfinite(total):
        return float("nan")
    target = qq * total
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = max(0, min(idx, v.size - 1))
    return float(v[idx])


def _validate_prefix_m(k_max: int, m: Optional[int]) -> int:
    if m is None:
        return int(k_max)
    mm = int(m)
    if mm < 1 or mm > int(k_max):
        raise ValueError(f"m must satisfy 1 <= m <= {k_max}, got {mm}")
    return mm


def _build_tridiag(alpha: np.ndarray, beta: np.ndarray, m: int) -> np.ndarray:
    T = np.diag(alpha[:m].astype(np.float64, copy=False))
    if m > 1:
        b = beta[: m - 1].astype(np.float64, copy=False)
        T[:-1, 1:] += np.diag(b)
        T[1:, :-1] += np.diag(b)
    return T


def _probe_nodes_weights(alpha: np.ndarray, beta: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    T = _build_tridiag(alpha, beta, m)
    vals, vecs = np.linalg.eigh(T)
    w = np.abs(vecs[0, :]) ** 2
    return vals.astype(np.float64, copy=False), w.astype(np.float64, copy=False)


def _sample_real_rademacher(xp: Any, n: int, rng: Any | None) -> Any:
    if rng is not None:
        u = rng.randint(0, 2, size=(n,))
    else:
        u = xp.random.randint(0, 2, size=(n,))
    return (2 * u - 1).astype(xp.float64)


def _sample_complex_rademacher(xp: Any, n: int, rng: Any | None) -> Any:
    if rng is not None:
        idx = rng.randint(0, 4, size=(n,))
    else:
        idx = xp.random.randint(0, 4, size=(n,))
    out = xp.empty((n,), dtype=xp.complex128)
    out[idx == 0] = 1.0 + 0.0j
    out[idx == 1] = -1.0 + 0.0j
    out[idx == 2] = 0.0 + 1.0j
    out[idx == 3] = 0.0 - 1.0j
    return out


def _local_reorthogonalize(
    backend: Any,
    z: Any,
    basis_window: Sequence[Any],
    *,
    passes: int = 2,
) -> None:
    """
    Partial reorthogonalization via local modified Gram-Schmidt.
    """
    if not basis_window:
        return
    xp = backend.xp
    p = max(int(passes), 1)
    for _ in range(p):
        for qv in basis_window:
            coeff = backend.linalg.vdot(qv, z)
            z -= coeff * qv
    _ = xp  # keep style parity with backend-driven ops


@dataclass
class SLQLanczosConfig:
    """
    Single-run SLQ Lanczos diagnostic configuration.
    """

    nv: int = 32
    k_max: int = 300
    hermitian_type: str = "complex"  # "real" or "complex"
    seed: Optional[int] = 0
    breakdown_abs_tol: float = 1e-14
    breakdown_rel_tol: float = 1e-12
    reorth_mode: str = "none"  # "none" or "local"
    reorth_window: int = 8
    reorth_passes: int = 2
    sync_timing: bool = True


@dataclass
class SLQLanczosResult:
    """
    Stores the full tridiagonal coefficients for all probes.
    """

    alpha: np.ndarray  # (nv, k_max)
    beta: np.ndarray  # (nv, k_max - 1)
    active_steps: np.ndarray  # (nv,)
    cfg: SLQLanczosConfig
    diagnostics: dict[str, Any]


@dataclass
class SLQAtomPack:
    """
    Aggregated Gaussian quadrature atoms from a Lanczos prefix.
    """

    nodes: np.ndarray
    weights: np.ndarray
    m: int
    nv_total: int
    nv_used: int


@dataclass
class SLQAnalysis:
    """
    Structured machine-readable diagnostics derived from one SLQ run.
    """

    prefix: list[dict[str, Any]]
    prefix_stability: dict[str, Any]
    final_atoms: SLQAtomPack
    final_quantiles: dict[str, float]
    final_kappa_eff: dict[str, float]
    final_spread_eff: dict[str, float]
    final_near_zero_mass: dict[str, float]
    final_tail_ratios: dict[str, float]
    final_labels: dict[str, bool]
    lambda_hat_min: float
    lambda_hat_max: float
    top_cloud: np.ndarray
    bottom_cloud: np.ndarray
    grid: dict[str, Any]
    spectrum_mode: str
    assumptions: str
    warnings: list[str]


@dataclass
class SLQReport:
    """
    Human-facing summary and plot-ready views.
    """

    headline: dict[str, Any]
    compact_metrics: dict[str, float]
    diagnosis_text: str
    plots: dict[str, Any]
    warnings: list[str]


def run_slq_lanczos_diagnostic(
    backend: Any,
    matvec: Callable[[Any, Any], Any],
    size: int,
    cfg: Optional[SLQLanczosConfig] = None,
) -> SLQLanczosResult:
    """
    Run one diagnostic-grade SLQ Lanczos pass:
    - ``nv`` probes
    - ``k_max`` steps per probe
    - stores full alpha/beta trajectories for all probes
    """
    cfg = cfg or SLQLanczosConfig()
    xp = backend.xp
    n = int(size)
    if n <= 1:
        raise ValueError("size must be > 1.")
    nv = int(cfg.nv)
    k_max = int(cfg.k_max)
    if nv < 1:
        raise ValueError("nv must be >= 1.")
    if k_max < 1:
        raise ValueError("k_max must be >= 1.")
    htype = str(cfg.hermitian_type).strip().lower()
    if htype not in ("real", "complex"):
        raise ValueError("hermitian_type must be one of: real, complex.")
    abs_tol = float(cfg.breakdown_abs_tol)
    rel_tol = float(cfg.breakdown_rel_tol)
    if abs_tol <= 0.0:
        raise ValueError("breakdown_abs_tol must be positive.")
    if rel_tol < 0.0:
        raise ValueError("breakdown_rel_tol must be >= 0.")
    reorth_mode = str(cfg.reorth_mode).strip().lower()
    if reorth_mode not in ("none", "local"):
        raise ValueError("reorth_mode must be one of: none, local.")
    reorth_window = max(int(cfg.reorth_window), 0)
    reorth_passes = max(int(cfg.reorth_passes), 1)
    sync_timing = bool(cfg.sync_timing)

    rng = None
    if cfg.seed is not None:
        try:
            rng = xp.random.RandomState(int(cfg.seed))
        except Exception:
            rng = None

    alpha_all = np.full((nv, k_max), np.nan, dtype=np.float64)
    beta_all = np.full((nv, max(k_max - 1, 0)), np.nan, dtype=np.float64)
    active = np.zeros((nv,), dtype=np.int32)

    t_matvec_total = 0.0
    n_matvec = 0
    t_reorth_total = 0.0
    n_reorth_calls = 0
    n_beta_checks = 0
    n_beta_threshold_hits = 0
    n_beta_nonfinite = 0
    breakdown_count = 0
    alpha_imag_sum = 0.0
    alpha_imag_max = 0.0
    alpha_imag_rel_max = 0.0
    n_alpha = 0
    t0 = time.perf_counter()

    q = xp.empty((n,), dtype=xp.complex128)
    q_prev = xp.empty((n,), dtype=xp.complex128)
    z = xp.empty((n,), dtype=xp.complex128)
    aq = xp.empty((n,), dtype=xp.complex128)

    for ell in range(nv):
        if htype == "real":
            q0 = _sample_real_rademacher(xp, n, rng).astype(xp.complex128)
        else:
            q0 = _sample_complex_rademacher(xp, n, rng)
        norm_q0 = float(backend.linalg.norm(q0))
        if not math.isfinite(norm_q0) or norm_q0 <= 0.0:
            raise RuntimeError("Probe normalization failed (non-finite or zero norm).")
        xp.divide(q0, norm_q0, out=q)
        q_prev.fill(0.0)
        beta_prev = 0.0
        used_steps = 0
        basis_window: list[Any] = []

        for j in range(k_max):
            if sync_timing:
                _sync_device(xp)
            tm0 = time.perf_counter()
            ret = matvec(q, aq)
            if ret is not None and ret is not aq:
                xp.copyto(aq, ret)
            if sync_timing:
                _sync_device(xp)
            t_matvec_total += time.perf_counter() - tm0
            n_matvec += 1

            if j > 0 and beta_prev != 0.0:
                z[:] = aq - beta_prev * q_prev
            else:
                xp.copyto(z, aq)

            alpha_raw = backend.linalg.vdot(q, z)
            alpha_j = float(xp.real(alpha_raw))
            alpha_imag = float(abs(xp.imag(alpha_raw)))
            alpha_imag_sum += alpha_imag
            alpha_imag_max = max(alpha_imag_max, alpha_imag)
            alpha_imag_rel = alpha_imag / max(1.0, abs(alpha_j))
            alpha_imag_rel_max = max(alpha_imag_rel_max, alpha_imag_rel)
            n_alpha += 1
            z -= alpha_j * q
            if reorth_mode == "local" and reorth_window > 0 and basis_window:
                tr0 = time.perf_counter()
                _local_reorthogonalize(
                    backend,
                    z,
                    basis_window,
                    passes=reorth_passes,
                )
                t_reorth_total += time.perf_counter() - tr0
                n_reorth_calls += 1
            alpha_all[ell, j] = alpha_j
            used_steps = j + 1

            if j == k_max - 1:
                break

            norm_aq = float(backend.linalg.norm(aq))
            beta_j = float(backend.linalg.norm(z))
            beta_all[ell, j] = beta_j
            beta_threshold = abs_tol + rel_tol * max(norm_aq, abs(alpha_j), abs(beta_prev), 1.0)
            n_beta_checks += 1
            if not math.isfinite(beta_j):
                n_beta_nonfinite += 1
                breakdown_count += 1
                break
            if beta_j <= beta_threshold:
                n_beta_threshold_hits += 1
                breakdown_count += 1
                break

            xp.copyto(q_prev, q)
            xp.divide(z, beta_j, out=q)
            beta_prev = beta_j
            if reorth_mode == "local" and reorth_window > 0:
                basis_window.append(q_prev.copy())
                if len(basis_window) > reorth_window:
                    basis_window.pop(0)

        active[ell] = int(used_steps)

    t1 = time.perf_counter()
    diag = {
        "nv": int(nv),
        "k_max": int(k_max),
        "n_matvec": int(n_matvec),
        "t_matvec_total": float(t_matvec_total),
        "t_matvec_avg": float(t_matvec_total / max(n_matvec, 1)),
        "t_total": float(t1 - t0),
        "active_min": int(np.min(active)),
        "active_max": int(np.max(active)),
        "active_mean": float(np.mean(active)),
        "hermitian_type": htype,
        "breakdown_abs_tol": abs_tol,
        "breakdown_rel_tol": rel_tol,
        "breakdown_count": int(breakdown_count),
        "breakdown_rate": float(breakdown_count / max(nv, 1)),
        "n_beta_checks": int(n_beta_checks),
        "n_beta_threshold_hits": int(n_beta_threshold_hits),
        "n_beta_nonfinite": int(n_beta_nonfinite),
        "beta_threshold_hit_rate": float(n_beta_threshold_hits / max(n_beta_checks, 1)),
        "alpha_imag_mean": float(alpha_imag_sum / max(n_alpha, 1)),
        "alpha_imag_max": float(alpha_imag_max),
        "alpha_imag_rel_max": float(alpha_imag_rel_max),
        "reorth_mode": reorth_mode,
        "reorth_window": int(reorth_window),
        "reorth_passes": int(reorth_passes),
        "n_reorth_calls": int(n_reorth_calls),
        "t_reorth_total": float(t_reorth_total),
        "t_reorth_avg": float(t_reorth_total / max(n_reorth_calls, 1)),
        "sync_timing": sync_timing,
    }
    return SLQLanczosResult(
        alpha=alpha_all,
        beta=beta_all,
        active_steps=active,
        cfg=cfg,
        diagnostics=diag,
    )


def atoms_from_prefix(result: SLQLanczosResult, m: Optional[int] = None) -> SLQAtomPack:
    """
    Build aggregated quadrature atoms from Lanczos prefix ``m``.
    """
    k_max = int(result.cfg.k_max)
    mm = _validate_prefix_m(k_max, m)
    nv = int(result.cfg.nv)
    nodes_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    used = 0
    for ell in range(nv):
        m_eff = min(mm, int(result.active_steps[ell]))
        if m_eff < 1:
            continue
        a = result.alpha[ell, :m_eff]
        b = result.beta[ell, : max(m_eff - 1, 0)]
        vals, w = _probe_nodes_weights(a, b, m_eff)
        nodes_list.append(vals)
        weights_list.append(w)
        used += 1
    if not nodes_list:
        return SLQAtomPack(
            nodes=np.asarray([], dtype=np.float64),
            weights=np.asarray([], dtype=np.float64),
            m=mm,
            nv_total=nv,
            nv_used=0,
        )
    nodes = np.concatenate(nodes_list).astype(np.float64, copy=False)
    weights = np.concatenate(weights_list).astype(np.float64, copy=False) / float(max(used, 1))
    return SLQAtomPack(nodes=nodes, weights=weights, m=mm, nv_total=nv, nv_used=used)


def evaluate_cdf(atoms: SLQAtomPack, x_grid: Sequence[float]) -> np.ndarray:
    """
    Evaluate empirical spectral CDF on ``x_grid``.
    """
    x = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    if atoms.nodes.size == 0:
        return np.full_like(x, np.nan)
    order = np.argsort(atoms.nodes)
    nodes = atoms.nodes[order]
    weights = atoms.weights[order]
    cdf = np.cumsum(weights)
    idx = np.searchsorted(nodes, x, side="right")
    out = np.zeros_like(x)
    mask = idx > 0
    out[mask] = cdf[idx[mask] - 1]
    return out


def evaluate_gaussian_density(atoms: SLQAtomPack, x_grid: Sequence[float], sigma: float) -> np.ndarray:
    """
    Evaluate Gaussian-smoothed density from atom pack.
    """
    s = float(sigma)
    if s <= 0.0:
        raise ValueError("sigma must be positive.")
    x = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    if atoms.nodes.size == 0:
        return np.full_like(x, np.nan)
    diff = (x[:, None] - atoms.nodes[None, :]) / s
    ker = np.exp(-0.5 * diff * diff) / (math.sqrt(2.0 * math.pi) * s)
    return ker @ atoms.weights


def _safe_ratio(num: float, den: float, floor: float = 1e-30) -> float:
    return float(num / max(abs(den), floor))


def _compute_prefix_steps(k_max: int, prefix_steps: Optional[Sequence[int]]) -> list[int]:
    if prefix_steps is None:
        steps = list(range(20, k_max + 1, 20))
        if not steps:
            steps = [k_max]
        if steps[-1] != k_max:
            steps.append(k_max)
        return steps
    steps = sorted({int(v) for v in prefix_steps if 1 <= int(v) <= k_max})
    if not steps:
        raise ValueError("prefix_steps has no valid step in [1, k_max].")
    return steps


def analyze_slq_result(
    result: SLQLanczosResult,
    *,
    prefix_steps: Optional[Sequence[int]] = None,
    spectrum_mode: str = "spd",
    quantiles: Sequence[float] = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999),
    kappa_taus: Sequence[float] = (1e-3, 1e-2, 5e-2),
    near_zero_r: Sequence[int] = (1, 2, 3, 4, 5, 6),
    density_sigma_factors: Sequence[float] = (0.002, 0.005, 0.01),
    density_grid_size: int = 512,
) -> tuple[SLQAnalysis, dict[str, Any]]:
    """
    Analyze one SLQ run and return:
    - derived structured analysis
    - extra raw payload not suitable for compact dataclass fields
    """
    k_max = int(result.cfg.k_max)
    mode = str(spectrum_mode).strip().lower()
    if mode not in ("spd", "hermitian"):
        raise ValueError("spectrum_mode must be one of: spd, hermitian.")
    steps = _compute_prefix_steps(k_max, prefix_steps)

    prefix_rows: list[dict[str, Any]] = []
    prefix_atoms_raw: list[dict[str, Any]] = []
    final_atoms = atoms_from_prefix(result, m=steps[-1])
    if final_atoms.nodes.size == 0:
        analysis = SLQAnalysis(
            prefix=[],
            prefix_stability={},
            final_atoms=final_atoms,
            final_quantiles={},
            final_kappa_eff={},
            final_spread_eff={},
            final_near_zero_mass={},
            final_tail_ratios={},
            final_labels={},
            lambda_hat_min=float("nan"),
            lambda_hat_max=float("nan"),
            top_cloud=np.asarray([], dtype=np.float64),
            bottom_cloud=np.asarray([], dtype=np.float64),
            grid={"x": np.asarray([], dtype=np.float64), "cdf": np.asarray([], dtype=np.float64), "density": {}},
            spectrum_mode=mode,
            assumptions="SPD/PSD non-negative spectrum" if mode == "spd" else "General Hermitian spectrum (signed).",
            warnings=["No atoms available at final prefix."],
        )
        raw_extra = {
            "prefix_atoms": [],
            "extremal_ritz_cloud": {"top": np.asarray([], dtype=np.float64), "bottom": np.asarray([], dtype=np.float64)},
        }
        return analysis, raw_extra

    q_map_final = {float(q): _weighted_quantile(final_atoms.nodes, final_atoms.weights, float(q)) for q in quantiles}
    q99 = float(q_map_final.get(0.99, np.nan))
    q99_valid_spd = bool(math.isfinite(q99) and q99 > 0.0)

    def _kappa_from_q(nodes: np.ndarray, weights: np.ndarray, tau: float) -> float:
        ql = _weighted_quantile(nodes, weights, float(tau))
        qh = _weighted_quantile(nodes, weights, float(1.0 - tau))
        denom = max(float(ql), 1e-30)
        return float(qh / denom)

    def _spread_from_q(nodes: np.ndarray, weights: np.ndarray, tau: float) -> float:
        ql = _weighted_quantile(nodes, weights, float(tau))
        qh = _weighted_quantile(nodes, weights, float(1.0 - tau))
        return float(qh - ql)

    for m in steps:
        atoms = atoms_from_prefix(result, m=m)
        if atoms.nodes.size == 0:
            continue
        prefix_atoms_raw.append(
            {
                "m": int(m),
                "nv_total": int(atoms.nv_total),
                "nv_used": int(atoms.nv_used),
                "nodes": atoms.nodes.copy(),
                "weights": atoms.weights.copy(),
            }
        )
        q_map = {float(q): _weighted_quantile(atoms.nodes, atoms.weights, float(q)) for q in quantiles}
        kappa_map = (
            {float(tau): _kappa_from_q(atoms.nodes, atoms.weights, float(tau)) for tau in kappa_taus}
            if mode == "spd"
            else {}
        )
        spread_map = (
            {float(tau): _spread_from_q(atoms.nodes, atoms.weights, float(tau)) for tau in kappa_taus}
            if mode == "hermitian"
            else {}
        )
        mz: dict[int, float] = {}
        if mode == "spd":
            if q99_valid_spd:
                for r in near_zero_r:
                    eps_r = (10.0 ** (-int(r))) * q99
                    mz[int(r)] = float(np.sum(atoms.weights[atoms.nodes <= eps_r]))
            else:
                for r in near_zero_r:
                    mz[int(r)] = float("nan")
        else:
            q01_abs = abs(float(q_map.get(0.01, np.nan)))
            q99_abs = abs(float(q_map.get(0.99, np.nan)))
            anchor = max(q01_abs, q99_abs, 1e-30)
            for r in near_zero_r:
                eps_r = (10.0 ** (-int(r))) * anchor
                mz[int(r)] = float(np.sum(atoms.weights[np.abs(atoms.nodes) <= eps_r]))
        prefix_rows.append(
            {
                "m": int(m),
                "quantiles": q_map,
                "kappa_eff": kappa_map,
                "spread_eff": spread_map,
                "near_zero_mass": mz,
            }
        )

    m_final = steps[-1]
    lambda_hat_min = float("nan")
    lambda_hat_max = float("nan")
    top_cloud: list[float] = []
    bottom_cloud: list[float] = []
    top_take = 5
    bottom_take = 5
    for ell in range(int(result.cfg.nv)):
        m_eff = min(m_final, int(result.active_steps[ell]))
        if m_eff < 1:
            continue
        vals, _w = _probe_nodes_weights(
            result.alpha[ell, :m_eff],
            result.beta[ell, : max(m_eff - 1, 0)],
            m_eff,
        )
        lambda_hat_min = min(lambda_hat_min, float(vals[0])) if math.isfinite(lambda_hat_min) else float(vals[0])
        lambda_hat_max = max(lambda_hat_max, float(vals[-1])) if math.isfinite(lambda_hat_max) else float(vals[-1])
        top_cloud.extend([float(v) for v in vals[-top_take:]])
        bottom_cloud.extend([float(v) for v in vals[:bottom_take]])

    support_lo = float(np.min(final_atoms.nodes))
    support_hi = float(np.max(final_atoms.nodes))
    span = max(support_hi - support_lo, 1e-30)
    sigma_max_fac = float(max(density_sigma_factors)) if len(density_sigma_factors) > 0 else 0.0
    sigma_max = max(sigma_max_fac * span, 0.0)
    pad = 4.0 * sigma_max
    x_grid = np.linspace(support_lo - pad, support_hi + pad, int(max(density_grid_size, 32)))
    cdf_grid = evaluate_cdf(final_atoms, x_grid)
    density = {}
    for fac in density_sigma_factors:
        s = float(fac) * span
        density[float(fac)] = evaluate_gaussian_density(final_atoms, x_grid, s)

    near_zero_final: dict[int, float] = {}
    if mode == "spd":
        if q99_valid_spd:
            for r in near_zero_r:
                eps_r = (10.0 ** (-int(r))) * q99
                near_zero_final[int(r)] = float(np.sum(final_atoms.weights[final_atoms.nodes <= eps_r]))
        else:
            for r in near_zero_r:
                near_zero_final[int(r)] = float("nan")
    else:
        q01_abs = abs(float(q_map_final.get(0.01, np.nan)))
        q99_abs = abs(float(q_map_final.get(0.99, np.nan)))
        anchor = max(q01_abs, q99_abs, 1e-30)
        for r in near_zero_r:
            eps_r = (10.0 ** (-int(r))) * anchor
            near_zero_final[int(r)] = float(np.sum(final_atoms.weights[np.abs(final_atoms.nodes) <= eps_r]))

    warnings: list[str] = []
    if str(result.cfg.reorth_mode).strip().lower() != "none":
        warnings.append(
            "Local reorthogonalization is enabled; tridiagonal T is used as an SLQ proxy."
        )
    if mode == "spd":
        q01 = float(q_map_final.get(0.01, np.nan))
        if not math.isfinite(q01) or q01 <= 0.0:
            warnings.append(
                "SPD summary assumes non-negative spectrum; observed low quantile is non-positive."
            )
        if not q99_valid_spd:
            warnings.append(
                "q99 is non-finite or non-positive; near-zero mass is marked as nan."
            )

    kappa_final = (
        {float(tau): _kappa_from_q(final_atoms.nodes, final_atoms.weights, float(tau)) for tau in kappa_taus}
        if mode == "spd"
        else {}
    )
    spread_final = (
        {float(tau): _spread_from_q(final_atoms.nodes, final_atoms.weights, float(tau)) for tau in kappa_taus}
        if mode == "hermitian"
        else {}
    )

    q001 = float(q_map_final.get(0.001, np.nan))
    q01 = float(q_map_final.get(0.01, np.nan))
    q05 = float(q_map_final.get(0.05, np.nan))
    q95 = float(q_map_final.get(0.95, np.nan))
    q99f = float(q_map_final.get(0.99, np.nan))
    q999 = float(q_map_final.get(0.999, np.nan))
    bulk_width = q95 - q05
    full_width = q99f - q01
    upper_tail = q999 - q99f
    lower_tail = q01 - q001
    tail_ratio = _safe_ratio(upper_tail, max(full_width, 1e-30))
    spike_ratio = _safe_ratio(lambda_hat_max, q999 if math.isfinite(q999) else lambda_hat_max)
    bulk_width_ratio = _safe_ratio(bulk_width, max(abs(q50 := float(q_map_final.get(0.5, np.nan))), 1e-30))
    near_zero_anchor = near_zero_final.get(3, float("nan"))
    near_zero_anchor_f = float(near_zero_anchor) if near_zero_anchor is not None else float("nan")

    labels = {
        "bulk_tight": bool(math.isfinite(bulk_width_ratio) and bulk_width_ratio < 5.0),
        "upper_tail_heavy": bool(math.isfinite(tail_ratio) and tail_ratio > 0.20),
        "near_zero_dominated": bool(math.isfinite(near_zero_anchor_f) and near_zero_anchor_f > 0.15),
    }

    prefix_stability: dict[str, Any] = {"window": 5, "metrics": {}, "stable": True}
    if len(prefix_rows) >= 2:
        take = prefix_rows[-min(5, len(prefix_rows)) :]
        q01_series = np.asarray([float(r["quantiles"].get(0.01, np.nan)) for r in take], dtype=np.float64)
        q99_series = np.asarray([float(r["quantiles"].get(0.99, np.nan)) for r in take], dtype=np.float64)
        nz3_series = np.asarray([float(r["near_zero_mass"].get(3, np.nan)) for r in take], dtype=np.float64)
        if mode == "spd":
            keff_series = np.asarray([float(r["kappa_eff"].get(0.01, np.nan)) for r in take], dtype=np.float64)
        else:
            keff_series = np.asarray([float(r["spread_eff"].get(0.01, np.nan)) for r in take], dtype=np.float64)

        def _rel_span(a: np.ndarray) -> float:
            if a.size == 0 or not np.any(np.isfinite(a)):
                return float("nan")
            mn = float(np.nanmin(a))
            mx = float(np.nanmax(a))
            mid = max(abs(float(np.nanmean(a))), 1e-30)
            return float((mx - mn) / mid)

        prefix_stability["metrics"] = {
            "q01_rel_span": _rel_span(q01_series),
            "q99_rel_span": _rel_span(q99_series),
            "shape_rel_span": _rel_span(keff_series),
            "near_zero_rel_span": _rel_span(nz3_series),
        }
        vals = [v for v in prefix_stability["metrics"].values() if math.isfinite(v)]
        prefix_stability["stable"] = bool(len(vals) > 0 and max(vals) < 0.15)
    else:
        prefix_stability["stable"] = False

    analysis = SLQAnalysis(
        prefix=prefix_rows,
        prefix_stability=prefix_stability,
        final_atoms=final_atoms,
        final_quantiles=q_map_final,
        final_kappa_eff=kappa_final,
        final_spread_eff=spread_final,
        final_near_zero_mass=near_zero_final,
        final_tail_ratios={
            "bulk_width": float(bulk_width),
            "full_width": float(full_width),
            "upper_tail": float(upper_tail),
            "lower_tail": float(lower_tail),
            "tail_ratio": float(tail_ratio),
            "bulk_width_ratio": float(bulk_width_ratio),
            "spike_ratio": float(spike_ratio),
        },
        final_labels=labels,
        lambda_hat_min=float(lambda_hat_min),
        lambda_hat_max=float(lambda_hat_max),
        top_cloud=np.asarray(top_cloud, dtype=np.float64),
        bottom_cloud=np.asarray(bottom_cloud, dtype=np.float64),
        grid={"x": x_grid, "cdf": cdf_grid, "density": density},
        spectrum_mode=mode,
        assumptions="SPD/PSD non-negative spectrum" if mode == "spd" else "General Hermitian spectrum (signed).",
        warnings=warnings,
    )

    raw_extra = {
        "prefix_atoms": prefix_atoms_raw,
        "extremal_ritz_cloud": {
            "top": np.asarray(top_cloud, dtype=np.float64),
            "bottom": np.asarray(bottom_cloud, dtype=np.float64),
        },
    }
    return analysis, raw_extra


def build_slq_views(result: SLQLanczosResult, analysis: SLQAnalysis) -> SLQReport:
    q = analysis.final_quantiles
    tail = analysis.final_tail_ratios
    stable = bool(analysis.prefix_stability.get("stable", False))
    labels = analysis.final_labels

    if labels.get("near_zero_dominated", False):
        dominant_issue = "near_zero_tail"
    elif labels.get("upper_tail_heavy", False):
        dominant_issue = "upper_spikes"
    else:
        dominant_issue = "bulk_width"

    health = "stable" if stable and len(analysis.warnings) == 0 else "caution"
    bulk_shape = "tight" if labels.get("bulk_tight", False) else "wide"
    near_zero_tail = "strong" if labels.get("near_zero_dominated", False) else "weak"

    headline = {
        "health": health,
        "dominant_issue": dominant_issue,
        "bulk_shape": bulk_shape,
        "near_zero_tail": near_zero_tail,
    }
    compact = {
        "q01": float(q.get(0.01, np.nan)),
        "q50": float(q.get(0.5, np.nan)),
        "q95": float(q.get(0.95, np.nan)),
        "q99": float(q.get(0.99, np.nan)),
        "q999": float(q.get(0.999, np.nan)),
        "lambda_max": float(analysis.lambda_hat_max),
        "bulk_width_ratio": float(tail.get("bulk_width_ratio", np.nan)),
        "tail_ratio": float(tail.get("tail_ratio", np.nan)),
        "spike_ratio": float(tail.get("spike_ratio", np.nan)),
    }
    diagnosis = (
        f"Health={health}; dominant issue={dominant_issue}; "
        f"bulk={bulk_shape}; near-zero tail={near_zero_tail}; "
        f"prefix stable={stable}."
    )
    plots = {
        "cdf_table": {"x": analysis.grid["x"], "cdf": analysis.grid["cdf"]},
        "density_table": {
            "x": analysis.grid["x"],
            "density": analysis.grid["density"],
        },
        "prefix_table": analysis.prefix,
        "ritz_cloud_table": {
            "top": analysis.top_cloud,
            "bottom": analysis.bottom_cloud,
        },
    }
    return SLQReport(
        headline=headline,
        compact_metrics=compact,
        diagnosis_text=diagnosis,
        plots=plots,
        warnings=list(analysis.warnings),
    )


def package_slq_output(
    result: SLQLanczosResult,
    analysis: SLQAnalysis,
    report: SLQReport,
    *,
    raw_extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    raw_extra = raw_extra or {}
    raw = {
        "config": result.cfg,
        "run_diagnostics": result.diagnostics,
        "alpha": result.alpha,
        "beta": result.beta,
        "active_steps": result.active_steps,
        "prefix_atoms": raw_extra.get("prefix_atoms", []),
        "x_grid": analysis.grid["x"],
        "cdf": analysis.grid["cdf"],
        "density": analysis.grid["density"],
        "extremal_ritz_cloud": raw_extra.get(
            "extremal_ritz_cloud",
            {"top": analysis.top_cloud, "bottom": analysis.bottom_cloud},
        ),
    }
    derived = asdict(analysis)
    views = asdict(report)
    return {"raw": raw, "derived": derived, "views": views}


def _lookup_quantile(q_map: dict[Any, Any], q: float) -> float:
    if not isinstance(q_map, dict):
        return float("nan")
    if q in q_map:
        return float(q_map[q])
    qs = str(q)
    if qs in q_map:
        return float(q_map[qs])
    for k, v in q_map.items():
        try:
            if abs(float(k) - float(q)) < 1e-15:
                return float(v)
        except Exception:
            pass
    return float("nan")


def build_slq_plot_payload(
    summary: dict[str, Any],
    *,
    left_zoom: Optional[tuple[float, float]] = None,
    left_zoom_points: int = 1000,
) -> dict[str, Any]:
    """
    Build plot-ready arrays from packaged SLQ summary.
    """
    raw = summary.get("raw", {}) if isinstance(summary, dict) else {}
    derived = summary.get("derived", {}) if isinstance(summary, dict) else {}

    x = np.asarray(raw.get("x_grid", []), dtype=np.float64)
    cdf = np.asarray(raw.get("cdf", []), dtype=np.float64)
    density = raw.get("density", {}) if isinstance(raw, dict) else {}
    dens_002 = np.asarray(density.get(0.002, density.get("0.002", [])), dtype=np.float64)
    dens_005 = np.asarray(density.get(0.005, density.get("0.005", [])), dtype=np.float64)
    dens_010 = np.asarray(density.get(0.01, density.get("0.01", [])), dtype=np.float64)

    prefix = derived.get("prefix", []) if isinstance(derived, dict) else []
    m_vals = np.asarray([int(r.get("m", 0)) for r in prefix], dtype=np.float64)
    q001 = np.asarray([_lookup_quantile(r.get("quantiles", {}), 0.001) for r in prefix], dtype=np.float64)
    q095 = np.asarray([_lookup_quantile(r.get("quantiles", {}), 0.95) for r in prefix], dtype=np.float64)
    q099 = np.asarray([_lookup_quantile(r.get("quantiles", {}), 0.99) for r in prefix], dtype=np.float64)
    kappa01 = np.asarray([_lookup_quantile(r.get("kappa_eff", {}), 0.01) for r in prefix], dtype=np.float64)
    kappa05 = np.asarray([_lookup_quantile(r.get("kappa_eff", {}), 0.05) for r in prefix], dtype=np.float64)

    q_map = derived.get("final_quantiles", {}) if isinstance(derived, dict) else {}
    lam_min = float(derived.get("lambda_hat_min", np.nan))
    if left_zoom is None:
        lo = 0.0
        hi = 1.0
        if math.isfinite(lam_min) and lam_min > 0.0:
            lo = max(0.0, 0.8 * lam_min)
            hi = max(0.5, 4.5 * lam_min)
    else:
        lo, hi = float(left_zoom[0]), float(left_zoom[1])
    x_zoom = np.linspace(lo, hi, int(max(left_zoom_points, 64)))

    prefix_atoms = raw.get("prefix_atoms", []) if isinstance(raw, dict) else []
    nodes = np.asarray([], dtype=np.float64)
    weights = np.asarray([], dtype=np.float64)
    if isinstance(prefix_atoms, list) and len(prefix_atoms) > 0:
        best = None
        best_m = -1
        for item in prefix_atoms:
            try:
                mm = int(item.get("m", -1))
            except Exception:
                mm = -1
            if mm > best_m:
                best_m = mm
                best = item
        if isinstance(best, dict):
            nodes = np.asarray(best.get("nodes", []), dtype=np.float64)
            weights = np.asarray(best.get("weights", []), dtype=np.float64)

    dens_zoom_005 = np.asarray([], dtype=np.float64)
    dens_zoom_010 = np.asarray([], dtype=np.float64)
    if nodes.size > 0 and weights.size == nodes.size:
        class _Atoms:
            pass

        atoms = _Atoms()
        atoms.nodes = nodes
        atoms.weights = weights
        span = max(float(np.max(nodes) - np.min(nodes)), 1e-30) if nodes.size > 1 else 1.0
        dens_zoom_005 = evaluate_gaussian_density(atoms, x_zoom, sigma=0.005 * span)
        dens_zoom_010 = evaluate_gaussian_density(atoms, x_zoom, sigma=0.01 * span)

    cloud = raw.get("extremal_ritz_cloud", {}) if isinstance(raw, dict) else {}
    top_cloud = np.asarray(cloud.get("top", []), dtype=np.float64)
    bottom_cloud = np.asarray(cloud.get("bottom", []), dtype=np.float64)

    return {
        "global": {"x": x, "cdf": cdf, "density_002": dens_002, "density_005": dens_005, "density_010": dens_010},
        "left_zoom": {
            "x": x_zoom,
            "density_005": dens_zoom_005,
            "density_010": dens_zoom_010,
            "lambda_min": lam_min,
            "q01": _lookup_quantile(q_map, 0.01),
            "q99": _lookup_quantile(q_map, 0.99),
        },
        "prefix": {
            "m": m_vals,
            "q001": q001,
            "q095": q095,
            "q099": q099,
            "kappa01": kappa01,
            "kappa05": kappa05,
        },
        "ritz": {
            "top": top_cloud,
            "bottom": bottom_cloud,
        },
    }


def save_slq_plots(
    summary: dict[str, Any],
    output_dir: str,
    *,
    dpi: int = 160,
    left_zoom: Optional[tuple[float, float]] = None,
) -> dict[str, str]:
    """
    Save the default SLQ diagnostic figures to disk.
    Returns a mapping of logical figure names to file paths.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    payload = build_slq_plot_payload(summary, left_zoom=left_zoom)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    x = payload["global"]["x"]
    cdf = payload["global"]["cdf"]
    xz = payload["left_zoom"]["x"]
    dz5 = payload["left_zoom"]["density_005"]
    dz10 = payload["left_zoom"]["density_010"]
    lam_min = float(payload["left_zoom"]["lambda_min"])
    m = payload["prefix"]["m"]
    q001 = payload["prefix"]["q001"]
    q095 = payload["prefix"]["q095"]
    q099 = payload["prefix"]["q099"]
    k01 = payload["prefix"]["kappa01"]
    k05 = payload["prefix"]["kappa05"]
    top = payload["ritz"]["top"]
    bottom = payload["ritz"]["bottom"]

    if x.size > 0 and cdf.size == x.size:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(x, cdf)
        ax.set_title("Global spectral CDF")
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("CDF")
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        p = out / "fig1_global_cdf.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved["fig1_global_cdf"] = str(p)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(x, cdf)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_title("Global spectral CDF (symlog-x)")
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("CDF")
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        p = out / "fig1b_global_cdf_symlogx.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved["fig1b_global_cdf_symlogx"] = str(p)

    if xz.size > 0:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if dz5.size == xz.size:
            ax.plot(xz, dz5, label="sigma=0.005")
        if dz10.size == xz.size:
            ax.plot(xz, dz10, label="sigma=0.01")
        if math.isfinite(lam_min):
            ax.axvline(lam_min, linestyle="--", alpha=0.6, label="lambda_min")
        ax.set_title("Density near left edge (local grid)")
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("density")
        ax.legend(loc="best")
        fig.tight_layout()
        p = out / "fig2_left_edge_density_local.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved["fig2_left_edge_density_local"] = str(p)

    if m.size > 0:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(m, q001, label="q0.001")
        ax.plot(m, q095, label="q0.95")
        ax.set_title("Prefix bulk quantiles vs m")
        ax.set_xlabel("Lanczos steps m")
        ax.set_ylabel("value")
        ax.legend(loc="best")
        fig.tight_layout()
        p = out / "fig3a_prefix_bulk_quantiles.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved["fig3a_prefix_bulk_quantiles"] = str(p)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(m, q099, label="q0.99")
        ax.set_title("Prefix upper-tail quantile vs m")
        ax.set_xlabel("Lanczos steps m")
        ax.set_ylabel("value")
        ax.legend(loc="best")
        fig.tight_layout()
        p = out / "fig3b_prefix_upper_tail_q99.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved["fig3b_prefix_upper_tail_q99"] = str(p)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if np.any(np.isfinite(k01)):
            ax.plot(m, k01, label="kappa_eff(0.01)")
        if np.any(np.isfinite(k05)):
            ax.plot(m, k05, label="kappa_eff(0.05)")
        ax.set_yscale("log")
        ax.set_title("Effective condition surrogates vs m")
        ax.set_xlabel("Lanczos steps m")
        ax.set_ylabel("value (log)")
        ax.legend(loc="best")
        fig.tight_layout()
        p = out / "fig3c_prefix_kappa_eff.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved["fig3c_prefix_kappa_eff"] = str(p)

    if bottom.size > 0:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.scatter(np.arange(bottom.size), bottom, s=10)
        ax.set_title("Bottom Ritz cloud")
        ax.set_xlabel("index")
        ax.set_ylabel("Ritz value")
        fig.tight_layout()
        p = out / "fig4a_bottom_ritz_cloud.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved["fig4a_bottom_ritz_cloud"] = str(p)

    if top.size > 0:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.scatter(np.arange(top.size), top, s=10)
        ax.set_title("Top Ritz cloud")
        ax.set_xlabel("index")
        ax.set_ylabel("Ritz value")
        fig.tight_layout()
        p = out / "fig4b_top_ritz_cloud.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved["fig4b_top_ritz_cloud"] = str(p)

    if x.size > 0 and cdf.size == x.size and m.size > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(x, cdf)
        axes[0, 0].set_xscale("symlog", linthresh=1.0)
        axes[0, 0].set_title("Global CDF (symlog-x)")
        axes[0, 0].set_xlabel("eigenvalue")
        axes[0, 0].set_ylabel("CDF")

        if xz.size > 0 and dz5.size == xz.size:
            axes[0, 1].plot(xz, dz5, label="sigma=0.005")
        if xz.size > 0 and dz10.size == xz.size:
            axes[0, 1].plot(xz, dz10, label="sigma=0.01")
        if math.isfinite(lam_min):
            axes[0, 1].axvline(lam_min, linestyle="--", alpha=0.6, label="lambda_min")
        axes[0, 1].set_title("Left-edge density (local grid)")
        axes[0, 1].set_xlabel("eigenvalue")
        axes[0, 1].set_ylabel("density")
        axes[0, 1].legend(loc="best")

        axes[1, 0].plot(m, q001, label="q0.001")
        axes[1, 0].plot(m, q095, label="q0.95")
        axes[1, 0].set_title("Prefix bulk quantiles")
        axes[1, 0].set_xlabel("m")
        axes[1, 0].set_ylabel("value")
        axes[1, 0].legend(loc="best")

        if top.size > 0:
            axes[1, 1].scatter(np.arange(top.size), top, s=10, label="top")
        if bottom.size > 0:
            axes[1, 1].scatter(np.arange(bottom.size), bottom, s=10, label="bottom")
        axes[1, 1].set_title("Extremal Ritz cloud")
        axes[1, 1].set_xlabel("index")
        axes[1, 1].set_ylabel("Ritz value")
        axes[1, 1].legend(loc="best")

        fig.tight_layout()
        p = out / "dashboard_2x2.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved["dashboard_2x2"] = str(p)

    return saved


def summarize_slq_diagnostics(
    result: SLQLanczosResult,
    *,
    prefix_steps: Optional[Sequence[int]] = None,
    spectrum_mode: str = "spd",
    quantiles: Sequence[float] = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999),
    kappa_taus: Sequence[float] = (1e-3, 1e-2, 5e-2),
    near_zero_r: Sequence[int] = (1, 2, 3, 4, 5, 6),
    density_sigma_factors: Sequence[float] = (0.002, 0.005, 0.01),
    density_grid_size: int = 512,
) -> dict[str, Any]:
    """
    High-level entry that returns a 3-layer output:
    { "raw": ..., "derived": ..., "views": ... }.
    """
    analysis, raw_extra = analyze_slq_result(
        result,
        prefix_steps=prefix_steps,
        spectrum_mode=spectrum_mode,
        quantiles=quantiles,
        kappa_taus=kappa_taus,
        near_zero_r=near_zero_r,
        density_sigma_factors=density_sigma_factors,
        density_grid_size=density_grid_size,
    )
    report = build_slq_views(result, analysis)
    return package_slq_output(result, analysis, report, raw_extra=raw_extra)

