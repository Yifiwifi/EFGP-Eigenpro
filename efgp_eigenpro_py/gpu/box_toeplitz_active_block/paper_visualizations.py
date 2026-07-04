from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...discretization import basis_weights, choose_grid_params
from ...efgp_solver import EFGPSolver
from ...kernels import make_matern, make_squared_exponential
from ..backends import build_gpu_backend_bundle
from ..contexts import GPUOperatorContext, ensure_gpu_data_context
from ..v1_ops import predict_v1
from ..v3_eigenspace import EigenspaceConfig
from ..versions import (
    _run_gpu_precompute,
    run_v1_pure_efgp,
    run_v3_full_gpu_eigenspace,
    run_v6_box_toeplitz_active_block,
    run_v7_box_eigenpro_active_block,
)
from .active_set import compute_rho
from .box_eigenpro import (
    _local_box_apply_block,
    build_box_eigenpro_preconditioner,
)
from .config import BTABConfig, BTABExperimentConfig, resolve_btab_experiment_route
from .run_experiments import build_gpu_run_config, load_processed_dataset, make_kernel


@dataclass
class PaperVisualizationConfig:
    output_dir: str | Path
    summary_root: str | Path | None = None
    residual_dataset_stem: str = "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain100000000"
    map_dataset_stem: str = "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain10000000"
    kernel_family: str = "matern"
    kernel_lengthscale: float = 0.1
    kernel_nu: float = 1.5
    kernel_variance: float = 1.0
    reg_lambda: float = 0.1
    eps: float = 1e-5
    nufft_tol: float = 1e-10
    l2_scaled: bool = True
    tol: float = 1e-6
    residual_rmse_ratio_tol: float = 1.10
    residual_maxiter: int | None = None
    residual_sample_size: int = 100_000
    residual_sample_seed: int = 17
    rmse_checkpoint_count: int = 70
    residual_trace_dense_until: int = 100
    residual_trace_mid_until: int = 1000
    residual_trace_stride_mid: int = 10
    residual_trace_stride_late: int = 50
    residual_zoom_iterations: int = 400
    spectrum_active_topk: int = 2048
    spectrum_boxeig_q: int = 256
    map_active_topk: int = 4096
    map_sample_size: int = 200_000
    map_sample_seed: int = 23
    map_bins: int = 256
    boxeig_sweep_n_train: int = 300_000_000
    boxeig_sweep_dataset_contains: str = "USGS"
    make_active_score: bool = True
    make_spectrum: bool = True
    make_mechanism_figure: bool = True
    make_residual: bool = True
    make_rmse: bool = True
    make_prediction_map: bool = True
    make_boxeig_sweep: bool = True


PAPER_LABELS = {
    "plain": "EFGP-CG",
    "v3": "Global EigenPro-style PCG",
    "active_inverse": "Active inverse",
    "boxeig": "Box-EigenPro",
}

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
BLACK = "#000000"
GRAY = "#999999"

PROFILE_STYLES = {
    "SE": {"color": BLUE, "linestyle": "-", "linewidth": 1.4},
    "matern": {"color": PURPLE, "linestyle": "-", "linewidth": 1.4},
}

METHOD_STYLES = {
    "EFGP-CG": {"color": BLUE, "linestyle": "-", "linewidth": 1.35},
    "Global EigenPro-style PCG": {"color": ORANGE, "linestyle": "--", "linewidth": 1.35},
    "best global EigenPro-style PCG": {"color": ORANGE, "linestyle": "--", "linewidth": 1.35},
    "Active inverse": {"color": GREEN, "linestyle": "-.", "linewidth": 1.35},
    "Exact active-block solve": {"color": GREEN, "linestyle": "-.", "linewidth": 1.35},
    "Box-EigenPro": {"color": RED, "linestyle": "-", "linewidth": 1.35},
}


def _asnumpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "get"):
        return np.asarray(arr.get())
    return np.asarray(arr)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k, "")) for k in fieldnames})


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_existing(out_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = out_dir / name
        if path.exists():
            return path
    return None


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "ok", "converged"}:
        return True
    if text in {"false", "0", "no", "n", "nan", "none", ""}:
        return False
    return default


def _save_figure(fig: Any, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    return {"png": str(png), "pdf": str(pdf)}


def _import_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _cfg_for_kernel(exp_cfg: BTABExperimentConfig, viz_cfg: PaperVisualizationConfig, family: str) -> BTABExperimentConfig:
    family_key = str(family).strip().lower()
    if family_key in {"se", "squared_exponential", "rbf", "gaussian"}:
        return replace(
            exp_cfg,
            kernel_family="SE",
            kernel_lengthscale=float(viz_cfg.kernel_lengthscale),
            kernel_variance=float(viz_cfg.kernel_variance),
            reg_lambda=float(viz_cfg.reg_lambda),
            eps=float(viz_cfg.eps),
            nufft_tol=float(viz_cfg.nufft_tol),
            l2_scaled=bool(viz_cfg.l2_scaled),
            tol=float(viz_cfg.tol),
        )
    return replace(
        exp_cfg,
        kernel_family="matern",
        kernel_lengthscale=float(viz_cfg.kernel_lengthscale),
        kernel_nu=float(viz_cfg.kernel_nu),
        kernel_variance=float(viz_cfg.kernel_variance),
        reg_lambda=float(viz_cfg.reg_lambda),
        eps=float(viz_cfg.eps),
        nufft_tol=float(viz_cfg.nufft_tol),
        l2_scaled=bool(viz_cfg.l2_scaled),
        tol=float(viz_cfg.tol),
    )


def _kernel_for_profile(family: str, dim: int, viz_cfg: PaperVisualizationConfig) -> Any:
    family_key = str(family).strip().lower()
    if family_key in {"se", "squared_exponential", "rbf", "gaussian"}:
        return make_squared_exponential(
            lengthscale=float(viz_cfg.kernel_lengthscale),
            dim=int(dim),
            variance=float(viz_cfg.kernel_variance),
        )
    return make_matern(
        nu=float(viz_cfg.kernel_nu),
        lengthscale=float(viz_cfg.kernel_lengthscale),
        dim=int(dim),
        variance=float(viz_cfg.kernel_variance),
    )


def _single_col_size(height: float = 2.3) -> tuple[float, float]:
    return (3.4, float(height))


def _double_col_size(height: float = 2.4) -> tuple[float, float]:
    return (6.8, float(height))


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        zorder=20,
    )


def _active_score_profile_data(viz_cfg: PaperVisualizationConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    profile_specs = [
        ("SE kernel, M=1225", "SE"),
        ("Matern 3/2 kernel, M=35721", "matern"),
    ]
    for label, family in profile_specs:
        kernel = _kernel_for_profile(family, 2, viz_cfg)
        grid = choose_grid_params(kernel, float(viz_cfg.eps), 1.0, l2scaled=bool(viz_cfg.l2_scaled))
        weights = basis_weights(kernel, grid.xis, grid.h)
        rho = compute_rho(1.0, weights, float(viz_cfg.reg_lambda))
        order = np.argsort(rho)[::-1]
        rho_sorted = np.asarray(rho[order], dtype=np.float64)
        denom = max(float(np.sum(rho_sorted)), 1e-300)
        cum = np.cumsum(rho_sorted) / denom
        rank = np.arange(1, int(cum.size) + 1)
        profiles.append(
            {
                "label": label,
                "kernel_family": family,
                "rank": rank,
                "cumulative_mass": cum,
                "M": int(cum.size),
                "mtot": int(grid.mtot),
                "h": float(grid.h),
            }
        )
        for idx in np.unique(np.geomspace(1, int(cum.size), min(400, int(cum.size))).astype(int)):
            rows.append(
                {
                    "profile_label": label,
                    "kernel_family": family,
                    "rank": int(idx),
                    "cumulative_rho_mass": float(cum[int(idx) - 1]),
                    "M": int(cum.size),
                    "mtot": int(grid.mtot),
                    "h": float(grid.h),
                    "eps": float(viz_cfg.eps),
                    "reg_lambda": float(viz_cfg.reg_lambda),
                }
            )
    return profiles, rows


def _plot_active_score_profiles(ax: Any, profiles: list[dict[str, Any]]) -> None:
    for profile in profiles:
        family = str(profile["kernel_family"])
        style = PROFILE_STYLES.get(family, {"color": BLACK, "linestyle": "-", "linewidth": 1.25})
        if family.lower() in {"se", "squared_exponential", "rbf", "gaussian"}:
            label = r"SE, $M=1225$"
        else:
            label = r"Matern $3/2$, $M=35721$"
        ax.plot(
            profile["rank"],
            profile["cumulative_mass"],
            label=label,
            **style,
        )
    for i, s_act in enumerate((512, 1024, 2048)):
        ax.axvline(
            s_act,
            color=GRAY,
            linestyle=":",
            linewidth=0.8,
            label="candidate top-k: 512, 1024, 2048" if i == 0 else None,
        )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("top-k Fourier modes")
    ax.set_ylabel("cumulative mass")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right", fontsize=6.2, handlelength=1.5, labelspacing=0.28)


def make_active_score_mass_figure(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    del exp_cfg
    plt = _import_pyplot()
    out_dir = Path(viz_cfg.output_dir)
    profiles, rows = _active_score_profile_data(viz_cfg)
    fig, ax = plt.subplots(figsize=_single_col_size(2.35))
    _plot_active_score_profiles(ax, profiles)
    ax.set_title("Active-score cumulative mass")
    paths = _save_figure(fig, out_dir, "active_score_cumulative_mass")
    plt.close(fig)
    csv_path = out_dir / "active_score_cumulative_mass.csv"
    _write_csv(csv_path, rows)
    return {
        "figure_name": "active_score_cumulative_mass",
        "output_csv": str(csv_path),
        **paths,
        "notes": "Kernel/grid-level diagnostic; no large dataset is loaded.",
    }


def _prepare_solver_context(
    cfg: BTABExperimentConfig,
    dataset_payload: dict[str, Any],
    *,
    maxiter: int | None = None,
) -> tuple[Any, Any, Any, EFGPSolver, Any]:
    x_train = np.asarray(dataset_payload["x_train"], dtype=np.float64)
    y_train = np.asarray(dataset_payload["y_train"], dtype=np.float64).reshape(-1)
    solver = EFGPSolver(
        make_kernel(cfg, int(dataset_payload["dim"])),
        reg_lambda=float(cfg.reg_lambda),
        eps=float(cfg.eps),
        nufft_tol=float(cfg.nufft_tol),
        l2scaled=bool(cfg.l2_scaled),
    )
    run_cfg = build_gpu_run_config(cfg, maxiter=maxiter)
    backend = build_gpu_backend_bundle(run_cfg.backend)
    data_ctx = ensure_gpu_data_context(backend, x_train, y_train, state=None)
    data_ctx.meta["debug_finite_checks"] = bool(run_cfg.debug_finite_checks)
    op_ctx = GPUOperatorContext()
    data_ctx = _run_gpu_precompute(
        backend,
        solver,
        run_cfg,
        data_ctx,
        op_ctx,
        use_original_precompute=False,
    )
    return backend, data_ctx, op_ctx, solver, run_cfg


def _active_window_spectrum_data(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    dataset = load_processed_dataset(viz_cfg.residual_dataset_stem)
    cfg = _cfg_for_kernel(exp_cfg, viz_cfg, "matern")
    backend, data_ctx, _op_ctx, _solver, _run_cfg = _prepare_solver_context(cfg, dataset)
    btab_cfg = BTABConfig(
        active_mode="topk",
        active_topk=int(viz_cfg.spectrum_active_topk),
        active_tau=None,
        box_budget=cfg.btab_box_budget,
        solve_mode="boxeig",
        exact_apply_mode="boxeig",
        eig_q=int(viz_cfg.spectrum_boxeig_q),
        eig_tol=float(cfg.btab_eig_tol),
        eig_maxiter=cfg.btab_eig_maxiter,
        eig_ncv=cfg.btab_eig_ncv,
    )
    precond = build_box_eigenpro_preconditioner(
        backend,
        data_ctx,
        float(cfg.reg_lambda),
        btab_cfg,
        q=int(viz_cfg.spectrum_boxeig_q),
        profile_apply_components=False,
    )
    xp = backend.xp
    n_box = int(np.prod(precond.box_shape, dtype=np.int64))
    eye = xp.eye(n_box, dtype=xp.complex128)
    A = _local_box_apply_block(backend, precond, float(cfg.reg_lambda), eye)
    A = 0.5 * (A + A.conj().T)
    raw = np.linalg.eigvalsh(np.asarray(_asnumpy(A), dtype=np.complex128))
    raw = np.sort(np.real(raw))[::-1]
    q = min(max(int(viz_cfg.spectrum_boxeig_q), 0), max(int(raw.size) - 1, 0))
    theta_q1 = max(float(raw[q]), 1e-300) if raw.size else 1e-300
    pre = np.maximum(raw / theta_q1, 1e-300) if raw.size else raw.copy()
    if q > 0:
        pre[:q] = 1.0
    rows = []
    for i, val in enumerate(raw, 1):
        rows.append({"spectrum": "raw_A_BB", "rank": int(i), "eigenvalue": float(val)})
    for i, val in enumerate(pre, 1):
        rows.append({"spectrum": "ideal_box_eigenpro_corrected", "rank": int(i), "eigenvalue": float(val)})
    return {
        "raw": raw,
        "preconditioned": pre,
        "rows": rows,
        "dataset_stem": str(dataset["stem"]),
        "N_train": int(dataset["n_train"]),
        "kernel": "matern",
        "M": int(data_ctx.meta["mtot"]) ** int(data_ctx.meta["dim"]),
        "active_topk": int(viz_cfg.spectrum_active_topk),
        "box_size": int(n_box),
        "btab_eig_q": int(q),
        "corrected_spectrum_kind": "ideal EigenPro-style flattening from raw A_BB eigenvalues",
    }


def _plot_raw_spectrum(ax: Any, raw: np.ndarray) -> None:
    ax.semilogy(np.arange(1, raw.size + 1), np.maximum(raw, 1e-300), color=BLACK, linewidth=1.1)
    ax.set_xlabel("rank")
    ax.set_ylabel("eigenvalue")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)


def _plot_corrected_spectrum(ax: Any, pre: np.ndarray, *, q: int | None = None) -> None:
    clipped = np.maximum(np.asarray(pre, dtype=np.float64), 1e-8)
    ax.semilogy(np.arange(1, clipped.size + 1), clipped, color=RED, linewidth=1.2)
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=0.9)
    if q is not None and int(q) > 0:
        ax.axvline(int(q), color=GRAY, linestyle="--", linewidth=0.9)
    pos = clipped[np.isfinite(clipped) & (clipped > 0)]
    if pos.size:
        ymin = max(1e-8, 10.0 ** np.floor(np.log10(float(np.min(pos)))))
    else:
        ymin = 1e-8
    ymax = max(1.2, float(np.nanmax(clipped)) * 1.05) if clipped.size else 1.2
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("rank")
    ax.set_ylabel(r"eig. of $P_B A_{BB}$")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    ax.text(
        0.60,
        0.86,
        "flattened level",
        transform=ax.transAxes,
        fontsize=6.8,
        color=GRAY,
        ha="left",
        va="center",
    )
    if q is not None and int(q) > 0:
        ax.text(
            0.08,
            0.18,
            r"$q$",
            transform=ax.transAxes,
            fontsize=7,
            color=GRAY,
            ha="left",
            va="center",
        )


def make_active_window_spectrum_figure(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    plt = _import_pyplot()
    out_dir = Path(viz_cfg.output_dir)
    data = _active_window_spectrum_data(exp_cfg, viz_cfg)
    raw = data["raw"]
    pre = data["preconditioned"]
    csv_path = out_dir / "active_window_spectrum_diagnostic.csv"
    _write_csv(csv_path, data["rows"])
    fig, axes = plt.subplots(1, 2, figsize=_double_col_size(2.35), constrained_layout=True)
    _plot_raw_spectrum(axes[0], raw)
    axes[0].set_title("Raw active-window spectrum")
    _plot_corrected_spectrum(axes[1], pre, q=int(data.get("btab_eig_q", 0)))
    axes[1].set_title("Ideal corrected spectrum")
    paths = _save_figure(fig, out_dir, "active_window_spectrum_diagnostic")
    plt.close(fig)
    return {
        "figure_name": "active_window_spectrum_diagnostic",
        "output_csv": str(csv_path),
        **paths,
        **{k: v for k, v in data.items() if k not in {"raw", "preconditioned", "rows"}},
    }


def make_mechanism_diagnostics_figure(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    plt = _import_pyplot()
    out_dir = Path(viz_cfg.output_dir)
    profiles, active_rows = _active_score_profile_data(viz_cfg)
    spectrum = _active_window_spectrum_data(exp_cfg, viz_cfg)
    active_csv = out_dir / "figure1_mechanism_active_score.csv"
    spectrum_csv = out_dir / "figure1_mechanism_spectrum.csv"
    _write_csv(active_csv, active_rows)
    _write_csv(spectrum_csv, spectrum["rows"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.20, wspace=0.35)
    _plot_active_score_profiles(axes[0], profiles)
    axes[0].set_title("Active-score mass")
    _panel_label(axes[0], "(a)")
    _plot_raw_spectrum(axes[1], spectrum["raw"])
    axes[1].set_title("Raw active-window spectrum")
    _panel_label(axes[1], "(b)")
    _plot_corrected_spectrum(axes[2], spectrum["preconditioned"], q=int(spectrum.get("btab_eig_q", 0)))
    axes[2].set_title("Ideal corrected spectrum")
    _panel_label(axes[2], "(c)")
    paths = _save_figure(fig, out_dir, "figure1_mechanism_diagnostics")
    plt.close(fig)
    return {
        "figure_name": "figure1_mechanism_diagnostics",
        "output_active_score_csv": str(active_csv),
        "output_spectrum_csv": str(spectrum_csv),
        **paths,
        **{k: v for k, v in spectrum.items() if k not in {"raw", "preconditioned", "rows"}},
    }


def _load_summary_frames(
    root: Path | None,
    *,
    prefer_group: str | None = None,
) -> tuple[Any | None, list[str]]:
    if root is None:
        return None, []
    try:
        import pandas as pd
    except Exception:
        return None, []
    paths: list[Path] = []
    root = root.resolve()
    if prefer_group:
        group_path = root / str(prefer_group) / "master_summary.csv"
        if group_path.exists():
            paths.append(group_path)
    if not paths:
        if (root / "master_summary.csv").exists():
            paths.append(root / "master_summary.csv")
        for p in sorted(root.glob("*/master_summary.csv")):
            if p not in paths:
                paths.append(p)
    if not paths:
        return None, []
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_summary_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False), [str(p) for p in paths]


def _num(row: Any, key: str, default: float = float("nan")) -> float:
    try:
        val = row.get(key, default)
    except AttributeError:
        val = default
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default


def _select_representative_methods(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    del exp_cfg
    root = Path(viz_cfg.summary_root).resolve() if viz_cfg.summary_root is not None else None
    df, sources = _load_summary_frames(root)
    selected: dict[str, Any] = {"source_summary_files": sources, "methods": {}}
    if df is None or df.empty:
        for key in PAPER_LABELS:
            selected["methods"][key] = {"available": False, "reason": "no master_summary.csv found"}
        return selected
    target = str(viz_cfg.residual_dataset_stem)
    ok = df.copy()
    ok = ok[ok.get("status", "").astype(str).str.lower().isin(["ok", "converged"])]
    ok = ok[ok.get("dataset_stem", "").astype(str) == target]
    ok = ok[ok.get("kernel_family", "").astype(str).str.lower().isin(["matern", "mat", "mat32"])]
    if "n_train" in ok.columns:
        ok = ok[np.isclose(ok["n_train"].astype(float), 100_000_000, rtol=0.0, atol=1.0)]
    plain = ok[ok.get("method", "").astype(str).str.lower().eq("plain_cg")]
    if plain.empty:
        baseline_rmse = float("nan")
    else:
        plain = plain.sort_values("time_total", kind="mergesort")
        baseline_rmse = float(plain.iloc[0].get("rmse_test", float("nan")))

    def _pick(key: str, mask: Any) -> dict[str, Any]:
        sub = ok[mask].copy()
        if sub.empty:
            return {"available": False, "reason": "no successful row in method family"}
        if "cg_relres" in sub.columns:
            sub = sub[sub["cg_relres"].astype(float) <= float(viz_cfg.tol)]
        if math.isfinite(baseline_rmse) and "rmse_test" in sub.columns:
            sub["rmse_ratio_to_plain"] = sub["rmse_test"].astype(float) / max(baseline_rmse, 1e-300)
            sub = sub[sub["rmse_ratio_to_plain"] <= float(viz_cfg.residual_rmse_ratio_tol)]
        if sub.empty:
            return {"available": False, "reason": "no row satisfies residual/RMSE criteria"}
        time_col = "time_total" if "time_total" in sub.columns else "time_solve"
        row = sub.sort_values(time_col, kind="mergesort").iloc[0].to_dict()
        row["paper_label"] = PAPER_LABELS[key]
        row["available"] = True
        if "rmse_ratio_to_plain" not in row and math.isfinite(baseline_rmse):
            row["rmse_ratio_to_plain"] = float(row.get("rmse_test", float("nan"))) / max(baseline_rmse, 1e-300)
        return _json_safe(row)

    method = ok.get("method", "").astype(str).str.lower()
    version = ok.get("version", "").astype(str).str.lower() if "version" in ok.columns else method
    selected["baseline_rmse_test"] = baseline_rmse
    selected["methods"]["plain"] = _pick("plain", method.eq("plain_cg"))
    selected["methods"]["v3"] = _pick("v3", method.str.startswith("eigenpro_pcg"))
    selected["methods"]["active_inverse"] = _pick(
        "active_inverse",
        version.eq("v6_btab") | method.str.startswith("btab_auto") | method.str.startswith("btab_exact"),
    )
    selected["methods"]["boxeig"] = _pick(
        "boxeig",
        version.eq("v7_btab_boxeig") | method.str.startswith("btab_boxeig"),
    )
    return selected


def _series_from_frame(df: Any, name: str, default: Any = "") -> Any:
    if name in df.columns:
        return df[name]
    return df.assign(**{name: default})[name]


def _numeric_column(df: Any, name: str, default: float = float("nan")) -> Any:
    import pandas as pd

    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _first_finite_series(*series: Any) -> Any:
    out = series[0].copy()
    for ser in series[1:]:
        out = out.where(np.isfinite(out), ser)
    return out


def _parse_int_from_method(method: Any, pattern: str) -> int | None:
    import re

    m = re.search(pattern, str(method))
    return int(m.group(1)) if m else None


def _group_c_boxeig_sweep_spec(exp_cfg: BTABExperimentConfig) -> dict[str, Any]:
    cfg = resolve_btab_experiment_route(
        replace(exp_cfg, btab_experiment_route="group_c"),
    )
    pairs = [(int(k), int(q)) for k, q in (cfg.btab_boxeig_topk_q_pairs or [])]
    return {
        "route": "group_c",
        "q_values": sorted({q for _, q in pairs}),
        "configured_active_topk_targets": sorted({k for k, _ in pairs}),
        "configured_active_topk_q_pairs": pairs,
        "scan_variables": ["btab_box_size", "btab_eig_q"],
        "selector_variables": ["btab_active_topk", "btab_eig_q"],
    }


def _filter_group_c_rows(work: Any) -> Any:
    for col in ("btab_route_group", "output_group", "btab_experiment_route"):
        if col in work.columns:
            route = _series_from_frame(work, col).astype(str).str.lower()
            filtered = work[route.eq("group_c")].copy()
            if not filtered.empty:
                return filtered
    return work


def _resolve_actual_box_size_series(work: Any) -> Any:
    size = _numeric_column(work, "btab_box_size")
    size = size.where(np.isfinite(size), _numeric_column(work, "box_size"))
    missing = ~np.isfinite(size)
    if bool(missing.any()) and "box_shape" in work.columns:
        size.loc[missing] = [
            _box_size_from_shape(value)
            for value in work.loc[missing, "box_shape"].tolist()
        ]
    return size


def _prefer_usgs_dataset(work: Any, contains: str) -> Any:
    if work.empty or "dataset_stem" not in work.columns:
        return work
    stems = sorted(_series_from_frame(work, "dataset_stem").astype(str).unique())
    if len(stems) <= 1:
        return work
    needle = str(contains or "").strip().lower()
    if not needle:
        return work
    preferred = [stem for stem in stems if needle in stem.lower()]
    if preferred:
        return work[_series_from_frame(work, "dataset_stem").astype(str).isin(preferred)].copy()
    return work


def _box_size_from_shape(value: Any) -> float:
    import ast
    import re

    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return float("nan")
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [int(v) for v in value]
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return float("nan")
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                vals = [int(v) for v in parsed]
            else:
                vals = [int(parsed)]
        except Exception:
            vals = [int(v) for v in re.findall(r"-?\d+", text)]
    if not vals or any(v <= 0 for v in vals):
        return float("nan")
    return float(np.prod(np.asarray(vals, dtype=np.int64), dtype=np.int64))


def _boxeig_sweep_tables(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    sweep_spec = _group_c_boxeig_sweep_spec(exp_cfg)
    root = Path(viz_cfg.summary_root).resolve() if viz_cfg.summary_root is not None else None
    df, sources = _load_summary_frames(root, prefer_group="group_c")
    if df is None or df.empty:
        return {"available": False, "reason": "no master_summary.csv found", "source_summary_files": sources}

    work_all = _filter_group_c_rows(df.copy())
    if work_all.empty:
        return {
            "available": False,
            "reason": "no group_c rows found in master_summary.csv",
            "source_summary_files": sources,
            "sweep_spec": sweep_spec,
        }
    work_all = work_all[
        _series_from_frame(work_all, "dataset_stem").astype(str).str.contains(
            str(viz_cfg.boxeig_sweep_dataset_contains),
            case=False,
            na=False,
        )
    ]
    work_all = _prefer_usgs_dataset(work_all, str(viz_cfg.boxeig_sweep_dataset_contains))
    work_all = work_all[_series_from_frame(work_all, "kernel_family").astype(str).str.lower().isin(["matern", "mat", "mat32"])]
    work_all = work_all[
        np.isclose(
            _numeric_column(work_all, "n_train").astype(float),
            float(viz_cfg.boxeig_sweep_n_train),
            rtol=0.0,
            atol=1.0,
        )
    ]
    work_ok = work_all.copy()
    if "status" in work_ok.columns:
        work_ok = work_ok[
            _series_from_frame(work_ok, "status").astype(str).str.lower().isin(["ok", "converged"])
        ]

    method = _series_from_frame(work_ok, "method").astype(str).str.lower()
    version = _series_from_frame(work_ok, "version").astype(str).str.lower()
    boxeig = work_ok[method.str.startswith("btab_boxeig") | version.eq("v7_btab_boxeig")].copy()
    if boxeig.empty:
        return {
            "available": False,
            "reason": "no Box-EigenPro rows after group_c filtering",
            "source_summary_files": sources,
            "sweep_spec": sweep_spec,
        }

    boxeig["btab_box_size_num"] = _resolve_actual_box_size_series(boxeig)
    boxeig["btab_eig_q_num"] = _numeric_column(boxeig, "btab_eig_q")
    missing_q = ~np.isfinite(boxeig["btab_eig_q_num"])
    if bool(missing_q.any()):
        boxeig.loc[missing_q, "btab_eig_q_num"] = [
            _parse_int_from_method(method, r"_q(\d+)$") or float("nan")
            for method in boxeig.loc[missing_q, "method"].tolist()
        ]
    boxeig["btab_active_topk_num"] = _numeric_column(boxeig, "btab_active_topk")
    missing_topk = ~np.isfinite(boxeig["btab_active_topk_num"])
    if bool(missing_topk.any()):
        boxeig.loc[missing_topk, "btab_active_topk_num"] = [
            _parse_int_from_method(method, r"topk_(\d+)") or float("nan")
            for method in boxeig.loc[missing_topk, "method"].tolist()
        ]
    if sweep_spec["q_values"]:
        q_values = np.asarray(sweep_spec["q_values"], dtype=np.int64)
        q_mask = np.isfinite(boxeig["btab_eig_q_num"]) & np.isin(
            boxeig["btab_eig_q_num"].where(np.isfinite(boxeig["btab_eig_q_num"]), -1).astype(int),
            q_values,
        )
        boxeig = boxeig[q_mask].copy()
    configured_pairs = {
        (int(k), int(q))
        for k, q in sweep_spec.get("configured_active_topk_q_pairs", [])
    }
    if configured_pairs and bool(np.isfinite(boxeig["btab_active_topk_num"]).any()):
        selector_pairs = list(zip(boxeig["btab_active_topk_num"], boxeig["btab_eig_q_num"]))
        selector_mask = [
            np.isfinite(topk)
            and np.isfinite(q)
            and (int(round(float(topk))), int(round(float(q)))) in configured_pairs
            for topk, q in selector_pairs
        ]
        boxeig = boxeig[selector_mask].copy()

    boxeig["iters"] = _first_finite_series(_numeric_column(boxeig, "cg_iters"), _numeric_column(boxeig, "outer_iters"))
    boxeig["relres"] = _first_finite_series(_numeric_column(boxeig, "cg_relres"), _numeric_column(boxeig, "outer_relres"))
    boxeig["time_precond_build_num"] = _numeric_column(boxeig, "time_precond_build")
    boxeig["time_solve_num"] = _numeric_column(boxeig, "time_solve")
    boxeig["time_total_num"] = _numeric_column(boxeig, "time_total")
    fallback_total = boxeig["time_precond_build_num"].fillna(0.0) + boxeig["time_solve_num"].fillna(0.0)
    boxeig["time_total_num"] = boxeig["time_total_num"].where(np.isfinite(boxeig["time_total_num"]), fallback_total)
    boxeig["rmse_test_num"] = _numeric_column(boxeig, "rmse_test")
    boxeig["converged"] = boxeig["relres"].astype(float) <= float(viz_cfg.tol)
    boxeig = boxeig[
        np.isfinite(boxeig["btab_box_size_num"])
        & np.isfinite(boxeig["btab_eig_q_num"])
        & np.isfinite(boxeig["iters"])
        & np.isfinite(boxeig["time_total_num"])
    ].copy()
    if boxeig.empty:
        return {
            "available": False,
            "reason": "no finite Box-EigenPro rows with actual btab_box_size and q",
            "source_summary_files": sources,
            "sweep_spec": sweep_spec,
        }

    raw_cols = [
        "dataset_stem",
        "method",
        "version",
        "n_train",
        "kernel_family",
        "btab_active_topk_num",
        "btab_box_size_num",
        "btab_eig_q_num",
        "iters",
        "relres",
        "converged",
        "time_precond_build_num",
        "time_solve_num",
        "time_total_num",
        "rmse_test_num",
        "source_summary_file",
    ]
    raw_rows = boxeig[[c for c in raw_cols if c in boxeig.columns]].copy()
    raw_rows["selector_role"] = "configured group_c active_topk/q row"
    raw_rows["scan_role"] = "btab_box_size/q after active-window expansion"

    collapsed_rows: list[dict[str, Any]] = []
    for (box_size, q), group in boxeig.groupby(["btab_box_size_num", "btab_eig_q_num"], sort=True):
        pool = group[group["converged"]].copy()
        if pool.empty:
            pool = group.copy()
        row = pool.sort_values("time_total_num", kind="mergesort").iloc[0]
        active_topk_values = sorted(
            {
                int(round(v))
                for v in group["btab_active_topk_num"].dropna().astype(float)
                if math.isfinite(float(v))
            }
        )
        collapsed_rows.append(
            {
                "row_type": "boxeig",
                "btab_box_size": int(round(float(box_size))),
                "btab_eig_q": int(round(float(q))),
                "btab_active_topk_values": ",".join(str(v) for v in active_topk_values),
                "iters": float(row["iters"]),
                "relres": float(row["relres"]),
                "converged": bool(row["converged"]),
                "time_precond_build": float(row["time_precond_build_num"]) if math.isfinite(float(row["time_precond_build_num"])) else float("nan"),
                "time_solve": float(row["time_solve_num"]) if math.isfinite(float(row["time_solve_num"])) else float("nan"),
                "time_total": float(row["time_total_num"]),
                "rmse_test": float(row["rmse_test_num"]) if math.isfinite(float(row["rmse_test_num"])) else float("nan"),
                "method": str(row.get("method", "")),
                "source_summary_file": str(row.get("source_summary_file", "")),
                "n_source_rows": int(len(group)),
                "collapse_rule": (
                    "fastest converged row per actual btab_box_size and btab_eig_q; "
                    "input btab_active_topk is diagnostic only"
                ),
            }
        )

    def _baseline(label: str, mask: Any, *, source: Any) -> dict[str, Any]:
        sub = source[mask].copy()
        if sub.empty:
            return {"row_type": "baseline", "baseline_label": label, "available": False, "reason": "no rows"}
        sub["iters"] = _first_finite_series(_numeric_column(sub, "cg_iters"), _numeric_column(sub, "outer_iters"))
        sub["relres"] = _first_finite_series(_numeric_column(sub, "cg_relres"), _numeric_column(sub, "outer_relres"))
        sub["time_total_num"] = _numeric_column(sub, "time_total")
        sub["time_solve_num"] = _numeric_column(sub, "time_solve")
        sub["time_precond_build_num"] = _numeric_column(sub, "time_precond_build")
        sub["time_total_num"] = sub["time_total_num"].where(
            np.isfinite(sub["time_total_num"]),
            sub["time_solve_num"].fillna(0.0) + sub["time_precond_build_num"].fillna(0.0),
        )
        sub["rmse_test_num"] = _numeric_column(sub, "rmse_test")
        sub["converged"] = sub["relres"].astype(float) <= float(viz_cfg.tol)
        finite = sub[np.isfinite(sub["iters"]) & np.isfinite(sub["time_total_num"])].copy()
        if finite.empty:
            return {"row_type": "baseline", "baseline_label": label, "available": False, "reason": "no finite rows"}
        pool = finite[finite["converged"]].copy()
        reason = ""
        if pool.empty:
            pool = finite
            reason = "no converged row; plotted as unavailable"
        row = pool.sort_values("time_total_num", kind="mergesort").iloc[0]
        return {
            "row_type": "baseline",
            "baseline_label": label,
            "available": bool(row["converged"]),
            "reason": reason,
            "method": str(row.get("method", "")),
            "iters": float(row["iters"]),
            "relres": float(row["relres"]),
            "converged": bool(row["converged"]),
            "time_total": float(row["time_total_num"]),
            "time_solve": float(row["time_solve_num"]) if math.isfinite(float(row["time_solve_num"])) else float("nan"),
            "time_precond_build": float(row["time_precond_build_num"]) if math.isfinite(float(row["time_precond_build_num"])) else float("nan"),
            "rmse_test": float(row["rmse_test_num"]) if math.isfinite(float(row["rmse_test_num"])) else float("nan"),
            "source_summary_file": str(row.get("source_summary_file", "")),
        }

    work_method = _series_from_frame(work_all, "method").astype(str).str.lower()
    work_version = _series_from_frame(work_all, "version").astype(str).str.lower()
    baseline_rows = [
        _baseline("EFGP-CG", work_method.eq("plain_cg"), source=work_all),
        _baseline("Global EigenPro-style PCG", work_method.str.startswith("eigenpro_pcg"), source=work_all),
        _baseline(
            "Active inverse",
            work_version.eq("v6_btab") | work_method.str.startswith("btab_auto") | work_method.str.startswith("btab_exact"),
            source=work_all,
        ),
    ]
    plot_rows = collapsed_rows + baseline_rows
    best = min(
        (row for row in collapsed_rows if bool(row.get("converged", False))),
        key=lambda row: float(row["time_total"]),
        default=None,
    )
    return {
        "available": True,
        "source_summary_files": sources,
        "sweep_spec": {
            **sweep_spec,
            "observed_btab_box_sizes": sorted(
                {
                    int(round(float(v)))
                    for v in boxeig["btab_box_size_num"].dropna().astype(float)
                    if math.isfinite(float(v))
                }
            ),
            "observed_scan_pairs": [
                [int(round(float(box_size))), int(round(float(q)))]
                for box_size, q in sorted(
                    {
                        (float(row["btab_box_size_num"]), float(row["btab_eig_q_num"]))
                        for _, row in boxeig.iterrows()
                    }
                )
            ],
        },
        "raw_rows": raw_rows.to_dict(orient="records"),
        "collapsed_rows": collapsed_rows,
        "baseline_rows": baseline_rows,
        "plot_rows": plot_rows,
        "best_boxeig": best,
    }


def _plot_boxeig_sweep_rows(plot_rows: list[dict[str, Any]], out_dir: Path, *, stem: str) -> dict[str, str]:
    plt = _import_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.78, top=0.82, bottom=0.18, wspace=0.28)
    box_rows = [row for row in plot_rows if str(row.get("row_type", "")) == "boxeig"]
    baseline_rows = [row for row in plot_rows if str(row.get("row_type", "")) == "baseline"]
    box_sizes = sorted(set(_to_int(row.get("btab_box_size"), 0) for row in box_rows if _to_int(row.get("btab_box_size"), 0) > 0))
    q_values_all = sorted(
        {
            _to_int(row.get("btab_eig_q"), 0)
            for row in box_rows
            if _to_int(row.get("btab_eig_q"), 0) > 0
        }
    )
    marker_x = q_values_all[0] if q_values_all else 0
    box_palette = [PURPLE, RED, BLACK, BLUE]
    markers = ["o", "s", "^", "D"]
    for i, box_size in enumerate(box_sizes):
        part = [row for row in box_rows if _to_int(row.get("btab_box_size"), 0) == box_size]
        part = sorted(part, key=lambda row: _to_int(row.get("btab_eig_q"), 0))
        q = [_to_int(row.get("btab_eig_q"), 0) for row in part]
        iters = [_to_float(row.get("iters")) for row in part]
        times = [_to_float(row.get("time_total")) for row in part]
        label = rf"$|B|={box_size}$"
        color = box_palette[i % len(box_palette)]
        marker = markers[i % len(markers)]
        axes[0].plot(q, iters, color=color, marker=marker, label=label, linewidth=1.25, markersize=4)
        axes[1].plot(q, times, color=color, marker=marker, label=label, linewidth=1.25, markersize=4)

    best_candidates = [
        row for row in box_rows
        if _to_bool(row.get("converged", False)) and math.isfinite(_to_float(row.get("time_total")))
    ]
    if best_candidates:
        best = min(best_candidates, key=lambda row: _to_float(row.get("time_total")))
        q_best = _to_int(best.get("btab_eig_q"), 0)
        axes[0].scatter([q_best], [_to_float(best.get("iters"))], marker="*", s=75, color=BLACK, zorder=5)
        axes[1].scatter([q_best], [_to_float(best.get("time_total"))], marker="*", s=75, color=BLACK, zorder=5, label="best")

    for base in baseline_rows:
        label = str(base.get("baseline_label", "baseline"))
        available = _to_bool(base.get("available", False))
        display_label = label
        if label == "Global EigenPro-style PCG":
            display_label = "global EP-style"
        elif label == "Active inverse":
            display_label = "exact block"
        style = METHOD_STYLES.get(display_label, METHOD_STYLES.get(label, {"color": GRAY, "linestyle": ":", "linewidth": 1.0}))
        if available:
            baseline_iters = _to_float(base.get("iters"))
            baseline_time = _to_float(base.get("time_total"))
            if math.isfinite(baseline_iters):
                axes[0].axhline(baseline_iters, label=display_label, alpha=0.78, **style)
            if math.isfinite(baseline_time):
                axes[1].axhline(baseline_time, label=display_label, alpha=0.78, **style)
        elif label == "Active inverse":
            baseline_iters = _to_float(base.get("iters"))
            baseline_time = _to_float(base.get("time_total"))
            fail_label = "exact block, not conv."
            if math.isfinite(baseline_iters) and marker_x > 0:
                axes[0].scatter(
                    [marker_x],
                    [max(baseline_iters, 1e-12)],
                    marker="x",
                    s=42,
                    color=GRAY,
                    linewidths=1.1,
                    label=fail_label,
                    zorder=6,
                )
            if math.isfinite(baseline_time) and marker_x > 0:
                axes[1].scatter(
                    [marker_x],
                    [baseline_time],
                    marker="x",
                    s=42,
                    color=GRAY,
                    linewidths=1.1,
                    label=fail_label,
                    zorder=6,
                )

    axes[0].set_xlabel("spectral rank q")
    axes[0].set_ylabel("PCG iterations")
    axes[0].set_yscale("log")
    axes[0].set_title("Iterations")
    axes[0].grid(True, which="both", alpha=0.22, linewidth=0.5)
    _panel_label(axes[0], "(a)")
    axes[1].set_xlabel("spectral rank q")
    axes[1].set_ylabel("total time (s)")
    axes[1].set_title("Total time")
    axes[1].grid(True, alpha=0.22, linewidth=0.5)
    _panel_label(axes[1], "(b)")
    fig.suptitle(r"Matern USGS, $N=3\times 10^8$, $M=35721$", fontsize=9)
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        for handle, text in zip(h, lab):
            if text not in labels:
                handles.append(handle)
                labels.append(text)
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(0.80, 0.50),
            ncol=1,
            handlelength=1.8,
            labelspacing=0.45,
            borderaxespad=0.0,
        )
    paths = _save_figure(fig, out_dir, stem)
    plt.close(fig)
    return paths


def make_group_c_boxeig_parameter_sweep_figure(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    out_dir = Path(viz_cfg.output_dir)
    tables = _boxeig_sweep_tables(exp_cfg, viz_cfg)
    if not bool(tables.get("available", False)):
        return {
            "figure_name": "group_c_boxeig_parameter_sweep",
            "available": False,
            "reason": tables.get("reason", "unavailable"),
            "source_summary_files": tables.get("source_summary_files", []),
        }
    raw_csv = out_dir / "group_c_boxeig_sweep_raw.csv"
    collapsed_csv = out_dir / "group_c_boxeig_sweep_collapsed.csv"
    plot_csv = out_dir / "group_c_boxeig_sweep_plot_data.csv"
    _write_csv(raw_csv, tables["raw_rows"])
    _write_csv(collapsed_csv, tables["collapsed_rows"])
    _write_csv(plot_csv, tables["plot_rows"])
    paths = _plot_boxeig_sweep_rows(tables["plot_rows"], out_dir, stem="group_c_boxeig_parameter_sweep")
    return {
        "figure_name": "group_c_boxeig_parameter_sweep",
        "available": True,
        "dataset_filter": str(viz_cfg.boxeig_sweep_dataset_contains),
        "N_train": int(viz_cfg.boxeig_sweep_n_train),
        "kernel": "matern",
        "tol": float(viz_cfg.tol),
        "output_raw_csv": str(raw_csv),
        "output_collapsed_csv": str(collapsed_csv),
        "output_plot_csv": str(plot_csv),
        "best_boxeig": tables.get("best_boxeig"),
        "baselines": tables.get("baseline_rows", []),
        "sweep_spec": tables.get("sweep_spec"),
        "notes": (
            "Group C Box-EigenPro sweep uses actual btab_box_size and btab_eig_q as scanned "
            "variables. Rows are filtered to group_c, q values come from "
            "config.btab_boxeig_topk_q_pairs, and input btab_active_topk is kept only as "
            "diagnostic metadata when multiple selector labels collapse to the same |B|."
        ),
        **paths,
    }


def _make_checkpoints(maxiter: int, count: int) -> set[int]:
    maxiter = max(1, int(maxiter))
    count = max(10, int(count))
    vals = {0, maxiter}
    vals.update(range(1, min(6, maxiter) + 1))
    geom = np.unique(np.round(np.geomspace(1, maxiter, count)).astype(int))
    vals.update(int(v) for v in geom if 0 <= int(v) <= maxiter)
    return vals


def _should_record_residual(iteration: int, viz_cfg: PaperVisualizationConfig) -> bool:
    iteration = int(iteration)
    if iteration <= int(viz_cfg.residual_trace_dense_until):
        return True
    if iteration <= int(viz_cfg.residual_trace_mid_until):
        return iteration % max(1, int(viz_cfg.residual_trace_stride_mid)) == 0
    return iteration % max(1, int(viz_cfg.residual_trace_stride_late)) == 0


class _TraceRecorder:
    def __init__(
        self,
        *,
        method_key: str,
        method_label: str,
        metadata: dict[str, Any],
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        checkpoints: set[int],
        viz_cfg: PaperVisualizationConfig,
    ) -> None:
        self.method_key = method_key
        self.method_label = method_label
        self.metadata = dict(metadata)
        self.x_eval = np.asarray(x_eval, dtype=np.float64)
        self.y_eval = np.asarray(y_eval, dtype=np.float64).reshape(-1)
        self.checkpoints = set(int(v) for v in checkpoints)
        self.viz_cfg = viz_cfg
        self.rows: list[dict[str, Any]] = []
        self.backend = None
        self.data_ctx = None

    def factory(self, backend: Any, data_ctx: Any) -> Callable[[dict[str, Any]], None]:
        self.backend = backend
        self.data_ctx = data_ctx

        def _callback(event: dict[str, Any]) -> None:
            iteration = int(event["iteration"])
            is_checkpoint = iteration in self.checkpoints
            is_final = bool(event.get("is_final", False))
            if not (is_checkpoint or is_final or _should_record_residual(iteration, self.viz_cfg)):
                return
            rmse = float("nan")
            checkpoint_kind = "residual"
            if is_checkpoint:
                yhat = predict_v1(backend, data_ctx, self.x_eval, event["x"])
                yp = _asnumpy(yhat).astype(np.float64, copy=False).reshape(-1)
                rmse = float(np.sqrt(np.mean((yp - self.y_eval) ** 2)))
                checkpoint_kind = "rmse_checkpoint"
            if is_final and checkpoint_kind == "residual":
                checkpoint_kind = "final_residual"
            row = {
                **self.metadata,
                "method_key": self.method_key,
                "method_label": self.method_label,
                "iteration": iteration,
                "elapsed_time": float(event.get("elapsed_time", float("nan"))),
                "relres": float(event.get("relres", float("nan"))),
                "rmse_test_sample": rmse,
                "checkpoint_kind": checkpoint_kind,
                "trace_sampling_policy": (
                    f"every iteration through {int(self.viz_cfg.residual_trace_dense_until)}, "
                    f"every {int(self.viz_cfg.residual_trace_stride_mid)} through "
                    f"{int(self.viz_cfg.residual_trace_mid_until)}, then every "
                    f"{int(self.viz_cfg.residual_trace_stride_late)}; RMSE checkpoints and final always kept"
                ),
            }
            self.rows.append(row)

        return _callback

    def ensure_final_rmse(self, beta_gpu: Any) -> None:
        if not self.rows or self.backend is None or self.data_ctx is None:
            return
        if math.isfinite(float(self.rows[-1].get("rmse_test_sample", float("nan")))):
            return
        yhat = predict_v1(self.backend, self.data_ctx, self.x_eval, beta_gpu)
        yp = _asnumpy(yhat).astype(np.float64, copy=False).reshape(-1)
        self.rows[-1]["rmse_test_sample"] = float(np.sqrt(np.mean((yp - self.y_eval) ** 2)))
        self.rows[-1]["checkpoint_kind"] = "final_rmse_checkpoint"


def _sample_test(dataset: dict[str, Any], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(dataset["x_test"], dtype=np.float64)
    y = np.asarray(dataset["y_test"], dtype=np.float64).reshape(-1)
    if int(x.shape[0]) <= int(n):
        return x, y
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(int(x.shape[0]), size=int(n), replace=False))
    return x[idx], y[idx]


def _btab_cfg_from_row(row: dict[str, Any], *, boxeig: bool) -> BTABConfig:
    active_topk = int(_num(row, "btab_active_topk", _num(row, "active_topk", 2048)))
    eig_q = int(_num(row, "btab_eig_q", _num(row, "eig_q", 128)))
    exact_box_max_size = int(_num(row, "btab_exact_box_max_size", 20000))
    box_budget_val = _num(row, "btab_box_budget", 80000)
    return BTABConfig(
        active_mode="topk",
        active_topk=active_topk,
        active_tau=None,
        box_budget=int(box_budget_val) if math.isfinite(box_budget_val) else None,
        solve_mode="boxeig" if boxeig else str(row.get("btab_solve_mode", "auto")),
        exact_box_max_size=exact_box_max_size,
        exact_apply_mode="boxeig" if boxeig else str(row.get("btab_exact_apply_mode", "inverse")),
        outer_solver=str(row.get("btab_outer_solver", "auto")),
        outer_gmres_restart=int(_num(row, "btab_outer_gmres_restart", 50)),
        inner_tol=float(_num(row, "btab_inner_tol", 1e-3)),
        inner_maxiter=int(_num(row, "btab_inner_maxiter", 50)),
        inner_precond=str(row.get("btab_inner_precond", "diag")),
        eig_q=eig_q,
        eig_tol=float(_num(row, "btab_eig_tol", 1e-3)),
        eig_maxiter=None,
        eig_ncv=None,
        diagnostic_mode="none",
    )


def make_residual_and_rmse_figures(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    plt = _import_pyplot()
    out_dir = Path(viz_cfg.output_dir)
    selected = _select_representative_methods(exp_cfg, viz_cfg)
    dataset = load_processed_dataset(viz_cfg.residual_dataset_stem)
    x_eval, y_eval = _sample_test(dataset, int(viz_cfg.residual_sample_size), int(viz_cfg.residual_sample_seed))
    cfg = _cfg_for_kernel(exp_cfg, viz_cfg, "matern")
    if viz_cfg.residual_maxiter is not None:
        cfg = replace(cfg, maxiter=int(viz_cfg.residual_maxiter), non_v1_maxiter=int(viz_cfg.residual_maxiter))
    solver = EFGPSolver(
        make_kernel(cfg, int(dataset["dim"])),
        reg_lambda=float(cfg.reg_lambda),
        eps=float(cfg.eps),
        nufft_tol=float(cfg.nufft_tol),
        l2scaled=bool(cfg.l2_scaled),
    )
    x_train = np.asarray(dataset["x_train"], dtype=np.float64)
    y_train = np.asarray(dataset["y_train"], dtype=np.float64).reshape(-1)
    all_rows: list[dict[str, Any]] = []
    run_manifest: dict[str, Any] = {"selection": selected, "traced_methods": {}}

    for method_key in ("plain", "v3", "active_inverse", "boxeig"):
        row = selected.get("methods", {}).get(method_key, {})
        if not bool(row.get("available", False)):
            run_manifest["traced_methods"][method_key] = row
            continue
        maxiter = int(_num(row, "maxiter", cfg.maxiter if method_key == "plain" else cfg.non_v1_maxiter))
        run_cfg = build_gpu_run_config(cfg, maxiter=maxiter)
        checkpoints = _make_checkpoints(maxiter, int(viz_cfg.rmse_checkpoint_count))
        metadata = {
            "dataset_stem": str(dataset["stem"]),
            "kernel_name": "matern",
            "N_train": int(dataset["n_train"]),
            "test_sample_size_for_rmse": int(x_eval.shape[0]),
            "test_sample_seed": int(viz_cfg.residual_sample_seed),
            "lambda": float(cfg.reg_lambda),
            "ell": float(cfg.kernel_lengthscale),
            "eps": float(cfg.eps),
            "tol": float(cfg.tol),
            "source_summary_file": row.get("source_summary_file", ""),
            "internal_method": row.get("method", ""),
            "top_q": row.get("top_q", ""),
            "btab_active_topk": row.get("btab_active_topk", row.get("active_topk", "")),
            "btab_eig_q": row.get("btab_eig_q", ""),
        }
        recorder = _TraceRecorder(
            method_key=method_key,
            method_label=PAPER_LABELS[method_key],
            metadata=metadata,
            x_eval=x_eval,
            y_eval=y_eval,
            checkpoints=checkpoints,
            viz_cfg=viz_cfg,
        )
        if method_key == "plain":
            out = run_v1_pure_efgp(
                solver,
                x_train,
                y_train,
                run_cfg,
                trace_callback_factory=recorder.factory,
            )
        elif method_key == "v3":
            q = int(_num(row, "top_q", 64))
            eig_cfg = EigenspaceConfig(
                q_max=q,
                block_size=max(q + 8, q + 1),
                n_iter=3,
                method="subspace_iter",
            )
            out = run_v3_full_gpu_eigenspace(
                solver,
                x_train,
                y_train,
                run_cfg,
                eig_cfg=eig_cfg,
                trace_callback_factory=recorder.factory,
            )
        elif method_key == "active_inverse":
            out = run_v6_box_toeplitz_active_block(
                solver,
                x_train,
                y_train,
                run_cfg,
                btab_cfg=_btab_cfg_from_row(row, boxeig=False),
                trace_callback_factory=recorder.factory,
            )
        else:
            out = run_v7_box_eigenpro_active_block(
                solver,
                x_train,
                y_train,
                run_cfg,
                btab_cfg=_btab_cfg_from_row(row, boxeig=True),
                trace_callback_factory=recorder.factory,
            )
        recorder.ensure_final_rmse(out.beta_gpu)
        all_rows.extend(recorder.rows)
        run_manifest["traced_methods"][method_key] = {
            "available": True,
            "paper_label": PAPER_LABELS[method_key],
            "selected_row": row,
            "diagnostics": _json_safe(out.diagnostics),
            "trace_run_for_visualization": True,
            "not_used_for_main_timing": True,
        }

    trace_csv = out_dir / "fig6_residual_rmse_trace.csv"
    _write_csv(trace_csv, all_rows)
    fig, axes = plt.subplots(1, 2, figsize=_double_col_size(2.35), constrained_layout=True)
    for label in PAPER_LABELS.values():
        part = [r for r in all_rows if r.get("method_label") == label]
        if not part:
            continue
        xs = [r["iteration"] for r in part]
        ys = [max(float(r["relres"]), 1e-300) for r in part]
        style = METHOD_STYLES.get(label, {"color": BLACK, "linestyle": "-", "linewidth": 1.25})
        axes[0].semilogy(xs, ys, label=label, **style)
        zoom = [(x, y) for x, y in zip(xs, ys) if int(x) <= int(viz_cfg.residual_zoom_iterations)]
        if zoom:
            axes[1].semilogy([x for x, _ in zoom], [y for _, y in zoom], label=label, **style)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("relative residual")
    axes[0].set_title("Full range")
    axes[0].axhline(
        float(viz_cfg.tol),
        color=GRAY,
        linestyle=":",
        linewidth=0.9,
        label=rf"$\varepsilon_{{pcg}}={float(viz_cfg.tol):.0e}$",
    )
    axes[1].axhline(float(viz_cfg.tol), color=GRAY, linestyle=":", linewidth=0.9)
    axes[0].grid(True, which="both", alpha=0.22, linewidth=0.5)
    axes[0].legend(frameon=False, loc="best")
    _panel_label(axes[0], "(a)")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("relative residual")
    axes[1].set_title(f"Zoom: first {int(viz_cfg.residual_zoom_iterations)} iterations")
    axes[1].grid(True, which="both", alpha=0.22, linewidth=0.5)
    _panel_label(axes[1], "(b)")
    residual_paths = _save_figure(fig, out_dir, "fig6_residual_convergence")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=_single_col_size(2.35))
    for label in PAPER_LABELS.values():
        part = [
            r for r in all_rows
            if r.get("method_label") == label and math.isfinite(float(r.get("rmse_test_sample", float("nan"))))
        ]
        if not part:
            continue
        style = METHOD_STYLES.get(label, {"color": BLACK, "linestyle": "-", "linewidth": 1.25})
        ax.semilogy(
            [r["iteration"] for r in part],
            [float(r["rmse_test_sample"]) for r in part],
            marker="o",
            markersize=2.4,
            label=label,
            **style,
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel("held-out sample RMSE")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    ax.legend(frameon=False, loc="best")
    rmse_paths = _save_figure(fig, out_dir, "rmse_checkpoint_convergence")
    plt.close(fig)

    return {
        "figure_name": "fig6_residual_and_rmse",
        "dataset_stem": str(dataset["stem"]),
        "N_train": int(dataset["n_train"]),
        "output_csv": str(trace_csv),
        "residual": residual_paths,
        "rmse": rmse_paths,
        **run_manifest,
    }


def _rasterize_points(x: np.ndarray, values: np.ndarray, bins: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x = np.asarray(x, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    xmin, xmax = float(np.min(x[:, 0])), float(np.max(x[:, 0]))
    ymin, ymax = float(np.min(x[:, 1])), float(np.max(x[:, 1]))
    sums, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1], bins=int(bins), range=[[xmin, xmax], [ymin, ymax]], weights=values)
    counts, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=int(bins), range=[[xmin, xmax], [ymin, ymax]])
    with np.errstate(invalid="ignore", divide="ignore"):
        img = sums / counts
    img[counts <= 0] = np.nan
    return img.T, (float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1]))


def make_prediction_error_map(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    plt = _import_pyplot()
    out_dir = Path(viz_cfg.output_dir)
    dataset = load_processed_dataset(viz_cfg.map_dataset_stem)
    cfg = _cfg_for_kernel(exp_cfg, viz_cfg, "matern")
    solver = EFGPSolver(
        make_kernel(cfg, int(dataset["dim"])),
        reg_lambda=float(cfg.reg_lambda),
        eps=float(cfg.eps),
        nufft_tol=float(cfg.nufft_tol),
        l2scaled=bool(cfg.l2_scaled),
    )
    btab_cfg = BTABConfig(
        active_mode="topk",
        active_topk=int(viz_cfg.map_active_topk),
        active_tau=None,
        box_budget=cfg.btab_box_budget,
        solve_mode="auto",
        exact_box_max_size=cfg.btab_exact_box_max_size,
        exact_apply_mode="inverse",
        diagnostic_mode="none",
    )
    out = run_v6_box_toeplitz_active_block(
        solver,
        np.asarray(dataset["x_train"], dtype=np.float64),
        np.asarray(dataset["y_train"], dtype=np.float64).reshape(-1),
        build_gpu_run_config(cfg, maxiter=int(cfg.non_v1_maxiter)),
        btab_cfg=btab_cfg,
    )
    x_eval, y_eval = _sample_test(dataset, int(viz_cfg.map_sample_size), int(viz_cfg.map_sample_seed))
    yhat = _asnumpy(predict_v1(out.backend, out.data_ctx, x_eval, out.beta_gpu)).astype(np.float64, copy=False).reshape(-1)
    err = np.abs(yhat - y_eval)
    truth_img, extent = _rasterize_points(x_eval, y_eval, int(viz_cfg.map_bins))
    pred_img, _ = _rasterize_points(x_eval, yhat, int(viz_cfg.map_bins))
    err_img, _ = _rasterize_points(x_eval, err, int(viz_cfg.map_bins))
    npz_path = out_dir / "usgs_prediction_error_map_rasters.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, truth=truth_img, prediction=pred_img, abs_error=err_img, extent=np.asarray(extent))
    metrics = {
        "rmse_sample": float(np.sqrt(np.mean((yhat - y_eval) ** 2))),
        "mae_sample": float(np.mean(err)),
        "sample_size": int(x_eval.shape[0]),
        "sample_seed": int(viz_cfg.map_sample_seed),
        "diagnostics": _json_safe(out.diagnostics),
    }
    metrics_path = out_dir / "usgs_prediction_error_map_metrics.json"
    _write_json(metrics_path, metrics)
    fig, axes = plt.subplots(1, 3, figsize=_double_col_size(2.35), constrained_layout=True)
    vals = truth_img[np.isfinite(truth_img)]
    vmin, vmax = (float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))) if vals.size else (None, None)
    im0 = axes[0].imshow(truth_img, origin="lower", extent=extent, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground truth")
    _panel_label(axes[0], "(a)")
    im1 = axes[1].imshow(pred_img, origin="lower", extent=extent, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    _panel_label(axes[1], "(b)")
    err_vals = err_img[np.isfinite(err_img)]
    err_vmax = float(np.nanpercentile(err_vals, 99)) if err_vals.size else None
    im2 = axes[2].imshow(err_img, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=err_vmax)
    axes[2].set_title("Absolute error")
    _panel_label(axes[2], "(c)")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im0, ax=axes[:2], shrink=0.78, label="standardized elevation")
    fig.colorbar(im2, ax=axes[2], shrink=0.78, label="absolute error")
    paths = _save_figure(fig, out_dir, "usgs_prediction_error_map")
    plt.close(fig)
    return {
        "figure_name": "usgs_prediction_error_map",
        "dataset_stem": str(dataset["stem"]),
        "N_train": int(dataset["n_train"]),
        "output_npz": str(npz_path),
        "output_metrics_json": str(metrics_path),
        **paths,
    }


def run_paper_visualizations(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    out_dir = Path(viz_cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_cfg = replace(viz_cfg, output_dir=out_dir)
    manifest: dict[str, Any] = {
        "paper_visualizations_dir": str(out_dir),
        "config": asdict(viz_cfg),
        "trace_run_for_visualization": True,
        "not_used_for_main_timing": True,
        "figures": {},
    }
    if bool(viz_cfg.make_mechanism_figure):
        manifest["figures"]["figure1_mechanism_diagnostics"] = make_mechanism_diagnostics_figure(exp_cfg, viz_cfg)
    elif bool(viz_cfg.make_active_score):
        manifest["figures"]["active_score_cumulative_mass"] = make_active_score_mass_figure(exp_cfg, viz_cfg)
    if (not bool(viz_cfg.make_mechanism_figure)) and bool(viz_cfg.make_spectrum):
        manifest["figures"]["active_window_spectrum_diagnostic"] = make_active_window_spectrum_figure(exp_cfg, viz_cfg)
    if bool(viz_cfg.make_residual) or bool(viz_cfg.make_rmse):
        trace_info = make_residual_and_rmse_figures(exp_cfg, viz_cfg)
        if not bool(viz_cfg.make_residual):
            trace_info.pop("residual", None)
        if not bool(viz_cfg.make_rmse):
            trace_info.pop("rmse", None)
        manifest["figures"]["fig6_residual_and_rmse"] = trace_info
    if bool(viz_cfg.make_prediction_map):
        manifest["figures"]["usgs_prediction_error_map"] = make_prediction_error_map(exp_cfg, viz_cfg)
    if bool(viz_cfg.make_boxeig_sweep):
        manifest["figures"]["group_c_boxeig_parameter_sweep"] = make_group_c_boxeig_parameter_sweep_figure(exp_cfg, viz_cfg)
    manifest_path = out_dir / "paper_visualization_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _profiles_from_saved_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        label = str(row.get("profile_label") or row.get("label") or row.get("kernel_family") or "profile")
        family = str(row.get("kernel_family") or label)
        grouped.setdefault((label, family), []).append(row)
    profiles: list[dict[str, Any]] = []
    for (label, family), part in grouped.items():
        part = sorted(part, key=lambda r: _to_int(r.get("rank"), 0))
        rank = np.asarray([_to_int(r.get("rank"), 0) for r in part], dtype=np.int64)
        mass = np.asarray([_to_float(r.get("cumulative_rho_mass")) for r in part], dtype=np.float64)
        keep = (rank > 0) & np.isfinite(mass)
        if not np.any(keep):
            continue
        profiles.append(
            {
                "label": label,
                "kernel_family": family,
                "rank": rank[keep],
                "cumulative_mass": mass[keep],
                "M": int(np.max(rank[keep])),
            }
        )
    return profiles


def _spectrum_from_saved_rows(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, int | None]:
    raw: list[tuple[int, float]] = []
    corrected: list[tuple[int, float]] = []
    for row in rows:
        name = str(row.get("spectrum", "")).strip().lower()
        pair = (_to_int(row.get("rank"), 0), _to_float(row.get("eigenvalue")))
        if pair[0] <= 0 or not math.isfinite(pair[1]):
            continue
        if "raw" in name:
            raw.append(pair)
        elif "preconditioned" in name or "box" in name or "corrected" in name:
            corrected.append(pair)
    raw_arr = np.asarray([v for _, v in sorted(raw)], dtype=np.float64)
    corrected_arr = np.asarray([v for _, v in sorted(corrected)], dtype=np.float64)
    q = None
    if corrected_arr.size:
        near_one = np.isfinite(corrected_arr) & np.isclose(corrected_arr, 1.0, rtol=1e-7, atol=1e-10)
        if bool(np.any(near_one)):
            q = int(np.argmax(~near_one)) if not bool(np.all(near_one)) else int(corrected_arr.size)
            if q <= 0:
                q = int(np.sum(near_one))
    return raw_arr, corrected_arr, q


def _rerender_mechanism_from_saved(out_dir: Path, active_csv: Path, spectrum_csv: Path) -> dict[str, Any]:
    plt = _import_pyplot()
    profiles = _profiles_from_saved_rows(_read_csv_rows(active_csv))
    raw, corrected, q = _spectrum_from_saved_rows(_read_csv_rows(spectrum_csv))
    if not profiles:
        raise ValueError(f"No active-score profiles could be read from {active_csv}")
    if raw.size == 0 or corrected.size == 0:
        raise ValueError(f"No raw/corrected spectrum could be read from {spectrum_csv}")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.20, wspace=0.35)
    _plot_active_score_profiles(axes[0], profiles)
    axes[0].set_title("Active-score mass")
    _panel_label(axes[0], "(a)")
    _plot_raw_spectrum(axes[1], raw)
    axes[1].set_title("Raw active-window spectrum")
    _panel_label(axes[1], "(b)")
    _plot_corrected_spectrum(axes[2], corrected, q=q)
    axes[2].set_title("Ideal corrected spectrum")
    _panel_label(axes[2], "(c)")
    paths = _save_figure(fig, out_dir, "figure1_mechanism_diagnostics")
    plt.close(fig)
    return {
        "figure_name": "figure1_mechanism_diagnostics",
        "source_active_score_csv": str(active_csv),
        "source_spectrum_csv": str(spectrum_csv),
        **paths,
    }


def _rerender_residual_from_saved(
    out_dir: Path,
    trace_csv: Path,
    *,
    zoom_iterations: int = 400,
    tol: float = 1e-6,
) -> dict[str, Any]:
    plt = _import_pyplot()
    rows = _read_csv_rows(trace_csv)
    residual_rows = [
        {
            **row,
            "iteration_int": _to_int(row.get("iteration"), 0),
            "relres_float": _to_float(row.get("relres")),
            "rmse_float": _to_float(row.get("rmse_test_sample")),
        }
        for row in rows
    ]
    residual_rows = [
        row for row in residual_rows
        if row["iteration_int"] >= 0 and math.isfinite(row["relres_float"])
    ]
    if not residual_rows:
        raise ValueError(f"No residual rows could be read from {trace_csv}")

    fig, axes = plt.subplots(1, 2, figsize=_double_col_size(2.35), constrained_layout=True)
    for label in PAPER_LABELS.values():
        part = [row for row in residual_rows if row.get("method_label") == label]
        if not part:
            continue
        part = sorted(part, key=lambda r: r["iteration_int"])
        xs = [row["iteration_int"] for row in part]
        ys = [max(float(row["relres_float"]), 1e-300) for row in part]
        style = METHOD_STYLES.get(label, {"color": BLACK, "linestyle": "-", "linewidth": 1.25})
        axes[0].semilogy(xs, ys, label=label, **style)
        zoom = [(x, y) for x, y in zip(xs, ys) if int(x) <= int(zoom_iterations)]
        if zoom:
            axes[1].semilogy([x for x, _ in zoom], [y for _, y in zoom], label=label, **style)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("relative residual")
    axes[0].set_title("Full range")
    axes[0].axhline(
        float(tol),
        color=GRAY,
        linestyle=":",
        linewidth=0.9,
        label=rf"$\varepsilon_{{pcg}}={float(tol):.0e}$",
    )
    axes[1].axhline(float(tol), color=GRAY, linestyle=":", linewidth=0.9)
    axes[0].grid(True, which="both", alpha=0.22, linewidth=0.5)
    axes[0].legend(frameon=False, loc="best")
    _panel_label(axes[0], "(a)")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("relative residual")
    axes[1].set_title(f"Zoom: first {int(zoom_iterations)} iterations")
    axes[1].grid(True, which="both", alpha=0.22, linewidth=0.5)
    _panel_label(axes[1], "(b)")
    residual_paths = _save_figure(fig, out_dir, "fig6_residual_convergence")
    plt.close(fig)

    rmse_rows = [row for row in residual_rows if math.isfinite(row["rmse_float"])]
    rmse_paths: dict[str, str] | None = None
    if rmse_rows:
        fig, ax = plt.subplots(figsize=_single_col_size(2.35))
        for label in PAPER_LABELS.values():
            part = [row for row in rmse_rows if row.get("method_label") == label]
            if not part:
                continue
            part = sorted(part, key=lambda r: r["iteration_int"])
            style = METHOD_STYLES.get(label, {"color": BLACK, "linestyle": "-", "linewidth": 1.25})
            ax.semilogy(
                [row["iteration_int"] for row in part],
                [row["rmse_float"] for row in part],
                marker="o",
                markersize=2.4,
                label=label,
                **style,
            )
        ax.set_xlabel("iteration")
        ax.set_ylabel("held-out sample RMSE")
        ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
        ax.legend(frameon=False, loc="best")
        rmse_paths = _save_figure(fig, out_dir, "rmse_checkpoint_convergence")
        plt.close(fig)
    return {
        "figure_name": "fig6_residual_and_rmse",
        "source_trace_csv": str(trace_csv),
        "residual": residual_paths,
        "rmse": rmse_paths,
    }


def _rerender_map_from_saved(out_dir: Path, rasters_npz: Path) -> dict[str, Any]:
    plt = _import_pyplot()
    with np.load(rasters_npz) as data:
        truth_img = np.asarray(data["truth"], dtype=np.float64)
        pred_img = np.asarray(data["prediction"], dtype=np.float64)
        err_img = np.asarray(data["abs_error"], dtype=np.float64)
        extent = tuple(float(v) for v in np.asarray(data["extent"], dtype=np.float64).reshape(-1))
    fig, axes = plt.subplots(1, 3, figsize=_double_col_size(2.35), constrained_layout=True)
    vals = truth_img[np.isfinite(truth_img)]
    vmin, vmax = (float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))) if vals.size else (None, None)
    im0 = axes[0].imshow(truth_img, origin="lower", extent=extent, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground truth")
    _panel_label(axes[0], "(a)")
    axes[1].imshow(pred_img, origin="lower", extent=extent, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    _panel_label(axes[1], "(b)")
    err_vals = err_img[np.isfinite(err_img)]
    err_vmax = float(np.nanpercentile(err_vals, 99)) if err_vals.size else None
    im2 = axes[2].imshow(err_img, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=err_vmax)
    axes[2].set_title("Absolute error")
    _panel_label(axes[2], "(c)")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im0, ax=axes[:2], shrink=0.78, label="standardized elevation")
    fig.colorbar(im2, ax=axes[2], shrink=0.78, label="absolute error")
    paths = _save_figure(fig, out_dir, "usgs_prediction_error_map")
    plt.close(fig)
    return {
        "figure_name": "usgs_prediction_error_map",
        "source_rasters_npz": str(rasters_npz),
        **paths,
    }


def _rerender_boxeig_sweep_from_saved(out_dir: Path, plot_csv: Path) -> dict[str, Any]:
    rows = _read_csv_rows(plot_csv)
    if not rows:
        raise ValueError(f"No Box-EigenPro sweep rows could be read from {plot_csv}")
    paths = _plot_boxeig_sweep_rows(rows, out_dir, stem="group_c_boxeig_parameter_sweep")
    return {
        "figure_name": "group_c_boxeig_parameter_sweep",
        "source_plot_csv": str(plot_csv),
        **paths,
    }


def rerender_paper_visualizations_from_saved_data(
    viz_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    zoom_iterations: int = 400,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Recreate paper figures from saved CSV/NPZ artifacts without running solvers.

    This path intentionally avoids loading large datasets, creating cufinufft
    plans, or touching the GPU. It is for visual restyling after the expensive
    trace/spectrum/map artifacts have already been generated.
    """

    source_dir = Path(viz_dir).resolve()
    out_dir = Path(output_dir).resolve() if output_dir is not None else source_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source_dir": str(source_dir),
        "output_dir": str(out_dir),
        "rerender_only": True,
        "uses_saved_plot_data": True,
        "does_not_run_solver": True,
        "does_not_load_large_training_data": True,
        "source_manifest": _read_json_if_exists(source_dir / "paper_visualization_manifest.json"),
        "figures": {},
        "skipped": {},
    }

    active_csv = _find_existing(
        source_dir,
        ["figure1_mechanism_active_score.csv", "active_score_cumulative_mass.csv"],
    )
    spectrum_csv = _find_existing(
        source_dir,
        ["figure1_mechanism_spectrum.csv", "active_window_spectrum_diagnostic.csv"],
    )
    if active_csv is not None and spectrum_csv is not None:
        manifest["figures"]["figure1_mechanism_diagnostics"] = _rerender_mechanism_from_saved(
            out_dir,
            active_csv,
            spectrum_csv,
        )
    else:
        manifest["skipped"]["figure1_mechanism_diagnostics"] = {
            "active_csv_found": active_csv is not None,
            "spectrum_csv_found": spectrum_csv is not None,
        }

    trace_csv = _find_existing(source_dir, ["fig6_residual_rmse_trace.csv"])
    if trace_csv is not None:
        manifest["figures"]["fig6_residual_and_rmse"] = _rerender_residual_from_saved(
            out_dir,
            trace_csv,
            zoom_iterations=int(zoom_iterations),
            tol=float(tol),
        )
    else:
        manifest["skipped"]["fig6_residual_and_rmse"] = {"trace_csv_found": False}

    rasters_npz = _find_existing(source_dir, ["usgs_prediction_error_map_rasters.npz"])
    if rasters_npz is not None:
        manifest["figures"]["usgs_prediction_error_map"] = _rerender_map_from_saved(out_dir, rasters_npz)
    else:
        manifest["skipped"]["usgs_prediction_error_map"] = {"rasters_npz_found": False}

    boxeig_sweep_csv = _find_existing(source_dir, ["group_c_boxeig_sweep_plot_data.csv"])
    if boxeig_sweep_csv is not None:
        manifest["figures"]["group_c_boxeig_parameter_sweep"] = _rerender_boxeig_sweep_from_saved(
            out_dir,
            boxeig_sweep_csv,
        )
    else:
        manifest["skipped"]["group_c_boxeig_parameter_sweep"] = {"plot_csv_found": False}

    manifest_path = out_dir / "paper_visualization_rerender_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
