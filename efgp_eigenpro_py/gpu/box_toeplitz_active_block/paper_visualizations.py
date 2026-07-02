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
    apply_box_eigenpro_local,
    build_box_eigenpro_preconditioner,
)
from .config import BTABConfig, BTABExperimentConfig
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
    spectrum_active_topk: int = 2048
    spectrum_boxeig_q: int = 256
    map_active_topk: int = 4096
    map_sample_size: int = 200_000
    map_sample_seed: int = 23
    map_bins: int = 256
    make_active_score: bool = True
    make_spectrum: bool = True
    make_residual: bool = True
    make_rmse: bool = True
    make_prediction_map: bool = True


PAPER_LABELS = {
    "plain": "EFGP-CG",
    "v3": "Global EigenPro-style PCG",
    "active_inverse": "Active inverse",
    "boxeig": "Box-EigenPro",
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


def _save_figure(fig: Any, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return {"png": str(png), "pdf": str(pdf)}


def _import_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

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


def make_active_score_mass_figure(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    del exp_cfg
    plt = _import_pyplot()
    out_dir = Path(viz_cfg.output_dir)
    rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
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
        ax.plot(rank, cum, label=f"{label} (mtot={grid.mtot})")
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
    for s_act in (512, 1024, 2048):
        ax.axvline(s_act, color="0.72", linestyle="--", linewidth=0.9)
        ax.text(s_act, 0.04, f"{s_act}", rotation=90, va="bottom", ha="right", fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("top-k Fourier modes")
    ax.set_ylabel("normalized cumulative active-score mass")
    ax.set_title("Active-score profiles for the Fourier grids used in the experiments")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
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


def make_active_window_spectrum_figure(
    exp_cfg: BTABExperimentConfig,
    viz_cfg: PaperVisualizationConfig,
) -> dict[str, Any]:
    plt = _import_pyplot()
    out_dir = Path(viz_cfg.output_dir)
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
    PA = apply_box_eigenpro_local(backend, precond, A)
    raw = np.linalg.eigvalsh(np.asarray(_asnumpy(A), dtype=np.complex128))
    pre = np.linalg.eigvals(np.asarray(_asnumpy(PA), dtype=np.complex128))
    raw = np.sort(np.real(raw))[::-1]
    pre = np.sort(np.real(pre))[::-1]
    rows = []
    for i, val in enumerate(raw, 1):
        rows.append({"spectrum": "raw_A_BB", "rank": int(i), "eigenvalue": float(val)})
    for i, val in enumerate(pre, 1):
        rows.append({"spectrum": "box_eigenpro_preconditioned", "rank": int(i), "eigenvalue": float(val)})
    csv_path = out_dir / "active_window_spectrum_diagnostic.csv"
    _write_csv(csv_path, rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].semilogy(np.arange(1, raw.size + 1), np.maximum(raw, 1e-300), color="#1f77b4")
    axes[0].set_title("Raw active-window spectrum")
    axes[0].set_xlabel("rank")
    axes[0].set_ylabel("eigenvalue")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].plot(np.arange(1, pre.size + 1), pre, color="#d62728", linewidth=1.2)
    axes[1].axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, label="exact inverse reference")
    axes[1].set_title("Box-EigenPro active-window spectrum")
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("eigenvalue of P_B A_BB")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Active-window spectrum diagnostic (Matern USGS N=1e8)")
    paths = _save_figure(fig, out_dir, "active_window_spectrum_diagnostic")
    plt.close(fig)
    return {
        "figure_name": "active_window_spectrum_diagnostic",
        "dataset_stem": str(dataset["stem"]),
        "N_train": int(dataset["n_train"]),
        "kernel": "matern",
        "M": int(data_ctx.meta["mtot"]) ** int(data_ctx.meta["dim"]),
        "active_topk": int(viz_cfg.spectrum_active_topk),
        "box_size": int(n_box),
        "btab_eig_q": int(viz_cfg.spectrum_boxeig_q),
        "output_csv": str(csv_path),
        **paths,
    }


def _load_summary_frames(root: Path | None) -> tuple[Any | None, list[str]]:
    if root is None:
        return None, []
    try:
        import pandas as pd
    except Exception:
        return None, []
    paths: list[Path] = []
    root = root.resolve()
    if (root / "master_summary.csv").exists():
        paths.append(root / "master_summary.csv")
    for p in sorted(root.glob("*/master_summary.csv")):
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


def _make_checkpoints(maxiter: int, count: int) -> set[int]:
    maxiter = max(1, int(maxiter))
    count = max(10, int(count))
    vals = {0, maxiter}
    vals.update(range(1, min(6, maxiter) + 1))
    geom = np.unique(np.round(np.geomspace(1, maxiter, count)).astype(int))
    vals.update(int(v) for v in geom if 0 <= int(v) <= maxiter)
    return vals


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
    ) -> None:
        self.method_key = method_key
        self.method_label = method_label
        self.metadata = dict(metadata)
        self.x_eval = np.asarray(x_eval, dtype=np.float64)
        self.y_eval = np.asarray(y_eval, dtype=np.float64).reshape(-1)
        self.checkpoints = set(int(v) for v in checkpoints)
        self.rows: list[dict[str, Any]] = []
        self.backend = None
        self.data_ctx = None

    def factory(self, backend: Any, data_ctx: Any) -> Callable[[dict[str, Any]], None]:
        self.backend = backend
        self.data_ctx = data_ctx

        def _callback(event: dict[str, Any]) -> None:
            iteration = int(event["iteration"])
            rmse = float("nan")
            checkpoint_kind = "residual"
            if iteration in self.checkpoints:
                yhat = predict_v1(backend, data_ctx, self.x_eval, event["x"])
                yp = _asnumpy(yhat).astype(np.float64, copy=False).reshape(-1)
                rmse = float(np.sqrt(np.mean((yp - self.y_eval) ** 2)))
                checkpoint_kind = "rmse_checkpoint"
            row = {
                **self.metadata,
                "method_key": self.method_key,
                "method_label": self.method_label,
                "iteration": iteration,
                "elapsed_time": float(event.get("elapsed_time", float("nan"))),
                "relres": float(event.get("relres", float("nan"))),
                "rmse_test_sample": rmse,
                "checkpoint_kind": checkpoint_kind,
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
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label in PAPER_LABELS.values():
        part = [r for r in all_rows if r.get("method_label") == label]
        if not part:
            continue
        ax.semilogy([r["iteration"] for r in part], [max(float(r["relres"]), 1e-300) for r in part], label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative residual")
    ax.set_title("Residual convergence on Matern USGS N=1e8")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    residual_paths = _save_figure(fig, out_dir, "fig6_residual_convergence")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label in PAPER_LABELS.values():
        part = [
            r for r in all_rows
            if r.get("method_label") == label and math.isfinite(float(r.get("rmse_test_sample", float("nan"))))
        ]
        if not part:
            continue
        ax.semilogy([r["iteration"] for r in part], [float(r["rmse_test_sample"]) for r in part], marker="o", markersize=2.8, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("held-out sample RMSE")
    ax.set_title("RMSE checkpoints on Matern USGS N=1e8")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
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
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    vals = truth_img[np.isfinite(truth_img)]
    vmin, vmax = (float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))) if vals.size else (None, None)
    im0 = axes[0].imshow(truth_img, origin="lower", extent=extent, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[0].set_title("ground truth")
    im1 = axes[1].imshow(pred_img, origin="lower", extent=extent, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[1].set_title("preconditioned prediction")
    err_vals = err_img[np.isfinite(err_img)]
    err_vmax = float(np.nanpercentile(err_vals, 99)) if err_vals.size else None
    im2 = axes[2].imshow(err_img, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=err_vmax)
    axes[2].set_title("absolute error")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im0, ax=axes[:2], shrink=0.78, label="standardized elevation")
    fig.colorbar(im2, ax=axes[2], shrink=0.78, label="absolute error")
    fig.suptitle("USGS prediction/error map (Matern)")
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
    if bool(viz_cfg.make_active_score):
        manifest["figures"]["active_score_cumulative_mass"] = make_active_score_mass_figure(exp_cfg, viz_cfg)
    if bool(viz_cfg.make_spectrum):
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
    manifest_path = out_dir / "paper_visualization_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
