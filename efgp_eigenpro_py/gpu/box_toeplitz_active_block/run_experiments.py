from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ...efgp_solver import EFGPSolver
from ...kernels import make_matern, make_squared_exponential
from ..v1_ops import predict_v1
from ..v3_eigenspace import EigenspaceConfig
from ..versions import (
    GPURunConfig,
    run_v1_pure_efgp,
    run_v3_full_gpu_eigenspace,
    run_v6_box_toeplitz_active_block,
)
from .active_set import format_box_tag
from .config import BTABConfig, BTABExperimentConfig


_HERE = Path(__file__).resolve().parent
_PROCESSED_DIR = _HERE.parent / "benchmark_dataset" / "processed"


def discover_processed_datasets() -> dict[str, Path]:
    return {p.stem: p for p in sorted(_PROCESSED_DIR.glob("*.npz"))}


def load_processed_dataset(stem: str) -> dict[str, Any]:
    dataset_map = discover_processed_datasets()
    if stem not in dataset_map:
        raise FileNotFoundError(
            f"Unknown dataset stem {stem!r}. Available: {', '.join(sorted(dataset_map))}"
        )
    npz_path = dataset_map[stem]
    meta_path = npz_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    loaded = np.load(npz_path)
    x_train = np.asarray(loaded["x_train"], dtype=np.float64)
    y_train = np.asarray(loaded["y_train"], dtype=np.float64).reshape(-1)
    x_test = np.asarray(loaded["x_test"], dtype=np.float64)
    y_test = np.asarray(loaded["y_test"], dtype=np.float64).reshape(-1)
    return {
        "stem": stem,
        "path": str(npz_path),
        "meta_path": str(meta_path),
        "meta": meta,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "dim": int(x_train.shape[1]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
    }


def make_kernel(cfg: BTABExperimentConfig, dim: int):
    family = str(cfg.kernel_family).strip().lower()
    if family in ("matern", "mat", "mat32", "mat52"):
        return make_matern(
            nu=float(cfg.kernel_nu),
            lengthscale=float(cfg.kernel_lengthscale),
            dim=int(dim),
            variance=float(cfg.kernel_variance),
        )
    if family in ("squared_exponential", "se", "rbf", "gaussian"):
        return make_squared_exponential(
            lengthscale=float(cfg.kernel_lengthscale),
            dim=int(dim),
            variance=float(cfg.kernel_variance),
        )
    raise ValueError(f"unknown kernel_family={cfg.kernel_family!r}")


def _backend_to_numpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "get"):
        return np.asarray(arr.get())
    return np.asarray(arr)


def regression_rmse(y_true: np.ndarray, y_pred: Any) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = _backend_to_numpy(y_pred).astype(np.float64, copy=False).reshape(-1)
    mse = float(np.mean((yp - yt) ** 2))
    return float(math.sqrt(max(mse, 0.0)))


def evaluate_output(out: Any, x_eval: np.ndarray, y_eval: np.ndarray) -> float:
    yhat = predict_v1(out.backend, out.data_ctx, x_eval, out.beta_gpu)
    return regression_rmse(y_eval, yhat)


def build_gpu_run_config(cfg: BTABExperimentConfig) -> GPURunConfig:
    return GPURunConfig(
        reg_lambda=float(cfg.reg_lambda),
        tol=float(cfg.tol),
        maxiter=int(cfg.maxiter),
        chunk_size=cfg.chunk_size,
        debug_finite_checks=bool(cfg.debug_finite_checks),
        profile_components=bool(cfg.profile_components),
        backend=cfg.backend,
    )


def method_rows_for_dataset(
    cfg: BTABExperimentConfig,
    dataset_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    x_train = np.asarray(dataset_payload["x_train"], dtype=np.float64)
    y_train = np.asarray(dataset_payload["y_train"], dtype=np.float64).reshape(-1)
    x_test = np.asarray(dataset_payload["x_test"], dtype=np.float64)
    y_test = np.asarray(dataset_payload["y_test"], dtype=np.float64).reshape(-1)
    solver = EFGPSolver(
        make_kernel(cfg, dataset_payload["dim"]),
        reg_lambda=float(cfg.reg_lambda),
        eps=float(cfg.eps),
        nufft_tol=float(cfg.nufft_tol),
        l2scaled=bool(cfg.l2_scaled),
    )
    gpu_cfg = build_gpu_run_config(cfg)

    rows: list[dict[str, Any]] = []

    def _run_case(tag: str, fn, *, extra: dict[str, Any] | None = None) -> None:
        extra = extra or {}
        base = {
            "dataset_stem": dataset_payload["stem"],
            "dataset_dim": int(dataset_payload["dim"]),
            "n_train": int(dataset_payload["n_train"]),
            "n_test": int(dataset_payload["n_test"]),
            "kernel_family": str(cfg.kernel_family),
            "kernel_lengthscale": float(cfg.kernel_lengthscale),
            "kernel_nu": float(cfg.kernel_nu),
            "kernel_variance": float(cfg.kernel_variance),
            "reg_lambda": float(cfg.reg_lambda),
            "eps": float(cfg.eps),
            "tol": float(cfg.tol),
            "maxiter": int(cfg.maxiter),
            "method": tag,
            **extra,
        }
        try:
            out = fn()
            diag = dict(out.diagnostics)
            row = {
                **base,
                **diag,
                "rmse_train": float(evaluate_output(out, x_train, y_train)),
                "rmse_test": float(evaluate_output(out, x_test, y_test)),
                "status": str(diag.get("status", "ok")),
            }
        except Exception as exc:
            row = {
                **base,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        rows.append(row)

    _run_case("plain_cg", lambda: run_v1_pure_efgp(solver, x_train, y_train, gpu_cfg))

    for top_q in cfg.eigenpro_topq_list:
        eig_cfg = EigenspaceConfig(
            q_max=int(top_q),
            block_size=max(int(top_q) + 8, int(top_q) + 1),
            n_iter=3,
            method="subspace_iter",
        )
        _run_case(
            f"eigenpro_pcg_q{int(top_q)}",
            lambda eig_cfg=eig_cfg: run_v3_full_gpu_eigenspace(
                solver,
                x_train,
                y_train,
                gpu_cfg,
                eig_cfg=eig_cfg,
            ),
            extra={"top_q": int(top_q)},
        )

    if str(cfg.btab_active_mode).lower() == "topk":
        configs = [
            BTABConfig(
                active_mode="topk",
                active_topk=int(k),
                active_tau=None,
                box_budget=cfg.btab_box_budget,
                solve_mode=str(cfg.btab_solve_mode),
                exact_box_max_size=cfg.btab_exact_box_max_size,
                exact_apply_mode=str(cfg.btab_exact_apply_mode),
                outer_solver=str(cfg.btab_outer_solver),
                outer_gmres_restart=int(cfg.btab_outer_gmres_restart),
                inner_tol=float(cfg.btab_inner_tol),
                inner_maxiter=int(cfg.btab_inner_maxiter),
                inner_precond=str(cfg.btab_inner_precond),
                keep_box_matrix=bool(cfg.btab_keep_box_matrix),
            )
            for k in cfg.btab_topk_list
        ]
    else:
        configs = [
            BTABConfig(
                active_mode="tau",
                active_topk=None,
                active_tau=float(tau),
                box_budget=cfg.btab_box_budget,
                solve_mode=str(cfg.btab_solve_mode),
                exact_box_max_size=cfg.btab_exact_box_max_size,
                exact_apply_mode=str(cfg.btab_exact_apply_mode),
                outer_solver=str(cfg.btab_outer_solver),
                outer_gmres_restart=int(cfg.btab_outer_gmres_restart),
                inner_tol=float(cfg.btab_inner_tol),
                inner_maxiter=int(cfg.btab_inner_maxiter),
                inner_precond=str(cfg.btab_inner_precond),
                keep_box_matrix=bool(cfg.btab_keep_box_matrix),
            )
            for tau in cfg.btab_tau_list
        ]
    for btab_cfg in configs:
        tag = format_box_tag(
            type("BoxTag", (), {
                "active_mode": btab_cfg.active_mode,
                "active_topk": btab_cfg.active_topk,
                "active_tau": btab_cfg.active_tau,
            })()
        )
        _run_case(
            f"btab_{str(btab_cfg.solve_mode).lower()}_{tag}",
            lambda btab_cfg=btab_cfg: run_v6_box_toeplitz_active_block(
                solver,
                x_train,
                y_train,
                gpu_cfg,
                btab_cfg=btab_cfg,
            ),
            extra={
                "btab_active_mode": str(btab_cfg.active_mode),
                "btab_active_topk": (
                    None if btab_cfg.active_topk is None else int(btab_cfg.active_topk)
                ),
                "btab_active_tau": (
                    None if btab_cfg.active_tau is None else float(btab_cfg.active_tau)
                ),
                "btab_box_budget": (
                    None if btab_cfg.box_budget is None else int(btab_cfg.box_budget)
                ),
                "btab_solve_mode": str(btab_cfg.solve_mode),
                "btab_exact_box_max_size": (
                    None
                    if btab_cfg.exact_box_max_size is None
                    else int(btab_cfg.exact_box_max_size)
                ),
                "btab_exact_apply_mode": str(btab_cfg.exact_apply_mode),
                "btab_outer_solver": str(btab_cfg.outer_solver),
                "btab_outer_gmres_restart": int(btab_cfg.outer_gmres_restart),
                "btab_inner_tol": float(btab_cfg.inner_tol),
                "btab_inner_maxiter": int(btab_cfg.inner_maxiter),
                "btab_inner_precond": str(btab_cfg.inner_precond),
            },
        )
    return rows


def write_rows(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    config_path = out_dir / "experiment_config.json"
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return None


def run_experiments(cfg: BTABExperimentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BTABExperimentConfig()
    out_dir = cfg.resolve_output_dir(_HERE)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for stem in cfg.dataset_stems:
        payload = load_processed_dataset(stem)
        rows.extend(method_rows_for_dataset(cfg, payload))
        write_rows(rows, out_dir)
    (out_dir / "experiment_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "output_dir": str(out_dir),
        "rows": rows,
    }


def _parse_int_list(raw: str) -> list[int]:
    return [int(tok.strip()) for tok in raw.split(",") if tok.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(tok.strip()) for tok in raw.split(",") if tok.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Box-Toeplitz Active Block GPU experiments.")
    p.add_argument("--dataset-stems", type=str, default=",".join(BTABExperimentConfig().dataset_stems))
    p.add_argument("--kernel-family", type=str, default=BTABExperimentConfig().kernel_family)
    p.add_argument("--kernel-lengthscale", type=float, default=BTABExperimentConfig().kernel_lengthscale)
    p.add_argument("--kernel-nu", type=float, default=BTABExperimentConfig().kernel_nu)
    p.add_argument("--kernel-variance", type=float, default=BTABExperimentConfig().kernel_variance)
    p.add_argument("--reg-lambda", type=float, default=BTABExperimentConfig().reg_lambda)
    p.add_argument("--eps", type=float, default=BTABExperimentConfig().eps)
    p.add_argument("--nufft-tol", type=float, default=BTABExperimentConfig().nufft_tol)
    p.add_argument("--tol", type=float, default=BTABExperimentConfig().tol)
    p.add_argument("--maxiter", type=int, default=BTABExperimentConfig().maxiter)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--eigenpro-topq-list", type=str, default="45,90")
    p.add_argument("--btab-active-mode", type=str, default=BTABExperimentConfig().btab_active_mode)
    p.add_argument("--btab-topk-list", type=str, default="512,1024,2048")
    p.add_argument("--btab-tau-list", type=str, default="1e-1,1e-2")
    p.add_argument("--btab-box-budget", type=int, default=BTABExperimentConfig().btab_box_budget)
    p.add_argument("--btab-solve-mode", type=str, default=BTABExperimentConfig().btab_solve_mode)
    p.add_argument("--btab-exact-box-max-size", type=int, default=BTABExperimentConfig().btab_exact_box_max_size)
    p.add_argument("--btab-exact-apply-mode", type=str, default=BTABExperimentConfig().btab_exact_apply_mode)
    p.add_argument("--btab-outer-solver", type=str, default=BTABExperimentConfig().btab_outer_solver)
    p.add_argument("--btab-outer-gmres-restart", type=int, default=BTABExperimentConfig().btab_outer_gmres_restart)
    p.add_argument("--btab-inner-tol", type=float, default=BTABExperimentConfig().btab_inner_tol)
    p.add_argument("--btab-inner-maxiter", type=int, default=BTABExperimentConfig().btab_inner_maxiter)
    p.add_argument("--btab-inner-precond", type=str, default=BTABExperimentConfig().btab_inner_precond)
    p.add_argument("--output-dir", type=str, default="")
    return p


def config_from_args(args: argparse.Namespace) -> BTABExperimentConfig:
    return BTABExperimentConfig(
        dataset_stems=[tok.strip() for tok in args.dataset_stems.split(",") if tok.strip()],
        kernel_family=str(args.kernel_family),
        kernel_lengthscale=float(args.kernel_lengthscale),
        kernel_nu=float(args.kernel_nu),
        kernel_variance=float(args.kernel_variance),
        reg_lambda=float(args.reg_lambda),
        eps=float(args.eps),
        nufft_tol=float(args.nufft_tol),
        tol=float(args.tol),
        maxiter=int(args.maxiter),
        chunk_size=args.chunk_size,
        eigenpro_topq_list=_parse_int_list(args.eigenpro_topq_list),
        btab_active_mode=str(args.btab_active_mode),
        btab_topk_list=_parse_int_list(args.btab_topk_list),
        btab_tau_list=_parse_float_list(args.btab_tau_list),
        btab_box_budget=args.btab_box_budget,
        btab_solve_mode=str(args.btab_solve_mode),
        btab_exact_box_max_size=args.btab_exact_box_max_size,
        btab_exact_apply_mode=str(args.btab_exact_apply_mode),
        btab_outer_solver=str(args.btab_outer_solver),
        btab_outer_gmres_restart=int(args.btab_outer_gmres_restart),
        btab_inner_tol=float(args.btab_inner_tol),
        btab_inner_maxiter=int(args.btab_inner_maxiter),
        btab_inner_precond=str(args.btab_inner_precond),
        output_dir=str(args.output_dir or ""),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    res = run_experiments(cfg)
    print(f"wrote results to: {res['output_dir']}")


if __name__ == "__main__":
    main()
