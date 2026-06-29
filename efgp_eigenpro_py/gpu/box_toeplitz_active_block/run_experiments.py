from __future__ import annotations

import argparse
import csv
import json
import math
import re
import traceback
from dataclasses import asdict, replace
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
    run_v7_box_eigenpro_active_block,
    run_v6_box_toeplitz_active_block,
)
from .active_set import format_box_tag
from .config import (
    BTABConfig,
    BTABExperimentConfig,
    expand_btab_experiment_routes,
    normalize_btab_experiment_route,
    resolve_btab_experiment_route,
)


_HERE = Path(__file__).resolve().parent
_PROCESSED_DIR = _HERE.parent / "benchmark_dataset" / "processed"
_STEM_SIZE_SUFFIX_RE = re.compile(r"(?:_ntrain\d+|_n\d+)$", flags=re.IGNORECASE)


def discover_processed_datasets() -> dict[str, Path]:
    return {p.stem: p for p in sorted(_PROCESSED_DIR.glob("*.npz"))}


def _stem_family_prefix(stem: str) -> str:
    return _STEM_SIZE_SUFFIX_RE.sub("", str(stem).strip())


def discover_processed_dataset_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stem, npz_path in discover_processed_datasets().items():
        meta_path = npz_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        shapes = meta.get("shapes", {}) if isinstance(meta, dict) else {}
        n_train = shapes.get("n_train", meta.get("n_train"))
        n_test = shapes.get("n_test", meta.get("n_test"))
        dim = shapes.get("dim", meta.get("dim"))
        if n_train is None or n_test is None or dim is None:
            loaded = np.load(npz_path)
            try:
                x_train = np.asarray(loaded["x_train"])
                x_test = np.asarray(loaded["x_test"])
                n_train = int(x_train.shape[0]) if n_train is None else int(n_train)
                n_test = int(x_test.shape[0]) if n_test is None else int(n_test)
                dim = int(x_train.shape[1]) if dim is None else int(dim)
            finally:
                loaded.close()
        records.append(
            {
                "stem": stem,
                "path": str(npz_path),
                "meta_path": str(meta_path),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "dim": int(dim),
                "family_prefix": _stem_family_prefix(stem),
            }
        )
    return records


def _pick_matching_stem(candidates: list[dict[str, Any]], prefix: str, n_train: int) -> str:
    preferred_names = [
        f"{prefix}_ntrain{int(n_train)}",
        f"{prefix}_n{int(n_train)}",
    ]
    for name in preferred_names:
        exact = [rec["stem"] for rec in candidates if rec["stem"] == name]
        if len(exact) == 1:
            return str(exact[0])
    if len(candidates) == 1:
        return str(candidates[0]["stem"])
    candidate_names = ", ".join(sorted(str(rec["stem"]) for rec in candidates))
    raise ValueError(
        f"Ambiguous dataset match for family {prefix!r} and n_train={int(n_train)}. "
        f"Candidates: {candidate_names}"
    )


def _n_train_from_sidecar_meta(meta: dict[str, Any]) -> int | None:
    if not isinstance(meta, dict):
        return None
    for path in (
        ("generation", "n_train"),
        ("shapes", "n_train"),
        ("n_train",),
    ):
        obj: Any = meta
        ok = True
        for key in path:
            if not isinstance(obj, dict) or key not in obj:
                ok = False
                break
            obj = obj[key]
        if ok and obj is not None:
            return int(obj)
    return None


def _json_sidecar_only_n_trains(family_prefix: str) -> list[int]:
    values: list[int] = []
    for json_path in sorted(_PROCESSED_DIR.glob("*.json")):
        stem = json_path.stem
        if _stem_family_prefix(stem) != family_prefix:
            continue
        if (_PROCESSED_DIR / f"{stem}.npz").exists():
            continue
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_train = _n_train_from_sidecar_meta(meta)
        if n_train is not None:
            values.append(int(n_train))
    return sorted(set(values))


def candidate_dataset_stems_for_restore(cfg: BTABExperimentConfig) -> list[str]:
    """Return plausible dataset stems to restore before resolve_dataset_stems runs."""
    if not cfg.n_train_list:
        return [str(stem).strip() for stem in cfg.dataset_stems if str(stem).strip()]
    family_prefixes = list(
        dict.fromkeys(_stem_family_prefix(stem) for stem in cfg.dataset_stems if str(stem).strip())
    )
    candidates: list[str] = []
    for prefix in family_prefixes:
        for n_train in cfg.n_train_list:
            candidates.append(f"{prefix}_ntrain{int(n_train)}")
            candidates.append(f"{prefix}_n{int(n_train)}")
    return list(dict.fromkeys(candidates))


def resolve_dataset_stems(cfg: BTABExperimentConfig) -> list[str]:
    if not cfg.n_train_list:
        return list(cfg.dataset_stems)
    records = discover_processed_dataset_records()
    family_prefixes = list(dict.fromkeys(_stem_family_prefix(stem) for stem in cfg.dataset_stems if str(stem).strip()))
    if not family_prefixes:
        raise ValueError("n_train_list was provided, but no dataset_stems were available to infer dataset families.")
    resolved: list[str] = []
    for prefix in family_prefixes:
        family_records = [rec for rec in records if rec["family_prefix"] == prefix]
        if not family_records:
            raise FileNotFoundError(f"No processed datasets found for dataset family prefix {prefix!r}.")
        for n_train in cfg.n_train_list:
            matches = [rec for rec in family_records if int(rec["n_train"]) == int(n_train)]
            if not matches:
                available = ", ".join(str(rec["n_train"]) for rec in sorted(family_records, key=lambda rec: int(rec["n_train"])))
                sidecar_only = _json_sidecar_only_n_trains(prefix)
                hint = ""
                if int(n_train) in sidecar_only:
                    hint = (
                        f"\nFound .json sidecar for n_train={int(n_train)} but no matching .npz under {_PROCESSED_DIR}. "
                        "Restore from Google Drive cache first (notebook: ensure_btab_datasets_available), "
                        "or copy the .npz into processed/."
                    )
                elif sidecar_only:
                    hint = (
                        "\nJson-only sidecars (no .npz) for this family include n_train: "
                        + ", ".join(str(v) for v in sidecar_only)
                        + "."
                    )
                raise FileNotFoundError(
                    f"No processed dataset found for family {prefix!r} with n_train={int(n_train)}. "
                    f"Available local .npz n_train values: {available}{hint}"
                )
            resolved.append(_pick_matching_stem(matches, prefix=prefix, n_train=int(n_train)))
    return list(dict.fromkeys(resolved))


def load_processed_dataset(stem: str) -> dict[str, Any]:
    dataset_map = discover_processed_datasets()
    if stem not in dataset_map:
        sidecar_only = sorted(
            p.stem
            for p in _PROCESSED_DIR.glob("*.json")
            if p.stem not in dataset_map
        )
        hint = ""
        if stem in sidecar_only:
            hint = (
                f"\nFound sidecar metadata for {stem!r} but no matching .npz file under {_PROCESSED_DIR}."
                "\nThis usually means the repository/Drive cache only has the .json preview metadata."
                "\nCopy the matching .npz into processed/ (or restore it from Drive cache) before running."
            )
        raise FileNotFoundError(
            f"Unknown dataset stem {stem!r}. Available .npz stems: {', '.join(sorted(dataset_map))}{hint}"
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


def _kernel_param_value(
    params: dict[str, Any],
    canonical_key: str,
    default: Any,
    *aliases: str,
) -> Any:
    for key in (canonical_key, *aliases):
        if key in params:
            return params[key]
    return default


def iter_kernel_configs(cfg: BTABExperimentConfig) -> list[BTABExperimentConfig]:
    """Expand cfg.kernel_family_list into concrete per-kernel configs."""
    raw_families = cfg.kernel_family_list or [cfg.kernel_family]
    families = [str(family).strip() for family in raw_families if str(family).strip()]
    if not families:
        raise ValueError("kernel_family_list is empty; provide at least one kernel family.")

    params_by_family = {
        str(key).strip().lower(): dict(value)
        for key, value in (cfg.kernel_params_by_family or {}).items()
        if str(key).strip()
    }
    kernel_cfgs: list[BTABExperimentConfig] = []
    for family in families:
        params = params_by_family.get(family.lower(), {})
        kernel_cfgs.append(
            replace(
                cfg,
                kernel_family=str(
                    _kernel_param_value(params, "kernel_family", family, "family")
                ),
                kernel_lengthscale=float(
                    _kernel_param_value(
                        params,
                        "kernel_lengthscale",
                        cfg.kernel_lengthscale,
                        "lengthscale",
                        "length_scale",
                    )
                ),
                kernel_nu=float(
                    _kernel_param_value(params, "kernel_nu", cfg.kernel_nu, "nu")
                ),
                kernel_variance=float(
                    _kernel_param_value(
                        params,
                        "kernel_variance",
                        cfg.kernel_variance,
                        "variance",
                    )
                ),
            )
        )
    return kernel_cfgs


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


def build_gpu_run_config(
    cfg: BTABExperimentConfig,
    *,
    maxiter: int | None = None,
) -> GPURunConfig:
    return GPURunConfig(
        reg_lambda=float(cfg.reg_lambda),
        tol=float(cfg.tol),
        maxiter=int(cfg.maxiter if maxiter is None else maxiter),
        chunk_size=cfg.chunk_size,
        debug_finite_checks=bool(cfg.debug_finite_checks),
        profile_components=bool(cfg.profile_components),
        backend=cfg.backend,
    )


def method_rows_for_dataset(
    cfg: BTABExperimentConfig,
    dataset_payload: dict[str, Any],
    *,
    repeat_idx: int = 0,
    is_warmup: bool = False,
) -> list[dict[str, Any]]:
    cfg = resolve_btab_experiment_route(
        cfg,
        n_train=int(dataset_payload["n_train"]),
    )
    x_train = np.asarray(dataset_payload["x_train"], dtype=np.float64)
    y_train = np.asarray(dataset_payload["y_train"], dtype=np.float64).reshape(-1)
    x_test = np.asarray(dataset_payload["x_test"], dtype=np.float64)
    y_test = np.asarray(dataset_payload["y_test"], dtype=np.float64).reshape(-1)
    run_seed = int(cfg.seed) + int(repeat_idx)
    solver = EFGPSolver(
        make_kernel(cfg, dataset_payload["dim"]),
        reg_lambda=float(cfg.reg_lambda),
        eps=float(cfg.eps),
        nufft_tol=float(cfg.nufft_tol),
        l2scaled=bool(cfg.l2_scaled),
    )
    v1_gpu_cfg = build_gpu_run_config(cfg, maxiter=int(cfg.maxiter))
    precond_gpu_cfg = build_gpu_run_config(cfg, maxiter=int(cfg.non_v1_maxiter))

    rows: list[dict[str, Any]] = []

    def _run_case(
        tag: str,
        fn,
        *,
        run_cfg: GPURunConfig,
        extra: dict[str, Any] | None = None,
    ) -> None:
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
            "maxiter": int(run_cfg.maxiter),
            "v1_maxiter": int(cfg.maxiter),
            "non_v1_maxiter": int(cfg.non_v1_maxiter),
            "repeat_idx": int(repeat_idx),
            "is_warmup": bool(is_warmup),
            "run_seed": int(run_seed),
            "method": tag,
            "btab_experiment_route": str(cfg.btab_experiment_route),
            "btab_route_group": str(cfg.btab_experiment_route),
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

    _run_case(
        "plain_cg",
        lambda: run_v1_pure_efgp(solver, x_train, y_train, v1_gpu_cfg),
        run_cfg=v1_gpu_cfg,
    )

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
                precond_gpu_cfg,
                eig_cfg=eig_cfg,
            ),
            run_cfg=precond_gpu_cfg,
            extra={"top_q": int(top_q)},
        )

    if str(cfg.btab_active_mode).lower() == "topk":
        inverse_topk_list = (
            cfg.btab_topk_list
            if cfg.btab_inverse_topk_list is None
            else cfg.btab_inverse_topk_list
        )
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
                diagnostic_mode=str(cfg.btab_diagnostic_mode),
                diagnostic_power_iter=int(cfg.btab_diagnostic_power_iter),
                diagnostic_tol=float(cfg.btab_diagnostic_tol),
            )
            for k in inverse_topk_list
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
                diagnostic_mode=str(cfg.btab_diagnostic_mode),
                diagnostic_power_iter=int(cfg.btab_diagnostic_power_iter),
                diagnostic_tol=float(cfg.btab_diagnostic_tol),
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
                precond_gpu_cfg,
                btab_cfg=btab_cfg,
            ),
            run_cfg=precond_gpu_cfg,
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
                "btab_diagnostic_mode": str(btab_cfg.diagnostic_mode),
                "btab_diagnostic_power_iter": int(btab_cfg.diagnostic_power_iter),
                "btab_diagnostic_tol": float(btab_cfg.diagnostic_tol),
            },
        )
    if cfg.btab_boxeig_topk_q_pairs is None:
        boxeig_configs = [
            (base_cfg, int(q))
            for base_cfg in configs
            for q in cfg.btab_eig_q_list
        ]
    else:
        boxeig_configs = [
            (
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
                    diagnostic_mode=str(cfg.btab_diagnostic_mode),
                    diagnostic_power_iter=int(cfg.btab_diagnostic_power_iter),
                    diagnostic_tol=float(cfg.btab_diagnostic_tol),
                ),
                int(q),
            )
            for k, q in cfg.btab_boxeig_topk_q_pairs
        ]
    for base_cfg, q in boxeig_configs:
        tag = format_box_tag(
            type("BoxTag", (), {
                "active_mode": base_cfg.active_mode,
                "active_topk": base_cfg.active_topk,
                "active_tau": base_cfg.active_tau,
            })()
        )
        btab_cfg = replace(
            base_cfg,
            solve_mode="boxeig",
            exact_apply_mode="boxeig",
            eig_q=int(q),
            eig_tol=float(cfg.btab_eig_tol),
            eig_maxiter=cfg.btab_eig_maxiter,
            eig_ncv=cfg.btab_eig_ncv,
            eig_apply_batch_cols=cfg.btab_eig_apply_batch_cols,
            diagnostic_mode=str(cfg.btab_diagnostic_mode),
            diagnostic_power_iter=int(cfg.btab_diagnostic_power_iter),
            diagnostic_tol=float(cfg.btab_diagnostic_tol),
        )
        _run_case(
            f"btab_boxeig_{tag}_q{int(q)}",
            lambda btab_cfg=btab_cfg: run_v7_box_eigenpro_active_block(
                solver,
                x_train,
                y_train,
                precond_gpu_cfg,
                btab_cfg=btab_cfg,
            ),
            run_cfg=precond_gpu_cfg,
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
                "btab_solve_mode": "boxeig",
                "btab_exact_box_max_size": (
                    None
                    if btab_cfg.exact_box_max_size is None
                    else int(btab_cfg.exact_box_max_size)
                ),
                "btab_exact_apply_mode": "boxeig",
                "btab_outer_solver": str(btab_cfg.outer_solver),
                "btab_outer_gmres_restart": int(btab_cfg.outer_gmres_restart),
                "btab_inner_tol": float(btab_cfg.inner_tol),
                "btab_inner_maxiter": int(btab_cfg.inner_maxiter),
                "btab_inner_precond": str(btab_cfg.inner_precond),
                "btab_eig_q": int(btab_cfg.eig_q),
                "btab_eig_tol": float(btab_cfg.eig_tol),
                "btab_eig_maxiter": btab_cfg.eig_maxiter,
                "btab_eig_ncv": btab_cfg.eig_ncv,
                "btab_eig_apply_batch_cols": btab_cfg.eig_apply_batch_cols,
                "btab_diagnostic_mode": str(btab_cfg.diagnostic_mode),
                "btab_diagnostic_power_iter": int(btab_cfg.diagnostic_power_iter),
                "btab_diagnostic_tol": float(btab_cfg.diagnostic_tol),
            },
        )
    return rows


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _derive_iterations(row: dict[str, Any]) -> float:
    for key in ("outer_iters", "cg_iters"):
        val = row.get(key)
        if isinstance(val, (int, float, np.integer, np.floating)) and np.isfinite(float(val)):
            return float(val)
    return float("nan")


def finalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "iterations" not in out:
        out["iterations"] = _derive_iterations(out)
    if "efgp_matrix_dim" in out and "M" not in out:
        out["M"] = out.get("efgp_matrix_dim")
    return out


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _freeze_group_value(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return "__nan__"
    if isinstance(value, (np.floating,)):
        val = float(value)
        return "__nan__" if not np.isfinite(val) else val
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    group_key_columns = {
        "dataset_stem",
        "dataset_dim",
        "n_train",
        "n_test",
        "kernel_family",
        "kernel_lengthscale",
        "kernel_nu",
        "kernel_variance",
        "reg_lambda",
        "eps",
        "tol",
        "maxiter",
        "v1_maxiter",
        "non_v1_maxiter",
        "method",
        "btab_experiment_route",
        "btab_route_group",
        "top_q",
        "version",
        "nufft_backend",
        "nufft_stage",
        "chunk_size",
        "debug_finite_checks",
        "profile_components",
        "device_name",
        "has_nufft",
        "precond_kind",
        "btab_active_mode",
        "btab_active_topk",
        "btab_active_tau",
        "btab_box_budget",
        "btab_solve_mode",
        "btab_exact_box_max_size",
        "btab_exact_apply_mode",
        "btab_outer_solver",
        "btab_outer_gmres_restart",
        "btab_inner_tol",
        "btab_inner_maxiter",
        "btab_inner_precond",
        "btab_eig_q",
        "btab_eig_tol",
        "btab_eig_maxiter",
        "btab_eig_ncv",
        "btab_eig_apply_batch_cols",
        "btab_diagnostic_mode",
        "btab_diagnostic_power_iter",
        "btab_diagnostic_tol",
        "active_mode",
        "active_topk",
        "active_tau",
        "solve_mode",
        "exact_apply_mode",
        "outer_solver",
        "outer_gmres_restart",
        "inner_tol",
        "inner_maxiter",
        "inner_precond",
        "diagnostic_mode",
        "btab_eig_backend",
        "btab_eig_ncv_actual",
        "mtot",
        "dim",
        "efgp_matrix_dim",
        "M",
        "box_shape",
        "box_radii",
    }
    excluded_columns = {"repeat_idx", "is_warmup", "run_seed", "status", "error_type", "error_message", "traceback"}
    present_group_columns = [col for col in group_key_columns if any(col in row for row in rows)]
    metric_columns = sorted(
        {
            col
            for row in rows
            for col, value in row.items()
            if col not in excluded_columns and col not in present_group_columns and _is_numeric_value(value)
        }
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(_freeze_group_value(row.get(col)) for col in present_group_columns)
        grouped.setdefault(key, []).append(row)
    aggregated: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        agg_row = {col: group_rows[0].get(col) for col in present_group_columns}
        agg_row["repeat_count"] = int(len(group_rows))
        ok_rows = [row for row in group_rows if str(row.get("status", "")).lower() == "ok"]
        agg_row["ok_count"] = int(len(ok_rows))
        agg_row["failed_count"] = int(len(group_rows) - len(ok_rows))
        if len(ok_rows) == len(group_rows):
            agg_row["status"] = "ok"
        elif ok_rows:
            agg_row["status"] = "partial_failed"
        else:
            agg_row["status"] = "failed"
        error_types = sorted({str(row.get("error_type")) for row in group_rows if row.get("error_type")})
        error_messages = sorted({str(row.get("error_message")) for row in group_rows if row.get("error_message")})
        agg_row["error_types"] = "|".join(error_types)
        agg_row["error_messages"] = " || ".join(error_messages)
        for col in metric_columns:
            vals = [
                float(row[col])
                for row in group_rows
                if col in row and _is_numeric_value(row[col]) and np.isfinite(float(row[col]))
            ]
            if not vals:
                agg_row[col] = float("nan")
                agg_row[f"{col}_median"] = float("nan")
                agg_row[f"{col}_std"] = float("nan")
                agg_row[f"{col}_min"] = float("nan")
                agg_row[f"{col}_max"] = float("nan")
                continue
            arr = np.asarray(vals, dtype=np.float64)
            agg_row[col] = float(np.median(arr))
            agg_row[f"{col}_median"] = float(np.median(arr))
            agg_row[f"{col}_std"] = float(np.std(arr, ddof=0))
            agg_row[f"{col}_min"] = float(np.min(arr))
            agg_row[f"{col}_max"] = float(np.max(arr))
        aggregated.append(agg_row)
    aggregated.sort(key=lambda row: tuple(str(row.get(col, "")) for col in ("dataset_stem", "method", "eps")))
    return aggregated


def write_rows(rows: list[dict[str, Any]], out_dir: Path, *, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return None


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    master_rows = [finalize_row(row) for row in rows]
    aggregate = aggregate_rows(master_rows)
    btab_master = [row for row in master_rows if "btab" in str(row.get("method", "")).lower()]
    btab_aggregate = [row for row in aggregate if "btab" in str(row.get("method", "")).lower()]
    write_rows(master_rows, out_dir, stem="master_summary")
    write_rows(aggregate, out_dir, stem="aggregate_summary")
    write_rows(btab_master, out_dir, stem="btab_master_summary")
    write_rows(btab_aggregate, out_dir, stem="btab_aggregate_summary")
    write_rows(master_rows, out_dir, stem="summary")


def _apply_named_route_experiment_defaults(cfg: BTABExperimentConfig) -> BTABExperimentConfig:
    route = normalize_btab_experiment_route(cfg.btab_experiment_route)
    if route in {"group_a", "group_b", "group_c"}:
        return resolve_btab_experiment_route(cfg)
    return cfg


def _run_experiments_single(
    cfg: BTABExperimentConfig,
    *,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = _apply_named_route_experiment_defaults(cfg)
    out_dir = out_dir or cfg.resolve_output_dir(_HERE)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    resolved_route_configs: dict[str, dict[str, Any]] = {}
    resolved_stems = resolve_dataset_stems(cfg)
    eps_values = [float(v) for v in (cfg.eps_list or [cfg.eps])]
    for stem in resolved_stems:
        payload = load_processed_dataset(stem)
        dataset_cfg = resolve_btab_experiment_route(
            cfg,
            n_train=int(payload["n_train"]),
        )
        kernel_cfgs = iter_kernel_configs(dataset_cfg)
        for kernel_cfg in kernel_cfgs:
            route_key = stem if len(kernel_cfgs) == 1 else f"{stem}|{kernel_cfg.kernel_family}"
            resolved_route_configs[route_key] = asdict(kernel_cfg)
            for eps in eps_values:
                run_cfg = replace(kernel_cfg, eps=float(eps), eps_list=list(eps_values))
                for warmup_idx in range(int(cfg.warmup_repeats)):
                    np.random.seed(int(cfg.seed) + int(warmup_idx))
                    method_rows_for_dataset(
                        run_cfg,
                        payload,
                        repeat_idx=int(warmup_idx),
                        is_warmup=True,
                    )
                for repeat_idx in range(int(cfg.measured_repeats)):
                    np.random.seed(int(cfg.seed) + int(repeat_idx))
                    rows.extend(
                        method_rows_for_dataset(
                            run_cfg,
                            payload,
                            repeat_idx=int(repeat_idx),
                            is_warmup=False,
                        )
                    )
                    write_outputs(rows, out_dir)
    (out_dir / "experiment_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "resolved_route_configs.json").write_text(
        json.dumps(resolved_route_configs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "output_dir": str(out_dir),
        "rows": rows,
        "dataset_stems": resolved_stems,
        "eps_values": eps_values,
        "resolved_route_configs": resolved_route_configs,
    }


def run_experiments(cfg: BTABExperimentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BTABExperimentConfig()
    route_cfgs = expand_btab_experiment_routes(cfg)
    if len(route_cfgs) == 1:
        return _run_experiments_single(route_cfgs[0])

    parent_dir = cfg.resolve_output_dir(_HERE)
    parent_dir.mkdir(parents=True, exist_ok=True)
    group_results: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    all_resolved_stems: list[str] = []
    all_eps_values: list[float] = []
    all_resolved_route_configs: dict[str, dict[str, Any]] = {}
    for route_cfg in route_cfgs:
        group = normalize_btab_experiment_route(route_cfg.btab_experiment_route)
        group_dir = parent_dir / group
        result = _run_experiments_single(route_cfg, out_dir=group_dir)
        group_results[group] = result
        all_rows.extend(result["rows"])
        all_resolved_stems.extend(result["dataset_stems"])
        all_eps_values.extend(result["eps_values"])
        for key, value in result["resolved_route_configs"].items():
            all_resolved_route_configs[f"{group}/{key}"] = value

    (parent_dir / "experiment_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (parent_dir / "multi_group_results.json").write_text(
        json.dumps(
            {
                group: {
                    "output_dir": result["output_dir"],
                    "dataset_stems": result["dataset_stems"],
                    "eps_values": result["eps_values"],
                    "n_rows": len(result["rows"]),
                }
                for group, result in group_results.items()
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(parent_dir),
        "rows": all_rows,
        "dataset_stems": list(dict.fromkeys(all_resolved_stems)),
        "eps_values": list(dict.fromkeys(all_eps_values)),
        "resolved_route_configs": all_resolved_route_configs,
        "group_results": group_results,
    }


def _parse_int_list(raw: str) -> list[int]:
    return [int(tok.strip()) for tok in raw.split(",") if tok.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(tok.strip()) for tok in raw.split(",") if tok.strip()]


def _parse_topk_q_pairs(raw: str) -> list[tuple[int, int]] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    pairs: list[tuple[int, int]] = []
    for token in text.split(","):
        parts = token.strip().split(":")
        if len(parts) != 2:
            raise ValueError(
                "Box-EigenPro pairs must use 'topk:q' format, for example "
                "'4096:128,8192:192'."
            )
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def _parse_optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in ("none", "null", "auto"):
        return None
    return int(text)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Box-Toeplitz Active Block GPU experiments.")
    p.add_argument("--dataset-stems", type=str, default=",".join(BTABExperimentConfig().dataset_stems))
    p.add_argument("--n-train-list", type=str, default="")
    p.add_argument("--kernel-family", type=str, default=BTABExperimentConfig().kernel_family)
    p.add_argument("--kernel-lengthscale", type=float, default=BTABExperimentConfig().kernel_lengthscale)
    p.add_argument("--kernel-nu", type=float, default=BTABExperimentConfig().kernel_nu)
    p.add_argument("--kernel-variance", type=float, default=BTABExperimentConfig().kernel_variance)
    p.add_argument("--reg-lambda", type=float, default=BTABExperimentConfig().reg_lambda)
    p.add_argument("--eps", type=float, default=BTABExperimentConfig().eps)
    p.add_argument("--eps-list", type=str, default="")
    p.add_argument("--nufft-tol", type=float, default=BTABExperimentConfig().nufft_tol)
    p.add_argument("--tol", type=float, default=BTABExperimentConfig().tol)
    p.add_argument("--maxiter", type=int, default=BTABExperimentConfig().maxiter)
    p.add_argument("--non-v1-maxiter", type=int, default=BTABExperimentConfig().non_v1_maxiter)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--warmup-repeats", type=int, default=BTABExperimentConfig().warmup_repeats)
    p.add_argument("--measured-repeats", type=int, default=BTABExperimentConfig().measured_repeats)
    p.add_argument("--seed", type=int, default=BTABExperimentConfig().seed)
    p.add_argument("--eigenpro-topq-list", type=str, default="45,90")
    p.add_argument(
        "--btab-experiment-route",
        type=str,
        default=BTABExperimentConfig().btab_experiment_route,
        choices=("cartesian", "custom", "group_a", "group_b", "group_c", "schedule"),
        help=(
            "BTAB parameter expansion. 'cartesian' uses the legacy full "
            "top-k x q sweep; named routes use curated non-Cartesian "
            "shortlists; 'schedule' selects one from each dataset's n_train."
        ),
    )
    p.add_argument(
        "--btab-experiment-routes",
        type=str,
        default="",
        help=(
            "Optional comma-separated named groups to run into subdirectories, "
            "for example 'group_a,group_b'."
        ),
    )
    p.add_argument("--btab-active-mode", type=str, default=BTABExperimentConfig().btab_active_mode)
    p.add_argument("--btab-topk-list", type=str, default="512,1024,2048")
    p.add_argument("--btab-tau-list", type=str, default="1e-1,1e-2")
    p.add_argument(
        "--btab-inverse-topk-list",
        type=str,
        default="",
        help="Explicit custom inverse shortlist, for example '1024,2048,4096'.",
    )
    p.add_argument(
        "--btab-boxeig-topk-q-pairs",
        type=str,
        default="",
        help=(
            "Explicit custom Box-EigenPro shortlist in topk:q format, "
            "for example '4096:128,8192:128,8192:192'."
        ),
    )
    p.add_argument("--btab-box-budget", type=int, default=BTABExperimentConfig().btab_box_budget)
    p.add_argument("--btab-solve-mode", type=str, default=BTABExperimentConfig().btab_solve_mode)
    p.add_argument("--btab-exact-box-max-size", type=int, default=BTABExperimentConfig().btab_exact_box_max_size)
    p.add_argument("--btab-exact-apply-mode", type=str, default=BTABExperimentConfig().btab_exact_apply_mode)
    p.add_argument("--btab-outer-solver", type=str, default=BTABExperimentConfig().btab_outer_solver)
    p.add_argument("--btab-outer-gmres-restart", type=int, default=BTABExperimentConfig().btab_outer_gmres_restart)
    p.add_argument("--btab-inner-tol", type=float, default=BTABExperimentConfig().btab_inner_tol)
    p.add_argument("--btab-inner-maxiter", type=int, default=BTABExperimentConfig().btab_inner_maxiter)
    p.add_argument("--btab-inner-precond", type=str, default=BTABExperimentConfig().btab_inner_precond)
    p.add_argument("--btab-eig-q-list", type=str, default="32,64,128")
    p.add_argument("--btab-eig-tol", type=float, default=BTABExperimentConfig().btab_eig_tol)
    p.add_argument("--btab-eig-maxiter", type=str, default="")
    p.add_argument("--btab-eig-ncv", type=str, default="")
    p.add_argument("--btab-eig-apply-batch-cols", type=str, default="")
    p.add_argument("--btab-diagnostic-mode", type=str, default=BTABExperimentConfig().btab_diagnostic_mode)
    p.add_argument("--btab-diagnostic-power-iter", type=int, default=BTABExperimentConfig().btab_diagnostic_power_iter)
    p.add_argument("--btab-diagnostic-tol", type=float, default=BTABExperimentConfig().btab_diagnostic_tol)
    p.add_argument("--output-dir", type=str, default="")
    return p


def config_from_args(args: argparse.Namespace) -> BTABExperimentConfig:
    dataset_stems = [tok.strip() for tok in args.dataset_stems.split(",") if tok.strip()]
    n_train_list = _parse_int_list(args.n_train_list) if str(args.n_train_list).strip() else []
    eps_list = _parse_float_list(args.eps_list) if str(args.eps_list).strip() else [float(args.eps)]
    return BTABExperimentConfig(
        dataset_stems=dataset_stems,
        n_train_list=n_train_list,
        kernel_family=str(args.kernel_family),
        kernel_lengthscale=float(args.kernel_lengthscale),
        kernel_nu=float(args.kernel_nu),
        kernel_variance=float(args.kernel_variance),
        reg_lambda=float(args.reg_lambda),
        eps=float(args.eps),
        eps_list=eps_list,
        nufft_tol=float(args.nufft_tol),
        tol=float(args.tol),
        maxiter=int(args.maxiter),
        non_v1_maxiter=int(args.non_v1_maxiter),
        chunk_size=args.chunk_size,
        warmup_repeats=int(args.warmup_repeats),
        measured_repeats=int(args.measured_repeats),
        seed=int(args.seed),
        eigenpro_topq_list=_parse_int_list(args.eigenpro_topq_list),
        btab_experiment_route=str(args.btab_experiment_route),
        btab_experiment_routes=(
            [tok.strip() for tok in str(args.btab_experiment_routes).split(",") if tok.strip()]
            if str(args.btab_experiment_routes).strip()
            else []
        ),
        btab_active_mode=str(args.btab_active_mode),
        btab_topk_list=_parse_int_list(args.btab_topk_list),
        btab_tau_list=_parse_float_list(args.btab_tau_list),
        btab_inverse_topk_list=(
            _parse_int_list(args.btab_inverse_topk_list)
            if str(args.btab_inverse_topk_list).strip()
            else None
        ),
        btab_boxeig_topk_q_pairs=_parse_topk_q_pairs(
            args.btab_boxeig_topk_q_pairs
        ),
        btab_box_budget=args.btab_box_budget,
        btab_solve_mode=str(args.btab_solve_mode),
        btab_exact_box_max_size=args.btab_exact_box_max_size,
        btab_exact_apply_mode=str(args.btab_exact_apply_mode),
        btab_outer_solver=str(args.btab_outer_solver),
        btab_outer_gmres_restart=int(args.btab_outer_gmres_restart),
        btab_inner_tol=float(args.btab_inner_tol),
        btab_inner_maxiter=int(args.btab_inner_maxiter),
        btab_inner_precond=str(args.btab_inner_precond),
        btab_eig_q_list=_parse_int_list(args.btab_eig_q_list),
        btab_eig_tol=float(args.btab_eig_tol),
        btab_eig_maxiter=_parse_optional_int(args.btab_eig_maxiter),
        btab_eig_ncv=_parse_optional_int(args.btab_eig_ncv),
        btab_eig_apply_batch_cols=_parse_optional_int(args.btab_eig_apply_batch_cols),
        btab_diagnostic_mode=str(args.btab_diagnostic_mode),
        btab_diagnostic_power_iter=int(args.btab_diagnostic_power_iter),
        btab_diagnostic_tol=float(args.btab_diagnostic_tol),
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
