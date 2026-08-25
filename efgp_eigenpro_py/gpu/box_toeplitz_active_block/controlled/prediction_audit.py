"""Out-of-timing prediction audit for a controlled fixed-system experiment.

This module deliberately reuses the controlled benchmark's system preparation,
method resolution, and single-solve entry point.  Prediction is evaluated only
after the audit solve, in bounded GPU chunks, and is never used for a speedup
claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ...benchmark_dataset.stored_npz import (
    StoredNpzError,
    inspect_stored_npy_member,
    load_stored_npz_prefix,
)
from ...v1_ops import predict_v1
from .benchmark import (
    ControlledConfig,
    MethodSpec,
    PreparedSystem,
    prepare_shared_system,
    resolve_method_specs,
    run_one_method,
    system_fingerprint,
)


_HERE = Path(__file__).resolve().parent
_ROW_FIELDS = (
    "system_id",
    "dataset",
    "method",
    "method_kind",
    "solve_status",
    "true_relres",
    "test_rmse",
    "test_rmse_ratio_vs_cg",
    "test_rmse_diff_vs_cg",
    "prediction_seconds",
    "iterations",
    "n_test",
    "prediction_chunk_size",
    "audit_solve_build_seconds",
    "audit_solve_seconds",
    "audit_only_not_for_speed_claim",
)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _synchronize(backend: Any) -> None:
    cuda = getattr(backend.xp, "cuda", None)
    if cuda is not None:
        cuda.runtime.deviceSynchronize()


def _parse_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(part.strip().lower() for part in str(raw).split(",") if part.strip())
    if not methods:
        raise argparse.ArgumentTypeError("methods must contain at least one name")
    return methods


def load_controlled_config(
    path: str | Path,
    *,
    methods: Sequence[str] | None = None,
) -> ControlledConfig:
    """Load a benchmark ``experiment_config.json`` without changing its system."""
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("controlled config must be a JSON object")
    valid_fields = {field.name for field in fields(ControlledConfig)}
    unknown = sorted(set(payload) - valid_fields)
    if unknown:
        raise ValueError(f"unknown ControlledConfig fields: {unknown}")
    if "methods" in payload:
        payload["methods"] = tuple(str(value) for value in payload["methods"])
    if "diagnostic_topk" in payload:
        payload["diagnostic_topk"] = tuple(int(value) for value in payload["diagnostic_topk"])
    cfg = ControlledConfig(**payload)
    if methods is not None:
        cfg = replace(cfg, methods=tuple(str(method).strip().lower() for method in methods))
    if "cg" not in cfg.methods:
        raise ValueError("prediction audit requires method 'cg' for RMSE comparisons")
    return cfg


def load_test_arrays(
    dataset_path: str | Path,
    *,
    max_test: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load host-resident test arrays from the exact NPZ used for training."""
    path = Path(dataset_path).expanduser().resolve()
    try:
        x_info = inspect_stored_npy_member(path, "x_test")
        y_info = inspect_stored_npy_member(path, "y_test")
        if not x_info.shape or not y_info.shape:
            raise StoredNpzError("x_test and y_test must have a row dimension")
        full_n_test = int(x_info.shape[0])
        if int(y_info.shape[0]) != full_n_test:
            raise ValueError("x_test and y_test row counts differ")
        limit = full_n_test if max_test is None else min(int(max_test), full_n_test)
        if limit <= 0:
            raise ValueError("max_test must be positive or None")
        x_test = load_stored_npz_prefix(path, "x_test", limit)
        y_test = load_stored_npz_prefix(
            path,
            "y_test",
            limit,
            dtype=np.float64,
        ).reshape(-1)
    except StoredNpzError:
        with np.load(path) as loaded:
            missing = [name for name in ("x_test", "y_test") if name not in loaded.files]
            if missing:
                raise KeyError(f"dataset {path} is missing test arrays: {missing}")
            x_test = np.asarray(loaded["x_test"])
            y_test = np.asarray(loaded["y_test"], dtype=np.float64).reshape(-1)
        full_n_test = int(y_test.size)
        if max_test is not None:
            limit = int(max_test)
            if limit <= 0:
                raise ValueError("max_test must be positive or None")
            limit = min(limit, full_n_test)
            x_test = x_test[:limit]
            y_test = y_test[:limit]
    if x_test.ndim != 2:
        raise ValueError("x_test must be a two-dimensional array")
    if x_test.shape[0] != y_test.size:
        raise ValueError("x_test and y_test row counts differ")
    if y_test.size == 0:
        raise ValueError("test set is empty")
    return x_test, y_test, full_n_test


def chunked_test_rmse(
    system: PreparedSystem,
    beta_host: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    chunk_size: int,
    predict_fn: Callable[[Any, Any, Any, Any], Any] | None = None,
) -> tuple[float, float]:
    """Compute RMSE while placing at most ``chunk_size`` test rows on the GPU."""
    size = int(chunk_size)
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if int(x_test.shape[0]) != int(y_test.size):
        raise ValueError("x_test and y_test row counts differ")

    backend = system.backend
    xp = backend.xp
    if predict_fn is None:
        predict_fn = predict_v1
    beta_gpu = xp.asarray(beta_host)
    squared_error_sum = 0.0
    _synchronize(backend)
    start_ns = time.perf_counter_ns()
    for start in range(0, int(y_test.size), size):
        stop = min(start + size, int(y_test.size))
        prediction_gpu = xp.asarray(
            predict_fn(backend, system.data_ctx, x_test[start:stop], beta_gpu),
            dtype=xp.float64,
        ).reshape(-1)
        target_gpu = xp.asarray(y_test[start:stop], dtype=xp.float64)
        if int(prediction_gpu.size) != stop - start:
            raise ValueError("prediction row count differs from the requested chunk")
        chunk_sse = xp.sum((prediction_gpu - target_gpu) ** 2)
        squared_error_sum += float(chunk_sse.item() if hasattr(chunk_sse, "item") else chunk_sse)
        del prediction_gpu, target_gpu, chunk_sse
    _synchronize(backend)
    prediction_seconds = (time.perf_counter_ns() - start_ns) * 1e-9
    rmse = math.sqrt(squared_error_sum / int(y_test.size))
    return float(rmse), float(prediction_seconds)


def attach_cg_rmse_comparisons(rows: list[dict[str, Any]]) -> None:
    """Add method/CG RMSE ratios and signed differences in place."""
    cg_rows = [row for row in rows if row.get("method") == "cg"]
    cg_rmse = _finite_or_none(cg_rows[0].get("test_rmse")) if len(cg_rows) == 1 else None
    for row in rows:
        rmse = _finite_or_none(row.get("test_rmse"))
        if row.get("method") == "cg" and rmse is not None:
            row["test_rmse_ratio_vs_cg"] = 1.0
            row["test_rmse_diff_vs_cg"] = 0.0
        elif rmse is not None and cg_rmse is not None:
            row["test_rmse_ratio_vs_cg"] = rmse / cg_rmse if cg_rmse > 0.0 else None
            row["test_rmse_diff_vs_cg"] = rmse - cg_rmse
        else:
            row["test_rmse_ratio_vs_cg"] = None
            row["test_rmse_diff_vs_cg"] = None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_ROW_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_prediction_audit(
    cfg: ControlledConfig,
    *,
    output_dir: str | Path,
    prediction_chunk_size: int = 100_000,
    warmup_solves: int = 1,
    max_test: int | None = None,
    config_source: str | Path | None = None,
) -> Path:
    """Run one audit solve per method and write prediction-only JSON/CSV evidence."""
    if int(warmup_solves) < 0:
        raise ValueError("warmup_solves must be nonnegative")
    if int(prediction_chunk_size) <= 0:
        raise ValueError("prediction_chunk_size must be positive")
    if "cg" not in cfg.methods:
        raise ValueError("prediction audit requires method 'cg' for RMSE comparisons")

    system = prepare_shared_system(cfg)
    specs, _rule = resolve_method_specs(system, cfg)
    dataset_path = Path(str(system.manifest["dataset_path"])).resolve()
    x_test, y_test, full_n_test = load_test_arrays(dataset_path, max_test=max_test)
    rows: list[dict[str, Any]] = []

    for order_position, spec in enumerate(specs):
        for warmup_idx in range(int(warmup_solves)):
            run_one_method(
                system,
                cfg,
                spec,
                repeat_idx=-(warmup_idx + 1),
                order_position=int(order_position),
                is_warmup=True,
            )
        solve_row, beta_host = run_one_method(
            system,
            cfg,
            spec,
            repeat_idx=0,
            order_position=int(order_position),
            is_warmup=False,
        )
        rmse: float | None = None
        prediction_seconds: float | None = None
        solve_status = str(solve_row.get("status", "unknown"))
        if beta_host is not None:
            try:
                rmse, prediction_seconds = chunked_test_rmse(
                    system,
                    beta_host,
                    x_test,
                    y_test,
                    chunk_size=int(prediction_chunk_size),
                )
            except Exception as exc:
                solve_status = f"prediction_error:{type(exc).__name__}:{exc}"
        rows.append(
            {
                "system_id": system.system_id,
                "dataset": str(system.manifest.get("dataset_stem", cfg.dataset_stem)),
                "method": spec.label,
                "method_kind": spec.kind,
                "solve_status": solve_status,
                "true_relres": _finite_or_none(solve_row.get("true_relres")),
                "test_rmse": rmse,
                "test_rmse_ratio_vs_cg": None,
                "test_rmse_diff_vs_cg": None,
                "prediction_seconds": prediction_seconds,
                "iterations": int(solve_row.get("iterations", -1)),
                "n_test": int(y_test.size),
                "prediction_chunk_size": int(prediction_chunk_size),
                "audit_solve_build_seconds": _finite_or_none(solve_row.get("build_seconds")),
                "audit_solve_seconds": _finite_or_none(solve_row.get("solve_seconds")),
                "audit_only_not_for_speed_claim": True,
            }
        )

    attach_cg_rmse_comparisons(rows)
    final_system_id = system_fingerprint(system.data_ctx, float(system.reg_lambda))
    if final_system_id != system.system_id:
        raise RuntimeError(
            "the arrays defining A,b changed during prediction audit: "
            f"{system.system_id} -> {final_system_id}"
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    config_source_path = Path(config_source).expanduser().resolve() if config_source else None
    payload = {
        "schema_version": 1,
        "audit_role": "prediction accuracy only; no solve or prediction speed claim",
        "system_id": system.system_id,
        "system_unchanged": True,
        "dataset": str(system.manifest.get("dataset_stem", cfg.dataset_stem)),
        "dataset_path": str(dataset_path),
        "dataset_content_index_sha256": system.manifest.get("dataset_content_index_sha256"),
        "config_source": str(config_source_path) if config_source_path else None,
        "config_source_sha256": (
            hashlib.sha256(config_source_path.read_bytes()).hexdigest()
            if config_source_path is not None
            else None
        ),
        "controlled_config": asdict(cfg),
        "warmup_solves_per_method": int(warmup_solves),
        "audit_solves_per_method": 1,
        "test_array_source": "x_test/y_test from the exact training NPZ",
        "test_target_scale": "stored NPZ y_test scale",
        "full_n_test": int(full_n_test),
        "evaluated_n_test": int(y_test.size),
        "test_subset_policy": "all" if int(y_test.size) == full_n_test else "first_n_prefix",
        "prediction_chunk_size": int(prediction_chunk_size),
        "rows": rows,
    }
    (destination / "prediction_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(destination / "prediction_audit.csv", rows)
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an out-of-timing, chunked test-RMSE audit from a controlled "
            "experiment_config.json."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=None,
        help="Optional comma-separated subset; it must include cg.",
    )
    parser.add_argument("--prediction-chunk-size", type=int, default=100_000)
    parser.add_argument("--warmup-solves", type=int, default=1)
    parser.add_argument(
        "--max-test",
        type=int,
        default=0,
        help="First-N smoke-test cap; zero evaluates the entire stored test set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to controlled/outputs/prediction_audit_TIMESTAMP.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_controlled_config(args.config, methods=args.methods)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _HERE / "outputs" / datetime.now().strftime("prediction_audit_%Y%m%d_%H%M%S")
    destination = run_prediction_audit(
        cfg,
        output_dir=output_dir,
        prediction_chunk_size=int(args.prediction_chunk_size),
        warmup_solves=int(args.warmup_solves),
        max_test=None if int(args.max_test) == 0 else int(args.max_test),
        config_source=args.config,
    )
    print(f"Wrote prediction audit to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
