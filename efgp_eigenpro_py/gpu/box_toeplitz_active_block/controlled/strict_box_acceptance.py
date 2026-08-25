"""Predeclared acceptance gate for a strict score-selected Fourier box.

The checker is deliberately read-only.  It recomputes timing comparisons from
``matched_runs.json`` so that a dataset cannot qualify through a hand-edited
summary or by selecting the best repeat after the run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
REQUIRED_METHODS = ("cg", "default", "full-eig")
MIN_PAIRED_REPEATS = 5
MIN_MEDIAN_SPEEDUP = 1.25
MIN_PAIRED_WIN_FRACTION = 0.8
MIN_LEVERAGE_CAPTURE = 0.90
MAX_SCORE_BOX_FRACTION = 0.50


class AcceptanceInputError(ValueError):
    """Raised when a required result file cannot be audited."""


def _load_json(path: Path, *, required: bool = True) -> Any | None:
    if not path.is_file():
        if required:
            raise AcceptanceInputError(f"required result file is missing: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceInputError(
            f"cannot read valid JSON from {path}: {exc}"
        ) from exc


def _gate(status: str, message: str, **evidence: Any) -> dict[str, Any]:
    return {"status": status, "message": message, "evidence": evidence}


def _finite_number(value: Any) -> float | None:
    try:
        cooked = float(value)
    except (TypeError, ValueError):
        return None
    return cooked if math.isfinite(cooked) else None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        cooked = int(value)
    except (TypeError, ValueError):
        return None
    try:
        if float(value) != cooked:
            return None
    except (TypeError, ValueError):
        return None
    return cooked


def _is_fp64_dtype(value: Any) -> bool:
    return str(value).lower() in {"float64", "complex128", "<f8", "<c16"}


def _measured_rows_by_method(
    rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    by_method = {method: [] for method in REQUIRED_METHODS}
    for row in rows:
        method = row.get("method")
        if method in by_method and row.get("is_warmup") is False:
            by_method[method].append(row)
    return by_method


def _repeat_map(
    rows: Iterable[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], list[Any]]:
    mapped: dict[int, dict[str, Any]] = {}
    duplicates: list[Any] = []
    for row in rows:
        repeat_idx = _int_value(row.get("repeat_idx"))
        if repeat_idx is None or repeat_idx < 0:
            duplicates.append(row.get("repeat_idx"))
            continue
        if repeat_idx in mapped:
            duplicates.append(repeat_idx)
            continue
        mapped[repeat_idx] = row
    return mapped, duplicates


def _fp64_gate(
    manifest: dict[str, Any], by_method: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    problems: list[str] = []
    if manifest.get("precision_mode") != "fp64":
        problems.append(f"manifest precision_mode={manifest.get('precision_mode')!r}")

    dtype_fields = (
        "x_host_dtype",
        "y_host_dtype",
        "weights_dtype",
        "gf_dtype",
        "rhs_storage_dtype",
        "rhs_solve_dtype",
        "matvec_requested_dtype",
        "real_component_dtype",
    )
    manifest_dtypes = {
        key: manifest[key] for key in dtype_fields if manifest.get(key) is not None
    }
    for key, value in manifest_dtypes.items():
        if not _is_fp64_dtype(value):
            problems.append(f"manifest {key}={value!r}")

    checked_rows = 0
    for method_rows in by_method.values():
        for row in method_rows:
            checked_rows += 1
            if row.get("precision_mode") != "fp64":
                problems.append(
                    f"{row.get('method')} repeat {row.get('repeat_idx')} "
                    f"precision_mode={row.get('precision_mode')!r}"
                )
            for key in ("solve_dtype", "true_residual_audit_dtype"):
                if row.get(key) is not None and not _is_fp64_dtype(row[key]):
                    problems.append(
                        f"{row.get('method')} repeat {row.get('repeat_idx')} "
                        f"{key}={row[key]!r}"
                    )

    status = "pass" if not problems else "fail"
    return _gate(
        status,
        (
            "manifest and measured solver records use fp64"
            if not problems
            else "non-fp64 evidence found"
        ),
        checked_measured_rows=checked_rows,
        manifest_dtypes=manifest_dtypes,
        problems=problems,
    )


def _same_system_gate(
    manifest: dict[str, Any], by_method: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    manifest_id = str(manifest.get("system_id") or "")
    final_id = str(manifest.get("final_system_id") or manifest_id)
    row_ids = {
        str(row.get("system_id") or "")
        for method_rows in by_method.values()
        for row in method_rows
    }
    problems: list[str] = []
    if not manifest_id:
        problems.append("manifest system_id is empty")
    if manifest.get("system_unchanged") is not True:
        problems.append("manifest system_unchanged is not true")
    if final_id != manifest_id:
        problems.append("final_system_id differs from system_id")
    if row_ids != {manifest_id}:
        problems.append("measured rows do not all match the manifest system_id")
    return _gate(
        "pass" if not problems else "fail",
        (
            "all audited rows share one unchanged system"
            if not problems
            else "same-system audit failed"
        ),
        manifest_system_id=manifest_id,
        final_system_id=final_id,
        measured_system_ids=sorted(row_ids),
        problems=problems,
    )


def _paired_repeats_gate(
    by_method: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    maps: dict[str, dict[int, dict[str, Any]]] = {}
    duplicates: dict[str, list[Any]] = {}
    for method, rows in by_method.items():
        maps[method], duplicates[method] = _repeat_map(rows)

    repeat_sets = {method: set(mapping) for method, mapping in maps.items()}
    counts = {method: len(mapping) for method, mapping in maps.items()}
    same_indices = len({frozenset(indices) for indices in repeat_sets.values()}) == 1
    common = set.intersection(*repeat_sets.values()) if repeat_sets else set()
    problems: list[str] = []
    if any(duplicates.values()):
        problems.append("invalid or duplicate repeat identifiers are present")
    if not same_indices:
        problems.append(
            "the three methods do not have exactly the same repeat identifiers"
        )
    if len(common) < MIN_PAIRED_REPEATS:
        problems.append(
            f"only {len(common)} exact pairs; at least {MIN_PAIRED_REPEATS} are required"
        )
    return (
        _gate(
            "pass" if not problems else "fail",
            (
                "at least five exact method-wise pairs are present"
                if not problems
                else "paired-repeat audit failed"
            ),
            repeat_counts=counts,
            paired_repeat_indices=sorted(common),
            duplicate_or_invalid_repeat_ids=duplicates,
            problems=problems,
        ),
        maps,
    )


def _convergence_gate(
    manifest: dict[str, Any], by_method: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    manifest_tol = _finite_number(manifest.get("tolerance"))
    problems: list[str] = []
    audited = 0
    for method in REQUIRED_METHODS:
        if not by_method[method]:
            problems.append(f"no measured {method} rows")
        for row in by_method[method]:
            audited += 1
            label = f"{method} repeat {row.get('repeat_idx')}"
            row_tol = _finite_number(row.get("tol"))
            true_relres = _finite_number(row.get("true_relres"))
            if row.get("status") != "converged":
                problems.append(f"{label} status={row.get('status')!r}")
            if row_tol is None:
                problems.append(f"{label} has no finite tolerance")
            elif manifest_tol is not None and not math.isclose(
                row_tol, manifest_tol, rel_tol=1e-12, abs_tol=0.0
            ):
                problems.append(f"{label} tolerance differs from manifest")
            if true_relres is None:
                problems.append(f"{label} has no finite audited true residual")
            elif row_tol is not None and true_relres > row_tol:
                problems.append(
                    f"{label} true_relres={true_relres:g} exceeds tol={row_tol:g}"
                )
    return _gate(
        "pass" if not problems else "fail",
        (
            "every measured solve reaches its audited tolerance"
            if not problems
            else "convergence audit failed"
        ),
        manifest_tolerance=manifest_tol,
        audited_rows=audited,
        problems=problems,
    )


def _constant_positive_rank(rows: list[dict[str, Any]]) -> tuple[int | None, list[Any]]:
    raw = [row.get("rank") for row in rows]
    ranks = {_int_value(value) for value in raw}
    if len(ranks) != 1:
        return None, raw
    rank = next(iter(ranks))
    return (rank if rank is not None and rank > 0 else None), raw


def _paired_performance_gates(
    repeat_maps: dict[str, dict[int, dict[str, Any]]], paired_gate: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    default_rank, default_raw_ranks = _constant_positive_rank(
        list(repeat_maps["default"].values())
    )
    full_rank, full_raw_ranks = _constant_positive_rank(
        list(repeat_maps["full-eig"].values())
    )
    rank_problems: list[str] = []
    if default_rank is None:
        rank_problems.append(
            "default does not have one positive rank across measured repeats"
        )
    if full_rank is None:
        rank_problems.append(
            "full-eig does not have one positive rank across measured repeats"
        )
    if default_rank is not None and full_rank is not None and default_rank != full_rank:
        rank_problems.append(
            f"default rank {default_rank} differs from full-eig rank {full_rank}"
        )
    rank_gate = _gate(
        "pass" if not rank_problems else "fail",
        (
            "default and full-eig use the same rank"
            if not rank_problems
            else "same-rank audit failed"
        ),
        default_rank=default_rank,
        full_eig_rank=full_rank,
        default_raw_ranks=default_raw_ranks,
        full_eig_raw_ranks=full_raw_ranks,
        problems=rank_problems,
    )

    common = sorted(
        set(repeat_maps["cg"])
        & set(repeat_maps["default"])
        & set(repeat_maps["full-eig"])
    )
    time_problems: list[str] = []
    times: dict[str, dict[int, float]] = {method: {} for method in REQUIRED_METHODS}
    for method in REQUIRED_METHODS:
        for repeat_idx in common:
            value = _finite_number(
                repeat_maps[method][repeat_idx].get("build_plus_solve_seconds")
            )
            if value is None or value <= 0.0:
                time_problems.append(
                    f"{method} repeat {repeat_idx} has invalid cold time"
                )
            else:
                times[method][repeat_idx] = value

    medians: dict[str, float | None] = {"cg": None, "full-eig": None}
    wins: dict[str, int] = {"cg": 0, "full-eig": 0}
    if not time_problems and common:
        for baseline in ("cg", "full-eig"):
            ratios = [times[baseline][idx] / times["default"][idx] for idx in common]
            medians[baseline] = float(statistics.median(ratios))
            wins[baseline] = sum(
                times["default"][idx] < times[baseline][idx] for idx in common
            )

    speed_problems = list(time_problems)
    if paired_gate["status"] != "pass":
        speed_problems.append("paired-repeat gate did not pass")
    if rank_gate["status"] != "pass":
        speed_problems.append("same-rank gate did not pass")
    for baseline, median in medians.items():
        if median is None or median < MIN_MEDIAN_SPEEDUP:
            speed_problems.append(
                f"median {baseline}/default cold speedup is below {MIN_MEDIAN_SPEEDUP:g}"
            )
    speed_gate = _gate(
        "pass" if not speed_problems else "fail",
        (
            "default is at least 1.25x faster in paired cold-time medians"
            if not speed_problems
            else "cold-time speedup gate failed"
        ),
        speedup_definition="baseline build+solve / default build+solve",
        minimum_median_speedup=MIN_MEDIAN_SPEEDUP,
        median_speedup_over_cg=medians["cg"],
        median_speedup_over_same_rank_full_eig=medians["full-eig"],
        problems=speed_problems,
    )

    required_wins = max(4, math.ceil(MIN_PAIRED_WIN_FRACTION * len(common)))
    win_problems = list(time_problems)
    if paired_gate["status"] != "pass":
        win_problems.append("paired-repeat gate did not pass")
    for baseline in ("cg", "full-eig"):
        if wins[baseline] < required_wins:
            win_problems.append(
                f"default beats {baseline} in {wins[baseline]}/{len(common)} pairs; "
                f"{required_wins}/{len(common)} are required"
            )
    win_gate = _gate(
        "pass" if not win_problems else "fail",
        (
            "default wins at least 80% of pairs against both baselines"
            if not win_problems
            else "paired-win gate failed"
        ),
        paired_repeats=len(common),
        minimum_win_fraction=MIN_PAIRED_WIN_FRACTION,
        required_wins=required_wins,
        wins_over_cg=wins["cg"],
        wins_over_full_eig=wins["full-eig"],
        problems=win_problems,
    )
    return rank_gate, speed_gate, win_gate


def _strict_box_gate(
    manifest: dict[str, Any], by_method: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    M = _int_value(manifest.get("M"))
    row_sizes = [_int_value(row.get("box_size")) for row in by_method["default"]]
    distinct_sizes = {value for value in row_sizes if value is not None}
    box_size = next(iter(distinct_sizes)) if len(distinct_sizes) == 1 else None
    manifest_box_size = _int_value(manifest.get("score_box_size"))
    problems: list[str] = []
    if M is None or M <= 0:
        problems.append("manifest M is not a positive integer")
    if len(distinct_sizes) != 1 or any(value is None for value in row_sizes):
        problems.append("default box_size is absent or changes across repeats")
    if M is not None and box_size is not None and not (0 < box_size < M):
        problems.append(f"default box is not strict: |B|={box_size}, M={M}")
    if (
        manifest_box_size is not None
        and box_size is not None
        and manifest_box_size != box_size
    ):
        problems.append(
            "manifest score_box_size differs from measured default box_size"
        )
    return _gate(
        "pass" if not problems else "fail",
        (
            "default uses one nonempty strict box"
            if not problems
            else "strict-box audit failed"
        ),
        box_size=box_size,
        manifest_box_size=manifest_box_size,
        fourier_dimension=M,
        problems=problems,
    )


def _memory_cap_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    explicit = manifest.get("score_cap_excludes_requested_threshold_modes")
    raw_size = _int_value(manifest.get("score_tau_raw_box_size"))
    selected_size = _int_value(manifest.get("score_box_size"))
    signals: dict[str, bool] = {}
    if isinstance(explicit, bool):
        signals["score_cap_excludes_requested_threshold_modes"] = explicit
    if raw_size is not None and selected_size is not None:
        signals["score_box_size versus score_tau_raw_box_size"] = (
            selected_size < raw_size
        )
    if not signals:
        return _gate(
            "not_applicable",
            "manifest does not expose whether the threshold box was memory-capped",
            score_tau_raw_box_size=raw_size,
            score_box_size=selected_size,
        )
    capped = any(signals.values())
    inconsistent = len(set(signals.values())) > 1
    return _gate(
        "fail" if capped or inconsistent else "pass",
        (
            "memory-cap metadata is inconsistent"
            if inconsistent
            else (
                "memory cap excluded threshold-selected modes"
                if capped
                else "threshold-selected modes were not excluded by a memory cap"
            )
        ),
        metadata_signals=signals,
        score_cap_excludes_requested_threshold_modes=explicit,
        score_tau_raw_box_size=raw_size,
        score_box_size=selected_size,
    )


def _score_diagnostic_gate(
    diagnostics: Any | None, manifest: dict[str, Any]
) -> dict[str, Any]:
    if diagnostics is None:
        return _gate(
            "pending",
            "score/leverage diagnostics are absent; this gate cannot pass yet",
            minimum_capture=MIN_LEVERAGE_CAPTURE,
            maximum_score_box_fraction=MAX_SCORE_BOX_FRACTION,
        )
    if not isinstance(diagnostics, list) or not all(
        isinstance(row, dict) for row in diagnostics
    ):
        return _gate("fail", "post_diagnostics.json is not a list of records")

    metric_rows = [
        row
        for row in diagnostics
        if _finite_number(row.get("score_box_leverage_capture")) is not None
        and _finite_number(row.get("score_box_fraction")) is not None
    ]
    candidates = [
        row
        for row in metric_rows
        if not manifest.get("system_id")
        or row.get("system_id") == manifest.get("system_id")
    ]
    if metric_rows and not candidates:
        return _gate(
            "fail",
            "score/leverage diagnostics belong to a different system",
            manifest_system_id=manifest.get("system_id"),
            diagnostic_system_ids=sorted(
                {str(row.get("system_id") or "") for row in metric_rows}
            ),
        )
    full_candidates = [row for row in candidates if row.get("method") == "full-eig"]
    if full_candidates:
        candidates = full_candidates
    if not candidates:
        return _gate(
            "pending",
            "diagnostics contain no score-box leverage capture and box fraction",
            minimum_capture=MIN_LEVERAGE_CAPTURE,
            maximum_score_box_fraction=MAX_SCORE_BOX_FRACTION,
        )
    if len(candidates) != 1:
        return _gate(
            "fail",
            "diagnostics contain multiple eligible score/leverage records",
            eligible_records=len(candidates),
        )

    row = candidates[0]
    capture = float(row["score_box_leverage_capture"])
    fraction = float(row["score_box_fraction"])
    enrichment = capture / fraction if fraction > 0.0 else None
    diagnostic_box_size = _int_value(row.get("score_box_size"))
    manifest_box_size = _int_value(manifest.get("score_box_size"))
    M = _int_value(manifest.get("M"))
    problems: list[str] = []
    if row.get("diagnostic_status") not in (None, "ok"):
        problems.append(f"diagnostic_status={row.get('diagnostic_status')!r}")
    if not (0.0 <= capture <= 1.0 + 1e-10):
        problems.append("leverage capture is outside [0, 1]")
    if not (0.0 < fraction < 1.0):
        problems.append("score box fraction is not strict")
    if (
        diagnostic_box_size is not None
        and manifest_box_size is not None
        and diagnostic_box_size != manifest_box_size
    ):
        problems.append("diagnostic score_box_size differs from the manifest box")
    if (
        diagnostic_box_size is not None
        and M is not None
        and M > 0
        and not math.isclose(
            fraction, diagnostic_box_size / M, rel_tol=1e-10, abs_tol=1e-12
        )
    ):
        problems.append("diagnostic score_box_fraction is inconsistent with |B|/M")
    if capture < MIN_LEVERAGE_CAPTURE:
        problems.append(
            f"leverage capture {capture:g} is below {MIN_LEVERAGE_CAPTURE:g}"
        )
    if fraction > MAX_SCORE_BOX_FRACTION:
        problems.append(
            f"score box fraction {fraction:g} exceeds {MAX_SCORE_BOX_FRACTION:g}"
        )
    return _gate(
        "pass" if not problems else "fail",
        (
            "score-selected box captures concentrated dominant-subspace leverage"
            if not problems
            else "score/leverage diagnostic gate failed"
        ),
        diagnostic_method=row.get("method"),
        score_box_leverage_capture=capture,
        score_box_fraction=fraction,
        diagnostic_score_box_size=diagnostic_box_size,
        manifest_score_box_size=manifest_box_size,
        leverage_enrichment=enrichment,
        minimum_capture=MIN_LEVERAGE_CAPTURE,
        maximum_score_box_fraction=MAX_SCORE_BOX_FRACTION,
        problems=problems,
    )


def evaluate_strict_box_run(run_dir: str | Path) -> dict[str, Any]:
    """Audit one controlled run directory against the fixed strict-box gates."""

    path = Path(run_dir).resolve()
    manifest = _load_json(path / "system_manifest.json")
    rows = _load_json(path / "matched_runs.json")
    diagnostics = _load_json(path / "post_diagnostics.json", required=False)
    if not isinstance(manifest, dict):
        raise AcceptanceInputError("system_manifest.json must contain one object")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise AcceptanceInputError("matched_runs.json must contain a list of records")

    by_method = _measured_rows_by_method(rows)
    paired_gate, repeat_maps = _paired_repeats_gate(by_method)
    rank_gate, speed_gate, wins_gate = _paired_performance_gates(
        repeat_maps, paired_gate
    )
    criteria = {
        "fp64": _fp64_gate(manifest, by_method),
        "same_system": _same_system_gate(manifest, by_method),
        "audited_convergence": _convergence_gate(manifest, by_method),
        "paired_repeats": paired_gate,
        "same_rank_full_eig": rank_gate,
        "cold_speedup": speed_gate,
        "paired_wins": wins_gate,
        "strict_box": _strict_box_gate(manifest, by_method),
        "threshold_not_memory_capped": _memory_cap_gate(manifest),
        "score_leverage": _score_diagnostic_gate(diagnostics, manifest),
    }
    failed = [name for name, gate in criteria.items() if gate["status"] == "fail"]
    pending = [name for name, gate in criteria.items() if gate["status"] == "pending"]
    overall = "fail" if failed else ("pending" if pending else "pass")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(path),
        "status": overall,
        "eligible": overall == "pass",
        "predeclared_thresholds": {
            "minimum_paired_repeats": MIN_PAIRED_REPEATS,
            "minimum_median_cold_speedup": MIN_MEDIAN_SPEEDUP,
            "minimum_paired_win_fraction": MIN_PAIRED_WIN_FRACTION,
            "minimum_score_box_leverage_capture": MIN_LEVERAGE_CAPTURE,
            "maximum_score_box_fraction": MAX_SCORE_BOX_FRACTION,
        },
        "failed_criteria": failed,
        "pending_criteria": pending,
        "criteria": criteria,
    }


def _error_report(run_dir: str | Path, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(Path(run_dir).resolve()),
        "status": "fail",
        "eligible": False,
        "failed_criteria": ["input_files"],
        "pending_criteria": [],
        "criteria": {
            "input_files": _gate("fail", str(exc), exception_type=type(exc).__name__)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a controlled run against predeclared strict-box criteria."
    )
    parser.add_argument(
        "run_dir", help="directory containing the controlled JSON outputs"
    )
    parser.add_argument("--output", type=Path, help="optional path for the JSON report")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)

    try:
        report = evaluate_strict_box_run(args.run_dir)
    except (AcceptanceInputError, OSError, ValueError) as exc:
        report = _error_report(args.run_dir, exc)
    indent = None if args.compact else 2
    rendered = json.dumps(report, indent=indent, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return {"pass": 0, "fail": 1, "pending": 2}[report["status"]]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
