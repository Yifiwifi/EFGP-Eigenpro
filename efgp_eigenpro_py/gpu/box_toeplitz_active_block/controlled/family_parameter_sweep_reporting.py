"""Reporting and selection for the two preconditioner parameter families.

This module deliberately does not launch experiments.  It consumes the
``pipeline_summary.json`` or ``pipeline_summary.csv`` artifacts produced by a
suite plan, records every inverse/eigen-family candidate, and selects one
fastest successful median candidate for each dataset/size/family group.

RMSE is carried into both reports for scientific inspection, but it is never
used to filter or rank candidates.  A candidate is selectable if and only if
its status is ``ok``, it has exactly three successful repeats, and its median
training total is finite.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_SUCCESSFUL_REPEATS = 3
SELECTION_RULE = (
    "status == 'ok' and successful_repeats == 3 and "
    "finite(train_total_seconds_median); minimize "
    "train_total_seconds_median; RMSE is reported only and is not used for "
    "filtering or ranking"
)

# ``ours-binned-active-eig`` and the full-grid reference route tune the same
# B/q eigenpair family.  Route remains a separate output column so that the
# selected implementation is never hidden by the family grouping.
METHOD_FAMILY = {
    "ours-binned-inverse": "inverse",
    "ours-binned-active-eig": "eigen",
    "efgp-standard-full-eig": "eigen",
}
METHOD_ROUTE = {
    "ours-binned-inverse": "localized",
    "ours-binned-active-eig": "localized",
    "efgp-standard-full-eig": "full-grid",
}

GROUP_FIELDS = (
    "dataset_family",
    "n_train",
    "parameter_family",
)

CANDIDATE_FIELDS = (
    "suite_profile",
    "case_id",
    "dataset_family",
    "dataset_stem",
    "n_train",
    "kernel_family",
    "parameter_family",
    "method",
    "route",
    "B",
    "q",
    "active_topk",
    "train_total_seconds_median",
    "train_total_seconds_max",
    "setup_seconds_median",
    "solving_phase_seconds_median",
    "test_rmse_median",
    "iterations_median",
    "status",
    "measured_repeats",
    "successful_repeats",
    "expected_measured_repeats",
    "selection_eligible",
    "selection_ineligibility_reason",
    "selection_rule",
    "rmse_used_for_selection",
    "summary_path",
    "summary_row_index",
)

WINNER_FIELDS = CANDIDATE_FIELDS + (
    "selection_rank",
    "fastest_time_tie_count",
)

UNBINNED_CG_METHOD = "efgp-standard-cg"
UNBINNED_CG_SPEEDUP_DEFINITION = (
    "unbinned_cg_train_total_seconds_median / "
    "family_train_total_seconds_median; values above 1 mean the selected "
    "preconditioner family is faster"
)
UNBINNED_CG_COMPARISON_FIELDS = (
    "dataset_family",
    "n_train",
    "parameter_family",
    "family_method",
    "B",
    "q",
    "active_topk",
    "unbinned_cg_method",
    "unbinned_cg_train_total_seconds_median",
    "unbinned_cg_test_rmse_median",
    "family_train_total_seconds_median",
    "family_test_rmse_median",
    "speedup_unbinned_cg_over_family",
)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    try:
        return dict(vars(value))
    except TypeError as exc:
        raise TypeError(
            "suite plan entries and their configs must be mappings, dataclasses, "
            "or objects with attributes"
        ) from exc


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _require_finite_metric(
    value: Any,
    *,
    label: str,
    strictly_positive: bool = False,
) -> float:
    number = _optional_float(value)
    if number is None or strictly_positive and number <= 0.0:
        qualifier = "finite and positive" if strictly_positive else "finite"
        raise ValueError(f"{label} must be {qualifier}, got {value!r}.")
    return number


def _require_path_within(path: str | Path, root: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    resolved_root = Path(root).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} is outside current_run_root: {resolved} not under "
            f"{resolved_root}."
        ) from exc
    return resolved


def _resolve_summary_path(plan_entry: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
    explicit = _first_present(
        plan_entry.get("pipeline_summary_path"),
        plan_entry.get("summary_path"),
        config.get("pipeline_summary_path"),
        config.get("summary_path"),
    )
    if explicit is not None:
        path = Path(str(explicit))
        if path.is_dir():
            for name in ("pipeline_summary.json", "pipeline_summary.csv"):
                candidate = path / name
                if candidate.is_file():
                    return candidate
        elif path.is_file():
            return path
        raise FileNotFoundError(f"pipeline summary does not exist: {path}")

    output_dir = _first_present(plan_entry.get("output_dir"), config.get("output_dir"))
    if output_dir is None:
        raise ValueError(
            "suite plan entry must provide config.output_dir, output_dir, or an "
            "explicit pipeline_summary_path"
        )
    output = Path(str(output_dir))
    # Prefer JSON when a normal suite run emitted both formats; unlike CSV it
    # preserves native booleans, nulls, and numeric types.
    for name in ("pipeline_summary.json", "pipeline_summary.csv"):
        candidate = output / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no pipeline_summary.json or pipeline_summary.csv found in {output}"
    )


def load_pipeline_summary(path: str | Path) -> list[dict[str, Any]]:
    """Load one pipeline summary in the suite's JSON or CSV representation."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = _first_present(
                payload.get("pipeline_summary"),
                payload.get("rows"),
                payload.get("candidates"),
            )
        if not isinstance(payload, list) or not all(
            isinstance(row, Mapping) for row in payload
        ):
            raise ValueError(
                f"pipeline summary JSON must contain a list of objects: {source}"
            )
        return [dict(row) for row in payload]
    raise ValueError(f"pipeline summary must be a .json or .csv file: {source}")


def _method_parameters(
    method: str,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None]:
    effective_b = _optional_int(row.get("effective_active_box_size"))
    effective_topk = _optional_int(row.get("effective_active_topk"))
    effective_q = _optional_int(row.get("effective_active_rank"))

    if method == "ours-binned-inverse":
        topk = _first_present(
            effective_topk,
            _optional_int(config.get("inverse_active_topk")),
            _optional_int(config.get("active_topk")),
            _optional_int(row.get("configured_active_topk")),
        )
        box_size = _first_present(
            effective_b,
            _optional_int(config.get("inverse_expected_active_box_size")),
            _optional_int(config.get("expected_active_box_size")),
            _optional_int(row.get("configured_expected_active_box_size")),
        )
        return _optional_int(box_size), None, _optional_int(topk)

    if method == "ours-binned-active-eig":
        topk = _first_present(
            effective_topk,
            _optional_int(config.get("active_eig_topk")),
            _optional_int(config.get("active_topk")),
            _optional_int(row.get("configured_active_topk")),
        )
        box_size = _first_present(
            effective_b,
            _optional_int(config.get("active_eig_expected_active_box_size")),
            _optional_int(config.get("expected_active_box_size")),
            _optional_int(row.get("configured_expected_active_box_size")),
        )
        q = _first_present(
            effective_q,
            _optional_int(config.get("active_eig_rank")),
            _optional_int(config.get("rank")),
            _optional_int(row.get("configured_active_rank")),
        )
        return _optional_int(box_size), _optional_int(q), _optional_int(topk)

    # The only remaining recognized method is the full-grid eigen route.
    topk = _first_present(
        effective_topk,
        _optional_int(config.get("active_topk")),
        _optional_int(row.get("configured_active_topk")),
    )
    box_size = _first_present(
        effective_b,
        _optional_int(config.get("expected_active_box_size")),
        _optional_int(row.get("configured_expected_active_box_size")),
    )
    q = _first_present(
        effective_q,
        _optional_int(config.get("full_eig_rank")),
        _optional_int(config.get("rank")),
        _optional_int(row.get("configured_full_eig_rank")),
    )
    return _optional_int(box_size), _optional_int(q), _optional_int(topk)


def _eligibility(
    *,
    status: str,
    successful_repeats: int | None,
    train_total_seconds_median: float | None,
    expected_successful_repeats: int,
) -> tuple[bool, str]:
    reasons: list[str] = []
    if status != "ok":
        reasons.append("status_not_ok")
    if successful_repeats != expected_successful_repeats:
        reasons.append(f"successful_repeats_not_{expected_successful_repeats}")
    if train_total_seconds_median is None:
        reasons.append("train_total_not_finite")
    return not reasons, ";".join(reasons)


def collect_family_parameter_sweep_candidates(
    suite_plan: Iterable[Mapping[str, Any] | Any],
    *,
    expected_successful_repeats: int = EXPECTED_SUCCESSFUL_REPEATS,
) -> list[dict[str, Any]]:
    """Collect normalized inverse/eigen candidates from a suite plan.

    Unrecognized methods are intentionally omitted because they do not belong
    to either B-only or B/q parameter family.  Failed and incomplete rows from
    recognized methods remain in the all-candidates result.
    """

    if expected_successful_repeats != EXPECTED_SUCCESSFUL_REPEATS:
        raise ValueError(
            "family sweep selection is fixed at exactly three successful repeats"
        )

    candidates: list[dict[str, Any]] = []
    for entry_value in suite_plan:
        entry = _mapping(entry_value)
        config = _mapping(entry.get("config"))
        source = _resolve_summary_path(entry, config)
        rows = load_pipeline_summary(source)
        for row_index, row in enumerate(rows):
            method = str(row.get("method", "")).strip()
            if method not in METHOD_FAMILY:
                continue

            status = str(row.get("status", "")).strip()
            successful_repeats = _optional_int(row.get("successful_repeats"))
            train_total = _optional_float(row.get("train_total_seconds_median"))
            eligible, reason = _eligibility(
                status=status,
                successful_repeats=successful_repeats,
                train_total_seconds_median=train_total,
                expected_successful_repeats=expected_successful_repeats,
            )
            box_size, q, topk = _method_parameters(method, row, config)

            candidate = {
                "suite_profile": _first_present(
                    entry.get("profile"), entry.get("suite_profile"), row.get("suite_profile")
                ),
                "case_id": _first_present(entry.get("case_id"), row.get("case_id")),
                "dataset_family": _first_present(
                    entry.get("dataset_family"),
                    row.get("dataset_family"),
                    config.get("dataset_family"),
                ),
                "dataset_stem": _first_present(
                    row.get("dataset_stem"), config.get("dataset_stem")
                ),
                "n_train": _optional_int(
                    _first_present(row.get("n_train"), config.get("n_train"))
                ),
                "kernel_family": _first_present(
                    row.get("kernel_family"), config.get("kernel_family")
                ),
                "parameter_family": METHOD_FAMILY[method],
                "method": method,
                "route": METHOD_ROUTE[method],
                "B": box_size,
                "q": q,
                "active_topk": topk,
                "train_total_seconds_median": train_total,
                "train_total_seconds_max": _optional_float(
                    row.get("train_total_seconds_max")
                ),
                "setup_seconds_median": _optional_float(
                    row.get("setup_seconds_median")
                ),
                "solving_phase_seconds_median": _optional_float(
                    row.get("solving_phase_seconds_median")
                ),
                "test_rmse_median": _optional_float(row.get("test_rmse_median")),
                "iterations_median": _optional_float(row.get("iterations_median")),
                "status": status,
                "measured_repeats": _optional_int(row.get("measured_repeats")),
                "successful_repeats": successful_repeats,
                "expected_measured_repeats": _optional_int(
                    _first_present(
                        row.get("expected_measured_repeats"),
                        config.get("measured_repeats"),
                    )
                ),
                "selection_eligible": eligible,
                "selection_ineligibility_reason": reason,
                "selection_rule": SELECTION_RULE,
                "rmse_used_for_selection": False,
                "summary_path": str(source),
                "summary_row_index": row_index,
            }
            candidates.append(candidate)

    return sorted(candidates, key=_candidate_sort_key)


def _sortable_text(value: Any) -> str:
    return "" if value is None else str(value)


def _sortable_int(value: Any) -> tuple[int, int]:
    number = _optional_int(value)
    return (1, 0) if number is None else (0, number)


def _group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in GROUP_FIELDS)


def _group_sort_key(group: Sequence[Any]) -> tuple[Any, ...]:
    dataset_family, n_train, parameter_family = group
    return (
        _sortable_text(dataset_family),
        _sortable_int(n_train),
        _sortable_text(parameter_family),
    )


def _candidate_tie_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _sortable_text(row.get("method")),
        _sortable_int(row.get("B")),
        _sortable_int(row.get("q")),
        _sortable_int(row.get("active_topk")),
        _sortable_text(row.get("suite_profile")),
        _sortable_text(row.get("case_id")),
        _sortable_text(row.get("summary_path")),
        _sortable_int(row.get("summary_row_index")),
    )


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return _group_sort_key(_group_key(row)) + _candidate_tie_key(row)


def select_fastest_successful_medians(
    candidates: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select the minimum finite training-time median in every family group."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for candidate_value in candidates:
        candidate = dict(candidate_value)
        total = _optional_float(candidate.get("train_total_seconds_median"))
        eligible, _ = _eligibility(
            status=str(candidate.get("status", "")).strip(),
            successful_repeats=_optional_int(candidate.get("successful_repeats")),
            train_total_seconds_median=total,
            expected_successful_repeats=EXPECTED_SUCCESSFUL_REPEATS,
        )
        # Recompute all three invariants so hand-built or edited candidate
        # tables cannot bypass the collector's fail-closed normalization.
        if not eligible:
            continue
        candidate["train_total_seconds_median"] = total
        grouped.setdefault(_group_key(candidate), []).append(candidate)

    winners: list[dict[str, Any]] = []
    for group in sorted(grouped, key=_group_sort_key):
        eligible = grouped[group]
        fastest_time = min(row["train_total_seconds_median"] for row in eligible)
        tied = [
            row
            for row in eligible
            if row["train_total_seconds_median"] == fastest_time
        ]
        winner = min(tied, key=_candidate_tie_key)
        winner = dict(winner)
        winner["selection_rank"] = 1
        winner["fastest_time_tie_count"] = len(tied)
        winners.append(winner)
    return winners


def build_family_parameter_sweep_reports(
    suite_plan: Iterable[Mapping[str, Any] | Any],
) -> dict[str, Any]:
    """Build both in-memory reports without writing files."""

    candidates = collect_family_parameter_sweep_candidates(suite_plan)
    winners = select_fastest_successful_medians(candidates)
    return {
        "all_candidates": candidates,
        "selected_winners": winners,
        "fastest_successful_median_winners": winners,
        "selection_rule": SELECTION_RULE,
        "expected_successful_repeats": EXPECTED_SUCCESSFUL_REPEATS,
        "group_fields": list(GROUP_FIELDS),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_family_parameter_sweep_reports(
    suite_plan: Iterable[Mapping[str, Any] | Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build and write all-candidate, winner, and selection-manifest artifacts."""

    report = build_family_parameter_sweep_reports(suite_plan)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_json = output / "all_candidates.json"
    all_csv = output / "all_candidates.csv"
    winners_json = output / "selected_winners.json"
    winners_csv = output / "selected_winners.csv"
    manifest_json = output / "family_parameter_sweep_manifest.json"

    candidates = report["all_candidates"]
    winners = report["selected_winners"]
    _write_json(all_json, candidates)
    _write_csv(all_csv, candidates, CANDIDATE_FIELDS)
    _write_json(winners_json, winners)
    _write_csv(winners_csv, winners, WINNER_FIELDS)

    manifest = {
        "schema_version": 1,
        "selection_rule": SELECTION_RULE,
        "expected_successful_repeats": EXPECTED_SUCCESSFUL_REPEATS,
        "rmse_used_for_selection": False,
        "group_fields": list(GROUP_FIELDS),
        "candidate_count": len(candidates),
        "selection_eligible_count": sum(
            row["selection_eligible"] is True for row in candidates
        ),
        "winner_count": len(winners),
        "source_pipeline_summaries": sorted(
            {row["summary_path"] for row in candidates}
        ),
        "artifacts": {
            "all_candidates_json": str(all_json),
            "all_candidates_csv": str(all_csv),
            "selected_winners_json": str(winners_json),
            "selected_winners_csv": str(winners_csv),
        },
    }
    _write_json(manifest_json, manifest)

    return {
        **report,
        "manifest": manifest,
        "paths": {
            "all_candidates_json": all_json,
            "all_candidates_csv": all_csv,
            "selected_winners_json": winners_json,
            "selected_winners_csv": winners_csv,
            "manifest_json": manifest_json,
        },
    }


def collect_unbinned_cg_reference_rows(
    suite_plan: Iterable[Mapping[str, Any] | Any],
    *,
    current_run_root: str | Path,
) -> list[dict[str, Any]]:
    """Load one complete three-repeat unbinned-CG median per dataset/size.

    The run-root containment check prevents an extension notebook from silently
    mixing the freshly measured family winners with an archived CG timing.  A
    missing or incomplete CG row raises instead of creating a partial speedup
    table.
    """

    references: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for entry_value in suite_plan:
        entry = _mapping(entry_value)
        config = _mapping(entry.get("config"))
        methods = tuple(str(method) for method in config.get("methods", ()))
        if methods != (UNBINNED_CG_METHOD,):
            raise ValueError(
                "unbinned-CG comparison plan must contain only "
                f"{UNBINNED_CG_METHOD!r}; got {methods!r}."
            )
        if _optional_int(config.get("measured_repeats")) != EXPECTED_SUCCESSFUL_REPEATS:
            raise ValueError(
                "unbinned-CG comparison requires exactly three measured repeats."
            )

        source = _resolve_summary_path(entry, config)
        source = _require_path_within(
            source,
            current_run_root,
            label="unbinned-CG pipeline summary",
        )
        rows = load_pipeline_summary(source)
        if len(rows) != 1 or str(rows[0].get("method", "")).strip() != UNBINNED_CG_METHOD:
            raise ValueError(
                "each unbinned-CG case must contain exactly one "
                f"{UNBINNED_CG_METHOD!r} summary row: {source}"
            )
        row = rows[0]
        status = str(row.get("status", "")).strip().lower()
        successful_repeats = _optional_int(row.get("successful_repeats"))
        if status != "ok" or successful_repeats != EXPECTED_SUCCESSFUL_REPEATS:
            raise ValueError(
                "unbinned-CG row is not a complete three-repeat measurement: "
                f"{source}"
            )

        dataset_family = str(
            _first_present(
                entry.get("dataset_family"),
                row.get("dataset_family"),
                config.get("dataset_family"),
            )
            or ""
        ).strip()
        n_train = _optional_int(
            _first_present(row.get("n_train"), config.get("n_train"))
        )
        if not dataset_family or n_train is None or n_train <= 0:
            raise ValueError(
                f"unbinned-CG row lacks a dataset family or positive N: {source}"
            )
        key = (dataset_family, n_train)
        if key in seen:
            raise ValueError(f"duplicate unbinned-CG dataset/size row: {key!r}.")
        seen.add(key)

        references.append(
            {
                "dataset_family": dataset_family,
                "n_train": n_train,
                "method": UNBINNED_CG_METHOD,
                "status": status,
                "train_total_seconds_median": _require_finite_metric(
                    row.get("train_total_seconds_median"),
                    label=f"{key!r} unbinned-CG training-time median",
                    strictly_positive=True,
                ),
                "test_rmse_median": _require_finite_metric(
                    row.get("test_rmse_median"),
                    label=f"{key!r} unbinned-CG test-RMSE median",
                ),
                "train_total_seconds_max": _optional_float(
                    row.get("train_total_seconds_max")
                ),
                "successful_repeats": successful_repeats,
                "summary_path": str(source),
                "summary_row_index": 0,
            }
        )
    if not references:
        raise ValueError("unbinned-CG comparison plan is empty.")
    return sorted(
        references,
        key=lambda row: (
            _sortable_text(row.get("dataset_family")),
            _sortable_int(row.get("n_train")),
        ),
    )


def build_unbinned_cg_speedup_rows(
    cg_suite_plan: Iterable[Mapping[str, Any] | Any],
    selected_family_winners: Iterable[Mapping[str, Any]],
    *,
    current_run_root: str | Path,
) -> dict[str, Any]:
    """Align current-run CG medians with both selected B/q family medians."""

    cg_rows = collect_unbinned_cg_reference_rows(
        cg_suite_plan,
        current_run_root=current_run_root,
    )
    cg_by_key = {
        (str(row["dataset_family"]), int(row["n_train"])): row for row in cg_rows
    }

    winner_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    winner_sources: set[str] = set()
    for winner_value in selected_family_winners:
        winner = dict(winner_value)
        dataset_family = str(winner.get("dataset_family", "")).strip()
        n_train = _optional_int(winner.get("n_train"))
        parameter_family = str(winner.get("parameter_family", "")).strip()
        if not dataset_family or n_train is None or parameter_family not in {
            "inverse",
            "eigen",
        }:
            raise ValueError(
                "selected family winner lacks a valid dataset/N/family key: "
                f"{winner!r}"
            )
        if (
            str(winner.get("status", "")).strip().lower() != "ok"
            or _optional_int(winner.get("successful_repeats"))
            != EXPECTED_SUCCESSFUL_REPEATS
            or not _truthy(winner.get("selection_eligible"))
            or _optional_int(winner.get("selection_rank")) != 1
        ):
            raise ValueError(
                "selected family winner is not a complete rank-1 three-repeat "
                f"measurement: {(dataset_family, n_train, parameter_family)!r}."
            )
        source = _require_path_within(
            str(winner.get("summary_path", "")),
            current_run_root,
            label="family-winner pipeline summary",
        )
        if not source.is_file():
            raise FileNotFoundError(f"family-winner pipeline summary is missing: {source}")
        winner_sources.add(str(source))
        key = (dataset_family, n_train, parameter_family)
        if key in winner_by_key:
            raise ValueError(f"duplicate selected family winner: {key!r}.")
        cooked = dict(winner)
        cooked["train_total_seconds_median"] = _require_finite_metric(
            winner.get("train_total_seconds_median"),
            label=f"{key!r} family training-time median",
            strictly_positive=True,
        )
        cooked["test_rmse_median"] = _require_finite_metric(
            winner.get("test_rmse_median"),
            label=f"{key!r} family test-RMSE median",
        )
        winner_by_key[key] = cooked

    expected_winner_keys = {
        (dataset_family, n_train, parameter_family)
        for dataset_family, n_train in cg_by_key
        for parameter_family in ("inverse", "eigen")
    }
    observed_winner_keys = set(winner_by_key)
    if observed_winner_keys != expected_winner_keys:
        raise ValueError(
            "current-run CG/family coverage mismatch; "
            f"missing={sorted(expected_winner_keys - observed_winner_keys)!r}, "
            f"extra={sorted(observed_winner_keys - expected_winner_keys)!r}."
        )

    comparisons: list[dict[str, Any]] = []
    for dataset_family, n_train, parameter_family in sorted(
        expected_winner_keys,
        key=lambda key: (
            _sortable_text(key[0]),
            _sortable_int(key[1]),
            _sortable_text(key[2]),
        ),
    ):
        cg_row = cg_by_key[(dataset_family, n_train)]
        winner = winner_by_key[(dataset_family, n_train, parameter_family)]
        family_time = float(winner["train_total_seconds_median"])
        comparisons.append(
            {
                "dataset_family": dataset_family,
                "n_train": n_train,
                "parameter_family": parameter_family,
                "family_method": winner.get("method"),
                "B": _optional_int(winner.get("B")),
                "q": _optional_int(winner.get("q")),
                "active_topk": _optional_int(winner.get("active_topk")),
                "unbinned_cg_method": UNBINNED_CG_METHOD,
                "unbinned_cg_train_total_seconds_median": float(
                    cg_row["train_total_seconds_median"]
                ),
                "unbinned_cg_test_rmse_median": float(cg_row["test_rmse_median"]),
                "family_train_total_seconds_median": family_time,
                "family_test_rmse_median": float(winner["test_rmse_median"]),
                "speedup_unbinned_cg_over_family": float(
                    cg_row["train_total_seconds_median"]
                )
                / family_time,
            }
        )

    return {
        "comparisons": comparisons,
        "speedup_definition": UNBINNED_CG_SPEEDUP_DEFINITION,
        "comparison_scope": "current_run_unpaired_three_repeat_median_ratio",
        "expected_successful_repeats": EXPECTED_SUCCESSFUL_REPEATS,
        "cg_source_pipeline_summaries": sorted(
            str(row["summary_path"]) for row in cg_rows
        ),
        "family_source_pipeline_summaries": sorted(winner_sources),
    }


def write_unbinned_cg_comparison_reports(
    cg_suite_plan: Iterable[Mapping[str, Any] | Any],
    selected_family_winners: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    current_run_root: str | Path,
) -> dict[str, Any]:
    """Write the final current-run Time/RMSE and CG-over-family speedup table."""

    report = build_unbinned_cg_speedup_rows(
        cg_suite_plan,
        selected_family_winners,
        current_run_root=current_run_root,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    comparison_json = output / "unbinned_cg_vs_family_winners.json"
    comparison_csv = output / "unbinned_cg_vs_family_winners.csv"
    manifest_json = output / "unbinned_cg_comparison_manifest.json"
    _write_json(comparison_json, report["comparisons"])
    _write_csv(
        comparison_csv,
        report["comparisons"],
        UNBINNED_CG_COMPARISON_FIELDS,
    )
    manifest = {
        "schema_version": 1,
        "comparison_scope": report["comparison_scope"],
        "speedup_definition": report["speedup_definition"],
        "expected_successful_repeats": report["expected_successful_repeats"],
        "paired_repeats": False,
        "comparison_row_count": len(report["comparisons"]),
        "current_run_root": str(Path(current_run_root).resolve()),
        "cg_source_pipeline_summaries": report["cg_source_pipeline_summaries"],
        "family_source_pipeline_summaries": report[
            "family_source_pipeline_summaries"
        ],
        "artifacts": {
            "comparison_json": str(comparison_json),
            "comparison_csv": str(comparison_csv),
        },
    }
    _write_json(manifest_json, manifest)
    return {
        **report,
        "manifest": manifest,
        "paths": {
            "comparison_json": comparison_json,
            "comparison_csv": comparison_csv,
            "manifest_json": manifest_json,
        },
    }


def load_serialized_suite_plan(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON-serialized suite plan for the command-line entry point."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = _first_present(payload.get("plan"), payload.get("suite_plan"))
    if not isinstance(payload, list) or not all(
        isinstance(entry, Mapping) for entry in payload
    ):
        raise ValueError(f"suite plan JSON must contain a list of objects: {source}")
    return [dict(entry) for entry in payload]


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select fastest three-successful-repeat median winners for the "
            "inverse and eigen B/q parameter families."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--plan-json",
        help="JSON-serialized build_profile_plan result (configs are objects).",
    )
    source.add_argument(
        "--suite-config",
        help="End-to-end suite JSON; use with --profile and --suite-output-root.",
    )
    parser.add_argument("--profile", help="Profile to expand from --suite-config.")
    parser.add_argument(
        "--dataset-dir",
        default=".",
        help="Dataset directory used while reconstructing the suite plan.",
    )
    parser.add_argument(
        "--suite-output-root",
        help="Root containing the profile/case pipeline_summary artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for all_candidates, selected_winners, and the manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for an existing serialized or configured plan."""

    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    if args.plan_json:
        plan = load_serialized_suite_plan(args.plan_json)
    else:
        if not args.profile or not args.suite_output_root:
            parser.error(
                "--suite-config requires both --profile and --suite-output-root"
            )
        suite_path = Path(args.suite_config)
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        if not isinstance(suite, dict):
            parser.error("--suite-config must contain a JSON object")
        # Local import keeps the in-memory reporting API independent of the
        # runner while reusing its canonical profile expansion for the CLI.
        from .end_to_end_suite import build_profile_plan

        plan = build_profile_plan(
            suite,
            args.profile,
            dataset_dir=args.dataset_dir,
            output_root=args.suite_output_root,
        )

    result = write_family_parameter_sweep_reports(plan, args.output_dir)
    manifest = result["manifest"]
    print(
        json.dumps(
            {
                "candidate_count": manifest["candidate_count"],
                "selection_eligible_count": manifest["selection_eligible_count"],
                "winner_count": manifest["winner_count"],
                "output_dir": str(Path(args.output_dir)),
            },
            ensure_ascii=False,
        )
    )
    return 0


__all__ = [
    "EXPECTED_SUCCESSFUL_REPEATS",
    "SELECTION_RULE",
    "METHOD_FAMILY",
    "GROUP_FIELDS",
    "UNBINNED_CG_METHOD",
    "UNBINNED_CG_SPEEDUP_DEFINITION",
    "UNBINNED_CG_COMPARISON_FIELDS",
    "collect_family_parameter_sweep_candidates",
    "select_fastest_successful_medians",
    "build_family_parameter_sweep_reports",
    "write_family_parameter_sweep_reports",
    "collect_unbinned_cg_reference_rows",
    "build_unbinned_cg_speedup_rows",
    "write_unbinned_cg_comparison_reports",
    "load_pipeline_summary",
    "load_serialized_suite_plan",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
