"""Suite orchestration for the two-step end-to-end KRR campaign.

The scale profile runs first.  A target regime is then selected by a declared,
data-independent rule: all six pipeline rows must be present, core executable
baselines must succeed (a prospectively declared RPCholesky hardware
``resource_limit`` is retained as a scalability result), the proposed and
standard full-eig pipelines must lie inside the prospectively declared broad
absolute usable-quality range, and the standard EFGP-CG iteration count should
lie in the predeclared interval.  Only after the selection is written to disk
are lambda, lengthscale, box-budget, and dataset robustness cases materialized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

from .end_to_end import (
    END_TO_END_METHODS,
    STAGE2_SYSTEM_CONFIG_FIELDS,
    TIMING_SCOPE,
    EndToEndConfig,
    run_end_to_end_experiment,
)


DEFAULT_SUITE_PATH = Path(__file__).with_name("end_to_end_suite.json")

# Method parameters do not define the fixed A,b system, but the Stage-1 winner
# must carry them prospectively into robustness and Stage 2.  Summary fields use
# explicit names so a per-method ``rank`` diagnostic cannot overwrite them.
FROZEN_METHOD_CONFIG_FROM_SUMMARY = {
    "rank": "configured_active_rank",
    "full_eig_rank": "configured_full_eig_rank",
    "active_topk": "configured_active_topk",
    "expected_active_box_size": "configured_expected_active_box_size",
    "allow_frozen_topk_capacity_adaptation": (
        "configured_allow_frozen_topk_capacity_adaptation"
    ),
    "box_budget": "box_budget",
    "parameter_selection_policy": "parameter_selection_policy",
    "parameter_source": "parameter_source",
}
BUDGET_ADAPTIVE_PARAMETER_POLICY = "budget_adaptive_score_rule"
BUDGET_ADAPTIVE_PARAMETER_SOURCE = (
    "robustness box-budget axis; score_tau selection under the declared cap, "
    "separate from frozen historical-transfer rows"
)


class TargetSelectionError(RuntimeError):
    def __init__(self, message: str, rejections: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.rejections = rejections


def load_suite_config(path: str | Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("base"), dict):
        raise ValueError("end-to-end suite requires an object-valued 'base'.")
    if not isinstance(payload.get("profiles"), dict):
        raise ValueError("end-to-end suite requires an object-valued 'profiles'.")
    return payload


def _config_fields() -> set[str]:
    return {field.name for field in fields(EndToEndConfig)}


def _normalize_config(payload: dict[str, Any]) -> EndToEndConfig:
    cooked = dict(payload)
    if "methods" in cooked:
        cooked["methods"] = tuple(str(method) for method in cooked["methods"])
    unknown = sorted(set(cooked) - _config_fields())
    if unknown:
        raise ValueError(f"unknown EndToEndConfig fields: {unknown}")
    return EndToEndConfig(**cooked)


def build_profile_plan(
    suite: dict[str, Any],
    profile: str,
    *,
    dataset_dir: str,
    output_root: str | Path,
) -> list[dict[str, Any]]:
    try:
        profile_payload = suite["profiles"][str(profile)]
    except KeyError as exc:
        raise KeyError(f"unknown end-to-end profile {profile!r}") from exc
    if "cases" not in profile_payload:
        raise ValueError(
            f"profile {profile!r} is a template profile, not a runnable case list."
        )
    base = dict(suite["base"])
    base.update(profile_payload.get("overrides", {}))
    plan: list[dict[str, Any]] = []
    for case in profile_payload["cases"]:
        case_id = str(case["id"])
        merged = dict(base)
        merged.update(
            {
                key: value
                for key, value in case.items()
                if key not in {"id", "dataset_family"}
            }
        )
        merged["dataset_dir"] = str(dataset_dir)
        merged["output_dir"] = str(Path(output_root) / str(profile) / case_id)
        config = _normalize_config(merged)
        plan.append(
            {
                "profile": str(profile),
                "case_id": case_id,
                "dataset_family": case.get("dataset_family"),
                "config": config,
            }
        )
    return plan


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _usability_eligible(row: dict[str, Any]) -> bool:
    """Read the separated usability flag, with legacy artifact compatibility."""
    if "usability_eligible" in row:
        return _truthy(row.get("usability_eligible"))
    return _truthy(row.get("accuracy_eligible"))


def select_target_regime(
    summaries: Iterable[dict[str, Any]],
    *,
    required_methods: Iterable[str] = END_TO_END_METHODS,
    cg_iteration_min: int = 3000,
    cg_iteration_max: int = 6000,
    dataset_priority: Iterable[str] = (),
    allowed_resource_limit_methods: Iterable[str] = ("rpcholesky-krr",),
) -> dict[str, Any]:
    """Select, never tune, the Stage-2 target from completed Stage-1 rows."""
    required = tuple(str(method) for method in required_methods)
    allowed_resource_limits = {str(method) for method in allowed_resource_limit_methods}
    if not allowed_resource_limits.issubset(required):
        raise ValueError(
            "allowed_resource_limit_methods must be a subset of required_methods"
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = (
        *STAGE2_SYSTEM_CONFIG_FIELDS,
        "accuracy_max_rmse",
        "accuracy_min_r2",
    )
    for row in summaries:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for group_key, rows in grouped.items():
        by_method = {str(row.get("method")): row for row in rows}
        missing = [method for method in required if method not in by_method]
        resource_limited = [
            method
            for method in required
            if method in by_method
            and str(by_method[method].get("status")) == "resource_limit"
        ]
        failed = [
            method
            for method in required
            if method in by_method
            and str(by_method[method].get("status")) != "ok"
            and not (
                method in allowed_resource_limits
                and str(by_method[method].get("status")) == "resource_limit"
            )
        ]
        usability_failures = [
            method
            for method in ("efgp-standard-full-eig", "ours-binned-default")
            if method not in by_method
            or not _usability_eligible(by_method[method])
        ]
        cg_row = by_method.get("efgp-standard-cg", {})
        cg_iterations = _finite(cg_row.get("iterations_median"))
        in_iteration_window = bool(
            cg_iterations is not None
            and int(cg_iteration_min) <= cg_iterations <= int(cg_iteration_max)
        )
        candidate = {
            **dict(zip(keys, group_key)),
            "cg_iterations": cg_iterations,
            "cg_iteration_min": int(cg_iteration_min),
            "cg_iteration_max": int(cg_iteration_max),
            "missing_methods": missing,
            "failed_methods": failed,
            "resource_limited_methods": resource_limited,
            "allowed_resource_limit_methods": sorted(allowed_resource_limits),
            "usability_failures": usability_failures,
            # Compatibility alias for consumers of frozen target artifacts.
            "accuracy_failures": usability_failures,
            "accuracy_failures_legacy_alias_for": "usability_failures",
            "in_iteration_window": in_iteration_window,
            "declared_dataset_family": (
                rows[0].get("declared_dataset_family") or rows[0].get("dataset_family")
            ),
        }
        for config_key, summary_key in FROZEN_METHOD_CONFIG_FROM_SUMMARY.items():
            value = rows[0].get(summary_key)
            candidate[config_key] = (
                _truthy(value)
                if config_key == "allow_frozen_topk_capacity_adaptation"
                else value
            )
        if not missing and not failed and not usability_failures and in_iteration_window:
            eligible.append(candidate)
        else:
            rejected.append(candidate)
    if not eligible:
        raise TargetSelectionError(
            "No target regime satisfies the frozen selection rule. Do not cherry-pick a "
            "replacement; inspect target_regime_rejections.json and revise the campaign prospectively.",
            rejected,
        )
    priority = {
        str(dataset_stem): index for index, dataset_stem in enumerate(dataset_priority)
    }

    def target_order(row: dict[str, Any]) -> tuple[int, int, str]:
        # ``eligible[-1]`` wins.  The configured first-priority dataset wins
        # an equal-N tie; unlisted datasets use a deterministic lexical tie.
        dataset_stem = str(row["dataset_stem"])
        priority_index = priority.get(dataset_stem, len(priority))
        return (int(row["n_train"]), -int(priority_index), dataset_stem)

    eligible.sort(key=target_order)
    selected = dict(eligible[-1])
    selected.update(
        {
            "selection_rule": (
                "largest N with all six declared pipeline rows present; successful "
                "EFGP/Nystrom rows (a declared RPCholesky hardware resource-limit row "
                "is retained as a valid scalability outcome); full-eig/ours inside "
                "the declared broad absolute usable-quality range (the "
                "reference-equivalence label is descriptive only); and EFGP-CG "
                "iterations in the "
                f"predeclared [{int(cg_iteration_min)},{int(cg_iteration_max)}] interval"
            ),
            "eligible_candidate_count": len(eligible),
            "rejected_candidate_count": len(rejected),
            "dataset_priority": list(dataset_priority),
            "rejections": rejected,
        }
    )
    return selected


def materialize_robustness_plan(
    suite: dict[str, Any],
    target: dict[str, Any],
    *,
    dataset_dir: str,
    output_root: str | Path,
) -> list[dict[str, Any]]:
    template = suite["profiles"]["robustness_at_selected_target"]
    base = dict(suite["base"])
    base.update(template.get("overrides", {}))
    for key in (
        *STAGE2_SYSTEM_CONFIG_FIELDS,
        "accuracy_max_rmse",
        "accuracy_min_r2",
    ):
        if target.get(key) is not None:
            base[key] = target[key]
    for config_key in FROZEN_METHOD_CONFIG_FROM_SUMMARY:
        if target.get(config_key) is not None:
            base[config_key] = target[config_key]
    base["dataset_stem"] = target["dataset_stem"]
    # Historical |B| is a scale-case provenance check.  A fixed top-k can form
    # a different enclosing box when lambda, lengthscale, or dataset changes.
    base["expected_active_box_size"] = None
    # This explicit authorization turns the frozen top-k into an upper bound
    # under the still-frozen box budget. It is deterministic capacity handling,
    # not a timing/accuracy-driven parameter search.
    base["allow_frozen_topk_capacity_adaptation"] = True

    target_family = target.get("declared_dataset_family") or target.get(
        "dataset_family"
    )
    variations: list[tuple[str, dict[str, Any], str | None]] = []
    for value in template["lambda_values"]:
        variations.append(
            (
                f"lambda_{str(value).replace('.', 'p')}",
                {"reg_lambda": value},
                str(target_family) if target_family else None,
            )
        )
    for value in template["lengthscale_values"]:
        variations.append(
            (
                f"lengthscale_{str(value).replace('.', 'p')}",
                {"lengthscale": value},
                str(target_family) if target_family else None,
            )
        )
    for value in template["box_budget_values"]:
        variations.append(
            (
                f"box_budget_{int(value)}",
                {
                    "box_budget": int(value),
                    "active_topk": None,
                    "expected_active_box_size": None,
                    "parameter_selection_policy": BUDGET_ADAPTIVE_PARAMETER_POLICY,
                    "parameter_source": BUDGET_ADAPTIVE_PARAMETER_SOURCE,
                },
                str(target_family) if target_family else None,
            )
        )
    for dataset in template["datasets"]:
        target_n = int(target["n_train"])
        stems_by_n = dataset.get("dataset_stems_by_n_train")
        if stems_by_n is not None:
            if str(target_n) not in stems_by_n:
                raise ValueError(
                    "robustness dataset has no exact artifact for selected target "
                    f"N={target_n}: {dataset.get('dataset_family')}"
                )
            dataset_stem = stems_by_n[str(target_n)]
        else:
            dataset_stem = dataset.get("dataset_stem")
        if not dataset_stem:
            raise ValueError(
                "robustness dataset requires dataset_stem or "
                "dataset_stems_by_n_train"
            )
        dataset_override = {
            "dataset_stem": dataset_stem,
            "n_train": target_n,
        }
        for key in ("accuracy_max_rmse", "accuracy_min_r2"):
            if key in dataset:
                dataset_override[key] = dataset[key]
        variations.append(
            (
                f"dataset_{dataset['dataset_family'].lower()}",
                dataset_override,
                str(dataset["dataset_family"]),
            )
        )

    # Deduplicate the reference case while retaining which axes include it.
    unique: dict[str, dict[str, Any]] = {}
    for label, override, dataset_family in variations:
        merged = dict(base)
        merged.update(override)
        identity = json.dumps(
            {
                key: merged.get(key)
                for key in (
                    *STAGE2_SYSTEM_CONFIG_FIELDS,
                    "box_budget",
                    "rank",
                    "full_eig_rank",
                    "active_topk",
                    "expected_active_box_size",
                    "parameter_selection_policy",
                    "parameter_source",
                    "accuracy_max_rmse",
                    "accuracy_min_r2",
                )
            },
            sort_keys=True,
        )
        if identity in unique:
            unique[identity]["robustness_axes"].append(label)
            continue
        case_id = label
        merged["dataset_dir"] = str(dataset_dir)
        merged["output_dir"] = str(
            Path(output_root) / "robustness_at_selected_target" / case_id
        )
        unique[identity] = {
            "profile": "robustness_at_selected_target",
            "case_id": case_id,
            "dataset_family": dataset_family,
            "robustness_axes": [label],
            "config": _normalize_config(merged),
        }
    return list(unique.values())


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (list, tuple, dict))
                            else value
                        )
                        for key, value in row.items()
                    }
                )


def _load_completed_case(cfg: EndToEndConfig) -> dict[str, Any] | None:
    output = Path(cfg.output_dir)
    config_path = output / "experiment_config.json"
    completion_path = output / "run_complete.json"
    summary_path = output / "pipeline_summary.json"
    runs_path = output / "pipeline_runs.csv"
    if not all(
        path.is_file()
        for path in (config_path, completion_path, summary_path, runs_path)
    ):
        return None
    try:
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_config = asdict(cfg)
    expected_config["methods"] = list(cfg.methods)
    if (
        saved_config != expected_config
        or completion.get("protocol_family") != "end_to_end_krr"
        or completion.get("timing_scope") != TIMING_SCOPE
        or list(completion.get("methods", [])) != list(cfg.methods)
        or completion.get("artifact_complete") is not True
        or completion.get("all_rows_present") is not True
        or completion.get("error_methods")
    ):
        return None
    if (
        not isinstance(summary, list)
        or len(summary) != len(cfg.methods)
        or {str(row.get("method")) for row in summary} != set(cfg.methods)
    ):
        return None
    required_summary_fields = {
        *STAGE2_SYSTEM_CONFIG_FIELDS,
        "accuracy_relative_tolerance",
        "accuracy_max_rmse",
        "accuracy_min_r2",
        "expected_measured_repeats",
        "accuracy_evaluated_repeats",
        "accuracy_passed_repeats",
        "usability_evaluated_repeats",
        "usability_passed_repeats",
        "usability_eligible",
        "execution_eligible",
        "quality_qualified_performance_eligible",
        "reference_evaluated_repeats",
        "reference_equivalent_repeats",
        "reference_equivalent",
        "setup_seconds_at_median_total",
        "solving_phase_seconds_at_median_total",
    }
    if any(
        required_summary_fields.difference(row)
        or row.get("timing_scope") != TIMING_SCOPE
        for row in summary
    ):
        return None
    return {
        "output_dir": str(output.resolve()),
        "summary": summary,
        "completion": completion,
        "resumed_existing": True,
    }


def run_plan(
    plan: Iterable[dict[str, Any]],
    *,
    index_root: str | Path,
    resume: bool = True,
) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    root = Path(index_root)
    for item in plan:
        cfg = item["config"]
        result = _load_completed_case(cfg) if resume else None
        try:
            if result is None:
                result = run_end_to_end_experiment(cfg)
                result["resumed_existing"] = False
            index_row = {
                "profile": item["profile"],
                "case_id": item["case_id"],
                "dataset_family": item.get("dataset_family"),
                "robustness_axes": item.get("robustness_axes"),
                "output_dir": result["output_dir"],
                "n_train": result["completion"]["n_train"],
                "n_test": result["completion"]["n_test"],
                "all_rows_present": result["completion"]["all_rows_present"],
                "status": str(
                    result["completion"].get(
                        "formal_result_status", "scientific_status_missing"
                    )
                ),
                "invocation_mode": (
                    "resumed_existing" if result.get("resumed_existing") else "executed"
                ),
                "resource_limit_methods": result["completion"].get(
                    "resource_limit_methods", []
                ),
                "performance_ineligible_methods": result["completion"].get(
                    "performance_ineligible_methods", []
                ),
                "error_type": None,
                "error_message": None,
            }
        except Exception as exc:
            index_row = {
                "profile": item["profile"],
                "case_id": item["case_id"],
                "dataset_family": item.get("dataset_family"),
                "robustness_axes": item.get("robustness_axes"),
                "output_dir": str(Path(cfg.output_dir).resolve()),
                "n_train": cfg.n_train,
                "n_test": None,
                "all_rows_present": False,
                "status": "error",
                "invocation_mode": "executed",
                "resource_limit_methods": [],
                "performance_ineligible_methods": [],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        index.append(index_row)
        _write_index(root / "end_to_end_suite_index", index)
    return index


def require_complete_plan(
    plan: Iterable[dict[str, Any]],
    index: Sequence[dict[str, Any]],
    *,
    phase: str,
) -> None:
    """Fail after checkpointing if any declared case lacks complete evidence.

    ``run_plan`` deliberately visits every case so that one failure does not
    erase diagnostics for the remaining cases.  A caller must nevertheless
    stop before target selection or downstream experiments when even one
    declared case failed or disappeared.
    """
    planned = [str(item["case_id"]) for item in plan]
    observed = [str(row.get("case_id", "")) for row in index]
    failures = [
        {
            "case_id": str(row.get("case_id", "")),
            "status": str(row.get("status", "")),
            "error_type": row.get("error_type"),
            "error_message": row.get("error_message"),
        }
        for row in index
        if (
            row.get("all_rows_present") is not True
            or str(row.get("status", "")).lower()
            in {"error", "execution_error", "scientific_status_missing"}
            or bool(row.get("error_type"))
        )
    ]
    if (
        len(planned) != len(index)
        or len(set(planned)) != len(planned)
        or len(set(observed)) != len(observed)
        or set(observed) != set(planned)
        or failures
    ):
        raise RuntimeError(
            f"{phase} is incomplete; refusing target selection/downstream work. "
            f"planned={planned}, observed={observed}, failures={failures}"
        )


def collect_summary_rows(plan: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in plan:
        completed = _load_completed_case(item["config"])
        if completed is None:
            continue
        payload = completed["summary"]
        if not isinstance(payload, list):
            raise TypeError(
                "validated pipeline summary is not a list: "
                f"{item['config'].output_dir}"
            )
        for row in payload:
            rows.append(
                {
                    **row,
                    "suite_profile": item["profile"],
                    "case_id": item["case_id"],
                    "declared_dataset_family": item.get("dataset_family"),
                    "robustness_axes": item.get("robustness_axes", []),
                    "run_dir": str(Path(item["config"].output_dir).resolve()),
                }
            )
    return rows


def run_stage1_campaign(
    suite: dict[str, Any],
    *,
    dataset_dir: str,
    output_root: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Run scale first, freeze the target artifact, then run robustness."""
    root = Path(output_root)
    scale_plan = build_profile_plan(
        suite,
        "scale_10m_300m",
        dataset_dir=dataset_dir,
        output_root=root,
    )
    scale_index = run_plan(scale_plan, index_root=root, resume=resume)
    require_complete_plan(scale_plan, scale_index, phase="Stage 1 scale campaign")
    scale_rows = collect_summary_rows(scale_plan)
    _write_index(root / "stage1_scale_summary", scale_rows)
    scale_summary_path = root / "stage1_scale_summary.csv"
    # The target is selected only from raw-repeat-verified rows.  Importing
    # locally avoids a module cycle: the reporter imports the selection rule
    # from this module when it performs the final cross-artifact audit.
    from .two_stage_reporting import (
        _validate_stage1_scale_design,
        load_stage1_summaries,
    )

    verified_scale_rows = load_stage1_summaries((scale_summary_path,))
    selection_cfg = dict(suite.get("target_selection", {}))
    try:
        target = select_target_regime(
            verified_scale_rows,
            cg_iteration_min=int(selection_cfg.get("cg_iteration_min", 3000)),
            cg_iteration_max=int(selection_cfg.get("cg_iteration_max", 6000)),
            dataset_priority=selection_cfg.get("dataset_priority", ()),
            allowed_resource_limit_methods=selection_cfg.get(
                "allowed_resource_limit_methods", ("rpcholesky-krr",)
            ),
        )
    except TargetSelectionError as exc:
        (root / "target_regime_rejections.json").write_text(
            json.dumps(exc.rejections, indent=2), encoding="utf-8"
        )
        raise
    _validate_stage1_scale_design(verified_scale_rows, target, suite)
    target_path = root / "selected_target_regime.json"
    target_path.write_text(json.dumps(target, indent=2), encoding="utf-8")
    (root / "target_regime_rejections.json").write_text(
        json.dumps(target.get("rejections", []), indent=2), encoding="utf-8"
    )
    robustness_plan = materialize_robustness_plan(
        suite,
        target,
        dataset_dir=dataset_dir,
        output_root=root,
    )
    robustness_index = run_plan(
        robustness_plan,
        index_root=root / "robustness_at_selected_target",
        resume=resume,
    )
    require_complete_plan(
        robustness_plan,
        robustness_index,
        phase="Stage 1 robustness campaign",
    )
    robustness_rows = collect_summary_rows(robustness_plan)
    _write_index(
        root / "robustness_at_selected_target" / "stage1_robustness_summary",
        robustness_rows,
    )
    return {
        "target": target,
        "target_path": str(target_path.resolve()),
        "scale_index": scale_index,
        "robustness_index": robustness_index,
        "scale_summary_rows": len(scale_rows),
        "robustness_summary_rows": len(robustness_rows),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end KRR suite profiles.")
    parser.add_argument("--suite-config", default=str(DEFAULT_SUITE_PATH))
    parser.add_argument("--profile", default="scale_10m_300m")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--run-robustness-after-selection",
        action="store_true",
        help=(
            "Run scale_10m_300m first, write the frozen target artifact, then "
            "materialize and run the declared robustness grid."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    suite = load_suite_config(args.suite_config)
    if args.run_robustness_after_selection:
        run_stage1_campaign(
            suite,
            dataset_dir=args.dataset_dir,
            output_root=args.output_root,
            resume=not args.no_resume,
        )
        return 0
    plan = build_profile_plan(
        suite,
        args.profile,
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
    )
    index = run_plan(plan, index_root=args.output_root, resume=not args.no_resume)
    require_complete_plan(plan, index, phase=f"Stage 1 profile {args.profile}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
