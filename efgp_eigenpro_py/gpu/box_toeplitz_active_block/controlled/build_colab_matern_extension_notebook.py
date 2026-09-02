"""Generate the focused Matérn sweep and literature-baseline Colab notebook.

The main one-click notebook remains the single source of orchestration logic.  This
builder changes only its configuration cell and fails closed if that cell drifts.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

try:
    from . import build_colab_all_experiments_notebook as main_builder
except ImportError:  # pragma: no cover - exercised by direct script invocation
    import build_colab_all_experiments_notebook as main_builder


HERE = Path(__file__).resolve().parent
OUTPUT_NOTEBOOK = HERE.parent / "colab_matern_sweep_and_literature_baselines.ipynb"
CONFIGURATION_MARKER = (
    "# ==================== 一键正式实验：通常无需修改 ===================="
)
FINAL_CHECKPOINT_MARKER = "CAMPAIGN_EXECUTION_FINISHED_UTC ="

_BASE_DISCONNECT_BLOCK = """\
if DISCONNECT_RUNTIME_WHEN_VERIFIED:
    if not run_verified or not FINAL_MANIFEST_PATH.is_file():
        raise RuntimeError("Selected workload is not fully verified; refusing to disconnect.")
    if IS_COLAB:
        from google.colab import runtime
        runtime.unassign()
"""

_BASE_TERMINAL_STATUS_BLOCK = """\
print(json.dumps(final_manifest, indent=2))
if run_verified:
    print("ONE-CLICK CAMPAIGN VERIFIED: all mandatory jobs passed.")
else:
    print(
        "ONE-CLICK CAMPAIGN COMPLETED WITH FAILURES/SKIPS. "
        "See campaign_jobs.csv and controlled_artifact_audit.csv; completed results remain usable."
    )
"""

_EXTENSION_TERMINAL_STATUS_BLOCK = """\
# Focused extension: terminal status stays in the internal manifest.  The only
# final user-facing result is the five-column RMSE--Time table below.
"""

_EXTENSION_DISCONNECT_BLOCK = """\
if DISCONNECT_RUNTIME_WHEN_VERIFIED:
    disconnect_validation_errors = []
    try:
        if not FINAL_MANIFEST_PATH.is_file():
            raise RuntimeError("Terminal campaign manifest is missing.")
        if final_manifest_partial.exists():
            raise RuntimeError("Terminal campaign manifest still has a partial file.")

        # Re-open every available focused result export from Drive before release.
        # A failed/timeout run intentionally lacks its public RMSE-Time artifact,
        # but its terminal manifest and campaign diagnostics must still be flushed.
        persisted_final_manifest = json.loads(
            FINAL_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if persisted_final_manifest != final_manifest:
            raise RuntimeError(
                "Persisted final manifest does not match the in-memory manifest."
            )

        required_persisted_exports = {
            "final_manifest": FINAL_MANIFEST_PATH,
            "campaign_jobs_csv": Path(persisted_final_manifest["campaign_jobs_csv"]),
            "campaign_jobs_json": Path(persisted_final_manifest["campaign_jobs_json"]),
            "unified_index": Path(persisted_final_manifest["unified_index"]),
        }
        sweep_section = persisted_final_manifest.get(
            "stage1_family_parameter_sweep", {}
        )
        if sweep_section.get("enabled"):
            for export_name, export_path in sweep_section.get(
                "report_paths", {}
            ).items():
                required_persisted_exports[
                    f"family_parameter_sweep.{export_name}"
                ] = Path(export_path)
        cg_section = persisted_final_manifest.get("matern_unbinned_cg", {})
        if cg_section.get("enabled"):
            for export_name, export_path in cg_section.get(
                "report_paths", {}
            ).items():
                required_persisted_exports[
                    f"matern_unbinned_cg.{export_name}"
                ] = Path(export_path)
        literature_section = persisted_final_manifest.get(
            "literature_baselines", {}
        )
        for phase_name, phase_payload in literature_section.items():
            if not isinstance(phase_payload, dict) or not phase_payload.get("enabled"):
                continue
            for field_name, field_value in phase_payload.items():
                if field_name.endswith("_path") and field_value:
                    required_persisted_exports[
                        f"literature_baselines.{phase_name}.{field_name}"
                    ] = Path(field_value)

        missing_or_empty_exports = [
            f"{label}={path}"
            for label, path in required_persisted_exports.items()
            if not path.is_file() or path.stat().st_size <= 0
        ]
        if missing_or_empty_exports:
            raise RuntimeError(
                "Focused terminal exports are missing or empty: "
                + "; ".join(missing_or_empty_exports)
            )

        final_100m_section = literature_section.get("final_100m", {})
        public_rmse_time_csv = final_100m_section.get(
            "public_rmse_time_table_csv_path"
        )
        if run_verified and final_100m_section.get("enabled"):
            public_rmse_time = pd.read_csv(Path(public_rmse_time_csv))
            expected_public_columns = [
                "dataset_family", "n_train", "method",
                "median_time_seconds", "rmse",
            ]
            if list(public_rmse_time.columns) != expected_public_columns:
                raise RuntimeError(
                    "Final public RMSE-Time table has an unexpected schema."
                )
            display(public_rmse_time)

        # Google Drive is FUSE-backed in Colab.  Request a filesystem flush, then
        # re-read the manifest once more before runtime release.
        if hasattr(os, "sync"):
            os.sync()
        persisted_after_sync = json.loads(
            FINAL_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if persisted_after_sync != final_manifest:
            raise RuntimeError("Final manifest changed during the Drive flush.")
    except Exception as exc:
        disconnect_validation_errors.append(
            {"error_type": type(exc).__name__, "error_message": str(exc)}
        )
        try:
            if hasattr(os, "sync"):
                os.sync()
        except Exception as sync_exc:
            disconnect_validation_errors.append(
                {
                    "error_type": type(sync_exc).__name__,
                    "error_message": str(sync_exc),
                }
            )
    finally:
        # Release even when a baseline timed out or a report was rejected.  The
        # internal manifest preserves that evidence; validation failure must not
        # leave a paid accelerator connected.
        if IS_COLAB:
            from google.colab import runtime
            runtime.unassign()
"""

_REPLACEMENTS = (
    ("from pathlib import Path", "import os\nimport sys\nfrom pathlib import Path"),
    ('RUN_TAG_PREFIX = "paper_one_click"', 'RUN_TAG_PREFIX = "matern_extension"'),
    ("RUN_ALL_FORMAL_EXPERIMENTS = True", "RUN_ALL_FORMAL_EXPERIMENTS = False"),
    (
        "RUN_STAGE1_END_TO_END_KRR = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_STAGE1_END_TO_END_KRR = False",
    ),
    (
        "RUN_STAGE1_ROBUSTNESS = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_STAGE1_ROBUSTNESS = False",
    ),
    (
        "RUN_STAGE1_FAMILY_SCALE = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_STAGE1_FAMILY_SCALE = False",
    ),
    (
        "RUN_STAGE1_FAMILY_PARAMETER_SWEEP = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_STAGE1_FAMILY_PARAMETER_SWEEP = True",
    ),
    (
        "RUN_LITERATURE_BASELINE_PILOT = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_LITERATURE_BASELINE_PILOT = False",
    ),
    (
        "RUN_LITERATURE_BASELINES_300M = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_LITERATURE_BASELINES_300M = False",
    ),
    (
        "RUN_MATERN_UNBINNED_CG = False",
        "RUN_MATERN_UNBINNED_CG = True",
    ),
    (
        "RUN_LITERATURE_BASELINES_100M = False",
        "RUN_LITERATURE_BASELINES_100M = True",
    ),
    (
        "RUN_STAGE1_FAMILY_ROBUSTNESS = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_STAGE1_FAMILY_ROBUSTNESS = False",
    ),
    (
        "RUN_STAGE1_FAMILY_KERNEL = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_STAGE1_FAMILY_KERNEL = False",
    ),
    (
        "RUN_STAGE2_FIXED_AB_SOLVERS = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_STAGE2_FIXED_AB_SOLVERS = False",
    ),
    (
        "RUN_PLUMBING_SMOKE = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_PLUMBING_SMOKE = True",
    ),
    (
        "RUN_PREDICTION_AUDIT = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_PREDICTION_AUDIT = False",
    ),
    (
        "# Release the paid Colab accelerator after every mandatory artifact\n"
        "# has been persisted and the final campaign manifest is verified.\n"
        "DISCONNECT_RUNTIME_WHEN_VERIFIED = True",
        "# Focused extension: flush terminal evidence, then always release Colab.\n"
        "# Set this to False only when retaining the runtime for interactive debugging.\n"
        "DISCONNECT_RUNTIME_WHEN_VERIFIED = True\n\n"
        "# Fail-safe: an unhandled error in any later cell immediately flushes and\n"
        "# releases the accelerator, even if the final checkpoint cell is never reached.\n"
        "if DISCONNECT_RUNTIME_WHEN_VERIFIED and \"google.colab\" in sys.modules:\n"
        "    from google.colab import runtime as _focused_runtime\n\n"
        "    def _focused_disconnect_on_unhandled_cell_error(result):\n"
        "        error = (\n"
        "            getattr(result, \"error_before_exec\", None)\n"
        "            or getattr(result, \"error_in_exec\", None)\n"
        "        )\n"
        "        if error is None:\n"
        "            return\n"
        "        try:\n"
        "            if hasattr(os, \"sync\"):\n"
        "                os.sync()\n"
        "        finally:\n"
        "            _focused_runtime.unassign()\n\n"
        "    get_ipython().events.register(\n"
        "        \"post_run_cell\", _focused_disconnect_on_unhandled_cell_error\n"
        "    )",
    ),
)

_EXPECTED_RUN_ASSIGNMENTS = {
    "RUN_TAG_PREFIX": "matern_extension",
    "RUN_TAG": None,
    "RUN_ALL_FORMAL_EXPERIMENTS": False,
    "RUN_STAGE1_END_TO_END_KRR": False,
    "RUN_STAGE1_ROBUSTNESS": False,
    "RUN_STAGE1_FAMILY_SCALE": False,
    "RUN_STAGE1_FAMILY_PARAMETER_SWEEP": True,
    "RUN_ORIGINAL_KRR_PROXY_FEASIBILITY": False,
    "RUN_ORIGINAL_KRR_FULL_SCALE_RESOURCE_AUDIT": False,
    "RUN_LITERATURE_BASELINE_PILOT": False,
    "RUN_LITERATURE_BASELINES_300M": False,
    "RUN_MATERN_UNBINNED_CG": True,
    "RUN_LITERATURE_BASELINES_100M": True,
    "RUN_STAGE1_FAMILY_ROBUSTNESS": False,
    "RUN_STAGE1_FAMILY_KERNEL": False,
    "RUN_STAGE2_FIXED_AB_SOLVERS": False,
    "RUN_LEGACY_GROUPS": [],
    "RUN_PLUMBING_SMOKE": True,
    "RUN_CG_SCREEN_10M": False,
    "RUN_Q256_CENTER_10M": False,
    "RUN_BOX_BUDGET_ABLATION": False,
    "RUN_ARCHIVED_EXACT_SCALE": False,
    "RUN_DEVELOPMENT_MASTER_SCALE": False,
    "RUN_MANITOWOC_SCALE": False,
    "RUN_WINNEBAGO_OAT_10M": False,
    "RUN_Q128_BRIDGE": False,
    "RUN_SE_FULL_INVERSE_CONTROL": False,
    "RUN_PREDICTION_AUDIT": False,
}


def _replace_exactly_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Configuration drift: expected exactly one occurrence of {old!r}, "
            f"found {count}."
        )
    return source.replace(old, new, 1)


def _literal_run_assignments(source: str) -> dict[str, object]:
    assignments: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("RUN_"):
            continue
        if target.id in assignments:
            raise RuntimeError(
                f"Configuration drift: duplicate assignment {target.id}."
            )
        try:
            assignments[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Configuration drift: {target.id} is not an explicit literal."
            ) from exc
    return assignments


def _validate_extension_configuration(source: str) -> None:
    assignments = _literal_run_assignments(source)
    if assignments != _EXPECTED_RUN_ASSIGNMENTS:
        missing = sorted(_EXPECTED_RUN_ASSIGNMENTS.keys() - assignments.keys())
        unexpected = sorted(assignments.keys() - _EXPECTED_RUN_ASSIGNMENTS.keys())
        mismatched = sorted(
            name
            for name in assignments.keys() & _EXPECTED_RUN_ASSIGNMENTS.keys()
            if assignments[name] != _EXPECTED_RUN_ASSIGNMENTS[name]
        )
        raise RuntimeError(
            "Extension workload flags do not match the fail-closed policy: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}."
        )

    disconnect_assignments = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "DISCONNECT_RUNTIME_WHEN_VERIFIED"
    ]
    if len(disconnect_assignments) != 1:
        raise RuntimeError(
            "Extension disconnect policy drift: expected exactly one explicit flag."
        )
    try:
        disconnect_default = ast.literal_eval(disconnect_assignments[0].value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Extension disconnect policy drift: flag is not an explicit literal."
        ) from exc
    if disconnect_default is not True:
        raise RuntimeError(
            "Extension disconnect policy drift: safe default must be automatic."
        )


def build_notebook() -> dict:
    notebook = main_builder.build_notebook()
    configuration_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and CONFIGURATION_MARKER in "".join(cell.get("source", []))
    ]
    if len(configuration_cells) != 1:
        raise RuntimeError(
            "Configuration drift: expected exactly one main configuration cell, "
            f"found {len(configuration_cells)}."
        )

    configuration_cell = configuration_cells[0]
    source = "".join(configuration_cell["source"])
    for old, new in _REPLACEMENTS:
        source = _replace_exactly_once(source, old, new)
    _validate_extension_configuration(source)
    configuration_cell["source"] = source.splitlines(keepends=True)

    checkpoint_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and FINAL_CHECKPOINT_MARKER in "".join(cell.get("source", []))
    ]
    if len(checkpoint_cells) != 1:
        raise RuntimeError(
            "Final-checkpoint drift: expected exactly one checkpoint cell, "
            f"found {len(checkpoint_cells)}."
        )
    checkpoint_cell = checkpoint_cells[0]
    checkpoint_source = "".join(checkpoint_cell["source"])
    checkpoint_source = _replace_exactly_once(
        checkpoint_source,
        _BASE_TERMINAL_STATUS_BLOCK,
        _EXTENSION_TERMINAL_STATUS_BLOCK,
    )
    checkpoint_source = _replace_exactly_once(
        checkpoint_source,
        _BASE_DISCONNECT_BLOCK,
        _EXTENSION_DISCONNECT_BLOCK,
    )
    checkpoint_cell["source"] = checkpoint_source.splitlines(keepends=True)
    return notebook


def main() -> None:
    notebook = build_notebook()
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_NOTEBOOK} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
