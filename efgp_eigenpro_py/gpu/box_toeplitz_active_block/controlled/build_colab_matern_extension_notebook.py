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

_REPLACEMENTS = (
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
        "RUN_ORIGINAL_KRR_PROXY_FEASIBILITY = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_ORIGINAL_KRR_PROXY_FEASIBILITY = True",
    ),
    (
        "RUN_ORIGINAL_KRR_FULL_SCALE_RESOURCE_AUDIT = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_ORIGINAL_KRR_FULL_SCALE_RESOURCE_AUDIT = True",
    ),
    (
        "RUN_LITERATURE_BASELINE_PILOT = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_LITERATURE_BASELINE_PILOT = True",
    ),
    (
        "RUN_LITERATURE_BASELINES_300M = RUN_ALL_FORMAL_EXPERIMENTS",
        "RUN_LITERATURE_BASELINES_300M = True",
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
)

_EXPECTED_RUN_ASSIGNMENTS = {
    "RUN_TAG_PREFIX": "matern_extension",
    "RUN_TAG": None,
    "RUN_ALL_FORMAL_EXPERIMENTS": False,
    "RUN_STAGE1_END_TO_END_KRR": False,
    "RUN_STAGE1_ROBUSTNESS": False,
    "RUN_STAGE1_FAMILY_SCALE": False,
    "RUN_STAGE1_FAMILY_PARAMETER_SWEEP": True,
    "RUN_ORIGINAL_KRR_PROXY_FEASIBILITY": True,
    "RUN_ORIGINAL_KRR_FULL_SCALE_RESOURCE_AUDIT": True,
    "RUN_LITERATURE_BASELINE_PILOT": True,
    "RUN_LITERATURE_BASELINES_300M": True,
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
