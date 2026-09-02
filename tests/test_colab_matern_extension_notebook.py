from __future__ import annotations

import json

import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    build_colab_all_experiments_notebook as main_builder,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    build_colab_matern_extension_notebook as extension_builder,
)


def _configuration_cell(notebook: dict) -> dict:
    matches = [
        cell
        for cell in notebook["cells"]
        if extension_builder.CONFIGURATION_MARKER in "".join(cell.get("source", []))
    ]
    assert len(matches) == 1
    return matches[0]


def _final_checkpoint_cell(notebook: dict) -> dict:
    matches = [
        cell
        for cell in notebook["cells"]
        if extension_builder.FINAL_CHECKPOINT_MARKER
        in "".join(cell.get("source", []))
    ]
    assert len(matches) == 1
    return matches[0]


def test_extension_changes_only_configuration_and_final_checkpoint_cells() -> None:
    main_notebook = main_builder.build_notebook()
    extension_notebook = extension_builder.build_notebook()
    assert len(extension_notebook["cells"]) == len(main_notebook["cells"])
    assert extension_notebook["metadata"] == main_notebook["metadata"]

    main_configuration = _configuration_cell(main_notebook)
    extension_configuration = _configuration_cell(extension_notebook)
    main_checkpoint = _final_checkpoint_cell(main_notebook)
    extension_checkpoint = _final_checkpoint_cell(extension_notebook)
    assert extension_configuration["id"] == main_configuration["id"]
    assert extension_configuration != main_configuration
    assert extension_checkpoint["id"] == main_checkpoint["id"]
    assert extension_checkpoint != main_checkpoint
    for main_cell, extension_cell in zip(
        main_notebook["cells"], extension_notebook["cells"], strict=True
    ):
        if main_cell["id"] in {
            main_configuration["id"],
            main_checkpoint["id"],
        }:
            continue
        assert extension_cell == main_cell


def test_extension_enables_only_requested_workloads() -> None:
    source = "".join(_configuration_cell(extension_builder.build_notebook())["source"])
    assignments = extension_builder._literal_run_assignments(source)
    assert assignments == extension_builder._EXPECTED_RUN_ASSIGNMENTS
    enabled = {
        name
        for name, value in assignments.items()
        if value is True and name not in {"RUN_TAG", "RUN_TAG_PREFIX"}
    }
    assert enabled == {
        "RUN_PLUMBING_SMOKE",
        "RUN_STAGE1_FAMILY_PARAMETER_SWEEP",
        "RUN_MATERN_UNBINNED_CG",
        "RUN_LITERATURE_BASELINES_100M",
    }
    assert assignments["RUN_LITERATURE_BASELINE_PILOT"] is False
    assert assignments["RUN_LITERATURE_BASELINES_300M"] is False
    assert assignments["RUN_TAG_PREFIX"] == "matern_extension"
    assert assignments["RUN_LEGACY_GROUPS"] == []


def test_extension_flushes_terminal_exports_and_always_disconnects() -> None:
    notebook = extension_builder.build_notebook()
    configuration_source = "".join(_configuration_cell(notebook)["source"])
    checkpoint_source = "".join(_final_checkpoint_cell(notebook)["source"])

    assert "DISCONNECT_RUNTIME_WHEN_VERIFIED = True" in configuration_source
    assert (
        "Set this to False only when retaining the runtime for interactive debugging."
        in configuration_source
    )
    assert 'get_ipython().events.register(' in configuration_source
    assert '"post_run_cell", _focused_disconnect_on_unhandled_cell_error' in (
        configuration_source
    )
    assert 'getattr(result, "error_before_exec", None)' in configuration_source
    assert "_focused_runtime.unassign()" in configuration_source
    assert "if not FINAL_MANIFEST_PATH.is_file():" in checkpoint_source
    assert "if final_manifest_partial.exists():" in checkpoint_source
    assert '"campaign_jobs_csv": Path(' in checkpoint_source
    assert '"campaign_jobs_json": Path(' in checkpoint_source
    assert '"unified_index": Path(' in checkpoint_source
    assert '"stage1_family_parameter_sweep"' in checkpoint_source
    assert '"matern_unbinned_cg"' in checkpoint_source
    assert '"literature_baselines"' in checkpoint_source
    assert "cg_section.get(" in checkpoint_source
    assert '"report_paths", {}' in checkpoint_source
    assert 'field_name.endswith("_path")' in checkpoint_source
    assert "missing_or_empty_exports" in checkpoint_source
    assert "expected_public_columns = [" in checkpoint_source
    assert "display(public_rmse_time)" in checkpoint_source
    assert "except Exception as exc:" in checkpoint_source
    assert "finally:" in checkpoint_source
    assert "validation failure must not" in checkpoint_source
    assert "print(json.dumps(final_manifest" not in checkpoint_source
    assert "ONE-CLICK CAMPAIGN VERIFIED" not in checkpoint_source
    assert "ONE-CLICK CAMPAIGN COMPLETED WITH FAILURES" not in checkpoint_source

    ordered_markers = [
        "final_manifest_partial.replace(FINAL_MANIFEST_PATH)",
        "persisted_final_manifest = json.loads(",
        "required_persisted_exports = {",
        "display(public_rmse_time)",
        'if hasattr(os, "sync"):',
        "persisted_after_sync = json.loads(",
        "    finally:",
        "        runtime.unassign()",
    ]
    marker_positions = [checkpoint_source.index(marker) for marker in ordered_markers]
    assert marker_positions == sorted(marker_positions)


def test_extension_replacement_is_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="expected exactly one occurrence"):
        extension_builder._replace_exactly_once("", "missing", "replacement")
    with pytest.raises(RuntimeError, match="expected exactly one occurrence"):
        extension_builder._replace_exactly_once("old old", "old", "new")


def test_committed_extension_notebook_matches_generator() -> None:
    committed = json.loads(
        extension_builder.OUTPUT_NOTEBOOK.read_text(encoding="utf-8")
    )
    assert committed == extension_builder.build_notebook()
