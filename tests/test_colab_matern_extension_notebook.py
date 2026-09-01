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


def test_extension_changes_only_the_main_configuration_cell() -> None:
    main_notebook = main_builder.build_notebook()
    extension_notebook = extension_builder.build_notebook()
    assert len(extension_notebook["cells"]) == len(main_notebook["cells"])
    assert extension_notebook["metadata"] == main_notebook["metadata"]

    main_configuration = _configuration_cell(main_notebook)
    extension_configuration = _configuration_cell(extension_notebook)
    assert extension_configuration["id"] == main_configuration["id"]
    assert extension_configuration != main_configuration
    for main_cell, extension_cell in zip(
        main_notebook["cells"], extension_notebook["cells"], strict=True
    ):
        if main_cell["id"] == main_configuration["id"]:
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
        "RUN_LITERATURE_BASELINE_PILOT",
        "RUN_LITERATURE_BASELINES_300M",
    }
    assert assignments["RUN_TAG_PREFIX"] == "matern_extension"
    assert assignments["RUN_LEGACY_GROUPS"] == []


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
