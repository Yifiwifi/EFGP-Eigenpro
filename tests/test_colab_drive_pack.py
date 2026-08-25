from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack import (
    add_exact_artifact,
    compare_nested_prefix,
    inspect_stored_npz,
    open_dataset_prefix,
    open_stored_npz_array,
    prepare_dataset,
    verify_catalog,
)


def _arrays(n_train: int) -> dict[str, np.ndarray]:
    n_test = n_train // 4
    return {
        "x_train": np.arange(n_train * 2, dtype=np.float32).reshape(n_train, 2),
        "y_train": np.arange(n_train, dtype=np.float32) + 100.0,
        "x_test": np.arange(n_test * 2, dtype=np.float32).reshape(n_test, 2) + 200.0,
        "y_test": np.arange(n_test, dtype=np.float32) + 300.0,
        "y_train_true": np.arange(n_train, dtype=np.float32) + 400.0,
        "y_center": np.asarray([12.5], dtype=np.float64),
    }


def _write_stored(path: Path, n_train: int) -> dict[str, np.ndarray]:
    arrays = _arrays(n_train)
    np.savez(path, **arrays)
    return arrays


def test_stored_npz_members_are_memmapped_at_exact_offsets(tmp_path: Path) -> None:
    path = tmp_path / "master.npz"
    expected = _write_stored(path, 8)

    specs = inspect_stored_npz(path)
    assert specs["x_train"].shape == (8, 2)
    assert specs["x_train"].dtype == np.dtype("float32")
    mapped = open_stored_npz_array(path, "x_train")
    assert isinstance(mapped, np.memmap)
    assert np.array_equal(mapped, expected["x_train"])

    prefix = open_dataset_prefix(path, 4)
    assert isinstance(prefix["x_train"], np.memmap)
    assert Path(prefix["x_train"].filename) == path
    assert np.array_equal(prefix["x_train"], expected["x_train"][:4])
    assert np.array_equal(prefix["x_test"], expected["x_test"][:1])
    assert np.array_equal(prefix["y_train_true"], expected["y_train_true"][:4])
    assert np.array_equal(prefix["y_center"], expected["y_center"])

    no_test = open_dataset_prefix(path, 4, include_test=False)
    assert "x_test" not in no_test
    assert "y_test" not in no_test


def test_compressed_npz_is_rejected_instead_of_silently_extracting(tmp_path: Path) -> None:
    path = tmp_path / "compressed.npz"
    np.savez_compressed(path, **_arrays(8))
    with pytest.raises(ValueError, match="is compressed"):
        inspect_stored_npz(path)


def test_prepare_creates_zero_copy_catalog_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "source" / "master.npz"
    source.parent.mkdir()
    _write_stored(source, 8)
    metadata = source.with_suffix(".json")
    metadata.write_text('{"frozen": true}\n', encoding="utf-8")
    stage = tmp_path / "stage"

    entry = prepare_dataset(
        source_npz=source,
        source_metadata=metadata,
        output_dir=stage,
        dataset_id="tiny-v1",
        prefix_sizes=(4, 8, 12),
        link_mode="hardlink",
        bundle_names=("paper_10m", "scale_masters"),
    )

    staged_master = stage / entry["master"]["drive_relative_path"]
    staged_metadata = stage / entry["metadata"]["drive_relative_path"]
    assert os.path.samefile(source, staged_master)
    assert os.path.samefile(metadata, staged_metadata)
    assert [row["status"] for row in entry["logical_prefixes"]] == [
        "ready",
        "ready",
        "planned",
    ]
    assert entry["logical_prefixes"][0]["n_test"] == 1
    assert entry["logical_prefixes"][2]["missing_train_rows"] == 4

    catalog = json.loads((stage / "drive_manifest.json").read_text(encoding="utf-8"))
    assert set(catalog["datasets"]) == {"tiny-v1"}
    assert {artifact["role"] for artifact in catalog["artifacts"]} == {
        "master_npz",
        "metadata_json",
    }
    assert catalog["bundles"]["paper_10m"] == [
        "tiny-v1:master",
        "tiny-v1:metadata",
    ]
    assert catalog["bundles"]["scale_masters"] == catalog["bundles"]["paper_10m"]
    assert verify_catalog(stage / "drive_manifest.json") == [
        entry["master"]["drive_relative_path"],
        entry["metadata"]["drive_relative_path"],
    ]
    checksum_text = (stage / "checksums.sha256").read_text(encoding="utf-8")
    assert entry["master"]["sha256"] in checksum_text
    assert entry["metadata"]["sha256"] in checksum_text

    # Idempotent preparation reuses the same hard links.
    repeated = prepare_dataset(
        source_npz=source,
        source_metadata=metadata,
        output_dir=stage,
        dataset_id="tiny-v1",
        prefix_sizes=(4, 8, 12),
        link_mode="hardlink",
        bundle_names=("paper_10m", "scale_masters"),
    )
    assert repeated["master"]["sha256"] == entry["master"]["sha256"]

    # A later call may add a bundle without silently dropping old memberships.
    prepare_dataset(
        source_npz=source,
        source_metadata=metadata,
        output_dir=stage,
        dataset_id="tiny-v1",
        prefix_sizes=(4, 8, 12),
        link_mode="hardlink",
        bundle_names=("new_bundle",),
    )
    updated = json.loads((stage / "drive_manifest.json").read_text(encoding="utf-8"))
    for bundle in ("paper_10m", "scale_masters", "new_bundle"):
        assert updated["bundles"][bundle] == ["tiny-v1:master", "tiny-v1:metadata"]


def test_manifest_only_catalog_merges_datasets_without_copying(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    for dataset_id in ("first", "second"):
        source = tmp_path / f"{dataset_id}.npz"
        _write_stored(source, 8)
        prepare_dataset(
            source_npz=source,
            output_dir=stage,
            dataset_id=dataset_id,
            prefix_sizes=(4, 8),
            link_mode="manifest-only",
        )

    catalog = json.loads((stage / "drive_manifest.json").read_text(encoding="utf-8"))
    assert set(catalog["datasets"]) == {"first", "second"}
    for dataset_id, entry in catalog["datasets"].items():
        assert not (stage / entry["master"]["drive_relative_path"]).exists()
        assert (stage / "data" / dataset_id / "dataset_manifest.json").is_file()


def test_compressed_legacy_artifact_remains_exact_and_has_no_prefix_claim(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paper_exact.npz"
    np.savez_compressed(source, **_arrays(8))
    stage = tmp_path / "stage"
    artifact = add_exact_artifact(
        source_path=source,
        output_dir=stage,
        name="paper-noise03-8:npz",
        dataset_family="paper-noise03",
        role="exact_npz",
        link_mode="hardlink",
        bundle_names=("paper_10m", "legacy_routes"),
    )
    assert os.path.samefile(source, stage / artifact["relative_path"])
    assert "array_schema" not in artifact
    catalog = json.loads((stage / "drive_manifest.json").read_text(encoding="utf-8"))
    assert catalog["bundles"]["paper_10m"] == ["paper-noise03-8:npz"]
    assert catalog["datasets"] == {}
    assert verify_catalog(stage / "drive_manifest.json") == [
        artifact["relative_path"]
    ]


def test_compare_nested_prefix_is_chunked_and_exact(tmp_path: Path) -> None:
    larger = tmp_path / "larger.npz"
    smaller = tmp_path / "smaller.npz"
    _write_stored(larger, 8)
    _write_stored(smaller, 4)

    report = compare_nested_prefix(
        larger_npz=larger, prefix_npz=smaller, chunk_rows=2
    )
    assert report["exact_prefix"] is True
    assert report["n_train"] == 4
    assert set(report["arrays_compared"]) == set(_arrays(4))

    broken = _arrays(4)
    broken["y_train"][3] = -999.0
    np.savez(smaller, **broken)
    with pytest.raises(ValueError, match="data differs for 'y_train'"):
        compare_nested_prefix(larger_npz=larger, prefix_npz=smaller, chunk_rows=2)


def test_compare_nested_prefix_accepts_a_compressed_frozen_prefix(tmp_path: Path) -> None:
    larger = tmp_path / "larger.npz"
    smaller = tmp_path / "smaller-compressed.npz"
    _write_stored(larger, 8)
    np.savez_compressed(smaller, **_arrays(4))

    report = compare_nested_prefix(
        larger_npz=larger,
        prefix_npz=smaller,
        chunk_rows=2,
    )
    assert report["exact_prefix"] is True
    assert report["n_train"] == 4
