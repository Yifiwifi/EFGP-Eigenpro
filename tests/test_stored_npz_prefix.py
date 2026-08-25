from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.benchmark_dataset.stored_npz import (
    StoredNpzError,
    inspect_stored_npy_member,
    load_stored_npz_prefix,
    mmap_stored_npz_array,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark import (
    _load_dataset,
)


def _write_sidecar(path: Path, n_train: int) -> None:
    path.with_suffix(".json").write_text(
        json.dumps({"shapes": {"n_train": n_train, "dim": 2}}),
        encoding="utf-8",
    )


def test_uncompressed_npz_member_can_be_mapped_and_prefixed(tmp_path: Path) -> None:
    path = tmp_path / "master.npz"
    x = np.arange(60, dtype=np.float32).reshape(30, 2)
    y = np.arange(30, dtype=np.float32)
    np.savez(path, x_train=x, y_train=y)

    info = inspect_stored_npy_member(path, "x_train")
    assert info.shape == (30, 2)
    assert info.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(mmap_stored_npz_array(path, "x_train"), x)

    prefix = load_stored_npz_prefix(path, "x_train", 7, dtype=np.float64)
    assert prefix.dtype == np.float64
    assert prefix.flags.c_contiguous
    np.testing.assert_array_equal(prefix, x[:7])


def test_compressed_member_is_rejected_for_mmap(tmp_path: Path) -> None:
    path = tmp_path / "compressed.npz"
    np.savez_compressed(path, x_train=np.arange(12).reshape(6, 2))
    with pytest.raises(StoredNpzError, match="compressed"):
        inspect_stored_npy_member(path, "x_train")


def test_controlled_loader_prefix_uses_exact_nested_rows(tmp_path: Path) -> None:
    stem = "fixture_master_n30"
    path = tmp_path / f"{stem}.npz"
    x = np.arange(60, dtype=np.float32).reshape(30, 2)
    y = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
    np.savez(path, x_train=x, y_train=y)
    _write_sidecar(path, len(x))

    loaded = _load_dataset(
        stem,
        n_train=11,
        subset_seed=999,
        dataset_dir=str(tmp_path),
        subset_mode="prefix",
    )
    assert loaded["source_n_train"] == 30
    assert loaded["subset_mode"] == "prefix"
    np.testing.assert_array_equal(loaded["x"], x[:11])
    np.testing.assert_array_equal(loaded["y"], y[:11])

