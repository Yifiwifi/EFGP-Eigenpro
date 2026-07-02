from __future__ import annotations

import argparse
import gc
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROCESSED_DIR = HERE / "processed"

DEFAULT_SOURCE_STEM = "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain300000000"
DEFAULT_OUTPUT_PREFIX = "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression"
DEFAULT_N_TRAIN_LIST = [
    1_000_000,
    3_000_000,
    10_000_000,
    30_000_000,
    100_000_000,
]
DEFAULT_GRID_SIZE = 64
DEFAULT_CHUNK_ROWS = 1_000_000
UNCOMPRESSED_NPZ_THRESHOLD_ROWS = 50_000_000


def _parse_int_list_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip().replace("_", "")
        if not part:
            continue
        val = int(part)
        if val <= 0:
            raise ValueError(f"all N values must be positive, got {val}")
        out.append(val)
    if not out:
        raise ValueError("empty N list")
    return out


def _iter_row_slices(n_rows: int, *, chunk_rows: int):
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, int(n_rows), chunk_rows):
        yield slice(start, min(int(n_rows), start + chunk_rows))


def _grid_cell_ids(x: np.ndarray, *, grid_size: int) -> np.ndarray:
    xy = np.asarray(x[:, :2], dtype=np.float32)
    ij = np.floor(np.clip(xy, 0.0, np.nextafter(np.float32(1.0), np.float32(0.0))) * int(grid_size)).astype(
        np.int32
    )
    return (ij[:, 0] * int(grid_size) + ij[:, 1]).astype(np.int32, copy=False)


def _count_grid_cells(x: np.ndarray, *, grid_size: int, chunk_rows: int) -> np.ndarray:
    n_cells = int(grid_size) * int(grid_size)
    counts = np.zeros(n_cells, dtype=np.int64)
    for sl in _iter_row_slices(int(x.shape[0]), chunk_rows=chunk_rows):
        cell = _grid_cell_ids(x[sl], grid_size=grid_size)
        counts += np.bincount(cell, minlength=n_cells).astype(np.int64, copy=False)
    return counts


def _allocate_proportional_counts(counts: np.ndarray, target: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    target = int(min(max(0, int(target)), total))
    if target <= 0 or total <= 0:
        return np.zeros_like(counts)
    if target >= total:
        return counts.copy()

    expected = counts.astype(np.float64) * (float(target) / float(total))
    alloc = np.floor(expected).astype(np.int64)
    alloc = np.minimum(alloc, counts)
    rem = int(target - int(alloc.sum()))
    if rem > 0:
        frac = expected - np.floor(expected)
        eligible = np.flatnonzero(alloc < counts)
        order = eligible[np.argsort(-frac[eligible], kind="stable")]
        alloc[order[:rem]] += 1
    return alloc


def _copy_all_chunked(src: np.ndarray, dst: np.ndarray, *, chunk_rows: int) -> None:
    for sl in _iter_row_slices(int(src.shape[0]), chunk_rows=chunk_rows):
        dst[sl] = src[sl]
    if hasattr(dst, "flush"):
        dst.flush()


def _build_spatial_nested_index(
    x_train_src: np.ndarray,
    *,
    n_train_list: list[int],
    grid_size: int,
    seed: int,
    chunk_rows: int,
    oversample: float,
    min_extra_per_cell: int,
) -> dict[str, Any]:
    n_train_pool = int(x_train_src.shape[0])
    max_n_train = int(max(n_train_list))
    if max_n_train > n_train_pool:
        raise ValueError(f"requested max N={max_n_train:,}, but source train pool has only {n_train_pool:,} rows")

    train_cell_counts = _count_grid_cells(x_train_src, grid_size=grid_size, chunk_rows=chunk_rows)
    target_counts_by_n = {
        int(n_train): _allocate_proportional_counts(train_cell_counts, int(n_train)) for n_train in n_train_list
    }
    max_counts = target_counts_by_n[max_n_train]
    occupied_cells = int(np.count_nonzero(train_cell_counts))

    threshold = np.zeros_like(train_cell_counts, dtype=np.float64)
    nonzero = train_cell_counts > 0
    requested = np.maximum(
        np.ceil(max_counts.astype(np.float64) * float(oversample)).astype(np.int64),
        max_counts + int(min_extra_per_cell),
    )
    threshold[nonzero] = np.minimum(1.0, requested[nonzero] / train_cell_counts[nonzero].astype(np.float64))

    rng = np.random.default_rng(int(seed))
    cand_idx_parts: list[np.ndarray] = []
    cand_cell_parts: list[np.ndarray] = []
    cand_key_parts: list[np.ndarray] = []
    n_candidates = 0
    for sl in _iter_row_slices(n_train_pool, chunk_rows=chunk_rows):
        local_len = int(sl.stop - sl.start)
        cell = _grid_cell_ids(x_train_src[sl], grid_size=grid_size)
        key = rng.integers(0, np.iinfo(np.uint64).max, size=local_len, dtype=np.uint64)
        keep = key < (threshold[cell] * float(np.iinfo(np.uint64).max)).astype(np.uint64)
        if not np.any(keep):
            continue
        local_idx = np.nonzero(keep)[0].astype(np.uint32, copy=False)
        cand_idx_parts.append((local_idx + np.uint32(sl.start)).astype(np.uint32, copy=False))
        cand_cell_parts.append(cell[keep].astype(np.uint16, copy=False))
        cand_key_parts.append(key[keep])
        n_candidates += int(local_idx.size)

    if n_candidates <= 0:
        raise RuntimeError("candidate sampling produced no rows")

    cand_idx = np.concatenate(cand_idx_parts)
    cand_cell = np.concatenate(cand_cell_parts)
    cand_key = np.concatenate(cand_key_parts)
    del cand_idx_parts, cand_cell_parts, cand_key_parts
    gc.collect()

    order = np.argsort(cand_cell, kind="stable")
    cand_idx = cand_idx[order]
    cand_cell = cand_cell[order]
    cand_key = cand_key[order]
    del order
    gc.collect()

    n_cells = int(grid_size) * int(grid_size)
    selected_by_cell: list[np.ndarray] = [np.asarray([], dtype=np.uint32) for _ in range(n_cells)]
    candidate_counts = np.bincount(cand_cell.astype(np.int32), minlength=n_cells).astype(np.int64, copy=False)
    cell_starts = np.r_[0, np.cumsum(candidate_counts)]
    short_cells: list[dict[str, int]] = []
    for cell_id in range(n_cells):
        k = int(max_counts[cell_id])
        if k <= 0:
            continue
        start = int(cell_starts[cell_id])
        stop = int(cell_starts[cell_id + 1])
        m = stop - start
        if m < k:
            short_cells.append({"cell": cell_id, "needed": k, "candidates": m, "pool_count": int(train_cell_counts[cell_id])})
            continue
        group_key = cand_key[start:stop]
        group_idx = cand_idx[start:stop]
        if k == m:
            chosen = np.argsort(group_key, kind="stable")
        else:
            chosen = np.argpartition(group_key, k - 1)[:k]
            chosen = chosen[np.argsort(group_key[chosen], kind="stable")]
        selected_by_cell[cell_id] = group_idx[chosen].astype(np.uint32, copy=True)

    if short_cells:
        raise RuntimeError(
            "not enough spatial candidates in some cells; rerun with larger --candidate-oversample. "
            + json.dumps(short_cells[:10], indent=2)
        )

    del cand_idx, cand_cell, cand_key
    gc.collect()

    return {
        "grid_size": int(grid_size),
        "seed": int(seed),
        "n_train_pool": n_train_pool,
        "max_n_train": max_n_train,
        "occupied_train_cells": occupied_cells,
        "train_cell_counts": train_cell_counts,
        "target_counts_by_n": target_counts_by_n,
        "selected_by_cell": selected_by_cell,
        "candidate_oversample": float(oversample),
        "candidate_min_extra_per_cell": int(min_extra_per_cell),
        "n_candidates": int(n_candidates),
    }


def _concat_nested_indices_for_n(
    selected_by_cell: list[np.ndarray],
    counts_for_n: np.ndarray,
    *,
    n_train: int,
    shuffle_seed: int,
) -> np.ndarray:
    pieces = [selected_by_cell[cell_id][: int(k)] for cell_id, k in enumerate(counts_for_n.tolist()) if int(k) > 0]
    out = np.concatenate(pieces).astype(np.uint32, copy=False)
    if int(out.size) != int(n_train):
        raise RuntimeError(f"nested index size mismatch for N={n_train:,}: got {out.size:,}")
    rng = np.random.default_rng(int(shuffle_seed))
    rng.shuffle(out)
    return out


def _copy_indexed_chunked(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_dst: np.ndarray,
    y_dst: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_rows: int,
) -> None:
    for sl in _iter_row_slices(int(indices.size), chunk_rows=chunk_rows):
        idx = indices[sl]
        x_dst[sl] = x_src[idx]
        y_dst[sl] = y_src[idx]
    if hasattr(x_dst, "flush"):
        x_dst.flush()
    if hasattr(y_dst, "flush"):
        y_dst.flush()


def _coverage_diagnostic(
    *,
    test_cell_counts: np.ndarray,
    train_counts_for_n: np.ndarray,
) -> dict[str, Any]:
    test_cells = test_cell_counts > 0
    covered = test_cells & (train_counts_for_n > 0)
    n_test_cells = int(np.count_nonzero(test_cells))
    n_covered = int(np.count_nonzero(covered))
    return {
        "n_test_occupied_cells": n_test_cells,
        "n_test_cells_with_train_points": n_covered,
        "test_cell_coverage_by_train": None if n_test_cells == 0 else float(n_covered / n_test_cells),
        "n_train_occupied_cells": int(np.count_nonzero(train_counts_for_n)),
    }


def _write_one_dataset(
    *,
    source_npz: Path,
    source_json: Path | None,
    source_meta: dict[str, Any],
    output_npz: Path,
    output_json: Path,
    dataset_stem: str,
    n_train: int,
    n_test: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    train_counts_for_n: np.ndarray,
    test_counts_for_n: np.ndarray,
    test_cell_counts: np.ndarray,
    spatial_plan_public: dict[str, Any],
    chunk_rows: int,
    overwrite: bool,
    x_train_src: np.ndarray,
    y_train_src: np.ndarray,
    x_test_src: np.ndarray,
    y_test_src: np.ndarray,
    transforms: dict[str, np.ndarray],
) -> dict[str, Any]:
    if output_npz.exists() and not overwrite:
        raise FileExistsError(f"{output_npz} already exists; pass --overwrite to replace it")
    if output_json.exists() and not overwrite:
        raise FileExistsError(f"{output_json} already exists; pass --overwrite to replace it")

    n_clean = int(n_train) + n_test

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="usgs_spatial_nested_tmp_", dir=str(output_npz.parent)) as tmp_dir:
        tmp = Path(tmp_dir)
        x_train = np.lib.format.open_memmap(
            tmp / "x_train.npy",
            mode="w+",
            dtype=x_train_src.dtype,
            shape=(int(n_train), int(x_train_src.shape[1])),
        )
        y_train = np.lib.format.open_memmap(
            tmp / "y_train.npy",
            mode="w+",
            dtype=y_train_src.dtype,
            shape=(int(n_train),),
        )
        x_test = np.lib.format.open_memmap(
            tmp / "x_test.npy",
            mode="w+",
            dtype=x_test_src.dtype,
            shape=(int(n_test), int(x_test_src.shape[1])),
        )
        y_test = np.lib.format.open_memmap(
            tmp / "y_test.npy",
            mode="w+",
            dtype=y_test_src.dtype,
            shape=(int(n_test),),
        )

        _copy_indexed_chunked(x_train_src, y_train_src, x_train, y_train, train_indices, chunk_rows=chunk_rows)
        _copy_indexed_chunked(x_test_src, y_test_src, x_test, y_test, test_indices, chunk_rows=chunk_rows)

        savez_fn = np.savez if n_clean >= UNCOMPRESSED_NPZ_THRESHOLD_ROWS else np.savez_compressed
        savez_fn(
            output_npz,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            x_min=np.asarray(transforms.get("x_min", np.asarray([], dtype=np.float64)), dtype=np.float64),
            x_scale=np.asarray(transforms.get("x_scale", np.asarray([], dtype=np.float64)), dtype=np.float64),
            y_mean=np.asarray(transforms.get("y_mean", np.asarray([], dtype=np.float64)), dtype=np.float64),
            y_std=np.asarray(transforms.get("y_std", np.asarray([], dtype=np.float64)), dtype=np.float64),
        )
        del x_train, y_train, x_test, y_test
        gc.collect()

    coverage = _coverage_diagnostic(test_cell_counts=test_counts_for_n, train_counts_for_n=train_counts_for_n)
    metadata = {
        "dataset_name": dataset_stem,
        "task_type": source_meta.get("task_type", "2d_lidar_elevation_regression"),
        "source_url": source_meta.get("source_url"),
        "processed_file": str(output_npz),
        "input_definition": source_meta.get("input_definition"),
        "target_definition": source_meta.get("target_definition"),
        "paper_task_statement": source_meta.get("paper_task_statement"),
        "derived_from": {
            "source_npz": str(source_npz),
            "source_json": None if source_json is None else str(source_json),
            "source_dataset_name": source_meta.get("dataset_name"),
            "source_shapes": source_meta.get("shapes", {}),
            "source_split": source_meta.get("split", {}),
            "source_cleaning": source_meta.get("cleaning", {}),
            "source_tile_selection_metadata": source_meta.get("tile_selection_metadata", {}),
        },
        "test_sampling": {
            "source": "x_test/y_test sampled from source_npz test pool",
            "n_test": n_test,
            "method": "spatial_grid_stratified_nested_prefix",
            "nested_within_sweep": True,
            "cell_count_allocation": "largest_remainder_proportional_to_source_test_cell_counts",
        },
        "train_sampling": {
            "method": "spatial_grid_stratified_nested_prefix",
            "grid_size": int(spatial_plan_public["grid_size"]),
            "seed": int(spatial_plan_public["seed"]),
            "n_train_pool": int(spatial_plan_public["n_train_pool"]),
            "n_selected": int(n_train),
            "nested_within_sweep": True,
            "cell_count_allocation": "largest_remainder_proportional_to_source_train_cell_counts",
            "output_order": "deterministically_shuffled_after_nested_selection",
        },
        "split": {
            "method": "source_npz_train_test_pools_spatial_grid_nested_subsets",
            "seed": int(spatial_plan_public["seed"]),
            "test_set_shared_across_sweep": False,
            "test_set_nested_within_sweep": True,
            "train_test_ratio_preserved": True,
            "split_indices_saved": False,
        },
        "diagnostics": {
            "grid_coverage": coverage,
        },
        "x_transform": source_meta.get("x_transform", {}),
        "y_transform": source_meta.get("y_transform", {}),
        "raw_bbox": source_meta.get("raw_bbox", {}),
        "tile_selection_metadata": source_meta.get("tile_selection_metadata", {}),
        "serialization": {
            "npz_mode": "stored" if n_clean >= UNCOMPRESSED_NPZ_THRESHOLD_ROWS else "deflated",
            "array_dtype": str(np.dtype(x_train_src.dtype)),
        },
        "shapes": {
            "n_train": int(n_train),
            "n_test": n_test,
            "n_clean": n_clean,
            "dim": int(source_meta.get("shapes", {}).get("dim", 2)),
            "n_train_pool": int(spatial_plan_public["n_train_pool"]),
        },
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "dataset_stem": dataset_stem,
        "output_npz": str(output_npz),
        "output_json": str(output_json),
        "n_train": int(n_train),
        "n_test": n_test,
        "n_clean": n_clean,
        "train_sampling_method": metadata["train_sampling"]["method"],
        "grid_coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a USGS N_train sweep from the 3e8 processed NPZ. Train and test "
            "subsets preserve the source train/test ratio and are sampled with a "
            "spatial-grid stratified nested-prefix design."
        )
    )
    parser.add_argument("--source-npz", type=Path, default=PROCESSED_DIR / f"{DEFAULT_SOURCE_STEM}.npz")
    parser.add_argument("--source-json", type=Path, default=PROCESSED_DIR / f"{DEFAULT_SOURCE_STEM}_618.json")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--dataset-stem-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--n-train-list", type=str, default=",".join(str(v) for v in DEFAULT_N_TRAIN_LIST))
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--candidate-oversample", type=float, default=1.15)
    parser.add_argument("--candidate-min-extra-per-cell", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_npz = Path(args.source_npz)
    source_json = Path(args.source_json) if str(args.source_json).strip() else None
    output_dir = Path(args.output_dir)
    n_train_list = sorted(_parse_int_list_csv(args.n_train_list))

    if not source_npz.exists():
        raise FileNotFoundError(source_npz)
    source_meta: dict[str, Any] = {}
    if source_json is not None:
        if not source_json.exists():
            raise FileNotFoundError(source_json)
        source_meta = json.loads(source_json.read_text(encoding="utf-8"))

    with np.load(source_npz) as src:
        n_train_pool = int(src["x_train"].shape[0])
        n_test = int(src["x_test"].shape[0])
        dim = int(src["x_train"].shape[1])
    source_test_size = float(n_test / float(n_train_pool + n_test))
    test_size = source_test_size if args.test_size is None else float(args.test_size)
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"--test-size must be between 0 and 1, got {test_size}")
    test_to_train_ratio = test_size / (1.0 - test_size)
    n_test_list = [int(round(int(n) * test_to_train_ratio)) for n in n_train_list]

    too_large = [n for n in n_train_list if int(n) > n_train_pool]
    if too_large:
        raise ValueError(f"requested N values exceed source train pool {n_train_pool:,}: {too_large}")
    too_large_test = [n for n in n_test_list if int(n) > n_test]
    if too_large_test:
        raise ValueError(f"requested test sizes exceed source test pool {n_test:,}: {too_large_test}")

    plan = {
        "source_npz": str(source_npz),
        "source_json": None if source_json is None else str(source_json),
        "output_dir": str(output_dir),
        "dataset_stem_prefix": str(args.dataset_stem_prefix),
        "n_train_pool": n_train_pool,
        "source_n_test_pool": n_test,
        "source_test_size": source_test_size,
        "test_size": test_size,
        "n_test_list": [int(v) for v in n_test_list],
        "dim": dim,
        "n_train_list": [int(v) for v in n_train_list],
        "grid_size": int(args.grid_size),
        "seed": int(args.seed),
        "chunk_rows": int(args.chunk_rows),
        "candidate_oversample": float(args.candidate_oversample),
        "candidate_min_extra_per_cell": int(args.candidate_min_extra_per_cell),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    with np.load(source_npz) as src:
        x_train_src = src["x_train"]
        y_train_src = src["y_train"].reshape(-1)
        x_test_src = src["x_test"]
        y_test_src = src["y_test"].reshape(-1)
        transforms = {
            "x_min": np.asarray(src["x_min"], dtype=np.float64) if "x_min" in src.files else np.asarray([], dtype=np.float64),
            "x_scale": np.asarray(src["x_scale"], dtype=np.float64) if "x_scale" in src.files else np.asarray([], dtype=np.float64),
            "y_mean": np.asarray(src["y_mean"], dtype=np.float64) if "y_mean" in src.files else np.asarray([], dtype=np.float64),
            "y_std": np.asarray(src["y_std"], dtype=np.float64) if "y_std" in src.files else np.asarray([], dtype=np.float64),
        }
        print("building spatial-grid nested training index")
        train_spatial_plan = _build_spatial_nested_index(
            x_train_src,
            n_train_list=n_train_list,
            grid_size=int(args.grid_size),
            seed=int(args.seed),
            chunk_rows=int(args.chunk_rows),
            oversample=float(args.candidate_oversample),
            min_extra_per_cell=int(args.candidate_min_extra_per_cell),
        )
        print("building spatial-grid nested test index")
        test_spatial_plan = _build_spatial_nested_index(
            x_test_src,
            n_train_list=n_test_list,
            grid_size=int(args.grid_size),
            seed=int(args.seed) + 17,
            chunk_rows=int(args.chunk_rows),
            oversample=float(args.candidate_oversample),
            min_extra_per_cell=int(args.candidate_min_extra_per_cell),
        )
        spatial_plan_public = {
            k: v
            for k, v in train_spatial_plan.items()
            if k
            not in {
                "selected_by_cell",
                "target_counts_by_n",
                "train_cell_counts",
            }
        }
        spatial_plan_public["train_cell_counts_summary"] = {
            "n_cells": int(train_spatial_plan["train_cell_counts"].size),
            "n_occupied_cells": int(np.count_nonzero(train_spatial_plan["train_cell_counts"])),
            "min_nonzero": int(train_spatial_plan["train_cell_counts"][train_spatial_plan["train_cell_counts"] > 0].min()),
            "max": int(train_spatial_plan["train_cell_counts"].max()),
        }
        spatial_plan_public["test_cell_counts_summary"] = {
            "n_cells": int(test_spatial_plan["train_cell_counts"].size),
            "n_occupied_cells": int(np.count_nonzero(test_spatial_plan["train_cell_counts"])),
            "min_nonzero": int(test_spatial_plan["train_cell_counts"][test_spatial_plan["train_cell_counts"] > 0].min()),
            "max": int(test_spatial_plan["train_cell_counts"].max()),
        }

        rows: list[dict[str, Any]] = []
        for n_train, n_test_for_n in zip(n_train_list, n_test_list):
            dataset_stem = f"{str(args.dataset_stem_prefix).strip()}_ntrain{int(n_train)}"
            print("=" * 100)
            print(f"deriving {dataset_stem}: n_train={int(n_train):,}, n_test={int(n_test_for_n):,}")
            train_indices = _concat_nested_indices_for_n(
                train_spatial_plan["selected_by_cell"],
                train_spatial_plan["target_counts_by_n"][int(n_train)],
                n_train=int(n_train),
                shuffle_seed=int(args.seed) + int(math.log10(max(10, int(n_train))) * 1000),
            )
            test_indices = _concat_nested_indices_for_n(
                test_spatial_plan["selected_by_cell"],
                test_spatial_plan["target_counts_by_n"][int(n_test_for_n)],
                n_train=int(n_test_for_n),
                shuffle_seed=int(args.seed) + 17 + int(math.log10(max(10, int(n_train))) * 1000),
            )
            row = _write_one_dataset(
                source_npz=source_npz,
                source_json=source_json,
                source_meta=source_meta,
                output_npz=output_dir / f"{dataset_stem}.npz",
                output_json=output_dir / f"{dataset_stem}.json",
                dataset_stem=dataset_stem,
                n_train=int(n_train),
                n_test=int(n_test_for_n),
                train_indices=train_indices,
                test_indices=test_indices,
                train_counts_for_n=train_spatial_plan["target_counts_by_n"][int(n_train)],
                test_counts_for_n=test_spatial_plan["target_counts_by_n"][int(n_test_for_n)],
                test_cell_counts=test_spatial_plan["train_cell_counts"],
                spatial_plan_public=spatial_plan_public,
                chunk_rows=int(args.chunk_rows),
                overwrite=bool(args.overwrite),
                x_train_src=x_train_src,
                y_train_src=y_train_src,
                x_test_src=x_test_src,
                y_test_src=y_test_src,
                transforms=transforms,
            )
            rows.append(row)
            print(json.dumps(row, indent=2))
            del train_indices, test_indices
            gc.collect()

    summary = {
        **plan,
        "spatial_plan": spatial_plan_public,
        "generated": rows,
    }
    summary_path = output_dir / f"{str(args.dataset_stem_prefix).strip()}_fixed_test_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 100)
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
