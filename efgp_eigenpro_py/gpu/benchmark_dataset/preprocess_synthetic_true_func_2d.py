from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---- Synthetic dataset controls ----
DIM = 2
DEFAULT_N_TRAIN = 100_000
DEFAULT_N_TEST = 1_000
DEFAULT_NOISE = 0.3
DEFAULT_SEED_TRAIN = 20260421
DEFAULT_SEED_TEST = 1

DEFAULT_STORAGE_DTYPE = np.float32
DEFAULT_CHUNK_ROWS = 1_000_000
UNCOMPRESSED_NPZ_THRESHOLD_ROWS = 50_000_000

# These mirror the benchmark notebook's weak-GPU synthetic setup.
KERNEL_FAMILY = "matern"
KERNEL_LENGTHSCALE = 0.1
KERNEL_NU = 1.5
REG_LAMBDA = 0.1
EPS = 1e-5
L2_SCALED = True


def _ensure_repo_root_on_path() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        pkg_dir = parent / "efgp_eigenpro_py"
        if pkg_dir.exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not locate repo root containing 'efgp_eigenpro_py'.")


REPO_ROOT = _ensure_repo_root_on_path()

from efgp_eigenpro_py.benchmark import make_test_set, true_func_2d  # noqa: E402


def _iter_row_slices(n_rows: int, *, chunk_rows: int = DEFAULT_CHUNK_ROWS):
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, int(n_rows), chunk_rows):
        yield slice(start, min(int(n_rows), start + chunk_rows))


def _close_memmaps(*arrays: np.ndarray | None) -> None:
    """Release disk-backed memmap handles (required on Windows before temp cleanup)."""
    for arr in arrays:
        if arr is None:
            continue
        try:
            if isinstance(arr, np.memmap):
                arr.flush()
                mmap_obj = getattr(arr, "_mmap", None)
                if mmap_obj is not None:
                    mmap_obj.close()
        except Exception:  # noqa: BLE001
            pass
    gc.collect()


def _generate_train_to_memmaps(
    work_dir: Path,
    *,
    n_train: int,
    dim: int,
    noise: float,
    seed: int,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    storage_dtype: np.dtype = DEFAULT_STORAGE_DTYPE,
) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap]:
    """Stream synthetic train arrays to disk-backed memmaps.

    Matches ``make_dataset(dim, n_train, true_func_2d, ...)`` by consuming the
    same RNG stream in fixed-size chunks instead of materializing all rows at once.
    """
    storage_dtype = np.dtype(storage_dtype)
    n_train = int(n_train)
    dim = int(dim)

    x_train = np.lib.format.open_memmap(
        work_dir / "x_train.npy",
        mode="w+",
        dtype=storage_dtype,
        shape=(n_train, dim),
    )
    y_train = np.lib.format.open_memmap(
        work_dir / "y_train.npy",
        mode="w+",
        dtype=storage_dtype,
        shape=(n_train,),
    )
    y_train_true = np.lib.format.open_memmap(
        work_dir / "y_train_true.npy",
        mode="w+",
        dtype=storage_dtype,
        shape=(n_train,),
    )
    train_noise = np.lib.format.open_memmap(
        work_dir / "train_noise.npy",
        mode="w+",
        dtype=storage_dtype,
        shape=(n_train,),
    )

    rng = np.random.default_rng(int(seed))
    noise_f = float(noise)
    for sl in _iter_row_slices(n_train, chunk_rows=int(chunk_rows)):
        n_chunk = int(sl.stop - sl.start)
        x_chunk = rng.uniform(0.0, 1.0, size=(n_chunk, dim)).astype(storage_dtype, copy=False)
        f_chunk = true_func_2d(x_chunk).astype(storage_dtype, copy=False)
        eps_chunk = (noise_f * rng.standard_normal(n_chunk)).astype(storage_dtype, copy=False)
        y_chunk = f_chunk + eps_chunk

        x_train[sl] = x_chunk
        y_train[sl] = y_chunk
        y_train_true[sl] = f_chunk
        train_noise[sl] = eps_chunk

    return x_train, y_train, y_train_true, train_noise


def _default_dataset_stem(n_train: int) -> str:
    return f"synthetic_true_func_2d_n{int(n_train)}"


def _default_output_paths(dataset_stem: str) -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    processed_dir = here / "processed"
    return processed_dir / f"{dataset_stem}.npz", processed_dir / f"{dataset_stem}.json"


def build_synthetic_dataset(
    output_npz: Path,
    output_json: Path,
    *,
    dataset_stem: str,
    n_train: int = DEFAULT_N_TRAIN,
    n_test: int = DEFAULT_N_TEST,
    noise: float = DEFAULT_NOISE,
    seed_train: int = DEFAULT_SEED_TRAIN,
    seed_test: int = DEFAULT_SEED_TEST,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> dict:
    if int(DIM) != 2:
        raise ValueError(f"true_func_2d requires DIM=2, got {DIM}")

    n_train = int(n_train)
    n_test = int(n_test)
    chunk_rows = max(1, int(chunk_rows))
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = output_npz.with_name(f".{output_npz.name}.tmp.npz")
    tmp_json = output_json.with_name(f".{output_json.name}.tmp")

    x_test, y_test = make_test_set(
        DIM,
        n_test,
        true_func_2d,
        seed=int(seed_test),
    )
    x_test = np.asarray(x_test, dtype=DEFAULT_STORAGE_DTYPE)
    y_test = np.asarray(y_test, dtype=DEFAULT_STORAGE_DTYPE)

    tmp_dir = tempfile.mkdtemp(prefix="synthetic_tmp_", dir=str(output_npz.parent))
    x_train = y_train = y_train_true = train_noise = None
    try:
        x_train, y_train, y_train_true, train_noise = _generate_train_to_memmaps(
            Path(tmp_dir),
            n_train=n_train,
            dim=int(DIM),
            noise=float(noise),
            seed=int(seed_train),
            chunk_rows=int(chunk_rows),
            storage_dtype=DEFAULT_STORAGE_DTYPE,
        )
        for arr in (x_train, y_train, y_train_true, train_noise):
            arr.flush()

        savez_fn = np.savez if n_train >= UNCOMPRESSED_NPZ_THRESHOLD_ROWS else np.savez_compressed
        savez_fn(
            tmp_npz,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            y_train_true=y_train_true,
            train_noise=train_noise,
        )
    finally:
        _close_memmaps(x_train, y_train, y_train_true, train_noise)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    metadata = {
        "dataset_name": str(dataset_stem),
        "task_type": "2d_synthetic_regression",
        "source_file": "generated://efgp_eigenpro_py.benchmark",
        "processed_file": str(output_npz),
        "input_definition": "x_train and x_test are sampled in [0,1]^2 using the benchmark helpers",
        "target_definition": "y_train = true_func_2d(x_train) + Gaussian noise, y_test = true_func_2d(x_test)",
        "generation": {
            "train_generator": "streamed_make_dataset_chunks",
            "test_generator": "make_test_set",
            "target_function": "true_func_2d",
            "dim": int(DIM),
            "n_train": n_train,
            "n_test": n_test,
            "noise_std": float(noise),
            "seed_train": int(seed_train),
            "seed_test": int(seed_test),
            "train_distribution": "uniform in [0,1]^2",
            "test_distribution": "uniform in [0,1]^2",
            "chunk_rows": int(chunk_rows),
            "storage_dtype": str(DEFAULT_STORAGE_DTYPE),
        },
        "split": {
            "method": "pre-generated train/test sets",
            "note": "This synthetic benchmark follows the sanity_check notebook directly instead of splitting one merged pool.",
        },
        "x_transform": {
            "method": "none",
            "domain": "[0,1]^2",
            "dim": int(DIM),
        },
        "y_transform": {
            "method": "none",
            "noise_model": "additive Gaussian noise on y_train only",
            "noise_std": float(noise),
        },
        "benchmark_reference": {
            "notebook": "gpu/sanity_check/v5_mat32_weak_gpu_complexity_benchmark.ipynb",
            "kernel_family": KERNEL_FAMILY,
            "lengthscale": float(KERNEL_LENGTHSCALE),
            "nu": float(KERNEL_NU),
            "reg_lambda": float(REG_LAMBDA),
            "eps": float(EPS),
            "l2_scaled": bool(L2_SCALED),
        },
        "serialization": {
            "npz_mode": "stored" if n_train >= UNCOMPRESSED_NPZ_THRESHOLD_ROWS else "deflated",
            "array_dtype": str(DEFAULT_STORAGE_DTYPE),
        },
        "shapes": {
            "n_train": n_train,
            "n_test": n_test,
            "dim": int(DIM),
        },
        "paper_task_statement": "(x1, x2) -> true_func_2d(x) with noisy train targets",
    }
    tmp_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(tmp_npz, output_npz)
    os.replace(tmp_json, output_json)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the weak-GPU synthetic true_func_2d benchmark dataset."
    )
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--seed-train", type=int, default=DEFAULT_SEED_TRAIN)
    parser.add_argument("--seed-test", type=int, default=DEFAULT_SEED_TEST)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--dataset-stem", type=str, default=None)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    dataset_stem = (
        str(args.dataset_stem).strip()
        if args.dataset_stem is not None and str(args.dataset_stem).strip()
        else _default_dataset_stem(args.n_train)
    )
    default_npz, default_json = _default_output_paths(dataset_stem)
    output_npz = args.output_npz if args.output_npz is not None else default_npz
    output_json = args.output_json if args.output_json is not None else default_json

    metadata = build_synthetic_dataset(
        output_npz,
        output_json,
        dataset_stem=dataset_stem,
        n_train=args.n_train,
        n_test=args.n_test,
        noise=args.noise,
        seed_train=args.seed_train,
        seed_test=args.seed_test,
        chunk_rows=args.chunk_rows,
    )
    print("saved npz:", output_npz)
    print("saved json:", output_json)
    print("summary:", json.dumps(metadata["shapes"], indent=2))


if __name__ == "__main__":
    main()
