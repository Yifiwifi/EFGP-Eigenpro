from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

try:
    from . import preprocess_synthetic_true_func_2d as _single_dataset
    from .preprocess_synthetic_true_func_2d import (
        DEFAULT_N_TEST,
        DEFAULT_NOISE,
        DEFAULT_SEED_TEST,
        DEFAULT_SEED_TRAIN,
        _default_dataset_stem,
        _default_output_paths,
        build_synthetic_dataset,
    )
except ImportError:  # Direct ``python path/to/script.py`` execution.
    import preprocess_synthetic_true_func_2d as _single_dataset
    from preprocess_synthetic_true_func_2d import (
        DEFAULT_N_TEST,
        DEFAULT_NOISE,
        DEFAULT_SEED_TEST,
        DEFAULT_SEED_TRAIN,
        _default_dataset_stem,
        _default_output_paths,
        build_synthetic_dataset,
    )


DEFAULT_N_TRAIN_LIST = [
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
    30_000_000,
    100_000_000,
    300_000_000,
]


def _parse_n_train_list(s: str | None) -> list[int]:
    if s is None or str(s).strip() == "":
        raise ValueError("n-train list must be a non-empty comma-separated string.")
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip().replace("_", "")
        if not part:
            continue
        val = int(part)
        if val <= 0:
            raise ValueError(f"every n-train value must be > 0, got {val}")
        out.append(val)
    if not out:
        raise ValueError("n-train list is empty after parsing.")
    return sorted(set(out))


def _validate_reusable_prefix_sizes(n_train_list: list[int], *, chunk_rows: int) -> None:
    """Reject size sets whose prefixes differ from independent streamed builds.

    ``_generate_train_to_memmaps`` consumes one RNG stream but interleaves the
    feature and noise draws *inside each chunk*. Consequently, ending an
    independent build in the middle of a chunk changes where its final noise
    draw occurs. Every reused non-maximum prefix must therefore end exactly on
    a chunk boundary to be bit-for-bit equivalent to ``build_synthetic_dataset``.
    """
    sizes = sorted(set(int(value) for value in n_train_list))
    if not sizes:
        raise ValueError("n-train list must be non-empty.")
    chunk_rows = max(1, int(chunk_rows))
    maximum = sizes[-1]
    incompatible = [value for value in sizes if value != maximum and value % chunk_rows != 0]
    if incompatible:
        rendered = ", ".join(str(value) for value in incompatible)
        raise ValueError(
            "--reuse-largest-prefix is not bit-for-bit safe for n-train values "
            f"[{rendered}] with chunk_rows={chunk_rows}. The streamed generator "
            "interleaves feature and noise RNG draws per chunk, so every non-maximum "
            "size must be divisible by chunk_rows. Choose an aligned chunk size or "
            "run without --reuse-largest-prefix."
        )


def _metadata_for_prefix(
    output_npz: Path,
    *,
    dataset_stem: str,
    n_train: int,
    n_test: int,
    noise: float,
    seed_train: int,
    seed_test: int,
    chunk_rows: int,
) -> dict:
    """Reproduce the single-dataset builder's public metadata protocol."""
    n_train = int(n_train)
    n_test = int(n_test)
    return {
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
            "dim": int(_single_dataset.DIM),
            "n_train": n_train,
            "n_test": n_test,
            "noise_std": float(noise),
            "seed_train": int(seed_train),
            "seed_test": int(seed_test),
            "train_distribution": "uniform in [0,1]^2",
            "test_distribution": "uniform in [0,1]^2",
            "chunk_rows": int(chunk_rows),
            "storage_dtype": str(_single_dataset.DEFAULT_STORAGE_DTYPE),
        },
        "split": {
            "method": "pre-generated train/test sets",
            "note": "This synthetic benchmark follows the sanity_check notebook directly instead of splitting one merged pool.",
        },
        "x_transform": {
            "method": "none",
            "domain": "[0,1]^2",
            "dim": int(_single_dataset.DIM),
        },
        "y_transform": {
            "method": "none",
            "noise_model": "additive Gaussian noise on y_train only",
            "noise_std": float(noise),
        },
        "benchmark_reference": {
            "notebook": "gpu/sanity_check/v5_mat32_weak_gpu_complexity_benchmark.ipynb",
            "kernel_family": _single_dataset.KERNEL_FAMILY,
            "lengthscale": float(_single_dataset.KERNEL_LENGTHSCALE),
            "nu": float(_single_dataset.KERNEL_NU),
            "reg_lambda": float(_single_dataset.REG_LAMBDA),
            "eps": float(_single_dataset.EPS),
            "l2_scaled": bool(_single_dataset.L2_SCALED),
        },
        "serialization": {
            "npz_mode": (
                "stored"
                if n_train >= int(_single_dataset.UNCOMPRESSED_NPZ_THRESHOLD_ROWS)
                else "deflated"
            ),
            "array_dtype": str(_single_dataset.DEFAULT_STORAGE_DTYPE),
        },
        "shapes": {
            "n_train": n_train,
            "n_test": n_test,
            "dim": int(_single_dataset.DIM),
        },
        "paper_task_statement": "(x1, x2) -> true_func_2d(x) with noisy train targets",
    }


def _serialize_prefix_dataset(
    output_npz: Path,
    output_json: Path,
    *,
    dataset_stem: str,
    n_train: int,
    n_test: int,
    noise: float,
    seed_train: int,
    seed_test: int,
    chunk_rows: int,
    x_train: np.memmap,
    y_train: np.memmap,
    y_train_true: np.memmap,
    train_noise: np.memmap,
) -> dict:
    """Serialize one output pair from the shared largest memmaps."""
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = output_npz.with_name(f".{output_npz.name}.tmp.npz")
    tmp_json = output_json.with_name(f".{output_json.name}.tmp")

    x_test, y_test = _single_dataset.make_test_set(
        int(_single_dataset.DIM),
        int(n_test),
        _single_dataset.true_func_2d,
        seed=int(seed_test),
    )
    x_test = np.asarray(x_test, dtype=_single_dataset.DEFAULT_STORAGE_DTYPE)
    y_test = np.asarray(y_test, dtype=_single_dataset.DEFAULT_STORAGE_DTYPE)
    metadata = _metadata_for_prefix(
        output_npz,
        dataset_stem=dataset_stem,
        n_train=int(n_train),
        n_test=int(n_test),
        noise=float(noise),
        seed_train=int(seed_train),
        seed_test=int(seed_test),
        chunk_rows=int(chunk_rows),
    )

    try:
        savez_fn = (
            np.savez
            if int(n_train) >= int(_single_dataset.UNCOMPRESSED_NPZ_THRESHOLD_ROWS)
            else np.savez_compressed
        )
        # Prefix slices remain disk-backed views. NumPy's NPZ writer consumes
        # them in bounded buffers instead of copying the largest train array.
        savez_fn(
            tmp_npz,
            x_train=x_train[: int(n_train)],
            x_test=x_test,
            y_train=y_train[: int(n_train)],
            y_test=y_test,
            y_train_true=y_train_true[: int(n_train)],
            train_noise=train_noise[: int(n_train)],
        )
        tmp_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(tmp_npz, output_npz)
        os.replace(tmp_json, output_json)
    finally:
        # Clean up only exact temporaries owned by this output attempt.
        for temporary in (tmp_npz, tmp_json):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return metadata


def build_synthetic_prefix_sweep(
    output_dir: Path,
    *,
    n_train_list: list[int],
    dataset_stem_prefix: str,
    size_token: str,
    noise: float = DEFAULT_NOISE,
    seed_train: int = DEFAULT_SEED_TRAIN,
    seed_test: int = DEFAULT_SEED_TEST,
    chunk_rows: int = 1_000_000,
    continue_on_error: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Generate one largest train memmap and serialize exact aligned prefixes."""
    sizes = sorted(set(int(value) for value in n_train_list))
    chunk_rows = max(1, int(chunk_rows))
    if not sizes:
        raise ValueError("n-train list must be non-empty.")
    if any(value <= 0 for value in sizes):
        raise ValueError("every n-train value must be > 0")
    _validate_reusable_prefix_sizes(sizes, chunk_rows=chunk_rows)
    if int(_single_dataset.DIM) != 2:
        raise ValueError(f"true_func_2d requires DIM=2, got {_single_dataset.DIM}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="synthetic_prefix_tmp_", dir=str(output_dir)))
    x_train = y_train = y_train_true = train_noise = None
    summary_rows: list[dict] = []
    failures: list[dict] = []
    try:
        x_train, y_train, y_train_true, train_noise = _single_dataset._generate_train_to_memmaps(
            work_dir,
            n_train=sizes[-1],
            dim=int(_single_dataset.DIM),
            noise=float(noise),
            seed=int(seed_train),
            chunk_rows=chunk_rows,
            storage_dtype=_single_dataset.DEFAULT_STORAGE_DTYPE,
        )
        for array in (x_train, y_train, y_train_true, train_noise):
            array.flush()

        for n_train in sizes:
            dataset_stem = f"{dataset_stem_prefix}_{size_token}{int(n_train)}"
            output_npz = output_dir / f"{dataset_stem}.npz"
            output_json = output_dir / f"{dataset_stem}.json"
            n_test = int(round(int(n_train) * 0.25))
            print("=" * 100)
            print(f"serializing shared prefix dataset_stem={dataset_stem} | n_train={n_train}")
            try:
                metadata = _serialize_prefix_dataset(
                    output_npz,
                    output_json,
                    dataset_stem=dataset_stem,
                    n_train=int(n_train),
                    n_test=n_test,
                    noise=float(noise),
                    seed_train=int(seed_train),
                    seed_test=int(seed_test),
                    chunk_rows=chunk_rows,
                    x_train=x_train,
                    y_train=y_train,
                    y_train_true=y_train_true,
                    train_noise=train_noise,
                )
                row = {
                    "dataset_stem": dataset_stem,
                    "n_train_requested": int(n_train),
                    "n_test_requested": n_test,
                    "output_npz": str(output_npz),
                    "output_json": str(output_json),
                    **metadata.get("shapes", {}),
                }
                summary_rows.append(row)
                print("saved npz:", output_npz)
                print("saved json:", output_json)
                print("summary:", json.dumps(row, indent=2))
            except Exception as exc:  # noqa: BLE001
                err = {
                    "dataset_stem": dataset_stem,
                    "n_train_requested": int(n_train),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(err)
                print("[ERROR]", json.dumps(err, indent=2))
                if not bool(continue_on_error):
                    raise
    finally:
        _single_dataset._close_memmaps(x_train, y_train, y_train_true, train_noise)
        shutil.rmtree(work_dir, ignore_errors=True)
    return summary_rows, failures


def _default_output_dir() -> Path:
    default_npz, _default_json = _default_output_paths(_default_dataset_stem(DEFAULT_N_TRAIN_LIST[0]))
    return default_npz.parent


def main() -> None:
    default_n_train_list = ",".join(str(v) for v in DEFAULT_N_TRAIN_LIST)
    parser = argparse.ArgumentParser(
        description=(
            "Batch controller for preprocess_synthetic_true_func_2d.py: "
            "generate multiple synthetic datasets with different n_train sizes."
        )
    )
    parser.add_argument("--n-train-list", type=str, default=default_n_train_list)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--seed-train", type=int, default=DEFAULT_SEED_TRAIN)
    parser.add_argument("--seed-test", type=int, default=DEFAULT_SEED_TEST)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=1_000_000,
        help="Streaming generation chunk; archived-paper reconstruction uses 5000000.",
    )
    parser.add_argument("--dataset-stem-prefix", type=str, default="synthetic_true_func_2d")
    parser.add_argument(
        "--size-token",
        choices=("n", "ntrain"),
        default="n",
        help=(
            "Filename token before the row count. Use ntrain with noise=0.3 to "
            "reconstruct the archived-paper Synthetic artifact family."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If set, continue generating remaining sizes after one size fails.",
    )
    parser.add_argument(
        "--reuse-largest-prefix",
        action="store_true",
        help=(
            "Generate the largest train memmaps once and serialize aligned smaller "
            "prefixes. Every non-maximum n-train must be divisible by --chunk-rows."
        ),
    )
    args = parser.parse_args()

    n_train_list = _parse_n_train_list(args.n_train_list)
    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_stem_prefix = str(args.dataset_stem_prefix).strip()
    if not dataset_stem_prefix:
        raise ValueError("dataset-stem-prefix must be non-empty.")

    summary_rows: list[dict] = []
    failures: list[dict] = []

    if bool(args.reuse_largest_prefix):
        summary_rows, failures = build_synthetic_prefix_sweep(
            output_dir,
            n_train_list=n_train_list,
            dataset_stem_prefix=dataset_stem_prefix,
            size_token=str(args.size_token),
            noise=float(args.noise),
            seed_train=int(args.seed_train),
            seed_test=int(args.seed_test),
            chunk_rows=int(args.chunk_rows),
            continue_on_error=bool(args.continue_on_error),
        )
    else:
        for n_train in n_train_list:
            dataset_stem = f"{dataset_stem_prefix}_{args.size_token}{int(n_train)}"
            output_npz = output_dir / f"{dataset_stem}.npz"
            output_json = output_dir / f"{dataset_stem}.json"
            print("=" * 100)
            print(f"generating dataset_stem={dataset_stem} | n_train={int(n_train)}")
            try:
                metadata = build_synthetic_dataset(
                    output_npz,
                    output_json,
                    dataset_stem=dataset_stem,
                    n_train=int(n_train),
                    n_test=int(round(int(n_train) * 0.25)),  # 25% of n_train, so 8:2 train:test ratio
                    noise=float(args.noise),
                    seed_train=int(args.seed_train),
                    seed_test=int(args.seed_test),
                    chunk_rows=int(args.chunk_rows),
                )
                row = {
                    "dataset_stem": dataset_stem,
                    "n_train_requested": int(n_train),
                    "n_test_requested": int(round(int(n_train) * 0.25)),
                    "output_npz": str(output_npz),
                    "output_json": str(output_json),
                    **metadata.get("shapes", {}),
                }
                summary_rows.append(row)
                print("saved npz:", output_npz)
                print("saved json:", output_json)
                print("summary:", json.dumps(row, indent=2))
            except Exception as exc:  # noqa: BLE001
                err = {
                    "dataset_stem": dataset_stem,
                    "n_train_requested": int(n_train),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(err)
                print("[ERROR]", json.dumps(err, indent=2))
                if not bool(args.continue_on_error):
                    raise

    batch_summary = {
        "dataset_stem_prefix": dataset_stem_prefix,
        "size_token": str(args.size_token),
        "output_dir": str(output_dir),
        "n_train_list": [int(v) for v in n_train_list],
        "n_test": int(round(int(n_train_list[0]) * 0.25)),
        "noise": float(args.noise),
        "seed_train": int(args.seed_train),
        "seed_test": int(args.seed_test),
        "chunk_rows": int(args.chunk_rows),
        "reuse_largest_prefix": bool(args.reuse_largest_prefix),
        "generated": summary_rows,
        "failures": failures,
    }
    summary_path = output_dir / f"{dataset_stem_prefix}_size_sweep_summary.json"
    summary_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print("=" * 100)
    print("batch summary json:", summary_path)
    print("generated datasets:", len(summary_rows))
    print("failed datasets:", len(failures))


if __name__ == "__main__":
    main()
