from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---- Synthetic dataset controls ----
DIM = 2
DEFAULT_N_TRAIN = 100_000
DEFAULT_N_TEST = 1_000
DEFAULT_NOISE = 0.02
DEFAULT_SEED_TRAIN = 20260421
DEFAULT_SEED_TEST = 1

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

from efgp_eigenpro_py.benchmark import make_dataset, make_test_set, true_func_2d  # noqa: E402


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
) -> dict:
    if int(DIM) != 2:
        raise ValueError(f"true_func_2d requires DIM=2, got {DIM}")

    x_train, y_train = make_dataset(
        DIM,
        int(n_train),
        true_func_2d,
        noise=float(noise),
        seed=int(seed_train),
    )
    x_test, y_test = make_test_set(
        DIM,
        int(n_test),
        true_func_2d,
        seed=int(seed_test),
    )

    y_train_true = true_func_2d(x_train)
    train_noise = y_train - y_train_true

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        x_train=np.asarray(x_train, dtype=np.float64),
        x_test=np.asarray(x_test, dtype=np.float64),
        y_train=np.asarray(y_train, dtype=np.float64),
        y_test=np.asarray(y_test, dtype=np.float64),
        y_train_true=np.asarray(y_train_true, dtype=np.float64),
        train_noise=np.asarray(train_noise, dtype=np.float64),
    )

    metadata = {
        "dataset_name": str(dataset_stem),
        "task_type": "2d_synthetic_regression",
        "source_file": "generated://efgp_eigenpro_py.benchmark",
        "processed_file": str(output_npz),
        "input_definition": "x_train and x_test are sampled in [0,1]^2 using the benchmark helpers",
        "target_definition": "y_train = true_func_2d(x_train) + Gaussian noise, y_test = true_func_2d(x_test)",
        "generation": {
            "train_generator": "make_dataset",
            "test_generator": "make_test_set",
            "target_function": "true_func_2d",
            "dim": int(DIM),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "noise_std": float(noise),
            "seed_train": int(seed_train),
            "seed_test": int(seed_test),
            "train_distribution": "uniform in [0,1]^2",
            "test_distribution": "uniform in [0,1]^2",
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
        "shapes": {
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "dim": int(x_train.shape[1]),
        },
        "paper_task_statement": "(x1, x2) -> true_func_2d(x) with noisy train targets",
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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
    )
    print("saved npz:", output_npz)
    print("saved json:", output_json)
    print("summary:", json.dumps(metadata["shapes"], indent=2))


if __name__ == "__main__":
    main()
