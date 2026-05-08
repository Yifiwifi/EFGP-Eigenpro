from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


# ---- Synthetic dataset controls ----
DATASET_STEM = "synthetic_true_func_2d_n100000"
DIM = 2
N_TRAIN = 100_000
N_TEST = 1_000
NOISE = 0.02
SEED_TRAIN = 20260421
SEED_TEST = 1

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


def _default_output_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    processed_dir = here / "processed"
    return processed_dir / f"{DATASET_STEM}.npz", processed_dir / f"{DATASET_STEM}.json"


def build_synthetic_dataset(
    output_npz: Path,
    output_json: Path,
) -> dict:
    x_train, y_train = make_dataset(
        DIM,
        N_TRAIN,
        true_func_2d,
        noise=NOISE,
        seed=SEED_TRAIN,
    )
    x_test, y_test = make_test_set(
        DIM,
        N_TEST,
        true_func_2d,
        seed=SEED_TEST,
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
        "dataset_name": DATASET_STEM,
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
            "n_train": int(N_TRAIN),
            "n_test": int(N_TEST),
            "noise_std": float(NOISE),
            "seed_train": int(SEED_TRAIN),
            "seed_test": int(SEED_TEST),
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
            "noise_std": float(NOISE),
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
    output_npz, output_json = _default_output_paths()
    metadata = build_synthetic_dataset(output_npz, output_json)
    print("saved npz:", output_npz)
    print("saved json:", output_json)
    print("summary:", json.dumps(metadata["shapes"], indent=2))


if __name__ == "__main__":
    main()
