from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return out


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
    parser.add_argument("--dataset-stem-prefix", type=str, default="synthetic_true_func_2d")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If set, continue generating remaining sizes after one size fails.",
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

    for n_train in n_train_list:
        dataset_stem = f"{dataset_stem_prefix}_n{int(n_train)}"
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
                n_test=int(args.n_test),
                noise=float(args.noise),
                seed_train=int(args.seed_train),
                seed_test=int(args.seed_test),
            )
            row = {
                "dataset_stem": dataset_stem,
                "n_train_requested": int(n_train),
                "n_test_requested": int(args.n_test),
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
        "output_dir": str(output_dir),
        "n_train_list": [int(v) for v in n_train_list],
        "n_test": int(args.n_test),
        "noise": float(args.noise),
        "seed_train": int(args.seed_train),
        "seed_test": int(args.seed_test),
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
