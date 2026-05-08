from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

FeatureMode = Literal["d1_time", "d2_time_daily"]
SplitMethod = Literal["random", "chronological"]
MissingStrategy = Literal["drop", "interpolate", "mean"]


def _default_input_path() -> Path:
    here = Path(__file__).resolve().parent
    return here / "household_power_consumption.txt"


def _default_output_paths(feature_mode: str) -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    processed_dir = here / "processed"
    stem = f"household_power_consumption_global_active_power_{feature_mode}"
    return processed_dir / f"{stem}.npz", processed_dir / f"{stem}.json"


def _read_raw_household_power(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    # pandas can read a zip file directly when it contains a single txt file.
    compression = "zip" if input_path.suffix.lower() == ".zip" else "infer"
    df = pd.read_csv(
        input_path,
        sep=";",
        na_values=["?", "nan", "NaN", ""],
        low_memory=False,
        compression=compression,
    )

    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"missing expected columns {missing_cols}; got columns {list(df.columns)}"
        )

    df = df[RAW_COLUMNS].copy()
    dt_str = df["Date"].astype(str) + " " + df["Time"].astype(str)
    df["datetime"] = pd.to_datetime(
        dt_str,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _clean_dataframe(
    df: pd.DataFrame,
    *,
    target_column: str,
    missing_strategy: MissingStrategy,
    drop_duplicate_timestamps: bool,
) -> tuple[pd.DataFrame, dict]:
    if target_column not in NUMERIC_COLUMNS:
        raise ValueError(f"target_column must be one of {NUMERIC_COLUMNS}, got {target_column}")

    n_raw = int(len(df))
    n_missing_datetime = int(df["datetime"].isna().sum())
    n_missing_target = int(df[target_column].isna().sum())
    n_rows_with_any_numeric_missing = int(df[NUMERIC_COLUMNS].isna().any(axis=1).sum())

    # Remove invalid calendar rows first. The official file should have none, but this
    # makes the script robust to malformed local copies.
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime", kind="mergesort")

    n_duplicate_timestamps = int(df["datetime"].duplicated().sum())
    if drop_duplicate_timestamps:
        df = df.drop_duplicates(subset=["datetime"], keep="first")

    if missing_strategy == "drop":
        # For the KRR/EFGP regression task we only need time coordinates and the
        # target. Dropping target-missing rows is the cleanest option and still leaves
        # about 98.75% of the original minute-level records.
        df = df.dropna(subset=[target_column]).copy()
    elif missing_strategy == "interpolate":
        # Keep the full calendar grid. This is useful if you want strictly regular
        # time coverage, although EFGP itself does not require regular sampling.
        df = df.set_index("datetime")
        df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].interpolate(
            method="time", limit_direction="both"
        )
        df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].ffill().bfill()
        df = df.reset_index()
        df = df.dropna(subset=[target_column]).copy()
    elif missing_strategy == "mean":
        # This matches a common tutorial-style preprocessing choice, but it is not the
        # recommended default for a paper benchmark because it injects flat values.
        means = df[NUMERIC_COLUMNS].mean(numeric_only=True)
        df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(means)
        df = df.dropna(subset=[target_column]).copy()
    else:
        raise ValueError(f"unknown missing_strategy: {missing_strategy}")

    n_clean = int(len(df))
    info = {
        "n_raw": n_raw,
        "n_missing_datetime": n_missing_datetime,
        "n_missing_target": n_missing_target,
        "n_rows_with_any_numeric_missing": n_rows_with_any_numeric_missing,
        "n_duplicate_timestamps": n_duplicate_timestamps,
        "n_clean": n_clean,
        "missing_strategy": str(missing_strategy),
        "drop_duplicate_timestamps": bool(drop_duplicate_timestamps),
    }
    return df, info


def _make_features(df: pd.DataFrame, *, feature_mode: FeatureMode) -> tuple[np.ndarray, dict]:
    dt = df["datetime"]
    t0 = dt.min()
    t1 = dt.max()
    total_seconds = float((t1 - t0).total_seconds())
    if not np.isfinite(total_seconds) or total_seconds <= 0.0:
        raise ValueError(f"invalid time span in seconds: {total_seconds}")

    elapsed_seconds = (dt - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    t_norm = elapsed_seconds / total_seconds

    if feature_mode == "d1_time":
        x = t_norm[:, None]
        feature_description = "normalized global timestamp t in [0,1]"
    elif feature_mode == "d2_time_daily":
        seconds_of_day = (
            dt.dt.hour.to_numpy(dtype=np.float64) * 3600.0
            + dt.dt.minute.to_numpy(dtype=np.float64) * 60.0
            + dt.dt.second.to_numpy(dtype=np.float64)
        )
        daily_phase = seconds_of_day / 86400.0
        x = np.column_stack([t_norm, daily_phase])
        feature_description = (
            "normalized global timestamp and non-periodic daily phase "
            "(seconds since midnight / 86400)"
        )
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode}")

    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite values found in generated features")

    info = {
        "feature_mode": str(feature_mode),
        "feature_description": feature_description,
        "time_origin": str(t0),
        "time_end": str(t1),
        "time_span_seconds": float(total_seconds),
        "x_domain_scaling": "calendar-time coordinates scaled by the full cleaned time domain",
        "dim": int(x.shape[1]),
    }
    return x, info


def _train_test_split_indices(
    n: int,
    *,
    train_ratio: float,
    seed: int,
    split_method: SplitMethod,
) -> tuple[np.ndarray, np.ndarray]:
    if n < 4:
        raise ValueError(f"dataset too small for train/test split: n={n}")
    if not (0.0 < float(train_ratio) < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    n_train = max(1, min(n - 1, int(round(n * float(train_ratio)))))
    if split_method == "random":
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n)
        train_idx = np.sort(perm[:n_train])
        test_idx = np.sort(perm[n_train:])
    elif split_method == "chronological":
        train_idx = np.arange(n_train, dtype=np.int64)
        test_idx = np.arange(n_train, n, dtype=np.int64)
    else:
        raise ValueError(f"unknown split_method: {split_method}")
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def preprocess_household_power_consumption(
    input_path: Path,
    output_npz: Path,
    output_json: Path,
    *,
    feature_mode: FeatureMode = "d1_time",
    target_column: str = "Global_active_power",
    train_ratio: float = 0.8,
    seed: int = 0,
    split_method: SplitMethod = "random",
    missing_strategy: MissingStrategy = "drop",
    drop_duplicate_timestamps: bool = True,
) -> dict:
    """Preprocess UCI household power data for low-dimensional KRR/EFGP.

    Default task:
        X = normalized calendar time, shape (N, 1)
        y = standardized Global_active_power

    Optional d=2 task:
        X = (normalized calendar time, daily phase), shape (N, 2)
        y = standardized Global_active_power
    """
    input_path = Path(input_path)
    output_npz = Path(output_npz)
    output_json = Path(output_json)

    df_raw = _read_raw_household_power(input_path)
    df, clean_info = _clean_dataframe(
        df_raw,
        target_column=target_column,
        missing_strategy=missing_strategy,
        drop_duplicate_timestamps=drop_duplicate_timestamps,
    )

    x_all, feature_info = _make_features(df, feature_mode=feature_mode)
    y_raw_all = df[target_column].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(y_raw_all)):
        raise ValueError("non-finite target values remain after cleaning")

    n = int(x_all.shape[0])
    train_idx, test_idx = _train_test_split_indices(
        n,
        train_ratio=train_ratio,
        seed=seed,
        split_method=split_method,
    )

    x_train = x_all[train_idx]
    x_test = x_all[test_idx]
    y_train_raw = y_raw_all[train_idx]
    y_test_raw = y_raw_all[test_idx]

    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError(f"invalid target std: {y_std}")

    y_train = (y_train_raw - y_mean) / y_std
    y_test = (y_test_raw - y_mean) / y_std

    dt_ns = df["datetime"].astype("int64").to_numpy(dtype=np.int64)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        x_train=np.asarray(x_train, dtype=np.float64),
        x_test=np.asarray(x_test, dtype=np.float64),
        y_train=np.asarray(y_train, dtype=np.float64),
        y_test=np.asarray(y_test, dtype=np.float64),
        y_train_raw=np.asarray(y_train_raw, dtype=np.float64),
        y_test_raw=np.asarray(y_test_raw, dtype=np.float64),
        y_mean=np.asarray([y_mean], dtype=np.float64),
        y_std=np.asarray([y_std], dtype=np.float64),
        train_idx=np.asarray(train_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
        datetime_train_ns=np.asarray(dt_ns[train_idx], dtype=np.int64),
        datetime_test_ns=np.asarray(dt_ns[test_idx], dtype=np.int64),
    )

    metadata = {
        "dataset_name": "individual_household_electric_power_consumption",
        "task_type": f"{feature_info['dim']}d_time_regression",
        "source_file": str(input_path),
        "processed_file": str(output_npz),
        "raw_columns": RAW_COLUMNS,
        "target_column_used": target_column,
        "target_definition": f"standardized {target_column}",
        "input_definition": feature_info["feature_description"],
        "cleaning": clean_info,
        "split": {
            "method": str(split_method),
            "train_ratio": float(train_ratio),
            "test_size": float(1.0 - train_ratio),
            "seed": int(seed),
        },
        "x_transform": feature_info,
        "y_transform": {
            "method": "train_standardization",
            "mean": float(y_mean),
            "std": float(y_std),
        },
        "shapes": {
            "n_clean": int(n),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "dim": int(x_train.shape[1]),
        },
        "paper_task_statement": (
            "t -> global active power"
            if feature_mode == "d1_time"
            else "(t, daily phase) -> global active power"
        ),
        "notes": {
            "recommended_main_task": "Use feature_mode='d1_time' as the cleanest EFGP/KRR sanity benchmark.",
            "d2_caveat": (
                "feature_mode='d2_time_daily' uses a non-periodic daily phase in [0,1). "
                "It can capture daily variation but has a boundary discontinuity at midnight. "
                "A fully periodic daily encoding would use sin/cos and therefore raise the input dimension to 3 if global time is also included."
            ),
        },
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess UCI Individual Household Electric Power Consumption "
            "into a low-dimensional KRR/EFGP regression dataset."
        )
    )
    parser.add_argument("--input", type=Path, default=_default_input_path())
    parser.add_argument(
        "--feature-mode",
        choices=["d1_time", "d2_time_daily"],
        default="d1_time",
        help="d1_time: X=t. d2_time_daily: X=(t, daily_phase).",
    )
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--target-column", type=str, default="Global_active_power")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-method",
        choices=["random", "chronological"],
        default="random",
        help="random is recommended for solver benchmarking; chronological is harder forecasting-style evaluation.",
    )
    parser.add_argument(
        "--missing-strategy",
        choices=["drop", "interpolate", "mean"],
        default="drop",
        help="drop is the clean paper default; interpolate keeps the calendar grid; mean matches common tutorial preprocessing.",
    )
    parser.add_argument(
        "--keep-duplicate-timestamps",
        action="store_true",
        help="Keep duplicate timestamps instead of dropping later duplicates.",
    )
    args = parser.parse_args()

    default_npz, default_json = _default_output_paths(args.feature_mode)
    output_npz = args.output_npz if args.output_npz is not None else default_npz
    output_json = args.output_json if args.output_json is not None else default_json

    metadata = preprocess_household_power_consumption(
        args.input,
        output_npz,
        output_json,
        feature_mode=args.feature_mode,
        target_column=args.target_column,
        train_ratio=args.train_ratio,
        seed=args.seed,
        split_method=args.split_method,
        missing_strategy=args.missing_strategy,
        drop_duplicate_timestamps=not bool(args.keep_duplicate_timestamps),
    )
    print("saved npz:", output_npz)
    print("saved json:", output_json)
    print("summary:", json.dumps(metadata["shapes"], indent=2))


if __name__ == "__main__":
    main()
#python "D:\NU\ML\efgp_eigenpro_py\gpu\benchmark_dataset\preprocess_household_power_consumption.py"