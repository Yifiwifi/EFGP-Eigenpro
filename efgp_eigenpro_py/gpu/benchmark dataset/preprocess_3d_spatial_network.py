from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


RAW_COLUMNS = ["osm_id", "longitude", "latitude", "altitude"]


def _default_paths() -> tuple[Path, Path, Path]:
    here = Path(__file__).resolve().parent
    raw_path = here / "3D_spatial_network.txt"
    processed_dir = here / "processed"
    npz_path = processed_dir / "3D_spatial_network_utm32_altitude_regression.npz"
    json_path = processed_dir / "3D_spatial_network_utm32_altitude_regression.json"
    return raw_path, npz_path, json_path


def _train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    if n < 4:
        raise ValueError(f"dataset too small for train/test split: n={n}")
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_test = max(1, min(n - 1, int(round(n * float(test_size)))))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def preprocess_3d_spatial_network(
    input_path: Path,
    output_npz: Path,
    output_json: Path,
    *,
    test_size: float = 0.2,
    seed: int = 0,
    drop_duplicates: bool = True,
) -> dict:
    df = pd.read_csv(
        input_path,
        header=None,
        names=RAW_COLUMNS,
    )
    n_raw = int(len(df))

    df = df.dropna()
    n_after_dropna = int(len(df))

    if drop_duplicates:
        df = df.drop_duplicates(subset=["longitude", "latitude", "altitude"])
    n_clean = int(len(df))

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    east, north = transformer.transform(
        df["longitude"].to_numpy(dtype=np.float64),
        df["latitude"].to_numpy(dtype=np.float64),
    )

    x_raw = np.column_stack([east, north]).astype(np.float64, copy=False)
    y_raw = df["altitude"].to_numpy(dtype=np.float64)

    x_train_raw, x_test_raw, y_train_raw, y_test_raw, train_idx, test_idx = _train_test_split(
        x_raw,
        y_raw,
        test_size=test_size,
        seed=seed,
    )

    x_min = np.min(x_train_raw, axis=0)
    x_train_shifted = x_train_raw - x_min
    x_test_shifted = x_test_raw - x_min
    scale = float(np.max(x_train_shifted))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid spatial scale: {scale}")

    x_train = x_train_shifted / scale
    x_test = x_test_shifted / scale

    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw))
    if not np.isfinite(y_std) or y_std <= 0.0:
        raise ValueError(f"invalid target std: {y_std}")

    y_train = (y_train_raw - y_mean) / y_std
    y_test = (y_test_raw - y_mean) / y_std

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        x_train=np.asarray(x_train, dtype=np.float64),
        x_test=np.asarray(x_test, dtype=np.float64),
        y_train=np.asarray(y_train, dtype=np.float64),
        y_test=np.asarray(y_test, dtype=np.float64),
        x_min=np.asarray(x_min, dtype=np.float64),
        x_scale=np.asarray([scale], dtype=np.float64),
        y_mean=np.asarray([y_mean], dtype=np.float64),
        y_std=np.asarray([y_std], dtype=np.float64),
        train_idx=np.asarray(train_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )

    metadata = {
        "dataset_name": "3D_spatial_network",
        "task_type": "2d_spatial_regression",
        "source_file": str(input_path),
        "processed_file": str(output_npz),
        "input_definition": "projected spatial coordinates (east, north), scaled with a shared scalar into [0,1]^2",
        "target_definition": "standardized altitude",
        "raw_columns": RAW_COLUMNS,
        "feature_columns_used": ["longitude", "latitude"],
        "target_column_used": "altitude",
        "projection": {
            "source_crs": "EPSG:4326",
            "target_crs": "EPSG:32632",
            "target_crs_name": "WGS 84 / UTM zone 32N",
        },
        "cleaning": {
            "dropna": True,
            "drop_duplicates": bool(drop_duplicates),
            "drop_duplicate_subset": ["longitude", "latitude", "altitude"],
            "drop_osm_id_from_model_input": True,
        },
        "split": {
            "method": "random",
            "test_size": float(test_size),
            "seed": int(seed),
        },
        "x_transform": {
            "method": "project_to_utm_then_shift_and_shared_scale",
            "x_min_train": [float(v) for v in x_min],
            "shared_scale": float(scale),
        },
        "y_transform": {
            "method": "train_standardization",
            "mean": float(y_mean),
            "std": float(y_std),
        },
        "shapes": {
            "n_raw": int(n_raw),
            "n_after_dropna": int(n_after_dropna),
            "n_clean": int(n_clean),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "dim": 2,
        },
        "paper_task_statement": "(east, north) -> altitude",
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    default_input, default_npz, default_json = _default_paths()

    parser = argparse.ArgumentParser(
        description="Preprocess 3D_spatial_network into a standard 2D spatial regression dataset."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-npz", type=Path, default=default_npz)
    parser.add_argument("--output-json", type=Path, default=default_json)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate (longitude, latitude, altitude) rows instead of dropping them.",
    )
    args = parser.parse_args()

    metadata = preprocess_3d_spatial_network(
        args.input,
        args.output_npz,
        args.output_json,
        test_size=args.test_size,
        seed=args.seed,
        drop_duplicates=not bool(args.keep_duplicates),
    )
    print("saved npz:", args.output_npz)
    print("saved json:", args.output_json)
    print("summary:", json.dumps(metadata["shapes"], indent=2))


if __name__ == "__main__":
    main()
