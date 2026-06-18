from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from preprocess_usgs_3dep_lidar_auto import (
    DEFAULT_PROJECT_URL,
    _default_paths,
    _parse_int_list,
    _safe_dataset_name_from_url,
    discover_lidar_urls,
    download_urls,
    preprocess_lidar_elevation,
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
    1000_000_000
]


def _parse_int_list_csv(s: str | None) -> list[int]:
    if s is None or str(s).strip() == "":
        raise ValueError("size list must be a non-empty comma-separated string.")
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip().replace("_", "")
        if not part:
            continue
        val = int(part)
        if val <= 0:
            raise ValueError(f"every size must be > 0, got {val}")
        out.append(val)
    if not out:
        raise ValueError("size list is empty after parsing.")
    return out


def _default_dataset_stem_prefix(project_url: str) -> str:
    dataset_name = _safe_dataset_name_from_url(project_url)
    return f"{dataset_name}_ground_elevation_regression"


def _default_output_dir(project_url: str) -> Path:
    dataset_name = _safe_dataset_name_from_url(project_url)
    _download_dir, default_npz, _default_json = _default_paths(dataset_name)
    return default_npz.parent


def _resolve_local_las_paths(download_dir: Path, max_files: int | None) -> list[Path]:
    las_paths = sorted([p for p in download_dir.iterdir() if p.suffix.lower() in {".las", ".laz"}])
    if max_files is not None:
        las_paths = las_paths[: int(max_files)]
    return las_paths


def _max_total_points_for_target_n_train(n_train: int, *, test_size: float) -> int:
    ts = float(test_size)
    if not (0.0 < ts < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    denom = max(1e-12, 1.0 - ts)
    return int(math.ceil(int(n_train) / denom))


def main() -> None:
    default_n_train_list = ",".join(str(v) for v in DEFAULT_N_TRAIN_LIST)
    parser = argparse.ArgumentParser(
        description=(
            "Batch controller for preprocess_usgs_3dep_lidar_auto.py: "
            "download/discover LAS/LAZ once, then generate multiple processed datasets "
            "with different target n_train sizes (converted to max_total_points using test_size)."
        )
    )
    parser.add_argument("--project-url", type=str, default=DEFAULT_PROJECT_URL)
    # New: user-facing list is N_train targets.
    parser.add_argument("--n-train-list", type=str, default=default_n_train_list)
    # Backward-compatible alias: treat as N_train list when provided.
    parser.add_argument("--size-list", type=str, default=None, help="Alias for --n-train-list (interpreted as N_train targets).")
    parser.add_argument("--dataset-stem-prefix", type=str, default=None)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-files",
        type=int,
        default=64,
        help="Number of LAS/LAZ tiles to use. 0 means all discovered files.",
    )
    parser.add_argument("--file-regex", type=str, default=None)
    parser.add_argument("--recursive-depth", type=int, default=2)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classification-keep",
        type=str,
        default="2",
        help="Comma-separated LAS classes to keep. Default '2' keeps ground points.",
    )
    parser.add_argument(
        "--max-points-per-file",
        type=int,
        default=0,
        help="Optional random subsample cap per tile after classification filtering. 0 means no per-file cap.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If set, continue generating remaining sizes after one size fails.",
    )
    args = parser.parse_args()

    project_url = str(args.project_url).strip()
    dataset_name = _safe_dataset_name_from_url(project_url)
    default_download_dir, _default_npz, _default_json = _default_paths(dataset_name)
    download_dir = args.download_dir if args.download_dir is not None else default_download_dir
    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir(project_url)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_train_list_raw = args.size_list if args.size_list is not None else args.n_train_list
    n_train_list = _parse_int_list_csv(n_train_list_raw)
    dataset_stem_prefix = (
        str(args.dataset_stem_prefix).strip()
        if args.dataset_stem_prefix is not None and str(args.dataset_stem_prefix).strip()
        else _default_dataset_stem_prefix(project_url)
    )
    max_files = None if int(args.max_files) == 0 else int(args.max_files)
    max_points_per_file = None if int(args.max_points_per_file) == 0 else int(args.max_points_per_file)
    classification_keep = _parse_int_list(args.classification_keep)

    if args.skip_download:
        urls: list[str] = []
        las_paths = _resolve_local_las_paths(download_dir, max_files)
    else:
        urls = discover_lidar_urls(
            project_url,
            recursive_depth=int(args.recursive_depth),
            file_regex=args.file_regex,
            max_files=max_files,
        )
        if not urls:
            raise RuntimeError(
                f"No LAS/LAZ files discovered from {project_url}. "
                "Try passing the exact LAZ/laz subdirectory URL."
            )
        print(f"discovered {len(urls)} LAS/LAZ URL(s)")
        las_paths = download_urls(
            urls,
            download_dir,
            overwrite=bool(args.overwrite_downloads),
            timeout=int(args.request_timeout),
            retries=int(args.max_retries),
        )

    if not las_paths:
        raise RuntimeError(f"No local LAS/LAZ files available in {download_dir}")

    print("local LAS/LAZ files:")
    for p in las_paths[:10]:
        print("  ", p)
    if len(las_paths) > 10:
        print(f"  ... and {len(las_paths) - 10} more")

    if args.download_only:
        print("download-only mode; finished")
        return

    summary_rows: list[dict] = []
    failures: list[dict] = []

    for n_train_target in n_train_list:
        max_total_points = _max_total_points_for_target_n_train(
            int(n_train_target),
            test_size=float(args.test_size),
        )
        dataset_stem = f"{dataset_stem_prefix}_ntrain{int(n_train_target)}"
        output_npz = output_dir / f"{dataset_stem}.npz"
        output_json = output_dir / f"{dataset_stem}.json"
        print("=" * 100)
        print(
            f"generating dataset_stem={dataset_stem} | "
            f"target_n_train={int(n_train_target)} -> max_total_points={int(max_total_points)} (test_size={float(args.test_size):g})"
        )
        try:
            metadata = preprocess_lidar_elevation(
                las_paths,
                output_npz,
                output_json,
                dataset_name=dataset_stem,
                source_url=project_url,
                downloaded_urls=urls,
                test_size=float(args.test_size),
                seed=int(args.seed),
                classification_keep=classification_keep,
                max_points_per_file=max_points_per_file,
                max_total_points=int(max_total_points),
            )
            row = {
                "dataset_stem": dataset_stem,
                "n_train_target": int(n_train_target),
                "max_total_points_used": int(max_total_points),
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
                "n_train_target": int(n_train_target),
                "max_total_points_used": int(max_total_points),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(err)
            print("[ERROR]", json.dumps(err, indent=2))
            if not bool(args.continue_on_error):
                raise

    batch_summary = {
        "project_url": project_url,
        "dataset_stem_prefix": dataset_stem_prefix,
        "download_dir": str(download_dir),
        "output_dir": str(output_dir),
        "n_train_list": [int(v) for v in n_train_list],
        "test_size": float(args.test_size),
        "max_files": None if max_files is None else int(max_files),
        "max_points_per_file": None if max_points_per_file is None else int(max_points_per_file),
        "classification_keep": classification_keep,
        "generated": summary_rows,
        "failures": failures,
    }
    summary_path = output_dir / f"{dataset_stem_prefix}_ntrain_sweep_summary.json"
    summary_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print("=" * 100)
    print("batch summary json:", summary_path)
    print("generated datasets:", len(summary_rows))
    print("failed datasets:", len(failures))


if __name__ == "__main__":
    main()
