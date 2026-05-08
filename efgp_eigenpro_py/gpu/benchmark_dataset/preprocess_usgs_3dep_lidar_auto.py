from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen, urlretrieve

import numpy as np


DEFAULT_PROJECT_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "USGS_LPC_IL_Winnebago_2018/laz/"
)
DEFAULT_DATASET_NAME = "usgs_3dep_lidar_winnebago_2018"

# The Rockyweb Apache index may time out on some networks.  For the default
# Winnebago project, keep a small deterministic fallback list so a quick
# benchmark can still start without crawling the whole directory.
DEFAULT_FALLBACK_LAZ_NAMES = [
    "USGS_LPC_IL_Winnebago_2018_2503_2049.laz",
    "USGS_LPC_IL_Winnebago_2018_2503_2051.laz",
    "USGS_LPC_IL_Winnebago_2018_2503_2053.laz",
    "USGS_LPC_IL_Winnebago_2018_2503_2055.laz",
    "USGS_LPC_IL_Winnebago_2018_2503_2057.laz",
    "USGS_LPC_IL_Winnebago_2018_2503_2059.laz",
    "USGS_LPC_IL_Winnebago_2018_2503_2061.laz",
    "USGS_LPC_IL_Winnebago_2018_2503_2063.laz",
]


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def _default_paths(dataset_name: str = DEFAULT_DATASET_NAME) -> tuple[Path, Path, Path]:
    here = Path(__file__).resolve().parent
    download_dir = here / "raw" / dataset_name
    processed_dir = here / "processed"
    npz_path = processed_dir / f"{dataset_name}_ground_elevation_regression.npz"
    json_path = processed_dir / f"{dataset_name}_ground_elevation_regression.json"
    return download_dir, npz_path, json_path


def _as_bool_nonempty(s: str | None) -> bool:
    return s is not None and str(s).strip() != ""


def _parse_int_list(s: str | None) -> list[int] | None:
    """Parse comma-separated integer class labels.

    Empty string means do not filter by classification.
    Default in this script is "2" = ground points for LAS/LAZ.
    """
    if not _as_bool_nonempty(s):
        return None
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out if out else None


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


def _safe_dataset_name_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    parts = [p for p in path.split("/") if p]
    # If URL ends in /laz or /LAZ, use the parent project directory.
    if parts and parts[-1].lower() in {"laz", "las", "laz_all", "classified_laz"}:
        parts = parts[:-1]
    if parts:
        name = parts[-1]
    else:
        name = DEFAULT_DATASET_NAME
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return name or DEFAULT_DATASET_NAME


# -----------------------------------------------------------------------------
# Rockyweb / Apache index discovery + download
# -----------------------------------------------------------------------------


def _fetch_text(url: str, *, timeout: int = 180, retries: int = 3, backoff_s: float = 2.0) -> str:
    """Fetch a text URL with retries.

    Rockyweb directory indexes can be slow from some networks.  A longer default
    timeout plus retry/backoff makes listing much more robust on Windows/PowerShell.
    """
    last_exc: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 lidar-preprocess-script",
                    "Accept": "text/plain,text/html,*/*",
                    "Connection": "close",
                },
            )
            with urlopen(req, timeout=int(timeout)) as resp:
                raw = resp.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < int(retries):
                print(f"[warn] fetch failed ({attempt}/{retries}) for {url}: {exc}; retrying...", file=sys.stderr)
                time.sleep(float(backoff_s) * attempt)
    assert last_exc is not None
    raise last_exc


def _hrefs_from_index(url: str) -> list[str]:
    html = _fetch_text(url)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    out: list[str] = []
    for href in hrefs:
        if href.startswith("?") or href.startswith("#"):
            continue
        if href in {"../", "/"}:
            continue
        out.append(urljoin(url, href))
    return out


def _is_current_rockyweb_lidar_base(base_url: str) -> bool:
    """Return True when base_url is a Rockyweb LAS/LAZ directory.

    Many Rockyweb 0_file_download_links.txt manifests contain old S3 links under
    prd-tnm.s3.amazonaws.com.  Some of those links now return 404 even though
    the same file is still available in the current Rockyweb directory.  When the
    manifest is fetched from a Rockyweb /laz/ directory, the most robust behavior
    is to keep only the discovered filename and rebuild the URL under base_url.
    """
    parsed = urlparse(base_url)
    path = parsed.path.lower().rstrip("/")
    return (
        parsed.netloc.lower().endswith("rockyweb.usgs.gov")
        and path.endswith(("/laz", "/las", "/classified_laz"))
    )


def _canonical_lidar_url(base_url: str, candidate_url_or_name: str) -> str:
    """Canonicalize a discovered LAS/LAZ candidate.

    If base_url is the authoritative Rockyweb directory, rebuild every candidate
    from its basename.  This avoids stale S3 manifest links.  Otherwise, preserve
    full URLs and resolve relative names against base_url.
    """
    candidate = candidate_url_or_name.strip().rstrip(".,);]")
    filename = Path(urlparse(candidate).path).name
    if not filename:
        filename = Path(candidate).name
    if _is_current_rockyweb_lidar_base(base_url):
        base = base_url if base_url.endswith("/") else base_url + "/"
        return urljoin(base, filename)
    if re.match(r"^https?://", candidate, flags=re.IGNORECASE):
        return candidate
    base = base_url if base_url.endswith("/") else base_url + "/"
    return urljoin(base, candidate)


def _extract_lidar_urls_from_text(
    base_url: str,
    text: str,
    *,
    file_regex: str | None = None,
    max_files: int | None = None,
) -> list[str]:
    """Extract LAS/LAZ URLs from either an Apache index or 0_file_download_links.txt.

    Important: for Rockyweb manifests, full S3 links may be stale.  We therefore
    canonicalize all matches to base_url/basename when base_url is a Rockyweb
    LAS/LAZ directory.
    """
    pat = re.compile(file_regex) if _as_bool_nonempty(file_regex) else None
    candidates: list[str] = []

    # 1) Full URLs inside manifest files.
    candidates.extend(re.findall(r"https?://[^\s'\"<>]+?\.(?:la[sz]|LA[ZS])", text))

    # 2) href entries from Apache HTML.
    candidates.extend(
        re.findall(r'href=["\']([^"\']+\.(?:la[sz]|LA[ZS]))["\']', text, flags=re.IGNORECASE)
    )

    # 3) Plain filenames inside text manifests.
    candidates.extend(re.findall(r"[A-Za-z0-9_.-]+\.(?:la[sz]|LA[ZS])", text))

    out: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        u = _canonical_lidar_url(base_url, cand)
        filename = Path(urlparse(u).path).name
        if pat is not None and not pat.search(filename):
            continue
        if not _is_lidar_file_url(u):
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
            if max_files is not None and len(out) >= int(max_files):
                break
    return out


def _manifest_candidates(root_url: str) -> list[str]:
    base = root_url if root_url.endswith("/") else root_url + "/"
    return [
        urljoin(base, "0_file_download_links.txt"),
        urljoin(base, "file_download_links.txt"),
    ]


def _is_lidar_file_url(url: str) -> bool:
    low = url.lower().split("?")[0]
    return low.endswith(".laz") or low.endswith(".las")


def _is_directory_url(url: str) -> bool:
    return url.endswith("/")


def discover_lidar_urls(
    root_url: str,
    *,
    recursive_depth: int = 2,
    file_regex: str | None = None,
    max_files: int | None = None,
) -> list[str]:
    """Discover LAS/LAZ file URLs from a Rockyweb-style directory.

    Pass a selected project directory or its LAZ/laz subdirectory, e.g.
      https://.../Projects/USGS_LPC_IL_Winnebago_2018/laz/

    Do not pass the entire Projects/ root unless you intentionally want a huge crawl.
    """
    root_url = root_url.strip()
    if not root_url.endswith("/") and not _is_lidar_file_url(root_url):
        root_url += "/"

    if _is_lidar_file_url(root_url):
        return [root_url]

    # Allow passing the manifest URL directly.
    if root_url.lower().split("?")[0].endswith(".txt"):
        text = _fetch_text(root_url)
        base = root_url.rsplit("/", 1)[0] + "/"
        return _extract_lidar_urls_from_text(base, text, file_regex=file_regex, max_files=max_files)

    if root_url.rstrip("/").endswith("/Projects"):
        raise ValueError(
            "Do not pass the global Projects/ root. Choose one project directory or its laz/ subdirectory. "
            f"Recommended default: {DEFAULT_PROJECT_URL}"
        )

    # Fast path: many USGS Rockyweb LAS/LAZ folders include a manifest named
    # 0_file_download_links.txt.  Reading it is usually more reliable than
    # parsing a huge HTML directory index.
    for manifest_url in _manifest_candidates(root_url):
        try:
            text = _fetch_text(manifest_url)
            urls = _extract_lidar_urls_from_text(
                root_url, text, file_regex=file_regex, max_files=max_files
            )
            if urls:
                print(f"discovered {len(urls)} LAS/LAZ URL(s) from manifest: {manifest_url}")
                return urls
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to read manifest {manifest_url}: {exc}", file=sys.stderr)

    pat = re.compile(file_regex) if _as_bool_nonempty(file_regex) else None
    found: list[str] = []
    seen_dirs: set[str] = set()

    def visit(url: str, depth: int) -> None:
        if max_files is not None and len(found) >= int(max_files):
            return
        if url in seen_dirs:
            return
        seen_dirs.add(url)

        try:
            hrefs = _hrefs_from_index(url)
        except Exception as exc:
            print(f"[warn] failed to list {url}: {exc}", file=sys.stderr)
            return

        files = sorted([h for h in hrefs if _is_lidar_file_url(h)])
        for f in files:
            name = Path(urlparse(f).path).name
            if pat is not None and not pat.search(name):
                continue
            found.append(f)
            if max_files is not None and len(found) >= int(max_files):
                return

        if depth <= 0:
            return

        # Prefer obvious LAS/LAZ subdirectories first.
        dirs = sorted([h for h in hrefs if _is_directory_url(h)])
        preferred = [d for d in dirs if Path(urlparse(d).path.rstrip("/")).name.lower() in {"laz", "las"}]
        rest = [d for d in dirs if d not in preferred]
        for d in preferred + rest:
            if max_files is not None and len(found) >= int(max_files):
                return
            visit(d, depth - 1)

    visit(root_url, int(recursive_depth))
    # De-duplicate while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for u in found:
        if u not in seen:
            out.append(u)
            seen.add(u)

    # Last-resort fallback for the default project if both manifest and HTML listing
    # fail due to timeouts.  This keeps the script usable for a small benchmark.
    if not out and root_url.rstrip("/") == DEFAULT_PROJECT_URL.rstrip("/"):
        fallback = [urljoin(DEFAULT_PROJECT_URL, name) for name in DEFAULT_FALLBACK_LAZ_NAMES]
        if file_regex:
            pat2 = re.compile(file_regex)
            fallback = [u for u in fallback if pat2.search(Path(urlparse(u).path).name)]
        if max_files is not None:
            fallback = fallback[: int(max_files)]
        print(f"[warn] using built-in fallback list with {len(fallback)} URL(s)", file=sys.stderr)
        return fallback

    return out


def _download_one_url(
    url: str,
    out: Path,
    *,
    timeout: int = 180,
    retries: int = 3,
    chunk_size: int = 1024 * 1024,
) -> None:
    last_exc: Exception | None = None
    tmp = out.with_suffix(out.suffix + ".part")
    for attempt in range(1, int(retries) + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 lidar-preprocess-script", "Connection": "close"})
            with urlopen(req, timeout=int(timeout)) as resp, tmp.open("wb") as f:
                while True:
                    chunk = resp.read(int(chunk_size))
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp.stat().st_size <= 0:
                raise RuntimeError(f"downloaded empty file: {url}")
            tmp.replace(out)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[warn] download failed ({attempt}/{retries}) for {url}: {exc}", file=sys.stderr)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if attempt < int(retries):
                time.sleep(2.0 * attempt)
    assert last_exc is not None
    raise last_exc


def download_urls(
    urls: Iterable[str],
    download_dir: Path,
    *,
    overwrite: bool = False,
    sleep_s: float = 0.0,
    timeout: int = 180,
    retries: int = 3,
) -> list[Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    urls = list(urls)
    for i, url in enumerate(urls, start=1):
        filename = Path(urlparse(url).path).name
        out = download_dir / filename
        if out.exists() and out.stat().st_size > 0 and not overwrite:
            print(f"[{i}/{len(urls)}] exists: {out}")
            paths.append(out)
            continue
        print(f"[{i}/{len(urls)}] downloading: {url}")
        print(f"             -> {out}")
        tmp = out.with_suffix(out.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        _download_one_url(url, out, timeout=timeout, retries=retries)
        paths.append(out)
        if sleep_s > 0:
            time.sleep(float(sleep_s))
    return paths


# -----------------------------------------------------------------------------
# LAS/LAZ preprocessing
# -----------------------------------------------------------------------------


def _require_laspy():
    try:
        import laspy  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "This script needs laspy and a LAZ backend. Install with: pip install laspy lazrs"
        ) from exc
    return laspy


def _read_las_laz_points(
    paths: list[Path],
    *,
    classification_keep: list[int] | None,
    max_points_per_file: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    laspy = _require_laspy()
    rng = np.random.default_rng(int(seed))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    file_summaries: list[dict] = []

    for path in paths:
        las = laspy.read(path)
        n_file = int(len(las.x))

        mask = np.ones(n_file, dtype=bool)
        has_classification = hasattr(las, "classification")
        if classification_keep is not None:
            if not has_classification:
                raise ValueError(
                    f"{path} has no LAS classification field, but classification_keep={classification_keep}. "
                    "Pass --classification-keep '' to disable filtering."
                )
            cls = np.asarray(las.classification)
            mask = np.isin(cls, np.asarray(classification_keep, dtype=cls.dtype))

        idx = np.flatnonzero(mask)
        n_kept_before_sample = int(idx.size)
        if n_kept_before_sample == 0:
            file_summaries.append(
                {
                    "path": str(path),
                    "n_file": n_file,
                    "n_kept_before_sample": 0,
                    "n_used": 0,
                }
            )
            continue

        if max_points_per_file is not None and max_points_per_file > 0 and idx.size > max_points_per_file:
            idx = rng.choice(idx, size=int(max_points_per_file), replace=False)

        # laspy scales coordinates transparently via las.x / las.y / las.z.
        xs.append(np.asarray(las.x[idx], dtype=np.float64))
        ys.append(np.asarray(las.y[idx], dtype=np.float64))
        zs.append(np.asarray(las.z[idx], dtype=np.float64))
        file_summaries.append(
            {
                "path": str(path),
                "n_file": n_file,
                "n_kept_before_sample": n_kept_before_sample,
                "n_used": int(idx.size),
            }
        )

    if not xs:
        raise ValueError("no points were loaded after filtering")

    x_raw = np.column_stack([np.concatenate(xs), np.concatenate(ys)]).astype(np.float64, copy=False)
    y_raw = np.concatenate(zs).astype(np.float64, copy=False)
    metadata = {
        "files": file_summaries,
        "n_files": len(paths),
        "n_loaded_points": int(x_raw.shape[0]),
    }
    return x_raw, y_raw, metadata


def preprocess_lidar_elevation(
    las_paths: list[Path],
    output_npz: Path,
    output_json: Path,
    *,
    dataset_name: str,
    source_url: str | None,
    downloaded_urls: list[str],
    test_size: float = 0.2,
    seed: int = 0,
    classification_keep: list[int] | None = None,
    max_points_per_file: int | None = None,
    max_total_points: int | None = None,
) -> dict:
    x_raw, y_raw, read_meta = _read_las_laz_points(
        las_paths,
        classification_keep=classification_keep,
        max_points_per_file=max_points_per_file,
        seed=seed,
    )

    n_before_total_subsample = int(x_raw.shape[0])
    if max_total_points is not None and max_total_points > 0 and x_raw.shape[0] > max_total_points:
        rng = np.random.default_rng(int(seed) + 1009)
        idx = rng.choice(x_raw.shape[0], size=int(max_total_points), replace=False)
        x_raw = x_raw[idx]
        y_raw = y_raw[idx]

    finite_mask = np.isfinite(x_raw).all(axis=1) & np.isfinite(y_raw)
    x_raw = x_raw[finite_mask]
    y_raw = y_raw[finite_mask]
    n_clean = int(x_raw.shape[0])
    if n_clean < 4:
        raise ValueError(f"too few finite points after cleaning: {n_clean}")

    x_train_raw, x_test_raw, y_train_raw, y_test_raw, train_idx, test_idx = _train_test_split(
        x_raw,
        y_raw,
        test_size=test_size,
        seed=seed,
    )

    # Shared spatial scaling preserves physical aspect ratio.
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
        "dataset_name": dataset_name,
        "task_type": "2d_lidar_elevation_regression",
        "source_url": source_url,
        "downloaded_urls": downloaded_urls,
        "downloaded_files": [str(p) for p in las_paths],
        "processed_file": str(output_npz),
        "input_definition": "LAS/LAZ horizontal coordinates (x, y), shifted and shared-scaled into a bounded 2D domain",
        "target_definition": "standardized LAS/LAZ elevation z",
        "paper_task_statement": "(x, y) -> elevation",
        "cleaning": {
            "finite_only": True,
            "classification_keep": classification_keep,
            "classification_keep_note": "LAS class 2 is ground in standard classified lidar; pass empty string to disable filtering.",
            "max_points_per_file": max_points_per_file,
            "max_total_points": max_total_points,
        },
        "split": {
            "method": "random",
            "test_size": float(test_size),
            "seed": int(seed),
        },
        "x_transform": {
            "method": "shift_and_shared_scale",
            "x_min_train": [float(v) for v in x_min],
            "shared_scale": float(scale),
        },
        "y_transform": {
            "method": "train_standardization",
            "mean": y_mean,
            "std": y_std,
        },
        "shapes": {
            "n_loaded_before_total_subsample": n_before_total_subsample,
            "n_clean": n_clean,
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "dim": 2,
        },
        "read_metadata": read_meta,
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    inferred_name = _safe_dataset_name_from_url(DEFAULT_PROJECT_URL)
    default_download_dir, default_npz, default_json = _default_paths(inferred_name)

    parser = argparse.ArgumentParser(
        description=(
            "Download selected USGS Rockyweb 3DEP LAS/LAZ tiles and preprocess them into "
            "a 2D elevation regression NPZ dataset."
        )
    )
    parser.add_argument(
        "--project-url",
        type=str,
        default=DEFAULT_PROJECT_URL,
        help=(
            "Selected Rockyweb project or LAZ/laz directory URL. Do not use the global Projects/ root. "
            f"Default: {DEFAULT_PROJECT_URL}"
        ),
    )
    parser.add_argument("--download-dir", type=Path, default=default_download_dir)
    parser.add_argument("--output-npz", type=Path, default=default_npz)
    parser.add_argument("--output-json", type=Path, default=default_json)
    parser.add_argument(
        "--max-files",
        type=int,
        default=8,
        help="Number of LAS/LAZ tiles to download/use. Start small; set 0 for all discovered files.",
    )
    parser.add_argument(
        "--file-regex",
        type=str,
        default=None,
        help="Optional regex applied to LAS/LAZ filenames before selecting max-files.",
    )
    parser.add_argument(
        "--recursive-depth",
        type=int,
        default=2,
        help="Directory crawl depth. If project-url already ends with /laz/, depth 0 is enough.",
    )
    parser.add_argument("--request-timeout", type=int, default=180, help="HTTP timeout in seconds for listing/downloading.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count for HTTP listing/downloading.")
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="Use existing LAS/LAZ files in download-dir.")
    parser.add_argument("--download-only", action="store_true", help="Only download files; do not preprocess.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classification-keep",
        type=str,
        default="2",
        help="Comma-separated LAS classes to keep. Default '2' keeps ground points. Empty string disables filtering.",
    )
    parser.add_argument(
        "--max-points-per-file",
        type=int,
        default=0,
        help="Optional random subsample cap per tile after classification filtering. 0 means no per-file cap.",
    )
    parser.add_argument(
        "--max-total-points",
        type=int,
        default=10_000_000,
        help="Optional random subsample cap after loading all tiles. 0 means no total cap.",
    )
    args = parser.parse_args()

    project_url = args.project_url.strip()
    dataset_name = _safe_dataset_name_from_url(project_url)

    max_files = None if args.max_files == 0 else int(args.max_files)
    max_points_per_file = None if args.max_points_per_file == 0 else int(args.max_points_per_file)
    max_total_points = None if args.max_total_points == 0 else int(args.max_total_points)
    classification_keep = _parse_int_list(args.classification_keep)

    if args.skip_download:
        las_paths = sorted(
            [p for p in args.download_dir.iterdir() if p.suffix.lower() in {".las", ".laz"}]
        )
        urls: list[str] = []
        if max_files is not None:
            las_paths = las_paths[:max_files]
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
            args.download_dir,
            overwrite=bool(args.overwrite_downloads),
            timeout=int(args.request_timeout),
            retries=int(args.max_retries),
        )

    if not las_paths:
        raise RuntimeError(f"No local LAS/LAZ files available in {args.download_dir}")

    print("local LAS/LAZ files:")
    for p in las_paths[:10]:
        print("  ", p)
    if len(las_paths) > 10:
        print(f"  ... and {len(las_paths) - 10} more")

    if args.download_only:
        print("download-only mode; finished")
        return

    metadata = preprocess_lidar_elevation(
        las_paths,
        args.output_npz,
        args.output_json,
        dataset_name=dataset_name,
        source_url=project_url,
        downloaded_urls=urls,
        test_size=args.test_size,
        seed=args.seed,
        classification_keep=classification_keep,
        max_points_per_file=max_points_per_file,
        max_total_points=max_total_points,
    )
    print("saved npz:", args.output_npz)
    print("saved json:", args.output_json)
    print("summary:", json.dumps(metadata["shapes"], indent=2))


if __name__ == "__main__":
    main()
