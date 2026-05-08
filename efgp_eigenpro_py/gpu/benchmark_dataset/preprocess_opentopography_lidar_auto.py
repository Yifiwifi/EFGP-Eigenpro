from __future__ import annotations

"""
Download and preprocess OpenTopography LiDAR point-cloud tiles for EFGP/KRR benchmarks.

Target task:
    X = horizontal point-cloud coordinates (x, y)
    y = elevation z

Recommended use cases:
  1) Preferred / robust: pass an OpenTopography tile-index file URL and a bbox.
       python preprocess_opentopography_lidar_auto.py \
         --tile-index-url "https://.../tile_index.zip" \
         --bbox "lon_min,lat_min,lon_max,lat_max" \
         --max-files 8 --max-total-points 1000000

  2) If you already have a local tile-index shapefile/GeoPackage/GeoJSON:
       python preprocess_opentopography_lidar_auto.py \
         --tile-index-path path/to/tile_index.zip \
         --bbox "lon_min,lat_min,lon_max,lat_max"

  3) If you have a text file of LAZ/LAS URLs:
       python preprocess_opentopography_lidar_auto.py \
         --tile-url-list urls.txt --max-files 8

  4) If you already downloaded LAS/LAZ files:
       python preprocess_opentopography_lidar_auto.py \
         --skip-download --download-dir raw/opentopography_lidar

Notes:
  * For tile-index mode, install geospatial deps:
        pip install geopandas shapely pyproj fiona
    or, preferably with conda:
        conda install -c conda-forge geopandas shapely pyproj fiona
  * For reading LAZ/LAS:
        pip install laspy lazrs numpy pandas
  * OpenTopography access depends on dataset/account permissions. Some USGS/NOAA/OT+
    datasets may require login, academic access, or OT+ subscription. This script
    cannot bypass those restrictions; it downloads only URLs that your network/account
    can access.
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np

LAS_EXTENSIONS = {".las", ".laz", ".copc.laz"}
TILE_INDEX_EXTENSIONS = {".zip", ".gpkg", ".geojson", ".json", ".shp"}

DEFAULT_DATASET_NAME = "OpenTopography_LiDAR_point_cloud"
DEFAULT_PROJECT_SLUG = "opentopography_lidar"


def _default_paths() -> tuple[Path, Path, Path, Path]:
    here = Path(__file__).resolve().parent
    raw_dir = here / "raw" / DEFAULT_PROJECT_SLUG
    processed_dir = here / "processed"
    npz_path = processed_dir / f"{DEFAULT_PROJECT_SLUG}_ground_elevation_regression.npz"
    json_path = processed_dir / f"{DEFAULT_PROJECT_SLUG}_ground_elevation_regression.json"
    tile_index_dir = here / "raw" / DEFAULT_PROJECT_SLUG / "tile_index"
    return raw_dir, npz_path, json_path, tile_index_dir


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v)


def _is_http_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")


def _read_url_text(url: str, *, timeout: int, retries: int) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (EFGP OpenTopography preprocessing)"})
            with urlopen(req, timeout=int(timeout)) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[warn] failed to read URL ({attempt}/{retries}) {url}: {exc}", file=sys.stderr)
            time.sleep(min(2.0 * attempt, 10.0))
    assert last_exc is not None
    raise last_exc


def _download_one_url(url: str, out_path: Path, *, timeout: int, retries: int, overwrite: bool = False) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        print(f"[skip] exists: {out_path}")
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"downloading: {url}\n        -> {out_path}")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (EFGP OpenTopography preprocessing)"})
            with urlopen(req, timeout=int(timeout)) as resp, tmp_path.open("wb") as f:
                shutil.copyfileobj(resp, f, length=1024 * 1024)
            if tmp_path.stat().st_size <= 0:
                raise IOError(f"downloaded empty file: {tmp_path}")
            tmp_path.replace(out_path)
            return out_path
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[warn] download failed ({attempt}/{retries}) for {url}: {exc}", file=sys.stderr)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            time.sleep(min(2.0 * attempt, 10.0))
    assert last_exc is not None
    raise last_exc


def _filename_from_url(url: str, default_suffix: str = "") -> str:
    path = urlparse(url).path
    name = Path(path).name
    if not name:
        name = "download" + default_suffix
    name = re.sub(r"[^A-Za-z0-9._\-]+", "_", name)
    return name


def _extract_links_from_dataset_page(dataset_page_url: str, *, timeout: int, retries: int) -> tuple[list[str], list[str]]:
    """Return candidate tile-index links and direct LAS/LAZ links from a dataset landing page.

    This is a best-effort scraper. OpenTopography pages can change and some download
    links are produced after login/job submission, so robust workflows should pass
    --tile-index-url, --tile-index-path, or --tile-url-list explicitly.
    """
    html = _read_url_text(dataset_page_url, timeout=timeout, retries=retries)
    parser = _HrefParser()
    parser.feed(html)
    links = [urljoin(dataset_page_url, href) for href in parser.hrefs]

    # Also capture raw absolute URLs that may appear in scripts or JSON snippets.
    links += re.findall(r"https?://[^\"'<>\s]+", html)
    links = sorted(set(link.strip().rstrip(")],;")) for link in links if link.strip())

    tile_index_links: list[str] = []
    lidar_links: list[str] = []
    for link in links:
        lower = link.lower()
        base = Path(urlparse(link).path).name.lower()
        if lower.endswith((".las", ".laz", ".copc.laz")):
            lidar_links.append(link)
        elif (
            any(lower.endswith(ext) for ext in TILE_INDEX_EXTENSIONS)
            and ("tile" in lower or "index" in lower or "tiles" in lower)
        ):
            tile_index_links.append(link)
        elif "tile" in lower and "index" in lower:
            tile_index_links.append(link)
        elif base.endswith((".zip", ".gpkg", ".geojson", ".json", ".shp")) and "index" in base:
            tile_index_links.append(link)

    return sorted(set(tile_index_links)), sorted(set(lidar_links))


def _parse_text_url_list(text: str, *, base_url: str | None = None) -> list[str]:
    urls: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip('"\'')
        if not line or line.startswith("#"):
            continue
        # Extract a URL from CSV-ish or whitespace-ish lines.
        found = re.findall(r"https?://[^,\s]+", line)
        if found:
            urls.extend([u.rstrip(")],;") for u in found])
            continue
        if base_url and (line.lower().endswith((".las", ".laz", ".copc.laz"))):
            urls.append(urljoin(base_url, line))
    return sorted(set(urls))


def _read_tile_url_list(path_or_url: str | Path, *, timeout: int, retries: int) -> list[str]:
    s = str(path_or_url)
    if _is_http_url(s):
        text = _read_url_text(s, timeout=timeout, retries=retries)
        return _parse_text_url_list(text, base_url=s)
    text = Path(s).read_text(encoding="utf-8", errors="replace")
    return _parse_text_url_list(text)


def _maybe_download_tile_index(tile_index_url: str, tile_index_dir: Path, *, timeout: int, retries: int) -> Path:
    tile_index_dir.mkdir(parents=True, exist_ok=True)
    name = _filename_from_url(tile_index_url, default_suffix=".zip")
    out = tile_index_dir / name
    return _download_one_url(tile_index_url, out, timeout=timeout, retries=retries)


def _extract_zip_if_needed(path: Path, extract_dir: Path) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return path
    out_dir = extract_dir / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        return out_dir
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return out_dir


def _parse_bbox(text: str | None) -> tuple[float, float, float, float] | None:
    if text is None or str(text).strip() == "":
        return None
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError("bbox must be 'xmin,ymin,xmax,ymax' in EPSG:4326 lon/lat coordinates")
    xmin, ymin, xmax, ymax = vals
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bbox must satisfy xmin < xmax and ymin < ymax")
    return xmin, ymin, xmax, ymax


def _find_vector_file(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates: list[Path] = []
    for ext in ("*.gpkg", "*.geojson", "*.json", "*.shp"):
        candidates.extend(sorted(path.rglob(ext)))
    if not candidates:
        raise FileNotFoundError(f"No vector tile-index file found under {path}; expected .shp/.gpkg/.geojson/.json")
    # Prefer files whose names suggest a tile index.
    for cand in candidates:
        lower = cand.name.lower()
        if "tile" in lower and "index" in lower:
            return cand
    return candidates[0]


def _discover_url_column(gdf) -> str:
    candidate_cols = []
    for col in gdf.columns:
        if col == getattr(gdf, "geometry", None):
            continue
        lower = str(col).lower()
        if any(k in lower for k in ["url", "href", "link", "download", "location", "path", "s3", "file"]):
            candidate_cols.append(col)

    # First try candidate columns with many URL-looking values.
    for col in candidate_cols + list(gdf.columns):
        if col == gdf.geometry.name:
            continue
        vals = gdf[col].dropna().astype(str).head(200).tolist()
        n_url = sum(("http://" in v or "https://" in v) and v.lower().endswith((".las", ".laz", ".copc.laz")) for v in vals)
        if n_url > 0:
            return str(col)

    # Then try columns that contain bare filenames.
    for col in candidate_cols + list(gdf.columns):
        if col == gdf.geometry.name:
            continue
        vals = gdf[col].dropna().astype(str).head(200).tolist()
        n_file = sum(v.lower().endswith((".las", ".laz", ".copc.laz")) for v in vals)
        if n_file > 0:
            return str(col)

    raise ValueError(
        "Could not find a LAS/LAZ URL or filename column in the tile index. "
        f"Columns: {list(gdf.columns)}. Pass --tile-url-column explicitly."
    )


def _urls_from_tile_index(
    tile_index_path: Path,
    *,
    bbox_lonlat: tuple[float, float, float, float] | None,
    tile_url_column: str | None,
    base_url: str | None,
) -> tuple[list[str], dict]:
    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import box  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Tile-index mode requires geopandas and shapely. Install with: "
            "conda install -c conda-forge geopandas shapely pyproj fiona"
        ) from exc

    path = _extract_zip_if_needed(Path(tile_index_path), Path(tile_index_path).parent / "_tile_index_extracted")
    vector_file = _find_vector_file(path)
    print(f"reading tile index: {vector_file}")
    gdf = gpd.read_file(vector_file)
    n_total = int(len(gdf))
    if n_total == 0:
        raise ValueError(f"tile index is empty: {vector_file}")

    if bbox_lonlat is not None:
        bbox_geom = gpd.GeoDataFrame(geometry=[box(*bbox_lonlat)], crs="EPSG:4326")
        if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
            bbox_geom = bbox_geom.to_crs(gdf.crs)
        elif gdf.crs is None:
            print("[warn] tile index has no CRS; assuming bbox is in the same CRS as tile geometries", file=sys.stderr)
        query_geom = bbox_geom.geometry.iloc[0]
        gdf = gdf[gdf.geometry.intersects(query_geom)]
        print(f"tile index bbox selection: {len(gdf)}/{n_total} intersect bbox")

    if len(gdf) == 0:
        raise ValueError("No tile-index records intersect the requested bbox.")

    col = tile_url_column or _discover_url_column(gdf)
    vals = gdf[col].dropna().astype(str).tolist()
    urls: list[str] = []
    for v in vals:
        v = v.strip().strip('"\'')
        if not v:
            continue
        found = re.findall(r"https?://[^,\s]+", v)
        if found:
            urls.extend([u.rstrip(")],;") for u in found if u.lower().endswith((".las", ".laz", ".copc.laz"))])
        elif v.lower().endswith((".las", ".laz", ".copc.laz")):
            if base_url:
                urls.append(urljoin(base_url.rstrip("/") + "/", v))
            else:
                # Local filename in tile index: not downloadable unless base-url is known.
                urls.append(v)

    urls = sorted(set(urls))
    if not urls:
        raise ValueError(f"Tile index column {col!r} produced no LAS/LAZ URLs or filenames.")

    meta = {
        "tile_index_path": str(tile_index_path),
        "tile_index_vector_file": str(vector_file),
        "tile_index_crs": str(gdf.crs) if gdf.crs is not None else None,
        "tile_url_column": str(col),
        "n_tiles_total": n_total,
        "n_tiles_selected": int(len(gdf)),
        "bbox_lonlat": list(bbox_lonlat) if bbox_lonlat is not None else None,
        "base_url_for_relative_tile_paths": base_url,
    }
    return urls, meta


def _normalize_download_urls(urls: Sequence[str], *, base_url: str | None = None) -> list[str]:
    out: list[str] = []
    for raw in urls:
        for part in re.split(r"[,\s]+", str(raw)):
            u = part.strip().strip('"\'')
            if not u:
                continue
            if _is_http_url(u):
                out.append(u)
            elif base_url and u.lower().endswith((".las", ".laz", ".copc.laz")):
                out.append(urljoin(base_url.rstrip("/") + "/", u))
    return sorted(set(out))


def _download_lidar_urls(
    urls: Sequence[str],
    out_dir: Path,
    *,
    max_files: int | None,
    timeout: int,
    retries: int,
    continue_on_error: bool,
) -> tuple[list[Path], list[dict]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = list(urls)
    if max_files is not None and int(max_files) > 0:
        selected = selected[: int(max_files)]
    if not selected:
        raise ValueError("No LAS/LAZ URLs selected for download.")

    paths: list[Path] = []
    failures: list[dict] = []
    for i, url in enumerate(selected, start=1):
        name = _filename_from_url(url, default_suffix=".laz")
        out = out_dir / name
        print(f"[{i}/{len(selected)}] {url}")
        try:
            paths.append(_download_one_url(url, out, timeout=timeout, retries=retries))
        except Exception as exc:  # noqa: BLE001
            failures.append({"url": url, "error": repr(exc)})
            if not continue_on_error:
                raise
            print(f"[warn] skipping failed URL: {url}: {exc}", file=sys.stderr)

    if not paths:
        raise RuntimeError("All downloads failed; no LAS/LAZ files available for preprocessing.")
    return paths, failures


def _find_lidar_files(input_path: Path, *, max_files: int | None = None) -> list[Path]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    files: list[Path]
    if input_path.is_file():
        lower = input_path.name.lower()
        if lower.endswith((".las", ".laz", ".copc.laz")):
            files = [input_path]
        else:
            raise ValueError(f"unsupported local point-cloud file: {input_path}")
    else:
        files = sorted(input_path.rglob("*.las")) + sorted(input_path.rglob("*.laz"))
        files = sorted(set(files))
    if max_files is not None and int(max_files) > 0:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"no .las/.laz files found under {input_path}")
    return files


def _parse_int_list(text: str | None) -> list[int] | None:
    if text is None or str(text).strip() == "":
        return None
    return [int(v.strip()) for v in str(text).split(",") if v.strip()]


def _maybe_import_laspy():
    try:
        import laspy  # type: ignore
    except ImportError as exc:
        raise ImportError("Install LAS/LAZ dependencies with: pip install laspy lazrs") from exc
    return laspy


def _maybe_import_pyproj():
    try:
        from pyproj import Transformer  # type: ignore
    except ImportError as exc:
        raise ImportError("Reprojection requires pyproj. Install with: pip install pyproj") from exc
    return Transformer


def _read_las_laz_points(
    files: Iterable[Path],
    *,
    classification_keep: list[int] | None,
    target_epsg: int | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    laspy = _maybe_import_laspy()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    summaries: list[dict] = []

    for fp in files:
        fp = Path(fp)
        print(f"reading point cloud: {fp}")
        las = laspy.read(fp)
        n_raw = int(len(las.points))
        mask = np.ones(n_raw, dtype=bool)
        cls_values = None
        if hasattr(las, "classification"):
            cls_values = np.asarray(las.classification)
        if classification_keep is not None:
            if cls_values is None:
                raise ValueError(f"classification filter requested but file has no classification: {fp}")
            mask = np.isin(cls_values, np.asarray(classification_keep, dtype=cls_values.dtype))

        x = np.asarray(las.x, dtype=np.float64)[mask]
        y = np.asarray(las.y, dtype=np.float64)[mask]
        z = np.asarray(las.z, dtype=np.float64)[mask]

        crs = None
        try:
            crs = las.header.parse_crs()
        except Exception:
            crs = None

        if target_epsg is not None:
            if crs is None:
                raise ValueError(
                    f"--target-epsg was requested, but CRS metadata is missing in {fp}. "
                    "Omit --target-epsg or reproject outside this script."
                )
            Transformer = _maybe_import_pyproj()
            target = f"EPSG:{int(target_epsg)}"
            if crs.to_string() != target:
                transformer = Transformer.from_crs(crs, target, always_xy=True)
                x, y = transformer.transform(x, y)

        xs.append(x)
        ys.append(y)
        zs.append(z)
        summaries.append(
            {
                "file": str(fp),
                "n_points_raw": n_raw,
                "n_points_kept_after_class_filter": int(mask.sum()),
                "classification_keep": classification_keep,
                "crs": crs.to_string() if crs is not None else None,
            }
        )

    xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)]).astype(np.float64, copy=False)
    z_all = np.concatenate(zs).astype(np.float64, copy=False)
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(z_all)
    xy = xy[finite]
    z_all = z_all[finite]
    meta = {
        "files": summaries,
        "n_files": len(summaries),
        "classification_keep": classification_keep,
        "target_epsg": int(target_epsg) if target_epsg is not None else None,
        "n_after_file_filters": int(sum(s["n_points_kept_after_class_filter"] for s in summaries)),
        "n_after_finite_filter": int(xy.shape[0]),
    }
    return xy, z_all, meta


def _clip_xy_bbox(xy: np.ndarray, z: np.ndarray, bbox_xy: tuple[float, float, float, float] | None) -> tuple[np.ndarray, np.ndarray, int]:
    if bbox_xy is None:
        return xy, z, int(xy.shape[0])
    xmin, ymin, xmax, ymax = bbox_xy
    mask = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
    return xy[mask], z[mask], int(mask.sum())


def _subsample(x: np.ndarray, y: np.ndarray, *, max_points: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    all_idx = np.arange(n, dtype=np.int64)
    if max_points is None or int(max_points) <= 0 or int(max_points) >= n:
        return x, y, all_idx
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
    return x[idx], y[idx], idx.astype(np.int64, copy=False)


def _train_test_split(x: np.ndarray, y: np.ndarray, *, test_size: float, seed: int):
    n = int(x.shape[0])
    if n < 4:
        raise ValueError(f"dataset too small for train/test split: n={n}")
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError(f"test_size must be in (0,1), got {test_size}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_test = max(1, min(n - 1, int(round(n * float(test_size)))))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def _drop_duplicate_xy_keep_first(xy: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if xy.shape[0] == 0:
        return xy, z, 0
    # Exact duplicate coordinates only. Avoid rounding because LiDAR coordinates are already quantized.
    structured = np.ascontiguousarray(xy).view([("x", xy.dtype), ("y", xy.dtype)]).reshape(-1)
    _, idx = np.unique(structured, return_index=True)
    idx = np.sort(idx)
    return xy[idx], z[idx], int(xy.shape[0] - idx.shape[0])


def preprocess_lidar_elevation_regression(
    las_paths: Sequence[Path],
    output_npz: Path,
    output_json: Path,
    *,
    dataset_name: str,
    source_description: dict,
    test_size: float,
    seed: int,
    classification_keep: list[int] | None,
    target_epsg: int | None,
    max_total_points: int | None,
    bbox_xy: tuple[float, float, float, float] | None,
    drop_duplicate_xy: bool,
) -> dict:
    xy_raw, z_raw, read_meta = _read_las_laz_points(
        las_paths,
        classification_keep=classification_keep,
        target_epsg=target_epsg,
    )
    n_after_read = int(xy_raw.shape[0])
    xy_raw, z_raw, n_after_bbox = _clip_xy_bbox(xy_raw, z_raw, bbox_xy)
    n_after_xy_bbox = int(xy_raw.shape[0])
    if drop_duplicate_xy:
        xy_raw, z_raw, n_dup_dropped = _drop_duplicate_xy_keep_first(xy_raw, z_raw)
    else:
        n_dup_dropped = 0
    n_after_dedup = int(xy_raw.shape[0])
    xy_raw, z_raw, subsample_idx = _subsample(xy_raw, z_raw, max_points=max_total_points, seed=seed)
    n_after_subsample = int(xy_raw.shape[0])
    if n_after_subsample < 4:
        raise ValueError(f"not enough points after filtering: {n_after_subsample}")

    x_train_raw, x_test_raw, y_train_raw, y_test_raw, train_idx, test_idx = _train_test_split(
        xy_raw,
        z_raw,
        test_size=test_size,
        seed=seed,
    )

    # Shared scale preserves horizontal physical aspect ratio.
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
        subsample_idx=np.asarray(subsample_idx, dtype=np.int64),
    )

    metadata = {
        "dataset_name": dataset_name,
        "task_type": "2d_lidar_elevation_regression",
        "source": source_description,
        "processed_file": str(output_npz),
        "input_definition": "LiDAR horizontal point coordinates (x, y), shifted by train min and scaled by one shared scalar",
        "target_definition": "standardized LiDAR elevation z",
        "feature_columns_used": ["x", "y"],
        "target_column_used": "z",
        "cleaning": {
            "classification_keep": classification_keep,
            "target_epsg": int(target_epsg) if target_epsg is not None else None,
            "bbox_xy": list(bbox_xy) if bbox_xy is not None else None,
            "drop_duplicate_xy": bool(drop_duplicate_xy),
            "n_duplicate_xy_dropped": int(n_dup_dropped),
            "max_total_points": int(max_total_points) if max_total_points is not None and int(max_total_points) > 0 else None,
        },
        "split": {
            "method": "random",
            "test_size": float(test_size),
            "train_ratio": float(1.0 - float(test_size)),
            "seed": int(seed),
        },
        "x_transform": {
            "method": "shift_and_shared_scale",
            "x_min_train": [float(v) for v in x_min],
            "shared_scale": float(scale),
        },
        "y_transform": {
            "method": "train_standardization",
            "mean": float(y_mean),
            "std": float(y_std),
        },
        "las_reading": read_meta,
        "shapes": {
            "n_after_read": int(n_after_read),
            "n_after_xy_bbox": int(n_after_xy_bbox),
            "n_after_dedup": int(n_after_dedup),
            "n_after_subsample": int(n_after_subsample),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "dim": 2,
        },
        "paper_task_statement": "(x, y) -> elevation",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _build_download_plan(args) -> tuple[list[Path], dict]:
    source_description: dict = {}

    if args.skip_download:
        files = _find_lidar_files(args.download_dir, max_files=args.max_files)
        source_description = {
            "download_mode": "skip_download_local_files",
            "download_dir": str(args.download_dir),
            "n_local_files_selected": len(files),
        }
        return files, source_description

    urls: list[str] = []
    tile_meta: dict = {}
    discovered: dict = {}

    if args.tile_index_path or args.tile_index_url:
        if args.tile_index_url:
            tile_index_path = _maybe_download_tile_index(
                args.tile_index_url,
                args.tile_index_dir,
                timeout=args.request_timeout,
                retries=args.max_retries,
            )
        else:
            tile_index_path = Path(args.tile_index_path)
        urls, tile_meta = _urls_from_tile_index(
            tile_index_path,
            bbox_lonlat=_parse_bbox(args.bbox),
            tile_url_column=args.tile_url_column,
            base_url=args.tile_base_url,
        )
    elif args.tile_url_list:
        urls = _read_tile_url_list(args.tile_url_list, timeout=args.request_timeout, retries=args.max_retries)
    elif args.download_url:
        urls = _normalize_download_urls(args.download_url, base_url=args.tile_base_url)
    elif args.dataset_page:
        tile_index_links, lidar_links = _extract_links_from_dataset_page(
            args.dataset_page,
            timeout=args.request_timeout,
            retries=args.max_retries,
        )
        discovered = {"tile_index_links": tile_index_links, "direct_lidar_links": lidar_links}
        print(f"dataset page discovery: {len(tile_index_links)} tile-index candidate(s), {len(lidar_links)} LAS/LAZ link(s)")
        if tile_index_links:
            tile_index_path = _maybe_download_tile_index(
                tile_index_links[0],
                args.tile_index_dir,
                timeout=args.request_timeout,
                retries=args.max_retries,
            )
            urls, tile_meta = _urls_from_tile_index(
                tile_index_path,
                bbox_lonlat=_parse_bbox(args.bbox),
                tile_url_column=args.tile_url_column,
                base_url=args.tile_base_url,
            )
        elif lidar_links:
            urls = lidar_links
        else:
            raise RuntimeError(
                "Could not discover tile index or LAS/LAZ links from the dataset page. "
                "Open the OpenTopography dataset page, download/copy the tile-index URL, then pass --tile-index-url."
            )
    else:
        raise ValueError(
            "No download source was specified. Use one of: --dataset-page, --tile-index-url, "
            "--tile-index-path, --tile-url-list, --download-url, or --skip-download."
        )

    urls = _normalize_download_urls(urls, base_url=args.tile_base_url)
    if args.max_files is not None and int(args.max_files) > 0:
        urls = urls[: int(args.max_files)]
    print(f"selected {len(urls)} LAS/LAZ URL(s)")

    paths, failures = _download_lidar_urls(
        urls,
        args.download_dir,
        max_files=None,
        timeout=args.request_timeout,
        retries=args.max_retries,
        continue_on_error=bool(args.continue_on_error),
    )
    source_description = {
        "download_mode": "opentopography_auto",
        "dataset_page": args.dataset_page,
        "tile_index_url": args.tile_index_url,
        "tile_index_path": str(args.tile_index_path) if args.tile_index_path else None,
        "tile_url_list": str(args.tile_url_list) if args.tile_url_list else None,
        "tile_base_url": args.tile_base_url,
        "bbox_lonlat": _parse_bbox(args.bbox),
        "n_urls_selected": len(urls),
        "n_files_downloaded_or_existing": len(paths),
        "download_failures": failures,
        "tile_index_metadata": tile_meta,
        "dataset_page_discovery": discovered,
    }
    return paths, source_description


def main() -> None:
    default_raw_dir, default_npz, default_json, default_tile_index_dir = _default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Download and preprocess OpenTopography LAS/LAZ point-cloud tiles into a "
            "2D elevation regression dataset for EFGP/KRR."
        )
    )
    src = parser.add_argument_group("download sources")
    src.add_argument("--dataset-page", type=str, default=None, help="OpenTopography dataset landing page URL. Best-effort scraper.")
    src.add_argument("--tile-index-url", type=str, default=None, help="URL to OpenTopography tile-index ZIP/GPKG/GeoJSON/Shapefile.")
    src.add_argument("--tile-index-path", type=Path, default=None, help="Local tile-index ZIP/GPKG/GeoJSON/Shapefile path.")
    src.add_argument("--tile-url-list", type=str, default=None, help="Local path or URL to a text file containing LAS/LAZ URLs.")
    src.add_argument(
        "--download-url",
        action="append",
        default=[],
        help="Direct LAS/LAZ URL. Can be repeated, or comma/space separated.",
    )
    src.add_argument("--skip-download", action="store_true", help="Skip download and use existing .las/.laz files in --download-dir.")
    src.add_argument("--tile-base-url", type=str, default=None, help="Base URL for relative LAS/LAZ filenames found in a tile index.")
    src.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Optional lon/lat AOI for tile-index selection: 'lon_min,lat_min,lon_max,lat_max' in EPSG:4326.",
    )
    src.add_argument("--tile-url-column", type=str, default=None, help="Explicit tile-index column containing LAS/LAZ URLs or filenames.")

    io = parser.add_argument_group("paths")
    io.add_argument("--download-dir", type=Path, default=default_raw_dir)
    io.add_argument("--tile-index-dir", type=Path, default=default_tile_index_dir)
    io.add_argument("--output-npz", type=Path, default=default_npz)
    io.add_argument("--output-json", type=Path, default=default_json)
    io.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)

    filt = parser.add_argument_group("preprocessing filters")
    filt.add_argument(
        "--classification-keep",
        type=str,
        default="2",
        help="Comma-separated LAS classification values to keep. Default '2' = ground. Empty string keeps all points.",
    )
    filt.add_argument("--target-epsg", type=int, default=None, help="Optional EPSG code to reproject x/y using LAS CRS metadata.")
    filt.add_argument("--max-files", type=int, default=8, help="Max LAS/LAZ files to use; <=0 means all discovered files.")
    filt.add_argument("--max-total-points", type=int, default=1_000_000, help="Random subsample cap after filters; <=0 means no cap.")
    filt.add_argument(
        "--bbox-xy",
        type=str,
        default=None,
        help="Optional post-read coordinate bbox in LAS/native or target EPSG: 'xmin,ymin,xmax,ymax'.",
    )
    filt.add_argument("--drop-duplicate-xy", action="store_true", help="Drop exact duplicate x/y coordinates, keeping first z.")

    split = parser.add_argument_group("split")
    split.add_argument("--test-size", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=0)

    net = parser.add_argument_group("network")
    net.add_argument("--request-timeout", type=int, default=240)
    net.add_argument("--max-retries", type=int, default=3)
    net.add_argument("--continue-on-error", action="store_true", help="Skip failed tile downloads instead of aborting.")

    args = parser.parse_args()
    if args.max_files is not None and int(args.max_files) <= 0:
        args.max_files = None
    if args.max_total_points is not None and int(args.max_total_points) <= 0:
        args.max_total_points = None

    las_paths, source_description = _build_download_plan(args)
    metadata = preprocess_lidar_elevation_regression(
        las_paths,
        args.output_npz,
        args.output_json,
        dataset_name=args.dataset_name,
        source_description=source_description,
        test_size=args.test_size,
        seed=args.seed,
        classification_keep=_parse_int_list(args.classification_keep),
        target_epsg=args.target_epsg,
        max_total_points=args.max_total_points,
        bbox_xy=_parse_bbox(args.bbox_xy),
        drop_duplicate_xy=bool(args.drop_duplicate_xy),
    )
    print("saved npz:", args.output_npz)
    print("saved json:", args.output_json)
    print("summary:", json.dumps(metadata["shapes"], indent=2))


if __name__ == "__main__":
    main()
