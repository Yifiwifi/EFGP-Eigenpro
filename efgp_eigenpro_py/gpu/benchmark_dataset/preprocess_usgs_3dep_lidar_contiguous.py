from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np


DEFAULT_PROJECT_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "USGS_LPC_IL_Winnebago_2018/laz/"
)
DEFAULT_DATASET_NAME = "usgs_3dep_lidar_winnebago_2018"

# Fallback names are now a compact vertical strip only used if Rockyweb listing fails.
# Normal behavior should discover all filenames, then choose a compact 2D block.
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
    if not _as_bool_nonempty(s):
        return None
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out if out else None


def _parse_int_list_csv(s: str | None) -> list[int]:
    if s is None or str(s).strip() == "":
        raise ValueError("list must be a non-empty comma-separated string")
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip().replace("_", "")
        if not part:
            continue
        val = int(part)
        if val <= 0:
            raise ValueError(f"all values must be positive, got {val}")
        out.append(val)
    if not out:
        raise ValueError("empty list after parsing")
    return out


def _parse_pair_ints(s: str | None, *, name: str) -> tuple[int, int] | None:
    if not _as_bool_nonempty(s):
        return None
    parts = [p.strip().replace("_", "") for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must have format 'a,b', got {s!r}")
    return int(parts[0]), int(parts[1])


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
    if parts and parts[-1].lower() in {"laz", "las", "laz_all", "classified_laz"}:
        parts = parts[:-1]
    if parts:
        name = parts[-1]
    else:
        name = DEFAULT_DATASET_NAME
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return name or DEFAULT_DATASET_NAME


def _max_total_points_for_target_n_train(n_train: int, *, test_size: float) -> int:
    ts = float(test_size)
    if not (0.0 < ts < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    return int(math.ceil(int(n_train) / max(1e-12, 1.0 - ts)))


# -----------------------------------------------------------------------------
# Rockyweb discovery and download
# -----------------------------------------------------------------------------


def _fetch_text(url: str, *, timeout: int = 180, retries: int = 3, backoff_s: float = 2.0) -> str:
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


def _hrefs_from_index(url: str, *, timeout: int, retries: int) -> list[str]:
    html = _fetch_text(url, timeout=timeout, retries=retries)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    out: list[str] = []
    for href in hrefs:
        if href.startswith("?") or href.startswith("#"):
            continue
        if href in {"../", "/"}:
            continue
        out.append(urljoin(url, href))
    return out


def _is_lidar_file_url(url: str) -> bool:
    low = url.lower().split("?")[0]
    return low.endswith(".laz") or low.endswith(".las")


def _is_directory_url(url: str) -> bool:
    return url.endswith("/")


def _is_current_rockyweb_lidar_base(base_url: str) -> bool:
    parsed = urlparse(base_url)
    path = parsed.path.lower().rstrip("/")
    return (
        parsed.netloc.lower().endswith("rockyweb.usgs.gov")
        and path.endswith(("/laz", "/las", "/classified_laz"))
    )


def _canonical_lidar_url(base_url: str, candidate_url_or_name: str) -> str:
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
) -> list[str]:
    pat = re.compile(file_regex) if _as_bool_nonempty(file_regex) else None
    candidates: list[str] = []
    candidates.extend(re.findall(r"https?://[^\s'\"<>]+?\.(?:la[sz]|LA[ZS])", text))
    candidates.extend(re.findall(r'href=["\']([^"\']+\.(?:la[sz]|LA[ZS]))["\']', text, flags=re.IGNORECASE))
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
    return out


def _manifest_candidates(root_url: str) -> list[str]:
    base = root_url if root_url.endswith("/") else root_url + "/"
    return [urljoin(base, "0_file_download_links.txt"), urljoin(base, "file_download_links.txt")]


def discover_lidar_urls(
    root_url: str,
    *,
    recursive_depth: int = 2,
    file_regex: str | None = None,
    request_timeout: int = 180,
    max_retries: int = 3,
) -> list[str]:
    root_url = root_url.strip()
    if not root_url.endswith("/") and not _is_lidar_file_url(root_url) and not root_url.lower().endswith(".txt"):
        root_url += "/"

    if _is_lidar_file_url(root_url):
        return [root_url]

    if root_url.lower().split("?")[0].endswith(".txt"):
        text = _fetch_text(root_url, timeout=request_timeout, retries=max_retries)
        base = root_url.rsplit("/", 1)[0] + "/"
        return _extract_lidar_urls_from_text(base, text, file_regex=file_regex)

    if root_url.rstrip("/").endswith("/Projects"):
        raise ValueError(
            "Do not pass the global Projects/ root. Choose one project directory or its laz/ subdirectory. "
            f"Recommended default: {DEFAULT_PROJECT_URL}"
        )

    for manifest_url in _manifest_candidates(root_url):
        try:
            text = _fetch_text(manifest_url, timeout=request_timeout, retries=max_retries)
            urls = _extract_lidar_urls_from_text(root_url, text, file_regex=file_regex)
            if urls:
                print(f"discovered {len(urls)} LAS/LAZ URL(s) from manifest: {manifest_url}")
                return urls
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to read manifest {manifest_url}: {exc}", file=sys.stderr)

    found: list[str] = []
    seen_dirs: set[str] = set()
    pat = re.compile(file_regex) if _as_bool_nonempty(file_regex) else None

    def visit(url: str, depth: int) -> None:
        if url in seen_dirs:
            return
        seen_dirs.add(url)
        try:
            hrefs = _hrefs_from_index(url, timeout=request_timeout, retries=max_retries)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to list {url}: {exc}", file=sys.stderr)
            return
        files = sorted([h for h in hrefs if _is_lidar_file_url(h)])
        for f in files:
            name = Path(urlparse(f).path).name
            if pat is not None and not pat.search(name):
                continue
            found.append(f)
        if depth <= 0:
            return
        dirs = sorted([h for h in hrefs if _is_directory_url(h)])
        preferred = [d for d in dirs if Path(urlparse(d).path.rstrip("/")).name.lower() in {"laz", "las"}]
        rest = [d for d in dirs if d not in preferred]
        for d in preferred + rest:
            visit(d, depth - 1)

    visit(root_url, int(recursive_depth))
    out: list[str] = []
    seen: set[str] = set()
    for u in found:
        if u not in seen:
            out.append(u)
            seen.add(u)

    if not out and root_url.rstrip("/") == DEFAULT_PROJECT_URL.rstrip("/"):
        fallback = [urljoin(DEFAULT_PROJECT_URL, name) for name in DEFAULT_FALLBACK_LAZ_NAMES]
        if file_regex:
            pat2 = re.compile(file_regex)
            fallback = [u for u in fallback if pat2.search(Path(urlparse(u).path).name)]
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
        _download_one_url(url, out, timeout=timeout, retries=retries)
        paths.append(out)
    return paths


# -----------------------------------------------------------------------------
# Continuous tile selection
# -----------------------------------------------------------------------------


def _tile_grid_key_from_name(name: str) -> tuple[int, int] | None:
    """Extract grid coordinates from common USGS LPC filenames.

    Example: USGS_LPC_IL_Winnebago_2018_2503_2073.laz -> (2503, 2073).
    The first number is treated as a column-like coordinate, the second as a row-like coordinate.
    Selection uses rank adjacency, so gaps such as 2049,2051,2053 still count as adjacent.
    """
    stem = Path(name).stem
    m = re.search(r"_(\d{3,})_(\d{3,})$", stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    nums = re.findall(r"\d{3,}", stem)
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])
    return None


def _tile_grid_key_from_url_or_path(u: str | Path) -> tuple[int, int] | None:
    if isinstance(u, Path):
        name = u.name
    else:
        name = Path(urlparse(str(u)).path).name
    return _tile_grid_key_from_name(name)


def _candidate_block_shapes(n: int) -> list[tuple[int, int]]:
    if n <= 0:
        return [(10**9, 10**9)]
    shapes: list[tuple[int, int]] = []
    max_side = max(1, int(math.ceil(math.sqrt(n))) + n)
    for w in range(1, max_side + 1):
        h = int(math.ceil(n / w))
        shapes.append((w, h))
        shapes.append((h, w))
    # Prefer small area and near-square compact shapes. Keep unique ordering.
    seen: set[tuple[int, int]] = set()
    uniq: list[tuple[int, int]] = []
    for wh in sorted(shapes, key=lambda p: ((p[0] * p[1]) - n, abs(math.log((p[0] + 1e-9) / (p[1] + 1e-9))), p[0] + p[1])):
        if wh not in seen:
            uniq.append(wh)
            seen.add(wh)
    return uniq[:80]


def _choose_contiguous_block(
    items: Sequence[str | Path],
    *,
    max_items: int | None,
    anchor: tuple[int, int] | None = None,
    block_shape: tuple[int, int] | None = None,
) -> tuple[list[str | Path], dict]:
    n_target = len(items) if max_items is None else min(int(max_items), len(items))
    if n_target <= 0:
        return list(items), {"tile_selection": "all", "reason": "max_items is none or all"}

    keyed: list[tuple[str | Path, tuple[int, int]]] = []
    unkeyed: list[str | Path] = []
    for it in items:
        key = _tile_grid_key_from_url_or_path(it)
        if key is None:
            unkeyed.append(it)
        else:
            keyed.append((it, key))

    if len(keyed) < n_target:
        selected = list(items)[:n_target]
        return selected, {
            "tile_selection": "first_fallback",
            "reason": f"only {len(keyed)} keyed tiles but need {n_target}",
            "n_keyed": len(keyed),
            "n_unkeyed": len(unkeyed),
        }

    x_vals = sorted({k[0] for _, k in keyed})
    y_vals = sorted({k[1] for _, k in keyed})
    x_rank = {v: i for i, v in enumerate(x_vals)}
    y_rank = {v: i for i, v in enumerate(y_vals)}
    rank_to_item: dict[tuple[int, int], tuple[str | Path, tuple[int, int]]] = {}
    for it, key in keyed:
        rank_to_item[(x_rank[key[0]], y_rank[key[1]])] = (it, key)

    if anchor is not None:
        # Snap anchor to nearest available rank coordinate in original tile space.
        ax = min(x_vals, key=lambda v: abs(v - anchor[0]))
        ay = min(y_vals, key=lambda v: abs(v - anchor[1]))
        anchor_rank = (x_rank[ax], y_rank[ay])
    else:
        anchor_rank = ((len(x_vals) - 1) / 2.0, (len(y_vals) - 1) / 2.0)

    shape_candidates = [block_shape] if block_shape is not None else _candidate_block_shapes(n_target)
    best: tuple[tuple, tuple[int, int], tuple[int, int], list[tuple[str | Path, tuple[int, int], tuple[int, int]]]] | None = None

    for w, h in shape_candidates:
        if w <= 0 or h <= 0:
            continue
        if w > len(x_vals) or h > len(y_vals):
            continue
        for x0 in range(0, len(x_vals) - w + 1):
            for y0 in range(0, len(y_vals) - h + 1):
                inside: list[tuple[str | Path, tuple[int, int], tuple[int, int]]] = []
                for rx in range(x0, x0 + w):
                    for ry in range(y0, y0 + h):
                        pair = rank_to_item.get((rx, ry))
                        if pair is not None:
                            it, key = pair
                            inside.append((it, key, (rx, ry)))
                count = len(inside)
                if count <= 0:
                    continue
                cx = x0 + (w - 1) / 2.0
                cy = y0 + (h - 1) / 2.0
                dist_anchor = (cx - anchor_rank[0]) ** 2 + (cy - anchor_rank[1]) ** 2
                fill = count / float(w * h)
                # Max count first; prefer enough tiles, filled rectangles, compact shapes, near anchor/center.
                enough = 1 if count >= n_target else 0
                score = (enough, min(count, n_target), fill, -((w * h) - min(count, n_target)), -abs(math.log((w + 1e-9) / (h + 1e-9))), -dist_anchor)
                if best is None or score > best[0]:
                    best = (score, (x0, y0), (w, h), inside)

    if best is None:
        selected = [it for it, _key in sorted(keyed, key=lambda p: (p[1][0], p[1][1]))[:n_target]]
        return selected, {"tile_selection": "sorted_key_fallback", "n_keyed": len(keyed)}

    _score, (x0, y0), (w, h), inside = best
    # If block has more tiles than needed, choose a compact central subset to avoid holes near the center.
    center = (x0 + (w - 1) / 2.0, y0 + (h - 1) / 2.0)
    inside_sorted = sorted(
        inside,
        key=lambda t: ((t[2][0] - center[0]) ** 2 + (t[2][1] - center[1]) ** 2, t[1][0], t[1][1]),
    )
    chosen = inside_sorted[:n_target]
    # For reproducible download/read order, sort chosen by grid coordinate.
    chosen_sorted = sorted(chosen, key=lambda t: (t[1][0], t[1][1]))
    selected = [t[0] for t in chosen_sorted]
    coords = [t[1] for t in chosen_sorted]

    meta = {
        "tile_selection": "contiguous_grid_block",
        "n_available_items": len(items),
        "n_keyed_items": len(keyed),
        "n_selected": len(selected),
        "target_count": n_target,
        "rank_block_origin": [int(x0), int(y0)],
        "rank_block_shape": [int(w), int(h)],
        "tile_coord_min": [int(min(c[0] for c in coords)), int(min(c[1] for c in coords))],
        "tile_coord_max": [int(max(c[0] for c in coords)), int(max(c[1] for c in coords))],
        "tile_coords": [[int(a), int(b)] for a, b in coords],
        "tile_names": [Path(urlparse(str(s)).path).name if not isinstance(s, Path) else s.name for s in selected],
    }
    if anchor is not None:
        meta["requested_anchor"] = [int(anchor[0]), int(anchor[1])]
    return selected, meta


def select_lidar_items(
    items: Sequence[str | Path],
    *,
    max_items: int | None,
    selection_mode: str,
    seed: int,
    anchor: tuple[int, int] | None = None,
    block_shape: tuple[int, int] | None = None,
) -> tuple[list[str | Path], dict]:
    items = list(items)
    if max_items is not None:
        max_items = min(int(max_items), len(items))
    mode = str(selection_mode).lower().strip()
    if mode == "all" or max_items is None:
        selected = sorted(items, key=lambda u: str(u))
        return selected, {"tile_selection": "all", "n_selected": len(selected)}
    if mode == "first":
        selected = sorted(items, key=lambda u: str(u))[:max_items]
        return selected, {"tile_selection": "first_sorted", "n_selected": len(selected)}
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(items), size=max_items, replace=False)
        selected = [items[i] for i in sorted(idx)]
        return selected, {"tile_selection": "random", "n_selected": len(selected), "seed": int(seed)}
    if mode == "contiguous":
        return _choose_contiguous_block(items, max_items=max_items, anchor=anchor, block_shape=block_shape)
    raise ValueError(f"unknown selection_mode={selection_mode!r}")


# -----------------------------------------------------------------------------
# LAS/LAZ preprocessing
# -----------------------------------------------------------------------------


def _require_laspy():
    try:
        import laspy  # type: ignore
    except ImportError as exc:
        raise ImportError("This script needs laspy and a LAZ backend. Install with: pip install laspy lazrs") from exc
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
            file_summaries.append({"path": str(path), "n_file": n_file, "n_kept_before_sample": 0, "n_used": 0})
            continue
        if max_points_per_file is not None and max_points_per_file > 0 and idx.size > max_points_per_file:
            idx = rng.choice(idx, size=int(max_points_per_file), replace=False)
        x = np.asarray(las.x[idx], dtype=np.float64)
        y = np.asarray(las.y[idx], dtype=np.float64)
        z = np.asarray(las.z[idx], dtype=np.float64)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        file_summaries.append(
            {
                "path": str(path),
                "tile_key": _tile_grid_key_from_name(path.name),
                "n_file": n_file,
                "n_kept_before_sample": n_kept_before_sample,
                "n_used": int(idx.size),
                "x_minmax": [float(np.min(x)), float(np.max(x))],
                "y_minmax": [float(np.min(y)), float(np.max(y))],
                "z_minmax": [float(np.min(z)), float(np.max(z))],
            }
        )
    if not xs:
        raise ValueError("no points were loaded after filtering")

    x_raw = np.column_stack([np.concatenate(xs), np.concatenate(ys)]).astype(np.float64, copy=False)
    y_raw = np.concatenate(zs).astype(np.float64, copy=False)
    metadata = {"files": file_summaries, "n_files": len(paths), "n_loaded_points": int(x_raw.shape[0])}
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
    selection_metadata: dict | None = None,
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

    raw_bbox = {
        "x_min": float(np.min(x_raw[:, 0])),
        "x_max": float(np.max(x_raw[:, 0])),
        "y_min": float(np.min(x_raw[:, 1])),
        "y_max": float(np.max(x_raw[:, 1])),
        "z_min": float(np.min(y_raw)),
        "z_max": float(np.max(y_raw)),
    }

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
        "split": {"method": "random", "test_size": float(test_size), "seed": int(seed)},
        "x_transform": {"method": "shift_and_shared_scale", "x_min_train": [float(v) for v in x_min], "shared_scale": float(scale)},
        "y_transform": {"method": "train_standardization", "mean": y_mean, "std": y_std},
        "raw_bbox": raw_bbox,
        "tile_selection_metadata": selection_metadata or {},
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
# Optional preview plot
# -----------------------------------------------------------------------------


def make_preview_png(npz_path: Path, png_path: Path, *, sample_train: int = 200_000, sample_test: int = 100_000, bins: int = 512, seed: int = 0) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError("preview plotting requires matplotlib: pip install matplotlib") from exc

    data = np.load(npz_path)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    rng = np.random.default_rng(int(seed))

    def subsample(x: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0 or x.shape[0] <= n:
            return x, y
        idx = rng.choice(x.shape[0], size=int(n), replace=False)
        return x[idx], y[idx]

    xt, yt = subsample(x_train, y_train, int(sample_train))
    xv, yv = subsample(x_test, y_test, int(sample_test))

    def raster_mean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bx = np.clip((x[:, 0] * (bins - 1)).astype(np.int64), 0, bins - 1)
        by = np.clip((x[:, 1] * (bins - 1)).astype(np.int64), 0, bins - 1)
        acc = np.zeros((bins, bins), dtype=np.float64)
        cnt = np.zeros((bins, bins), dtype=np.float64)
        np.add.at(acc, (by, bx), y)
        np.add.at(cnt, (by, bx), 1.0)
        out = np.full((bins, bins), np.nan, dtype=np.float64)
        mask = cnt > 0
        out[mask] = acc[mask] / cnt[mask]
        return out

    def raster_count(x: np.ndarray) -> np.ndarray:
        bx = np.clip((x[:, 0] * (bins - 1)).astype(np.int64), 0, bins - 1)
        by = np.clip((x[:, 1] * (bins - 1)).astype(np.int64), 0, bins - 1)
        cnt = np.zeros((bins, bins), dtype=np.float64)
        np.add.at(cnt, (by, bx), 1.0)
        cnt[cnt == 0] = np.nan
        return cnt

    cov = raster_count(np.vstack([xt, xv]))
    elev_train = raster_mean(xt, yt)
    elev_test = raster_mean(xv, yv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(np.log10(cov), origin="lower", extent=(0, 1, 0, 1))
    axes[0].set_title("(a) point density")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="log10 count")

    im1 = axes[1].imshow(elev_train, origin="lower", extent=(0, 1, 0, 1))
    axes[1].set_title("(b) train elevation")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="standardized z")

    im2 = axes[2].imshow(elev_test, origin="lower", extent=(0, 1, 0, 1))
    axes[2].set_title("(c) held-out elevation")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="standardized z")

    fig.suptitle(f"USGS LiDAR elevation regression preview: {npz_path.name}")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _resolve_local_las_paths(download_dir: Path) -> list[Path]:
    return sorted([p for p in download_dir.iterdir() if p.suffix.lower() in {".las", ".laz"}])


def _default_dataset_stem_prefix(project_url: str) -> str:
    dataset_name = _safe_dataset_name_from_url(project_url)
    return f"{dataset_name}_ground_elevation_regression"


def main() -> None:
    inferred_name = _safe_dataset_name_from_url(DEFAULT_PROJECT_URL)
    default_download_dir, default_npz, default_json = _default_paths(inferred_name)
    default_n_train_list = ",".join(str(v) for v in DEFAULT_N_TRAIN_LIST)

    parser = argparse.ArgumentParser(
        description=(
            "Download a spatially contiguous block of USGS Rockyweb 3DEP LAS/LAZ tiles and preprocess "
            "them into one or several 2D elevation regression NPZ datasets."
        )
    )
    parser.add_argument("--project-url", type=str, default=DEFAULT_PROJECT_URL)
    parser.add_argument("--download-dir", type=Path, default=default_download_dir)
    parser.add_argument("--output-npz", type=Path, default=default_npz)
    parser.add_argument("--output-json", type=Path, default=default_json)
    parser.add_argument("--output-dir", type=Path, default=default_npz.parent, help="Used only when --n-train-list is provided.")
    parser.add_argument("--dataset-stem-prefix", type=str, default=None)
    parser.add_argument("--n-train-list", type=str, default=None, help=f"Optional comma-separated sweep list. Example default set: {default_n_train_list}")
    parser.add_argument("--size-list", type=str, default=None, help="Alias for --n-train-list, interpreted as target N_train values.")

    parser.add_argument("--max-files", type=int, default=0, help="Number of tiles to select. 0 means auto: 16 for single run, 64 for sweep.")
    parser.add_argument("--file-regex", type=str, default=None)
    parser.add_argument("--recursive-depth", type=int, default=2)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")

    parser.add_argument("--tile-selection", type=str, default="contiguous", choices=["contiguous", "first", "random", "all"])
    parser.add_argument("--tile-anchor", type=str, default=None, help="Optional tile-coordinate anchor 'col,row', e.g. 2503,2057.")
    parser.add_argument("--tile-block-shape", type=str, default=None, help="Optional block shape 'cols,rows', e.g. 8,8. Usually leave empty.")

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classification-keep", type=str, default="2")
    parser.add_argument("--max-points-per-file", type=int, default=0)
    parser.add_argument("--max-total-points", type=int, default=10_000_000, help="Single-run cap. 0 means no cap. Ignored in sweep mode.")
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--make-preview", action="store_true", help="Save preview PNG(s) next to output JSON/NPZ.")
    parser.add_argument("--preview-bins", type=int, default=512)
    parser.add_argument("--preview-train", type=int, default=200_000)
    parser.add_argument("--preview-test", type=int, default=100_000)

    args = parser.parse_args()

    project_url = str(args.project_url).strip()
    dataset_name = _safe_dataset_name_from_url(project_url)
    n_train_list_raw = args.size_list if args.size_list is not None else args.n_train_list
    n_train_list = _parse_int_list_csv(n_train_list_raw) if _as_bool_nonempty(n_train_list_raw) else None
    sweep_mode = n_train_list is not None

    max_files_auto = 64 if sweep_mode else 16
    max_files = None if int(args.max_files) == 0 and args.tile_selection == "all" else (max_files_auto if int(args.max_files) == 0 else int(args.max_files))
    max_points_per_file = None if int(args.max_points_per_file) == 0 else int(args.max_points_per_file)
    max_total_points = None if int(args.max_total_points) == 0 else int(args.max_total_points)
    classification_keep = _parse_int_list(args.classification_keep)
    anchor = _parse_pair_ints(args.tile_anchor, name="--tile-anchor")
    block_shape = _parse_pair_ints(args.tile_block_shape, name="--tile-block-shape")

    if args.skip_download:
        all_local = _resolve_local_las_paths(args.download_dir)
        selected_local_any, selection_meta = select_lidar_items(
            all_local,
            max_items=max_files,
            selection_mode=args.tile_selection,
            seed=int(args.seed),
            anchor=anchor,
            block_shape=block_shape,
        )
        urls: list[str] = []
        las_paths = [Path(p) for p in selected_local_any]
    else:
        all_urls = discover_lidar_urls(
            project_url,
            recursive_depth=int(args.recursive_depth),
            file_regex=args.file_regex,
            request_timeout=int(args.request_timeout),
            max_retries=int(args.max_retries),
        )
        if not all_urls:
            raise RuntimeError(f"No LAS/LAZ files discovered from {project_url}.")
        print(f"discovered {len(all_urls)} total LAS/LAZ URL(s)")
        selected_urls_any, selection_meta = select_lidar_items(
            all_urls,
            max_items=max_files,
            selection_mode=args.tile_selection,
            seed=int(args.seed),
            anchor=anchor,
            block_shape=block_shape,
        )
        urls = [str(u) for u in selected_urls_any]
        print("tile selection:", json.dumps({k: v for k, v in selection_meta.items() if k != "tile_names"}, indent=2))
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
    for p in las_paths[:12]:
        print("  ", p)
    if len(las_paths) > 12:
        print(f"  ... and {len(las_paths) - 12} more")

    if args.download_only:
        print("download-only mode; finished")
        return

    if sweep_mode:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_stem_prefix = (
            str(args.dataset_stem_prefix).strip()
            if args.dataset_stem_prefix is not None and str(args.dataset_stem_prefix).strip()
            else _default_dataset_stem_prefix(project_url)
        )
        summary_rows: list[dict] = []
        failures: list[dict] = []
        for n_train_target in n_train_list or []:
            cap = _max_total_points_for_target_n_train(int(n_train_target), test_size=float(args.test_size))
            dataset_stem = f"{dataset_stem_prefix}_ntrain{int(n_train_target)}"
            output_npz = output_dir / f"{dataset_stem}.npz"
            output_json = output_dir / f"{dataset_stem}.json"
            print("=" * 100)
            print(f"generating {dataset_stem}: target_n_train={n_train_target}, max_total_points={cap}")
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
                    max_total_points=int(cap),
                    selection_metadata=selection_meta,
                )
                if args.make_preview:
                    make_preview_png(
                        output_npz,
                        output_json.with_suffix(".preview.png"),
                        sample_train=int(args.preview_train),
                        sample_test=int(args.preview_test),
                        bins=int(args.preview_bins),
                        seed=int(args.seed),
                    )
                row = {
                    "dataset_stem": dataset_stem,
                    "n_train_target": int(n_train_target),
                    "max_total_points_used": int(cap),
                    "output_npz": str(output_npz),
                    "output_json": str(output_json),
                    **metadata.get("shapes", {}),
                }
                summary_rows.append(row)
                print("summary:", json.dumps(row, indent=2))
            except Exception as exc:  # noqa: BLE001
                err = {
                    "dataset_stem": dataset_stem,
                    "n_train_target": int(n_train_target),
                    "max_total_points_used": int(cap),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(err)
                print("[ERROR]", json.dumps(err, indent=2))
                if not bool(args.continue_on_error):
                    raise
        batch_summary = {
            "project_url": project_url,
            "dataset_stem_prefix": dataset_stem_prefix,
            "download_dir": str(args.download_dir),
            "output_dir": str(output_dir),
            "n_train_list": [int(v) for v in n_train_list or []],
            "test_size": float(args.test_size),
            "max_files": None if max_files is None else int(max_files),
            "tile_selection_metadata": selection_meta,
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
        return

    metadata = preprocess_lidar_elevation(
        las_paths,
        args.output_npz,
        args.output_json,
        dataset_name=dataset_name,
        source_url=project_url,
        downloaded_urls=urls,
        test_size=float(args.test_size),
        seed=int(args.seed),
        classification_keep=classification_keep,
        max_points_per_file=max_points_per_file,
        max_total_points=max_total_points,
        selection_metadata=selection_meta,
    )
    if args.make_preview:
        make_preview_png(
            args.output_npz,
            args.output_json.with_suffix(".preview.png"),
            sample_train=int(args.preview_train),
            sample_test=int(args.preview_test),
            bins=int(args.preview_bins),
            seed=int(args.seed),
        )
    print("saved npz:", args.output_npz)
    print("saved json:", args.output_json)
    print("summary:", json.dumps(metadata["shapes"], indent=2))


if __name__ == "__main__":
    main()
