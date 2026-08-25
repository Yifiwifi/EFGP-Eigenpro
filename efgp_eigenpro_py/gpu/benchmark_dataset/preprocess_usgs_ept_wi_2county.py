"""Build a reproducible USGS 3DEP EPT elevation-regression benchmark.

The benchmark is deliberately tied to one EPT project and one metric AOI.  EPT
stores a progressive point hierarchy: coarse levels contain a sparse spatial
summary and deeper levels add detail.  We preserve that property in the sample
order.  All eligible points from a shallower level precede every point from a
deeper level; points within one level are ordered by an independent SplitMix64
permutation.  Consequently every requested data set is an exact prefix of the
largest one, without scanning the full-density point cloud for a 10M pilot.

Only standard project dependencies are used: urllib, laspy, numpy, and pyproj.
No PDAL installation is required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Mapping, Sequence

import laspy
import numpy as np
from pyproj import Transformer


DEFAULT_EPT_JSON = (
    "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/"
    "WI_2County_2_B23/ept.json"
)
DEFAULT_STEM_PREFIX = "USGS_EPT_WI_2County_2_B23_ground_elevation"
DEFAULT_SOURCE_PROJECT = "WI_2County_2_B23"
DEFAULT_NOAA_INPORT_RECORD = "https://www.fisheries.noaa.gov/inport/item/72914"
DEFAULT_USGS_PROJECT_REPORT = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/metadata/"
    "WI_2County_B23/USGS_WI_2County_B23_Project_Report.pdf"
)
DEFAULT_USGS_WORK_UNIT_REPORT = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/metadata/"
    "WI_2County_B23/WI_2County_2_B23/reports/"
    "WI_2County_WU300412_ProjectReport.pdf"
)
DEFAULT_OFFICIAL_MEAN_GROUND_DENSITY = 8.54
DEFAULT_OFFICIAL_WORK_UNIT_AREA_KM2 = 718.0 * 2.589988110336
DEFAULT_N_TRAIN_LIST = (100_000, 1_000_000, 10_000_000)
DEFAULT_AOI_CRS = "EPSG:6344"  # NAD83(2011) / UTM zone 15N
DEFAULT_SOURCE_CRS = "EPSG:3857"
DEFAULT_AOI_CENTER = (595_000.0, 4_907_000.0)
DEFAULT_AOI_SIDE_M = 8_000.0
DEFAULT_CLASSIFICATION = 2
DEFAULT_TEST_MODULUS = 5
DEFAULT_SPLIT_SEED = np.uint64(0x243F6A8885A308D3)
DEFAULT_ORDER_SEED = np.uint64(0x13198A2E03707344)
DEFAULT_DOWNLOAD_WORKERS = 8
DEFAULT_CHUNK_POINTS = 1_000_000
UNCOMPRESSED_NPZ_THRESHOLD_ROWS = 1_000_000
LARGE_OUTPUT_WARNING_ROWS = 10_000_000
Y_CALIBRATION_PREFIX_ROWS = 1_000_000


SPOOL_DTYPE = np.dtype(
    [
        ("source_id", "<u8"),
        ("order_hash", "<u8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("depth", "u1"),
    ],
    align=False,
)


@dataclass(frozen=True)
class FixedAoi:
    """Square AOI expressed in a declared projected CRS."""

    center_x: float = DEFAULT_AOI_CENTER[0]
    center_y: float = DEFAULT_AOI_CENTER[1]
    side_m: float = DEFAULT_AOI_SIDE_M
    crs: str = DEFAULT_AOI_CRS

    def __post_init__(self) -> None:
        if not np.isfinite([self.center_x, self.center_y, self.side_m]).all():
            raise ValueError("AOI center and side must be finite")
        if self.side_m <= 0.0:
            raise ValueError("AOI side must be positive")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        half = 0.5 * float(self.side_m)
        return (
            float(self.center_x) - half,
            float(self.center_y) - half,
            float(self.center_x) + half,
            float(self.center_y) + half,
        )


@dataclass(frozen=True)
class EptNode:
    key: str
    depth: int
    x_index: int
    y_index: int
    z_index: int
    hierarchy_points: int
    bounds: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class MaterializedNode:
    node: EptNode
    path: Path
    source_url: str
    sha256: str
    byte_count: int


@dataclass
class DepthSpool:
    depth: int
    train_path: Path
    test_path: Path
    train_count: int = 0
    test_count: int = 0


class SourceAccessor:
    """Uniform byte access for an EPT URL or a local fixture directory."""

    def __init__(self, ept_json: str | Path):
        raw = str(ept_json)
        parsed = urllib.parse.urlparse(raw)
        self.is_remote = parsed.scheme.lower() in {"http", "https"}
        if self.is_remote:
            if not raw.lower().split("?", 1)[0].endswith("/ept.json"):
                raise ValueError("remote EPT reference must end in ept.json")
            self.ept_reference = raw
            self.root_reference = raw.rsplit("/", 1)[0] + "/"
            self.local_root: Path | None = None
        else:
            ept_path = Path(ept_json).expanduser().resolve()
            if ept_path.is_dir():
                ept_path = ept_path / "ept.json"
            if not ept_path.is_file():
                raise FileNotFoundError(ept_path)
            self.ept_reference = str(ept_path)
            self.root_reference = str(ept_path.parent)
            self.local_root = ept_path.parent

    def describe(self, relative: str) -> str:
        relative = relative.replace("\\", "/")
        if self.is_remote:
            return urllib.parse.urljoin(self.root_reference, relative)
        assert self.local_root is not None
        return str((self.local_root / Path(relative)).resolve())

    def read_bytes(self, relative: str, *, timeout_seconds: float) -> bytes:
        relative = relative.replace("\\", "/")
        if self.is_remote:
            url = self.describe(relative)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "efgp-usgs-ept-benchmark/1.0"},
            )
            with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
                return response.read()
        assert self.local_root is not None
        return (self.local_root / Path(relative)).read_bytes()

    def local_path(self, relative: str) -> Path | None:
        if self.is_remote:
            return None
        assert self.local_root is not None
        return (self.local_root / Path(relative)).resolve()


def _parse_positive_sizes(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        values = [
            int(part.strip().replace("_", ""))
            for part in raw.split(",")
            if part.strip()
        ]
    else:
        values = [int(value) for value in raw]
    if not values or any(value <= 0 for value in values):
        raise ValueError("n-train-list must contain positive values")
    if len(set(values)) != len(values):
        raise ValueError("n-train-list contains duplicate values")
    return sorted(values)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _splitmix64(values: np.ndarray, seed: np.uint64) -> np.ndarray:
    """Vectorized SplitMix64; a bijection over uint64 for each fixed seed."""
    with np.errstate(over="ignore"):
        z = np.asarray(values, dtype=np.uint64) + np.uint64(seed)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _parse_node_key(key: str) -> tuple[int, int, int, int]:
    parts = key.split("-")
    if len(parts) != 4:
        raise ValueError(f"invalid EPT node key {key!r}")
    depth, x_index, y_index, z_index = (int(part) for part in parts)
    if depth < 0 or min(x_index, y_index, z_index) < 0:
        raise ValueError(f"negative component in EPT node key {key!r}")
    limit = 1 << depth
    if max(x_index, y_index, z_index) >= limit:
        raise ValueError(f"EPT node index exceeds depth grid in {key!r}")
    return depth, x_index, y_index, z_index


def _node_bounds(
    key: str, root_bounds: Sequence[float]
) -> tuple[float, float, float, float, float, float]:
    if len(root_bounds) != 6:
        raise ValueError("EPT root bounds must have six entries")
    depth, ix, iy, iz = _parse_node_key(key)
    subdivisions = float(1 << depth)
    mins = np.asarray(root_bounds[:3], dtype=np.float64)
    maxs = np.asarray(root_bounds[3:], dtype=np.float64)
    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        raise ValueError("EPT root bounds must be finite")
    widths = (maxs - mins) / subdivisions
    if np.any(widths <= 0.0):
        raise ValueError("EPT root bounds must be strictly increasing")
    indices = np.asarray([ix, iy, iz], dtype=np.float64)
    lower = mins + indices * widths
    upper = lower + widths
    return (
        float(lower[0]),
        float(lower[1]),
        float(lower[2]),
        float(upper[0]),
        float(upper[1]),
        float(upper[2]),
    )


def _xy_intersects(
    node_bounds: Sequence[float], source_aoi_bounds: Sequence[float]
) -> bool:
    nx0, ny0, _nz0, nx1, ny1, _nz1 = node_bounds
    ax0, ay0, ax1, ay1 = source_aoi_bounds
    return bool(nx1 >= ax0 and nx0 <= ax1 and ny1 >= ay0 and ny0 <= ay1)


def _source_aoi_bounds(aoi: FixedAoi, source_crs: str) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs(aoi.crs, source_crs, always_xy=True)
    xmin, ymin, xmax, ymax = aoi.bounds
    transformed = transformer.transform_bounds(
        xmin, ymin, xmax, ymax, densify_pts=21
    )
    if not np.isfinite(transformed).all():
        raise ValueError("AOI-to-source CRS transform produced non-finite bounds")
    return tuple(float(value) for value in transformed)


def _load_ept_metadata(
    accessor: SourceAccessor, *, timeout_seconds: float
) -> tuple[dict, dict]:
    payload = accessor.read_bytes("ept.json", timeout_seconds=timeout_seconds)
    metadata = json.loads(payload.decode("utf-8"))
    required = {"bounds", "boundsConforming", "dataType", "hierarchyType", "points", "srs"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"ept.json is missing required fields: {missing}")
    if metadata["dataType"] != "laszip" or metadata["hierarchyType"] != "json":
        raise ValueError(
            "this preprocessor requires EPT dataType=laszip and hierarchyType=json"
        )
    authority = str(metadata.get("srs", {}).get("authority", "EPSG"))
    horizontal = str(metadata.get("srs", {}).get("horizontal", ""))
    source_crs = f"{authority}:{horizontal}"
    if source_crs.upper() != DEFAULT_SOURCE_CRS:
        raise ValueError(
            f"frozen benchmark expects {DEFAULT_SOURCE_CRS}, ept.json declares {source_crs}"
        )
    provenance = {
        "url": accessor.describe("ept.json"),
        "sha256": _sha256_bytes(payload),
        "byte_count": len(payload),
    }
    return metadata, provenance


def discover_intersecting_nodes(
    accessor: SourceAccessor,
    ept_metadata: Mapping,
    *,
    aoi: FixedAoi,
    timeout_seconds: float = 300.0,
    max_lod_depth: int | None = None,
) -> tuple[list[EptNode], list[dict], tuple[float, float, float, float]]:
    """Recursively read JSON hierarchy pages and retain AOI-intersecting nodes."""
    if max_lod_depth is not None and int(max_lod_depth) < 0:
        raise ValueError("max_lod_depth must be nonnegative or None")
    root_bounds = tuple(float(value) for value in ept_metadata["bounds"])
    source_aoi = _source_aoi_bounds(aoi, DEFAULT_SOURCE_CRS)
    queue = ["0-0-0-0"]
    seen_pages: set[str] = set()
    nodes_by_key: dict[str, EptNode] = {}
    page_provenance: list[dict] = []

    while queue:
        page_root = queue.pop(0)
        if page_root in seen_pages:
            continue
        seen_pages.add(page_root)
        relative = f"ept-hierarchy/{page_root}.json"
        payload = accessor.read_bytes(relative, timeout_seconds=timeout_seconds)
        page = json.loads(payload.decode("utf-8"))
        if not isinstance(page, dict):
            raise ValueError(f"hierarchy page {relative} is not a JSON object")
        page_provenance.append(
            {
                "root_key": page_root,
                "url": accessor.describe(relative),
                "sha256": _sha256_bytes(payload),
                "byte_count": len(payload),
                "entry_count": len(page),
            }
        )
        for key, raw_count in sorted(page.items(), key=lambda item: _parse_node_key(item[0])):
            depth, ix, iy, iz = _parse_node_key(key)
            if max_lod_depth is not None and depth > int(max_lod_depth):
                continue
            bounds = _node_bounds(key, root_bounds)
            if not _xy_intersects(bounds, source_aoi):
                continue
            count = int(raw_count)
            if count == -1:
                if max_lod_depth is not None and depth == int(max_lod_depth):
                    # EPT continuation sentinels still have a data node at this key.
                    # Its point count is recovered from the LAZ header during scan.
                    nodes_by_key[key] = EptNode(key, depth, ix, iy, iz, -1, bounds)
                else:
                    queue.append(key)
                continue
            if count < -1:
                raise ValueError(f"unsupported hierarchy count {count} for node {key}")
            if count == 0:
                continue
            node = EptNode(key, depth, ix, iy, iz, count, bounds)
            previous = nodes_by_key.get(key)
            if previous is not None and previous != node:
                raise ValueError(f"conflicting hierarchy entries for {key}")
            nodes_by_key[key] = node

    nodes = sorted(
        nodes_by_key.values(),
        key=lambda node: (node.depth, node.x_index, node.y_index, node.z_index),
    )
    if not nodes:
        raise ValueError("no positive-count EPT nodes intersect the fixed AOI")
    return nodes, page_provenance, source_aoi


def estimate_hierarchy_capacity(nodes: Sequence[EptNode]) -> dict:
    """Return the raw-node point upper bound before class and exact-AOI filtering."""
    by_depth: dict[int, dict[str, int | list[str]]] = {}
    for node in nodes:
        row = by_depth.setdefault(
            node.depth,
            {
                "node_count": 0,
                "known_point_node_count": 0,
                "unknown_point_node_count": 0,
                "raw_node_points_known": 0,
                "unknown_point_node_keys": [],
            },
        )
        row["node_count"] = int(row["node_count"]) + 1
        if node.hierarchy_points >= 0:
            row["known_point_node_count"] = int(row["known_point_node_count"]) + 1
            row["raw_node_points_known"] = int(row["raw_node_points_known"]) + int(
                node.hierarchy_points
            )
        else:
            row["unknown_point_node_count"] = int(row["unknown_point_node_count"]) + 1
            unknown_keys = row["unknown_point_node_keys"]
            assert isinstance(unknown_keys, list)
            unknown_keys.append(node.key)
    cumulative = 0
    cumulative_unknown = 0
    depth_rows = []
    for depth in sorted(by_depth):
        cumulative += int(by_depth[depth]["raw_node_points_known"])
        cumulative_unknown += int(by_depth[depth]["unknown_point_node_count"])
        depth_rows.append(
            {
                "depth": int(depth),
                **by_depth[depth],
                "cumulative_raw_node_points_known": int(cumulative),
                "cumulative_unknown_point_nodes": int(cumulative_unknown),
            }
        )
    return {
        "meaning": (
            "known total from AOI-intersecting hierarchy counts before exact filtering; "
            "terminal continuation sentinels are listed separately and obtain exact counts "
            "from LAZ headers during a build"
        ),
        "depths": depth_rows,
        "total_intersecting_node_points_known": int(cumulative),
        "total_intersecting_node_points_upper_bound": (
            int(cumulative) if cumulative_unknown == 0 else None
        ),
        "unknown_terminal_node_count": int(cumulative_unknown),
        "has_unknown_terminal_counts": bool(cumulative_unknown),
    }


def _download_remote_atomic(
    url: str, destination: Path, *, timeout_seconds: float
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "efgp-usgs-ept-benchmark/1.0"}
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            with temporary.open("wb") as target:
                shutil.copyfileobj(response, target, length=8 * 1024 * 1024)
        if temporary.stat().st_size <= 0:
            raise RuntimeError(f"downloaded empty EPT node: {url}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _materialize_one_node(
    accessor: SourceAccessor,
    node: EptNode,
    *,
    cache_dir: Path,
    timeout_seconds: float,
) -> MaterializedNode:
    relative = f"ept-data/{node.key}.laz"
    source_url = accessor.describe(relative)
    local = accessor.local_path(relative)
    if local is None:
        local = cache_dir / f"{node.key}.laz"
        if not local.is_file() or local.stat().st_size <= 0:
            _download_remote_atomic(source_url, local, timeout_seconds=timeout_seconds)
    elif not local.is_file():
        raise FileNotFoundError(local)
    return MaterializedNode(
        node=node,
        path=local,
        source_url=source_url,
        sha256=_sha256_file(local),
        byte_count=int(local.stat().st_size),
    )


def materialize_nodes(
    accessor: SourceAccessor,
    nodes: Sequence[EptNode],
    *,
    cache_dir: Path,
    timeout_seconds: float,
    max_download_workers: int,
) -> list[MaterializedNode]:
    """Download a depth's small LAZ nodes concurrently, then return stable order."""
    workers = int(max_download_workers)
    if workers <= 0:
        raise ValueError("max_download_workers must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[MaterializedNode] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _materialize_one_node,
                accessor,
                node,
                cache_dir=cache_dir,
                timeout_seconds=timeout_seconds,
            ): node
            for node in nodes
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return sorted(
        results,
        key=lambda item: (
            item.node.depth,
            item.node.x_index,
            item.node.y_index,
            item.node.z_index,
        ),
    )


def _append_records(stream: BinaryIO, records: np.ndarray) -> None:
    if records.dtype != SPOOL_DTYPE:
        records = np.asarray(records, dtype=SPOOL_DTYPE)
    stream.write(records.tobytes(order="C"))


def _scan_depth_to_spools(
    materialized: Sequence[MaterializedNode],
    *,
    node_ranks: Mapping[str, int],
    depth_spool: DepthSpool,
    aoi: FixedAoi,
    source_crs: str,
    chunk_points: int,
) -> tuple[dict, list[dict]]:
    if chunk_points <= 0:
        raise ValueError("chunk_points must be positive")
    xmin, ymin, xmax, ymax = aoi.bounds
    to_aoi = Transformer.from_crs(source_crs, aoi.crs, always_xy=True)
    aggregate = {
        "header_points": 0,
        "iterated_points": 0,
        "classification_2_points": 0,
        "exact_aoi_ground_points": 0,
        "classification_rejected": 0,
        "aoi_rejected_after_classification": 0,
        "train_bucket_points": 0,
        "test_bucket_points": 0,
    }
    node_rows: list[dict] = []
    depth_spool.train_path.parent.mkdir(parents=True, exist_ok=True)
    with depth_spool.train_path.open("wb") as train_stream, depth_spool.test_path.open(
        "wb"
    ) as test_stream:
        for item in materialized:
            rank = int(node_ranks[item.node.key])
            if rank < 0 or rank >= (1 << 32):
                raise ValueError("selected-node rank does not fit the uint64 source-ID scheme")
            row = {
                "key": item.node.key,
                "depth": int(item.node.depth),
                "hierarchy_points": int(item.node.hierarchy_points),
                "source_url": item.source_url,
                "local_cache_file": str(item.path),
                "sha256": item.sha256,
                "byte_count": item.byte_count,
                "header_points": 0,
                "iterated_points": 0,
                "classification_2_points": 0,
                "exact_aoi_ground_points": 0,
                "train_bucket_points": 0,
                "test_bucket_points": 0,
            }
            ordinal_offset = 0
            with laspy.open(item.path) as reader:
                header_points = int(reader.header.point_count)
                row["header_points"] = header_points
                aggregate["header_points"] += header_points
                for points in reader.chunk_iterator(int(chunk_points)):
                    n_chunk = len(points)
                    if ordinal_offset + n_chunk > (1 << 32):
                        raise ValueError(
                            f"node {item.node.key} has too many points for 32-bit ordinal IDs"
                        )
                    row["iterated_points"] += n_chunk
                    aggregate["iterated_points"] += n_chunk
                    classes = np.asarray(points.classification, dtype=np.uint8)
                    is_ground = classes == np.uint8(DEFAULT_CLASSIFICATION)
                    n_ground = int(np.count_nonzero(is_ground))
                    row["classification_2_points"] += n_ground
                    aggregate["classification_2_points"] += n_ground
                    aggregate["classification_rejected"] += n_chunk - n_ground
                    if n_ground:
                        ground_local = np.flatnonzero(is_ground)
                        # laspy 2.7 ScaledArrayView.__getitem__ mistakes a length-two
                        # integer index array for a two-axis tuple and attempts to
                        # index its scalar scale/offset.  Materialize the scaled 1D
                        # views before NumPy advanced indexing.  This preserves the
                        # exact laspy scaling semantics for every chunk size.
                        source_x = np.asarray(points.x, dtype=np.float64)[ground_local]
                        source_y = np.asarray(points.y, dtype=np.float64)[ground_local]
                        elevation = np.asarray(points.z, dtype=np.float64)[ground_local]
                        east, north = to_aoi.transform(source_x, source_y)
                        east = np.asarray(east, dtype=np.float64)
                        north = np.asarray(north, dtype=np.float64)
                        exact = (
                            np.isfinite(east)
                            & np.isfinite(north)
                            & np.isfinite(elevation)
                            & (east >= xmin)
                            & (east <= xmax)
                            & (north >= ymin)
                            & (north <= ymax)
                        )
                        accepted_local = ground_local[exact]
                        accepted = int(accepted_local.size)
                        row["exact_aoi_ground_points"] += accepted
                        aggregate["exact_aoi_ground_points"] += accepted
                        aggregate["aoi_rejected_after_classification"] += n_ground - accepted
                        if accepted:
                            ordinal = accepted_local.astype(np.uint64) + np.uint64(
                                ordinal_offset
                            )
                            source_id = (
                                np.uint64(rank) << np.uint64(32)
                            ) | ordinal
                            split_hash = _splitmix64(source_id, DEFAULT_SPLIT_SEED)
                            order_hash = _splitmix64(source_id, DEFAULT_ORDER_SEED)
                            records = np.empty(accepted, dtype=SPOOL_DTYPE)
                            records["source_id"] = source_id
                            records["order_hash"] = order_hash
                            records["x"] = np.asarray(
                                (east[exact] - xmin) / aoi.side_m, dtype=np.float32
                            )
                            records["y"] = np.asarray(
                                (north[exact] - ymin) / aoi.side_m, dtype=np.float32
                            )
                            records["z"] = np.asarray(elevation[exact], dtype=np.float32)
                            records["depth"] = np.uint8(item.node.depth)
                            is_test = (
                                split_hash % np.uint64(DEFAULT_TEST_MODULUS)
                                == np.uint64(0)
                            )
                            train_records = records[~is_test]
                            test_records = records[is_test]
                            _append_records(train_stream, train_records)
                            _append_records(test_stream, test_records)
                            n_train = int(train_records.size)
                            n_test = int(test_records.size)
                            row["train_bucket_points"] += n_train
                            row["test_bucket_points"] += n_test
                            aggregate["train_bucket_points"] += n_train
                            aggregate["test_bucket_points"] += n_test
                    ordinal_offset += n_chunk
            if row["iterated_points"] != row["header_points"]:
                raise RuntimeError(
                    f"LAS header/stream count mismatch for {item.node.key}: "
                    f"{row['header_points']} vs {row['iterated_points']}"
                )
            node_rows.append(row)
    depth_spool.train_count = int(aggregate["train_bucket_points"])
    depth_spool.test_count = int(aggregate["test_bucket_points"])
    return aggregate, node_rows


def _spool_count(path: Path) -> int:
    size = int(path.stat().st_size)
    if size % SPOOL_DTYPE.itemsize:
        raise ValueError(f"corrupt spool byte count: {path}")
    return size // SPOOL_DTYPE.itemsize


def _iter_spool_chunks(
    path: Path, *, chunk_records: int = 2_000_000
) -> Iterator[np.ndarray]:
    count = _spool_count(path)
    records = np.memmap(path, dtype=SPOOL_DTYPE, mode="r", shape=(count,))
    try:
        for start in range(0, count, int(chunk_records)):
            yield records[start : min(count, start + int(chunk_records))]
    finally:
        del records


def _kth_hash_threshold(path: Path, k: int) -> np.uint64:
    """Find the kth (one-based) uint64 hash using constant-memory radix passes."""
    count = _spool_count(path)
    k = int(k)
    if k <= 0 or k > count:
        raise ValueError(f"cannot select k={k} from spool with {count} records")
    rank = k - 1
    prefix = np.uint64(0)
    prefix_mask = np.uint64(0)
    for shift in range(56, -1, -8):
        histogram = np.zeros(256, dtype=np.int64)
        for chunk in _iter_spool_chunks(path):
            hashes = np.asarray(chunk["order_hash"], dtype=np.uint64)
            if int(prefix_mask) != 0:
                hashes = hashes[(hashes & prefix_mask) == prefix]
            if hashes.size:
                byte = np.asarray(
                    (hashes >> np.uint64(shift)) & np.uint64(0xFF), dtype=np.int64
                )
                histogram += np.bincount(byte, minlength=256)
        cumulative = np.cumsum(histogram)
        bucket = int(np.searchsorted(cumulative, rank + 1, side="left"))
        before = int(cumulative[bucket - 1]) if bucket else 0
        rank -= before
        byte_mask = np.uint64(0xFF) << np.uint64(shift)
        prefix = (prefix & ~byte_mask) | (np.uint64(bucket) << np.uint64(shift))
        prefix_mask |= byte_mask
    if rank != 0:
        # SplitMix64 is one-to-one over unique source IDs, so the terminal bucket
        # contains one row.  A nonzero rank signals duplicate source IDs.
        raise RuntimeError("duplicate order hashes detected in EPT spool")
    return prefix


def _write_hash_ordered_prefix(
    source_path: Path,
    count_to_take: int,
    destination: BinaryIO,
    *,
    temporary_dir: Path,
) -> int:
    """Append an exact hash-sorted prefix from one LOD depth to destination."""
    count = _spool_count(source_path)
    take = int(count_to_take)
    if take <= 0 or take > count:
        raise ValueError(f"cannot take {take} rows from {count}-row spool")
    candidate_path = source_path
    temporary_candidate: Path | None = None
    if take < count:
        threshold = _kth_hash_threshold(source_path, take)
        temporary_candidate = temporary_dir / (
            f"selected_{source_path.stem}_{take}.bin"
        )
        written = 0
        with temporary_candidate.open("wb") as selected_stream:
            for chunk in _iter_spool_chunks(source_path):
                chosen = chunk[chunk["order_hash"] <= threshold]
                _append_records(selected_stream, chosen)
                written += int(chosen.size)
        if written != take:
            raise RuntimeError(
                f"radix threshold selected {written} records, expected {take}"
            )
        candidate_path = temporary_candidate

    candidate_count = _spool_count(candidate_path)
    sorted_path = _external_lsd_radix_sort(
        candidate_path,
        temporary_dir=temporary_dir,
        tag=f"{source_path.stem}_{take}",
    )
    try:
        for chunk in _iter_spool_chunks(sorted_path):
            _append_records(destination, chunk)
        del chunk
    finally:
        if sorted_path.exists():
            sorted_path.unlink()
        if temporary_candidate is not None and temporary_candidate.exists():
            temporary_candidate.unlink()
    return candidate_count


def _external_lsd_radix_sort(
    source_path: Path,
    *,
    temporary_dir: Path,
    tag: str,
) -> Path:
    """Stable external uint64 radix sort with bounded RAM.

    Eight least-significant-byte passes alternate disk files.  Each pass first
    counts 256 buckets and then writes them at fixed offsets.  No N-sized index
    array is allocated, which keeps the 10M route safe on a memory-constrained
    host and leaves a genuine (though I/O-heavy) path to larger disk artifacts.
    """
    count = _spool_count(source_path)
    current = source_path
    owned_current = False
    safe_tag = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in tag)
    for pass_index, shift in enumerate(range(0, 64, 8)):
        histogram = np.zeros(256, dtype=np.int64)
        for chunk in _iter_spool_chunks(current):
            byte = np.asarray(
                (chunk["order_hash"] >> np.uint64(shift)) & np.uint64(0xFF),
                dtype=np.int64,
            )
            histogram += np.bincount(byte, minlength=256)
        del chunk, byte
        offsets = np.zeros(256, dtype=np.int64)
        offsets[1:] = np.cumsum(histogram[:-1])
        cursors = offsets.copy()
        next_path = temporary_dir / f"radix_{safe_tag}_pass{pass_index}.bin"
        output = np.memmap(next_path, dtype=SPOOL_DTYPE, mode="w+", shape=(count,))
        for chunk in _iter_spool_chunks(current):
            byte = np.asarray(
                (chunk["order_hash"] >> np.uint64(shift)) & np.uint64(0xFF),
                dtype=np.uint8,
            )
            for bucket in np.unique(byte):
                bucket_int = int(bucket)
                selected = np.asarray(chunk[byte == bucket])
                start = int(cursors[bucket_int])
                stop = start + int(selected.size)
                output[start:stop] = selected
                cursors[bucket_int] = stop
        del chunk, byte, selected
        output.flush()
        del output
        if not np.array_equal(cursors, offsets + histogram):
            raise RuntimeError("external radix pass wrote an unexpected bucket count")
        if owned_current and current.exists():
            current.unlink()
        current = next_path
        owned_current = True
    return current


def _build_ordered_population(
    depth_spools: Sequence[DepthSpool],
    *,
    split: str,
    required: int,
    output_path: Path,
    temporary_dir: Path,
) -> list[dict]:
    if split not in {"train", "test"}:
        raise ValueError(split)
    remaining = int(required)
    rows: list[dict] = []
    with output_path.open("wb") as destination:
        for spool in sorted(depth_spools, key=lambda item: item.depth):
            path = spool.train_path if split == "train" else spool.test_path
            available = spool.train_count if split == "train" else spool.test_count
            if remaining <= 0:
                take = 0
            else:
                take = min(remaining, int(available))
            if take:
                _write_hash_ordered_prefix(
                    path, take, destination, temporary_dir=temporary_dir
                )
                remaining -= take
            rows.append(
                {
                    "depth": int(spool.depth),
                    "available": int(available),
                    "selected": int(take),
                }
            )
    if remaining:
        raise ValueError(
            f"not enough {split} points after scanned LODs: short by {remaining}"
        )
    if _spool_count(output_path) != int(required):
        raise RuntimeError("ordered population has an unexpected record count")
    return rows


def _calibrated_y_transform(ordered_train_path: Path) -> dict:
    """Fit once on the canonical first-1M training prefix.

    The prefix is defined by the frozen LOD/hash order, not by a requested output
    size.  A later 300M build therefore uses exactly the same calibration rows as
    today's 10M build.  Small local fixtures use every available ordered row.
    """
    available = _spool_count(ordered_train_path)
    n_fit = min(Y_CALIBRATION_PREFIX_ROWS, available)
    if n_fit <= 1:
        raise ValueError("at least two canonical training rows are needed for y calibration")
    records = np.memmap(
        ordered_train_path, dtype=SPOOL_DTYPE, mode="r", shape=(available,)
    )
    values = np.asarray(records["z"][:n_fit], dtype=np.float64)
    center = float(np.mean(values))
    scale = float(np.std(values))
    del records
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("canonical y-calibration prefix has invalid mean/std")
    return {
        "method": "canonical_training_order_prefix_standardization",
        "center_m": center,
        "scale_m": scale,
        "calibration_prefix_rows_requested": Y_CALIBRATION_PREFIX_ROWS,
        "n_fit": int(n_fit),
        "tiny_fixture_fallback_used": bool(n_fit < Y_CALIBRATION_PREFIX_ROWS),
        "fit_population": (
            "first min(1,000,000, available) rows of the frozen progressive-LOD "
            "training order"
        ),
        "fixed_across_requested_sizes_and_future_LODs": bool(
            n_fit == Y_CALIBRATION_PREFIX_ROWS
        ),
    }


def _n_test_for_train(n_train: int) -> int:
    return int(round(int(n_train) / (DEFAULT_TEST_MODULUS - 1)))


def _write_npz_and_sidecar(
    *,
    output_dir: Path,
    stem: str,
    n_train: int,
    ordered_train_path: Path,
    ordered_test_path: Path,
    common_metadata: Mapping,
    include_row_provenance: bool,
) -> dict:
    n_train = int(n_train)
    n_test = _n_test_for_train(n_train)
    train_count = _spool_count(ordered_train_path)
    test_count = _spool_count(ordered_test_path)
    train = np.memmap(
        ordered_train_path, dtype=SPOOL_DTYPE, mode="r", shape=(train_count,)
    )
    test = np.memmap(
        ordered_test_path, dtype=SPOOL_DTYPE, mode="r", shape=(test_count,)
    )
    y_transform = common_metadata["y_transform"]
    center = float(y_transform["center_m"])
    scale = float(y_transform["scale_m"])
    x_train = np.empty((n_train, 2), dtype=np.float32)
    x_test = np.empty((n_test, 2), dtype=np.float32)
    x_train[:, 0] = train["x"][:n_train]
    x_train[:, 1] = train["y"][:n_train]
    x_test[:, 0] = test["x"][:n_test]
    x_test[:, 1] = test["y"][:n_test]
    y_train = np.asarray((train["z"][:n_train] - center) / scale, dtype=np.float32)
    y_test = np.asarray((test["z"][:n_test] - center) / scale, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"
    temporary_npz = output_dir / f".{stem}.tmp.npz"
    temporary_json = output_dir / f".{stem}.tmp.json"
    savez = (
        np.savez
        if n_train + n_test >= UNCOMPRESSED_NPZ_THRESHOLD_ROWS
        else np.savez_compressed
    )
    arrays = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_center": np.asarray([center], dtype=np.float64),
        "y_scale": np.asarray([scale], dtype=np.float64),
    }
    if include_row_provenance:
        arrays.update(
            {
                "source_id_train": np.asarray(
                    train["source_id"][:n_train], dtype=np.uint64
                ),
                "source_id_test": np.asarray(
                    test["source_id"][:n_test], dtype=np.uint64
                ),
                "ept_lod_depth_train": np.asarray(
                    train["depth"][:n_train], dtype=np.uint8
                ),
                "ept_lod_depth_test": np.asarray(
                    test["depth"][:n_test], dtype=np.uint8
                ),
            }
        )
    savez(
        temporary_npz,
        **arrays,
    )
    metadata = dict(common_metadata)
    metadata.update(
        {
            "dataset_name": stem,
            "processed_file": str(npz_path),
            "shapes": {"n_train": n_train, "n_test": n_test, "dim": 2},
            "serialization": {
                "npz_mode": (
                    "stored"
                    if n_train + n_test >= UNCOMPRESSED_NPZ_THRESHOLD_ROWS
                    else "deflated"
                ),
                "feature_target_dtype": "float32",
                "source_id_dtype": "uint64",
                "lod_depth_dtype": "uint8",
                "row_provenance_arrays_included": bool(include_row_provenance),
                "row_provenance_storage_note": (
                    "disabled by default because uint64 source IDs plus uint8 depths add "
                    "9 bytes per train/test row (3.375 GB at a 300M/75M split)"
                ),
            },
        }
    )
    temporary_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary_npz, npz_path)
    os.replace(temporary_json, json_path)
    del train, test
    return metadata


def build_ept_datasets(
    *,
    output_dir: Path,
    n_train_list: Iterable[int] = DEFAULT_N_TRAIN_LIST,
    ept_json: str | Path = DEFAULT_EPT_JSON,
    dataset_stem_prefix: str = DEFAULT_STEM_PREFIX,
    cache_dir: Path | None = None,
    temporary_dir: Path | None = None,
    aoi: FixedAoi = FixedAoi(),
    max_download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    chunk_points: int = DEFAULT_CHUNK_POINTS,
    timeout_seconds: float = 300.0,
    allow_large_output: bool = False,
    include_row_provenance: bool = False,
    max_lod_depth: int | None = None,
    source_project: str = DEFAULT_SOURCE_PROJECT,
    noaa_inport_record: str = DEFAULT_NOAA_INPORT_RECORD,
    usgs_project_report: str = DEFAULT_USGS_PROJECT_REPORT,
    usgs_work_unit_report: str = DEFAULT_USGS_WORK_UNIT_REPORT,
    official_mean_ground_density_points_per_m2: float | None = (
        DEFAULT_OFFICIAL_MEAN_GROUND_DENSITY
    ),
    official_work_unit_area_km2: float | None = DEFAULT_OFFICIAL_WORK_UNIT_AREA_KM2,
) -> list[dict]:
    """Build exact progressive-LOD prefixes for all requested training sizes."""
    sizes = _parse_positive_sizes(n_train_list)
    if max(sizes) > LARGE_OUTPUT_WARNING_ROWS and not allow_large_output:
        raise ValueError(
            f"N>{LARGE_OUTPUT_WARNING_ROWS} is a preprocessing/scalability artifact, "
            "not a resident 4 GB GPU run; pass allow_large_output=True explicitly"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(cache_dir)
        if cache_dir is not None
        else output_dir.parent / "raw" / f"USGS_EPT_{source_project}"
    )
    if not str(source_project).strip():
        raise ValueError("source_project must be nonempty")
    if official_mean_ground_density_points_per_m2 is not None:
        if not np.isfinite(official_mean_ground_density_points_per_m2):
            raise ValueError("official mean ground density must be finite")
        if official_mean_ground_density_points_per_m2 <= 0.0:
            raise ValueError("official mean ground density must be positive")
    if official_work_unit_area_km2 is not None:
        if not np.isfinite(official_work_unit_area_km2):
            raise ValueError("official work-unit area must be finite")
        if official_work_unit_area_km2 <= 0.0:
            raise ValueError("official work-unit area must be positive")
    accessor = SourceAccessor(ept_json)
    ept_metadata, ept_provenance = _load_ept_metadata(
        accessor, timeout_seconds=timeout_seconds
    )
    nodes, hierarchy_pages, source_aoi = discover_intersecting_nodes(
        accessor,
        ept_metadata,
        aoi=aoi,
        timeout_seconds=timeout_seconds,
        max_lod_depth=max_lod_depth,
    )
    capacity = estimate_hierarchy_capacity(nodes)
    node_ranks = {node.key: rank for rank, node in enumerate(nodes)}
    max_train = max(sizes)
    max_test = max(_n_test_for_train(size) for size in sizes)

    owned_temporary: tempfile.TemporaryDirectory[str] | None = None
    if temporary_dir is None:
        owned_temporary = tempfile.TemporaryDirectory(
            prefix="usgs_ept_spool_", dir=output_dir
        )
        spool_root = Path(owned_temporary.name)
    else:
        spool_root = Path(temporary_dir)
        spool_root.mkdir(parents=True, exist_ok=True)

    depth_spools: list[DepthSpool] = []
    depth_statistics: list[dict] = []
    scanned_node_rows: list[dict] = []
    cumulative_train = 0
    cumulative_test = 0
    try:
        for depth in sorted({node.depth for node in nodes}):
            depth_nodes = [node for node in nodes if node.depth == depth]
            materialized = materialize_nodes(
                accessor,
                depth_nodes,
                cache_dir=cache_dir,
                timeout_seconds=timeout_seconds,
                max_download_workers=max_download_workers,
            )
            spool = DepthSpool(
                depth=depth,
                train_path=spool_root / f"depth_{depth:02d}_train.bin",
                test_path=spool_root / f"depth_{depth:02d}_test.bin",
            )
            stats, node_rows = _scan_depth_to_spools(
                materialized,
                node_ranks=node_ranks,
                depth_spool=spool,
                aoi=aoi,
                source_crs=DEFAULT_SOURCE_CRS,
                chunk_points=int(chunk_points),
            )
            depth_spools.append(spool)
            scanned_node_rows.extend(node_rows)
            cumulative_train += spool.train_count
            cumulative_test += spool.test_count
            depth_statistics.append(
                {
                    "depth": int(depth),
                    "intersecting_node_count": len(depth_nodes),
                    **stats,
                    "cumulative_train_bucket_points": int(cumulative_train),
                    "cumulative_test_bucket_points": int(cumulative_test),
                }
            )
            # Production builds continue through the canonical 1M calibration
            # prefix even when only a smaller N is requested.  A tiny fixture is
            # allowed to exhaust all hierarchy levels and calibrate on what exists.
            if (
                cumulative_train >= max(max_train, Y_CALIBRATION_PREFIX_ROWS)
                and cumulative_test >= max_test
            ):
                break
        if cumulative_train < max_train or cumulative_test < max_test:
            raise ValueError(
                "EPT hierarchy cannot satisfy the requested exact split through all "
                f"intersecting LODs: available train/test={cumulative_train}/{cumulative_test}, "
                f"requested={max_train}/{max_test}"
            )

        ordered_train = spool_root / "ordered_train.bin"
        ordered_test = spool_root / "ordered_test.bin"
        ordered_train_rows = max(
            max_train, min(Y_CALIBRATION_PREFIX_ROWS, cumulative_train)
        )
        selected_train_by_depth = _build_ordered_population(
            depth_spools,
            split="train",
            required=ordered_train_rows,
            output_path=ordered_train,
            temporary_dir=spool_root,
        )
        selected_test_by_depth = _build_ordered_population(
            depth_spools,
            split="test",
            required=max_test,
            output_path=ordered_test,
            temporary_dir=spool_root,
        )
        terminal_depth = int(depth_spools[-1].depth)
        y_transform = _calibrated_y_transform(ordered_train)
        xmin, ymin, xmax, ymax = aoi.bounds
        aoi_area_km2 = float(aoi.side_m) ** 2 / 1_000_000.0
        capacity_basis_area_km2 = (
            aoi_area_km2
            if official_work_unit_area_km2 is None
            else min(aoi_area_km2, float(official_work_unit_area_km2))
        )
        report_capacity_basis = None
        if official_mean_ground_density_points_per_m2 is not None:
            estimated_ground_points = int(
                round(
                    capacity_basis_area_km2
                    * 1_000_000.0
                    * float(official_mean_ground_density_points_per_m2)
                )
            )
            estimated_training_points = int(
                round(estimated_ground_points * (1.0 - 1.0 / DEFAULT_TEST_MODULUS))
            )
            report_capacity_basis = {
                "official_mean_ground_density_points_per_m2": float(
                    official_mean_ground_density_points_per_m2
                ),
                "density_source": usgs_work_unit_report,
                "fixed_square_aoi_area_km2": aoi_area_km2,
                "official_work_unit_area_km2": (
                    None
                    if official_work_unit_area_km2 is None
                    else float(official_work_unit_area_km2)
                ),
                "capacity_basis_area_km2": capacity_basis_area_km2,
                "capacity_basis_area_rule": (
                    "min(frozen square AOI area, official work-unit area); this avoids "
                    "counting the square's area outside an irregular work-unit boundary"
                ),
                "estimated_ground_points": estimated_ground_points,
                "train_bucket_fraction": 1.0 - 1.0 / DEFAULT_TEST_MODULUS,
                "estimated_training_bucket_points": estimated_training_points,
                "arithmetic": (
                    f"{capacity_basis_area_km2:.6f} km^2 * 1e6 m^2/km^2 * "
                    f"{float(official_mean_ground_density_points_per_m2):.6f} /m^2 * "
                    f"{1.0 - 1.0 / DEFAULT_TEST_MODULUS:.1f}"
                ),
                "meaning": (
                    "planning estimate only; a formal large artifact requires an exact "
                    "class-2/AOI scan and capacity validation"
                ),
            }
        common_metadata = {
            "task_type": "2d_lidar_ground_elevation_regression",
            "paper_task_statement": (
                f"{aoi.crs} horizontal position -> ground elevation"
            ),
            "input_definition": (
                f"EPT {DEFAULT_SOURCE_CRS} horizontal coordinates transformed pointwise "
                f"to {aoi.crs}, strictly filtered to the frozen {aoi.side_m:g} m square "
                "AOI, then fixed-AOI scaled"
            ),
            "target_definition": (
                "LAS classification-2 NAVD88 elevation in meters, standardized by the "
                "canonical first-1M training-order prefix"
            ),
            "source": {
                "collection": "USGS 3DEP Lidar Point Cloud",
                "project": source_project,
                "license": (
                    "U.S. Government data in the public domain; no use restrictions "
                    "reported by the USGS open-data registry"
                ),
                "official_urls": {
                    "aws_open_data_registry": "https://registry.opendata.aws/usgs-lidar/",
                    "noaa_inport_record": noaa_inport_record,
                    "usgs_project_report": usgs_project_report,
                    "usgs_work_unit_report": usgs_work_unit_report,
                    "ept": accessor.describe("ept.json"),
                },
                "ept_json": ept_provenance,
                "ept_total_points": int(ept_metadata["points"]),
                "ept_data_type": ept_metadata["dataType"],
                "ept_hierarchy_type": ept_metadata["hierarchyType"],
                "source_crs": DEFAULT_SOURCE_CRS,
                "hierarchy_pages": hierarchy_pages,
                "scanned_laz_nodes": scanned_node_rows,
            },
            "aoi": {
                "crs": aoi.crs,
                "center_m": [float(aoi.center_x), float(aoi.center_y)],
                "side_m": float(aoi.side_m),
                "bounds_m_inclusive": [xmin, ymin, xmax, ymax],
                "conservative_bounds_in_source_crs": list(source_aoi),
                "choice_rule": (
                    "center, side length, and target CRS frozen before solver screening; "
                    "node bounds are only a coarse filter and every point receives an exact "
                    f"{aoi.crs} square-AOI test"
                ),
            },
            "selection": {
                "classification_keep": [DEFAULT_CLASSIFICATION],
                "requested_max_lod_depth": (
                    None if max_lod_depth is None else int(max_lod_depth)
                ),
                "finite_coordinates_and_elevation_only": True,
                "terminal_lod_depth": terminal_depth,
                "depth_statistics": depth_statistics,
                "actual_train_bucket_capacity_through_terminal_depth": int(
                    cumulative_train
                ),
                "actual_test_bucket_capacity_through_terminal_depth": int(
                    cumulative_test
                ),
                "hierarchy_capacity_estimate": capacity,
                "report_based_capacity_basis": report_capacity_basis,
            },
            "sampling": {
                "population_definition": (
                    "EPT progressive level-of-detail population, not a uniform random sample "
                    "from the full-density source cloud"
                ),
                "lod_order": (
                    "shallower EPT depth first; independent SplitMix64 order within each depth"
                ),
                "split_method": (
                    "SplitMix64(source_id + split seed) modulo 5; residue 0 is test"
                ),
                "order_method": (
                    "SplitMix64(source_id + order seed); constant-memory radix threshold "
                    "selection followed by an eight-pass external stable radix sort per depth"
                ),
                "split_seed_uint64": int(DEFAULT_SPLIT_SEED),
                "order_seed_uint64": int(DEFAULT_ORDER_SEED),
                "source_id_scheme": (
                    "uint64 = (stable rank of full AOI-intersecting node key << 32) | "
                    "zero-based point ordinal in that LAZ node"
                ),
                "train_without_replacement": True,
                "test_without_replacement": True,
                "train_test_disjoint": True,
                "nested_prefixes_across_requested_sizes": True,
                "uses_targets_for_split_or_order": False,
                "selected_train_by_depth_for_output_and_calibration": selected_train_by_depth,
                "selected_test_by_depth_at_max_n": selected_test_by_depth,
            },
            "x_transform": {
                "method": (
                    f"{DEFAULT_SOURCE_CRS}_to_{aoi.crs}_then_fixed_AOI_shift_and_scale"
                ),
                "source_crs": DEFAULT_SOURCE_CRS,
                "target_crs": aoi.crs,
                "origin_m": [xmin, ymin],
                "shared_scale_m": float(aoi.side_m),
                "fit_from_selected_points": False,
                "fixed_across_requested_sizes_and_future_LODs": True,
            },
            "y_transform": y_transform,
            "limitations": [
                (
                    "Progressive EPT LOD prefixes answer a scale-up question; they are not "
                    "uniform subsamples of the terminal full-density point population."
                ),
                (
                    "A 300M NPZ is a disk/preprocessing artifact and is not loaded as resident "
                    "training data on the current 4 GB GPU."
                ),
            ],
        }
        rows = []
        for n_train in sizes:
            stem = f"{dataset_stem_prefix}_n{n_train}"
            rows.append(
                _write_npz_and_sidecar(
                    output_dir=output_dir,
                    stem=stem,
                    n_train=n_train,
                    ordered_train_path=ordered_train,
                    ordered_test_path=ordered_test,
                    common_metadata=common_metadata,
                    include_row_provenance=include_row_provenance,
                )
            )
        summary = {
            "dataset_stem_prefix": dataset_stem_prefix,
            "n_train_list": sizes,
            "ept_json": ept_provenance,
            "aoi": common_metadata["aoi"],
            "terminal_lod_depth": terminal_depth,
            "generated": [
                {
                    "dataset_name": row["dataset_name"],
                    "processed_file": row["processed_file"],
                    **row["shapes"],
                }
                for row in rows
            ],
        }
        summary_path = output_dir / f"{dataset_stem_prefix}_size_sweep_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return rows
    finally:
        if owned_temporary is not None:
            owned_temporary.cleanup()


def capacity_report(
    *,
    ept_json: str | Path = DEFAULT_EPT_JSON,
    aoi: FixedAoi = FixedAoi(),
    timeout_seconds: float = 300.0,
    max_lod_depth: int | None = None,
) -> dict:
    """Inspect only ept.json/hierarchy JSON; do not download point nodes."""
    accessor = SourceAccessor(ept_json)
    ept_metadata, ept_provenance = _load_ept_metadata(
        accessor, timeout_seconds=timeout_seconds
    )
    nodes, hierarchy_pages, source_aoi = discover_intersecting_nodes(
        accessor,
        ept_metadata,
        aoi=aoi,
        timeout_seconds=timeout_seconds,
        max_lod_depth=max_lod_depth,
    )
    return {
        "ept_json": ept_provenance,
        "hierarchy_pages": hierarchy_pages,
        "aoi": {
            "crs": aoi.crs,
            "center_m": [aoi.center_x, aoi.center_y],
            "side_m": aoi.side_m,
            "bounds_m_inclusive": list(aoi.bounds),
            "conservative_bounds_in_source_crs": list(source_aoi),
        },
        "capacity": estimate_hierarchy_capacity(nodes),
        "requested_max_lod_depth": (
            None if max_lod_depth is None else int(max_lod_depth)
        ),
        "caveat": (
            "Hierarchy counts are an upper bound. Exact class-2/AOI capacity is validated "
            "while scanning LAZ nodes for a requested output."
        ),
    }


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build reproducible progressive-LOD USGS EPT ground-elevation datasets."
        )
    )
    parser.add_argument("--ept-json", default=DEFAULT_EPT_JSON)
    parser.add_argument(
        "--n-train-list",
        default=",".join(str(value) for value in DEFAULT_N_TRAIN_LIST),
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--dataset-stem-prefix", default=DEFAULT_STEM_PREFIX)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--temporary-dir",
        "--work-dir",
        dest="temporary_dir",
        type=Path,
        default=None,
        help=(
            "Persistent spool/work directory. LAZ downloads in --cache-dir are always "
            "resumed when a nonempty cached node exists; deterministic spools are rebuilt."
        ),
    )
    parser.add_argument("--aoi-center-x", type=float, default=DEFAULT_AOI_CENTER[0])
    parser.add_argument("--aoi-center-y", type=float, default=DEFAULT_AOI_CENTER[1])
    parser.add_argument("--aoi-side-m", type=float, default=DEFAULT_AOI_SIDE_M)
    parser.add_argument(
        "--aoi-crs",
        default=DEFAULT_AOI_CRS,
        help="Projected target CRS used for the exact square AOI and model inputs.",
    )
    parser.add_argument("--source-project", default=DEFAULT_SOURCE_PROJECT)
    parser.add_argument("--noaa-inport-record", default=DEFAULT_NOAA_INPORT_RECORD)
    parser.add_argument("--usgs-project-report", default=DEFAULT_USGS_PROJECT_REPORT)
    parser.add_argument("--usgs-work-unit-report", default=DEFAULT_USGS_WORK_UNIT_REPORT)
    parser.add_argument(
        "--official-mean-ground-density",
        type=float,
        default=DEFAULT_OFFICIAL_MEAN_GROUND_DENSITY,
        help="Official mean class-2 ground density in points/m^2.",
    )
    parser.add_argument(
        "--official-work-unit-area-km2",
        type=float,
        default=DEFAULT_OFFICIAL_WORK_UNIT_AREA_KM2,
        help=(
            "Official irregular work-unit area in km^2; used only for the report-based "
            "capacity estimate, never for point selection."
        ),
    )
    parser.add_argument(
        "--max-download-workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS
    )
    parser.add_argument("--chunk-points", type=int, default=DEFAULT_CHUNK_POINTS)
    parser.add_argument(
        "--max-lod-depth",
        type=int,
        default=None,
        help=(
            "Optional inclusive EPT depth cap. A continuation sentinel at this depth "
            "is read as a terminal LAZ node without fetching its child hierarchy page."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument(
        "--allow-large-output",
        action="store_true",
        help=(
            "Required above 10M rows; 300M is a disk/preprocessing artifact and is not "
            "run resident on the current 4 GB GPU."
        ),
    )
    parser.add_argument(
        "--include-row-provenance",
        action="store_true",
        help=(
            "Store uint64 source IDs and uint8 EPT depths in NPZ. Disabled by default "
            "to avoid 9 extra bytes per row at large N; hashes remain in the sidecar."
        ),
    )
    parser.add_argument(
        "--cleanup-cache-after-success",
        action="store_true",
        help=(
            "After successful output, remove only *.laz files directly inside an "
            "explicit --cache-dir. JSON sidecars retain URLs and SHA-256 hashes."
        ),
    )
    args = parser.parse_args()
    aoi = FixedAoi(
        center_x=args.aoi_center_x,
        center_y=args.aoi_center_y,
        side_m=args.aoi_side_m,
        crs=args.aoi_crs,
    )
    if args.capacity_only:
        report = capacity_report(
            ept_json=args.ept_json,
            aoi=aoi,
            timeout_seconds=args.timeout_seconds,
            max_lod_depth=args.max_lod_depth,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"{args.dataset_stem_prefix}_capacity.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["capacity"], indent=2))
        print(f"saved: {path}")
        return
    rows = build_ept_datasets(
        output_dir=args.output_dir,
        n_train_list=_parse_positive_sizes(args.n_train_list),
        ept_json=args.ept_json,
        dataset_stem_prefix=args.dataset_stem_prefix,
        cache_dir=args.cache_dir,
        temporary_dir=args.temporary_dir,
        aoi=aoi,
        max_download_workers=args.max_download_workers,
        chunk_points=args.chunk_points,
        timeout_seconds=args.timeout_seconds,
        allow_large_output=args.allow_large_output,
        include_row_provenance=args.include_row_provenance,
        max_lod_depth=args.max_lod_depth,
        source_project=args.source_project,
        noaa_inport_record=args.noaa_inport_record,
        usgs_project_report=args.usgs_project_report,
        usgs_work_unit_report=args.usgs_work_unit_report,
        official_mean_ground_density_points_per_m2=(
            args.official_mean_ground_density
        ),
        official_work_unit_area_km2=args.official_work_unit_area_km2,
    )
    if args.cleanup_cache_after_success:
        if args.cache_dir is None:
            raise ValueError(
                "--cleanup-cache-after-success requires an explicit --cache-dir"
            )
        cache_root = args.cache_dir.expanduser().resolve()
        removed = 0
        for candidate in cache_root.glob("*.laz"):
            resolved = candidate.resolve()
            if resolved.parent != cache_root:
                raise RuntimeError(f"refusing cache cleanup outside {cache_root}")
            resolved.unlink()
            removed += 1
        print(f"removed {removed} cached LAZ node(s) from {cache_root}")
    print(json.dumps([row["shapes"] for row in rows], indent=2))


if __name__ == "__main__":
    main()
