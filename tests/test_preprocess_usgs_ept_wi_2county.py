from __future__ import annotations

import hashlib
import json
from pathlib import Path

import laspy
import numpy as np
from pyproj import Transformer

from efgp_eigenpro_py.gpu.benchmark_dataset.preprocess_usgs_ept_wi_2county import (
    DepthSpool,
    EptNode,
    FixedAoi,
    MaterializedNode,
    SPOOL_DTYPE,
    _scan_depth_to_spools,
    _splitmix64,
    build_ept_datasets,
    capacity_report,
)


def _write_laz(
    path: Path,
    *,
    east: np.ndarray,
    north: np.ndarray,
    elevation: np.ndarray,
    classification: np.ndarray,
    target_crs: str = "EPSG:6344",
) -> None:
    to_mercator = Transformer.from_crs(target_crs, "EPSG:3857", always_xy=True)
    x, y = to_mercator.transform(east, north)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    header.offsets = np.asarray([0.0, 0.0, 0.0])
    cloud = laspy.LasData(header)
    cloud.x = np.asarray(x)
    cloud.y = np.asarray(y)
    cloud.z = np.asarray(elevation)
    cloud.classification = np.asarray(classification, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.write(path, do_compress=True)


def _node_points(aoi: FixedAoi, *, depth: int, n: int = 100) -> tuple[np.ndarray, ...]:
    # Most points are eligible.  The final ten exercise exact AOI and class filters.
    t = np.arange(n, dtype=np.float64)
    east = aoi.center_x - 0.35 * aoi.side_m + (t % 13) / 12.0 * 0.7 * aoi.side_m
    north = aoi.center_y - 0.35 * aoi.side_m + (t % 17) / 16.0 * 0.7 * aoi.side_m
    elevation = 240.0 + 0.7 * east / 1000.0 + 0.4 * north / 1000.0 + depth
    classification = np.full(n, 2, dtype=np.uint8)
    classification[-5:] = 5
    east[-10:-5] = aoi.center_x + 0.75 * aoi.side_m
    return east, north, elevation, classification


def _write_ept_fixture(root: Path, aoi: FixedAoi) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    to_mercator = Transformer.from_crs(aoi.crs, "EPSG:3857", always_xy=True)
    sx0, sy0, sx1, sy1 = to_mercator.transform_bounds(*aoi.bounds, densify_pts=21)
    # Put the complete AOI inside the x/y index-zero cell even at depth two.
    x0 = sx0 - 1_000.0
    y0 = sy0 - 1_000.0
    root_width = 4.0 * ((sx1 - sx0) + 2_000.0)
    root_height = 4.0 * ((sy1 - sy0) + 2_000.0)
    ept = {
        "bounds": [x0, y0, 0.0, x0 + root_width, y0 + root_height, 1_000.0],
        "boundsConforming": [sx0, sy0, 100.0, sx1, sy1, 600.0],
        "dataType": "laszip",
        "hierarchyType": "json",
        "points": 310,
        "schema": [],
        "span": 128,
        "srs": {"authority": "EPSG", "horizontal": "3857"},
        "version": "1.0.0",
    }
    (root / "ept.json").write_text(json.dumps(ept, indent=2), encoding="utf-8")
    hierarchy = root / "ept-hierarchy"
    hierarchy.mkdir()
    (hierarchy / "0-0-0-0.json").write_text(
        json.dumps(
            {
                "0-0-0-0": 100,
                "1-0-0-0": -1,
                # This node does not intersect the AOI and intentionally has no LAZ.
                "1-1-1-0": 10,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (hierarchy / "1-0-0-0.json").write_text(
        json.dumps({"1-0-0-0": 100, "2-0-0-1": 100}, indent=2),
        encoding="utf-8",
    )
    for depth, key in ((0, "0-0-0-0"), (1, "1-0-0-0"), (2, "2-0-0-1")):
        east, north, elevation, classification = _node_points(aoi, depth=depth)
        _write_laz(
            root / "ept-data" / f"{key}.laz",
            east=east,
            north=north,
            elevation=elevation,
            classification=classification,
            target_crs=aoi.crs,
        )
    return root / "ept.json"


def test_local_ept_fixture_is_recursive_filtered_hashed_and_nested(tmp_path: Path) -> None:
    aoi = FixedAoi()
    source = _write_ept_fixture(tmp_path / "source", aoi)
    output = tmp_path / "processed"
    rows = build_ept_datasets(
        output_dir=output,
        n_train_list=[20, 80],
        ept_json=source,
        dataset_stem_prefix="EPT_fixture",
        cache_dir=tmp_path / "cache",
        aoi=aoi,
        max_download_workers=2,
        chunk_points=19,
        include_row_provenance=True,
    )
    assert [row["shapes"]["n_train"] for row in rows] == [20, 80]

    small = np.load(output / "EPT_fixture_n20.npz")
    large = np.load(output / "EPT_fixture_n80.npz")
    assert small["x_train"].shape == (20, 2)
    assert small["x_test"].shape == (5, 2)
    assert large["x_train"].shape == (80, 2)
    assert large["x_test"].shape == (20, 2)
    for key in ("x_train", "y_train", "source_id_train", "ept_lod_depth_train"):
        assert np.array_equal(small[key], large[key][:20])
    for key in ("x_test", "y_test", "source_id_test", "ept_lod_depth_test"):
        assert np.array_equal(small[key], large[key][:5])
    assert np.unique(large["source_id_train"]).size == 80
    assert not np.intersect1d(
        large["source_id_train"], large["source_id_test"]
    ).size
    assert np.all((large["x_train"] >= 0.0) & (large["x_train"] <= 1.0))
    assert np.all(np.diff(large["ept_lod_depth_train"].astype(np.int16)) >= 0)
    assert np.array_equal(small["y_center"], large["y_center"])
    assert np.array_equal(small["y_scale"], large["y_scale"])

    sidecar_path = output / "EPT_fixture_n80.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["source"]["project"] == "WI_2County_2_B23"
    assert len(sidecar["source"]["hierarchy_pages"]) == 2
    assert len(sidecar["source"]["scanned_laz_nodes"]) == 3
    assert all(
        len(row["sha256"]) == 64 for row in sidecar["source"]["scanned_laz_nodes"]
    )
    assert sidecar["selection"]["classification_keep"] == [2]
    depth_rows = sidecar["selection"]["depth_statistics"]
    assert sum(row["classification_rejected"] for row in depth_rows) == 15
    assert sum(row["aoi_rejected_after_classification"] for row in depth_rows) == 15
    assert sidecar["sampling"]["nested_prefixes_across_requested_sizes"] is True
    assert "not a uniform random sample" in sidecar["sampling"]["population_definition"]
    assert sidecar["x_transform"]["origin_m"] == list(aoi.bounds[:2])
    assert sidecar["x_transform"]["shared_scale_m"] == 8_000.0
    assert sidecar["y_transform"]["n_fit"] > 80
    assert sidecar["y_transform"]["tiny_fixture_fallback_used"] is True

    expected_ept_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sidecar["source"]["ept_json"]["sha256"] == expected_ept_hash


def test_alternate_work_unit_crs_and_provenance_are_not_hardcoded(
    tmp_path: Path,
) -> None:
    aoi = FixedAoi(
        center_x=437_100.0,
        center_y=4_884_400.0,
        side_m=1_000.0,
        crs="EPSG:6345",
    )
    source = _write_ept_fixture(tmp_path / "source_6345", aoi)
    output = tmp_path / "processed_6345"
    report_url = "https://example.test/manitowoc_work_unit_report.pdf"
    build_ept_datasets(
        output_dir=output,
        n_train_list=[20],
        ept_json=source,
        dataset_stem_prefix="EPT_fixture_6345",
        cache_dir=tmp_path / "cache_6345",
        aoi=aoi,
        chunk_points=19,
        source_project="WI_2County_1_B23",
        usgs_work_unit_report=report_url,
        official_mean_ground_density_points_per_m2=11.37,
        official_work_unit_area_km2=1_660.18,
    )
    sidecar = json.loads(
        (output / "EPT_fixture_6345_n20.json").read_text(encoding="utf-8")
    )
    assert sidecar["source"]["project"] == "WI_2County_1_B23"
    assert sidecar["paper_task_statement"].startswith("EPSG:6345")
    assert "EPSG:6345" in sidecar["input_definition"]
    assert "1000 m square" in sidecar["input_definition"]
    assert "8 km" not in sidecar["aoi"]["choice_rule"]
    assert "EPSG:6345" in sidecar["aoi"]["choice_rule"]
    assert sidecar["x_transform"]["target_crs"] == "EPSG:6345"
    basis = sidecar["selection"]["report_based_capacity_basis"]
    assert basis["density_source"] == report_url
    assert basis["capacity_basis_area_km2"] == 1.0
    assert basis["estimated_ground_points"] == 11_370_000
    assert basis["estimated_training_bucket_points"] == 9_096_000


def test_capacity_only_does_not_open_nonintersecting_laz(tmp_path: Path) -> None:
    aoi = FixedAoi()
    source = _write_ept_fixture(tmp_path / "source", aoi)
    report = capacity_report(ept_json=source, aoi=aoi)
    depths = report["capacity"]["depths"]
    assert [row["depth"] for row in depths] == [0, 1, 2]
    assert report["capacity"]["total_intersecting_node_points_upper_bound"] == 300
    assert "upper bound" in report["caveat"]


def test_max_lod_terminal_sentinel_does_not_fetch_child_page(tmp_path: Path) -> None:
    aoi = FixedAoi()
    source = _write_ept_fixture(tmp_path / "source", aoi)
    # Depth one is a -1 continuation entry.  With an inclusive cap it becomes an
    # unknown-count terminal data node, so the continuation JSON is unnecessary.
    (source.parent / "ept-hierarchy" / "1-0-0-0.json").unlink()
    report = capacity_report(ept_json=source, aoi=aoi, max_lod_depth=1)
    capacity = report["capacity"]
    assert report["requested_max_lod_depth"] == 1
    assert [row["depth"] for row in capacity["depths"]] == [0, 1]
    assert capacity["total_intersecting_node_points_known"] == 100
    assert capacity["unknown_terminal_node_count"] == 1
    assert capacity["has_unknown_terminal_counts"] is True
    assert capacity["total_intersecting_node_points_upper_bound"] is None
    assert capacity["depths"][1]["unknown_point_node_keys"] == ["1-0-0-0"]


def test_splitmix_order_is_deterministic_and_bijective_on_fixture_ids() -> None:
    source_ids = np.arange(10_000, dtype=np.uint64)
    first = _splitmix64(source_ids, np.uint64(123))
    second = _splitmix64(source_ids, np.uint64(123))
    assert np.array_equal(first, second)
    assert np.unique(first).size == source_ids.size


def test_scan_chunk_with_exactly_two_ground_points_avoids_laspy_scaled_view_bug(
    tmp_path: Path,
) -> None:
    aoi = FixedAoi()
    laz_path = tmp_path / "two_ground.laz"
    east = np.asarray(
        [aoi.center_x - 10.0, aoi.center_x + 10.0, aoi.center_x, aoi.center_x]
    )
    north = np.asarray(
        [aoi.center_y - 20.0, aoi.center_y + 20.0, aoi.center_y, aoi.center_y]
    )
    elevation = np.asarray([250.0, 251.0, 400.0, 401.0])
    classification = np.asarray([2, 2, 5, 5], dtype=np.uint8)
    _write_laz(
        laz_path,
        east=east,
        north=north,
        elevation=elevation,
        classification=classification,
    )
    node = EptNode(
        key="0-0-0-0",
        depth=0,
        x_index=0,
        y_index=0,
        z_index=0,
        hierarchy_points=4,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    )
    materialized = MaterializedNode(
        node=node,
        path=laz_path,
        source_url=str(laz_path),
        sha256=hashlib.sha256(laz_path.read_bytes()).hexdigest(),
        byte_count=laz_path.stat().st_size,
    )
    spool = DepthSpool(
        depth=0,
        train_path=tmp_path / "train.bin",
        test_path=tmp_path / "test.bin",
    )
    stats, rows = _scan_depth_to_spools(
        [materialized],
        node_ranks={node.key: 0},
        depth_spool=spool,
        aoi=aoi,
        source_crs="EPSG:3857",
        chunk_points=4,
    )
    assert stats["iterated_points"] == 4
    assert stats["classification_2_points"] == 2
    assert stats["exact_aoi_ground_points"] == 2
    assert stats["classification_rejected"] == 2
    assert spool.train_count + spool.test_count == 2
    assert rows[0]["exact_aoi_ground_points"] == 2
    assert (
        spool.train_path.stat().st_size + spool.test_path.stat().st_size
        == 2 * SPOOL_DTYPE.itemsize
    )
