from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.suite import (
    build_suite_plan,
    load_suite_config,
    validate_suite_case,
)


CONTROLLED_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = CONTROLLED_DIR / "three_dataset_suite.json"
DATASET_DIR = Path(__file__).resolve().parents[3] / "benchmark_dataset" / "processed"


class GeoLifeSuiteConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_suite_config(CONFIG_PATH)

    def test_main_profiles_use_geolife_and_exclude_mur(self) -> None:
        self.assertEqual(
            set(self.suite["dataset_aliases"]),
            {"geolife_n100000", "geolife_n1000000", "geolife_n10000000"},
        )
        expected_alias = {
            "demo": "geolife_n100000",
            "local_1m": "geolife_n1000000",
            "scale_10m": "geolife_n10000000",
        }
        for profile_name, alias_name in expected_alias.items():
            cases = self.suite["profiles"][profile_name]["cases"]
            self.assertEqual(cases[0]["dataset_alias"], alias_name)
            self.assertEqual(len(cases), 3)
        scale_100m = self.suite["profiles"]["scale_100m"]["cases"]
        self.assertEqual(
            [case["id"] for case in scale_100m],
            ["synthetic_n100000000", "usgs_n100000000"],
        )
        self.assertNotIn("mur", json.dumps(self.suite["dataset_aliases"]).lower())
        self.assertNotIn("mur", json.dumps(self.suite["profiles"]).lower())

    def test_all_local_profiles_validate_against_real_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            for profile_name in ("demo", "local_1m", "scale_10m", "scale_100m"):
                plan = build_suite_plan(
                    self.suite,
                    profile_name=profile_name,
                    output_root=output_root / profile_name,
                )
                self.assertEqual(len(plan), len(self.suite["profiles"][profile_name]["cases"]))

    def test_geolife_aliases_freeze_protocol_and_capacity(self) -> None:
        expected = {
            "geolife_n100000": (100_000, 25_000),
            "geolife_n1000000": (1_000_000, 250_000),
            "geolife_n10000000": (10_000_000, 2_500_000),
        }
        for alias_name, (n_train, n_test) in expected.items():
            alias = self.suite["dataset_aliases"][alias_name]
            equal = alias["metadata_equals"]
            self.assertEqual(equal["shapes.n_train"], n_train)
            self.assertEqual(equal["shapes.n_test"], n_test)
            self.assertEqual(equal["source.archive_sha256"], (
                "1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6"
            ))
            self.assertEqual(equal["sampling.split_unit"], "complete PLT trajectory")
            self.assertTrue(equal["sampling.train_without_replacement"])
            self.assertTrue(equal["sampling.source_record_ids_unique"])
            self.assertFalse(equal["sampling.input_coordinates_required_unique"])
            self.assertTrue(equal["frozen_policy.uses_targets_for_row_selection"])
            self.assertIn("non-commercial", equal["source.license"])
            self.assertIn("may not be distributed", equal["source.redistribution_note"])
            self.assertGreaterEqual(
                alias["metadata_minimums"]["cleaning_audit.available_train_bucket_rows"],
                10_000_000,
            )
            validation = validate_suite_case(
                {
                    "id": f"validate_{alias_name}",
                    "dataset_alias": alias_name,
                    "n_train": 0,
                    "expected_n_train": n_train,
                },
                dataset_dir=DATASET_DIR,
                aliases=self.suite["dataset_aliases"],
            )
            self.assertEqual(validation["n_train"], n_train)

    def test_example_configs_use_the_geolife_100k_artifact(self) -> None:
        expected_stem = "GeoLife_Beijing_GPS_altitude_regression_ntrain100000"
        for name in ("example_spatial_config.json", "example_spatial_strict_box_config.json"):
            payload = json.loads((CONTROLLED_DIR / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_stem"], expected_stem)
            self.assertEqual(payload["kernel_family"], "se")
            self.assertEqual(payload["lengthscale"], 0.02)

    def test_changed_frozen_metadata_is_rejected_before_npz_loading(self) -> None:
        alias_name = "geolife_n100000"
        alias = copy.deepcopy(self.suite["dataset_aliases"][alias_name])
        source_json = DATASET_DIR / f"{alias['dataset_stem']}.json"
        metadata = json.loads(source_json.read_text(encoding="utf-8"))
        metadata["frozen_policy"]["lon_min"] = 116.09
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary)
            (dataset_dir / f"{alias['dataset_stem']}.npz").touch()
            (dataset_dir / f"{alias['dataset_stem']}.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "frozen_policy.lon_min"):
                validate_suite_case(
                    {
                        "id": "changed_crop",
                        "dataset_alias": alias_name,
                        "n_train": 0,
                        "expected_n_train": 100_000,
                    },
                    dataset_dir=dataset_dir,
                    aliases={alias_name: alias},
                )


if __name__ == "__main__":
    unittest.main()
