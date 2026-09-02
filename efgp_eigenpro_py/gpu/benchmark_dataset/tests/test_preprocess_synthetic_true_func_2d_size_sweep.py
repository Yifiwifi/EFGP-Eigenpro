from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from efgp_eigenpro_py.gpu.benchmark_dataset import (
    preprocess_synthetic_true_func_2d as single,
)
from efgp_eigenpro_py.gpu.benchmark_dataset import (
    preprocess_synthetic_true_func_2d_size_sweep as sweep,
)


ARRAY_KEYS = (
    "x_train",
    "x_test",
    "y_train",
    "y_test",
    "y_train_true",
    "train_noise",
)


class SyntheticPrefixSweepTest(unittest.TestCase):
    def test_parse_normalizes_unordered_duplicate_sizes(self) -> None:
        self.assertEqual(sweep._parse_n_train_list("8, 4,8, 12"), [4, 8, 12])

    def test_reused_aligned_prefixes_match_independent_builds_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            expected_arrays: dict[int, dict[str, np.ndarray]] = {}
            expected_metadata: dict[int, dict] = {}
            for n_train in (4, 8):
                stem = f"equivalence_n{n_train}"
                output_npz = output_dir / f"{stem}.npz"
                output_json = output_dir / f"{stem}.json"
                expected_metadata[n_train] = single.build_synthetic_dataset(
                    output_npz,
                    output_json,
                    dataset_stem=stem,
                    n_train=n_train,
                    n_test=int(round(n_train * 0.25)),
                    noise=0.3,
                    seed_train=123,
                    seed_test=456,
                    chunk_rows=4,
                )
                with np.load(output_npz) as archive:
                    expected_arrays[n_train] = {
                        key: np.array(archive[key], copy=True) for key in ARRAY_KEYS
                    }

            with mock.patch.object(
                single,
                "_generate_train_to_memmaps",
                wraps=single._generate_train_to_memmaps,
            ) as generate:
                rows, failures = sweep.build_synthetic_prefix_sweep(
                    output_dir,
                    n_train_list=[8, 4, 8],
                    dataset_stem_prefix="equivalence",
                    size_token="n",
                    noise=0.3,
                    seed_train=123,
                    seed_test=456,
                    chunk_rows=4,
                )

            self.assertEqual(generate.call_count, 1)
            self.assertEqual([row["n_train"] for row in rows], [4, 8])
            self.assertEqual(failures, [])
            self.assertFalse(any(output_dir.glob("synthetic_prefix_tmp_*")))

            for n_train in (4, 8):
                stem = f"equivalence_n{n_train}"
                output_npz = output_dir / f"{stem}.npz"
                output_json = output_dir / f"{stem}.json"
                with np.load(output_npz) as archive:
                    for key in ARRAY_KEYS:
                        np.testing.assert_array_equal(
                            archive[key],
                            expected_arrays[n_train][key],
                            err_msg=f"mismatch for n_train={n_train}, array={key}",
                        )
                actual_metadata = json.loads(output_json.read_text(encoding="utf-8"))
                self.assertEqual(actual_metadata, expected_metadata[n_train])

    def test_unaligned_prefix_fails_closed_before_creating_work_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with self.assertRaisesRegex(ValueError, "not bit-for-bit safe"):
                sweep.build_synthetic_prefix_sweep(
                    output_dir,
                    n_train_list=[5, 8],
                    dataset_stem_prefix="unsafe",
                    size_token="n",
                    chunk_rows=4,
                )
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_failed_serialization_removes_owned_temporary_files_and_memmaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)

            def write_partial_then_fail(path, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                Path(path).write_bytes(b"partial")
                raise OSError("injected serialization failure")

            with mock.patch.object(np, "savez_compressed", side_effect=write_partial_then_fail):
                with self.assertRaisesRegex(OSError, "injected serialization failure"):
                    sweep.build_synthetic_prefix_sweep(
                        output_dir,
                        n_train_list=[4, 8],
                        dataset_stem_prefix="cleanup",
                        size_token="n",
                        chunk_rows=4,
                    )

            self.assertFalse(any(output_dir.glob("synthetic_prefix_tmp_*")))
            self.assertFalse(any(output_dir.glob(".*.tmp*")))
            self.assertFalse(any(output_dir.glob("cleanup_n*.npz")))
            self.assertFalse(any(output_dir.glob("cleanup_n*.json")))


if __name__ == "__main__":
    unittest.main()
