from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import benchmark
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import end_to_end
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    end_to_end_suite,
)


def test_batch_cache_loads_once_and_reprobes_live_memory(monkeypatch, tmp_path):
    cfg = end_to_end.EndToEndConfig(
        dataset_dir=str(tmp_path),
        dataset_stem="shared",
        n_train=4,
        max_test_rows=2,
    )
    dataset = {
        "x": np.zeros((4, 2)),
        "y": np.zeros(4),
        "x_test": np.zeros((2, 2)),
        "y_test": np.zeros(2),
        "n_test": 2,
        "source_n_train": 4,
    }
    calls = {"load": 0, "validate": 0, "probe": 0, "enable": 0}

    def fake_load(_cfg):
        calls["load"] += 1
        return dataset

    def fake_validate(_cfg, payload):
        calls["validate"] += 1
        assert payload is dataset
        return {"validated": True}

    def fake_probe():
        calls["probe"] += 1
        return 123

    def fake_enable(payload):
        calls["enable"] += 1
        assert payload is dataset

    monkeypatch.setattr(end_to_end, "load_end_to_end_dataset", fake_load)
    monkeypatch.setattr(
        end_to_end, "validate_dataset_generation_provenance", fake_validate
    )
    monkeypatch.setattr(
        end_to_end, "_probe_available_device_bytes_without_allocation", fake_probe
    )
    monkeypatch.setattr(
        end_to_end.fixed_ab, "enable_batch_gpu_training_reuse", fake_enable
    )
    monkeypatch.setattr(end_to_end, "_release_gpu_allocator_cache", lambda: None)

    cache = end_to_end.EndToEndBatchCache()
    assert cache.probe_available_device_bytes() == 123
    assert cache.probe_available_device_bytes() == 123
    first, first_reused = cache.acquire_dataset(cfg)
    second, second_reused = cache.acquire_dataset(
        end_to_end.replace(cfg, rank=320, output_dir=str(tmp_path / "case2"))
    )

    assert first is second is dataset
    assert first_reused is False
    assert second_reused is True
    assert dataset["validated"] is True
    assert calls == {"load": 1, "validate": 1, "probe": 2, "enable": 1}
    with pytest.raises(ValueError, match="cannot mix dataset groups"):
        cache.acquire_dataset(end_to_end.replace(cfg, n_train=5))
    cache.close()


def test_controlled_gpu_training_arrays_are_staged_once():
    class FakeXP:
        def __init__(self):
            self.asarray_calls = 0

        def asarray(self, value):
            self.asarray_calls += 1
            return np.asarray(value).copy()

    xp = FakeXP()
    backend = SimpleNamespace(xp=xp)
    dataset = {
        "x": np.arange(8, dtype=np.float64).reshape(4, 2),
        "y": np.arange(4, dtype=np.float64),
    }
    benchmark.enable_batch_gpu_training_reuse(dataset)

    first = benchmark._training_data_context(
        backend, dataset["x"], dataset["y"], dataset
    )
    second = benchmark._training_data_context(
        backend, dataset["x"], dataset["y"], dataset
    )

    assert xp.asarray_calls == 2
    assert second.x_gpu is first.x_gpu
    assert second.y_gpu is first.y_gpu
    assert benchmark.batch_gpu_training_reuse_diagnostics(dataset) == {
        "enabled": True,
        "staged": True,
        "stage_count": 1,
        "reuse_count": 1,
        "cached_bytes": dataset["x"].nbytes + dataset["y"].nbytes,
    }


def test_run_plan_stably_groups_dataset_cases_and_keeps_case_outputs(
    monkeypatch, tmp_path
):
    def item(case_id, stem, n_train):
        return {
            "profile": "sweep",
            "case_id": case_id,
            "dataset_family": stem,
            "config": end_to_end.EndToEndConfig(
                dataset_dir=str(tmp_path / "data"),
                dataset_stem=stem,
                n_train=n_train,
                methods=("ours-binned-active-eig",),
                output_dir=str(tmp_path / "out" / "sweep" / case_id),
            ),
        }

    plan = [
        item("a1", "a", 10),
        item("b1", "b", 10),
        item("a2", "a", 10),
    ]
    calls = []

    def fake_run(cfg, *, batch_cache):
        calls.append((cfg.dataset_stem, cfg.output_dir, batch_cache))
        return {
            "output_dir": cfg.output_dir,
            "completion": {
                "n_train": cfg.n_train,
                "n_test": 2,
                "all_rows_present": True,
                "formal_result_status": "claim_eligible_complete",
                "resource_limit_methods": [],
                "performance_ineligible_methods": [],
            },
        }

    monkeypatch.setattr(end_to_end_suite, "_load_completed_case", lambda cfg: None)
    monkeypatch.setattr(end_to_end_suite, "run_end_to_end_experiment", fake_run)
    monkeypatch.setattr(end_to_end, "_release_gpu_allocator_cache", lambda: None)

    index = end_to_end_suite.run_plan(
        plan, index_root=tmp_path / "index", resume=True
    )

    assert [row[0] for row in calls] == ["a", "a", "b"]
    assert calls[0][2] is calls[1][2]
    assert calls[2][2] is not calls[0][2]
    assert [row["case_id"] for row in index] == ["a1", "a2", "b1"]
    assert [row["output_dir"] for row in index] == [
        str(tmp_path / "out" / "sweep" / "a1"),
        str(tmp_path / "out" / "sweep" / "a2"),
        str(tmp_path / "out" / "sweep" / "b1"),
    ]


def test_empty_dataset_dir_identity_uses_loader_environment_without_io(
    monkeypatch, tmp_path
):
    processed = tmp_path / "configured-processed"
    monkeypatch.setenv("BTAB_PROCESSED_DIR", str(processed))
    cfg = end_to_end.EndToEndConfig(dataset_dir="", dataset_stem="shared")

    identity = end_to_end.dataset_execution_identity(cfg)

    assert identity[0] == str(processed.resolve())
    assert not processed.exists()


def test_live_memory_probe_adds_back_only_retained_training_arrays(
    monkeypatch,
):
    probes = iter((1_000, 600))
    releases = []
    monkeypatch.setattr(
        end_to_end,
        "_probe_available_device_bytes_without_allocation",
        lambda: next(probes),
    )
    monkeypatch.setattr(
        end_to_end, "_release_gpu_allocator_cache", lambda: releases.append(True)
    )
    cache = end_to_end.EndToEndBatchCache()
    assert cache.probe_available_device_bytes() == 1_000

    dataset = {"x": np.zeros((4, 2)), "y": np.zeros(4)}
    benchmark.enable_batch_gpu_training_reuse(dataset)
    state = dataset[benchmark._BATCH_GPU_TRAINING_CACHE_KEY]
    state["x_gpu"] = np.zeros((4, 2), dtype=np.float64)
    state["y_gpu"] = np.zeros(4, dtype=np.float64)
    cache.dataset = dataset

    cached_bytes = state["x_gpu"].nbytes + state["y_gpu"].nbytes
    assert cache.probe_available_device_bytes() == 600 + cached_bytes
    assert releases == [True]
    cache.close()


def test_empty_batch_close_does_not_touch_gpu_allocator(monkeypatch):
    releases = []
    monkeypatch.setattr(
        end_to_end, "_release_gpu_allocator_cache", lambda: releases.append(True)
    )

    end_to_end.EndToEndBatchCache().close()

    assert releases == []


def test_run_plan_closes_batch_cache_on_keyboard_interrupt(monkeypatch, tmp_path):
    cfg = end_to_end.EndToEndConfig(
        dataset_dir=str(tmp_path),
        dataset_stem="shared",
        n_train=4,
        output_dir=str(tmp_path / "out"),
    )
    plan = [{"profile": "sweep", "case_id": "case", "config": cfg}]
    closed = []
    original_close = end_to_end.EndToEndBatchCache.close

    def tracked_close(cache):
        closed.append(True)
        original_close(cache)

    monkeypatch.setattr(end_to_end_suite, "_load_completed_case", lambda _cfg: None)
    monkeypatch.setattr(
        end_to_end_suite,
        "run_end_to_end_experiment",
        lambda _cfg, *, batch_cache: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(end_to_end.EndToEndBatchCache, "close", tracked_close)
    monkeypatch.setattr(end_to_end, "_release_gpu_allocator_cache", lambda: None)

    with pytest.raises(KeyboardInterrupt):
        end_to_end_suite.run_plan(plan, index_root=tmp_path / "index")

    assert closed == [True]


def _completed_result(
    cfg,
    *,
    content_sha="old-content",
    metadata_sha="old-metadata",
    dataset_loaded=True,
):
    return {
        "output_dir": cfg.output_dir,
        "resumed_existing": True,
        "completion": {
            "n_train": cfg.n_train,
            "n_test": 2,
            "all_rows_present": True,
            "formal_result_status": "claim_eligible_complete",
            "resource_limit_methods": [],
            "performance_ineligible_methods": [],
            "dataset_loaded": dataset_loaded,
            "dataset_provenance": {
                "content_index_sha256": content_sha if dataset_loaded else None,
                "metadata_sha256": metadata_sha if dataset_loaded else None,
            },
        },
        "summary": [
            {
                "dataset_content_index_sha256": (
                    content_sha if dataset_loaded else None
                ),
                "dataset_metadata_sha256": metadata_sha if dataset_loaded else None,
            }
        ],
    }


def test_mixed_resume_rejects_changed_dataset_before_gpu_staging(
    monkeypatch, tmp_path
):
    def item(case_id):
        return {
            "profile": "sweep",
            "case_id": case_id,
            "config": end_to_end.EndToEndConfig(
                dataset_dir=str(tmp_path),
                dataset_stem="shared",
                n_train=4,
                output_dir=str(tmp_path / case_id),
            ),
        }

    resumed, execute = item("resumed"), item("execute")
    loads = []
    gpu_enables = []

    def load_completed(cfg):
        return _completed_result(cfg) if cfg.output_dir.endswith("resumed") else None

    def load_dataset(_cfg):
        loads.append(True)
        return {
            "x": np.zeros((4, 2)),
            "y": np.zeros(4),
            "content_index_sha256": "new-content",
            "metadata_sha256": "new-metadata",
        }

    def fake_run(cfg, *, batch_cache):
        batch_cache.acquire_dataset(cfg)
        raise AssertionError("fingerprint mismatch must stop before execution")

    monkeypatch.setattr(end_to_end_suite, "_load_completed_case", load_completed)
    monkeypatch.setattr(end_to_end_suite, "run_end_to_end_experiment", fake_run)
    monkeypatch.setattr(end_to_end, "load_end_to_end_dataset", load_dataset)
    monkeypatch.setattr(
        end_to_end.fixed_ab,
        "enable_batch_gpu_training_reuse",
        lambda _dataset: gpu_enables.append(True),
    )
    monkeypatch.setattr(end_to_end, "_release_gpu_allocator_cache", lambda: None)

    index = end_to_end_suite.run_plan(
        [resumed, execute], index_root=tmp_path / "index", resume=True
    )

    assert loads == [True]
    assert gpu_enables == []
    assert index[0]["invocation_mode"] == "resumed_existing"
    assert index[1]["status"] == "error"
    assert "does not match the completed cases resumed" in index[1]["error_message"]


def test_all_resumed_batch_does_not_open_source_dataset(monkeypatch, tmp_path):
    items = []
    completed = {}
    for case_id in ("one", "two"):
        cfg = end_to_end.EndToEndConfig(
            dataset_dir=str(tmp_path / "missing-source"),
            dataset_stem="shared",
            n_train=4,
            output_dir=str(tmp_path / case_id),
        )
        items.append({"profile": "sweep", "case_id": case_id, "config": cfg})
        completed[cfg.output_dir] = _completed_result(cfg)

    monkeypatch.setattr(
        end_to_end_suite,
        "_load_completed_case",
        lambda cfg: completed[cfg.output_dir],
    )
    monkeypatch.setattr(
        end_to_end_suite,
        "run_end_to_end_experiment",
        lambda *_args, **_kwargs: pytest.fail("all-resumed batch must not execute"),
    )
    monkeypatch.setattr(
        end_to_end,
        "load_end_to_end_dataset",
        lambda _cfg: pytest.fail("all-resumed batch must not load source data"),
    )
    monkeypatch.setattr(
        end_to_end,
        "_release_gpu_allocator_cache",
        lambda: pytest.fail("all-resumed batch must not touch GPU allocator"),
    )

    index = end_to_end_suite.run_plan(
        items, index_root=tmp_path / "index", resume=True
    )

    assert [row["invocation_mode"] for row in index] == [
        "resumed_existing",
        "resumed_existing",
    ]


def test_no_batch_reuse_flag_reaches_two_phase_campaign(monkeypatch, tmp_path):
    observed = {}
    monkeypatch.setattr(end_to_end_suite, "load_suite_config", lambda _path: {})

    def fake_campaign(_suite, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(end_to_end_suite, "run_stage1_campaign", fake_campaign)

    assert (
        end_to_end_suite.main(
            [
                "--dataset-dir",
                str(tmp_path / "data"),
                "--output-root",
                str(tmp_path / "out"),
                "--run-robustness-after-selection",
                "--no-dataset-batch-reuse",
            ]
        )
        == 0
    )
    assert observed["reuse_dataset_batches"] is False
