from __future__ import annotations

import math
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    end_to_end as end_to_end_module,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end import (
    ALL_END_TO_END_METHODS,
    END_TO_END_METHODS,
    FAMILY_END_TO_END_METHODS,
    LITERATURE_END_TO_END_METHODS,
    SCALABLE_LITERATURE_END_TO_END_METHODS,
    PROTOCOL_FAMILY,
    STAGE2_SYSTEM_CONFIG_FIELDS,
    TIMING_SCOPE,
    EndToEndConfig,
    _run_efgp_method,
    _run_literature_method,
    _validate_config,
    choose_rpcholesky_landmarks,
    choose_uniform_landmarks,
    fit_restricted_krr,
    kernel_cross,
    predict_restricted_krr,
    preflight_end_to_end_resources,
    run_end_to_end_experiment,
    run_pipeline_once,
    summarize_pipeline_rows,
    validate_dataset_generation_provenance,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end_suite import (
    _load_completed_case,
    build_profile_plan,
    load_suite_config,
    materialize_robustness_plan,
    materialize_family_robustness_plan,
    require_complete_plan,
    select_target_regime,
)


def test_protocol_methods_are_complete_krr_pipelines() -> None:
    assert PROTOCOL_FAMILY == "end_to_end_krr"
    assert set(END_TO_END_METHODS) == {
        "nystrom-krr",
        "rpcholesky-krr",
        "efgp-standard-cg",
        "efgp-standard-jacobi",
        "efgp-standard-full-eig",
        "ours-binned-default",
    }
    assert "cg" not in END_TO_END_METHODS
    assert "nystrom" not in END_TO_END_METHODS
    assert "rpcholesky" not in END_TO_END_METHODS
    assert EndToEndConfig().inverse_max_size == 6000
    assert set(FAMILY_END_TO_END_METHODS) == {
        "efgp-standard-cg",
        "efgp-standard-full-eig",
        "ours-binned-inverse",
        "ours-binned-active-eig",
    }
    assert set(END_TO_END_METHODS).issubset(ALL_END_TO_END_METHODS)
    assert set(LITERATURE_END_TO_END_METHODS) == {
        "native-falkon-krr",
        "matern-rff-ridge",
        "randomized-nystrom-fourier-pcg",
        "ski-kissgp-krr",
        "original-krr-nystrom-pcg",
    }
    assert set(SCALABLE_LITERATURE_END_TO_END_METHODS) == {
        "native-falkon-krr",
        "matern-rff-ridge",
        "randomized-nystrom-fourier-pcg",
        "ski-kissgp-krr",
    }
    assert set(LITERATURE_END_TO_END_METHODS).issubset(ALL_END_TO_END_METHODS)


def test_explicit_family_route_config_is_validated() -> None:
    cfg = EndToEndConfig(
        methods=FAMILY_END_TO_END_METHODS,
        inverse_active_topk=2048,
        inverse_expected_active_box_size=2601,
        active_eig_topk=8192,
        active_eig_expected_active_box_size=10609,
        active_eig_rank=320,
    )
    _validate_config(cfg)
    with pytest.raises(ValueError, match="inverse_active_topk"):
        _validate_config(EndToEndConfig(inverse_active_topk=0))


def test_literature_baseline_config_is_validated() -> None:
    cfg = EndToEndConfig(
        methods=LITERATURE_END_TO_END_METHODS,
        native_falkon_nystrom_centers=512,
        native_falkon_maxiter=30,
        native_falkon_tolerance=1e-5,
        rff_num_features=512,
        fourier_nystrom_rank=256,
        ski_interpolation="linear",
        ski_grid_spacing=1.0 / 128.0,
    )
    _validate_config(cfg)
    with pytest.raises(ValueError, match="rff_num_features"):
        _validate_config(EndToEndConfig(rff_num_features=0))
    with pytest.raises(ValueError, match="matern-rff-ridge requires"):
        _validate_config(
            EndToEndConfig(
                methods=("matern-rff-ridge",),
                kernel_family="se",
            )
        )
    with pytest.raises(ValueError, match="ski_interpolation"):
        _validate_config(EndToEndConfig(ski_interpolation="nearest"))
    with pytest.raises(ValueError, match="fourier_nystrom_rank"):
        _validate_config(EndToEndConfig(fourier_nystrom_rank=0))
    with pytest.raises(ValueError, match="original_krr_nystrom_rank"):
        _validate_config(EndToEndConfig(original_krr_nystrom_rank=0))


@pytest.mark.parametrize(
    ("method", "adapter_name", "adapter_result", "expected_setup"),
    [
        (
            "native-falkon-krr",
            "run_native_falkon_krr",
            {
                "status": "converged",
                "pipeline_family": "literature_data_space_krr",
                "setup_seconds": 1.0,
                "solver_build_seconds": 2.0,
                "iterative_solve_seconds": 3.0,
                "train_total_seconds": 6.0,
                "prediction_seconds": 0.5,
                "test_rmse": 0.2,
                "test_mae": 0.1,
                "test_r2": 0.8,
                "iterations": 7,
                "citations": ["rudi2017falkon"],
                "timing_scope": "test",
                "backend": "numpy",
                "nystrom_centers": 11,
                "falkon_penalty": 0.01,
                "relative_residual": 1e-6,
                "converged": True,
                "implementation": "native_falkon_algorithm",
                "official_falkon_package": False,
            },
            1.0,
        ),
        (
            "matern-rff-ridge",
            "run_matern_rff_ridge",
            {
                "status": "ok",
                "pipeline_family": "literature_random_features_krr",
                "setup_seconds": 1.0,
                "feature_accumulation_seconds": 2.0,
                "solve_seconds": 3.0,
                "train_total_seconds": 6.0,
                "prediction_seconds": 0.5,
                "test_rmse": 0.2,
                "test_mae": 0.1,
                "test_r2": 0.8,
                "citations": ["rahimi2007random"],
                "timing_scope": "test",
                "backend": "numpy",
                "num_features": 13,
                "implementation": "native_streaming_rff",
            },
            3.0,
        ),
    ],
)
def test_literature_adapter_maps_to_pipeline_schema(
    monkeypatch,
    method,
    adapter_name,
    adapter_result,
    expected_setup,
) -> None:
    from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
        literature_baselines,
    )

    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end.build_gpu_backend_bundle",
        lambda config: SimpleNamespace(xp=np),
    )
    monkeypatch.setattr(
        literature_baselines,
        adapter_name,
        lambda *args, **kwargs: dict(adapter_result),
    )
    dataset = {
        "x": np.zeros((5, 2)),
        "y": np.zeros(5),
        "x_test": np.zeros((3, 2)),
        "y_test": np.zeros(3),
    }
    row = _run_literature_method(
        method,
        EndToEndConfig(methods=(method,)),
        dataset,
        repeat_idx=2,
        is_warmup=False,
    )
    assert row["status"] == adapter_result["status"]
    assert row["setup_seconds"] == expected_setup
    expected_solving = 5.0 if method == "native-falkon-krr" else 3.0
    assert row["solving_phase_seconds"] == expected_solving
    assert row["train_total_seconds"] == 6.0
    assert row["test_rmse"] == 0.2
    assert row["test_r2"] == 0.8


def test_ski_adapter_maps_gpu_result_without_renaming_linear_ski(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
        structured_kernel_interpolation as ski,
    )

    observed: dict[str, object] = {}

    def fake_run(x_train, y_train, x_test, y_test, config):
        observed["config"] = config
        return {
            "status": "ok",
            "implementation": "native_streamed_ski_krr_cupy",
            "setup_seconds": 4.0,
            "solving_phase_seconds": 5.0,
            "train_total_seconds": 9.0,
            "prediction_seconds": 0.5,
            "test_rmse": 0.2,
            "test_mae": 0.1,
            "test_r2": 0.8,
            "diagnostics": {
                "interpolation": "linear",
                "grid_spacing": 1.0 / 128.0,
                "grid_shape": [133, 133],
                "grid_size": 17_689,
                "original_inducing_relative_residual": 1e-8,
                "kronecker_product_used": False,
                "stores_full_interpolation_matrix": False,
                "cg_iterations": 17,
                "timing_scope": "streamed GPU setup and inducing CG",
                "backend": "cupy",
            },
        }

    monkeypatch.setattr(ski, "run_structured_kernel_interpolation", fake_run)
    cfg = EndToEndConfig(
        methods=("ski-kissgp-krr",),
        ski_interpolation="linear",
        ski_grid_spacing=1.0 / 128.0,
    )
    dataset = {
        "x": np.zeros((5, 2)),
        "y": np.zeros(5),
        "x_test": np.zeros((3, 2)),
        "y_test": np.zeros(3),
    }

    row = _run_literature_method(
        "ski-kissgp-krr", cfg, dataset, repeat_idx=0, is_warmup=False
    )

    assert observed["config"].backend == "cupy"
    assert observed["config"].interpolation == "linear"
    assert row["pipeline_family"] == "structured_kernel_interpolation_krr"
    assert row["train_total_seconds"] == 9.0
    assert row["effective_ski_grid_shape"] == [133, 133]
    assert row["ski_kronecker_product_used"] is False
    assert row["ski_strict_kissgp_cubic"] is False
    assert row["literature_citations"] == ["wilson2015kissgp"]
    assert row["iterations"] == 17


def test_original_krr_nystrom_adapter_keeps_exact_operator_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
        original_krr_nystrom as original,
    )

    observed: dict[str, object] = {}

    def fake_run(x_train, y_train, x_test, y_test, config):
        observed["config"] = config
        return {
            "status": "converged",
            "implementation": "exact_blocked_data_space_krr_column_nystrom_pcg",
            "pipeline_family": "literature_original_data_space_krr",
            "citations": ["frangella2023randomized"],
            "data_staging_seconds": 1.0,
            "preconditioner_setup_seconds": 2.0,
            "setup_seconds": 3.0,
            "solve_seconds": 4.0,
            "train_total_seconds": 7.0,
            "prediction_seconds": 0.5,
            "rmse": 0.2,
            "mae": 0.1,
            "r2": 0.8,
            "iterations": 11,
            "effective_nystrom_rank": 64,
            "true_relative_residual": 8e-4,
            "exact_matvec_count": 12,
            "kernel_pair_evaluations": 1_200,
            "operator_approximation": False,
            "solved_system": "original_data_space_K_plus_absolute_ridge_I",
            "timing_scope": "exact original KRR",
            "backend": "cupy",
        }

    monkeypatch.setattr(original, "run_original_krr_nystrom_pcg", fake_run)
    cfg = EndToEndConfig(
        methods=("original-krr-nystrom-pcg",),
        original_krr_nystrom_rank=64,
    )
    dataset = {
        "x": np.zeros((10, 2)),
        "y": np.zeros(10),
        "x_test": np.zeros((4, 2)),
        "y_test": np.zeros(4),
    }

    row = _run_literature_method(
        "original-krr-nystrom-pcg", cfg, dataset, repeat_idx=0, is_warmup=False
    )

    assert observed["config"].backend == "cupy"
    assert observed["config"].rank == 64
    assert row["setup_seconds"] == 1.0
    assert row["solver_build_seconds"] == 2.0
    assert row["iterative_solve_seconds"] == 4.0
    assert row["solving_phase_seconds"] == 6.0
    assert row["train_total_seconds"] == 7.0
    assert row["test_rmse"] == 0.2
    assert row["original_krr_operator_approximation"] is False
    assert row["original_krr_solved_system"].startswith("original_data_space")


def test_original_krr_full_scale_resource_gate_is_retained_in_pipeline_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
        original_krr_nystrom as original,
    )

    audit = {
        "exact_matvec_pairs": 10**14,
        "dense_kernel_matrix_bytes": 800_000_000_000_000,
        "prediction_pairs": 10**10,
        "preconditioner_factor_bytes": 10_240_000_000,
        "resource_preflight_before_backend": True,
    }

    def gated(*args, **kwargs):
        raise original.OriginalKRRResourceLimit("exact_matvec_pair_cap", audit)

    monkeypatch.setattr(original, "run_original_krr_nystrom_pcg", gated)
    dataset = {
        "x": np.zeros((2, 2)),
        "y": np.zeros(2),
        "x_test": np.zeros((1, 2)),
        "y_test": np.zeros(1),
        "n_test": 1,
        "source_n_train": 2,
        "metadata": {},
        "content_index_sha256": "content",
        "metadata_sha256": "metadata",
    }

    row = run_pipeline_once(
        "original-krr-nystrom-pcg",
        EndToEndConfig(methods=("original-krr-nystrom-pcg",)),
        dataset,
        repeat_idx=0,
        is_warmup=False,
    )

    assert row["status"] == "resource_limit"
    assert row["resource_limit_reason"] == "exact_matvec_pair_cap"
    assert row["original_krr_exact_matvec_pairs"] == 10**14
    assert row["original_krr_dense_kernel_matrix_bytes"] == 800_000_000_000_000
    assert row["original_krr_preconditioner_factor_bytes"] == 10_240_000_000
    assert row["resource_preflight_before_backend"] is True


def test_full_scale_original_krr_is_excluded_before_dataset_or_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight exclusion must not touch data or CUDA")

    monkeypatch.setattr(end_to_end_module, "load_end_to_end_dataset", forbidden)
    monkeypatch.setattr(
        end_to_end_module,
        "_probe_available_device_bytes_without_allocation",
        forbidden,
    )
    monkeypatch.setattr(end_to_end_module, "_release_gpu_allocator_cache", forbidden)
    cfg = EndToEndConfig(
        dataset_stem="unread_original_krr_n10m",
        n_train=10_000_000,
        max_test_rows=10_000,
        methods=("original-krr-nystrom-pcg",),
        warmup_repeats=0,
        measured_repeats=1,
        output_dir=str(tmp_path / "original"),
    )

    result = run_end_to_end_experiment(cfg)

    assert result["completion"]["dataset_loaded"] is False
    assert result["completion"]["gpu_work_launched"] is False
    assert result["completion"]["cuda_runtime_memory_queried"] is False
    assert result["completion"]["cuda_runtime_memory_query_attempted"] is False
    assert result["completion"]["cuda_runtime_memory_query_succeeded"] is False
    assert result["completion"]["resource_preflight_all_methods_excluded"] is True
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["status"] == "resource_limit"
    assert row["resource_required_bytes"] is None
    assert row["resource_preflight_before_dataset_load"] is True
    assert row["gpu_backend_initialized_for_method"] is False
    assert row["gpu_work_launched"] is False
    audit = json.loads((tmp_path / "original" / "resource_preflight.json").read_text())
    assert audit["all_methods_excluded"] is True
    assert audit["gpu_work_required"] is False


def test_cuda_memory_probe_failure_excludes_before_dataset_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("failed CUDA probe must stop before data/GPU work")

    monkeypatch.setattr(
        end_to_end_module,
        "_probe_available_device_bytes_without_allocation",
        lambda: None,
    )
    monkeypatch.setattr(end_to_end_module, "load_end_to_end_dataset", forbidden)
    monkeypatch.setattr(end_to_end_module, "_release_gpu_allocator_cache", forbidden)
    cfg = EndToEndConfig(
        dataset_stem="unread_cuda_probe_failure",
        n_train=10_000_000,
        methods=("ours-binned-inverse",),
        inverse_active_topk=512,
        inverse_expected_active_box_size=625,
        warmup_repeats=0,
        measured_repeats=1,
        output_dir=str(tmp_path / "cuda-probe-failure"),
    )

    result = run_end_to_end_experiment(cfg)

    row = result["rows"][0]
    assert row["status"] == "resource_limit"
    assert row["resource_limit_reason"] == "cuda_memory_probe_unavailable"
    assert row["cuda_runtime_memory_query_attempted"] is True
    assert row["cuda_runtime_memory_query_succeeded"] is False
    assert result["completion"]["dataset_loaded"] is False
    assert result["completion"]["gpu_work_launched"] is False
    assert result["completion"]["cuda_runtime_memory_query_attempted"] is True
    assert result["completion"]["cuda_runtime_memory_query_succeeded"] is False


def test_full_scale_rpcholesky_is_excluded_before_full_gpu_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("factor preflight must precede data/CUDA access")

    monkeypatch.setattr(end_to_end_module, "load_end_to_end_dataset", forbidden)
    monkeypatch.setattr(
        end_to_end_module,
        "_probe_available_device_bytes_without_allocation",
        forbidden,
    )
    monkeypatch.setattr(end_to_end_module, "_release_gpu_allocator_cache", forbidden)
    cfg = EndToEndConfig(
        dataset_stem="unread_rpcholesky_n300m",
        n_train=300_000_000,
        methods=("rpcholesky-krr",),
        rpcholesky_rank=256,
        warmup_repeats=0,
        measured_repeats=1,
        output_dir=str(tmp_path / "rpcholesky"),
    )

    result = run_end_to_end_experiment(cfg)

    row = result["rows"][0]
    assert row["status"] == "resource_limit"
    assert row["resource_limit_reason"] == "rpcholesky_factor_memory_cap"
    assert row["resource_required_bytes"] == (
        300_000_000 * 256 * 8 + 300_000_000 * 3 * 8
    )
    assert row["resource_audit"]["rpcholesky_factor_bytes"] == (
        300_000_000 * 256 * 8
    )
    assert result["completion"]["dataset_loaded"] is False
    assert result["completion"]["gpu_work_launched"] is False


def test_bq_preflight_uses_conservative_peak_and_available_memory() -> None:
    inverse = EndToEndConfig(
        n_train=300_000_000,
        methods=("ours-binned-inverse",),
        inverse_active_topk=12_288,
        inverse_expected_active_box_size=15_625,
        inverse_max_size=16_384,
    )
    audit = preflight_end_to_end_resources(
        inverse,
        available_device_bytes=40 * 2**30,
    )
    assert audit["methods"]["ours-binned-inverse"]["status"] == (
        "excluded_resource_limit"
    )
    assert audit["methods"]["ours-binned-inverse"]["resource_limit_reason"] == (
        "active_inverse_peak_memory_cap"
    )

    smaller = replace(
        inverse,
        inverse_active_topk=8_192,
        inverse_expected_active_box_size=10_609,
    )
    smaller_audit = preflight_end_to_end_resources(
        smaller,
        available_device_bytes=40 * 2**30,
    )
    assert smaller_audit["methods"]["ours-binned-inverse"]["status"] == "eligible"

    eigen = EndToEndConfig(
        n_train=300_000_000,
        methods=("ours-binned-active-eig",),
        active_eig_topk=35_721,
        active_eig_expected_active_box_size=35_721,
        active_eig_rank=448,
    )
    eigen_audit = preflight_end_to_end_resources(
        eigen,
        available_device_bytes=40 * 2**30,
    )
    assert eigen_audit["methods"]["ours-binned-active-eig"]["status"] == "eligible"

    factor_work = EndToEndConfig(
        n_train=10_000_000,
        methods=("ours-binned-inverse",),
        inverse_active_topk=19_000,
        inverse_expected_active_box_size=20_000,
        inverse_max_size=25_000,
        resource_preflight_gpu_memory_cap_bytes=1024**4,
    )
    factor_work_audit = preflight_end_to_end_resources(factor_work)
    factor_work_decision = factor_work_audit["methods"]["ours-binned-inverse"]
    assert factor_work_decision["status"] == "excluded_resource_limit"
    assert factor_work_decision["resource_limit_reason"] == "dense_inverse_work_cap"

    with pytest.raises(ValueError, match="strictly smaller"):
        preflight_end_to_end_resources(
            EndToEndConfig(
                methods=("ours-binned-active-eig",),
                active_eig_expected_active_box_size=448,
                active_eig_rank=448,
            )
        )


def _synthetic_generation_dataset(noise_std: float) -> dict[str, object]:
    stem = "synthetic_true_func_2d_ntrain100"
    return {
        "metadata": {
            "dataset_name": stem,
            "generation": {
                "noise_std": noise_std,
                "seed_train": 20260421,
                "seed_test": 1,
                "chunk_rows": 5_000_000,
                "target_function": "true_func_2d",
                "n_train": 100,
                "n_test": 25,
                "dim": 2,
            },
            "shapes": {"n_train": 100, "n_test": 25, "dim": 2},
            "y_transform": {"noise_std": noise_std},
        },
        "source_n_train": 100,
        "x": np.zeros((8, 2), dtype=np.float64),
        "content_index_sha256": "content-sha",
        "metadata_sha256": "metadata-sha",
    }


def test_synthetic_generation_provenance_accepts_noise_03_family() -> None:
    cfg = EndToEndConfig(
        dataset_stem="synthetic_true_func_2d_ntrain100",
        expected_dataset_noise_std=0.3,
        expected_dataset_seed_train=20260421,
        expected_dataset_seed_test=1,
        expected_dataset_generation_chunk_rows=5_000_000,
        expected_dataset_target_function="true_func_2d",
    )
    observed = validate_dataset_generation_provenance(
        cfg, _synthetic_generation_dataset(0.3)
    )
    assert observed["observed_dataset_noise_std"] == 0.3
    assert observed["dataset_content_index_sha256"] == "content-sha"


def test_synthetic_generation_provenance_rejects_noise_002_master() -> None:
    cfg = EndToEndConfig(
        dataset_stem="synthetic_true_func_2d_ntrain100",
        expected_dataset_noise_std=0.3,
        expected_dataset_seed_train=20260421,
        expected_dataset_seed_test=1,
        expected_dataset_generation_chunk_rows=5_000_000,
        expected_dataset_target_function="true_func_2d",
    )
    with pytest.raises(ValueError, match="noise_std"):
        validate_dataset_generation_provenance(
            cfg, _synthetic_generation_dataset(0.02)
        )


def test_efgp_pipeline_total_charges_score_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frozen active-box choice is timed work, not free preprocessing."""

    class FakeBackend:
        xp = np

    class FakeSystem:
        backend = FakeBackend()
        rhs_gpu = np.asarray([1.0])
        data_ctx = object()
        setup_seconds = 7.0
        manifest = {"M": 1, "mtot": 1}
        system_id = "fixed-system"

    fake_system = FakeSystem()
    fake_spec = type(
        "Spec",
        (),
        {"label": "default", "active_set": None, "btab_config": None},
    )()
    method_row = {
        "status": "ok",
        "selection_seconds": 2.0,
        "preconditioner_build_seconds": 3.0,
        # The fixed-A,b runner's compatibility field already includes selection.
        "build_seconds": 5.0,
        "solve_seconds": 5.0,
        "selection_rule": "frozen_score_topk_clamped_to_box_budget",
        "active_topk": 28,
        "box_size": 49,
        "rank": 16,
    }

    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end._fixed_config",
        lambda cfg, methods: object(),
    )
    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end._prepare_binned_system",
        lambda cfg, dataset: (fake_system, {}),
    )
    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end.fixed_ab.resolve_method_specs",
        lambda system, cfg: ([fake_spec], {}),
    )
    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end.fixed_ab.run_one_method",
        lambda *args, **kwargs: (method_row, np.asarray([1.0])),
    )
    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end.predict_v1",
        lambda *args, **kwargs: np.asarray([0.0]),
    )
    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end._sync",
        lambda xp: None,
    )

    row = _run_efgp_method(
        "ours-binned-default",
        EndToEndConfig(),
        {"x_test": np.asarray([[0.0]]), "y_test": np.asarray([0.0])},
        repeat_idx=0,
        is_warmup=False,
    )

    assert row["solver_build_seconds"] == 3.0
    assert row["solving_phase_seconds"] == 10.0
    assert row["train_total_seconds"] == 17.0
    assert row["active_selection_rule"] == method_row["selection_rule"]
    assert row["effective_active_topk"] == 28
    assert row["effective_active_box_size"] == 49
    assert row["effective_active_rank"] == 16
    assert row["capacity_adapted"] is True


def test_fourier_nystrom_pipeline_keeps_fourier_setup_and_reports_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        xp = np

    class FakeSystem:
        backend = FakeBackend()
        rhs_gpu = np.asarray([1.0 + 0.0j])
        data_ctx = object()
        setup_seconds = 4.0
        manifest = {"M": 1, "mtot": 1}
        system_id = "fourier-system"

    fake_spec = SimpleNamespace(
        label="fourier-nystrom-precond", active_set=None, btab_config=None
    )
    captured: dict[str, object] = {}

    def fake_fixed_config(cfg, methods):
        captured["methods"] = methods
        captured["rank"] = cfg.fourier_nystrom_rank
        captured["seed"] = cfg.fourier_nystrom_seed
        return object()

    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end._fixed_config",
        fake_fixed_config,
    )
    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end.fixed_ab.prepare_shared_system",
        lambda cfg, dataset_payload: FakeSystem(),
    )
    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end.fixed_ab.resolve_method_specs",
        lambda system, cfg: ([fake_spec], {}),
    )
    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end.fixed_ab.run_one_method",
        lambda *args, **kwargs: (
            {
                "status": "converged",
                "selection_seconds": 0.0,
                "preconditioner_build_seconds": 2.0,
                "solve_seconds": 3.0,
                "rank": 256,
            },
            np.asarray([1.0 + 0.0j]),
        ),
    )
    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end.predict_v1",
        lambda *args, **kwargs: np.asarray([0.25, -0.25]),
    )
    monkeypatch.setattr(
        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end._sync",
        lambda xp: None,
    )

    row = _run_efgp_method(
        "randomized-nystrom-fourier-pcg",
        EndToEndConfig(
            methods=("randomized-nystrom-fourier-pcg",),
            fourier_nystrom_rank=256,
            fourier_nystrom_seed=17,
        ),
        {
            "x_test": np.asarray([[0.0, 0.0], [1.0, 1.0]]),
            "y_test": np.asarray([0.25, -0.25]),
        },
        repeat_idx=1,
        is_warmup=False,
    )

    assert captured == {
        "methods": ("cg", "fourier-nystrom-precond"),
        "rank": 256,
        "seed": 17,
    }
    assert row["pipeline_family"] == "fourier_randomized_nystrom_pcg"
    assert row["setup_seconds"] == 4.0
    assert row["solver_build_seconds"] == 2.0
    assert row["iterative_solve_seconds"] == 3.0
    assert row["train_total_seconds"] == 9.0
    assert row["effective_fourier_nystrom_rank"] == 256
    assert row["literature_citations"] == ["frangella2023randomized"]
    assert row["test_rmse"] == 0.0


def test_mixed_pipeline_config_rejects_mismatched_regularization_scaling() -> None:
    with pytest.raises(ValueError, match="mismatched ridge scaling"):
        _validate_config(EndToEndConfig(regularization_convention="mean_loss"))


def test_kernel_cross_matches_closed_form_matern_three_halves() -> None:
    x = np.asarray([[0.0, 0.0], [0.3, 0.4]], dtype=np.float64)
    z = np.asarray([[0.0, 0.0]], dtype=np.float64)
    result = kernel_cross(
        np,
        x,
        z,
        family="matern",
        lengthscale=0.2,
        nu=1.5,
        variance=1.7,
        dtype=np.float64,
    ).reshape(-1)
    scaled = math.sqrt(3.0) * 0.5 / 0.2
    expected = np.asarray([1.7, 1.7 * (1.0 + scaled) * math.exp(-scaled)])
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


def test_streamed_restricted_krr_matches_dense_normal_equations() -> None:
    rng = np.random.default_rng(9)
    x = rng.uniform(size=(31, 2))
    y = np.sin(2.0 * np.pi * x[:, 0]) + 0.2 * x[:, 1]
    cfg = EndToEndConfig(
        methods=("nystrom-krr",),
        nystrom_rank=7,
        low_rank_chunk_size=6,
        low_rank_dtype="fp64",
        kernel_family="matern",
        nu=1.5,
        lengthscale=0.25,
        reg_lambda=0.07,
    )
    indices = choose_uniform_landmarks(len(x), 7, seed=4)
    coefficients, landmarks, diagnostics = fit_restricted_krr(np, x, y, indices, cfg)

    C = kernel_cross(
        np,
        x,
        x[indices],
        family=cfg.kernel_family,
        lengthscale=cfg.lengthscale,
        nu=cfg.nu,
        variance=cfg.variance,
        dtype=np.float64,
    )
    W = kernel_cross(
        np,
        x[indices],
        x[indices],
        family=cfg.kernel_family,
        lengthscale=cfg.lengthscale,
        nu=cfg.nu,
        variance=cfg.variance,
        dtype=np.float64,
    )
    dense_system = C.T @ C + cfg.reg_lambda * W
    # Reproduce the documented numerical jitter.
    dense_system += diagnostics["jitter"] * np.eye(len(indices))
    expected = np.linalg.solve(dense_system, C.T @ y)
    np.testing.assert_allclose(coefficients, expected, rtol=2e-11, atol=2e-11)
    predicted = predict_restricted_krr(np, x, landmarks, coefficients, cfg)
    np.testing.assert_allclose(predicted, C @ expected, rtol=2e-12, atol=2e-12)
    assert diagnostics["regularization_convention"] == "absolute"


def test_exact_rpcholesky_is_reproducible_and_resource_gated() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(40, 2))
    cfg = EndToEndConfig(
        methods=("rpcholesky-krr",),
        rpcholesky_rank=6,
        low_rank_chunk_size=11,
        low_rank_dtype="fp64",
        rpcholesky_max_factor_bytes=10_000_000,
    )
    first, diag_first = choose_rpcholesky_landmarks(
        np, x, cfg, rank=6, dtype=np.float64
    )
    second, diag_second = choose_rpcholesky_landmarks(
        np, x, cfg, rank=6, dtype=np.float64
    )
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == len(first)
    assert diag_first["selection_algorithm"] == "exact_simple_rpcholesky"
    assert diag_first["relative_trace_final"] <= 1.0
    assert diag_first == diag_second

    gated = EndToEndConfig(
        methods=("rpcholesky-krr",),
        rpcholesky_rank=6,
        rpcholesky_max_factor_bytes=1,
    )
    with pytest.raises(MemoryError, match="will record resource_limit") as exc_info:
        choose_rpcholesky_landmarks(np, x, gated, rank=6, dtype=np.float64)
    assert exc_info.value.required_bytes == 6 * len(x) * 8
    assert exc_info.value.effective_cap_bytes == 1


def test_rpcholesky_stops_cleanly_when_residual_is_exhausted() -> None:
    x = np.zeros((12, 2), dtype=np.float64)
    cfg = EndToEndConfig(
        methods=("rpcholesky-krr",),
        rpcholesky_rank=6,
        rpcholesky_max_factor_bytes=10_000_000,
    )
    pivots, diagnostics = choose_rpcholesky_landmarks(
        np, x, cfg, rank=6, dtype=np.float64
    )
    assert len(pivots) == 1
    assert diagnostics["effective_rank"] == 1
    assert diagnostics["relative_trace_final"] <= 1e-12


def test_summary_retains_speedup_for_usable_non_equivalent_method() -> None:
    cfg = EndToEndConfig(
        methods=(
            "nystrom-krr",
            "efgp-standard-full-eig",
            "ours-binned-default",
        ),
        measured_repeats=1,
        accuracy_relative_tolerance=0.01,
        accuracy_max_rmse=1.5,
        accuracy_min_r2=0.5,
    )
    common = {
        "protocol_family": PROTOCOL_FAMILY,
        "dataset_stem": "toy",
        "n_train": 100,
        "n_test": 20,
        "dim": 2,
        "kernel_family": "matern",
        "lengthscale": 0.1,
        "nu": 1.5,
        "reg_lambda": 0.1,
        "box_budget": 64,
        "repeat_idx": 0,
        "is_warmup": False,
        "status": "converged",
        "prediction_seconds": 0.1,
        "test_mae": 0.1,
        "test_r2": 0.9,
    }
    rows = [
        {
            **common,
            "method": "efgp-standard-full-eig",
            "setup_seconds": 4.0,
            "solver_build_seconds": 1.0,
            "iterative_solve_seconds": 5.0,
            "solving_phase_seconds": 6.0,
            "train_total_seconds": 10.0,
            "test_rmse": 1.0,
        },
        {
            **common,
            "method": "ours-binned-default",
            "setup_seconds": 1.0,
            "solver_build_seconds": 1.0,
            "iterative_solve_seconds": 2.0,
            "solving_phase_seconds": 3.0,
            "train_total_seconds": 4.0,
            "test_rmse": 1.005,
        },
        {
            **common,
            "method": "nystrom-krr",
            "setup_seconds": 1.0,
            "solver_build_seconds": 0.0,
            "iterative_solve_seconds": 1.0,
            "solving_phase_seconds": 1.0,
            "train_total_seconds": 2.0,
            "test_rmse": 1.2,
        },
    ]
    summary = {row["method"]: row for row in summarize_pipeline_rows(rows, cfg)}
    assert summary["ours-binned-default"]["usability_eligible"] is True
    assert summary["ours-binned-default"]["reference_equivalent"] is True
    assert summary["efgp-standard-full-eig"]["ours_total_speedup"] == 2.5
    assert summary["efgp-standard-full-eig"]["fourier_eps"] == cfg.fourier_eps
    assert (
        summary["efgp-standard-full-eig"]["setup_seconds_at_median_total"]
        + summary["efgp-standard-full-eig"]["solving_phase_seconds_at_median_total"]
        == summary["efgp-standard-full-eig"]["train_total_seconds_median"]
    )
    assert summary["nystrom-krr"]["usability_eligible"] is True
    assert summary["nystrom-krr"]["reference_equivalent"] is False
    assert summary["nystrom-krr"]["quality_qualified_performance_eligible"] is True
    assert summary["nystrom-krr"]["ours_total_speedup"] == 0.5
    assert summary["nystrom-krr"]["comparison_rmse_ratio_to_ours"] == pytest.approx(
        1.2 / 1.005
    )
    assert summary["nystrom-krr"]["comparison_rmse_delta_from_ours"] == pytest.approx(
        0.195
    )
    assert summary["nystrom-krr"]["ours_speedup_claim_eligible"] is True


def test_usability_requires_every_repeat_and_absolute_quality() -> None:
    cfg = EndToEndConfig(
        methods=("efgp-standard-full-eig", "ours-binned-default"),
        measured_repeats=3,
        accuracy_relative_tolerance=0.01,
        accuracy_max_rmse=1.1,
        accuracy_min_r2=0.5,
    )
    common = {
        "protocol_family": PROTOCOL_FAMILY,
        "dataset_stem": "toy",
        "dataset_family": "Toy",
        "n_train": 100,
        "n_test": 20,
        "dim": 2,
        "kernel_family": "matern",
        "lengthscale": 0.1,
        "nu": 1.5,
        "variance": 1.0,
        "reg_lambda": 0.1,
        "regularization_convention": "absolute",
        "fourier_eps": 1e-5,
        "box_budget": 64,
        "is_warmup": False,
        "status": "converged",
        "setup_seconds": 1.0,
        "solver_build_seconds": 0.25,
        "iterative_solve_seconds": 0.75,
        "solving_phase_seconds": 1.0,
        "train_total_seconds": 2.0,
        "prediction_seconds": 0.1,
        "test_mae": 0.1,
    }
    rows = []
    for repeat_idx in range(3):
        rows.append(
            {
                **common,
                "method": "efgp-standard-full-eig",
                "repeat_idx": repeat_idx,
                "test_rmse": 1.0,
                "test_r2": 0.8,
            }
        )
        rows.append(
            {
                **common,
                "method": "ours-binned-default",
                "repeat_idx": repeat_idx,
                "test_rmse": 2.0 if repeat_idx == 2 else 0.9,
                "test_r2": 0.1 if repeat_idx == 2 else 0.85,
            }
        )
    summary = {row["method"]: row for row in summarize_pipeline_rows(rows, cfg)}
    assert summary["efgp-standard-full-eig"]["usability_eligible"] is True
    assert summary["efgp-standard-full-eig"]["reference_equivalent"] is True
    assert summary["ours-binned-default"]["test_rmse_median"] == 0.9
    assert summary["ours-binned-default"]["usability_passed_repeats"] == 2
    assert summary["ours-binned-default"]["accuracy_passed_repeats"] == 2
    assert summary["ours-binned-default"]["usability_eligible"] is False
    assert summary["ours-binned-default"]["accuracy_eligible"] is False
    assert summary["ours-binned-default"]["reference_equivalent"] is False
    assert summary["ours-binned-default"]["performance_claim_eligible"] is False
    assert summary["ours-binned-default"]["ours_total_speedup"] == 1.0
    assert summary["ours-binned-default"]["ours_speedup_claim_eligible"] is False


def _rows_with_effective_active_set_diagnostics() -> tuple[EndToEndConfig, list[dict]]:
    cfg = EndToEndConfig(
        methods=("efgp-standard-full-eig", "ours-binned-default"),
        measured_repeats=2,
        allow_frozen_topk_capacity_adaptation=True,
    )
    common = {
        "protocol_family": PROTOCOL_FAMILY,
        "dataset_stem": "toy",
        "n_train": 100,
        "n_test": 20,
        "dim": 2,
        "kernel_family": "matern",
        "lengthscale": 0.05,
        "nu": 1.5,
        "reg_lambda": 0.1,
        "box_budget": 49,
        "configured_active_topk": 40,
        "configured_allow_frozen_topk_capacity_adaptation": True,
        "is_warmup": False,
        "status": "converged",
        "setup_seconds": 1.0,
        "solver_build_seconds": 0.25,
        "iterative_solve_seconds": 0.75,
        "solving_phase_seconds": 1.0,
        "train_total_seconds": 2.0,
        "prediction_seconds": 0.1,
        "test_rmse": 0.9,
        "test_mae": 0.1,
        "test_r2": 0.8,
    }
    rows = []
    for repeat_idx in range(2):
        rows.extend(
            [
                {
                    **common,
                    "method": "efgp-standard-full-eig",
                    "repeat_idx": repeat_idx,
                    "active_selection_rule": "full_eig",
                    "effective_active_topk": 64,
                    "effective_active_box_size": 64,
                    "effective_active_rank": 64,
                    "capacity_adapted": False,
                },
                {
                    **common,
                    "method": "ours-binned-default",
                    "repeat_idx": repeat_idx,
                    "active_selection_rule": (
                        "frozen_score_topk_clamped_to_box_budget"
                    ),
                    "effective_active_topk": 28,
                    "effective_active_box_size": 49,
                    "effective_active_rank": 16,
                    "capacity_adapted": True,
                },
            ]
        )
    return cfg, rows


def test_summary_propagates_effective_active_set_diagnostics() -> None:
    cfg, rows = _rows_with_effective_active_set_diagnostics()
    summary = {row["method"]: row for row in summarize_pipeline_rows(rows, cfg)}
    ours = summary["ours-binned-default"]
    assert ours["configured_allow_frozen_topk_capacity_adaptation"] is True
    assert ours["active_selection_rule"] == (
        "frozen_score_topk_clamped_to_box_budget"
    )
    assert ours["effective_active_topk"] == 28
    assert ours["effective_active_box_size"] == 49
    assert ours["effective_active_rank"] == 16
    assert ours["capacity_adapted"] is True


def test_summary_fails_closed_when_effective_active_set_changes() -> None:
    cfg, rows = _rows_with_effective_active_set_diagnostics()
    changed = [dict(row) for row in rows]
    next(
        row
        for row in changed
        if row["method"] == "ours-binned-default" and row["repeat_idx"] == 1
    )["effective_active_topk"] = 27
    with pytest.raises(
        RuntimeError,
        match="ours-binned-default changed effective_active_topk",
    ):
        summarize_pipeline_rows(changed, cfg)


def _target_rows(n_train: int, cg_iterations: int) -> list[dict[str, object]]:
    common: dict[str, object] = {
        "dataset_stem": f"synthetic_true_func_2d_ntrain{int(n_train)}",
        "n_train": n_train,
        "kernel_family": "matern",
        "lengthscale": 0.1,
        "nu": 1.5,
        "reg_lambda": 0.1,
        "fourier_eps": 1e-5,
        "configured_allow_frozen_topk_capacity_adaptation": False,
        "status": "ok",
        "usability_eligible": True,
        # The legacy reference-equivalence gate must not control target selection.
        "accuracy_eligible": False,
    }
    rows = [{**common, "method": method} for method in END_TO_END_METHODS]
    for row in rows:
        if row["method"] == "efgp-standard-cg":
            row["iterations_median"] = cg_iterations
    return rows


def test_target_selection_is_frozen_and_uses_largest_eligible_n() -> None:
    rows = [
        *_target_rows(10_000_000, 3500),
        *_target_rows(30_000_000, 5500),
        *_target_rows(100_000_000, 9000),
    ]
    selected = select_target_regime(rows)
    assert selected["n_train"] == 30_000_000
    assert selected["eligible_candidate_count"] == 2
    assert selected["rejected_candidate_count"] == 1
    assert selected["allow_frozen_topk_capacity_adaptation"] is False


def test_target_selection_uses_prospectively_declared_equal_n_priority() -> None:
    synthetic = _target_rows(10_000_000, 3500)
    winnebago = [
        {
            **row,
            "dataset_stem": "winnebago_master",
        }
        for row in _target_rows(10_000_000, 3500)
    ]
    selected = select_target_regime(
        [*synthetic, *winnebago],
        dataset_priority=(
            "winnebago_master",
            "synthetic_true_func_2d_ntrain10000000",
        ),
    )
    assert selected["dataset_stem"] == "winnebago_master"


def test_target_selection_accepts_declared_rpcholesky_resource_limit() -> None:
    rows = [*_target_rows(10_000_000, 4000), *_target_rows(30_000_000, 5000)]
    for row in rows:
        if row["method"] == "rpcholesky-krr" and row["n_train"] == 30_000_000:
            row["status"] = "resource_limit"
    selected = select_target_regime(rows)
    assert selected["n_train"] == 30_000_000
    assert selected["resource_limited_methods"] == ["rpcholesky-krr"]


def test_target_selection_fails_closed_for_undeclared_resource_limit() -> None:
    rows = _target_rows(10_000_000, 4000)
    next(row for row in rows if row["method"] == "efgp-standard-jacobi")[
        "status"
    ] = "resource_limit"
    with pytest.raises(RuntimeError, match="Do not cherry-pick"):
        select_target_regime(rows)


def test_suite_refuses_partial_plan_before_downstream_work() -> None:
    plan = [{"case_id": "case-a"}, {"case_id": "case-b"}]
    partial_index = [
        {
            "case_id": "case-a",
            "all_rows_present": True,
            "status": "claim_eligible_complete",
            "error_type": None,
        }
    ]
    with pytest.raises(RuntimeError, match="refusing target selection/downstream"):
        require_complete_plan(plan, partial_index, phase="test scale")

    complete_index = [
        {
            "case_id": case_id,
            "all_rows_present": True,
            "status": "complete_with_resource_limits",
            "error_type": None,
        }
        for case_id in ("case-a", "case-b")
    ]
    require_complete_plan(plan, complete_index, phase="test scale")


def test_shipped_suite_covers_10m_to_300m_and_materializes_robustness(
    tmp_path: Path,
) -> None:
    suite = load_suite_config()
    scale = build_profile_plan(
        suite,
        "scale_10m_300m",
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    assert {item["config"].n_train for item in scale} == {
        10_000_000,
        30_000_000,
        100_000_000,
        300_000_000,
    }
    assert {item["dataset_family"] for item in scale} == {
        "Synthetic",
        "Winnebago",
    }
    expected_synthetic_stems = {
        n_train: f"synthetic_true_func_2d_ntrain{n_train}"
        for n_train in (10_000_000, 30_000_000, 100_000_000, 300_000_000)
    }
    assert {
        int(item["config"].n_train): item["config"].dataset_stem
        for item in scale
        if item["dataset_family"] == "Synthetic"
    } == expected_synthetic_stems
    assert set(expected_synthetic_stems.values()).issubset(
        suite["target_selection"]["dataset_priority"]
    )
    assert suite["base"]["dataset_stem"] == "synthetic_true_func_2d_ntrain10000000"
    assert all(
        item["config"].allow_frozen_topk_capacity_adaptation is False
        for item in scale
    )
    target = select_target_regime(_target_rows(10_000_000, 4000))
    robust = materialize_robustness_plan(
        suite,
        target,
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    configs = [item["config"] for item in robust]
    assert {cfg.reg_lambda for cfg in configs} >= {0.01, 0.1, 1.0}
    assert {cfg.lengthscale for cfg in configs} >= {0.05, 0.1, 0.2}
    assert {cfg.box_budget for cfg in configs} >= {4096, 8192, 16384}
    assert {cfg.dataset_stem for cfg in configs} >= {
        "synthetic_true_func_2d_ntrain10000000",
        "USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain10000000",
    }
    adaptive = [
        cfg
        for cfg in configs
        if cfg.parameter_selection_policy == "budget_adaptive_score_rule"
    ]
    frozen = [cfg for cfg in configs if cfg not in adaptive]
    assert {cfg.box_budget for cfg in adaptive} == {4096, 8192, 16384}
    assert all(cfg.active_topk is None for cfg in adaptive)
    assert all(cfg.expected_active_box_size is None for cfg in configs)
    assert all(cfg.allow_frozen_topk_capacity_adaptation is True for cfg in configs)
    assert all(
        cfg.parameter_selection_policy
        == "historical_selected_transfer_no_current_scan"
        for cfg in frozen
    )


def test_shipped_suite_freezes_archived_full_eig_and_ours_winners(
    tmp_path: Path,
) -> None:
    suite = load_suite_config()
    scale = build_profile_plan(
        suite,
        "scale_10m_300m",
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    observed = {
        item["case_id"]: (
            item["config"].rank,
            item["config"].full_eig_rank,
            item["config"].active_topk,
            item["config"].expected_active_box_size,
        )
        for item in scale
    }
    assert observed == {
        "synthetic_matern_n10m": (256, 256, 4096, 5329),
        "synthetic_matern_n30m": (320, 256, 8192, 10609),
        "synthetic_matern_n100m": (320, 320, 35721, 35721),
        "synthetic_matern_n300m": (320, 384, 35721, 35721),
        "winnebago_matern_n10m": (320, 128, 8192, 10609),
        "winnebago_matern_n30m": (256, 128, 4096, 5329),
        "winnebago_matern_n100m": (320, 256, 35721, 35721),
        "winnebago_matern_n300m": (384, 320, 35721, 35721),
    }
    assert all(
        item["config"].parameter_selection_policy
        == "historical_selected_transfer_no_current_scan"
        for item in scale
    )
    assert all("0D6827265" in item["config"].parameter_source for item in scale)
    synthetic_parameter_sources = [
        item["config"].parameter_source
        for item in scale
        if item["dataset_family"] == "Synthetic"
    ]
    assert synthetic_parameter_sources
    assert all("generated diagnostic table" in source for source in synthetic_parameter_sources)
    assert all("matching noise=0.3" in source for source in synthetic_parameter_sources)
    assert all("old timings excluded" in source for source in synthetic_parameter_sources)
    synthetic_configs = [
        item["config"] for item in scale if item["dataset_family"] == "Synthetic"
    ]
    assert all(cfg.expected_dataset_noise_std == 0.3 for cfg in synthetic_configs)
    assert all(
        cfg.expected_dataset_seed_train == 20260421 for cfg in synthetic_configs
    )
    assert all(cfg.expected_dataset_seed_test == 1 for cfg in synthetic_configs)
    assert all(
        cfg.expected_dataset_generation_chunk_rows == 5_000_000
        for cfg in synthetic_configs
    )
    assert all(
        cfg.expected_dataset_target_function == "true_func_2d"
        for cfg in synthetic_configs
    )
    assert suite["base"]["inverse_max_size"] == 6000
    assert suite["base"]["allow_frozen_topk_capacity_adaptation"] is False
    assert suite["stage2_fixed_ab"]["inverse_max_size"] == 16384
    assert suite["stage2_fixed_ab"]["default_inverse_max_size"] == 6000


def test_shipped_family_profiles_separate_inverse_and_eigenpair_branches(
    tmp_path: Path,
) -> None:
    suite = load_suite_config()
    family_scale = build_profile_plan(
        suite,
        "family_scale_10m_300m",
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    assert len(family_scale) == 8
    assert all(item["config"].methods == FAMILY_END_TO_END_METHODS for item in family_scale)
    by_case = {item["case_id"]: item["config"] for item in family_scale}
    assert by_case["synthetic_matern_n30m"].inverse_active_topk == 2048
    assert by_case["synthetic_matern_n30m"].active_eig_topk == 8192
    assert by_case["synthetic_matern_n30m"].active_eig_rank == 320
    assert by_case["winnebago_matern_n30m"].inverse_active_topk == 512
    assert by_case["winnebago_matern_n30m"].active_eig_topk == 35721
    assert by_case["winnebago_matern_n30m"].active_eig_rank == 128

    kernel = build_profile_plan(
        suite,
        "family_kernel_at_30m",
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    assert {(item["dataset_family"], item["config"].kernel_family) for item in kernel} == {
        ("Synthetic", "se"),
        ("Synthetic", "matern"),
        ("Winnebago", "se"),
        ("Winnebago", "matern"),
    }
    assert all(item["config"].methods == FAMILY_END_TO_END_METHODS for item in kernel)


def test_matern_family_parameter_sweep_plan_size_coverage_and_repeats(
    tmp_path: Path,
) -> None:
    suite = load_suite_config()
    plan = build_profile_plan(
        suite,
        "matern_family_parameter_sweep_10m_300m",
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )

    assert len(plan) == 144
    assert len({item["case_id"] for item in plan}) == len(plan)
    assert {
        (item["dataset_family"], item["config"].n_train) for item in plan
    } == {
        (family, n_train)
        for family in ("Synthetic", "Winnebago")
        for n_train in (10_000_000, 30_000_000, 100_000_000, 300_000_000)
    }
    method_counts = {
        method: sum(item["config"].methods == (method,) for item in plan)
        for method in ("ours-binned-inverse", "ours-binned-active-eig")
    }
    assert method_counts == {
        "ours-binned-inverse": 32,
        "ours-binned-active-eig": 112,
    }
    assert all(len(item["config"].methods) == 1 for item in plan)
    assert all(item["config"].kernel_family == "matern" for item in plan)
    assert all(item["config"].warmup_repeats == 1 for item in plan)
    assert all(item["config"].measured_repeats == 3 for item in plan)
    assert all(
        item["config"].allow_frozen_topk_capacity_adaptation is False
        for item in plan
    )
    assert all(
        item["config"].parameter_selection_policy
        == "predeclared_matern_two_family_parameter_sweep"
        for item in plan
    )


def test_literature_baseline_profiles_are_executable_and_three_repeat(
    tmp_path: Path,
) -> None:
    suite = load_suite_config()
    pilot = build_profile_plan(
        suite,
        "literature_baseline_pilot_10m",
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    final = build_profile_plan(
        suite,
        "literature_baselines_300m",
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    assert len(pilot) == 18
    assert len(final) == 8
    assert {item["dataset_family"] for item in pilot} == {
        "Synthetic",
        "Winnebago",
    }
    assert all(item["config"].n_train == 10_000_000 for item in pilot)
    assert all(len(item["config"].methods) == 1 for item in pilot)
    assert all(item["config"].warmup_repeats == 1 for item in pilot)
    assert all(item["config"].measured_repeats == 3 for item in pilot)
    assert {
        item["config"].native_falkon_nystrom_centers
        for item in pilot
        if item["config"].methods == ("native-falkon-krr",)
    } == {64, 128}
    assert {
        item["config"].rff_num_features
        for item in pilot
        if item["config"].methods == ("matern-rff-ridge",)
    } == {128, 256}
    assert {
        item["config"].fourier_nystrom_rank
        for item in pilot
        if item["config"].methods == ("randomized-nystrom-fourier-pcg",)
    } == {128, 256, 512}
    assert {
        item["config"].ski_grid_spacing
        for item in pilot
        if item["config"].methods == ("ski-kissgp-krr",)
    } == {1.0 / 64.0, 1.0 / 128.0}
    assert all(
        item["config"].ski_interpolation == "linear"
        for item in pilot
        if item["config"].methods == ("ski-kissgp-krr",)
    )
    assert all(
        item["config"].native_falkon_maxiter == 8
        and item["config"].native_falkon_tolerance == 1e-3
        for item in pilot
        if item["config"].methods == ("native-falkon-krr",)
    )
    assert all(item["config"].n_train == 300_000_000 for item in final)
    assert all(
        len(item["config"].methods) == 1
        and item["config"].methods[0] in LITERATURE_END_TO_END_METHODS
        for item in final
    )
    assert all(item["config"].warmup_repeats == 1 for item in final)
    assert all(item["config"].measured_repeats == 3 for item in final)
    assert all(item["config"].native_falkon_nystrom_centers == 128 for item in final)
    assert all(item["config"].rff_num_features == 256 for item in final)
    assert all(item["config"].fourier_nystrom_rank == 256 for item in final)
    assert all(item["config"].ski_interpolation == "linear" for item in final)
    assert all(item["config"].ski_grid_spacing == 1.0 / 128.0 for item in final)
    assert all(item["config"].native_falkon_train_chunk_size == 250_000 for item in final)
    assert all(item["config"].rff_train_chunk_size == 250_000 for item in final)


def test_original_full_scale_profile_is_single_pre_dataset_exclusion(
    tmp_path: Path,
) -> None:
    plan = build_profile_plan(
        load_suite_config(),
        "original_krr_full_scale_resource_audit",
        dataset_dir=str(tmp_path / "unread-data"),
        output_root=tmp_path / "out",
    )

    assert len(plan) == 4
    assert {
        (item["dataset_family"], int(item["config"].n_train)) for item in plan
    } == {
        (family, n_train)
        for family in ("Synthetic", "Winnebago")
        for n_train in (10_000_000, 300_000_000)
    }
    assert all(
        item["config"].methods == ("original-krr-nystrom-pcg",)
        and item["config"].warmup_repeats == 0
        and item["config"].measured_repeats == 1
        and item["config"].parameter_selection_policy
        == "single_pre_dataset_resource_exclusion_no_execution_selection"
        for item in plan
    )


def test_matern_family_parameter_sweep_candidates_and_box_assertions(
    tmp_path: Path,
) -> None:
    suite = load_suite_config()
    profile = suite["profiles"]["matern_family_parameter_sweep_10m_300m"]
    sweep = profile["family_parameter_sweep"]
    topk_to_box = {
        int(topk): int(box_size)
        for topk, box_size in sweep["topk_to_expected_box_size"].items()
    }
    assert topk_to_box == {
        512: 625,
        728: 961,
        1024: 1369,
        2048: 2601,
        4096: 5329,
        8192: 10609,
        12288: 15625,
        16384: 21025,
        20480: 25921,
        25720: 32761,
        35721: 35721,
    }
    plan = build_profile_plan(
        suite,
        "matern_family_parameter_sweep_10m_300m",
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )

    observed: dict[tuple[str, int, str], set[tuple[int, ...]]] = {}
    for item in plan:
        cfg = item["config"]
        method = cfg.methods[0]
        key = (str(item["dataset_family"]), int(cfg.n_train), method)
        if method == "ours-binned-inverse":
            assert cfg.inverse_active_topk == cfg.active_topk
            assert cfg.inverse_expected_active_box_size == cfg.expected_active_box_size
            assert cfg.active_eig_topk is None
            assert cfg.active_eig_expected_active_box_size is None
            assert cfg.active_eig_rank is None
            candidate = (int(cfg.active_topk), int(cfg.expected_active_box_size))
        else:
            assert method == "ours-binned-active-eig"
            assert cfg.active_eig_topk == cfg.active_topk
            assert cfg.active_eig_expected_active_box_size == cfg.expected_active_box_size
            assert cfg.active_eig_rank == cfg.rank
            assert cfg.inverse_active_topk is None
            assert cfg.inverse_expected_active_box_size is None
            candidate = (
                int(cfg.active_topk),
                int(cfg.expected_active_box_size),
                int(cfg.active_eig_rank),
            )
            assert candidate[2] <= candidate[1]
        assert topk_to_box[candidate[0]] == candidate[1]
        assert (
            f"asserted topk={candidate[0]}->|B|={candidate[1]}"
            in cfg.parameter_source
        )
        observed.setdefault(key, set()).add(candidate)

    small_inverse = {
        (512, 625),
        (728, 961),
        (1024, 1369),
        (2048, 2601),
        (4096, 5329),
    }
    high_inverse = {(4096, 5329), (8192, 10609), (12288, 15625)}
    small_active = {
        (2048, 2601, 192),
        (4096, 5329, 192),
        (4096, 5329, 256),
        *((8192, 10609, rank) for rank in (192, 256, 320)),
        *((16384, 21025, rank) for rank in (192, 256, 320)),
        *((35721, 35721, rank) for rank in (128, 192, 256, 320)),
    }
    high_active = {
        (topk, topk_to_box[topk], rank)
        for topk in (20480, 25720, 35721)
        for rank in (192, 256, 320, 384, 448)
    }
    for family in ("Synthetic", "Winnebago"):
        for n_train in (10_000_000, 30_000_000):
            assert observed[(family, n_train, "ours-binned-inverse")] == small_inverse
            assert observed[(family, n_train, "ours-binned-active-eig")] == small_active
        for n_train in (100_000_000, 300_000_000):
            assert observed[(family, n_train, "ours-binned-inverse")] == high_inverse
            assert observed[(family, n_train, "ours-binned-active-eig")] == high_active

    source_winners = {
        (str(case["dataset_family"]), int(case["n_train"])): {
            "ours-binned-inverse": {
                (
                    int(case["inverse_active_topk"]),
                    int(case["inverse_expected_active_box_size"]),
                )
            },
            "ours-binned-active-eig": {
                (
                    int(case["active_eig_topk"]),
                    int(case["active_eig_expected_active_box_size"]),
                    int(case["active_eig_rank"]),
                )
            },
        }
        for case in suite["profiles"]["scale_10m_300m"]["cases"]
    }
    for (family, n_train), winners in source_winners.items():
        for method, candidates in winners.items():
            assert candidates <= observed[(family, n_train, method)]


def test_family_robustness_freezes_both_family_configs(tmp_path: Path) -> None:
    suite = load_suite_config()
    target = select_target_regime(_target_rows(30_000_000, 4000))
    target.update({
        "inverse_active_topk": 2048,
        "inverse_expected_active_box_size": 2601,
        "active_eig_topk": 8192,
        "active_eig_expected_active_box_size": 10609,
        "active_eig_rank": 320,
    })
    plan = materialize_family_robustness_plan(
        suite,
        target,
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    assert plan
    assert all(item["config"].methods == FAMILY_END_TO_END_METHODS for item in plan)
    budget_rows = [
        item for item in plan
        if any(str(axis).startswith("box_budget_") for axis in item["robustness_axes"])
    ]
    frozen_rows = [item for item in plan if item not in budget_rows]
    assert all(item["config"].inverse_active_topk is None for item in budget_rows)
    assert all(item["config"].active_eig_topk is None for item in budget_rows)
    assert all(item["config"].inverse_active_topk == 2048 for item in frozen_rows)
    assert all(item["config"].active_eig_topk == 8192 for item in frozen_rows)
    assert all(item["config"].active_eig_rank == 320 for item in plan)


def test_robustness_uses_exact_synthetic_and_winnebago_at_selected_n(tmp_path: Path) -> None:
    suite = load_suite_config()
    datasets = {
        item["dataset_family"]: item
        for item in suite["profiles"]["robustness_at_selected_target"]["datasets"]
    }
    assert datasets["Synthetic"]["dataset_stems_by_n_train"] == {
        "10000000": "synthetic_true_func_2d_ntrain10000000",
        "30000000": "synthetic_true_func_2d_ntrain30000000",
        "100000000": "synthetic_true_func_2d_ntrain100000000",
        "300000000": "synthetic_true_func_2d_ntrain300000000",
    }
    assert "dataset_stem" not in datasets["Synthetic"]
    target = select_target_regime(_target_rows(30_000_000, 4000))
    robust = materialize_robustness_plan(
        suite,
        target,
        dataset_dir=str(tmp_path / "data"),
        output_root=tmp_path / "out",
    )
    winnebago = [item for item in robust if item.get("dataset_family") == "Winnebago"]
    assert len(winnebago) == 1
    assert winnebago[0]["config"].n_train == 30_000_000
    assert winnebago[0]["config"].dataset_stem.endswith("ntrain30000000")
    assert winnebago[0]["config"].accuracy_max_rmse == 0.15
    synthetic = [
        item
        for item in robust
        if "dataset_synthetic" in item.get("robustness_axes", [])
    ]
    assert len(synthetic) == 1
    assert synthetic[0]["config"].n_train == 30_000_000
    assert (
        synthetic[0]["config"].dataset_stem
        == "synthetic_true_func_2d_ntrain30000000"
    )


def test_resume_reuses_resource_outcome_but_not_execution_error(tmp_path: Path) -> None:
    cfg = EndToEndConfig(
        methods=("rpcholesky-krr",),
        output_dir=str(tmp_path),
    )
    config_payload = asdict(cfg)
    config_payload["methods"] = list(cfg.methods)
    (tmp_path / "experiment_config.json").write_text(
        json.dumps(config_payload), encoding="utf-8"
    )
    summary = {
        "method": "rpcholesky-krr",
        "status": "resource_limit",
        "timing_scope": TIMING_SCOPE,
        **{field: getattr(cfg, field) for field in STAGE2_SYSTEM_CONFIG_FIELDS},
        "accuracy_relative_tolerance": cfg.accuracy_relative_tolerance,
        "accuracy_max_rmse": cfg.accuracy_max_rmse,
        "accuracy_min_r2": cfg.accuracy_min_r2,
        "expected_measured_repeats": cfg.measured_repeats,
        "accuracy_evaluated_repeats": 0,
        "accuracy_passed_repeats": 0,
        "usability_evaluated_repeats": 0,
        "usability_passed_repeats": 0,
        "usability_eligible": False,
        "execution_eligible": False,
        "quality_qualified_performance_eligible": False,
        "reference_evaluated_repeats": 0,
        "reference_equivalent_repeats": 0,
        "reference_equivalent": False,
        "setup_seconds_at_median_total": None,
        "solving_phase_seconds_at_median_total": None,
    }
    (tmp_path / "pipeline_summary.json").write_text(
        json.dumps([summary]), encoding="utf-8"
    )
    (tmp_path / "pipeline_runs.csv").write_text(
        "method,status\nrpcholesky-krr,resource_limit\n", encoding="utf-8"
    )
    completion = {
        "protocol_family": PROTOCOL_FAMILY,
        "timing_scope": TIMING_SCOPE,
        "methods": ["rpcholesky-krr"],
        "artifact_complete": True,
        "all_rows_present": True,
        "error_methods": [],
    }
    completion_path = tmp_path / "run_complete.json"
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    assert _load_completed_case(cfg) is not None

    completion["error_methods"] = ["rpcholesky-krr"]
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    assert _load_completed_case(cfg) is None
