from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from efgp_eigenpro_py.discretization import basis_weights, choose_grid_params
from efgp_eigenpro_py.gpu.box_toeplitz_active_block import active_set as active_set_module
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.active_set import score_rank_order
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
    benchmark as benchmark_module,
)
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark import (
    ControlledConfig,
    PreparedSystem,
    _load_dataset,
    _normalize_n_train,
    _validate_config,
    pairwise_comparisons,
    resolve_method_specs,
    resolve_score_box_rule,
    summarize_rows,
    system_component_fingerprints,
    system_fingerprint,
)
from efgp_eigenpro_py.kernels import make_matern


def _fake_system() -> PreparedSystem:
    mtot = 7
    backend = SimpleNamespace(xp=np, fft=np.fft)
    xtxcol = np.zeros(2 * mtot - 1, dtype=np.complex128)
    xtxcol[mtot - 1] = 100.0
    weights = np.array([0.1, 0.3, 1.0, 2.0, 1.0, 0.3, 0.1])
    data_ctx = SimpleNamespace(
        xtxcol_gpu=xtxcol,
        gf_gpu=np.fft.fftn(xtxcol),
        weights_gpu_flat=weights,
        weights_np_flat=weights.copy(),
        rhs_gpu=np.arange(mtot, dtype=np.complex128),
        meta={"mtot": mtot, "dim": 1},
    )
    fingerprint = system_fingerprint(data_ctx, 0.1)
    return PreparedSystem(
        backend=backend,
        data_ctx=data_ctx,
        rhs_gpu=data_ctx.rhs_gpu,
        reg_lambda=0.1,
        setup_seconds=1.0,
        system_id=fingerprint,
        manifest={"M": mtot, "mtot": mtot, "dim": 1},
    )


def _expanded_lengthscale_system() -> PreparedSystem:
    kernel = make_matern(lengthscale=0.05, nu=1.5, dim=2, variance=1.0)
    grid = choose_grid_params(kernel, 1e-5, L=1.0, l2scaled=True)
    weights = np.ascontiguousarray(
        basis_weights(kernel, grid.xis, grid.h).reshape(-1)
    )
    assert grid.mtot == 291
    assert weights.size == 84_681

    # _active_from_cfg only reads the lag-grid center to obtain gamma.
    xtxcol = np.zeros((2 * grid.mtot - 1,) * 2, dtype=np.float64)
    xtxcol[(grid.mtot - 1,) * 2] = 1.0
    data_ctx = SimpleNamespace(
        xtxcol_gpu=xtxcol,
        weights_np_flat=weights,
        weights_gpu_flat=weights,
        meta={"mtot": grid.mtot, "dim": 2},
    )
    return PreparedSystem(
        backend=SimpleNamespace(xp=np, fft=np.fft),
        data_ctx=data_ctx,
        rhs_gpu=np.zeros(weights.size, dtype=np.complex128),
        reg_lambda=0.1,
        setup_seconds=0.0,
        system_id="expanded-lengthscale-fixture",
        manifest={"M": weights.size, "mtot": grid.mtot, "dim": 2},
    )


def test_formal_default_methods_are_fixed_system_controls() -> None:
    config = ControlledConfig()
    assert config.methods == ("cg", "jacobi", "default", "full-eig")
    assert config.inverse_max_size == 6000


def test_new_configs_reject_ambiguous_krr_method_names_but_artifacts_can_load() -> None:
    legacy = ControlledConfig(methods=("cg", "nystrom"), measured_repeats=5)

    with pytest.raises(ValueError, match="ambiguous fixed-system method"):
        _validate_config(legacy)

    _validate_config(legacy, allow_legacy_method_names=True)


def test_fourier_adaptations_use_explicit_names_and_roles() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=(
            "cg",
            "fourier-nystrom-precond",
            "fourier-rpcholesky-precond",
        ),
        nystrom_rank=2,
        rpcholesky_rank=2,
        measured_repeats=5,
    )

    _validate_config(cfg)
    specs, _ = resolve_method_specs(system, cfg)
    adaptations = [spec for spec in specs if spec.label != "cg"]

    assert [spec.kind for spec in adaptations] == [
        "fourier-nystrom-precond",
        "fourier-rpcholesky-precond",
    ]
    assert all(
        spec.result_role == "exploratory_fourier_adaptation"
        for spec in adaptations
    )


def test_score_rule_uses_only_declared_memory_cap() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "default", "full-eig"),
        score_tau=1.0,
        box_budget=3,
        inverse_max_size=3,
        rank=2,
        measured_repeats=5,
    )

    rule = resolve_score_box_rule(system, cfg)
    specs, _ = resolve_method_specs(system, cfg)

    assert rule.raw_tau_box_size > 3
    assert int(rule.active.box_idx.size) <= 3
    assert rule.selection_rule == "score_ranked_memory_capped_box"
    default = next(spec for spec in specs if spec.label == "default")
    assert default.kind == "active-inverse"


def test_frozen_topk_and_method_specific_full_eig_rank_are_preserved() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "default", "full-eig"),
        active_topk=3,
        box_budget=system.manifest["M"],
        inverse_max_size=1,
        rank=2,
        full_eig_rank=4,
        measured_repeats=5,
        parameter_selection_policy="frozen_original_best",
        parameter_source="paper_table1_selected.csv",
    )

    specs, rule = resolve_method_specs(system, cfg)

    assert rule.selection_rule == "frozen_score_topk"
    assert rule.active.active_mode == "topk"
    assert rule.active.active_topk == 3
    assert next(spec for spec in specs if spec.label == "default").rank == 2
    assert next(spec for spec in specs if spec.label == "full-eig").rank == 4


def test_frozen_topk_checks_declared_observed_box_size() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "default"),
        active_topk=3,
        expected_active_box_size=system.manifest["M"] + 1,
        box_budget=system.manifest["M"],
        rank=2,
        measured_repeats=5,
    )

    with pytest.raises(ValueError, match="provenance check failed"):
        resolve_score_box_rule(system, cfg)


def test_frozen_topk_clamps_to_a_smaller_robustness_grid_without_rescanning() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "default"),
        active_topk=20,
        expected_active_box_size=None,
        box_budget=20,
        inverse_max_size=1,
        rank=2,
        measured_repeats=5,
        parameter_selection_policy="historical_selected_transfer_no_current_scan",
    )

    rule = resolve_score_box_rule(system, cfg)

    assert rule.config.active_topk == system.manifest["M"]
    assert rule.selection_rule == "frozen_score_topk_clamped_to_grid"


def test_frozen_topk_clamps_to_box_budget_on_an_expanded_robustness_grid() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "default"),
        # The two highest scores occupy the center and one neighbour.  Their
        # centered enclosing box has size three, which exceeds this fixed cap.
        active_topk=2,
        expected_active_box_size=None,
        allow_frozen_topk_capacity_adaptation=True,
        box_budget=2,
        inverse_max_size=1,
        rank=2,
        measured_repeats=5,
        parameter_selection_policy="historical_selected_transfer_no_current_scan",
    )

    rule = resolve_score_box_rule(system, cfg)

    assert rule.raw_tau_box_size == 3
    assert rule.config.active_topk == 1
    assert rule.active.box_idx.size == 1
    assert rule.selection_rule == "frozen_score_topk_clamped_to_box_budget"


def test_frozen_topk_combines_grid_and_box_budget_capacity_clamps() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "default"),
        active_topk=20,
        expected_active_box_size=None,
        allow_frozen_topk_capacity_adaptation=True,
        box_budget=2,
        inverse_max_size=1,
        rank=2,
        measured_repeats=5,
    )

    rule = resolve_score_box_rule(system, cfg)

    assert rule.config.active_topk == 1
    assert rule.selection_rule == (
        "frozen_score_topk_clamped_to_grid_and_box_budget"
    )


def test_capacity_clamp_computes_one_deterministic_score_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _fake_system()
    calls = 0
    original = active_set_module.score_rank_order

    def counted(rho: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(rho)

    monkeypatch.setattr(active_set_module, "score_rank_order", counted)
    rule = resolve_score_box_rule(
        system,
        ControlledConfig(
            methods=("default",),
            active_topk=2,
            allow_frozen_topk_capacity_adaptation=True,
            box_budget=2,
            inverse_max_size=1,
            rank=2,
            measured_repeats=5,
        ),
    )

    assert rule.config.active_topk == 1
    assert calls == 1


def test_score_order_uses_flat_index_as_the_tie_break() -> None:
    np.testing.assert_array_equal(
        score_rank_order(np.array([1.0, 2.0, 2.0, 1.0])),
        np.array([1, 2, 0, 3]),
    )


def test_real_lengthscale_0p05_frozen_topk_is_capped_without_rescanning() -> None:
    system = _expanded_lengthscale_system()
    cfg = ControlledConfig(
        methods=("default",),
        rank=320,
        active_topk=35_721,
        expected_active_box_size=None,
        allow_frozen_topk_capacity_adaptation=True,
        box_budget=35_721,
        inverse_max_size=1_024,
        measured_repeats=5,
        parameter_selection_policy="historical_selected_transfer_no_current_scan",
    )

    specs, rule = resolve_method_specs(system, cfg)

    assert rule.raw_tau_box_size == 45_369
    assert rule.config.active_topk == 28_333
    assert rule.active.active_idx.size == 28_333
    assert rule.active.box_idx.size == 35_721
    np.testing.assert_array_equal(rule.active.radii, [94, 94])
    assert rule.selection_rule == "frozen_score_topk_clamped_to_box_budget"
    default = next(spec for spec in specs if spec.label == "default")
    assert default.btab_config is not None
    assert default.btab_config.active_topk == 28_333
    assert "clamped_to_box_budget" in default.selection_rule

    order = score_rank_order(rule.active.rho)
    next_prefix_multi = np.stack(
        np.unravel_index(order[:28_334], (291, 291)), axis=1
    )
    next_radii = np.max(np.abs(next_prefix_multi - np.array([145, 145])), axis=0)
    assert int(np.prod(2 * next_radii + 1)) == 36_099


def test_frozen_topk_box_budget_does_not_relax_scale_provenance() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "default"),
        active_topk=2,
        expected_active_box_size=3,
        box_budget=2,
        rank=2,
        measured_repeats=5,
    )

    with pytest.raises(ValueError, match="capacity adaptation is not authorized"):
        resolve_score_box_rule(system, cfg)


def test_explicit_active_methods_are_charged_shared_score_selection() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "active-inverse", "active-eig"),
        score_tau=1.0,
        box_budget=3,
        inverse_max_size=3,
        rank=2,
        measured_repeats=5,
    )

    specs, rule = resolve_method_specs(system, cfg)
    selected = {
        spec.label: spec for spec in specs if spec.label.startswith("active-")
    }

    assert set(selected) == {"active-inverse", "active-eig"}
    assert all(
        spec.selection_seconds == rule.selection_seconds
        for spec in selected.values()
    )
    assert all(spec.active_set is rule.active for spec in selected.values())


def test_default_route_cap_is_independent_from_explicit_inverse_cap() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("default", "active-inverse", "active-eig"),
        active_topk=3,
        box_budget=system.manifest["M"],
        inverse_max_size=system.manifest["M"],
        default_inverse_max_size=1,
        rank=2,
        measured_repeats=5,
    )

    specs, rule = resolve_method_specs(system, cfg)
    by_label = {spec.label: spec for spec in specs}

    assert int(rule.active.box_idx.size) > cfg.default_inverse_max_size
    assert by_label["default"].kind == "active-eig"
    assert by_label["active-inverse"].kind == "active-inverse"
    assert by_label["default"].active_set is rule.active
    assert by_label["active-inverse"].active_set is rule.active


def test_full_inverse_spec_uses_the_complete_fourier_grid() -> None:
    system = _fake_system()
    cfg = ControlledConfig(
        methods=("cg", "full-inverse"),
        score_tau=1.0,
        box_budget=3,
        inverse_max_size=3,
        rank=2,
        measured_repeats=5,
    )

    specs, _ = resolve_method_specs(system, cfg)
    full = next(spec for spec in specs if spec.label == "full-inverse")

    assert full.kind == "active-inverse"
    assert full.selection_rule == "full_grid"
    assert full.result_role == "baseline"
    assert full.btab_config is not None
    assert full.btab_config.active_mode == "topk"
    assert full.btab_config.active_topk == system.manifest["M"]
    assert full.btab_config.box_budget == system.manifest["M"]
    assert full.btab_config.exact_box_max_size == system.manifest["M"]


def test_system_fingerprint_detects_rhs_change() -> None:
    system = _fake_system()
    before = system_fingerprint(system.data_ctx, system.reg_lambda)
    system.data_ctx.rhs_gpu = system.data_ctx.rhs_gpu.copy()
    system.data_ctx.rhs_gpu[0] += 1.0
    after = system_fingerprint(system.data_ctx, system.reg_lambda)
    assert before != after


@pytest.mark.parametrize(
    ("attribute", "changed_fields"),
    [
        ("weights_gpu_flat", {"weights_sha256"}),
        ("gf_gpu", {"gf_sha256"}),
        # Without an explicit cast solve RHS, the data-context RHS is both the
        # solve array and its separately named storage provenance anchor.
        ("rhs_gpu", {"rhs_sha256", "rhs_storage_sha256"}),
    ],
)
def test_system_component_fingerprints_localize_array_change(
    attribute: str,
    changed_fields: set[str],
) -> None:
    system = _fake_system()
    before = system_component_fingerprints(system.data_ctx)
    changed = np.asarray(getattr(system.data_ctx, attribute)).copy()
    changed.reshape(-1)[0] += 1.0
    setattr(system.data_ctx, attribute, changed)
    after = system_component_fingerprints(system.data_ctx)

    assert {field for field in before if before[field] != after[field]} == changed_fields


def test_timing_prediction_artifact_selects_lowest_converged_measured_beta(
    tmp_path,
) -> None:
    system = _fake_system()
    cfg = ControlledConfig(methods=("cg", "default"), measured_repeats=2)

    def row(method: str, repeat: int, status: str, relres: float) -> dict:
        return {
            "system_id": system.system_id,
            "method": method,
            "method_kind": "cg" if method == "cg" else "active-eig",
            "repeat_idx": repeat,
            "order_position": 0,
            "is_warmup": False,
            "status": status,
            "true_relres": relres,
            "iterations": 10 - repeat,
            "build_seconds": 0.1,
            "solve_seconds": 0.2,
        }

    cg0 = row("cg", 0, "converged", 1e-9)
    default0 = row("default", 0, "maxiter", 2e-7)
    default1 = row("default", 1, "converged", 1e-9)
    rows = [cg0, default0, default1]
    saved = [
        (cg0, np.ones(system.manifest["M"], dtype=np.complex128)),
        (default0, np.zeros(system.manifest["M"], dtype=np.complex128)),
        (default1, np.full(system.manifest["M"], 2.0, dtype=np.complex128)),
    ]

    payload = benchmark_module.save_timing_prediction_solutions(
        system,
        cfg,
        rows,
        saved,
        tmp_path,
    )

    assert payload["solution_count"] == 2
    records = {record["method"]: record for record in payload["solutions"]}
    assert records["cg"]["timing_repeat_idx"] == 0
    assert records["default"]["timing_repeat_idx"] == 1
    assert records["default"]["selection_eligible"] is True
    stored = json.loads(
        (tmp_path / benchmark_module.TIMING_SOLUTIONS_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert stored["timing_solution_artifact_sha256"] == payload[
        "timing_solution_artifact_sha256"
    ]
    with np.load(
        tmp_path / benchmark_module.TIMING_SOLUTIONS_ARTIFACT_FILENAME,
        allow_pickle=False,
    ) as loaded:
        selected = np.asarray(loaded[records["default"]["array_key"]])
    assert np.all(selected == 2.0)


def test_empty_score_box_makes_default_fall_back_to_jacobi() -> None:
    system = _fake_system()
    system.reg_lambda = 1e12
    cfg = ControlledConfig(
        methods=("cg", "default"),
        score_tau=1.0,
        box_budget=3,
        inverse_max_size=3,
        rank=2,
        measured_repeats=5,
    )

    specs, rule = resolve_method_specs(system, cfg)
    default = next(spec for spec in specs if spec.label == "default")

    assert rule.active.box_idx.size == 0
    assert default.kind == "jacobi"
    assert "empty_fallback_jacobi" in default.selection_rule


def test_mixed32_rejects_tolerance_near_machine_epsilon() -> None:
    cfg = ControlledConfig(
        methods=("cg",),
        precision="mixed32",
        tol=1e-7,
        measured_repeats=5,
    )
    with pytest.raises(ValueError, match="machine eps"):
        _validate_config(cfg)


def test_zero_n_train_means_all_but_negative_is_invalid() -> None:
    assert _normalize_n_train(None) is None
    assert _normalize_n_train(0) is None
    assert _normalize_n_train(12) == 12
    with pytest.raises(ValueError, match="n_train"):
        _normalize_n_train(-1)


def test_loader_rejects_requested_rows_beyond_source(tmp_path) -> None:
    np.savez(
        tmp_path / "tiny.npz",
        x_train=np.zeros((3, 2), dtype=np.float32),
        y_train=np.zeros(3, dtype=np.float32),
    )
    (tmp_path / "tiny.json").write_text(
        '{"shapes":{"n_train":3,"dim":2}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exceeds"):
        _load_dataset("tiny", 4, 0, str(tmp_path))


def test_summary_uses_paired_cold_speedup_and_requested_order() -> None:
    rows = []
    for repeat, cg_time, pcg_total in [(0, 10.0, 5.0), (1, 20.0, 5.0)]:
        rows.extend(
            [
                {
                    "method": "pcg",
                    "method_kind": "active-inverse",
                    "repeat_idx": repeat,
                    "is_warmup": False,
                    "build_seconds": 1.0,
                    "solve_seconds": pcg_total - 1.0,
                    "build_plus_solve_seconds": pcg_total,
                    "iterations": 2,
                    "true_relres": 1e-9,
                    "status": "converged",
                },
                {
                    "method": "cg",
                    "method_kind": "cg",
                    "repeat_idx": repeat,
                    "is_warmup": False,
                    "build_seconds": 0.0,
                    "solve_seconds": cg_time,
                    "build_plus_solve_seconds": cg_time,
                    "iterations": 10,
                    "true_relres": 1e-9,
                    "status": "converged",
                },
            ]
        )

    summary = summarize_rows(rows, 100.0, 1e-7, method_order=("cg", "pcg"))

    assert [row["method"] for row in summary] == ["cg", "pcg"]
    pcg = summary[1]
    assert pcg["cold_speedup_median"] == pytest.approx(3.0)
    assert pcg["solver_total_speedup_over_cg_median"] == pytest.approx(3.0)
    assert pcg["solver_total_seconds_median"] == pytest.approx(5.0)
    assert pcg["paired_wins_over_cg"] == 2
    comparison = pairwise_comparisons(rows)[0]
    assert comparison["reference_method"] == "cg"
    assert comparison["candidate_method"] == "pcg"
    assert comparison["total_speedup_median"] == pytest.approx(3.0)
    assert comparison["solver_total_speedup_median"] == pytest.approx(3.0)

    restored_rows = [
        {**row, "setup_inclusive_timing_eligible": False}
        for row in rows
    ]
    restored_summary = summarize_rows(
        restored_rows, 100.0, 1e-7, method_order=("cg", "pcg")
    )
    assert all(
        row["setup_inclusive_timing_eligible"] is False
        for row in restored_summary
    )
    assert all(
        np.isnan(row["shared_fourier_setup_plus_method_speedup_median"])
        for row in restored_summary
    )


def test_solver_total_seconds_is_canonical_over_legacy_alias() -> None:
    rows = [
        {
            "method": "cg",
            "repeat_idx": 0,
            "is_warmup": False,
            "build_seconds": 0.0,
            "solve_seconds": 12.0,
            "solver_total_seconds": 12.0,
            "build_plus_solve_seconds": 120.0,
            "true_relres": 1e-9,
            "status": "converged",
        },
        {
            "method": "candidate",
            "repeat_idx": 0,
            "is_warmup": False,
            "build_seconds": 2.0,
            "solve_seconds": 4.0,
            "solver_total_seconds": 6.0,
            "build_plus_solve_seconds": 1.0,
            "true_relres": 1e-9,
            "status": "converged",
        },
    ]

    summaries = summarize_rows(rows, 1.0, 1e-7, method_order=("cg", "candidate"))
    candidate = summaries[1]
    comparison = pairwise_comparisons(rows, 1e-7)[0]

    assert candidate["solver_total_seconds_median"] == pytest.approx(6.0)
    assert candidate["solver_total_speedup_over_cg_median"] == pytest.approx(2.0)
    assert comparison["solver_total_speedup_median"] == pytest.approx(2.0)


def test_run_row_normalizes_selection_build_and_solve_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _fake_system()
    cfg = ControlledConfig(methods=("cg", "jacobi"), measured_repeats=5)
    spec = benchmark_module.MethodSpec(
        label="jacobi",
        kind="jacobi",
        selection_seconds=1.25,
    )
    elapsed = iter((2.5, 4.0))

    monkeypatch.setattr(
        benchmark_module,
        "_timed",
        lambda backend, operation: (operation(), next(elapsed)),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_build_preconditioner",
        lambda supplied_system, supplied_cfg, supplied_spec: (
            lambda vector, out: None,
            object(),
            {},
        ),
    )
    monkeypatch.setattr(
        benchmark_module,
        "pcg_solve_gpu",
        lambda *args, **kwargs: (
            np.zeros_like(system.rhs_gpu),
            3,
            1e-9,
            {"status": "converged", "n_matvec": 3},
        ),
    )
    monkeypatch.setattr(benchmark_module, "_true_residual", lambda *args: 1e-9)

    row, _ = benchmark_module.run_one_method(
        system,
        cfg,
        spec,
        repeat_idx=0,
        order_position=0,
        is_warmup=False,
    )

    assert row["selection_seconds"] == pytest.approx(1.25)
    assert row["preconditioner_build_seconds"] == pytest.approx(2.5)
    assert row["build_seconds"] == pytest.approx(3.75)
    assert row["solve_seconds"] == pytest.approx(4.0)
    assert row["solver_total_seconds"] == pytest.approx(7.75)
    assert row["build_plus_solve_seconds"] == pytest.approx(7.75)


def test_unconverged_candidate_is_excluded_from_speedup_claims() -> None:
    rows = [
        {
            "method": "cg",
            "method_kind": "cg",
            "repeat_idx": 0,
            "is_warmup": False,
            "build_seconds": 0.0,
            "solve_seconds": 10.0,
            "build_plus_solve_seconds": 10.0,
            "iterations": 10,
            "true_relres": 5e-8,
            "status": "converged",
            "tol": 1e-7,
        },
        {
            "method": "fast-but-wrong",
            "method_kind": "jacobi",
            "repeat_idx": 0,
            "is_warmup": False,
            "build_seconds": 0.1,
            "solve_seconds": 0.1,
            "build_plus_solve_seconds": 0.2,
            "iterations": 1,
            "true_relres": 2e-7,
            "status": "maxiter",
            "tol": 1e-7,
        },
    ]

    summary = summarize_rows(
        rows,
        1.0,
        1e-7,
        method_order=("cg", "fast-but-wrong"),
    )
    bad = summary[1]
    assert bad["converged_repeats"] == 0
    assert bad["paired_comparisons"] == 0
    assert not bad["performance_claim_eligible"]
    assert np.isnan(bad["cold_speedup_median"])

    comparison = pairwise_comparisons(rows, 1e-7)[0]
    assert comparison["available_paired_repeats"] == 1
    assert comparison["paired_repeats"] == 0
    assert not comparison["performance_claim_eligible"]
    assert np.isnan(comparison["total_speedup_median"])


def test_error_repeat_cannot_disappear_from_pairwise_eligibility() -> None:
    rows = []
    for repeat in range(5):
        rows.append(
            {
                "method": "cg",
                "repeat_idx": repeat,
                "is_warmup": False,
                "build_plus_solve_seconds": 10.0,
                "solve_seconds": 10.0,
                "true_relres": 5e-8,
                "status": "converged",
                "tol": 1e-7,
            }
        )
        rows.append(
            {
                "method": "candidate",
                "repeat_idx": repeat,
                "is_warmup": False,
                "build_plus_solve_seconds": 5.0 if repeat else np.nan,
                "solve_seconds": 4.0 if repeat else np.nan,
                "true_relres": 5e-8 if repeat else np.nan,
                "status": "converged" if repeat else "error",
                "tol": 1e-7,
            }
        )

    comparison = pairwise_comparisons(rows, 1e-7)[0]
    assert comparison["available_paired_repeats"] == 5
    assert comparison["paired_repeats"] == 4
    assert not comparison["performance_claim_eligible"]


def test_pairwise_reports_crossover_for_higher_build_lower_solve() -> None:
    rows = []
    for repeat in range(5):
        rows.extend(
            [
                {
                    "method": "cg",
                    "repeat_idx": repeat,
                    "is_warmup": False,
                    "build_seconds": 0.0,
                    "solve_seconds": 10.0,
                    "build_plus_solve_seconds": 10.0,
                    "true_relres": 1e-9,
                    "status": "converged",
                    "tol": 1e-7,
                },
                {
                    "method": "candidate",
                    "repeat_idx": repeat,
                    "is_warmup": False,
                    "build_seconds": 12.0,
                    "solve_seconds": 4.0,
                    "build_plus_solve_seconds": 16.0,
                    "true_relres": 1e-9,
                    "status": "converged",
                    "tol": 1e-7,
                },
            ]
        )

    comparison = pairwise_comparisons(rows, 1e-7)[0]
    assert comparison["crossover_status"] == "candidate_higher_build_lower_solve"
    assert comparison["cold_to_reuse_crossover_rhs"] == pytest.approx(2.0)
    assert np.isnan(comparison["candidate_faster_through_rhs"])


def test_post_diagnostics_records_pcg_error_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _fake_system()
    cfg = ControlledConfig(post_diagnostic_mode="cheap")
    specs = [
        benchmark_module.MethodSpec(label="bad-pcg", kind="active-eig"),
        benchmark_module.MethodSpec(label="good-pcg", kind="active-eig"),
    ]
    pcg_calls: list[str] = []

    def fake_build_preconditioner(supplied_system, supplied_cfg, spec):
        del supplied_system, supplied_cfg
        return (
            lambda vector, out: None,
            SimpleNamespace(),
            {"preconditioner_probe": spec.label},
        )

    def fake_pcg_solve(*args, **kwargs):
        del args
        work_prefix = str(kwargs["work_prefix"])
        pcg_calls.append(work_prefix)
        if "bad_pcg" in work_prefix:
            raise RuntimeError(
                "PCG denominator is non-positive or non-finite (denom=nan)"
            )
        return (
            np.zeros_like(system.rhs_gpu),
            3,
            1e-9,
            {"status": "converged"},
        )

    monkeypatch.setattr(
        benchmark_module,
        "_build_preconditioner",
        fake_build_preconditioner,
    )
    monkeypatch.setattr(
        benchmark_module,
        "run_btab_post_diagnostics",
        lambda *args, **kwargs: {"epsilon_T": 0.1, "eta_eig": 0.2},
    )
    monkeypatch.setattr(benchmark_module, "pcg_solve_gpu", fake_pcg_solve)
    monkeypatch.setattr(
        benchmark_module,
        "_true_residual",
        lambda supplied_system, supplied_cfg, beta: 1e-9,
    )

    rows, arrays = benchmark_module._post_diagnostics(
        system,
        cfg,
        specs,
        SimpleNamespace(),
    )

    assert [row["method"] for row in rows] == ["bad-pcg", "good-pcg"]
    assert len(pcg_calls) == 2
    assert arrays == {}

    failed, succeeded = rows
    assert failed["preconditioner_probe"] == "bad-pcg"
    assert failed["diagnostic_status"] == "error"
    assert failed["diagnostic_pcg_status"] == "error"
    assert failed["diagnostic_error_stage"] == "diagnostic_pcg"
    assert failed["error_type"] == "RuntimeError"
    assert "denom=nan" in failed["error_message"]
    assert "Traceback (most recent call last):" in failed["traceback"]
    assert "RuntimeError" in failed["traceback"]

    assert succeeded["preconditioner_probe"] == "good-pcg"
    assert succeeded["diagnostic_status"] == "ok"
    assert succeeded["diagnostic_pcg_status"] == "converged"
