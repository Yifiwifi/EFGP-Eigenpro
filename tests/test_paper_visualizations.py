from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.config import BTABExperimentConfig
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.paper_visualizations import (
    PaperVisualizationConfig,
    make_active_score_mass_figure,
    make_group_c_boxeig_parameter_sweep_figure,
    rerender_paper_visualizations_from_saved_data,
)
from efgp_eigenpro_py.gpu.iterative_solvers import cg_solve_gpu, pcg_solve_gpu


def _backend():
    return SimpleNamespace(
        xp=np,
        linalg=SimpleNamespace(norm=np.linalg.norm, vdot=np.vdot),
    )


def test_cg_trace_callback_preserves_solution():
    backend = _backend()
    A = np.diag(np.array([2.0, 3.0, 5.0], dtype=np.float64))
    b = np.array([1.0, -2.0, 3.0], dtype=np.float64)

    def matvec(v, out):
        out[...] = A @ v

    x0, it0, rel0, stats0 = cg_solve_gpu(
        backend,
        matvec,
        b,
        SimpleNamespace(),
        1e-12,
        20,
        return_stats=True,
    )
    trace = []
    x1, it1, rel1, stats1 = cg_solve_gpu(
        backend,
        matvec,
        b,
        SimpleNamespace(),
        1e-12,
        20,
        return_stats=True,
        trace_callback=lambda event: trace.append((event["iteration"], event["relres"])),
    )

    np.testing.assert_allclose(x1, x0, rtol=1e-12, atol=1e-12)
    assert it1 == it0
    assert rel1 == rel0
    assert stats1["n_matvec"] == stats0["n_matvec"]
    assert trace[0][0] == 0
    assert trace[-1][0] == it1


def test_pcg_trace_callback_records_initial_and_final():
    backend = _backend()
    A = np.diag(np.array([2.0, 4.0, 8.0], dtype=np.float64))
    b = np.array([1.0, 2.0, 1.0], dtype=np.float64)

    def matvec(v, out):
        out[...] = A @ v

    def precond(v, out):
        out[...] = v

    trace = []
    _x, it, _rel, _stats = pcg_solve_gpu(
        backend,
        matvec,
        precond,
        b,
        SimpleNamespace(),
        1e-12,
        20,
        return_stats=True,
        trace_callback=lambda event: trace.append((event["iteration"], event["relres"])),
    )
    assert trace[0][0] == 0
    assert trace[-1][0] == it
    assert trace[-1][1] <= trace[0][1]


def test_active_score_visualization_writes_artifacts(tmp_path):
    cfg = BTABExperimentConfig()
    viz_cfg = PaperVisualizationConfig(
        output_dir=tmp_path,
        make_spectrum=False,
        make_residual=False,
        make_rmse=False,
        make_prediction_map=False,
    )
    info = make_active_score_mass_figure(cfg, viz_cfg)

    assert (tmp_path / "active_score_cumulative_mass.png").exists()
    assert (tmp_path / "active_score_cumulative_mass.pdf").exists()
    assert (tmp_path / "active_score_cumulative_mass.csv").exists()
    assert info["notes"].startswith("Kernel/grid-level")


def test_group_c_boxeig_parameter_sweep_from_summary(tmp_path):
    rows = [
        {
            "status": "ok",
            "dataset_stem": "USGS_matern_group_c",
            "kernel_family": "matern",
            "n_train": 300000000,
            "method": "plain_cg",
            "version": "v1",
            "cg_iters": 21445,
            "cg_relres": 5e-8,
            "time_total": 27.23,
            "time_solve": 27.23,
            "time_precond_build": 0.0,
            "rmse_test": 1.0,
        },
        {
            "status": "ok",
            "dataset_stem": "USGS_matern_group_c",
            "kernel_family": "matern",
            "n_train": 300000000,
            "method": "eigenpro_pcg_q384",
            "version": "v3",
            "cg_iters": 296,
            "cg_relres": 8e-8,
            "time_total": 3.66,
            "time_solve": 2.7,
            "time_precond_build": 0.96,
            "rmse_test": 1.0,
        },
        {
            "status": "ok",
            "dataset_stem": "USGS_matern_group_c",
            "kernel_family": "matern",
            "n_train": 300000000,
            "method": "btab_auto_topk_2048",
            "version": "v6_btab",
            "cg_iters": 5000,
            "cg_relres": 2e-5,
            "time_total": 7.0,
            "time_solve": 6.8,
            "time_precond_build": 0.2,
            "rmse_test": 1.2,
        },
        {
            "status": "ok",
            "dataset_stem": "USGS_matern_group_c",
            "kernel_family": "matern",
            "n_train": 300000000,
            "method": "btab_boxeig_topk_20480_q320",
            "version": "v7_btab_boxeig",
            "btab_active_topk": 20480,
            "btab_box_size": "",
            "box_shape": [161, 161],
            "btab_eig_q": 320,
            "cg_iters": 360,
            "cg_relres": 9e-8,
            "time_total": 3.2,
            "time_solve": 2.4,
            "time_precond_build": 0.8,
            "rmse_test": 1.01,
        },
        {
            "status": "ok",
            "dataset_stem": "USGS_matern_group_c",
            "kernel_family": "matern",
            "n_train": 300000000,
            "method": "btab_boxeig_topk_25720_q384",
            "version": "v7_btab_boxeig",
            "btab_active_topk": 25720,
            "btab_box_size": 35721,
            "box_shape": [189, 189],
            "btab_eig_q": 384,
            "cg_iters": 319,
            "cg_relres": 7e-8,
            "time_total": 2.8977,
            "time_solve": 2.0,
            "time_precond_build": 0.8977,
            "rmse_test": 1.01,
        },
        {
            "status": "ok",
            "dataset_stem": "USGS_matern_group_c",
            "kernel_family": "matern",
            "n_train": 300000000,
            "method": "btab_boxeig_topk_35721_q384",
            "version": "v7_btab_boxeig",
            "btab_active_topk": 35721,
            "btab_box_size": 35721,
            "box_shape": [189, 189],
            "btab_eig_q": 384,
            "cg_iters": 318,
            "cg_relres": 6e-8,
            "time_total": 3.1,
            "time_solve": 2.05,
            "time_precond_build": 1.05,
            "rmse_test": 1.01,
        },
        {
            "status": "ok",
            "dataset_stem": "USGS_matern_group_c",
            "kernel_family": "matern",
            "n_train": 300000000,
            "method": "btab_boxeig_topk_35721_q448",
            "version": "v7_btab_boxeig",
            "btab_active_topk": 35721,
            "btab_box_size": 35721,
            "box_shape": [189, 189],
            "btab_eig_q": 448,
            "cg_iters": 270,
            "cg_relres": 7e-8,
            "time_total": 2.9953,
            "time_solve": 1.9,
            "time_precond_build": 1.0953,
            "rmse_test": 1.01,
        },
        {
            "status": "ok",
            "dataset_stem": "USGS_matern_group_c",
            "kernel_family": "matern",
            "n_train": 300000000,
            "method": "btab_boxeig_topk_12345_q384",
            "version": "v7_btab_boxeig",
            "btab_active_topk": 12345,
            "btab_box_size": 15625,
            "box_shape": [125, 125],
            "btab_eig_q": 384,
            "cg_iters": 100,
            "cg_relres": 5e-8,
            "time_total": 1.0,
            "time_solve": 0.8,
            "time_precond_build": 0.2,
            "rmse_test": 1.01,
        },
    ]
    fieldnames = sorted({key for row in rows for key in row})
    with (tmp_path / "master_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    info = make_group_c_boxeig_parameter_sweep_figure(
        BTABExperimentConfig(),
        PaperVisualizationConfig(output_dir=tmp_path, summary_root=tmp_path, tol=1e-7),
    )

    assert info["available"] is True
    assert (tmp_path / "group_c_boxeig_parameter_sweep.pdf").exists()
    assert (tmp_path / "group_c_boxeig_sweep_raw.csv").exists()
    assert (tmp_path / "group_c_boxeig_sweep_collapsed.csv").exists()
    assert (tmp_path / "group_c_boxeig_sweep_plot_data.csv").exists()
    assert info["best_boxeig"]["btab_box_size"] == 35721
    assert info["best_boxeig"]["btab_eig_q"] == 384
    assert info["best_boxeig"]["time_total"] == 2.8977
    assert info["sweep_spec"]["scan_variables"] == ["btab_box_size", "btab_eig_q"]
    assert 25720 in info["sweep_spec"]["configured_active_topk_targets"]
    assert 35721 in info["sweep_spec"]["observed_btab_box_sizes"]
    assert 15625 not in info["sweep_spec"]["observed_btab_box_sizes"]
    with (tmp_path / "group_c_boxeig_sweep_collapsed.csv").open(encoding="utf-8") as f:
        collapsed = list(csv.DictReader(f))
    assert any(row["btab_box_size"] == "25921" for row in collapsed)
    assert all(row["btab_box_size"] != "15625" for row in collapsed)
    assert any(
        row["btab_box_size"] == "35721"
        and row["btab_eig_q"] == "384"
        and row["btab_active_topk_values"] == "25720,35721"
        for row in collapsed
    )
    baselines = {row["baseline_label"]: row for row in info["baselines"]}
    assert baselines["EFGP-CG"]["available"] is True
    assert baselines["Global EigenPro-style PCG"]["available"] is True
    assert baselines["Active inverse"]["available"] is False


def test_rerender_paper_visualizations_from_saved_data(tmp_path):
    (tmp_path / "figure1_mechanism_active_score.csv").write_text(
        "\n".join(
                [
                    "profile_label,kernel_family,rank,cumulative_rho_mass",
                    "\"SE kernel, M=1225\",SE,1,0.7",
                    "\"SE kernel, M=1225\",SE,10,0.95",
                    "\"Matern 3/2 kernel, M=35721\",matern,1,0.2",
                    "\"Matern 3/2 kernel, M=35721\",matern,10,0.5",
                ]
            ),
        encoding="utf-8",
    )
    (tmp_path / "figure1_mechanism_spectrum.csv").write_text(
        "\n".join(
            [
                "spectrum,rank,eigenvalue",
                "raw_A_BB,1,10.0",
                "raw_A_BB,2,2.0",
                "box_eigenpro_preconditioned,1,1.1",
                "box_eigenpro_preconditioned,2,0.9",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "fig6_residual_rmse_trace.csv").write_text(
        "\n".join(
            [
                "method_label,iteration,relres,rmse_test_sample",
                "EFGP-CG,0,1.0,1.2",
                "EFGP-CG,10,0.1,0.8",
                "Box-EigenPro,0,1.0,1.2",
                "Box-EigenPro,10,0.01,0.75",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "group_c_boxeig_sweep_plot_data.csv").write_text(
        "\n".join(
            [
                "row_type,btab_box_size,btab_eig_q,iters,time_total,converged,baseline_label,available",
                "boxeig,25921,320,360,3.2,True,,",
                "boxeig,35721,384,319,2.9,True,,",
                "baseline,,,21445,27.23,True,EFGP-CG,True",
            ]
        ),
        encoding="utf-8",
    )
    truth = np.arange(16, dtype=np.float64).reshape(4, 4)
    pred = truth + 0.1
    err = np.abs(pred - truth)
    np.savez_compressed(
        tmp_path / "usgs_prediction_error_map_rasters.npz",
        truth=truth,
        prediction=pred,
        abs_error=err,
        extent=np.asarray([0.0, 1.0, 0.0, 1.0]),
    )

    manifest = rerender_paper_visualizations_from_saved_data(tmp_path)

    assert (tmp_path / "figure1_mechanism_diagnostics.pdf").exists()
    assert (tmp_path / "fig6_residual_convergence.pdf").exists()
    assert (tmp_path / "rmse_checkpoint_convergence.pdf").exists()
    assert (tmp_path / "usgs_prediction_error_map.pdf").exists()
    assert (tmp_path / "group_c_boxeig_parameter_sweep.pdf").exists()
    assert (tmp_path / "paper_visualization_rerender_manifest.json").exists()
    assert manifest["does_not_run_solver"] is True
