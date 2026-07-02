from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.config import BTABExperimentConfig
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.paper_visualizations import (
    PaperVisualizationConfig,
    make_active_score_mass_figure,
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
    assert (tmp_path / "paper_visualization_rerender_manifest.json").exists()
    assert manifest["does_not_run_solver"] is True
