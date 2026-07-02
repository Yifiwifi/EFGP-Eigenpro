from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from efgp_eigenpro_py.gpu.box_toeplitz_active_block.config import BTABExperimentConfig
from efgp_eigenpro_py.gpu.box_toeplitz_active_block.paper_visualizations import (
    PaperVisualizationConfig,
    make_active_score_mass_figure,
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
