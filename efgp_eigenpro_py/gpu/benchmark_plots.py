from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_complexity_benchmark_plots(
    summary_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    dpi: int = 180,
    show: bool = False,
) -> list[Path]:
    """
    Save Figure1-5 complexity benchmark plots from grouped summary dataframe.
    """
    plot_dir = Path(out_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Figure 1: end-to-end median wall time vs N (log-log)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (mode, top_q), g in summary_df.groupby(["mode", "top_q"]):
        g = g.sort_values("N")
        ax.plot(g["N"], g["wall_s_total_median"], marker="o", label=f"{mode}_q{int(top_q)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("median wall_s_total")
    ax.set_title("Figure 1: End-to-end time vs N")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig1_path = plot_dir / "fig1_total_time_vs_n_loglog.png"
    fig.savefig(fig1_path, dpi=dpi)
    saved.append(fig1_path)
    if show:
        plt.show()
    plt.close(fig)

    # Figure 2: stage time vs N by mode
    for (mode, top_q), g in summary_df.groupby(["mode", "top_q"]):
        g = g.sort_values("N")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(g["N"], g["time_precompute_median"], marker="o", label="precompute")
        ax.plot(g["N"], g["time_eigenspace_median"], marker="o", label="eigenspace")
        ax.plot(g["N"], g["time_solve_median"], marker="o", label="solve")
        ax.plot(g["N"], g["time_predict_median"], marker="o", label="predict")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("median stage time")
        ax.set_title(f"Figure 2: Stage time vs N | {mode}_q{int(top_q)}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = plot_dir / f"fig2_stage_vs_n_{mode}_q{int(top_q)}.png"
        fig.savefig(p, dpi=dpi)
        saved.append(p)
        if show:
            plt.show()
        plt.close(fig)

    # Figure 3: cg_iters vs N
    fig, ax = plt.subplots(figsize=(8, 5))
    for (mode, top_q), g in summary_df.groupby(["mode", "top_q"]):
        g = g.sort_values("N")
        ax.plot(g["N"], g["cg_iters_median"], marker="o", label=f"{mode}_q{int(top_q)}")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("median cg_iters")
    ax.set_title("Figure 3: CG iterations vs N")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig3_path = plot_dir / "fig3_cg_iters_vs_n.png"
    fig.savefig(fig3_path, dpi=dpi)
    saved.append(fig3_path)
    if show:
        plt.show()
    plt.close(fig)

    # Figure 4: solve decomposition vs N
    fig, ax = plt.subplots(figsize=(8, 5))
    for (mode, top_q), g in summary_df.groupby(["mode", "top_q"]):
        g = g.sort_values("N")
        solve_other = g["time_solve_median"] - g["t_matvec_total_median"] - g["t_precond_total_median"]
        solve_other = solve_other.clip(lower=0)
        ax.plot(g["N"], g["t_matvec_total_median"], marker="o", linestyle="-", label=f"matvec {mode}_q{int(top_q)}")
        ax.plot(g["N"], g["t_precond_total_median"], marker="s", linestyle="--", label=f"precond {mode}_q{int(top_q)}")
        ax.plot(g["N"], solve_other, marker="^", linestyle=":", label=f"other {mode}_q{int(top_q)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("median solve sub-time")
    ax.set_title("Figure 4: Solve decomposition vs N")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig4_path = plot_dir / "fig4_solve_decompose_vs_n.png"
    fig.savefig(fig4_path, dpi=dpi)
    saved.append(fig4_path)
    if show:
        plt.show()
    plt.close(fig)

    # Figure 5: time share vs N (per mode)
    for (mode, top_q), g in summary_df.groupby(["mode", "top_q"]):
        g = g.sort_values("N")
        total = g["wall_s_total_median"].replace(0, np.nan)
        r_pre = g["time_precompute_median"] / total
        r_eig = g["time_eigenspace_median"] / total
        r_sol = g["time_solve_median"] / total
        r_pred = g["time_predict_median"] / total

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(g["N"], r_pre, marker="o", label="precompute/total")
        ax.plot(g["N"], r_eig, marker="o", label="eigenspace/total")
        ax.plot(g["N"], r_sol, marker="o", label="solve/total")
        ax.plot(g["N"], r_pred, marker="o", label="predict/total")
        ax.set_xscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("time share")
        ax.set_title(f"Figure 5: Stage share vs N | {mode}_q{int(top_q)}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = plot_dir / f"fig5_stage_share_{mode}_q{int(top_q)}.png"
        fig.savefig(p, dpi=dpi)
        saved.append(p)
        if show:
            plt.show()
        plt.close(fig)

    return saved
