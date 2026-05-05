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

    Expects per-mode aggregates including ``time_train_median``
    (`time_train` = precompute + eigenspace + precond_build + solve) and timing medians used in figures.
    Fig1/Fig5 use training time ``T_train``; end-to-end ``wall_s_total`` is for tables / exports.
    """
    plot_dir = Path(out_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Figure 1: median training time T_train vs N (precompute+eigenspace+precond_build+solve; excludes predict)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (mode, top_q), g in summary_df.groupby(["mode", "top_q"]):
        g = g.sort_values("N")
        ax.plot(g["N"], g["time_train_median"], marker="o", label=f"{mode}_q{int(top_q)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("median time_train")
    ax.set_title("Figure 1: Training time T_train vs N")
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
        ax.plot(g["N"], g["time_precond_build_median"], marker="o", label="precond_build")
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

    # Figure 5: share of training time T_train vs N (excludes predict; see wall_s_total in tables)
    for (mode, top_q), g in summary_df.groupby(["mode", "top_q"]):
        g = g.sort_values("N")
        denom = g["time_train_median"].replace(0, np.nan)
        r_pre = g["time_precompute_median"] / denom
        r_eig = g["time_eigenspace_median"] / denom
        r_pb = g["time_precond_build_median"] / denom
        r_sol = g["time_solve_median"] / denom

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(g["N"], r_pre, marker="o", label="precompute/T_train")
        ax.plot(g["N"], r_eig, marker="o", label="eigenspace/T_train")
        ax.plot(g["N"], r_pb, marker="o", label="precond_build/T_train")
        ax.plot(g["N"], r_sol, marker="o", label="solve/T_train")
        ax.set_xscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("fraction of training time")
        ax.set_title(f"Figure 5: Training-stage share vs N | {mode}_q{int(top_q)}")
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
