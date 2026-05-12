from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mode_display_name(mode: str, top_q: int | float) -> str:
    mode = str(mode)
    q = int(top_q)
    if mode == "gpu_v1_topq0":
        return "baseline EFGP"
    if mode == "gpu_v3_topq_eigenpro_nystrom":
        return f"ours_q{q}"
    return f"{mode}_q{q}"


def save_complexity_benchmark_plots(
    summary_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    dpi: int = 180,
    show: bool = False,
    precompute_methods_by_mode: dict[str, Iterable[str] | None] | None = None,
    precompute_methods_default: Iterable[str] | None = None,
) -> list[Path]:
    """
    Save Figure1-5 complexity benchmark plots from grouped summary dataframe.

    Expects per-mode aggregates including ``time_train_median``
    (`time_train` = precompute + eigenspace + precond_build + solve) and timing medians used in figures.
    Fig1/Fig5 use training time ``T_train``; end-to-end ``wall_s_total`` is for tables / exports.
    """
    def _norm_choice(v: Iterable[str] | None) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        out: list[str] = []
        for s in v:
            ss = str(s).strip().lower()
            if ss == "" or ss == "none":
                continue
            out.append(ss)
        if not out:
            return None
        # dedupe, keep order
        seen: set[str] = set()
        uniq: list[str] = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    # Caller-provided selector: filter only (allow multiple methods).
    if "precompute_method" in summary_df.columns and not summary_df.empty and (
        precompute_methods_by_mode is not None or precompute_methods_default is not None
    ):
        df = summary_df.copy()
        df["_pcm_lc"] = df["precompute_method"].astype(str).str.strip().str.lower()
        sel_map = precompute_methods_by_mode or {}
        sel_default = _norm_choice(precompute_methods_default)

        kept = []
        for mode, g in df.groupby("mode", dropna=False):
            m = str(mode)
            choice = _norm_choice(sel_map.get(m, sel_default))
            if choice is None:
                kept.append(g)
                continue
            sub = g[g["_pcm_lc"].isin(choice)]
            kept.append(sub if not sub.empty else g)
        summary_df = pd.concat(kept, ignore_index=True).drop(columns=["_pcm_lc"], errors="ignore")

    # Default behavior (no selector): pick one method per curve to avoid duplicated labels.
    elif "precompute_method" in summary_df.columns and not summary_df.empty:
        df = summary_df.copy()
        df["_pcm_lc"] = df["precompute_method"].astype(str).str.strip().str.lower()
        group_cols = [c for c in ("mode", "top_q", "N", "eps") if c in df.columns]
        if not group_cols:
            group_cols = ["mode", "top_q"]

        def _pick_one(g: pd.DataFrame) -> pd.DataFrame:
            mode = str(g["mode"].iloc[0])
            if mode == "gpu_v1_topq0":
                gg = g[g["_pcm_lc"] == "original"]
                if not gg.empty:
                    return gg.iloc[[0]]
            if mode == "gpu_v3_topq_eigenpro_nystrom":
                gg = g[g["_pcm_lc"] == "c1"]
                if not gg.empty:
                    return gg.iloc[[0]]
                gg = g[g["_pcm_lc"] == "original"]
                if not gg.empty:
                    return gg.iloc[[0]]

            gg = g[g["_pcm_lc"] == "original"]
            if not gg.empty:
                return gg.iloc[[0]]
            return g.iloc[[0]]

        picked = []
        for _, g in df.groupby(group_cols, dropna=False):
            picked.append(_pick_one(g))
        summary_df = pd.concat(picked, ignore_index=True).drop(columns=["_pcm_lc"], errors="ignore")

    plot_dir = Path(out_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Figure 1: median training time T_train vs N (precompute+eigenspace+precond_build+solve; excludes predict)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (mode, top_q), g in summary_df.groupby(["mode", "top_q"]):
        g = g.sort_values("N")
        ax.plot(g["N"], g["time_train_median"], marker="o", label=_mode_display_name(mode, top_q))
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
        display_name = _mode_display_name(mode, top_q)
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
        ax.set_title(f"Figure 2: Stage time vs N | {display_name}")
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
        ax.plot(g["N"], g["cg_iters_median"], marker="o", label=_mode_display_name(mode, top_q))
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
        display_name = _mode_display_name(mode, top_q)
        solve_other = g["time_solve_median"] - g["t_matvec_total_median"] - g["t_precond_total_median"]
        solve_other = solve_other.clip(lower=0)
        ax.plot(g["N"], g["t_matvec_total_median"], marker="o", linestyle="-", label=f"matvec {display_name}")
        ax.plot(g["N"], g["t_precond_total_median"], marker="s", linestyle="--", label=f"precond {display_name}")
        ax.plot(g["N"], solve_other, marker="^", linestyle=":", label=f"other {display_name}")
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
        display_name = _mode_display_name(mode, top_q)
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
        ax.set_title(f"Figure 5: Training-stage share vs N | {display_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = plot_dir / f"fig5_stage_share_{mode}_q{int(top_q)}.png"
        fig.savefig(p, dpi=dpi)
        saved.append(p)
        if show:
            plt.show()
        plt.close(fig)

    # Figure 6: per-iteration time vs N for q=0 and q=q_max.
    # - y-axis: time per iteration (average)
    # - color: q (0 vs q_max)
    # - line style: solve (solid) vs matvec (dashed)
    need_cols = {"mode", "top_q", "N", "time_solve_median", "t_matvec_total_median", "cg_iters_median"}
    if not summary_df.empty and need_cols.issubset(set(summary_df.columns)):
        q_vals = pd.to_numeric(summary_df["top_q"], errors="coerce")
        q_pos = q_vals[q_vals > 0]
        q_max = int(q_pos.max()) if not q_pos.empty else 0

        base = summary_df[(summary_df["mode"] == "gpu_v1_topq0") & (summary_df["top_q"] == 0)].copy()

        hi_mode = None
        if q_max > 0:
            if (summary_df["mode"] == "gpu_v3_topq_eigenpro_nystrom").any():
                hi_mode = "gpu_v3_topq_eigenpro_nystrom"
            elif (summary_df["mode"] == "gpu_v3_topq").any():
                hi_mode = "gpu_v3_topq"

        hi = (
            summary_df[(summary_df["mode"] == hi_mode) & (summary_df["top_q"] == q_max)].copy()
            if hi_mode is not None
            else summary_df.iloc[0:0].copy()
        )

        def _per_iter_curves(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            g = df.sort_values("N").copy()
            it = pd.to_numeric(g["cg_iters_median"], errors="coerce")
            it = it.where(it > 0, np.nan)
            solve_per_iter = pd.to_numeric(g["time_solve_median"], errors="coerce") / it
            # In CG/PCG, matvec count is typically iters+1 (extra at init). We don't have n_matvec in summary,
            # so approximate per-iteration matvec time by dividing by (iters+1).
            matvec_per_iter = pd.to_numeric(g["t_matvec_total_median"], errors="coerce") / (it + 1.0)
            return (
                g["N"].to_numpy(dtype=float),
                solve_per_iter.to_numpy(dtype=float),
                matvec_per_iter.to_numpy(dtype=float),
            )

        if not base.empty and (q_max == 0 or not hi.empty):
            fig, ax = plt.subplots(figsize=(8, 5))
            c0 = "C0"
            n0, s0, m0 = _per_iter_curves(base)
            base_name = _mode_display_name("gpu_v1_topq0", 0)
            ax.plot(n0, s0, color=c0, linestyle="-", marker="o", label=f"{base_name} solve/iter")
            ax.plot(n0, m0, color=c0, linestyle="--", marker="o", label=f"{base_name} matvec/iter")

            if not hi.empty:
                c1 = "C1"
                n1, s1, m1 = _per_iter_curves(hi)
                hi_name = _mode_display_name(hi_mode, q_max)
                ax.plot(n1, s1, color=c1, linestyle="-", marker="s", label=f"{hi_name} solve/iter")
                ax.plot(n1, m1, color=c1, linestyle="--", marker="s", label=f"{hi_name} matvec/iter")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("N")
            ax.set_ylabel("median time per iteration")
            if hi_mode is None:
                ax.set_title(f"Figure 6: per-iteration time vs N | {base_name}")
            else:
                ax.set_title(f"Figure 6: per-iteration time vs N | {base_name} vs {hi_name}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig6_path = plot_dir / "fig6_per_iter_time_vs_n_loglog.png"
            fig.savefig(fig6_path, dpi=dpi)
            saved.append(fig6_path)
            if show:
                plt.show()
            plt.close(fig)

    return saved
