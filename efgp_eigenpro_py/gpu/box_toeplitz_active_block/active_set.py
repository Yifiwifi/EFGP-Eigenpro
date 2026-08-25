from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BoxActiveSet:
    gamma: float
    rho: np.ndarray
    active_idx: np.ndarray
    box_idx: np.ndarray
    tail_idx: np.ndarray
    radii: np.ndarray
    center_multi: np.ndarray
    active_mode: str
    active_topk: int | None
    active_tau: float | None


def compute_rho(gamma: float, weights: np.ndarray, reg_lambda: float) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return (float(gamma) * (np.abs(w) ** 2)) / max(float(reg_lambda), 1e-300)


def select_active_indices(
    rho: np.ndarray,
    *,
    mode: str,
    topk: int | None = None,
    tau: float | None = None,
) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    mode_key = str(mode).strip().lower()
    if mode_key == "topk":
        if topk is None or int(topk) <= 0:
            raise ValueError("active_topk must be > 0 when active_mode='topk'.")
        k = min(int(topk), int(rho.size))
        order = np.argsort(rho)[::-1]
        return np.sort(order[:k].astype(np.int64, copy=False))
    if mode_key == "tau":
        if tau is None:
            raise ValueError("active_tau is required when active_mode='tau'.")
        return np.flatnonzero(rho > float(tau)).astype(np.int64, copy=False)
    raise ValueError(f"unknown active_mode={mode!r}; expected 'topk' or 'tau'.")


def _flatten_center_index(mtot: int, dim: int) -> np.ndarray:
    hm = (int(mtot) - 1) // 2
    return np.full((int(dim),), hm, dtype=np.int64)


def build_box_active_set(
    *,
    gamma: float,
    weights: np.ndarray,
    reg_lambda: float,
    mtot: int,
    dim: int,
    active_mode: str,
    active_topk: int | None,
    active_tau: float | None,
    box_budget: int | None = None,
) -> BoxActiveSet:
    rho = compute_rho(gamma, weights, reg_lambda)
    active_idx = select_active_indices(
        rho,
        mode=active_mode,
        topk=active_topk,
        tau=active_tau,
    )
    center_multi = _flatten_center_index(mtot, dim)
    if active_idx.size == 0:
        all_idx = np.arange(int(weights.size), dtype=np.int64)
        return BoxActiveSet(
            gamma=float(gamma),
            rho=rho,
            active_idx=active_idx,
            box_idx=np.empty((0,), dtype=np.int64),
            tail_idx=all_idx,
            radii=np.zeros((int(dim),), dtype=np.int64),
            center_multi=center_multi,
            active_mode=str(active_mode),
            active_topk=None if active_topk is None else int(active_topk),
            active_tau=None if active_tau is None else float(active_tau),
        )

    active_multi = np.stack(
        np.unravel_index(active_idx, (int(mtot),) * int(dim)),
        axis=1,
    ).astype(np.int64, copy=False)
    centered = active_multi - center_multi[None, :]
    radii = np.max(np.abs(centered), axis=0).astype(np.int64, copy=False)

    ranges = [
        np.arange(
            int(center_multi[ax] - radii[ax]),
            int(center_multi[ax] + radii[ax] + 1),
            dtype=np.int64,
        )
        for ax in range(int(dim))
    ]
    mesh = np.meshgrid(*ranges, indexing="ij")
    box_multi = np.stack([m.reshape(-1) for m in mesh], axis=1)
    box_idx = np.ravel_multi_index(
        box_multi.T,
        dims=(int(mtot),) * int(dim),
    ).astype(np.int64, copy=False)
    box_idx.sort()
    if box_budget is not None and int(box_idx.size) > int(box_budget):
        raise ValueError(
            "constructed box exceeds box_budget: "
            f"|S_g|={int(box_idx.size)} > {int(box_budget)}"
        )
    tail_mask = np.ones((int(weights.size),), dtype=bool)
    tail_mask[box_idx] = False
    tail_idx = np.flatnonzero(tail_mask).astype(np.int64, copy=False)
    return BoxActiveSet(
        gamma=float(gamma),
        rho=rho,
        active_idx=active_idx,
        box_idx=box_idx,
        tail_idx=tail_idx,
        radii=radii,
        center_multi=center_multi,
        active_mode=str(active_mode),
        active_topk=None if active_topk is None else int(active_topk),
        active_tau=None if active_tau is None else float(active_tau),
    )


def format_box_tag(active: BoxActiveSet) -> str:
    if active.active_mode.lower() == "topk":
        return f"topk_{int(active.active_topk or 0)}"
    tau = float(active.active_tau or 0.0)
    if tau == 0.0:
        return "tau_0"
    exp = int(math.floor(math.log10(abs(tau)))) if tau != 0.0 else 0
    mant = tau / (10 ** exp) if tau != 0.0 else 0.0
    return f"tau_{mant:.3g}e{exp:+d}"
