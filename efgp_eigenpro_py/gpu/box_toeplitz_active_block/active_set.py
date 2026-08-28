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


def score_rank_order(rho: np.ndarray) -> np.ndarray:
    """Rank scores descending with flat Fourier index as a deterministic tie-break."""
    scores = np.asarray(rho, dtype=np.float64).reshape(-1)
    flat_index = np.arange(scores.size, dtype=np.int64)
    return np.lexsort((flat_index, -scores)).astype(np.int64, copy=False)


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
        order = score_rank_order(rho)
        return np.sort(order[:k].astype(np.int64, copy=False))
    if mode_key == "tau":
        if tau is None:
            raise ValueError("active_tau is required when active_mode='tau'.")
        return np.flatnonzero(rho > float(tau)).astype(np.int64, copy=False)
    raise ValueError(f"unknown active_mode={mode!r}; expected 'topk' or 'tau'.")


def _flatten_center_index(mtot: int, dim: int) -> np.ndarray:
    hm = (int(mtot) - 1) // 2
    return np.full((int(dim),), hm, dtype=np.int64)


def _build_box_active_set_from_indices(
    *,
    gamma: float,
    rho: np.ndarray,
    weights_size: int,
    mtot: int,
    dim: int,
    active_idx: np.ndarray,
    active_mode: str,
    active_topk: int | None,
    active_tau: float | None,
    box_budget: int | None,
) -> BoxActiveSet:
    active_idx = np.asarray(active_idx, dtype=np.int64).reshape(-1)
    center_multi = _flatten_center_index(mtot, dim)
    if active_idx.size == 0:
        all_idx = np.arange(int(weights_size), dtype=np.int64)
        return BoxActiveSet(
            gamma=float(gamma),
            rho=np.asarray(rho, dtype=np.float64).reshape(-1),
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
    tail_mask = np.ones((int(weights_size),), dtype=bool)
    tail_mask[box_idx] = False
    tail_idx = np.flatnonzero(tail_mask).astype(np.int64, copy=False)
    return BoxActiveSet(
        gamma=float(gamma),
        rho=np.asarray(rho, dtype=np.float64).reshape(-1),
        active_idx=active_idx,
        box_idx=box_idx,
        tail_idx=tail_idx,
        radii=radii,
        center_multi=center_multi,
        active_mode=str(active_mode),
        active_topk=None if active_topk is None else int(active_topk),
        active_tau=None if active_tau is None else float(active_tau),
    )


def build_memory_capped_topk_active_set(
    *,
    gamma: float,
    weights: np.ndarray,
    reg_lambda: float,
    mtot: int,
    dim: int,
    requested_topk: int,
    box_budget: int,
) -> tuple[BoxActiveSet, int]:
    """Build the largest requested score prefix whose centered box fits the cap.

    The score order is computed exactly once and the final active box is built
    exactly once.  The returned integer is the unbounded requested prefix's box
    size, which lets callers disclose whether deterministic capacity adaptation
    occurred.
    """
    if int(requested_topk) <= 0:
        raise ValueError("requested_topk must be positive.")
    if int(box_budget) <= 0:
        raise ValueError("box_budget must be positive.")
    weights_flat = np.asarray(weights, dtype=np.float64).reshape(-1)
    rho = compute_rho(gamma, weights_flat, reg_lambda)
    order = score_rank_order(rho)
    requested = min(int(requested_topk), int(order.size))
    prefix_order = order[:requested]
    shape = (int(mtot),) * int(dim)
    prefix_multi = np.stack(
        np.unravel_index(prefix_order, shape), axis=1
    ).astype(np.int64, copy=False)
    center = _flatten_center_index(int(mtot), int(dim))
    cumulative_radii = np.maximum.accumulate(
        np.abs(prefix_multi - center[None, :]), axis=0
    )
    box_sizes = np.prod(2 * cumulative_radii + 1, axis=1, dtype=np.int64)
    raw_box_size = int(box_sizes[-1])
    feasible = np.flatnonzero(box_sizes <= int(box_budget))
    if feasible.size == 0:
        raise ValueError(
            "box_budget is too small even for the highest-score mode's centered box."
        )
    effective_topk = int(feasible[-1]) + 1
    active_idx = np.sort(
        prefix_order[:effective_topk].astype(np.int64, copy=False)
    )
    active = _build_box_active_set_from_indices(
        gamma=float(gamma),
        rho=rho,
        weights_size=int(weights_flat.size),
        mtot=int(mtot),
        dim=int(dim),
        active_idx=active_idx,
        active_mode="topk",
        active_topk=effective_topk,
        active_tau=None,
        box_budget=int(box_budget),
    )
    return active, raw_box_size


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
    return _build_box_active_set_from_indices(
        gamma=float(gamma),
        rho=rho,
        weights_size=int(np.asarray(weights).size),
        mtot=int(mtot),
        dim=int(dim),
        active_idx=active_idx,
        active_mode=str(active_mode),
        active_topk=None if active_topk is None else int(active_topk),
        active_tau=None if active_tau is None else float(active_tau),
        box_budget=box_budget,
    )


def validate_precomputed_active_set(
    active: BoxActiveSet,
    *,
    gamma: float,
    weights: np.ndarray,
    reg_lambda: float,
    mtot: int,
    dim: int,
    active_mode: str,
    active_topk: int | None,
    active_tau: float | None,
    box_budget: int | None,
) -> None:
    """Fail closed if a reused active set came from different selection inputs."""
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    expected_rho = compute_rho(float(gamma), weights, float(reg_lambda))
    expected_center = _flatten_center_index(int(mtot), int(dim))
    valid = bool(
        np.isclose(float(active.gamma), float(gamma), rtol=1e-12, atol=1e-14)
        and np.asarray(active.rho).shape == expected_rho.shape
        and np.allclose(
            np.asarray(active.rho, dtype=np.float64),
            expected_rho,
            rtol=1e-12,
            atol=1e-14,
        )
        and np.array_equal(
            np.asarray(active.center_multi, dtype=np.int64).reshape(-1),
            expected_center,
        )
        and str(active.active_mode).lower() == str(active_mode).lower()
        and active.active_topk == active_topk
        and active.active_tau == active_tau
        and np.all(np.asarray(active.box_idx, dtype=np.int64) >= 0)
        and np.all(np.asarray(active.box_idx, dtype=np.int64) < weights.size)
        and (
            box_budget is None
            or int(active.box_idx.size) <= int(box_budget)
        )
    )
    if not valid:
        raise ValueError(
            "precomputed_active_set does not match the current system and "
            "preconditioner configuration"
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
