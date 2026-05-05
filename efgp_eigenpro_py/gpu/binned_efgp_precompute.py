"""
Binned EFGP precompute (C0/C1/C2): spatial binning plus type-1 NUFFT on bin centers.

独立模块，可与 ``v1_ops.gpu_precompute_v1`` 并行使用：在不改动现有流水线的前提下，
由调用方选择传入 ``v_tilde`` / ``b_tilde``（或后续的 ``Gf``）替换路径。

NUFFT 相位约定与仓库其余部分一致（``benchmark.py`` / ``v1_ops``）：默认 ``isign=-1``，
离散输出与 ``exp(i * isign * tphx · mode)`` 一致。C1/C2 **一阶**修正含因子 ``isign``；
二阶项系数不含 ``isign``（``(i·isign)^2=-1`` 已与 ``-0.5 ω^T Q ω`` 吸收）。
非均匀坐标 ``tphx`` 须在 ``[-pi, pi]``：默认若超出 ``pi + tol`` 则报错（避免静默 clip 改相位）；
仅当 ``allow_tphx_clip=True`` 时才裁剪（兼容极端用法）。

默认 ``x_center=0.5`` 对应 ``[0,1]^d``；若要与 ``gpu_precompute_v1`` 完全一致，
可传入 ``(x_min+x_max)/2``。

``use_sparse_bins=True`` 时使用 **稀疏 binning**（Python dict 路径，偏向正确性校验，大批量可能很慢）。
默认 ``build_binned_efgp_system(use_sparse_bins=False, use_gpu_dense_bins=True)`` 走 **融合 CUDA binning + GPU 上 compact + 批量 strengths 的 cuFINUFFT**（需 ``backend``）。

GPU dense（``return_gpu=True``）：``D``、``x_center`` 等按需求 H2D；bin 统计、fused launch、compact、NUFFT 组合与 ``b_tilde=D*rhs`` 尽量留在设备上。``gpu_timing=False`` 时不为细粒度计时而频繁 ``cuda`` 同步。

纯 CPU 或调试 dict 路径：``use_gpu_dense_bins=False``，可按需 ``use_sparse_bins=True``。
诊断字段会标明 binning 是否曾分配 dense。
"""
from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# 便于 ``python efgp_eigenpro_py/gpu/binned_efgp_precompute.py``（仓库根不在 PYTHONPATH 时）
_here = Path(__file__).resolve()
if _here.parent.name == "gpu":
    _repo_root = _here.parents[2]
    _sr = str(_repo_root)
    if _sr not in sys.path:
        sys.path.insert(0, _sr)

try:
    from ..discretization import generate_multi_index
    from ..nufft_ops import nufftnd1
except ImportError:
    from efgp_eigenpro_py.discretization import generate_multi_index
    from efgp_eigenpro_py.nufft_ops import nufftnd1

# 归一化盒子边界容差：允许极小浮点越界，随后 clip bin 索引。
X_NORM_TOL = 1e-10
# tphx 相对 ``pi`` 的容差：以内视为数值误差，允许 clip（见 ``tphx_from_centers``）。
TPHX_BOUNDARY_TOL = 1e-12

THETA_MAX: dict[str, dict[str, float]] = {
    "fast": {"C0": 1.00, "C1": 1.50, "C2": 2.00},
    "balanced": {"C0": 0.50, "C1": 0.80, "C2": 1.00},
    "accurate": {"C0": 0.25, "C1": 0.40, "C2": 0.60},
}


def _resolve_order(order: str, d: int) -> str:
    if order == "auto":
        return "C1"
    if order not in ("C0", "C1", "C2"):
        raise ValueError(f"order must be one of C0, C1, C2, auto; got {order!r}")
    return order


def _bytes_per_cell(order: str, d: int) -> int:
    if order == "C0":
        n_stat = 2
    elif order == "C1":
        n_stat = 2 + 2 * d
    elif order == "C2":
        n_stat = 2 + 2 * d + d * (d + 1)
    else:
        raise ValueError(order)
    return 8 * n_stat


def choose_binning_grid(
    N: int,
    d: int,
    h: float,
    m: int,
    order: str = "C1",
    quality: str = "balanced",
    memory_budget_bytes: int = 4 * 1024**3,
    min_avg_count: float = 8.0,
    r_max: int | None = None,
) -> tuple[int, int, float, float, dict[str, Any]]:
    """
    按规范第 7–8 节选择 ``r``，并返回 ``G, Delta, theta_actual`` 及诊断信息。
    """
    order = _resolve_order(order, d)
    if quality not in THETA_MAX:
        raise ValueError(f"quality must be one of {list(THETA_MAX.keys())}; got {quality!r}")

    theta_max = THETA_MAX[quality][order]
    r_phase = int(math.ceil(2.0 * math.pi * float(h) * int(m) * int(d) / float(theta_max)))

    bytes_per_cell = _bytes_per_cell(order, d)
    G_mem = max(1, int(memory_budget_bytes) // max(bytes_per_cell, 1))
    r_mem = int(G_mem ** (1.0 / d))

    G_occ = max(1, int(float(N) / float(min_avg_count)))
    r_occ = int(G_occ ** (1.0 / d))

    candidates = [r_phase, r_mem, r_occ]
    if r_max is not None:
        candidates.append(int(r_max))

    r = max(1, min(candidates))
    G = int(r**d)
    Delta = 1.0 / float(r)
    theta_actual = 2.0 * math.pi * float(h) * int(m) * int(d) / float(r)

    diagnostics: dict[str, Any] = {
        "order": order,
        "quality": quality,
        "theta_target": theta_max,
        "theta_actual": theta_actual,
        "r_phase": r_phase,
        "r_mem": r_mem,
        "r_occ": r_occ,
        "r_auto": r,
        "G": G,
        "Delta": Delta,
        "bytes_per_cell": bytes_per_cell,
        "estimated_dense_memory_bytes": G * bytes_per_cell,
        "avg_occupancy": float(N) / float(max(G, 1)),
        "compression_ratio": float(N) / float(max(G, 1)),
        "phase_target_met": r >= r_phase,
    }
    if r < r_phase:
        diagnostics["warning_phase"] = (
            "phase target not met; binning error may dominate"
        )
    return r, G, Delta, theta_actual, diagnostics


def linearize_indices(q: np.ndarray, r: int) -> np.ndarray:
    """q: (B, d), 坐标取值 ``0..r-1``；返回线性 bin 索引（与规范伪代码一致）。"""
    q = np.asarray(q, dtype=np.int64)
    if q.ndim != 2:
        raise ValueError("q must be 2D (B, d).")
    idx = q[:, 0].copy()
    mult = int(r)
    for k in range(1, q.shape[1]):
        idx = idx + mult * q[:, k].astype(np.int64, copy=False)
        mult *= int(r)
    return idx


def _pair_indices_upper(d: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(d) for j in range(i, d)]


def validate_normalized_X(X: np.ndarray, d: int, tol: float = X_NORM_TOL) -> np.ndarray:
    """
    检查 ``X`` 是否落在 ``[0,1]^d``（允许 ``tol`` 以内浮点误差）。
    返回 ``float64`` 视图供后续计算。
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != int(d):
        raise ValueError(f"X must have shape (N, {d}), got {X.shape}.")
    if np.any(X < -tol) or np.any(X > 1.0 + tol):
        raise ValueError(
            "X must be normalized to [0, 1]^d (within tol={}); "
            "got min={}, max={}.".format(tol, float(np.min(X)), float(np.max(X)))
        )
    return X


def tphx_from_centers(
    z_centers: np.ndarray,
    h: float,
    x_center: np.ndarray,
    *,
    allow_clip: bool = False,
    boundary_tol: float = TPHX_BOUNDARY_TOL,
) -> np.ndarray:
    """
    ``tphx = 2*pi*h*(z - x_center)``。

    默认 ``allow_clip=False``：若 ``abs(tphx) > pi + boundary_tol`` 则 ``ValueError``，
    否则仅将落在 ``(pi, pi+tol]`` 之类的数值噪声 clip 到 ``[-pi, nextafter(pi,0)]``。
    ``allow_clip=True`` 时对所有分量静默裁剪（不推荐，除非确认与 ``v1_ops`` 同款启发式）。
    """
    z_centers = np.asarray(z_centers, dtype=np.float64)
    xc = np.asarray(x_center, dtype=np.float64).reshape(1, -1)
    raw = 2.0 * np.pi * float(h) * (z_centers - xc)
    raw = np.asarray(raw, dtype=np.float64)
    upper = float(np.nextafter(np.pi, 0.0))
    amax = float(np.max(np.abs(raw)))
    if allow_clip:
        return np.clip(raw, -np.pi, upper)
    if amax > np.pi + float(boundary_tol):
        raise ValueError(
            "tphx outside [-pi, pi] beyond numerical tolerance; "
            "check h, scaling, x_center, or set allow_clip=True. "
            f"max|tphx|={amax}, tol={boundary_tol}."
        )
    return np.clip(raw, -np.pi, upper)


def direct_fourier_sum_type1(
    X_points: np.ndarray,
    weights: np.ndarray,
    modes: np.ndarray,
    h: float,
    x_center: np.ndarray,
    isign: int = -1,
    *,
    tphx_allow_clip: bool = False,
    max_points_per_chunk: int = 4096,
) -> np.ndarray:
    """
    直接求 ``sum_n w_n exp(i * isign * tphx_n · modes)``，与 type-1 NUFFT（``isign``）对照用。

    ``tphx`` 由 ``tphx_from_centers(X_points, ...)`` 生成（默认严格范围检查）。
    ``modes`` 须与 ``generate_multi_index`` 行顺序一致。

    按样本分块累加，避免分配 ``(N, M)`` 的 ``phase``（大 ``N``、大 ``M`` 时会爆 RAM）。
    """
    X_points = np.asarray(X_points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.complex128).reshape(-1)
    if w.shape[0] != X_points.shape[0]:
        raise ValueError("weights length must match X_points.")
    modes = np.asarray(modes, dtype=np.float64)
    n = int(X_points.shape[0])
    m = int(modes.shape[0])
    if n == 0:
        return np.zeros(m, dtype=np.complex128)
    out = np.zeros(m, dtype=np.complex128)
    isign_f = float(isign)
    bs = max(1, int(max_points_per_chunk))
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        tphx_chunk = tphx_from_centers(X_points[i0:i1], h, x_center, allow_clip=tphx_allow_clip)
        phase = tphx_chunk @ modes.T
        contrib = (w[i0:i1, None] * np.exp(1j * isign_f * phase)).sum(axis=0)
        out += contrib
    return out


def _bin_indices_and_delta(X64: np.ndarray, rr: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """``X64`` 已为 ``float64`` 且校验过盒子；返回 ``idx, delta``。"""
    q = np.floor(rr * X64).astype(np.int64, copy=False)
    np.clip(q, 0, rr - 1, out=q)
    idx = linearize_indices(q, rr)
    z = (q.astype(np.float64, copy=False) + 0.5) / float(rr)
    delta = X64 - z
    return idx, delta


def _aggregate_chunk_to_unique_bins(
    idx: np.ndarray,
    y64: np.ndarray,
    delta: np.ndarray,
    d: int,
    order: str,
    pairs_c2: list[tuple[int, int]],
    rt: type,
) -> dict[str, Any]:
    """单块内 ``np.unique`` + ``bincount``，长度 ``nu << N``。"""
    u, inv = np.unique(idx.astype(np.int64, copy=False), return_inverse=True)
    nu = int(u.shape[0])
    npts = int(idx.shape[0])
    c_loc = np.bincount(inv, weights=np.ones(npts, dtype=rt), minlength=nu)
    s_loc = np.bincount(inv, weights=y64, minlength=nu)
    out: dict[str, Any] = {"u": u.astype(np.int64, copy=False), "c": c_loc, "s": s_loc}
    if order in ("C1", "C2"):
        a_loc = np.zeros((nu, d), dtype=rt)
        ay_loc = np.zeros((nu, d), dtype=rt)
        for k in range(d):
            a_loc[:, k] = np.bincount(inv, weights=delta[:, k], minlength=nu)
            ay_loc[:, k] = np.bincount(inv, weights=y64 * delta[:, k], minlength=nu)
        out["a"] = a_loc
        out["ay"] = ay_loc
    if order == "C2":
        npairs = len(pairs_c2)
        Q_loc = np.zeros((nu, npairs), dtype=rt)
        Qy_loc = np.zeros((nu, npairs), dtype=rt)
        for p, (i, j) in enumerate(pairs_c2):
            dij = delta[:, i] * delta[:, j]
            Q_loc[:, p] = np.bincount(inv, weights=dij, minlength=nu)
            Qy_loc[:, p] = np.bincount(inv, weights=y64 * dij, minlength=nu)
        out["Q"] = Q_loc
        out["Qy"] = Qy_loc
    return out


def _linear_idx_to_q_z_numpy(idx_occ: np.ndarray, rr: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """线性 bin 索引 ``0..G-1`` 解码为整数格点 ``q`` 与 bin 中心 ``z``（CPU）。"""
    idx = np.asarray(idx_occ, dtype=np.int64).reshape(-1)
    q = np.zeros((idx.size, int(d)), dtype=np.int64)
    rem = idx.copy()
    for k in range(int(d)):
        q[:, k] = rem % rr
        rem //= rr
    z = (q.astype(np.float64, copy=False) + 0.5) / float(rr)
    return q, z


class _SparseGlobalBins:
    """
    跨 chunk 合并 unique bin，不占 ``G`` 长度向量。

    实现为 Python ``dict`` + 逐 bin 更新；正确性优先。若单 chunk 内 unique bin 极多、
    chunk 数量大，可能成为瓶颈，后续可改为 sorted-array merge / 批量归并。
    """

    __slots__ = ("d", "order", "pairs_c2", "c", "s", "a", "ay", "Q", "Qy")

    def __init__(self, d: int, order: str) -> None:
        self.d = int(d)
        self.order = order
        self.pairs_c2 = _pair_indices_upper(d) if order == "C2" else []
        self.c: dict[int, float] = {}
        self.s: dict[int, float] = {}
        self.a: dict[int, np.ndarray] = {}
        self.ay: dict[int, np.ndarray] = {}
        self.Q: dict[int, np.ndarray] = {}
        self.Qy: dict[int, np.ndarray] = {}

    def ingest(self, pack: dict[str, Any]) -> None:
        u = pack["u"]
        for j in range(len(u)):
            g = int(u[j])
            self.c[g] = self.c.get(g, 0.0) + float(pack["c"][j])
            self.s[g] = self.s.get(g, 0.0) + float(pack["s"][j])
            if self.order in ("C1", "C2"):
                aj = pack["a"][j]
                ayj = pack["ay"][j]
                if g not in self.a:
                    self.a[g] = np.zeros(self.d, dtype=np.float64)
                    self.ay[g] = np.zeros(self.d, dtype=np.float64)
                self.a[g] += aj
                self.ay[g] += ayj
            if self.order == "C2":
                Qj = pack["Q"][j]
                Qyj = pack["Qy"][j]
                if g not in self.Q:
                    nq = len(self.pairs_c2)
                    self.Q[g] = np.zeros(nq, dtype=np.float64)
                    self.Qy[g] = np.zeros(nq, dtype=np.float64)
                self.Q[g] += Qj
                self.Qy[g] += Qyj

    def to_sparse_bin_stats(self, r: int) -> dict[str, Any]:
        keys = sorted(self.c.keys())
        idx_occ = np.array(keys, dtype=np.int64)
        rr = int(r)
        d = self.d
        q, z = _linear_idx_to_q_z_numpy(idx_occ, rr, d)
        order = self.order
        out: dict[str, Any] = {
            "idx_occ": idx_occ,
            "q_occ": q,
            "z_occ": z,
            "c_occ": np.array([self.c[g] for g in keys], dtype=np.float64),
            "s_occ": np.array([self.s[g] for g in keys], dtype=np.float64),
            "r": rr,
            "d": d,
            "order": order,
            "dense": False,
            "binning_dense_allocated": False,
        }
        if order in ("C1", "C2"):
            out["a_occ"] = np.stack([self.a[g] for g in keys], axis=0)
            out["ay_occ"] = np.stack([self.ay[g] for g in keys], axis=0)
        if order == "C2":
            out["Q_occ"] = np.stack([self.Q[g] for g in keys], axis=0)
            out["Qy_occ"] = np.stack([self.Qy[g] for g in keys], axis=0)
            out["pairs"] = self.pairs_c2
        return out


def _grid_diag_for_fixed_r(
    r_use: int,
    N: int,
    d: int,
    h: float,
    m: int,
    order_eff: str,
    quality: str,
    theta_target: float,
) -> tuple[int, float, float, dict[str, Any]]:
    """用户指定 ``r`` 时与 ``choose_binning_grid`` 对齐的诊断块。"""
    r_u = max(1, int(r_use))
    G = int(r_u**d)
    Delta = 1.0 / float(r_u)
    theta_actual = 2.0 * math.pi * float(h) * int(m) * int(d) / float(r_u)
    bytes_pc = _bytes_per_cell(order_eff, int(d))
    r_phase = int(math.ceil(2.0 * math.pi * float(h) * int(m) * int(d) / float(theta_target)))
    grid_diag: dict[str, Any] = {
        "order": order_eff,
        "quality": quality,
        "theta_target": theta_target,
        "theta_actual": theta_actual,
        "r_phase": r_phase,
        "r_auto": r_u,
        "G": G,
        "Delta": Delta,
        "bytes_per_cell": bytes_pc,
        "estimated_dense_memory_bytes": G * bytes_pc,
        "avg_occupancy": float(N) / float(max(G, 1)),
        "compression_ratio": float(N) / float(max(G, 1)),
        "phase_target_met": r_u >= r_phase,
    }
    if r_u < r_phase:
        grid_diag["warning_phase"] = (
            "phase target not met; binning error may dominate"
        )
    return G, Delta, theta_actual, grid_diag


def build_bin_stats_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    r: int,
    d: int,
    order: str,
    *,
    use_dense_bins: bool = True,
) -> dict[str, Any]:
    """
    向量化 binning。

    ``use_dense_bins=True``：分配 ``G=r^d`` 稠密数组。
    ``use_dense_bins=False``：稀疏 binning，不占 ``G`` 向量（输出同 ``sparsify_bin_stats``）。
    """
    order = _resolve_order(order, d)
    X64 = validate_normalized_X(X, d)
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    if X64.shape[0] != y64.shape[0]:
        raise ValueError("X and y length mismatch.")

    rr = int(r)
    rt = np.float64
    pairs_c2 = _pair_indices_upper(d) if order == "C2" else []

    idx, delta = _bin_indices_and_delta(X64, rr, d)

    if not use_dense_bins:
        pack = _aggregate_chunk_to_unique_bins(idx, y64, delta, d, order, pairs_c2, rt)
        glob = _SparseGlobalBins(d, order)
        glob.ingest(pack)
        return glob.to_sparse_bin_stats(rr)

    G = rr**d
    c = np.bincount(idx, weights=np.ones(idx.shape[0], dtype=rt), minlength=G).astype(rt, copy=False)
    s = np.bincount(idx, weights=y64, minlength=G).astype(rt, copy=False)

    out: dict[str, Any] = {
        "c": c,
        "s": s,
        "r": rr,
        "d": d,
        "order": order,
        "dense": True,
        "binning_dense_allocated": True,
    }

    if order in ("C1", "C2"):
        a = np.zeros((G, d), dtype=rt)
        ay = np.zeros((G, d), dtype=rt)
        for k in range(d):
            a[:, k] = np.bincount(idx, weights=delta[:, k], minlength=G)
            ay[:, k] = np.bincount(idx, weights=y64 * delta[:, k], minlength=G)
        out["a"] = a
        out["ay"] = ay

    if order == "C2":
        pairs = pairs_c2
        Q = np.zeros((G, len(pairs)), dtype=rt)
        Qy = np.zeros((G, len(pairs)), dtype=rt)
        for p, (i, j) in enumerate(pairs):
            dij = delta[:, i] * delta[:, j]
            Q[:, p] = np.bincount(idx, weights=dij, minlength=G)
            Qy[:, p] = np.bincount(idx, weights=y64 * dij, minlength=G)
        out["Q"] = Q
        out["Qy"] = Qy
        out["pairs"] = pairs

    return out


def build_bin_stats_streaming(
    X_loader: Iterator[np.ndarray],
    y_loader: Iterator[np.ndarray],
    r: int,
    d: int,
    order: str,
    *,
    use_dense_bins: bool = True,
) -> dict[str, Any]:
    """多块流式：``use_dense_bins=False`` 时用 dict 合并，不分配 ``G`` 长度向量。"""
    order = _resolve_order(order, d)
    rr = int(r)
    rt = np.float64
    pairs_c2 = _pair_indices_upper(d) if order == "C2" else []

    n_seen = 0

    if use_dense_bins:
        G = rr**d
        c = np.zeros(G, dtype=rt)
        s = np.zeros(G, dtype=rt)
        if order in ("C1", "C2"):
            a = np.zeros((G, d), dtype=rt)
            ay = np.zeros((G, d), dtype=rt)
        if order == "C2":
            Q = np.zeros((G, len(pairs_c2)), dtype=rt)
            Qy = np.zeros((G, len(pairs_c2)), dtype=rt)

        for X_chunk, y_chunk in zip(X_loader, y_loader):
            xc = np.asarray(X_chunk)
            yc = np.asarray(y_chunk, dtype=np.float64).reshape(-1)
            if xc.shape[0] != yc.shape[0]:
                raise ValueError("chunk X/y length mismatch in streaming loader.")
            n_seen += int(xc.shape[0])
            Xv = validate_normalized_X(xc, d)
            idx, delta = _bin_indices_and_delta(Xv, rr, d)
            np.add.at(c, idx, 1.0)
            np.add.at(s, idx, yc)
            if order in ("C1", "C2"):
                for k in range(int(d)):
                    np.add.at(a[:, k], idx, delta[:, k])
                    np.add.at(ay[:, k], idx, yc * delta[:, k])
            if order == "C2":
                for p, (i, j) in enumerate(pairs_c2):
                    dij = delta[:, i] * delta[:, j]
                    np.add.at(Q[:, p], idx, dij)
                    np.add.at(Qy[:, p], idx, yc * dij)

        out: dict[str, Any] = {
            "c": c,
            "s": s,
            "r": rr,
            "d": d,
            "order": order,
            "dense": True,
            "binning_dense_allocated": True,
            "n_seen": int(n_seen),
        }
        if order in ("C1", "C2"):
            out["a"] = a
            out["ay"] = ay
        if order == "C2":
            out["Q"] = Q
            out["Qy"] = Qy
            out["pairs"] = pairs_c2
        return out

    glob = _SparseGlobalBins(d, order)
    for X_chunk, y_chunk in zip(X_loader, y_loader):
        X64 = validate_normalized_X(X_chunk, d)
        y64 = np.asarray(y_chunk, dtype=np.float64).reshape(-1)
        if X64.shape[0] != y64.shape[0]:
            raise ValueError("chunk X/y length mismatch.")
        n_seen += int(X64.shape[0])
        idx, delta = _bin_indices_and_delta(X64, rr, d)
        pack = _aggregate_chunk_to_unique_bins(idx, y64, delta, d, order, pairs_c2, rt)
        glob.ingest(pack)
    out_sp = glob.to_sparse_bin_stats(rr)
    out_sp["n_seen"] = int(n_seen)
    return out_sp


def sparsify_bin_stats(bin_stats: dict[str, Any]) -> dict[str, Any]:
    """仅保留 ``c>0`` 的格子（由 dense 统计压缩；本就 sparse 的无需再调）。"""
    if not bin_stats.get("dense", True):
        return bin_stats
    c = np.asarray(bin_stats["c"])
    mask = c > 0
    idx_occ = np.nonzero(mask)[0].astype(np.int64)
    rr = int(bin_stats["r"])
    d = int(bin_stats["d"])
    order = str(bin_stats["order"])

    q, z = _linear_idx_to_q_z_numpy(idx_occ, rr, d)

    out: dict[str, Any] = {
        "idx_occ": idx_occ,
        "q_occ": q,
        "z_occ": z,
        "c_occ": c[mask],
        "s_occ": np.asarray(bin_stats["s"])[mask],
        "r": rr,
        "d": d,
        "order": order,
        "dense": False,
        "binning_dense_allocated": bool(bin_stats.get("binning_dense_allocated", True)),
    }
    if order in ("C1", "C2"):
        out["a_occ"] = np.asarray(bin_stats["a"])[mask, :]
        out["ay_occ"] = np.asarray(bin_stats["ay"])[mask, :]
    if order == "C2":
        out["Q_occ"] = np.asarray(bin_stats["Q"])[mask, :]
        out["Qy_occ"] = np.asarray(bin_stats["Qy"])[mask, :]
        out["pairs"] = bin_stats["pairs"]
    return out


def _import_cupy_binned() -> Any:
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CuPy is required for GPU binning (build_bin_stats_from_arrays_gpu_dense).") from exc
    return cp


def _sync_cuda_stream(xp: Any) -> None:
    """等待 CuPy 已提交 CUDA 内核完成（wall 计时有意义）。无 CUDA 或无 Stream 时为 no-op。"""
    if xp is None:
        return
    cuda_mod = getattr(xp, "cuda", None)
    stream_cls = getattr(cuda_mod, "Stream", None) if cuda_mod is not None else None
    if stream_cls is None:
        return
    try:
        stream_cls.null.synchronize()
    except Exception:
        pass


def _sync_cuda_for_timings(xp: Any, timings_s: dict[str, float] | None) -> None:
    if timings_s is not None:
        _sync_cuda_stream(xp)


# 单次 kernel 遍历 N，对 occupied bin 原子累加全部标量矩（等价于多次 bincount 合并）。
_FUSED_BIN_MAX_D = 8
_FUSED_DENSE_CUDA = r"""
extern "C" __global__ void fused_dense_bin_accum(
    const double* __restrict__ X,
    const double* __restrict__ y,
    int N,
    int rr,
    int d,
    long long G,
    int order_code,
    int npairs,
    const int* __restrict__ pi,
    const int* __restrict__ pj,
    double* __restrict__ c_,
    double* __restrict__ s_,
    double* __restrict__ a_,
    double* __restrict__ ay_,
    double* __restrict__ Q_,
    double* __restrict__ Qy_)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    const double yi = y[i];
    double delta[8];
    int qq;
    double xv;
    long long mult;
    long long g;
    int k;

    mult = 1;
    g = 0;
#pragma unroll 1
    for (k = 0; k < d; ++k)
    {
        xv = X[(size_t)i * (size_t)d + (size_t)k];
        qq = (int)floor((double)rr * xv);
        if (qq < 0)
            qq = 0;
        if (qq >= rr)
            qq = rr - 1;
        double zc = ((double)qq + 0.5) / (double)rr;
        delta[(size_t)k] = xv - zc;
        g += mult * (long long)qq;
        mult *= (long long)rr;
    }
    if (g < 0 || g >= G)
        return;

    atomicAdd(&c_[g], 1.0);
    atomicAdd(&s_[g], yi);

    if (order_code >= 1)
    {
        long long gd = g * (long long)d;
        for (int kk = 0; kk < d; ++kk)
        {
            long long ix = gd + kk;
            double dk = delta[kk];
            atomicAdd(&a_[ix], dk);
            atomicAdd(&ay_[ix], yi * dk);
        }
    }
    if (order_code >= 2)
    {
        long long gn = g * (long long)npairs;
        for (int p = 0; p < npairs; ++p)
        {
            int ii = pi[p];
            int jj = pj[p];
            double dij = delta[ii] * delta[jj];
            long long ix = gn + (long long)p;
            atomicAdd(&Q_[ix], dij);
            atomicAdd(&Qy_[ix], yi * dij);
        }
    }
}
"""


_fused_bin_kernel_cache: dict[int, Any] = {}
_gpu_dummy_scratch: dict[int, Any] = {}  # per-device 1-double placeholder
# C2：`pi/pj` 只依赖 ``d`` 与 device，缓存避免每 chunk 重复 H2D。
_pair_ij_gpu_cache: dict[tuple[int, int], tuple[Any, Any]] = {}


def _get_fused_bin_kernel(cp: Any) -> Any:
    dev_id = int(cp.cuda.Device().id)
    kern = _fused_bin_kernel_cache.get(dev_id)
    if kern is None:
        kern = cp.RawKernel(_FUSED_DENSE_CUDA, "fused_dense_bin_accum")
        _fused_bin_kernel_cache[dev_id] = kern
    return kern


def _order_code_from_str(order: str) -> int:
    if order == "C0":
        return 0
    if order == "C1":
        return 1
    if order == "C2":
        return 2
    raise ValueError(order)


_ORDER_RANK = {"C0": 0, "C1": 1, "C2": 2}


def _resolved_order_rank(order: str) -> int:
    if order not in _ORDER_RANK:
        raise ValueError(f"unknown order rank key: {order!r}")
    return _ORDER_RANK[order]


def _gpu_dense_stats_tail_view(stats_full: dict[str, Any], order_target: str, d: int) -> dict[str, Any]:
    """同一套 dense GPU 缓冲区上截取 C0/C1/C2 子集，避免 auto-downgrade 重复 fused binning。"""
    full_order = _resolve_order(str(stats_full["order"]), d)
    o_t = _resolve_order(order_target, d)
    if _resolved_order_rank(full_order) < _resolved_order_rank(o_t):
        raise ValueError(f"cannot view order {o_t} from stats computed with lower order {full_order}")
    rr = int(stats_full["r"])
    dd = int(stats_full["d"])
    out: dict[str, Any] = {
        "c": stats_full["c"],
        "s": stats_full["s"],
        "r": rr,
        "d": dd,
        "order": o_t,
        "dense": True,
        "binning_dense_allocated": bool(stats_full.get("binning_dense_allocated", True)),
        "on_gpu": True,
    }
    if o_t in ("C1", "C2"):
        out["a"] = stats_full["a"]
        out["ay"] = stats_full["ay"]
    if o_t == "C2":
        out["Q"] = stats_full["Q"]
        out["Qy"] = stats_full["Qy"]
        out["pairs"] = stats_full["pairs"]
    return out


def _get_pair_ij_gpu_arrays(
    cp: Any,
    d: int,
    pairs_c2: list[tuple[int, int]],
) -> tuple[Any, Any]:
    dev_id = int(cp.cuda.Device().id)
    if not pairs_c2:
        z = cp.zeros(1, dtype=cp.int32)
        return z, z
    key = (int(d), dev_id)
    cached = _pair_ij_gpu_cache.get(key)
    if cached is not None:
        return cached
    pi_np = np.array([p[0] for p in pairs_c2], dtype=np.int32)
    pj_np = np.array([p[1] for p in pairs_c2], dtype=np.int32)
    pi_g = cp.asarray(pi_np)
    pj_g = cp.asarray(pj_np)
    _pair_ij_gpu_cache[key] = (pi_g, pj_g)
    return pi_g, pj_g


def _occupied_bin_atomic_skew_diag(c_occ: Any) -> dict[str, float]:
    """由 occupied bin 的 ``c_occ``（NumPy/CuPy）计算 skew 等指标（宿主标量）。"""
    co = np.asarray(_device_to_numpy(c_occ), dtype=np.float64).reshape(-1)
    if co.size <= 0:
        return {
            "bin_count_max_occ": 0.0,
            "bin_count_mean_occ": 0.0,
            "bin_count_p99_occ": 0.0,
            "atomic_skew_occ_counts": float("nan"),
        }
    mx = float(co.max())
    mu = float(co.mean())
    p99 = float(np.percentile(co, 99.0))
    skew = mx / mu if mu > 1e-30 else float("inf")
    return {
        "bin_count_max_occ": mx,
        "bin_count_mean_occ": mu,
        "bin_count_p99_occ": p99,
        "atomic_skew_occ_counts": skew,
    }


def _bin_count_skew_from_bin_stats(st: dict[str, Any]) -> dict[str, float]:
    """优先用 compact ``c_occ``，否则用 dense ``c[c>0]``（宿主上算分位数）。"""
    co = st.get("c_occ")
    if co is not None:
        return _occupied_bin_atomic_skew_diag(co)
    c = st.get("c")
    if c is None:
        return {
            "bin_count_max_occ": 0.0,
            "bin_count_mean_occ": 0.0,
            "bin_count_p99_occ": 0.0,
            "atomic_skew_occ_counts": float("nan"),
        }
    cnp = np.asarray(_device_to_numpy(c), dtype=np.float64).reshape(-1)
    pos = cnp[cnp > 0]
    if pos.size <= 0:
        return {
            "bin_count_max_occ": 0.0,
            "bin_count_mean_occ": 0.0,
            "bin_count_p99_occ": 0.0,
            "atomic_skew_occ_counts": float("nan"),
        }
    mx = float(pos.max())
    mu = float(pos.mean())
    p99 = float(np.percentile(pos, 99.0))
    skew = mx / mu if mu > 1e-30 else float("inf")
    return {
        "bin_count_max_occ": mx,
        "bin_count_mean_occ": mu,
        "bin_count_p99_occ": p99,
        "atomic_skew_occ_counts": skew,
    }


def _gpu_dummy_1(cp: Any) -> Any:
    dev_id = int(cp.cuda.Device().id)
    if dev_id not in _gpu_dummy_scratch:
        _gpu_dummy_scratch[dev_id] = cp.zeros(1, dtype=cp.float64)
    return _gpu_dummy_scratch[dev_id]


def _launch_fused_dense_bin_kernel(
    cp: Any,
    Xg: Any,
    yg: Any,
    rr: int,
    d: int,
    G: int,
    order_str: str,
    pairs_c2: list[tuple[int, int]],
    c: Any,
    s: Any,
    a: Any | None,
    ay: Any | None,
    Q: Any | None,
    Qy: Any | None,
) -> None:
    if d > _FUSED_BIN_MAX_D:
        raise ValueError(f"fused_dense_bin_accum supports d <= {_FUSED_BIN_MAX_D}, got {d}")
    oc = _order_code_from_str(order_str)
    npairs = len(pairs_c2) if oc >= 2 else 0
    if npairs > 0:
        pi_g, pj_g = _get_pair_ij_gpu_arrays(cp, int(d), pairs_c2)
    else:
        z = cp.zeros(1, dtype=cp.int32)
        pi_g = pj_g = z
    dm = _gpu_dummy_1(cp)
    ap = dm if oc < 1 or a is None else a
    ayp = dm if oc < 1 or ay is None else ay
    Qp = dm if oc < 2 or Q is None else Q
    Qyp = dm if oc < 2 or Qy is None else Qy

    kern = _get_fused_bin_kernel(cp)
    n = int(Xg.shape[0])
    threads = 256
    blocks = max(1, (n + threads - 1) // threads)
    kern(
        (blocks,),
        (threads,),
        (
            Xg,
            yg,
            np.int32(n),
            np.int32(int(rr)),
            np.int32(int(d)),
            np.int64(int(G)),
            np.int32(int(oc)),
            np.int32(int(npairs)),
            pi_g,
            pj_g,
            c,
            s,
            ap,
            ayp,
            Qp,
            Qyp,
        ),
    )


def build_bin_stats_from_arrays_gpu_dense(
    X: np.ndarray | None,
    y: np.ndarray | None,
    r: int,
    d: int,
    order: str,
    *,
    xp: Any | None = None,
    dtype: Any | None = None,
    timings_s: dict[str, float] | None = None,
    gpu_chunk_rows: int | None = None,
    skip_normalized_check: bool = False,
    X_gpu: Any | None = None,
    y_gpu: Any | None = None,
) -> dict[str, Any]:
    """
    在 GPU 上构造 ``G=r^d`` 稠密度量：使用 **单次 fused kernel**（可多次 launch 切块）累加 C0/C1/C2 矩。

    若 ``X_gpu`` / ``y_gpu`` 已提供（CuPy，形状 ``(N,d)`` / ``(N,)``），则不再从宿主上传整块 ``X``，
    仅按 ``gpu_chunk_rows`` 对 **显存子区间** 分块 launch（子块仍在 GPU 上切片，无 H2D）。
    否则从 NumPy ``X,y`` 分块上传（宿主仍持全量数据）。

    **dtype**：CUDA kernel 仅为 **float64**；仅允许 ``dtype=None`` 或 ``cp.float64``（等价），否则抛错。
    """
    order = _resolve_order(order, d)
    cp = xp if xp is not None else _import_cupy_binned()
    if dtype is None:
        rt = cp.float64
    else:
        dt = cp.dtype(dtype)
        if dt != cp.dtype("float64"):
            raise ValueError(
                "GPU fused binning uses double-precision CUDA kernel only; "
                "use dtype=None or cp.float64."
            )
        rt = cp.float64

    rr = int(r)
    pairs_c2 = _pair_indices_upper(d) if order == "C2" else []
    G = int(rr**d)
    if int(d) > _FUSED_BIN_MAX_D:
        raise ValueError(
            f"GPU fused binning requires d <= {_FUSED_BIN_MAX_D}; use CPU path or smaller d."
        )

    c = cp.zeros(G, dtype=rt)
    s = cp.zeros(G, dtype=rt)
    a = ay = Q = Qy = None
    if order in ("C1", "C2"):
        a = cp.zeros((G, int(d)), dtype=rt)
        ay = cp.zeros((G, int(d)), dtype=rt)
    if order == "C2":
        npairs = len(pairs_c2)
        Q = cp.zeros((G, npairs), dtype=rt)
        Qy = cp.zeros((G, npairs), dtype=rt)

    _sync_cuda_for_timings(cp, timings_s)
    t_h2d0 = time.perf_counter()
    t_fm0 = time.perf_counter()

    if X_gpu is not None:
        if y_gpu is None:
            raise ValueError("build_bin_stats_from_arrays_gpu_dense: y_gpu required with X_gpu")
        Xg_full = cp.asarray(X_gpu, dtype=cp.float64)
        yg_full = cp.asarray(y_gpu, dtype=cp.float64).reshape(-1)
        if Xg_full.ndim != 2 or int(Xg_full.shape[1]) != int(d):
            raise ValueError("X_gpu must have shape (N, d)")
        n_full = int(Xg_full.shape[0])
        if yg_full.shape[0] != n_full:
            raise ValueError("X_gpu / y_gpu length mismatch")
        chunk = n_full if gpu_chunk_rows is None else min(n_full, int(gpu_chunk_rows))
        for s0 in range(0, n_full, chunk):
            s1 = min(s0 + chunk, n_full)
            _launch_fused_dense_bin_kernel(
                cp,
                Xg_full[s0:s1],
                yg_full[s0:s1],
                rr,
                int(d),
                G,
                order,
                pairs_c2,
                c,
                s,
                a,
                ay,
                Q,
                Qy,
            )
    else:
        if X is None or y is None:
            raise ValueError("Provide X,y on host or X_gpu,y_gpu on device")
        if skip_normalized_check:
            X64 = np.asarray(X, dtype=np.float64).reshape(-1, int(d))
            if X64.ndim != 2 or X64.shape[1] != int(d):
                raise ValueError("X shape mismatch.")
        else:
            X64 = validate_normalized_X(X, d)
        y64 = np.asarray(y, dtype=np.float64).reshape(-1)
        if X64.shape[0] != y64.shape[0]:
            raise ValueError("X and y length mismatch.")
        n_full = int(X64.shape[0])
        chunk = min(n_full, int(gpu_chunk_rows or n_full))
        for s0 in range(0, n_full, chunk):
            s1 = min(s0 + chunk, n_full)
            Xg = cp.asarray(X64[s0:s1], dtype=cp.float64)
            yg = cp.asarray(y64[s0:s1], dtype=cp.float64).reshape(-1)
            _launch_fused_dense_bin_kernel(
                cp, Xg, yg, rr, int(d), G, order, pairs_c2, c, s, a, ay, Q, Qy
            )

    _sync_cuda_for_timings(cp, timings_s)
    if timings_s is not None:
        timings_s["t_h2d_xy_s"] = float(time.perf_counter() - t_h2d0)
        timings_s["t_gpu_fused_bin_moments_s"] = float(time.perf_counter() - t_fm0)

    out: dict[str, Any] = {
        "c": c,
        "s": s,
        "r": rr,
        "d": int(d),
        "order": order,
        "dense": True,
        "binning_dense_allocated": True,
        "on_gpu": True,
    }
    if a is not None:
        assert ay is not None
        out["a"] = a
        out["ay"] = ay
    if order == "C2":
        assert Q is not None and Qy is not None
        out["Q"] = Q
        out["Qy"] = Qy
        out["pairs"] = pairs_c2

    _sync_cuda_for_timings(cp, timings_s)

    return out


def sparsify_bin_stats_gpu(
    bin_stats: dict[str, Any],
    xp: Any | None = None,
    *,
    timings_s: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Compress dense GPU stats to occupied bins only; arrays remain on GPU.
    """
    if not bin_stats.get("dense", True):
        return bin_stats
    if not bin_stats.get("on_gpu"):
        raise ValueError("sparsify_bin_stats_gpu expects dense GPU stats (on_gpu=True).")

    cp = xp if xp is not None else _import_cupy_binned()
    _sync_cuda_for_timings(cp, timings_s)
    t0 = time.perf_counter()
    c = bin_stats["c"]
    mask = c > 0
    idx_occ = cp.nonzero(mask)[0].astype(cp.int64)
    rr = int(bin_stats["r"])
    d = int(bin_stats["d"])
    order = str(bin_stats["order"])

    nocc = int(idx_occ.size)
    q = cp.zeros((nocc, d), dtype=cp.int64)
    rem = idx_occ.astype(cp.int64, copy=True)
    for k in range(d):
        q[:, k] = rem % rr
        rem = rem // rr

    z_occ = (q.astype(cp.float64, copy=False) + 0.5) / float(rr)

    out: dict[str, Any] = {
        "idx_occ": idx_occ,
        "q_occ": q,
        "z_occ": z_occ,
        "c_occ": c[idx_occ],
        "s_occ": bin_stats["s"][idx_occ],
        "r": rr,
        "d": d,
        "order": order,
        "dense": False,
        "on_gpu": True,
        "binning_dense_allocated": bool(bin_stats.get("binning_dense_allocated", True)),
    }
    if order in ("C1", "C2"):
        out["a_occ"] = bin_stats["a"][idx_occ, :]
        out["ay_occ"] = bin_stats["ay"][idx_occ, :]
    if order == "C2":
        out["Q_occ"] = bin_stats["Q"][idx_occ, :]
        out["Qy_occ"] = bin_stats["Qy"][idx_occ, :]
        out["pairs"] = bin_stats["pairs"]
    _sync_cuda_for_timings(cp, timings_s)
    if timings_s is not None:
        timings_s["t_compact_occupied_s"] = float(time.perf_counter() - t0)
    return out


def tphx_from_centers_gpu(
    z_centers: Any,
    h: float,
    x_center: np.ndarray,
    xp: Any,
    *,
    allow_clip: bool = False,
    boundary_tol: float = TPHX_BOUNDARY_TOL,
) -> Any:
    """GPU analogue of ``tphx_from_centers`` (``z_centers`` CuPy array)."""
    cp = xp
    z_centers = cp.asarray(z_centers, dtype=cp.float64)
    xc = cp.asarray(np.asarray(x_center, dtype=np.float64).reshape(1, -1), dtype=cp.float64)
    raw = 2.0 * cp.pi * float(h) * (z_centers - xc)
    upper = float(np.nextafter(np.pi, 0.0))
    amax = float(cp.max(cp.abs(raw)))
    if allow_clip:
        return cp.clip(raw, -cp.pi, upper)
    if amax > np.pi + float(boundary_tol):
        raise ValueError(
            "tphx outside [-pi, pi] beyond numerical tolerance; "
            "check h, scaling, x_center, or set allow_clip=True. "
            f"max|tphx|={amax}, tol={boundary_tol}."
        )
    return cp.clip(raw, -cp.pi, upper)


def _dispatch_type1_keep_gpu(
    backend: Any,
    tphx_gpu: Any,
    weights_real: Any,
    dim: int,
    ms: int,
    eps: float,
    isign: int,
) -> Any:
    """Like ``_dispatch_type1`` but avoids host round-trip; returns device FFT coeffs."""
    xp = backend.xp
    w = xp.asarray(weights_real, dtype=xp.float64).reshape(-1)
    c = w.astype(xp.complex128)
    tphx_gpu = xp.asarray(tphx_gpu, dtype=xp.float64)
    return _type1_coeffs_gpu(backend, tphx_gpu, c, dim, ms, eps, isign)


def _dispatch_type1_keep_gpu_batched(
    backend: Any,
    tphx_gpu: Any,
    strengths_real: Any,
    dim: int,
    ms: int,
    eps: float,
    isign: int,
) -> Any:
    """多块实数强度 ``(n_tr, M)`` 一次批量 type-1，返回 ``(n_tr, n_modes)`` _device complex。"""
    xp = backend.xp
    w = xp.asarray(strengths_real, dtype=xp.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    cplx = w.astype(xp.complex128)
    tphx_gpu = xp.asarray(tphx_gpu, dtype=xp.float64)
    return _type1_coeffs_gpu(backend, tphx_gpu, cplx, dim, ms, eps, isign)


_omega_modes_gpu_cache: dict[tuple[int, int, float, int], tuple[Any, Any]] = {}


def _omega_modes_device_cached(backend: Any, m: int, dim: int, h: float) -> tuple[Any, Any]:
    xp = backend.xp
    key = (int(m), int(dim), float(h), int(xp.cuda.Device().id))
    cached = _omega_modes_gpu_cache.get(key)
    if cached is not None:
        return cached
    modes_rhs = generate_multi_index(int(m), dim)
    modes_v = generate_multi_index(2 * int(m), dim)
    om_rhs = _omega_modes(modes_rhs, h)
    om_v = _omega_modes(modes_v, h)
    pair = (
        xp.asarray(om_v, dtype=xp.float64),
        xp.asarray(om_rhs, dtype=xp.float64),
    )
    _omega_modes_gpu_cache[key] = pair
    return pair


def _to_host_complex128(arr: Any) -> np.ndarray:
    """CuPy / NumPy 阵列 -> host ``complex128``。"""
    if arr is None:
        raise TypeError("expected array")
    return np.asarray(_device_to_numpy(arr), dtype=np.complex128)


def compute_binned_fourier_sums_gpu(
    bin_stats: dict[str, Any],
    h: float,
    m: int,
    *,
    x_center: np.ndarray,
    backend: Any,
    nufft_tol: float,
    isign: int = -1,
    tphx_allow_clip: bool = False,
    return_gpu: bool = True,
    timings_s: dict[str, float] | None = None,
) -> tuple[np.ndarray | Any, np.ndarray | Any, dict[str, Any]]:
    """
    Fourier sums for **GPU-compacted** bin stats (``dense=False``, ``on_gpu=True``).

    When ``return_gpu=True`` (默认), ``v_tilde`` / ``rhs_tilde`` 留在 GPU。
    ``return_gpu=False`` 时拷回 NumPy，并在 ``timings_s`` 写入 ``t_gpu_to_cpu_copy_s``。
    """
    if backend is None:
        raise ValueError("compute_binned_fourier_sums_gpu requires a GPU backend bundle.")
    if not bin_stats.get("on_gpu"):
        raise ValueError("compute_binned_fourier_sums_gpu expects bin_stats['on_gpu'] True.")
    if bin_stats.get("dense", True):
        raise ValueError("Pass sparsify_bin_stats_gpu output (dense=False).")

    xp = backend.xp
    _sync_cuda_for_timings(xp, timings_s)
    t_ft0 = time.perf_counter()
    order = _resolve_order(str(bin_stats["order"]), int(bin_stats["d"]))
    dim = int(bin_stats["d"])
    z_used = bin_stats["z_occ"]
    c_w = bin_stats["c_occ"]
    s_w = bin_stats["s_occ"]

    om_v_g, om_rhs_g = _omega_modes_device_cached(backend, int(m), dim, float(h))

    ms_rhs = 2 * int(m) + 1
    ms_v = 4 * int(m) + 1

    tphx = tphx_from_centers_gpu(z_used, h, x_center, xp, allow_clip=tphx_allow_clip)

    jc = 1.0j * float(isign)
    om_v_c = om_v_g.astype(xp.complex128, copy=False)
    om_rhs_c = om_rhs_g.astype(xp.complex128, copy=False)

    if order == "C0":
        info = {"num_transforms_v": 1, "num_transforms_rhs": 1}
        v_tilde = _dispatch_type1_keep_gpu(backend, tphx, c_w, dim, ms_v, nufft_tol, isign)
        rhs_tilde = _dispatch_type1_keep_gpu(backend, tphx, s_w, dim, ms_rhs, nufft_tol, isign)

    elif order == "C1":
        a = bin_stats["a_occ"]
        ay = bin_stats["ay_occ"]
        n_tr_v = 1 + dim
        n_tr_r = 1 + dim
        stacks_v = xp.stack([c_w] + [a[:, kk] for kk in range(dim)], axis=0)
        stacks_r = xp.stack([s_w] + [ay[:, kk] for kk in range(dim)], axis=0)
        fv = _dispatch_type1_keep_gpu_batched(backend, tphx, stacks_v, dim, ms_v, nufft_tol, isign)
        fr = _dispatch_type1_keep_gpu_batched(backend, tphx, stacks_r, dim, ms_rhs, nufft_tol, isign)
        v_tilde = fv[0].astype(xp.complex128, copy=False)
        for kk in range(dim):
            v_tilde = v_tilde + jc * om_v_c[:, kk] * fv[1 + kk]
        rhs_tilde = fr[0].astype(xp.complex128, copy=False)
        for kk in range(dim):
            rhs_tilde = rhs_tilde + jc * om_rhs_c[:, kk] * fr[1 + kk]
        info = {"num_transforms_v": n_tr_v, "num_transforms_rhs": n_tr_r}

    else:
        pairs: list[tuple[int, int]] = bin_stats["pairs"]  # type: ignore[assignment]
        a = bin_stats["a_occ"]
        ay = bin_stats["ay_occ"]
        Qm = bin_stats["Q_occ"]
        Qym = bin_stats["Qy_occ"]
        npair = len(pairs)
        n_tr_v = 1 + dim + npair
        n_tr_r = 1 + dim + npair
        qcols_v = [Qm[:, p] for p in range(npair)]
        qcols_r = [Qym[:, p] for p in range(npair)]
        stacks_v = xp.stack([c_w] + [a[:, kk] for kk in range(dim)] + qcols_v, axis=0)
        stacks_r = xp.stack([s_w] + [ay[:, kk] for kk in range(dim)] + qcols_r, axis=0)
        fv = _dispatch_type1_keep_gpu_batched(backend, tphx, stacks_v, dim, ms_v, nufft_tol, isign)
        fr = _dispatch_type1_keep_gpu_batched(backend, tphx, stacks_r, dim, ms_rhs, nufft_tol, isign)

        v_tilde = fv[0].astype(xp.complex128, copy=False)
        for kk in range(dim):
            v_tilde = v_tilde + jc * om_v_c[:, kk] * fv[1 + kk]
        rhs_tilde = fr[0].astype(xp.complex128, copy=False)
        for kk in range(dim):
            rhs_tilde = rhs_tilde + jc * om_rhs_c[:, kk] * fr[1 + kk]

        for p, (i, j) in enumerate(pairs):
            fv2 = fv[1 + dim + p]
            factor_v = om_v_g[:, i] * om_v_g[:, j]
            if i != j:
                factor_v = factor_v * 2.0
            v_tilde = v_tilde + (-0.5 * factor_v.astype(xp.complex128) * fv2)

            fr2 = fr[1 + dim + p]
            factor_r = om_rhs_g[:, i] * om_rhs_g[:, j]
            if i != j:
                factor_r = factor_r * 2.0
            rhs_tilde = rhs_tilde + (-0.5 * factor_r.astype(xp.complex128) * fr2)

        info = {"num_transforms_v": n_tr_v, "num_transforms_rhs": n_tr_r}

    _sync_cuda_for_timings(xp, timings_s)
    if timings_s is not None:
        timings_s["t_binned_cufinufft_on_centers_s"] = float(time.perf_counter() - t_ft0)

    if return_gpu:
        return v_tilde, rhs_tilde, info

    _sync_cuda_stream(xp)
    t_cp0 = time.perf_counter()
    out_v = _to_host_complex128(v_tilde)
    out_r = _to_host_complex128(rhs_tilde)
    if timings_s is not None:
        timings_s["t_gpu_to_cpu_copy_s"] = float(time.perf_counter() - t_cp0)
    return out_v, out_r, info


def _device_to_numpy(x: Any) -> np.ndarray:
    """CuPy 等到 host；已是 NumPy 则视作数组。"""
    if hasattr(x, "get"):
        return np.asarray(x.get())
    return np.asarray(x)


def _type1_coeffs_cpu(
    tphx: np.ndarray,
    coeffs: np.ndarray,
    dim: int,
    ms: int,
    eps: float,
    isign: int,
) -> np.ndarray:
    tphx = np.ascontiguousarray(tphx, dtype=np.float64)
    coeffs = np.ascontiguousarray(coeffs, dtype=np.complex128)
    out = nufftnd1(tphx, coeffs, ms, eps, isign)
    return np.asarray(out, dtype=np.complex128).reshape(-1)


def _type1_coeffs_gpu(
    backend: Any,
    tphx: Any,
    coeffs: Any,
    dim: int,
    ms: int,
    eps: float,
    isign: int,
) -> Any:
    xp = backend.xp
    tphx_gpu = xp.asarray(tphx, dtype=xp.float64)
    n = int(tphx_gpu.shape[0])
    c_full = xp.asarray(coeffs, dtype=xp.complex128)
    batched = bool(c_full.ndim == 2)
    if c_full.ndim == 1:
        if int(c_full.shape[0]) != n:
            raise ValueError(f"type-1 coeffs length {c_full.shape[0]} != tphx rows {n}")
        c_call = c_full
        n_rows = 1
    elif c_full.ndim == 2:
        if int(c_full.shape[1]) != n:
            raise ValueError(f"type-1 coeffs last dim {c_full.shape[1]} != tphx rows {n}")
        c_call = c_full
        n_rows = int(c_full.shape[0])
    else:
        raise ValueError("coeffs must be 1-D or 2-D complex on device")

    if backend.has_nufft and backend.nufft is not None:
        cuf = backend.nufft
        try:
            if dim == 1:
                x0 = xp.ascontiguousarray(tphx_gpu[:, 0])
                out = cuf.nufft1d1(x0, c_call, (int(ms),), eps=eps, isign=isign)
            elif dim == 2:
                x0 = xp.ascontiguousarray(tphx_gpu[:, 0])
                x1 = xp.ascontiguousarray(tphx_gpu[:, 1])
                out = cuf.nufft2d1(x0, x1, c_call, (int(ms), int(ms)), eps=eps, isign=isign)
            elif dim == 3:
                x0 = xp.ascontiguousarray(tphx_gpu[:, 0])
                x1 = xp.ascontiguousarray(tphx_gpu[:, 1])
                x2 = xp.ascontiguousarray(tphx_gpu[:, 2])
                out = cuf.nufft3d1(
                    x0,
                    x1,
                    x2,
                    c_call,
                    (int(ms), int(ms), int(ms)),
                    eps=eps,
                    isign=isign,
                )
            else:
                raise NotImplementedError("cuFINUFFT 路径仅支持 dim<=3")
            out_arr = xp.ascontiguousarray(out)
            if batched:
                return out_arr.reshape(n_rows, -1)
            return out_arr.reshape(-1)
        except Exception as exc:
            if getattr(backend, "allow_cpu_fallback", False):
                warnings.warn(
                    f"cuFINUFFT failed, falling back to CPU finufft: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise

    tphx_np = np.ascontiguousarray(_device_to_numpy(tphx_gpu), dtype=np.float64)
    if not batched:
        c_np = np.ascontiguousarray(_device_to_numpy(c_full), dtype=np.complex128)
        host_out = _type1_coeffs_cpu(tphx_np, c_np, dim, ms, eps, isign)
        return xp.asarray(host_out, dtype=xp.complex128)

    mats: list[np.ndarray] = []
    for t in range(n_rows):
        c_np = np.ascontiguousarray(_device_to_numpy(c_full[int(t)]), dtype=np.complex128)
        mats.append(_type1_coeffs_cpu(tphx_np, c_np, dim, ms, eps, isign))
    stacked = np.stack(mats, axis=0)
    return xp.asarray(stacked, dtype=xp.complex128)


def _dispatch_type1(
    backend: Any | None,
    tphx_np: np.ndarray,
    weights_real: np.ndarray,
    dim: int,
    ms: int,
    eps: float,
    isign: int,
) -> np.ndarray:
    coeffs_np = np.asarray(weights_real, dtype=np.complex128)
    if backend is None:
        return _type1_coeffs_cpu(tphx_np, coeffs_np, dim, ms, eps, isign)

    xp = backend.xp
    tphx_gpu = xp.asarray(np.ascontiguousarray(tphx_np, dtype=np.float64))
    coeffs_gpu = xp.asarray(coeffs_np)
    out_gpu = _type1_coeffs_gpu(backend, tphx_gpu, coeffs_gpu, dim, ms, eps, isign)
    return np.asarray(_device_to_numpy(out_gpu), dtype=np.complex128)


def _omega_modes(modes: np.ndarray, h: float) -> np.ndarray:
    return 2.0 * np.pi * float(h) * np.asarray(modes, dtype=np.float64)


def compute_binned_fourier_sums_C0(
    bin_stats: dict[str, Any],
    h: float,
    m: int,
    *,
    x_center: np.ndarray,
    backend: Any | None,
    nufft_tol: float,
    isign: int = -1,
    tphx_allow_clip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dim = int(bin_stats["d"])
    modes_rhs = generate_multi_index(int(m), dim)
    modes_v = generate_multi_index(2 * int(m), dim)
    ms_rhs = 2 * int(m) + 1
    ms_v = 4 * int(m) + 1

    if bin_stats.get("dense", True):
        z = _dense_bin_centers(int(bin_stats["r"]), dim)
        mask = np.asarray(bin_stats["c"]) > 0
        z_used = z[mask]
        c_w = np.asarray(bin_stats["c"])[mask]
        s_w = np.asarray(bin_stats["s"])[mask]
    else:
        z_used = np.asarray(bin_stats["z_occ"], dtype=np.float64)
        c_w = np.asarray(bin_stats["c_occ"])
        s_w = np.asarray(bin_stats["s_occ"])

    tphx = tphx_from_centers(z_used, h, x_center, allow_clip=tphx_allow_clip)
    info = {"num_transforms_v": 1, "num_transforms_rhs": 1}

    v_tilde = _dispatch_type1(backend, tphx, c_w, dim, ms_v, nufft_tol, isign)
    rhs_tilde = _dispatch_type1(backend, tphx, s_w, dim, ms_rhs, nufft_tol, isign)
    return v_tilde, rhs_tilde, info


def _dense_bin_centers(r: int, d: int) -> np.ndarray:
    """与 ``linearize_indices`` 一致：``flat = q0 + q1 r + ... + q_{d-1} r^{d-1}``。"""
    rr = int(r)
    G = rr**d
    flat = np.arange(G, dtype=np.int64)
    q = np.zeros((G, d), dtype=np.float64)
    rem = flat.copy()
    for k in range(d):
        q[:, k] = (rem % rr).astype(np.float64)
        rem //= rr
    return (q + 0.5) / float(rr)


def compute_binned_fourier_sums_C1(
    bin_stats: dict[str, Any],
    h: float,
    m: int,
    *,
    x_center: np.ndarray,
    backend: Any | None,
    nufft_tol: float,
    isign: int = -1,
    tphx_allow_clip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dim = int(bin_stats["d"])
    modes_rhs = generate_multi_index(int(m), dim)
    modes_v = generate_multi_index(2 * int(m), dim)
    om_rhs = _omega_modes(modes_rhs, h)
    om_v = _omega_modes(modes_v, h)
    ms_rhs = 2 * int(m) + 1
    ms_v = 4 * int(m) + 1

    if bin_stats.get("dense", True):
        z = _dense_bin_centers(int(bin_stats["r"]), dim)
        mask = np.asarray(bin_stats["c"]) > 0
        z_used = z[mask]
        c_w = np.asarray(bin_stats["c"])[mask]
        s_w = np.asarray(bin_stats["s"])[mask]
        a = np.asarray(bin_stats["a"])[mask, :]
        ay = np.asarray(bin_stats["ay"])[mask, :]
    else:
        z_used = np.asarray(bin_stats["z_occ"], dtype=np.float64)
        c_w = np.asarray(bin_stats["c_occ"])
        s_w = np.asarray(bin_stats["s_occ"])
        a = np.asarray(bin_stats["a_occ"])
        ay = np.asarray(bin_stats["ay_occ"])

    tphx = tphx_from_centers(z_used, h, x_center, allow_clip=tphx_allow_clip)
    n_tr_v = 1 + dim
    n_tr_r = 1 + dim

    T0v = _dispatch_type1(backend, tphx, c_w, dim, ms_v, nufft_tol, isign)
    v_tilde = np.array(T0v, dtype=np.complex128, copy=True)
    for k in range(dim):
        Tk = _dispatch_type1(backend, tphx, a[:, k], dim, ms_v, nufft_tol, isign)
        v_tilde += 1.0j * float(isign) * om_v[:, k] * Tk

    T0r = _dispatch_type1(backend, tphx, s_w, dim, ms_rhs, nufft_tol, isign)
    rhs_tilde = np.array(T0r, dtype=np.complex128, copy=True)
    for k in range(dim):
        Tk = _dispatch_type1(backend, tphx, ay[:, k], dim, ms_rhs, nufft_tol, isign)
        rhs_tilde += 1.0j * float(isign) * om_rhs[:, k] * Tk

    return v_tilde, rhs_tilde, {"num_transforms_v": n_tr_v, "num_transforms_rhs": n_tr_r}


def compute_binned_fourier_sums_C2(
    bin_stats: dict[str, Any],
    h: float,
    m: int,
    *,
    x_center: np.ndarray,
    backend: Any | None,
    nufft_tol: float,
    isign: int = -1,
    tphx_allow_clip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dim = int(bin_stats["d"])
    modes_rhs = generate_multi_index(int(m), dim)
    modes_v = generate_multi_index(2 * int(m), dim)
    om_rhs = _omega_modes(modes_rhs, h)
    om_v = _omega_modes(modes_v, h)
    pairs: list[tuple[int, int]] = bin_stats["pairs"]  # type: ignore[assignment]
    ms_rhs = 2 * int(m) + 1
    ms_v = 4 * int(m) + 1

    npair = len(pairs)
    n_tr_v = 1 + dim + npair
    n_tr_r = 1 + dim + npair

    if bin_stats.get("dense", True):
        z = _dense_bin_centers(int(bin_stats["r"]), dim)
        mask = np.asarray(bin_stats["c"]) > 0
        z_used = z[mask]
        c_w = np.asarray(bin_stats["c"])[mask]
        s_w = np.asarray(bin_stats["s"])[mask]
        a = np.asarray(bin_stats["a"])[mask, :]
        ay = np.asarray(bin_stats["ay"])[mask, :]
        Qm = np.asarray(bin_stats["Q"])[mask, :]
        Qym = np.asarray(bin_stats["Qy"])[mask, :]
    else:
        z_used = np.asarray(bin_stats["z_occ"], dtype=np.float64)
        c_w = np.asarray(bin_stats["c_occ"])
        s_w = np.asarray(bin_stats["s_occ"])
        a = np.asarray(bin_stats["a_occ"])
        ay = np.asarray(bin_stats["ay_occ"])
        Qm = np.asarray(bin_stats["Q_occ"])
        Qym = np.asarray(bin_stats["Qy_occ"])

    tphx = tphx_from_centers(z_used, h, x_center, allow_clip=tphx_allow_clip)

    T0v = _dispatch_type1(backend, tphx, c_w, dim, ms_v, nufft_tol, isign)
    v_tilde = np.array(T0v, dtype=np.complex128, copy=True)
    for k in range(dim):
        Tk = _dispatch_type1(backend, tphx, a[:, k], dim, ms_v, nufft_tol, isign)
        v_tilde += 1.0j * float(isign) * om_v[:, k] * Tk

    T0r = _dispatch_type1(backend, tphx, s_w, dim, ms_rhs, nufft_tol, isign)
    rhs_tilde = np.array(T0r, dtype=np.complex128, copy=True)
    for k in range(dim):
        Tk = _dispatch_type1(backend, tphx, ay[:, k], dim, ms_rhs, nufft_tol, isign)
        rhs_tilde += 1.0j * float(isign) * om_rhs[:, k] * Tk

    for p, (i, j) in enumerate(pairs):
        Tqv = _dispatch_type1(backend, tphx, Qm[:, p], dim, ms_v, nufft_tol, isign)
        factor_v = om_v[:, i] * om_v[:, j]
        if i != j:
            factor_v = factor_v * 2.0
        v_tilde += -0.5 * factor_v * Tqv

        Tqr = _dispatch_type1(backend, tphx, Qym[:, p], dim, ms_rhs, nufft_tol, isign)
        factor_r = om_rhs[:, i] * om_rhs[:, j]
        if i != j:
            factor_r = factor_r * 2.0
        rhs_tilde += -0.5 * factor_r * Tqr

    return v_tilde, rhs_tilde, {"num_transforms_v": n_tr_v, "num_transforms_rhs": n_tr_r}


def compute_binned_fourier_sums(
    bin_stats: dict[str, Any],
    h: float,
    m: int,
    *,
    x_center: np.ndarray,
    backend: Any | None = None,
    nufft_tol: float = 1e-9,
    tphx_allow_clip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    order = _resolve_order(str(bin_stats["order"]), int(bin_stats["d"]))
    if order == "C0":
        return compute_binned_fourier_sums_C0(
            bin_stats,
            h,
            m,
            x_center=x_center,
            backend=backend,
            nufft_tol=nufft_tol,
            tphx_allow_clip=tphx_allow_clip,
        )
    if order == "C1":
        return compute_binned_fourier_sums_C1(
            bin_stats,
            h,
            m,
            x_center=x_center,
            backend=backend,
            nufft_tol=nufft_tol,
            tphx_allow_clip=tphx_allow_clip,
        )
    return compute_binned_fourier_sums_C2(
        bin_stats,
        h,
        m,
        x_center=x_center,
        backend=backend,
        nufft_tol=nufft_tol,
        tphx_allow_clip=tphx_allow_clip,
    )


def _count_occ_from_dense_c(c: Any) -> int:
    """dense 计数向量（NumPy / CuPy）上非空的 bin 数。"""
    try:
        import cupy as cp  # type: ignore

        if isinstance(c, cp.ndarray):
            return int(cp.count_nonzero(c > 0))
    except Exception:
        pass
    return int(np.sum(np.asarray(c) > 0))


def _count_occ_from_compact_c_occ(c_occ: Any) -> int:
    """已 compact 时 ``c_occ`` 的行数即占用 bin 数。"""
    try:
        import cupy as cp  # type: ignore

        if isinstance(c_occ, cp.ndarray):
            return int(c_occ.shape[0])
    except Exception:
        pass
    return int(np.asarray(c_occ).shape[0])


def summarize_bin_stats_layout(stats: dict[str, Any]) -> dict[str, Any]:
    """默认 diagnostics 中代替完整 ``bin_stats`` 的轻量摘要。"""
    r = int(stats["r"])
    d = int(stats["d"])
    Gtot = int(r**d)
    dense = bool(stats.get("dense", True))
    if dense:
        nocc = _count_occ_from_dense_c(stats["c"])
    else:
        nocc = _count_occ_from_compact_c_occ(stats["c_occ"])
    bda = stats.get("binning_dense_allocated")
    return {
        "num_occupied_bins": nocc,
        "G_total_bins": Gtot,
        "bin_stats_layout_dense": dense,
        "binning_dense_allocated": bda,
    }


def _collect_warnings(
    N: int,
    d: int,
    G: int,
    theta_actual: float,
    theta_target: float,
    phase_target_met: bool,
) -> list[str]:
    w: list[str] = []
    if not phase_target_met or theta_actual > theta_target + 1e-15:
        w.append("Binning phase error target not met.")
    if G > 0 and N / float(G) < 2.0:
        w.append("Average occupancy too small; binning may not compress effectively.")
    if d >= 4:
        w.append("Regular grid binning may suffer from curse of dimensionality.")
    if G > 0 and float(G) > 0.2 * float(N):
        w.append("G is close to N; binned EFGP may not be faster than original NUFFT.")
    return w


def _n_tr_total_for_bin_order(bin_order: str, d_: int) -> int:
    o = _resolve_order(bin_order, d_)
    ntv, ntr = (1, 1)
    if o == "C0":
        ntv, ntr = 1, 1
    elif o == "C1":
        dd = int(d_)
        ntv = ntr = 1 + dd
    else:
        dd = int(d_)
        npair = len(_pair_indices_upper(dd))
        ntv = ntr = 1 + dd + npair
    return int(ntv + ntr)


def _gpu_exact_dual_type1_dense_points(
    backend: Any,
    X_pts: Any,
    y_pts: Any,
    *,
    dim: int,
    h: float,
    m: int,
    x_center: np.ndarray,
    nufft_tol: float,
    isign: int,
    tphx_allow_clip: bool,
    timings_s: dict[str, float] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    xp = backend.xp
    X_pts = xp.asarray(X_pts, dtype=xp.float64)
    y_pts = xp.asarray(y_pts, dtype=xp.float64).reshape(-1)
    ms_rhs = 2 * int(m) + 1
    ms_v = 4 * int(m) + 1
    _sync_cuda_for_timings(xp, timings_s)
    t0 = time.perf_counter()
    tphx = tphx_from_centers_gpu(X_pts, float(h), x_center, xp, allow_clip=tphx_allow_clip)
    nloc = int(X_pts.shape[0])
    ones = xp.ones(nloc, dtype=xp.float64)
    v_tilde = _dispatch_type1_keep_gpu(backend, tphx, ones, dim, ms_v, nufft_tol, isign)
    rhs_tilde = _dispatch_type1_keep_gpu(backend, tphx, y_pts, dim, ms_rhs, nufft_tol, isign)
    _sync_cuda_for_timings(xp, timings_s)
    if timings_s is not None:
        timings_s["t_exact_dense_point_dual_nufft_s"] = float(time.perf_counter() - t0)
    info = {"num_transforms_v": 1, "num_transforms_rhs": 1}
    return v_tilde, rhs_tilde, info


def build_binned_efgp_system(
    X: np.ndarray | Any,
    y: np.ndarray | Any,
    N: int,
    d: int,
    h: float,
    m: int,
    D: np.ndarray,
    order: str = "C1",
    quality: str = "balanced",
    r: int | None = None,
    memory_budget_bytes: int = 4 * 1024**3,
    min_avg_count: float = 8.0,
    use_sparse_bins: bool = False,
    r_max: int | None = None,
    dtype: Any = np.float64,
    backend: Any | None = None,
    nufft_tol: float = 1e-9,
    x_center: np.ndarray | None = None,
    *,
    return_bin_stats: bool = False,
    tphx_allow_clip: bool = False,
    use_gpu_dense_bins: bool = True,
    return_gpu: bool = True,
    gpu_timing: bool = False,
    input_on_gpu: bool = False,
    assume_normalized: bool = False,
    skip_cpu_validation: bool = False,
    gpu_chunk_rows: int | None = None,
    auto_downgrade_order: bool = True,
    allow_exact_nufft_fallback: bool = False,
    nufft_allow_cpu_fallback: bool | None = None,
) -> tuple[np.ndarray | Any, np.ndarray | Any, dict[str, Any]]:
    """
    由分箱经验测度构造近似 EFGP 预计算量。

    GPU dense 默认：**融合 CUDA binning（单次 N-pass） + 批量 strengths 的 cuFINUFFT(type-1)**。
    ``gpu_timing=True`` 时记录 ``binned_precompute_breakdown_s`` / ``gpu_precompute_timings_s`` 并在关键点同步；
    否则多数内部步骤 **不额外同步**，便于与其它 CUDA work 重叠。

    ``auto_downgrade_order``（默认启用）：**只 fused binning 一次**（按请求的 ``order``），之后 C2→C1→C0 仅
    在 **同一套 dense GPU 缓冲** 上做子集视图 + 重新 compact，不重跑 N-pass。
    compact 后以 ``ratio = ((#v)+(#rhs))*G_eff/N`` 判别；``ratio>1`` 时若 ``allow_exact_nufft_fallback=True`` 则切到
    全点双 type-1。**纯 binned 评测请保持** ``allow_exact_nufft_fallback=False``（默认），否则结果与 speedup 会与 exact 混淆。

    ``input_on_gpu=True``：``X,y`` 为 CuPy 数组，已满足 ``[0,1]^d`` 时可设 ``assume_normalized=True`` 只做轻检查。
    ``gpu_chunk_rows``：宿主模式为 H2D 块大小；``input_on_gpu`` 时为设备上行块大小（显存子区间 launch）。

    ``nufft_allow_cpu_fallback``：若传入则临时覆盖 ``backend.allow_cpu_fallback``（结束后恢复），benchmark 建议 ``False``。
    """
    order_req = _resolve_order(order, d)
    order_work = str(order_req)

    saved_cpu_fb: bool | None = None
    fb_mutated = False
    if backend is not None and nufft_allow_cpu_fallback is not None:
        saved_cpu_fb = bool(getattr(backend, "allow_cpu_fallback", False))
        backend.allow_cpu_fallback = bool(nufft_allow_cpu_fallback)
        fb_mutated = True

    try:
        if quality not in THETA_MAX:
            raise ValueError(f"quality must be one of {list(THETA_MAX.keys())}; got {quality!r}")

        y_work_np: np.ndarray | None = None
        gpu_X = gpu_y = None
        if input_on_gpu:
            if not use_gpu_dense_bins:
                raise ValueError("input_on_gpu=True 需要 use_gpu_dense_bins=True")
            if backend is None:
                raise ValueError("input_on_gpu 需要 backend")
            xp_in = backend.xp
            gpu_X = xp_in.asarray(X, dtype=xp_in.float64)
            gpu_y = xp_in.asarray(y, dtype=xp_in.float64).reshape(-1)
            if int(gpu_X.shape[0]) != int(N) or int(gpu_X.shape[1]) != int(d):
                raise ValueError(f"X must have shape (N, {d}) on GPU")
            if int(gpu_y.shape[0]) != int(N):
                raise ValueError("N must match len(y) on GPU")
            if not skip_cpu_validation and not assume_normalized:
                lo = float(xp_in.min(gpu_X))
                hi = float(xp_in.max(gpu_X))
                if lo < -X_NORM_TOL or hi > 1.0 + X_NORM_TOL:
                    raise ValueError(
                        "input_on_gpu: X 超出 [0,1] 容许范围；可设 assume_normalized=True 跳过此检查"
                    )
        else:
            if np.asarray(X).ndim != 2 or np.asarray(X).shape[1] != int(d):
                raise ValueError(f"X must have shape (N, {d}).")
            y_work_np = np.asarray(y, dtype=dtype).reshape(-1)
            if np.asarray(X).shape[0] != int(N) or y_work_np.shape[0] != int(N):
                raise ValueError("N must match len(X) and len(y).")
            if not skip_cpu_validation and not assume_normalized:
                validate_normalized_X(np.asarray(X), int(d))

        theta_target = THETA_MAX[quality][order_req]

        if r is None:
            r_use, G, Delta, theta_actual, grid_diag = choose_binning_grid(
                int(N),
                int(d),
                float(h),
                int(m),
                order=order_req,
                quality=quality,
                memory_budget_bytes=memory_budget_bytes,
                min_avg_count=min_avg_count,
                r_max=r_max,
            )
        else:
            G, Delta, theta_actual, grid_diag = _grid_diag_for_fixed_r(
                int(r),
                int(N),
                int(d),
                float(h),
                int(m),
                order_req,
                quality,
                float(theta_target),
            )
            r_use = int(grid_diag["r_auto"])

        if x_center is None:
            x_center_np = np.full(int(d), 0.5, dtype=np.float64)
        else:
            x_center_np = np.asarray(x_center, dtype=np.float64).reshape(-1)
            if x_center_np.size != int(d):
                raise ValueError("x_center must have length d.")

        timing_payload: dict[str, float] | None = {} if gpu_timing else None
        used_exact_nufft = False
        downgrade_notes: list[str] = []

        if use_gpu_dense_bins:
            if backend is None:
                raise ValueError("use_gpu_dense_bins=True requires a GPU backend bundle.")
            xp = backend.xp
            stats: dict[str, Any] = {}
            ft_info: dict[str, Any] = {"num_transforms_v": 0, "num_transforms_rhs": 0}
            v_tilde: Any
            rhs_tilde: Any
            stats_dense_full: dict[str, Any]

            order_work = str(order_req)
            if input_on_gpu:
                stats_dense_full = build_bin_stats_from_arrays_gpu_dense(
                    None,
                    None,
                    r_use,
                    int(d),
                    str(order_req),
                    xp=xp,
                    timings_s=timing_payload,
                    gpu_chunk_rows=gpu_chunk_rows,
                    skip_normalized_check=True,
                    X_gpu=gpu_X,
                    y_gpu=gpu_y,
                )
            else:
                assert y_work_np is not None
                stats_dense_full = build_bin_stats_from_arrays_gpu_dense(
                    np.asarray(X, dtype=np.float64),
                    y_work_np,
                    r_use,
                    int(d),
                    str(order_req),
                    xp=xp,
                    timings_s=timing_payload,
                    gpu_chunk_rows=gpu_chunk_rows,
                )

            loop_guard = 0
            while True:
                loop_guard += 1
                if loop_guard > 8:
                    raise RuntimeError("auto_downgrade_order: exceeded max refinement loops")

                dense_tail = _gpu_dense_stats_tail_view(stats_dense_full, order_work, int(d))
                stats = sparsify_bin_stats_gpu(dense_tail, xp=xp, timings_s=timing_payload)

                nocc = int(stats["c_occ"].shape[0])
                n_tot_tr = _n_tr_total_for_bin_order(str(order_work), int(d))
                ratio = float(n_tot_tr) * float(nocc) / float(max(int(N), 1))

                if ratio > 1.0:
                    if not allow_exact_nufft_fallback:
                        raise ValueError(
                            "effective_work_ratio>1.0：binned NUFFT 在 transform 层难以优于双 NUFFT；"
                            "请提高 r / 自动降阶，或显式设 allow_exact_nufft_fallback=True 走全点 double type-1。"
                        )
                    used_exact_nufft = True
                    downgrade_notes.append(
                        f"exact dual type-1 on N points (ratio={ratio:.4g}>1)"
                    )
                    if input_on_gpu:
                        v_tilde, rhs_tilde, ft_info = _gpu_exact_dual_type1_dense_points(
                            backend,
                            gpu_X,
                            gpu_y,
                            dim=int(d),
                            h=float(h),
                            m=int(m),
                            x_center=x_center_np,
                            nufft_tol=nufft_tol,
                            isign=-1,
                            tphx_allow_clip=tphx_allow_clip,
                            timings_s=timing_payload,
                        )
                    else:
                        v_tilde, rhs_tilde, ft_info = _gpu_exact_dual_type1_dense_points(
                            backend,
                            xp.asarray(np.asarray(X, dtype=np.float64)),
                            xp.asarray(y_work_np.astype(np.float64)),
                            dim=int(d),
                            h=float(h),
                            m=int(m),
                            x_center=x_center_np,
                            nufft_tol=nufft_tol,
                            isign=-1,
                            tphx_allow_clip=tphx_allow_clip,
                            timings_s=timing_payload,
                        )
                    break

                if auto_downgrade_order and str(order_work) == "C2" and ratio > 0.4:
                    order_work = "C1"
                    downgrade_notes.append(
                        f"auto downgrade C2->C1 (ratio={ratio:.4g}); reuse dense stats view"
                    )
                    continue
                if auto_downgrade_order and str(order_work) == "C1" and ratio > 0.8:
                    order_work = "C0"
                    downgrade_notes.append(
                        f"auto downgrade C1->C0 (ratio={ratio:.4g}); reuse dense stats view"
                    )
                    continue

                v_tilde, rhs_tilde, ft_info = compute_binned_fourier_sums_gpu(
                    stats,
                    float(h),
                    int(m),
                    x_center=x_center_np,
                    backend=backend,
                    nufft_tol=nufft_tol,
                    tphx_allow_clip=tphx_allow_clip,
                    return_gpu=bool(return_gpu),
                    timings_s=timing_payload,
                )
                break
        else:
            assert y_work_np is not None
            use_dense_bins = not bool(use_sparse_bins)
            stats = build_bin_stats_from_arrays(
                X, y_work_np, r_use, int(d), order_req, use_dense_bins=use_dense_bins
            )
            v_tilde, rhs_tilde, ft_info = compute_binned_fourier_sums(
                stats,
                float(h),
                int(m),
                x_center=x_center_np,
                backend=backend,
                nufft_tol=nufft_tol,
                tphx_allow_clip=tphx_allow_clip,
            )

        binned_breakdown_s = timing_payload if timing_payload is not None else {}

        D_flat = np.asarray(D, dtype=np.float64).reshape(-1)
        nf = int((2 * int(m) + 1) ** int(d))
        if D_flat.shape[0] != nf:
            raise ValueError(f"D must have length (2m+1)^d = {nf}, got {D_flat.shape[0]}.")

        if use_gpu_dense_bins:
            xp_dm = backend.xp
            _sync_cuda_for_timings(xp_dm, timing_payload)
            _t_dm0 = time.perf_counter()
            if return_gpu:
                Dg = xp_dm.asarray(D_flat, dtype=xp_dm.float64).reshape(-1)
                rhs_flat = rhs_tilde.reshape(-1)
                if Dg.shape[0] != rhs_flat.shape[0]:
                    raise ValueError(f"D rhs length mismatch: {Dg.shape[0]} vs {rhs_flat.shape[0]}")
                b_tilde = Dg * rhs_flat
            else:
                rhs_np = np.asarray(rhs_tilde, dtype=np.complex128).reshape(-1)
                if D_flat.shape[0] != rhs_np.shape[0]:
                    raise ValueError(f"D rhs length mismatch: {D_flat.shape[0]} vs {rhs_np.shape[0]}")
                b_tilde = D_flat.astype(np.complex128, copy=False) * rhs_np
            _sync_cuda_for_timings(xp_dm, timing_payload)
            if timing_payload is not None:
                binned_breakdown_s["t_rhs_D_multiply_s"] = float(time.perf_counter() - _t_dm0)
        else:
            rhs_np = np.asarray(rhs_tilde, dtype=np.complex128).reshape(-1)
            b_tilde = D_flat.astype(np.complex128, copy=False) * rhs_np
        Mf = (2 * int(m) + 1) ** int(d)
        phase_met = bool(grid_diag.get("phase_target_met", True))
        warnings = _collect_warnings(
            int(N), int(d), G, theta_actual, theta_target, phase_met
        )
        if "warning_phase" in grid_diag:
            warnings.append(str(grid_diag["warning_phase"]))

        if downgrade_notes:
            warnings.extend(["auto_route: " + s for s in downgrade_notes])

        if use_gpu_dense_bins:
            bin_note = (
                "GPU fused binning kernel + compaction + batched strengths cuFINUFFT on occupied centers; "
                "or exact dual type-1 on all training points when route selects it."
            )
        else:
            bin_note = (
                "sparse binning: never allocated dense length-G accumulator vectors."
                if not stats.get("binning_dense_allocated", True)
                else (
                    "dense binning: allocated full G=r^d vectors (see estimated_dense_memory_bytes in grid)."
                )
            )

        if used_exact_nufft:
            bin_layout = {
                "num_occupied_bins": int(N),
                "G_total_bins": int(G),
                "bin_stats_layout_dense": False,
                "binning_dense_allocated": False,
            }
        else:
            bin_layout = summarize_bin_stats_layout(stats)

        try:
            skew_sn = _bin_count_skew_from_bin_stats(stats)
        except Exception:
            skew_sn = {
                "bin_count_max_occ": 0.0,
                "bin_count_mean_occ": 0.0,
                "bin_count_p99_occ": 0.0,
                "atomic_skew_occ_counts": float("nan"),
            }

        diagnostics: dict[str, Any] = {
            "N": int(N),
            "d": int(d),
            "h": float(h),
            "m": int(m),
            "M_f": int(Mf),
            "order": order_req,
            "order_requested": str(order_req),
            "order_bins_final": str(stats.get("order", order_req)) if not used_exact_nufft else "EXACT_DUAL_NUFFT",
            "quality": quality,
            "r": int(r_use),
            "G": int(G),
            "Delta": float(Delta),
            "theta_actual": float(theta_actual),
            "avg_occupancy": float(N) / float(max(G, 1)),
            "compression_ratio": float(N) / float(max(G, 1)),
            "estimated_memory_gb": float(grid_diag["estimated_dense_memory_bytes"]) / (1024.0**3),
            "phase_target_met": phase_met,
            "num_transforms_v": ft_info["num_transforms_v"],
            "num_transforms_rhs": ft_info["num_transforms_rhs"],
            "grid": grid_diag,
            "warnings": warnings,
            "use_sparse_bins_request": bool(use_sparse_bins),
            "binning_dense_allocated": stats.get("binning_dense_allocated"),
            "binning_memory_note": bin_note,
            "use_gpu_dense_bins": bool(use_gpu_dense_bins),
            "input_on_gpu": bool(input_on_gpu),
            "assume_normalized": bool(assume_normalized),
            "skip_cpu_validation": bool(skip_cpu_validation),
            "auto_downgrade_order": bool(auto_downgrade_order),
            "allow_exact_nufft_fallback": bool(allow_exact_nufft_fallback),
            "used_exact_dense_point_nufft": bool(used_exact_nufft),
            "nufft_cpu_fallback_overridden": fb_mutated,
            "nufft_allow_cpu_fallback_effective": (
                bool(nufft_allow_cpu_fallback) if nufft_allow_cpu_fallback is not None else None
            ),
            **skew_sn,
            **bin_layout,
        }
        nocc_eff = int(diagnostics.get("num_occupied_bins", 0))
        diagnostics["occupied_fraction_over_G"] = float(nocc_eff) / float(max(int(G), 1))
        skew_m = diagnostics.get("atomic_skew_occ_counts", float("nan"))
        if isinstance(skew_m, float) and not math.isnan(skew_m) and skew_m > 100.0:
            warnings.append(
                "bin_count max/mean skew>100: fused binning may be limited by atomic contention on hot bins."
            )
        if diagnostics["occupied_fraction_over_G"] > 0.5:
            warnings.append(
                "G_eff/G>0.5: compaction + per-occupied NUFFT may be a weak win; "
                "consider exact-N NUFFT or a dense regular-grid FFT path."
            )
        diagnostics["warnings"] = warnings
        diagnostics["benchmark_report_field_names"] = [
            "order_requested",
            "order_bins_final",
            "used_exact_dense_point_nufft",
            "allow_exact_nufft_fallback",
            "N",
            "d",
            "m",
            "h",
            "r",
            "G",
            "num_occupied_bins",
            "occupied_fraction_over_G",
            "theta_actual",
            "phase_target_met",
            "effective_work_ratio",
            "avg_occupancy",
            "bin_count_max_occ",
            "bin_count_mean_occ",
            "bin_count_p99_occ",
            "atomic_skew_occ_counts",
            "t_gpu_fused_bin_moments_s",
            "t_compact_occupied_s",
            "t_binned_cufinufft_on_centers_s",
            "t_exact_dense_point_dual_nufft_s",
            "t_rhs_D_multiply_s",
        ]
        n_tr_total = int(ft_info["num_transforms_v"]) + int(ft_info["num_transforms_rhs"])
        diagnostics["effective_work_ratio"] = (
            float(n_tr_total) * float(nocc_eff) / float(max(int(N), 1))
        )
        diagnostics["return_gpu"] = bool(return_gpu and use_gpu_dense_bins)
        if use_gpu_dense_bins and binned_breakdown_s:
            diagnostics["binned_precompute_breakdown_s"] = dict(binned_breakdown_s)
            diagnostics["binned_rhs_note"] = (
                "Compaction + batched strengths NUFFT; D multiply "
                "(``gpu_timing`` 打开时有 cuda 分段同步；omega/modes GPU 缓存复用)."
            )
        if gpu_timing and use_gpu_dense_bins and binned_breakdown_s:
            diagnostics["gpu_precompute_timings_s"] = dict(binned_breakdown_s)
        if return_bin_stats:
            diagnostics["bin_stats"] = stats
        return v_tilde, b_tilde, diagnostics
    finally:
        if fb_mutated and backend is not None:
            setattr(backend, "allow_cpu_fallback", saved_cpu_fb)


def build_binned_efgp_system_streaming(
    X_loader: Iterator[np.ndarray],
    y_loader: Iterator[np.ndarray],
    N: int,
    d: int,
    h: float,
    m: int,
    D: np.ndarray,
    order: str = "C1",
    quality: str = "balanced",
    r: int | None = None,
    memory_budget_bytes: int = 4 * 1024**3,
    min_avg_count: float = 8.0,
    use_sparse_bins: bool = False,
    r_max: int | None = None,
    backend: Any | None = None,
    nufft_tol: float = 1e-9,
    x_center: np.ndarray | None = None,
    *,
    return_bin_stats: bool = False,
    tphx_allow_clip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """流式多块 CPU binning（正确性/debug）；timing benchmark 请用 GPU fused 路径或不计本路径。"""
    order_eff = _resolve_order(order, d)
    if quality not in THETA_MAX:
        raise ValueError(f"quality must be one of {list(THETA_MAX.keys())}; got {quality!r}")

    theta_target = THETA_MAX[quality][order_eff]

    if r is None:
        r_use, G, Delta, theta_actual, grid_diag = choose_binning_grid(
            int(N),
            int(d),
            float(h),
            int(m),
            order=order_eff,
            quality=quality,
            memory_budget_bytes=memory_budget_bytes,
            min_avg_count=min_avg_count,
            r_max=r_max,
        )
    else:
        G, Delta, theta_actual, grid_diag = _grid_diag_for_fixed_r(
            int(r),
            int(N),
            int(d),
            float(h),
            int(m),
            order_eff,
            quality,
            float(theta_target),
        )
        r_use = int(grid_diag["r_auto"])

    if x_center is None:
        x_center_np = np.full(int(d), 0.5, dtype=np.float64)
    else:
        x_center_np = np.asarray(x_center, dtype=np.float64).reshape(-1)
        if x_center_np.size != int(d):
            raise ValueError("x_center must have length d.")

    use_dense_bins = not bool(use_sparse_bins)
    stats = build_bin_stats_streaming(
        X_loader,
        y_loader,
        r_use,
        int(d),
        order_eff,
        use_dense_bins=use_dense_bins,
    )
    if int(stats["n_seen"]) != int(N):
        raise ValueError(
            f"streaming loader yielded {stats['n_seen']} samples, expected {N}"
        )

    v_tilde, rhs_tilde, ft_info = compute_binned_fourier_sums(
        stats,
        float(h),
        int(m),
        x_center=x_center_np,
        backend=backend,
        nufft_tol=nufft_tol,
        tphx_allow_clip=tphx_allow_clip,
    )

    D = np.asarray(D, dtype=np.float64).reshape(-1)
    if D.shape[0] != (2 * int(m) + 1) ** int(d):
        raise ValueError(
            f"D must have length (2m+1)^d = {(2 * int(m) + 1) ** int(d)}, got {D.shape[0]}."
        )
    b_tilde = D * rhs_tilde

    Mf = (2 * int(m) + 1) ** int(d)
    phase_met = bool(grid_diag.get("phase_target_met", True))
    warnings = _collect_warnings(
        int(N), int(d), G, theta_actual, theta_target, phase_met
    )
    if "warning_phase" in grid_diag:
        warnings.append(str(grid_diag["warning_phase"]))

    bin_note = (
        "sparse binning: never allocated dense length-G accumulator vectors."
        if not stats.get("binning_dense_allocated", True)
        else (
            "dense binning: allocated full G=r^d vectors (see estimated_dense_memory_bytes in grid)."
        )
    )

    diagnostics: dict[str, Any] = {
        "N": int(N),
        "d": int(d),
        "h": float(h),
        "m": int(m),
        "M_f": int(Mf),
        "order": order_eff,
        "quality": quality,
        "r": int(r_use),
        "G": int(G),
        "Delta": float(Delta),
        "theta_actual": float(theta_actual),
        "avg_occupancy": float(N) / float(max(G, 1)),
        "compression_ratio": float(N) / float(max(G, 1)),
        "estimated_memory_gb": float(grid_diag["estimated_dense_memory_bytes"]) / (1024.0**3),
        "phase_target_met": phase_met,
        "num_transforms_v": ft_info["num_transforms_v"],
        "num_transforms_rhs": ft_info["num_transforms_rhs"],
        "grid": grid_diag,
        "warnings": warnings,
        "use_sparse_bins_request": bool(use_sparse_bins),
        "binning_dense_allocated": stats.get("binning_dense_allocated"),
        "binning_memory_note": bin_note,
        **summarize_bin_stats_layout(stats),
    }
    nocc_eff = int(diagnostics.get("num_occupied_bins", 0))
    n_tr_total = int(ft_info["num_transforms_v"]) + int(ft_info["num_transforms_rhs"])
    diagnostics["effective_work_ratio"] = float(n_tr_total) * float(nocc_eff) / float(max(int(N), 1))
    diagnostics["return_gpu"] = False
    if return_bin_stats:
        diagnostics["bin_stats"] = stats
    return v_tilde, b_tilde, diagnostics


def make_chunk_iterators_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    chunk_size: int,
) -> tuple[Iterator[np.ndarray], Iterator[np.ndarray], int]:
    """
    返回 **可直接传入** ``build_bin_stats_streaming`` / ``build_binned_efgp_system_streaming``
    的一对迭代器 ``(x_iter, y_iter, n)``（单次遍历生成器，勿重复使用）。
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n = int(X.shape[0])
    cs = int(chunk_size)

    def x_iter() -> Iterator[np.ndarray]:
        for i in range(0, n, cs):
            yield X[i : i + cs]

    def y_iter() -> Iterator[np.ndarray]:
        for i in range(0, n, cs):
            yield y[i : i + cs]

    return x_iter(), y_iter(), n


def _run_self_tests() -> None:
    """模块内自检（NUFFT 顺序、流式一致性、C0/C1/C2 相对误差趋势；可选 GPU 上 original N 点双 NUFFT 计时）。"""
    np.random.seed(42)
    d, m, h, r = 2, 2, 0.12, 14
    xc = np.full(d, 0.5)
    n_pt = 120
    X = np.random.rand(n_pt, d)
    y = np.random.randn(n_pt)

    print("Test 1: C0 NUFFT vs direct sum (generate_multi_index order)")
    st0 = build_bin_stats_from_arrays(X, y, r, d, "C0", use_dense_bins=True)
    z = _dense_bin_centers(r, d)
    mask = st0["c"] > 0
    z_used = z[mask]
    c_w = st0["c"][mask]
    s_w = st0["s"][mask]
    modes_v = generate_multi_index(2 * m, d)
    modes_rhs = generate_multi_index(m, d)
    v_c0, rhs_c0, _ = compute_binned_fourier_sums_C0(
        st0, h, m, x_center=xc, backend=None, nufft_tol=1e-12
    )
    v_dir = direct_fourier_sum_type1(z_used, c_w, modes_v, h, xc, isign=-1)
    rhs_dir = direct_fourier_sum_type1(z_used, s_w, modes_rhs, h, xc, isign=-1)
    assert np.allclose(v_c0, v_dir, rtol=2e-7, atol=1e-10), float(
        np.max(np.abs(v_c0 - v_dir))
    )
    assert np.allclose(rhs_c0, rhs_dir, rtol=2e-7, atol=1e-10)
    print("  OK")

    print("Test 2: relative error C0 vs C1 vs C2 vs exact on ones weights (typical trend)")
    modes_v_ex = generate_multi_index(2 * m, d)
    v_exact = direct_fourier_sum_type1(
        X, np.ones(n_pt, dtype=np.float64), modes_v_ex, h, xc, isign=-1
    )
    errs = []
    for ord_name in ("C0", "C1", "C2"):
        st = build_bin_stats_from_arrays(X, y, r, d, ord_name, use_dense_bins=True)
        if ord_name == "C0":
            v_apx, _, _ = compute_binned_fourier_sums_C0(
                st, h, m, x_center=xc, backend=None, nufft_tol=1e-12
            )
        elif ord_name == "C1":
            v_apx, _, _ = compute_binned_fourier_sums_C1(
                st, h, m, x_center=xc, backend=None, nufft_tol=1e-12
            )
        else:
            v_apx, _, _ = compute_binned_fourier_sums_C2(
                st, h, m, x_center=xc, backend=None, nufft_tol=1e-12
            )
        num = float(np.linalg.norm(v_apx - v_exact))
        den = max(float(np.linalg.norm(v_exact)), 1e-30)
        errs.append(num / den)
    print(f"  rel_err C0,C1,C2 = {errs[0]:.4e}, {errs[1]:.4e}, {errs[2]:.4e}")
    if not (errs[2] <= errs[0] + 1e-9 and errs[1] <= errs[0] + 1e-9):
        print(
            "  NOTE: err ordering err_C2<=err_C1<=err_C0 not guaranteed on every draw "
            "(Taylor validity depends on bin geometry); soft check only."
        )
    else:
        print("  OK (err_C2 <= err_C1 <= err_C0)")

    print("Test 3: dense streaming equals single-pass array")
    dense_a = build_bin_stats_from_arrays(X, y, r, d, "C2", use_dense_bins=True)
    xi, yi, _n = make_chunk_iterators_from_arrays(X, y, chunk_size=17)
    dense_s = build_bin_stats_streaming(xi, yi, r, d, "C2", use_dense_bins=True)
    assert int(dense_s["n_seen"]) == int(_n)
    np.testing.assert_allclose(dense_a["c"], dense_s["c"])
    np.testing.assert_allclose(dense_a["s"], dense_s["s"])
    np.testing.assert_allclose(dense_a["a"], dense_s["a"])
    np.testing.assert_allclose(dense_a["ay"], dense_s["ay"])
    np.testing.assert_allclose(dense_a["Q"], dense_s["Q"])
    np.testing.assert_allclose(dense_a["Qy"], dense_s["Qy"])
    print("  OK")

    print("Test 4: sparse binning path equals sparsify(dense) (C1)")
    ds = build_bin_stats_from_arrays(X, y, r, d, "C1", use_dense_bins=True)
    sp_ref = sparsify_bin_stats(ds)
    sp_direct = build_bin_stats_from_arrays(X, y, r, d, "C1", use_dense_bins=False)
    assert not sp_direct["dense"]
    order_ref = np.argsort(sp_ref["idx_occ"])
    order_dr = np.argsort(sp_direct["idx_occ"])
    np.testing.assert_array_equal(
        sp_ref["idx_occ"][order_ref], sp_direct["idx_occ"][order_dr]
    )
    np.testing.assert_allclose(sp_ref["c_occ"][order_ref], sp_direct["c_occ"][order_dr])
    np.testing.assert_allclose(sp_ref["s_occ"][order_ref], sp_direct["s_occ"][order_dr])
    np.testing.assert_allclose(sp_ref["a_occ"][order_ref], sp_direct["a_occ"][order_dr])
    np.testing.assert_allclose(sp_ref["ay_occ"][order_ref], sp_direct["ay_occ"][order_dr])
    print("  OK")

    print("Test 5: GPU cuFINUFFT vs CPU finufft (flatten order, optional)")
    try:
        from efgp_eigenpro_py.gpu.backends import build_gpu_backend_bundle

        bundle = build_gpu_backend_bundle()
    except Exception as exc:
        print(f"  SKIP ({exc})")
    else:
        if not bundle.has_nufft:
            print("  SKIP (no GPU NUFFT)")
        else:
            st_gpu = build_bin_stats_from_arrays(X, y, r, d, "C1", use_dense_bins=True)
            v_cpu, rhs_cpu, _ = compute_binned_fourier_sums(
                st_gpu,
                h,
                m,
                x_center=xc,
                backend=None,
                nufft_tol=1e-9,
            )
            v_gpu, rhs_gpu, _ = compute_binned_fourier_sums(
                st_gpu,
                h,
                m,
                x_center=xc,
                backend=bundle,
                nufft_tol=1e-9,
            )
            v_gpu = np.asarray(_device_to_numpy(v_gpu))
            rhs_gpu = np.asarray(_device_to_numpy(rhs_gpu))
            np.testing.assert_allclose(v_gpu, v_cpu, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(rhs_gpu, rhs_cpu, rtol=1e-6, atol=1e-8)
            print("  OK")

            print("Test 5c: fused GPU binning vs CPU dense moments (C2)")
            xp5 = bundle.xp
            st_ref = build_bin_stats_from_arrays(X, y, r, d, "C2", use_dense_bins=True)
            st_bd = build_bin_stats_from_arrays_gpu_dense(
                X, y, r, d, "C2", xp=xp5, timings_s=None
            )
            _sync_cuda_stream(xp5)
            for ak in ("c", "s", "a", "ay", "Q", "Qy"):
                a_cpu = np.asarray(st_ref[ak])
                a_g = np.asarray(_device_to_numpy(st_bd[ak]))
                np.testing.assert_allclose(a_g, a_cpu, rtol=1e-5, atol=1e-5)

            print("  OK")

            ms_rhs_tm = 2 * int(m) + 1
            ms_v_tm = 4 * int(m) + 1
            eps_tm = float(1e-9)
            xp_tm = bundle.xp
            Xv = validate_normalized_X(np.asarray(X, dtype=np.float64), int(d))
            y64_tm = np.asarray(y, dtype=np.float64).reshape(-1)
            tphx_pts = tphx_from_centers(Xv, float(h), xc, allow_clip=False)
            tphxg = xp_tm.asarray(tphx_pts, dtype=xp_tm.float64)
            wo_tm = xp_tm.ones(int(n_pt), dtype=xp_tm.float64)
            wy_tm = xp_tm.asarray(y64_tm, dtype=xp_tm.float64)
            print("Test 5d: cuFINUFFT batched type-1 (3 strengths) vs looped single transforms")
            fb_sv = bool(getattr(bundle, "allow_cpu_fallback", False))
            bundle.allow_cpu_fallback = False
            try:
                xp_tm.random.seed(123)
                w_0 = xp_tm.random.standard_normal(int(n_pt), dtype=xp_tm.float64)
                w_1 = xp_tm.random.standard_normal(int(n_pt), dtype=xp_tm.float64)
                w_2 = xp_tm.random.standard_normal(int(n_pt), dtype=xp_tm.float64)
                w_stack = xp_tm.stack([w_0, w_1, w_2], axis=0)
                _sync_cuda_stream(xp_tm)
                out_batch = _dispatch_type1_keep_gpu_batched(
                    bundle, tphxg, w_stack, d, ms_v_tm, eps_tm, -1
                )
                o0 = _dispatch_type1_keep_gpu(bundle, tphxg, w_0, d, ms_v_tm, eps_tm, -1)
                o1 = _dispatch_type1_keep_gpu(bundle, tphxg, w_1, d, ms_v_tm, eps_tm, -1)
                o2 = _dispatch_type1_keep_gpu(bundle, tphxg, w_2, d, ms_v_tm, eps_tm, -1)
                _sync_cuda_stream(xp_tm)
                out_loop = xp_tm.stack([o0, o1, o2], axis=0)
                np.testing.assert_allclose(
                    np.asarray(_device_to_numpy(out_batch)),
                    np.asarray(_device_to_numpy(out_loop)),
                    rtol=5e-5,
                    atol=5e-6,
                )
            finally:
                bundle.allow_cpu_fallback = fb_sv
            print("  OK")

            print(
                "Test 5b: original — N-point type-1 cuNUFFT wall time "
                "(v + rhs, 2 transforms, CUDA sync)"
            )
            for _warm in range(2):
                _sync_cuda_stream(xp_tm)
                _ = _dispatch_type1_keep_gpu(
                    bundle, tphxg, wo_tm, d, ms_v_tm, eps_tm, -1
                )
                _ = _dispatch_type1_keep_gpu(
                    bundle, tphxg, wy_tm, d, ms_rhs_tm, eps_tm, -1
                )
                _sync_cuda_stream(xp_tm)
            _sync_cuda_stream(xp_tm)
            t_ori0 = time.perf_counter()
            _ = _dispatch_type1_keep_gpu(
                bundle, tphxg, wo_tm, d, ms_v_tm, eps_tm, -1
            )
            _ = _dispatch_type1_keep_gpu(
                bundle, tphxg, wy_tm, d, ms_rhs_tm, eps_tm, -1
            )
            _sync_cuda_stream(xp_tm)
            t_original_cunufft_s = float(time.perf_counter() - t_ori0)
            print(
                f"  t_original_dual_cunufft_s (N={n_pt}, ms_v={ms_v_tm}, ms_rhs={ms_rhs_tm}): "
                f"{t_original_cunufft_s:.6e} s wall"
            )

    print("All self-tests finished.")


if __name__ == "__main__":
    _run_self_tests()
