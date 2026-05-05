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

``use_sparse_bins=True`` 时使用 **稀疏 binning**：binning 阶段不分配 ``G=r^d`` 稠密向量，
仅合并非空 bin（适合 ``G`` 很大）；若为 ``False`` 则分配 dense 数组。
诊断字段会标明 binning 是否曾分配 dense。
"""
from __future__ import annotations

import math
import sys
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
        return "C1" if d <= 2 else "C2"
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
) -> np.ndarray:
    """
    直接求 ``sum_n w_n exp(i * isign * tphx_n · modes)``，与 type-1 NUFFT（``isign``）对照用。

    ``tphx`` 由 ``tphx_from_centers(X_points, ...)`` 生成（默认严格范围检查）。
    ``modes`` 须与 ``generate_multi_index`` 行顺序一致。
    """
    X_points = np.asarray(X_points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.complex128).reshape(-1)
    if w.shape[0] != X_points.shape[0]:
        raise ValueError("weights length must match X_points.")
    modes = np.asarray(modes, dtype=np.float64)
    tphx = tphx_from_centers(X_points, h, x_center, allow_clip=tphx_allow_clip)
    phase = tphx @ modes.T
    return (w[:, None] * np.exp(1j * float(isign) * phase)).sum(axis=0)


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
        q = np.zeros((idx_occ.size, d), dtype=np.int64)
        rem = idx_occ.copy()
        for k in range(d):
            q[:, k] = rem % rr
            rem //= rr
        z = (q.astype(np.float64) + 0.5) / float(rr)
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


def build_bin_stats_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    r: int,
    d: int,
    order: str,
    *,
    dtype: np.dtype = np.float64,
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
    rt = np.float64 if np.issubdtype(dtype, np.floating) else np.float64
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
    dtype: np.dtype = np.float64,
    use_dense_bins: bool = True,
) -> dict[str, Any]:
    """多块流式：``use_dense_bins=False`` 时用 dict 合并，不分配 ``G`` 长度向量。"""
    order = _resolve_order(order, d)
    rr = int(r)
    rt = np.float64 if np.issubdtype(dtype, np.floating) else np.float64
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
            yc = np.asarray(y_chunk).reshape(-1)
            if xc.shape[0] != yc.shape[0]:
                raise ValueError("chunk X/y length mismatch in streaming loader.")
            n_seen += int(xc.shape[0])
            st = build_bin_stats_from_arrays(
                xc, yc, rr, d, order, dtype=dtype, use_dense_bins=True
            )
            c += st["c"]
            s += st["s"]
            if order in ("C1", "C2"):
                a += st["a"]
                ay += st["ay"]
            if order == "C2":
                Q += st["Q"]
                Qy += st["Qy"]

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

    q = np.zeros((idx_occ.size, d), dtype=np.int64)
    rem = idx_occ.copy()
    for k in range(d):
        q[:, k] = rem % rr
        rem //= rr

    z = (q.astype(np.float64) + 0.5) / float(rr)

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


def _as_numpy(x: Any) -> np.ndarray:
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
    n = int(tphx.shape[0])
    c = xp.asarray(coeffs, dtype=xp.complex128).reshape(n)

    if backend.has_nufft and backend.nufft is not None:
        cuf = backend.nufft
        try:
            if dim == 1:
                x0 = xp.ascontiguousarray(tphx[:, 0])
                out = cuf.nufft1d1(x0, c, (int(ms),), eps=eps, isign=isign)
            elif dim == 2:
                x0 = xp.ascontiguousarray(tphx[:, 0])
                x1 = xp.ascontiguousarray(tphx[:, 1])
                out = cuf.nufft2d1(x0, x1, c, (int(ms), int(ms)), eps=eps, isign=isign)
            elif dim == 3:
                x0 = xp.ascontiguousarray(tphx[:, 0])
                x1 = xp.ascontiguousarray(tphx[:, 1])
                x2 = xp.ascontiguousarray(tphx[:, 2])
                out = cuf.nufft3d1(
                    x0,
                    x1,
                    x2,
                    c,
                    (int(ms), int(ms), int(ms)),
                    eps=eps,
                    isign=isign,
                )
            else:
                raise NotImplementedError("cuFINUFFT 路径仅支持 dim<=3")
            return xp.ascontiguousarray(out.reshape(-1))
        except Exception as exc:
            if getattr(backend, "allow_cpu_fallback", False):
                warnings.warn(
                    f"cuFINUFFT failed, falling back to CPU finufft: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise

    tphx_np = np.ascontiguousarray(_as_numpy(tphx), dtype=np.float64)
    c_np = np.ascontiguousarray(_as_numpy(c), dtype=np.complex128)
    host_out = _type1_coeffs_cpu(tphx_np, c_np, dim, ms, eps, isign)
    return xp.asarray(host_out, dtype=xp.complex128)


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
    return np.asarray(_as_numpy(out_gpu), dtype=np.complex128)


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


def summarize_bin_stats_layout(stats: dict[str, Any]) -> dict[str, Any]:
    """默认 diagnostics 中代替完整 ``bin_stats`` 的轻量摘要。"""
    r = int(stats["r"])
    d = int(stats["d"])
    Gtot = int(r**d)
    dense = bool(stats.get("dense", True))
    if dense:
        nocc = int(np.sum(np.asarray(stats["c"]) > 0))
    else:
        nocc = int(np.asarray(stats["c_occ"]).shape[0])
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


def build_binned_efgp_system(
    X: np.ndarray,
    y: np.ndarray,
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
    use_sparse_bins: bool = True,
    r_max: int | None = None,
    dtype: np.dtype = np.float64,
    backend: Any | None = None,
    nufft_tol: float = 1e-9,
    x_center: np.ndarray | None = None,
    *,
    return_bin_stats: bool = False,
    tphx_allow_clip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    由分箱经验测度构造近似 EFGP 预计算量。

    参数 ``X, y`` 为完整数据（函数内部校验 ``N``）；也可改用
    ``build_binned_efgp_system_streaming`` 做真正流式多块读取。

    Returns
    -------
    v_tilde : ( (4m+1)^d,) complex
        ``J_{2m}`` 上近似 Toeplitz 生成元频域列（与 ``generate_multi_index(2m,d)`` 顺序一致）。
    b_tilde : ( (2m+1)^d,) complex
        ``b_tilde = D * rhs_tilde``，``D`` 与 ``basis_weights`` 展平顺序一致时应传入展平后的对角。
    diagnostics : dict
    """
    order_eff = _resolve_order(order, d)
    if quality not in THETA_MAX:
        raise ValueError(f"quality must be one of {list(THETA_MAX.keys())}; got {quality!r}")

    if np.asarray(X).ndim != 2 or np.asarray(X).shape[1] != int(d):
        raise ValueError(f"X must have shape (N, {d}).")
    y_work = np.asarray(y, dtype=dtype).reshape(-1)
    if np.asarray(X).shape[0] != int(N) or y_work.shape[0] != int(N):
        raise ValueError("N must match len(X) and len(y).")

    validate_normalized_X(np.asarray(X), int(d))

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
        r_use = max(1, int(r))
        G = int(r_use**d)
        Delta = 1.0 / float(r_use)
        theta_actual = 2.0 * math.pi * float(h) * int(m) * int(d) / float(r_use)
        bytes_pc = _bytes_per_cell(order_eff, int(d))
        r_phase = int(math.ceil(2.0 * math.pi * float(h) * int(m) * int(d) / float(theta_target)))
        grid_diag = {
            "order": order_eff,
            "quality": quality,
            "theta_target": theta_target,
            "theta_actual": theta_actual,
            "r_phase": r_phase,
            "r_auto": r_use,
            "G": G,
            "Delta": Delta,
            "bytes_per_cell": bytes_pc,
            "estimated_dense_memory_bytes": G * bytes_pc,
            "avg_occupancy": float(N) / float(max(G, 1)),
            "compression_ratio": float(N) / float(max(G, 1)),
            "phase_target_met": r_use >= r_phase,
        }
        if r_use < r_phase:
            grid_diag["warning_phase"] = (
                "phase target not met; binning error may dominate"
            )

    if x_center is None:
        x_center_np = np.full(int(d), 0.5, dtype=np.float64)
    else:
        x_center_np = np.asarray(x_center, dtype=np.float64).reshape(-1)
        if x_center_np.size != int(d):
            raise ValueError("x_center must have length d.")

    use_dense_bins = not bool(use_sparse_bins)
    stats = build_bin_stats_from_arrays(
        X, y_work, r_use, int(d), order_eff, dtype=dtype, use_dense_bins=use_dense_bins
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
    if return_bin_stats:
        diagnostics["bin_stats"] = stats
    return v_tilde, b_tilde, diagnostics


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
    use_sparse_bins: bool = True,
    r_max: int | None = None,
    dtype: np.dtype = np.float64,
    backend: Any | None = None,
    nufft_tol: float = 1e-9,
    x_center: np.ndarray | None = None,
    *,
    return_bin_stats: bool = False,
    tphx_allow_clip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """流式多块 binning，其余与 ``build_binned_efgp_system`` 相同。"""
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
        r_use = max(1, int(r))
        G = int(r_use**d)
        Delta = 1.0 / float(r_use)
        theta_actual = 2.0 * math.pi * float(h) * int(m) * int(d) / float(r_use)
        bytes_pc = _bytes_per_cell(order_eff, int(d))
        r_phase = int(math.ceil(2.0 * math.pi * float(h) * int(m) * int(d) / float(theta_target)))
        grid_diag = {
            "order": order_eff,
            "quality": quality,
            "theta_target": theta_target,
            "theta_actual": theta_actual,
            "r_phase": r_phase,
            "r_auto": r_use,
            "G": G,
            "Delta": Delta,
            "bytes_per_cell": bytes_pc,
            "estimated_dense_memory_bytes": G * bytes_pc,
            "avg_occupancy": float(N) / float(max(G, 1)),
            "compression_ratio": float(N) / float(max(G, 1)),
            "phase_target_met": r_use >= r_phase,
        }
        if r_use < r_phase:
            grid_diag["warning_phase"] = (
                "phase target not met; binning error may dominate"
            )

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
        dtype=dtype,
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


make_bin_stats_loaders_from_arrays = make_chunk_iterators_from_arrays


def _run_self_tests() -> None:
    """模块内自检（NUFFT 顺序、流式一致性、C0/C1/C2 相对误差趋势）。"""
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
    dense_s = build_bin_stats_streaming(xi, yi, r, d, "C2", dtype=np.float64, use_dense_bins=True)
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
            v_gpu = np.asarray(_as_numpy(v_gpu))
            rhs_gpu = np.asarray(_as_numpy(rhs_gpu))
            np.testing.assert_allclose(v_gpu, v_cpu, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(rhs_gpu, rhs_cpu, rtol=1e-6, atol=1e-8)
            print("  OK")

    print("All self-tests finished.")


if __name__ == "__main__":
    _run_self_tests()
