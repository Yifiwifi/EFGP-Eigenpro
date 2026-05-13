from __future__ import annotations

import gc
import importlib
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from efgp_eigenpro_py.discretization import basis_weights, choose_grid_params
from efgp_eigenpro_py.efgp_solver import EFGPSolver
from efgp_eigenpro_py.gpu.backends import BackendConfig
from efgp_eigenpro_py.gpu.contexts import ensure_gpu_data_context
from efgp_eigenpro_py.gpu.v1_ops import _device_array_to_numpy, predict_v1
from efgp_eigenpro_py.gpu.v3_eigenspace import EigenspaceConfig
from efgp_eigenpro_py.kernels import make_matern, make_squared_exponential

import efgp_eigenpro_py.gpu as _gpu_pkg_bm
import efgp_eigenpro_py.gpu.binned_efgp_precompute as _binned_pc_mod
import efgp_eigenpro_py.gpu.v1_ops as _gpu_v1_ops_bm
import efgp_eigenpro_py.gpu.versions as _gpu_versions_bm

try:
    import cupy as cp
except Exception:  # pragma: no cover - depends on local GPU env
    cp = None


BENCHMARK_DIR = _HERE.parent
PROCESSED_DATA_DIR = BENCHMARK_DIR / "processed"

_BENCHMARK_PC_METHOD_ACTIVE: str | None = None
_GPU_PC_ORIGINAL_FN = None
_LAST_PC_PATCH_EXTRA: dict[str, Any] = {}


@dataclass
class AccuracyBenchmarkConfig:
    dataset_stems: list[str] = field(default_factory=lambda: ["synthetic_true_func_2d_n100000"])
    kernel_specs: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "mat32_ls0.1",
                "family": "matern",
                "nu": 1.5,
                "lengthscale": 0.1,
                "variance": 1.0,
            }
        ]
    )
    seed_base: int = 0
    train_core_fraction: float = 0.9
    n_val_eval: int = 100_000
    n_test_eval: int = 100_000
    target_delta: float = 0.02
    target_window: int = 3
    max_epochs: int = 40
    repeats: int = 1
    reg_lambda: float = 0.1
    solve_tol: float = 1e-6
    gpu_maxiter: int = 6000
    eps_list: list[float] = field(default_factory=lambda: [1e-5])
    l2_scaled: bool = True
    gpu_nufft: str = "auto"
    precompute_methods: dict[str, str] = field(
        default_factory=lambda: {"EFGP-CG": "original", "ours": "c1"}
    )
    precompute_c1_min_n_total: int | None = None
    binned_quality: str = "balanced"
    binned_use_sparse_bins: bool = False
    binned_use_gpu_dense_bins: bool = True
    binned_allow_exact_nufft_fallback: bool = False
    binned_nufft_allow_cpu_fallback: bool = False
    binned_r_user: int | None = None
    v3_topq_list: list[int] = field(default_factory=lambda: [45, 90, 135, 180, 360])
    nystrom_topq_list: list[int] = field(default_factory=lambda: [45, 90, 135, 180, 360])
    extra_topq_list: list[int] = field(default_factory=lambda: [45, 90, 135, 180, 360])
    eig_method_toggles: dict[str, bool] = field(
        default_factory=lambda: {
            "baseline_v1_topq0": True,
            "baseline_v3_topq": True,
            "nystrom_compact_coordinate": True,
            "extra_rand_range_onepass": True,
        }
    )
    v3_oversample: int = 16
    v3_n_iter: int = 3
    eigenpro_nystrom_precond_kind: str = "coordinate_nystrom"
    eigenpro_nystrom_refine_mode: str = "auto"
    eigenpro_nystrom_surrogate_size: int = 1600
    eigenpro_nystrom_lowfreq_ratio: float = 0.5
    eigenpro_nystrom_oversample: int = 10
    eigenpro_nystrom_ritz_refine: bool = True
    eigenpro_nystrom_seed: int = 0
    eigenpro_nystrom_block_rows: int | None = 8192
    eigenpro_nystrom_ritz_block_cols: int = 16
    eigenpro_nystrom_lift: bool = True
    eigenpro_nystrom_refine_iters: int = 1
    eigenpro_coord_nystrom_gamma: float = 1.0
    extra_rand_range_oversample: int = 16
    extra_rand_range_power_iters: int = 0
    extra_rand_range_omega_kind: str = "gaussian"
    extra_rand_range_block_cols: int = 16
    eigenpro3_p_centers_list: list[int] = field(
        default_factory=lambda: [10_000, 30_000, 100_000, 300_000]
    )
    eigenpro2_enabled: bool = True
    eigenpro3_enabled: bool = True
    efgp_cg_enabled: bool = True
    ours_enabled: bool = True
    ep2_mem_gb: float = 8.0
    ep2_top_q: int | None = None
    ep2_n_subsamples: int | None = None
    ep2_lr_scale: float = 0.01
    ep2_stop_on_divergence: bool = True
    ep2_divergence_abs_rmse: float = 5.0
    ep2_divergence_target_factor: float = 50.0
    ep3_nystrom_samples: int = 512
    ep3_data_precond_level: int = 64
    ep3_loader_batch_size: int = 512
    predict_batch_size: int = 2048
    center_chunk: int = 8192
    run_tag: str = field(default_factory=lambda: datetime.now().strftime("accuracy_baselines_%Y%m%d_%H%M%S"))

    @classmethod
    def smoke(cls) -> "AccuracyBenchmarkConfig":
        return cls(
            dataset_stems=["synthetic_true_func_2d_n100000"],
            kernel_specs=[
                {
                    "name": "mat32_ls0.1",
                    "family": "matern",
                    "nu": 1.5,
                    "lengthscale": 0.1,
                    "variance": 1.0,
                }
            ],
            n_val_eval=5_000,
            n_test_eval=5_000,
            max_epochs=3,
            target_window=3,
            v3_topq_list=[45],
            nystrom_topq_list=[45],
            extra_topq_list=[45],
            eigenpro3_p_centers_list=[2_000],
        )


class TensorRegressionDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _clear_state(clear_pool: bool = True) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if clear_pool and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def _sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if cp is not None:
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass


def discover_processed_datasets() -> dict[str, Path]:
    return {p.stem: p for p in sorted(PROCESSED_DATA_DIR.glob("*.npz"))}


def load_dataset(stem: str) -> dict[str, Any]:
    dataset_map = discover_processed_datasets()
    if stem not in dataset_map:
        available = ", ".join(sorted(dataset_map))
        raise FileNotFoundError(f"Unknown dataset stem {stem!r}. Available: {available}")

    path = dataset_map[stem]
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    loaded = np.load(path)
    required = ("x_train", "x_test", "y_train", "y_test")
    missing = [k for k in required if k not in loaded.files]
    if missing:
        raise ValueError(f"Processed dataset {path.name} missing arrays: {missing}")

    x_train = np.asarray(loaded["x_train"], dtype=np.float64)
    x_test = np.asarray(loaded["x_test"], dtype=np.float64)
    y_train = np.asarray(loaded["y_train"], dtype=np.float64).reshape(-1, 1)
    y_test = np.asarray(loaded["y_test"], dtype=np.float64).reshape(-1, 1)
    shapes = meta.get("shapes", {}) if isinstance(meta, dict) else {}
    n_total = shapes.get("n_clean", shapes.get("n_total", int(x_train.shape[0] + x_test.shape[0])))

    return {
        "name": stem,
        "path": str(path),
        "metadata_path": str(meta_path),
        "metadata": meta,
        "available_arrays": sorted(list(loaded.files)),
        "dim": int(x_train.shape[1]),
        "n_total": int(n_total),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "y_std": _resolve_y_std(loaded, meta),
    }


def _resolve_y_std(loaded: np.lib.npyio.NpzFile, meta: dict[str, Any]) -> float:
    if "y_std" in loaded.files:
        arr = np.asarray(loaded["y_std"], dtype=np.float64).reshape(-1)
        if arr.size and np.isfinite(arr[0]) and arr[0] > 0:
            return float(arr[0])
    y_tf = meta.get("y_transform", {}) if isinstance(meta, dict) else {}
    if str(y_tf.get("method", "")).lower() == "train_standardization":
        val = float(y_tf.get("std", np.nan))
        if np.isfinite(val) and val > 0:
            return val
    return float("nan")


def split_train_val(
    dataset_payload: dict[str, Any],
    cfg: AccuracyBenchmarkConfig,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    x_pool = np.asarray(dataset_payload["x_train"], dtype=np.float32)
    y_pool = np.asarray(dataset_payload["y_train"], dtype=np.float32).reshape(-1, 1)
    x_test = np.asarray(dataset_payload["x_test"], dtype=np.float32)
    y_test = np.asarray(dataset_payload["y_test"], dtype=np.float32).reshape(-1, 1)

    rng = np.random.default_rng(int(seed))
    n_pool = int(x_pool.shape[0])
    perm = rng.permutation(n_pool)
    n_core = max(1, min(n_pool - 1, int(round(float(cfg.train_core_fraction) * n_pool))))
    core_idx = perm[:n_core]
    val_idx = perm[n_core:]
    if val_idx.size == 0:
        val_idx = perm[-1:]
        core_idx = perm[:-1]

    x_core = x_pool[core_idx]
    y_core = y_pool[core_idx]
    x_val = x_pool[val_idx]
    y_val = y_pool[val_idx]

    x_val_eval, y_val_eval = sample_eval_subset(x_val, y_val, cfg.n_val_eval, seed=seed + 101)
    x_test_eval, y_test_eval = sample_eval_subset(x_test, y_test, cfg.n_test_eval, seed=seed + 202)

    return {
        "x_train_core": x_core,
        "y_train_core": y_core,
        "x_val": x_val,
        "y_val": y_val,
        "x_val_eval": x_val_eval,
        "y_val_eval": y_val_eval,
        "x_test_eval": x_test_eval,
        "y_test_eval": y_test_eval,
    }


def sample_eval_subset(
    x: np.ndarray,
    y: np.ndarray,
    cap: int | None,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1, 1)
    if cap is None or int(cap) <= 0 or x.shape[0] <= int(cap):
        return x, y
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(x.shape[0], size=int(cap), replace=False)
    return x[idx], y[idx]


def pairwise_sqdist(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    z_norm = (z * z).sum(dim=1, keepdim=True).T
    out = x @ z.T
    out.mul_(-2.0)
    out.add_(x_norm)
    out.add_(z_norm)
    return out.clamp_min_(0.0)


def make_torch_kernel(name: str, bandwidth: float):
    kernel_name = str(name).lower()
    bw = float(bandwidth)
    if bw <= 0.0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth}")

    def gaussian(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(-pairwise_sqdist(x, z) / (2.0 * bw * bw))

    def laplacian(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.sqrt(pairwise_sqdist(x, z)) / bw)

    def matern32(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s = math.sqrt(3.0) * torch.sqrt(pairwise_sqdist(x, z)) / bw
        return (1.0 + s) * torch.exp(-s)

    def matern52(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s = math.sqrt(5.0) * torch.sqrt(pairwise_sqdist(x, z)) / bw
        return (1.0 + s + s * s / 3.0) * torch.exp(-s)

    if kernel_name in {"gaussian", "se", "rbf"}:
        return gaussian
    if kernel_name in {"laplacian", "matern12", "mat12"}:
        return laplacian
    if kernel_name in {"matern32", "mat32"}:
        return matern32
    if kernel_name in {"matern52", "mat52"}:
        return matern52
    raise ValueError(f"unknown torch kernel: {name}")


def make_efgp_kernel(kernel_cfg: dict[str, Any], dim: int):
    fam = str(kernel_cfg.get("family", "")).strip().lower()
    lengthscale = float(kernel_cfg["lengthscale"])
    variance = float(kernel_cfg.get("variance", 1.0))
    if fam in ("matern", "mat"):
        return make_matern(
            lengthscale=lengthscale,
            nu=float(kernel_cfg.get("nu", 1.5)),
            dim=int(dim),
            variance=variance,
        )
    if fam in ("gaussian", "se", "squared_exponential", "squared-exponential", "rbf"):
        return make_squared_exponential(lengthscale=lengthscale, dim=int(dim), variance=variance)
    raise ValueError(f"unsupported EFGP kernel family: {fam!r}")


def eigenpro_kernel_from_cfg(kernel_cfg: dict[str, Any]) -> tuple[str, float]:
    if "kernel" in kernel_cfg and "bandwidth" in kernel_cfg:
        return str(kernel_cfg["kernel"]).strip().lower(), float(kernel_cfg["bandwidth"])
    fam = str(kernel_cfg.get("family", "")).strip().lower()
    ls = float(kernel_cfg["lengthscale"])
    if fam in ("matern", "mat"):
        nu = float(kernel_cfg.get("nu", 1.5))
        if abs(nu - 1.5) < 1e-9:
            return "matern32", ls
        if abs(nu - 2.5) < 1e-9:
            return "matern52", ls
        if abs(nu - 0.5) < 1e-9:
            return "laplacian", ls
        raise ValueError(f"unsupported Matern nu for EigenPro: {nu}")
    if fam in ("gaussian", "se", "squared_exponential", "squared-exponential", "rbf"):
        return "gaussian", ls
    raise ValueError(f"unsupported kernel family for EigenPro: {fam!r}")


def regression_metrics_std(y_true: np.ndarray, y_pred: np.ndarray, y_std: float = float("nan")) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    mse = float(np.mean(err * err))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float("nan") if denom <= 0 else float(1.0 - np.sum(err * err) / denom)
    meter_scale = float(y_std) if np.isfinite(y_std) and y_std > 0 else float("nan")
    return {
        "rmse_std": rmse,
        "mae_std": mae,
        "r2": r2,
        "mse_std": mse,
        "rmse_meter": rmse * meter_scale if np.isfinite(meter_scale) else float("nan"),
        "mae_meter": mae * meter_scale if np.isfinite(meter_scale) else float("nan"),
    }


def _gpu_scalar(x: Any) -> float:
    arr = np.asarray(_device_array_to_numpy(x)).reshape(-1)
    return float(arr[0]) if arr.size else float("nan")


def _gpu_regression_metrics(backend, yhat_gpu, y_true_np: np.ndarray, y_std: float) -> dict[str, float]:
    xp = backend.xp
    yhat_gpu = xp.asarray(yhat_gpu, dtype=xp.float64).reshape(-1)
    y_true_gpu = xp.asarray(np.asarray(y_true_np, dtype=np.float64)).reshape(-1)
    resid_gpu = yhat_gpu - y_true_gpu
    mse = _gpu_scalar(xp.mean(resid_gpu * resid_gpu))
    rmse = float(math.sqrt(max(mse, 0.0)))
    mae = _gpu_scalar(xp.mean(xp.abs(resid_gpu)))
    ss_res = _gpu_scalar(xp.sum(resid_gpu * resid_gpu))
    centered_true = y_true_gpu - xp.mean(y_true_gpu)
    ss_tot = _gpu_scalar(xp.sum(centered_true * centered_true))
    r2 = float("nan") if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    meter_scale = float(y_std) if np.isfinite(y_std) and y_std > 0 else float("nan")
    return {
        "rmse_std": rmse,
        "mae_std": mae,
        "r2": r2,
        "mse_std": mse,
        "rmse_meter": rmse * meter_scale if np.isfinite(meter_scale) else float("nan"),
        "mae_meter": mae * meter_scale if np.isfinite(meter_scale) else float("nan"),
    }


def install_gpu_precompute_patch(cfg: AccuracyBenchmarkConfig) -> None:
    global _GPU_PC_ORIGINAL_FN, _gpu_v1_ops_bm, _gpu_versions_bm

    _gpu_v1_ops_bm = importlib.reload(_gpu_v1_ops_bm)
    _gpu_versions_bm = importlib.reload(_gpu_versions_bm)
    _GPU_PC_ORIGINAL_FN = _gpu_v1_ops_bm.gpu_precompute_v1
    build_binned_efgp_system = _binned_pc_mod.build_binned_efgp_system

    def _wrapped(
        backend,
        kernel,
        eps,
        nufft_tol,
        data_ctx,
        op_ctx=None,
        *,
        l2scaled=False,
        force=False,
        chunk_size=None,
    ):
        global _LAST_PC_PATCH_EXTRA
        pcm = (_BENCHMARK_PC_METHOD_ACTIVE or "original").strip().lower()
        if pcm not in ("c0", "c1", "c2"):
            t0 = time.perf_counter()
            ctx = _GPU_PC_ORIGINAL_FN(
                backend,
                kernel,
                eps,
                nufft_tol,
                data_ctx,
                op_ctx,
                l2scaled=l2scaled,
                force=force,
                chunk_size=chunk_size,
            )
            elapsed = float(time.perf_counter() - t0)
            _LAST_PC_PATCH_EXTRA = {
                "precompute_method_effective": "original",
                "time_precompute_NUFFT": elapsed,
                "time_precompute_binned": float("nan"),
            }
            return ctx

        xp = backend.xp
        ctx = data_ctx
        x_gpu = xp.asarray(ctx.x_gpu, dtype=xp.float64)
        y_gpu = xp.asarray(ctx.y_gpu, dtype=xp.float64).reshape(-1)
        n = int(x_gpu.shape[0])
        dim = int(kernel.dim)
        x_min = xp.min(x_gpu, axis=0)
        x_max = xp.max(x_gpu, axis=0)
        L = float(xp.max(x_max - x_min))
        x_center_gpu = (x_min + x_max) / 2.0
        grid = choose_grid_params(kernel, eps, L, l2scaled=l2scaled)
        mtot = int(grid.mtot)
        hm = (mtot - 1) // 2
        if mtot != 2 * hm + 1:
            raise RuntimeError(f"unexpected mtot={mtot}")

        weights_np = np.ascontiguousarray(basis_weights(kernel, grid.xis, grid.h).reshape(-1))
        weights_flat = xp.asarray(weights_np, dtype=xp.float64).reshape(-1)
        weights_nd = weights_flat.reshape((mtot,) * dim)
        x_center_np = np.asarray(_device_array_to_numpy(x_center_gpu, np.float64)).reshape(-1)

        t0 = time.perf_counter()
        v_tilde, b_tilde, diag_bin = build_binned_efgp_system(
            x_gpu,
            y_gpu,
            n,
            dim,
            float(grid.h),
            hm,
            weights_np,
            order=pcm.upper(),
            quality=str(cfg.binned_quality),
            r=int(cfg.binned_r_user) if cfg.binned_r_user is not None else None,
            use_sparse_bins=bool(cfg.binned_use_sparse_bins),
            use_gpu_dense_bins=bool(cfg.binned_use_gpu_dense_bins),
            return_bin_stats=False,
            x_center=x_center_np,
            backend=backend,
            nufft_tol=float(nufft_tol),
            gpu_timing=True,
            input_on_gpu=True,
            assume_normalized=True,
            skip_cpu_validation=True,
            allow_exact_nufft_fallback=bool(cfg.binned_allow_exact_nufft_fallback),
            nufft_allow_cpu_fallback=bool(cfg.binned_nufft_allow_cpu_fallback),
        )

        ms_xtx = 2 * int(mtot) - 1
        expected_v_size = int(ms_xtx) ** int(dim)
        vt_flat = xp.asarray(v_tilde).reshape(-1)
        if int(vt_flat.size) != expected_v_size:
            raise RuntimeError(f"binned v_tilde size {vt_flat.size} != expected {expected_v_size}")
        xtxcol_gpu = xp.ascontiguousarray(vt_flat.reshape((ms_xtx,) * int(dim)))
        gf_gpu = xp.ascontiguousarray(backend.fft.fftn(xtxcol_gpu))

        expected_rhs_size = int(mtot**dim)
        if cp is None or not isinstance(b_tilde, cp.ndarray):
            raise RuntimeError("C1 binned precompute requires GPU b_tilde from build_binned_efgp_system.")
        rhs_gpu = b_tilde.reshape(-1).astype(xp.complex128, copy=False)
        if int(rhs_gpu.size) != expected_rhs_size:
            raise RuntimeError(f"b_tilde size {rhs_gpu.size} != expected {expected_rhs_size}")

        ctx.weights_gpu_nd = weights_nd
        ctx.weights_gpu_flat = weights_flat
        ctx.weights_np_flat = np.ascontiguousarray(weights_np.reshape(-1))
        ctx.rhs_gpu = rhs_gpu
        ctx.xtxcol_gpu = xtxcol_gpu
        ctx.gf_gpu = gf_gpu
        ctx.x_center_gpu = x_center_gpu
        _sync_gpu()
        elapsed = float(time.perf_counter() - t0)
        breakdown = diag_bin.get("binned_precompute_breakdown_s", None) or {}
        _LAST_PC_PATCH_EXTRA = {
            "precompute_method_effective": pcm,
            "time_precompute_NUFFT": float("nan"),
            "time_precompute_binned": elapsed,
            "precompute_benchmark_note": "binned pcm path: build_binned_efgp_system provides XtXcol + rhs; exact gpu_precompute_v1 is skipped.",
            "binned_theta_actual": float(diag_bin.get("theta_actual", np.nan)),
            "binned_G": int(diag_bin.get("G", -1)),
            "binned_num_occupied_bins": int(diag_bin.get("num_occupied_bins", -1)),
            "effective_work_ratio": float(diag_bin.get("effective_work_ratio", np.nan)),
            "binned_order_bins_final": str(diag_bin.get("order_bins_final", "")),
        }
        for key, value in breakdown.items():
            try:
                _LAST_PC_PATCH_EXTRA[str(key)] = float(value)
            except (TypeError, ValueError):
                _LAST_PC_PATCH_EXTRA[str(key)] = float("nan")

        ctx.meta.update(
            {
                "mtot": mtot,
                "dim": dim,
                "h": float(grid.h),
                "weight_shape": tuple(int(s) for s in weights_nd.shape),
                "gf_shape": tuple(int(s) for s in gf_gpu.shape),
                "rhs_shape": tuple(int(s) for s in rhs_gpu.shape),
                "nufft_tol": float(nufft_tol),
                "nufft_stage": f"binned_{pcm}",
                "chunk_size": None,
                "gf_absmax": float(xp.max(xp.abs(gf_gpu))),
                "rhs_variant": pcm.upper(),
            }
        )
        return ctx

    _gpu_v1_ops_bm.gpu_precompute_v1 = _wrapped
    _gpu_versions_bm.gpu_precompute_v1 = _wrapped
    if hasattr(_gpu_pkg_bm, "gpu_precompute_v1"):
        _gpu_pkg_bm.gpu_precompute_v1 = _wrapped


def _resolve_precompute_method(method_name: str, dataset_payload: dict[str, Any], cfg: AccuracyBenchmarkConfig) -> str:
    policy_key = "ours" if str(method_name).startswith("ours") else str(method_name)
    requested = str(cfg.precompute_methods.get(policy_key, "original")).strip().lower()
    if requested == "c1":
        threshold = cfg.precompute_c1_min_n_total
        if threshold is not None and int(dataset_payload["n_total"]) < int(threshold):
            return "original"
    return requested


def _eig_enabled(cfg: AccuracyBenchmarkConfig, name: str) -> bool:
    return bool((cfg.eig_method_toggles or {}).get(str(name), False))


def make_ours_eigenspace_config(top_q: int, cfg: AccuracyBenchmarkConfig) -> EigenspaceConfig:
    return EigenspaceConfig(
        q_max=int(top_q),
        block_size=int(top_q + cfg.v3_oversample),
        n_iter=int(cfg.v3_n_iter),
        eig_method="eigenpro_nystrom",
        method_cfg={
            "precond_kind": str(cfg.eigenpro_nystrom_precond_kind),
            "coord_nystrom_gamma": float(cfg.eigenpro_coord_nystrom_gamma),
        },
        surrogate_size=int(cfg.eigenpro_nystrom_surrogate_size),
        surrogate_lowfreq_ratio=float(cfg.eigenpro_nystrom_lowfreq_ratio),
        surrogate_oversample=int(cfg.eigenpro_nystrom_oversample),
        surrogate_seed=int(cfg.eigenpro_nystrom_seed),
        surrogate_block_rows=None if cfg.eigenpro_nystrom_block_rows is None else int(cfg.eigenpro_nystrom_block_rows),
        surrogate_ritz_refine=bool(cfg.eigenpro_nystrom_ritz_refine),
        surrogate_ritz_block_cols=int(cfg.eigenpro_nystrom_ritz_block_cols),
        surrogate_lift=bool(cfg.eigenpro_nystrom_lift),
        surrogate_refine_mode=str(cfg.eigenpro_nystrom_refine_mode),
        surrogate_refine_iters=int(cfg.eigenpro_nystrom_refine_iters),
    )


def make_v3_eigenspace_config(method_variant: str, top_q: int, cfg: AccuracyBenchmarkConfig) -> EigenspaceConfig:
    variant = str(method_variant)
    if variant == "baseline_v3_topq":
        return EigenspaceConfig(
            q_max=int(top_q),
            block_size=int(top_q + cfg.v3_oversample),
            n_iter=int(cfg.v3_n_iter),
        )
    if variant == "nystrom_compact_coordinate":
        return make_ours_eigenspace_config(top_q, cfg)
    if variant == "rand_range_onepass":
        return EigenspaceConfig(
            q_max=int(top_q),
            block_size=int(top_q + cfg.v3_oversample),
            n_iter=int(cfg.v3_n_iter),
            eig_method="rand_range_onepass",
            method_cfg={
                "oversample": int(cfg.extra_rand_range_oversample),
                "power_iters": int(cfg.extra_rand_range_power_iters),
                "omega_kind": str(cfg.extra_rand_range_omega_kind),
                "block_cols": int(cfg.extra_rand_range_block_cols),
            },
        )
    raise ValueError(f"Unsupported EFGP eigenspace method_variant: {variant!r}")


def build_fixed_efgp_method_specs(cfg: AccuracyBenchmarkConfig) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if _eig_enabled(cfg, "baseline_v3_topq"):
        specs.extend(
            {
                "method": f"EFGP-V3-q{int(q)}",
                "method_variant": "baseline_v3_topq",
                "top_q": int(q),
            }
            for q in cfg.v3_topq_list
        )
    if bool(cfg.ours_enabled) and _eig_enabled(cfg, "nystrom_compact_coordinate"):
        specs.extend(
            {
                "method": f"ours_q{int(q)}",
                "method_variant": "nystrom_compact_coordinate",
                "top_q": int(q),
            }
            for q in cfg.nystrom_topq_list
        )
    if _eig_enabled(cfg, "extra_rand_range_onepass"):
        specs.extend(
            {
                "method": f"EFGP-rand_range_onepass-q{int(q)}",
                "method_variant": "rand_range_onepass",
                "top_q": int(q),
            }
            for q in cfg.extra_topq_list
        )
    return specs


def run_fixed_efgp_case(
    dataset_payload: dict[str, Any],
    split: dict[str, np.ndarray],
    kernel_cfg: dict[str, Any],
    cfg: AccuracyBenchmarkConfig,
    *,
    eps: float,
    method_name: str,
    method_variant: str,
    top_q: int,
    repeat_idx: int,
) -> dict[str, Any]:
    global _BENCHMARK_PC_METHOD_ACTIVE

    method = str(method_name)
    policy_key = "ours" if method.startswith("ours") else method
    requested_precompute = str(cfg.precompute_methods.get(policy_key, "original")).lower()
    effective_precompute = _resolve_precompute_method(method, dataset_payload, cfg)
    kernel = make_efgp_kernel(kernel_cfg, int(dataset_payload["dim"]))
    solver = EFGPSolver(
        kernel=kernel,
        reg_lambda=float(cfg.reg_lambda),
        eps=float(eps),
        nufft_tol=1e-10,
        l2scaled=bool(cfg.l2_scaled),
    )
    run_cfg = _gpu_versions_bm.GPURunConfig(
        reg_lambda=float(cfg.reg_lambda),
        tol=float(cfg.solve_tol),
        maxiter=int(cfg.gpu_maxiter),
        chunk_size=None,
        debug_finite_checks=False,
        backend=BackendConfig(nufft=str(cfg.gpu_nufft)),
    )

    try:
        _BENCHMARK_PC_METHOD_ACTIVE = effective_precompute
        if method == "EFGP-CG":
            out = _gpu_versions_bm.run_v1_pure_efgp(
                solver,
                split["x_train_core"].astype(np.float64, copy=False),
                split["y_train_core"].reshape(-1).astype(np.float64, copy=False),
                run_cfg,
            )
        else:
            out = _gpu_versions_bm.run_v3_full_gpu_eigenspace(
                solver,
                split["x_train_core"].astype(np.float64, copy=False),
                split["y_train_core"].reshape(-1).astype(np.float64, copy=False),
                run_cfg,
                make_v3_eigenspace_config(method_variant, top_q, cfg),
            )

        diag = dict(out.diagnostics or {})
        patch_extra = dict(_LAST_PC_PATCH_EXTRA)
        yhat_val = predict_v1(out.backend, out.data_ctx, split["x_val_eval"], out.beta_gpu)
        val_metrics = _gpu_regression_metrics(out.backend, yhat_val, split["y_val_eval"], dataset_payload["y_std"])
        yhat_test = predict_v1(out.backend, out.data_ctx, split["x_test_eval"], out.beta_gpu)
        test_metrics = _gpu_regression_metrics(out.backend, yhat_test, split["y_test_eval"], dataset_payload["y_std"])

        train_time = _efgp_train_time(diag)
        return {
            "status": "ok",
            "error": "",
            "method": method,
            "method_variant": str(method_variant),
            "p": np.nan,
            "top_q": int(top_q) if top_q > 0 else np.nan,
            "epochs_to_target": np.nan,
            "fit_time_to_target_s": float(train_time),
            "wall_time_to_target_s": float(train_time),
            "best_val_rmse_std": float(val_metrics["rmse_std"]),
            "val_rmse_std": float(val_metrics["rmse_std"]),
            "test_rmse_std": float(test_metrics["rmse_std"]),
            "test_mae_std": float(test_metrics["mae_std"]),
            "test_r2": float(test_metrics["r2"]),
            "test_rmse_meter": float(test_metrics["rmse_meter"]),
            "test_mae_meter": float(test_metrics["mae_meter"]),
            "time_train_s": float(train_time),
            "time_predict_s": float(diag.get("time_predict", np.nan)),
            "cg_iters": int(diag.get("cg_iters", -1)),
            "cg_relres": float(diag.get("cg_relres", np.nan)),
            "precompute_method_requested": requested_precompute,
            "precompute_method_effective": str(patch_extra.get("precompute_method_effective", effective_precompute)),
            "nufft_stage": str(out.data_ctx.meta.get("nufft_stage", "")),
            **_prefix_keys(diag, "diag_"),
            **_prefix_keys(patch_extra, "pc_"),
        }
    finally:
        _BENCHMARK_PC_METHOD_ACTIVE = None
        _LAST_PC_PATCH_EXTRA.clear()
        _clear_state()


def _efgp_train_time(diag: dict[str, Any]) -> float:
    pieces = [
        diag.get("time_precompute", 0.0),
        diag.get("time_eigenspace", 0.0),
        diag.get("time_precond_build", 0.0),
        diag.get("time_solve", 0.0),
    ]
    total = 0.0
    for val in pieces:
        try:
            f = float(val)
        except (TypeError, ValueError):
            f = 0.0
        if np.isfinite(f):
            total += f
    return float(total)


def _prefix_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, (str, int, float, bool, np.integer, np.floating)) or value is None:
            out[prefix + str(key)] = value
    return out


def _target_reached(history: list[dict[str, Any]], target: float, cfg: AccuracyBenchmarkConfig) -> bool:
    if len(history) < int(cfg.target_window):
        return False
    vals = [float(row["val_rmse_std"]) for row in history[-int(cfg.target_window) :]]
    return float(np.mean(vals)) <= (1.0 + float(cfg.target_delta)) * float(target)


def _select_best_history(history: list[dict[str, Any]], *, budget_s: float | None = None) -> dict[str, Any] | None:
    rows = history
    if budget_s is not None and np.isfinite(budget_s):
        rows = [r for r in rows if float(r.get("elapsed_wall_s", np.inf)) <= float(budget_s)]
    if not rows:
        return None
    return min(rows, key=lambda r: float(r["val_rmse_std"]))


def _is_ep2_divergent(val_rmse: float, target_val_rmse_std: float, cfg: AccuracyBenchmarkConfig) -> bool:
    if not np.isfinite(float(val_rmse)):
        return True
    target_threshold = float(cfg.ep2_divergence_target_factor) * max(float(target_val_rmse_std), 1e-12)
    threshold = max(float(cfg.ep2_divergence_abs_rmse), target_threshold)
    return float(val_rmse) > threshold


def _predict_ep2(model, x: np.ndarray, device: torch.device, batch_size: int, weight_cpu=None) -> np.ndarray:
    x_t = torch.as_tensor(x, dtype=torch.float32)
    weight = None if weight_cpu is None else weight_cpu.to(device)
    preds = []
    with torch.no_grad():
        for start in range(0, x_t.shape[0], int(batch_size)):
            xb = x_t[start : start + int(batch_size)].to(device)
            preds.append(model.forward(xb, weight=weight).detach().cpu().numpy())
    if weight is not None:
        del weight
    return np.vstack(preds)


def run_eigenpro2_target_case(
    dataset_payload: dict[str, Any],
    split: dict[str, np.ndarray],
    kernel_cfg: dict[str, Any],
    cfg: AccuracyBenchmarkConfig,
    *,
    target_val_rmse_std: float,
    time_budget_s: float,
    repeat_idx: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    from eigenpro2.models import KernelModel, asm_eigenpro_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel_name, bandwidth = eigenpro_kernel_from_cfg(kernel_cfg)
    kernel_fn = make_torch_kernel(kernel_name, bandwidth)
    x_train = np.asarray(split["x_train_core"], dtype=np.float32)
    y_train = np.asarray(split["y_train_core"], dtype=np.float32).reshape(-1, 1)
    x_val = np.asarray(split["x_val_eval"], dtype=np.float32)
    y_val = np.asarray(split["y_val_eval"], dtype=np.float32).reshape(-1, 1)

    setup_start = time.perf_counter()
    model = KernelModel(
        kernel_fn,
        torch.as_tensor(x_train, dtype=torch.float32),
        y_dim=1,
        device=device,
    )
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    n_samples, n_labels = y_train_t.shape
    n_subsamples = cfg.ep2_n_subsamples
    if n_subsamples is None:
        n_subsamples = min(n_samples, 2000) if n_samples < 100_000 else 12_000
    mem_bytes = (float(cfg.ep2_mem_gb) - 1.0) * 1024**3
    bsizes = torch.arange(int(n_subsamples), device=device)
    mem_usages = ((model.x_dim + 3 * n_labels + bsizes + 1) * model.n_centers + int(n_subsamples) * 1000) * 4
    bs_gpu = int(torch.sum(mem_usages < mem_bytes).detach().cpu().item())
    bs_gpu = max(1, bs_gpu)

    sample_ids = torch.randperm(n_samples, device=device)[: int(n_subsamples)]
    samples = model.centers[sample_ids]
    eigenpro_f, gap, top_eigval, beta = asm_eigenpro_fn(
        samples, model.kernel_fn, cfg.ep2_top_q, bs_gpu, alpha=0.95, seed=int(cfg.seed_base + repeat_idx)
    )
    new_top_eigval = top_eigval / gap
    bs, eta = model._compute_opt_params(None, bs_gpu, beta, new_top_eigval)
    bs = max(1, int(bs.detach().cpu().item() if torch.is_tensor(bs) else bs))
    eta_val = float(eta.detach().cpu().item() if torch.is_tensor(eta) else eta)
    eta_eff = float(cfg.ep2_lr_scale) * eta_val
    eta_t = model.tensor(eta_eff / bs, dtype=torch.float)
    setup_time = float(time.perf_counter() - setup_start)

    elapsed_fit = setup_time
    elapsed_wall = setup_time
    history: list[dict[str, Any]] = []
    best_weight = model.weight.detach().cpu().clone()
    budget_weight = model.weight.detach().cpu().clone()
    best_row: dict[str, Any] | None = None
    budget_row: dict[str, Any] | None = None
    best_val = float("inf")
    budget_val = float("inf")
    reached = False
    target_fit = float("nan")
    target_wall = float("nan")
    target_epoch = float("nan")
    stop_reason = ""

    for epoch in range(1, int(cfg.max_epochs) + 1):
        epoch_start = time.perf_counter()
        epoch_ids = torch.randperm(n_samples, device=device)
        for batch_ids in torch.split(epoch_ids, bs):
            x_batch = x_train_t[batch_ids]
            y_batch = y_train_t[batch_ids]
            model.eigenpro_iterate(samples, x_batch, y_batch, eigenpro_f, eta_t, sample_ids, batch_ids)
        _sync_gpu()
        epoch_fit = float(time.perf_counter() - epoch_start)
        elapsed_fit += epoch_fit
        elapsed_wall += epoch_fit

        eval_start = time.perf_counter()
        y_val_pred = _predict_ep2(model, x_val, device, int(cfg.predict_batch_size))
        val_eval_time = float(time.perf_counter() - eval_start)
        elapsed_wall += val_eval_time
        val_metrics = regression_metrics_std(y_val, y_val_pred, dataset_payload["y_std"])
        hist_row = {
            "method": "EigenPro2",
            "p": int(x_train.shape[0]),
            "epoch": int(epoch),
            "step": int(epoch),
            "elapsed_fit_s": float(elapsed_fit),
            "elapsed_wall_s": float(elapsed_wall),
            "validation_eval_time_s": float(val_eval_time),
            "val_rmse_std": float(val_metrics["rmse_std"]),
            "accepted_epoch": True,
            "ep2_lr_scale": float(cfg.ep2_lr_scale),
            "ep2_eta_raw": float(eta_val),
            "ep2_eta_effective": float(eta_eff),
            "ep2_batch_size": int(bs),
            "ep2_n_subsamples": int(n_subsamples),
        }
        history.append(hist_row)
        val_rmse = float(hist_row["val_rmse_std"])
        if val_rmse < best_val:
            best_val = val_rmse
            best_row = hist_row
            best_weight = model.weight.detach().cpu().clone()
        if np.isfinite(float(time_budget_s)) and float(elapsed_wall) <= float(time_budget_s) and val_rmse < budget_val:
            budget_val = val_rmse
            budget_row = hist_row
            budget_weight = model.weight.detach().cpu().clone()
        if (not reached) and _target_reached(history, target_val_rmse_std, cfg):
            reached = True
            target_fit = float(elapsed_fit)
            target_wall = float(elapsed_wall)
            target_epoch = int(epoch)
            break
        if bool(cfg.ep2_stop_on_divergence) and _is_ep2_divergent(val_rmse, target_val_rmse_std, cfg):
            hist_row["stopped_reason"] = "diverged"
            stop_reason = "diverged"
            break

    if best_row is None:
        best_row = _select_best_history(history)
    summary = _finalize_eigenpro_summary(
        dataset_payload,
        split,
        cfg,
        method="EigenPro2",
        p=int(x_train.shape[0]),
        reached=reached,
        target_fit=target_fit,
        target_wall=target_wall,
        target_epoch=target_epoch,
        best_row=best_row,
        predict_fn=lambda x: _predict_ep2(model, x, device, int(cfg.predict_batch_size), best_weight),
    )
    if stop_reason:
        summary["stopped_reason"] = stop_reason
    budget = _finalize_budget_summary(
        dataset_payload,
        split,
        cfg,
        method="EigenPro2",
        p=int(x_train.shape[0]),
        time_budget_s=time_budget_s,
        budget_row=budget_row,
        predict_fn=lambda x: _predict_ep2(model, x, device, int(cfg.predict_batch_size), budget_weight),
    )
    _clear_state()
    return summary, budget, history


def sample_centers(x_train: np.ndarray, p: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    p_eff = min(int(p), int(x_train.shape[0]))
    idx = rng.choice(x_train.shape[0], size=p_eff, replace=False)
    return x_train[idx].astype(np.float32, copy=False)


def _predict_ep3_manual(
    x: np.ndarray,
    centers_cpu: torch.Tensor,
    weights_cpu: torch.Tensor,
    kernel_fn,
    *,
    device: torch.device,
    batch_size: int,
    center_chunk: int,
) -> np.ndarray:
    centers = centers_cpu.to(device)
    weights = weights_cpu.to(device)
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], int(batch_size)):
            xb = torch.as_tensor(x[start : start + int(batch_size)], dtype=torch.float32, device=device)
            out = torch.zeros((xb.shape[0], weights.shape[1]), dtype=torch.float32, device=device)
            for c0 in range(0, centers.shape[0], int(center_chunk)):
                out += kernel_fn(xb, centers[c0 : c0 + int(center_chunk)]) @ weights[c0 : c0 + int(center_chunk)]
            preds.append(out.detach().cpu().numpy())
    del centers, weights
    return np.vstack(preds)


def run_eigenpro3_target_case(
    dataset_payload: dict[str, Any],
    split: dict[str, np.ndarray],
    kernel_cfg: dict[str, Any],
    cfg: AccuracyBenchmarkConfig,
    *,
    p_centers: int,
    target_val_rmse_std: float,
    time_budget_s: float,
    repeat_idx: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    from eigenpro3.models import KernelModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernel_name, bandwidth = eigenpro_kernel_from_cfg(kernel_cfg)
    kernel_fn = make_torch_kernel(kernel_name, bandwidth)
    x_train = np.asarray(split["x_train_core"], dtype=np.float32)
    y_train = np.asarray(split["y_train_core"], dtype=np.float32).reshape(-1, 1)
    x_val = np.asarray(split["x_val_eval"], dtype=np.float32)
    y_val = np.asarray(split["y_val_eval"], dtype=np.float32).reshape(-1, 1)
    p_eff = min(int(p_centers), int(x_train.shape[0]))
    centers_np = sample_centers(x_train, p_eff, seed=int(cfg.seed_base + repeat_idx))
    rng = np.random.default_rng(int(cfg.seed_base + repeat_idx) + 17)
    ns = min(int(cfg.ep3_nystrom_samples), int(x_train.shape[0]))
    nys_idx = rng.choice(x_train.shape[0], size=ns, replace=False)

    train_loader = DataLoader(
        TensorRegressionDataset(x_train, y_train),
        batch_size=int(cfg.ep3_loader_batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    setup_start = time.perf_counter()
    model = KernelModel(
        1,
        torch.as_tensor(centers_np, dtype=torch.float32),
        kernel_fn,
        X=None,
        y=None,
        devices=[device],
        make_dataloader=False,
        nystrom_samples=torch.as_tensor(x_train[nys_idx], dtype=torch.float32),
        n_nystrom_samples=ns,
        data_preconditioner_level=int(cfg.ep3_data_precond_level),
    )
    setup_time = float(time.perf_counter() - setup_start)

    elapsed_fit = setup_time
    elapsed_wall = setup_time
    history: list[dict[str, Any]] = []
    best_weights = model.weights.detach().cpu().clone()
    budget_weights = model.weights.detach().cpu().clone()
    best_row: dict[str, Any] | None = None
    budget_row: dict[str, Any] | None = None
    best_val = float("inf")
    budget_val = float("inf")
    centers_cpu = model.centers.detach().cpu().clone()
    reached = False
    target_fit = float("nan")
    target_wall = float("nan")
    target_epoch = float("nan")

    for epoch in range(1, int(cfg.max_epochs) + 1):
        epoch_start = time.perf_counter()
        model.epoch = epoch - 1
        model.fit_epoch([train_loader])
        _sync_gpu()
        epoch_fit = float(time.perf_counter() - epoch_start)
        elapsed_fit += epoch_fit
        elapsed_wall += epoch_fit

        eval_start = time.perf_counter()
        y_val_pred = _predict_ep3_manual(
            x_val,
            centers_cpu,
            model.weights.detach().cpu(),
            kernel_fn,
            device=device,
            batch_size=int(cfg.predict_batch_size),
            center_chunk=int(cfg.center_chunk),
        )
        val_eval_time = float(time.perf_counter() - eval_start)
        elapsed_wall += val_eval_time
        val_metrics = regression_metrics_std(y_val, y_val_pred, dataset_payload["y_std"])
        hist_row = {
            "method": "EigenPro3",
            "p": int(p_eff),
            "epoch": int(epoch),
            "step": int(epoch),
            "elapsed_fit_s": float(elapsed_fit),
            "elapsed_wall_s": float(elapsed_wall),
            "validation_eval_time_s": float(val_eval_time),
            "val_rmse_std": float(val_metrics["rmse_std"]),
            "accepted_epoch": True,
        }
        history.append(hist_row)
        val_rmse = float(hist_row["val_rmse_std"])
        if val_rmse < best_val:
            best_val = val_rmse
            best_row = hist_row
            best_weights = model.weights.detach().cpu().clone()
        if np.isfinite(float(time_budget_s)) and float(elapsed_wall) <= float(time_budget_s) and val_rmse < budget_val:
            budget_val = val_rmse
            budget_row = hist_row
            budget_weights = model.weights.detach().cpu().clone()
        if (not reached) and _target_reached(history, target_val_rmse_std, cfg):
            reached = True
            target_fit = float(elapsed_fit)
            target_wall = float(elapsed_wall)
            target_epoch = int(epoch)
            break

    if best_row is None:
        best_row = _select_best_history(history)
    summary = _finalize_eigenpro_summary(
        dataset_payload,
        split,
        cfg,
        method="EigenPro3",
        p=int(p_eff),
        reached=reached,
        target_fit=target_fit,
        target_wall=target_wall,
        target_epoch=target_epoch,
        best_row=best_row,
        predict_fn=lambda x: _predict_ep3_manual(
            x,
            centers_cpu,
            best_weights,
            kernel_fn,
            device=device,
            batch_size=int(cfg.predict_batch_size),
            center_chunk=int(cfg.center_chunk),
        ),
    )
    budget = _finalize_budget_summary(
        dataset_payload,
        split,
        cfg,
        method="EigenPro3",
        p=int(p_eff),
        time_budget_s=time_budget_s,
        budget_row=budget_row,
        predict_fn=lambda x: _predict_ep3_manual(
            x,
            centers_cpu,
            budget_weights,
            kernel_fn,
            device=device,
            batch_size=int(cfg.predict_batch_size),
            center_chunk=int(cfg.center_chunk),
        ),
    )
    _clear_state()
    return summary, budget, history


def _finalize_eigenpro_summary(
    dataset_payload: dict[str, Any],
    split: dict[str, np.ndarray],
    cfg: AccuracyBenchmarkConfig,
    *,
    method: str,
    p: int,
    reached: bool,
    target_fit: float,
    target_wall: float,
    target_epoch: float,
    best_row: dict[str, Any] | None,
    predict_fn,
) -> dict[str, Any]:
    if best_row is None:
        raise RuntimeError(f"{method} produced no validation history.")
    y_pred = predict_fn(split["x_test_eval"])
    test_metrics = regression_metrics_std(split["y_test_eval"], y_pred, dataset_payload["y_std"])
    stopped_reason = ""
    if best_row is not None:
        stopped_reason = str(best_row.get("stopped_reason", ""))
    return {
        "status": "ok",
        "error": "",
        "method": method,
        "p": int(p),
        "top_q": np.nan,
        "reached_target": bool(reached),
        "fit_time_to_target_s": float(target_fit) if reached else np.nan,
        "wall_time_to_target_s": float(target_wall) if reached else np.nan,
        "epochs_to_target": float(target_epoch) if reached else np.nan,
        "best_val_rmse_std": float(best_row["val_rmse_std"]),
        "test_rmse_std": float(test_metrics["rmse_std"]),
        "test_mae_std": float(test_metrics["mae_std"]),
        "test_r2": float(test_metrics["r2"]),
        "test_rmse_meter": float(test_metrics["rmse_meter"]),
        "test_mae_meter": float(test_metrics["mae_meter"]),
        "precompute_method_requested": np.nan,
        "precompute_method_effective": np.nan,
        "nystrom_samples": np.nan if method == "EigenPro2" else int(cfg.ep3_nystrom_samples),
        "epochs": int(cfg.max_epochs),
        "stopped_reason": stopped_reason,
    }


def _finalize_budget_summary(
    dataset_payload: dict[str, Any],
    split: dict[str, np.ndarray],
    cfg: AccuracyBenchmarkConfig,
    *,
    method: str,
    p: int,
    time_budget_s: float,
    budget_row: dict[str, Any] | None,
    predict_fn,
) -> dict[str, Any]:
    if budget_row is None:
        return {
            "method": method,
            "p": int(p),
            "time_budget_source": "EFGP-CG",
            "time_budget_s": float(time_budget_s),
            "best_val_rmse_std_within_budget": np.nan,
            "epoch_at_best_val": np.nan,
            "test_rmse_std_at_best_val": np.nan,
            "test_mae_std_at_best_val": np.nan,
            "test_r2_at_best_val": np.nan,
            "test_rmse_meter_at_best_val": np.nan,
            "test_mae_meter_at_best_val": np.nan,
        }
    y_pred = predict_fn(split["x_test_eval"])
    test_metrics = regression_metrics_std(split["y_test_eval"], y_pred, dataset_payload["y_std"])
    return {
        "method": method,
        "p": int(p),
        "time_budget_source": "EFGP-CG",
        "time_budget_s": float(time_budget_s),
        "best_val_rmse_std_within_budget": float(budget_row["val_rmse_std"]),
        "epoch_at_best_val": int(budget_row["epoch"]),
        "test_rmse_std_at_best_val": float(test_metrics["rmse_std"]),
        "test_mae_std_at_best_val": float(test_metrics["mae_std"]),
        "test_r2_at_best_val": float(test_metrics["r2"]),
        "test_rmse_meter_at_best_val": float(test_metrics["rmse_meter"]),
        "test_mae_meter_at_best_val": float(test_metrics["mae_meter"]),
    }


def _base_case_cols(
    dataset_payload: dict[str, Any],
    split: dict[str, np.ndarray],
    kernel_cfg: dict[str, Any],
    cfg: AccuracyBenchmarkConfig,
    *,
    eps: float,
    repeat_idx: int,
) -> dict[str, Any]:
    return {
        "run_tag": cfg.run_tag,
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_payload["name"],
        "dataset_path": dataset_payload["path"],
        "N_train": int(split["x_train_core"].shape[0]),
        "N_val_eval": int(split["x_val_eval"].shape[0]),
        "N_test_eval": int(split["x_test_eval"].shape[0]),
        "n_train_pool": int(dataset_payload["n_train"]),
        "n_test_full": int(dataset_payload["n_test"]),
        "dim": int(dataset_payload["dim"]),
        "kernel": str(kernel_cfg["name"]),
        "kernel_name": str(kernel_cfg["name"]),
        "kernel_family": str(kernel_cfg.get("family", "")),
        "kernel_lengthscale": float(kernel_cfg.get("lengthscale", np.nan)),
        "kernel_bandwidth": float(kernel_cfg.get("bandwidth", kernel_cfg.get("lengthscale", np.nan))),
        "kernel_nu": float(kernel_cfg.get("nu", np.nan)) if kernel_cfg.get("nu", None) is not None else np.nan,
        "reg_lambda": float(cfg.reg_lambda),
        "eps": float(eps),
        "repeat_idx": int(repeat_idx),
        "target_delta": float(cfg.target_delta),
        "target_window": int(cfg.target_window),
    }


def _fixed_budget_row_from_summary(summary: dict[str, Any], time_budget_s: float) -> dict[str, Any]:
    within = float(summary.get("time_train_s", np.inf)) <= float(time_budget_s)
    return {
        "method": summary["method"],
        "p": summary.get("p", np.nan),
        "time_budget_source": "EFGP-CG",
        "time_budget_s": float(time_budget_s),
        "best_val_rmse_std_within_budget": float(summary["best_val_rmse_std"]) if within else np.nan,
        "epoch_at_best_val": np.nan,
        "test_rmse_std_at_best_val": float(summary["test_rmse_std"]) if within else np.nan,
        "test_mae_std_at_best_val": float(summary["test_mae_std"]) if within else np.nan,
        "test_r2_at_best_val": float(summary["test_r2"]) if within else np.nan,
        "test_rmse_meter_at_best_val": float(summary["test_rmse_meter"]) if within else np.nan,
        "test_mae_meter_at_best_val": float(summary["test_mae_meter"]) if within else np.nan,
    }


def run_accuracy_benchmark(cfg: AccuracyBenchmarkConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AccuracyBenchmarkConfig()
    install_gpu_precompute_patch(cfg)

    out_dir = BENCHMARK_DIR / "outputs" / cfg.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "benchmark_config.json"
    time_to_target_path = out_dir / "time_to_target_summary.csv"
    same_budget_path = out_dir / "same_time_budget_summary.csv"
    history_path = out_dir / "raw_eval_history.csv"

    config_payload = asdict(cfg)
    config_payload["precompute_policy_note"] = "EFGP-CG uses original; ours_q* requests c1 by default. Set precompute_c1_min_n_total to force small datasets back to original."
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    for dataset_stem in cfg.dataset_stems:
        dataset_payload = load_dataset(dataset_stem)
        for repeat_idx in range(int(cfg.repeats)):
            seed = int(cfg.seed_base + repeat_idx)
            split = split_train_val(dataset_payload, cfg, seed=seed)
            for kernel_cfg in cfg.kernel_specs:
                for eps in cfg.eps_list:
                    base = _base_case_cols(dataset_payload, split, kernel_cfg, cfg, eps=float(eps), repeat_idx=repeat_idx)
                    target_row: dict[str, Any] | None = None

                    if cfg.efgp_cg_enabled:
                        try:
                            fixed = run_fixed_efgp_case(
                                dataset_payload,
                                split,
                                kernel_cfg,
                                cfg,
                                eps=float(eps),
                                method_name="EFGP-CG",
                                method_variant="baseline_v1_topq0",
                                top_q=0,
                                repeat_idx=repeat_idx,
                            )
                            target_row = {
                                **base,
                                **fixed,
                                "target_val_rmse_std": float(fixed["best_val_rmse_std"]),
                                "reached_target": True,
                            }
                        except Exception as exc:
                            target_row = {
                                **base,
                                "method": "EFGP-CG",
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "target_val_rmse_std": np.nan,
                                "reached_target": False,
                            }
                            traceback.print_exc()
                        summary_rows.append(target_row)

                    if target_row is None or not np.isfinite(float(target_row.get("target_val_rmse_std", np.nan))):
                        pd.DataFrame(summary_rows).to_csv(time_to_target_path, index=False)
                        continue

                    target_val = float(target_row["target_val_rmse_std"])
                    time_budget_s = float(target_row.get("time_train_s", target_row.get("fit_time_to_target_s", np.nan)))
                    budget_rows.append({**base, **_fixed_budget_row_from_summary(target_row, time_budget_s)})

                    for fixed_spec in build_fixed_efgp_method_specs(cfg):
                        method_name = str(fixed_spec["method"])
                        method_variant = str(fixed_spec["method_variant"])
                        q = int(fixed_spec["top_q"])
                        if method_name.startswith("ours") and not bool(cfg.ours_enabled):
                            continue
                        try:
                            fixed = run_fixed_efgp_case(
                                dataset_payload,
                                split,
                                kernel_cfg,
                                cfg,
                                eps=float(eps),
                                method_name=method_name,
                                method_variant=method_variant,
                                top_q=int(q),
                                repeat_idx=repeat_idx,
                            )
                            fixed["reached_target"] = bool(
                                float(fixed["best_val_rmse_std"]) <= (1.0 + float(cfg.target_delta)) * target_val
                            )
                            row = {**base, **fixed, "target_val_rmse_std": target_val}
                            summary_rows.append(row)
                            budget_rows.append({**base, **_fixed_budget_row_from_summary(row, time_budget_s)})
                        except Exception as exc:
                            traceback.print_exc()
                            summary_rows.append(
                                {
                                    **base,
                                    "method": method_name,
                                    "method_variant": method_variant,
                                    "top_q": q,
                                    "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "target_val_rmse_std": target_val,
                                    "reached_target": False,
                                }
                            )
                        finally:
                            pd.DataFrame(summary_rows).to_csv(time_to_target_path, index=False)
                            pd.DataFrame(budget_rows).to_csv(same_budget_path, index=False)

                    if cfg.eigenpro2_enabled:
                        try:
                            ep_summary, ep_budget, ep_history = run_eigenpro2_target_case(
                                dataset_payload,
                                split,
                                kernel_cfg,
                                cfg,
                                target_val_rmse_std=target_val,
                                time_budget_s=time_budget_s,
                                repeat_idx=repeat_idx,
                            )
                            summary_rows.append({**base, **ep_summary, "target_val_rmse_std": target_val})
                            budget_rows.append({**base, **ep_budget})
                            history_rows.extend({**base, **h, "target_val_rmse_std": target_val} for h in ep_history)
                        except Exception as exc:
                            traceback.print_exc()
                            summary_rows.append(
                                {
                                    **base,
                                    "method": "EigenPro2",
                                    "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "target_val_rmse_std": target_val,
                                    "reached_target": False,
                                }
                            )
                        finally:
                            pd.DataFrame(summary_rows).to_csv(time_to_target_path, index=False)
                            pd.DataFrame(budget_rows).to_csv(same_budget_path, index=False)
                            pd.DataFrame(history_rows).to_csv(history_path, index=False)

                    if cfg.eigenpro3_enabled:
                        seen_p: set[int] = set()
                        for p in cfg.eigenpro3_p_centers_list:
                            p_eff = min(int(p), int(split["x_train_core"].shape[0]))
                            if p_eff in seen_p:
                                continue
                            seen_p.add(p_eff)
                            try:
                                ep_summary, ep_budget, ep_history = run_eigenpro3_target_case(
                                    dataset_payload,
                                    split,
                                    kernel_cfg,
                                    cfg,
                                    p_centers=p_eff,
                                    target_val_rmse_std=target_val,
                                    time_budget_s=time_budget_s,
                                    repeat_idx=repeat_idx,
                                )
                                summary_rows.append({**base, **ep_summary, "target_val_rmse_std": target_val})
                                budget_rows.append({**base, **ep_budget})
                                history_rows.extend({**base, **h, "target_val_rmse_std": target_val} for h in ep_history)
                            except Exception as exc:
                                traceback.print_exc()
                                summary_rows.append(
                                    {
                                        **base,
                                        "method": "EigenPro3",
                                        "p": p_eff,
                                        "status": "error",
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "target_val_rmse_std": target_val,
                                        "reached_target": False,
                                    }
                                )
                            finally:
                                pd.DataFrame(summary_rows).to_csv(time_to_target_path, index=False)
                                pd.DataFrame(budget_rows).to_csv(same_budget_path, index=False)
                                pd.DataFrame(history_rows).to_csv(history_path, index=False)

    pd.DataFrame(summary_rows).to_csv(time_to_target_path, index=False)
    pd.DataFrame(budget_rows).to_csv(same_budget_path, index=False)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    return {
        "out_dir": out_dir,
        "time_to_target_summary": pd.DataFrame(summary_rows),
        "same_time_budget_summary": pd.DataFrame(budget_rows),
        "raw_eval_history": pd.DataFrame(history_rows),
        "config_path": config_path,
        "time_to_target_path": time_to_target_path,
        "same_budget_path": same_budget_path,
        "history_path": history_path,
    }
