"""Generate the single Colab orchestration notebook for all BTAB experiments."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


HERE = Path(__file__).resolve().parent
REFERENCE_NOTEBOOK = HERE.parent / "boxeig_inverse_diagnostics_experiment.ipynb"
OUTPUT_NOTEBOOK = HERE.parent / "colab_all_experiments_10m_300m.ipynb"


def _source(text: str) -> list[str]:
    cooked = textwrap.dedent(text).strip("\n") + "\n"
    return cooked.splitlines(keepends=True)


def _markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def build_notebook() -> dict:
    reference = json.loads(REFERENCE_NOTEBOOK.read_text(encoding="utf-8"))
    legacy_policy = "".join(reference["cells"][8]["source"])

    cells: list[dict] = []
    cells.append(
        _markdown(
            r"""
            # EFGP / Active-Box 全实验 Colab 总控（10M–300M）

            本 notebook 把原有实验和新增实验放在同一入口中，但保留三条**不可混表计时**的证据链：

            | protocol | 内容 | 可以说明什么 |
            |---|---|---|
            | `archived_complete_pipeline` | 原 `group_a/group_b/group_c`，direct CG 与 binned-C1 candidates，含 setup/solve/prediction | 原论文探索性规模图与候选筛选 |
            | `controlled_fixed_system` | 所有方法共享同一个哈希 (A\beta=b)，1 warm-up + 5 paired repeats | 方法间严格配对加速、构造/求解/存储权衡 |
            | `prediction_audit` | 另建固定系统，仅验证 test RMSE | 解与预测等价性；其中时间不进入 speedup claim |

            使用方式：先选择 Colab 的 **GPU + High-RAM** runtime，再按顺序执行。所有重计算开关默认关闭，避免误触发 300M。建议先完成 10M smoke/center，再逐个启用 30M、100M、300M；每个 controlled case 通过 `suite --resume` 独立检查点恢复。

            参考并保留了 `boxeig_inverse_diagnostics_experiment.ipynb` 中最关键的 direct/binned precompute policy；统一结果区重新整理 legacy、controlled 和 audit 三种 schema。
            """
        )
    )
    cells.append(
        _code(
            r"""
            # ==================== 用户总开关：只改这一格 ====================
            from pathlib import Path

            REPO_URL = "https://github.com/Yifiwifi/EFGP-Eigenpro.git"
            REPO_REF = "main"  # 正式运行前建议改成已 push 的 commit SHA
            LOCAL_REPO = Path("/content/EFGP-Eigenpro")

            DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/EFGP_Colab")
            DRIVE_DATA_ROOT = DRIVE_PROJECT_ROOT / "data_bundle"
            RUN_TAG = "paper_colab_20260824"  # 协议或代码改变时换一个 tag；同 tag 用于断点续跑
            DRIVE_RUN_ROOT = DRIVE_PROJECT_ROOT / "runs" / RUN_TAG
            LOCAL_DATA_DIR = Path("/content/efgp_data")

            # 从 drive_manifest.json 选择数据。下方会根据实验开关自动补齐必需 bundle；
            # 这里只放希望额外缓存到本地 SSD 的 bundle。
            DATA_BUNDLES = ["controlled_10m"]
            CACHE_DATA_LOCALLY = True       # 正式计时推荐 True，避免 Drive FUSE 进入 timing
            VERIFY_FULL_SHA256 = False      # 首次上传后设 True；11GB 全校验会花几分钟

            # 原 notebook 的完整 exploratory pipeline；可选 group_a/group_b/group_c。
            RUN_LEGACY_GROUPS = []

            # Controlled 开关。
            RUN_PLUMBING_SMOKE = False
            RUN_CG_SCREEN_10M = False
            RUN_Q256_CENTER_10M = False
            RUN_ARCHIVED_EXACT_SCALE = False     # 原数据定义；每个 N 是独立 exact artifact
            RUN_DEVELOPMENT_MASTER_SCALE = False  # low-noise Synthetic + raw-prefix Winnebago，新 scale protocol
            RUN_MANITOWOC_SCALE = False           # 需要先生成 Manitowoc 300M master
            RUN_WINNEBAGO_OAT_10M = False
            RUN_Q128_BRIDGE = False
            RUN_SE_FULL_INVERSE_CONTROL = False
            # 新 runtime 只开 audit 时，仍应同时保留产生相应 config 的 RUN_* 开关，
            # 以便自动 staging 那个 profile 所需的数据 bundle；suite 会用 --resume 快速跳过已完成 case。
            RUN_PREDICTION_AUDIT = False

            ACTIVE_SIZES = [10_000_000]
            ALLOW_100M = False
            ALLOW_300M = False
            PREDICTION_AUDIT_MAX_TRAIN_N = 10_000_000

            # 缺失数据的可选生成动作；默认不执行。
            # controlled scale 缺 30/100/300M；完整 legacy groups 还需补 1/3M。
            GENERATE_ARCHIVED_SYNTHETIC_SIZES = []  # noise=.3 / chunk=5M / _ntrainN
            GENERATE_MANITOWOC_300M = False
            MANITOWOC_START_LOD = 8  # 只是起点；容量不足时必须提高并重新精确扫描
            SYNC_GENERATED_DATA_TO_DRIVE = False

            required_bundles = []
            if RUN_LEGACY_GROUPS:
                required_bundles.append("legacy_named_route_inputs")
            if RUN_ARCHIVED_EXACT_SCALE:
                required_bundles.append("archived_exact_available")
            if RUN_DEVELOPMENT_MASTER_SCALE:
                required_bundles.append("development_scale_masters")
            if RUN_MANITOWOC_SCALE:
                required_bundles.append("manitowoc_10m")
            if any([
                RUN_CG_SCREEN_10M, RUN_Q256_CENTER_10M, RUN_WINNEBAGO_OAT_10M,
                RUN_Q128_BRIDGE, RUN_SE_FULL_INVERSE_CONTROL,
            ]):
                required_bundles.append("controlled_10m")
            DATA_BUNDLES = list(dict.fromkeys([*DATA_BUNDLES, *required_bundles]))

            requested_size_hints = [int(n) for n in ACTIVE_SIZES]
            if any(group in {"group_b", "group_c"} for group in RUN_LEGACY_GROUPS):
                requested_size_hints.append(300_000_000)
            elif "group_a" in RUN_LEGACY_GROUPS:
                requested_size_hints.append(30_000_000)
            requested_size_hints.extend(int(n) for n in GENERATE_ARCHIVED_SYNTHETIC_SIZES)
            if GENERATE_MANITOWOC_300M:
                requested_size_hints.append(300_000_000)
            if RUN_PREDICTION_AUDIT:
                requested_size_hints.append(int(PREDICTION_AUDIT_MAX_TRAIN_N))
            MAX_REQUESTED_N = max(requested_size_hints or [0])

            DISCONNECT_RUNTIME_WHEN_VERIFIED = False
            print("Configuration loaded. No heavy experiment has started.")
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## 1. 挂载 Drive、固定输出目录

            数据与结果分开：Drive 长期保存 master/manifest 和 run artifacts；当前 case 的数据先复制到 `/content` 本地 SSD 后再计时。
            """
        )
    )
    cells.append(
        _code(
            r"""
            import os, sys, json, shutil, subprocess, time, hashlib, platform
            from pathlib import Path
            from IPython.display import display

            IS_COLAB = "google.colab" in sys.modules
            if IS_COLAB:
                from google.colab import drive
                drive.mount("/content/drive")
            else:
                print("Not running in Colab; Drive mount skipped.")

            DRIVE_PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            DRIVE_RUN_ROOT.mkdir(parents=True, exist_ok=True)
            LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
            print("Drive project:", DRIVE_PROJECT_ROOT)
            print("Local data cache:", LOCAL_DATA_DIR)
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## 2. Clone/pin 代码并安装 Colab 依赖

            正式论文运行必须把 `REPO_REF` 固定到 commit SHA，并在最终 run manifest 中记录。不要同时安装多个 CuPy wheel；本格先检查已有 CuPy，再补齐 cuFINUFFT 和通用依赖。当前内部 runner 不需要额外安装 EigenPro2/3。
            """
        )
    )
    cells.append(
        _code(
            r"""
            def run_cmd(args, *, cwd=None, env=None):
                args = [str(x) for x in args]
                print("+", " ".join(args))
                return subprocess.run(args, cwd=cwd, env=env, check=True)

            if not LOCAL_REPO.exists():
                run_cmd(["git", "clone", REPO_URL, str(LOCAL_REPO)])
            run_cmd(["git", "fetch", "--all", "--tags"], cwd=LOCAL_REPO)
            checkout_ref = f"origin/{REPO_REF}" if REPO_REF in {"main", "master"} else REPO_REF
            run_cmd(["git", "checkout", "--detach", checkout_ref], cwd=LOCAL_REPO)

            run_cmd([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
            installed = json.loads(subprocess.check_output(
                [sys.executable, "-m", "pip", "list", "--format=json"], text=True
            ))
            cupy_distributions = sorted(
                row["name"] for row in installed
                if row["name"].lower() == "cupy" or row["name"].lower().startswith("cupy-cuda")
            )
            if len(cupy_distributions) > 1:
                raise RuntimeError(f"Multiple CuPy distributions are installed: {cupy_distributions}")
            if not cupy_distributions:
                run_cmd([sys.executable, "-m", "pip", "install", "cupy-cuda12x"])

            colab_dependencies = [
                sys.executable, "-m", "pip", "install",
                "numpy>=1.23,<3", "scipy>=1.10,<2", "finufft>=2.3,<3",
                "psutil>=5.9,<8", "pandas", "matplotlib", "cufinufft>=2.4,<3",
            ]
            if GENERATE_MANITOWOC_300M:
                colab_dependencies.extend(["pyproj>=3.6,<4", "laspy", "lazrs"])
            run_cmd(colab_dependencies)

            import cupy as cp
            print("CuPy distribution:", cupy_distributions or ["cupy-cuda12x (installed above)"])
            print("CuPy version:", cp.__version__)

            os.chdir(LOCAL_REPO)
            if str(LOCAL_REPO) not in sys.path:
                sys.path.insert(0, str(LOCAL_REPO))
            GIT_SHA = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=LOCAL_REPO, text=True
            ).strip()
            print("Pinned Git SHA:", GIT_SHA)
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## 3. GPU / RAM / 磁盘 preflight

            当前 controlled loader 的训练 (x,y) 会转成 float64，最低常驻量约为 (24N) bytes。300M 仅训练数组就约 6.71 GiB GPU；建议 A100 40/80GB 且可用 host RAM ≥20GB。L4 的 300M 只能先试 `cg/default/full-eig`，不能预先承诺全六方法成功。
            """
        )
    )
    cells.append(
        _code(
            r"""
            import numpy as np, pandas as pd, psutil
            import cupy as cp
            import scipy, cufinufft, finufft

            run_cmd(["nvidia-smi"])
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
            gpu_total = int(cp.cuda.runtime.memGetInfo()[1])
            host_available = int(psutil.virtual_memory().available)
            disk_free = int(shutil.disk_usage("/content").free) if Path("/content").exists() else int(shutil.disk_usage(".").free)
            runtime_info = {
                "git_sha": GIT_SHA,
                "gpu": gpu_name,
                "gpu_total_bytes": gpu_total,
                "host_available_bytes": host_available,
                "local_disk_free_bytes": disk_free,
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "cupy": cp.__version__,
                "cufinufft": getattr(cufinufft, "__version__", "unknown"),
                "finufft": getattr(finufft, "__version__", "unknown"),
            }
            display(pd.DataFrame([
                {
                    "N": n,
                    "train_x_y_fp64_GiB": 24*n/2**30,
                    "host_load_peak_est_GiB": 28*n/2**30,
                    "logical_test_rows": n//4,
                }
                for n in [10_000_000, 30_000_000, 100_000_000, 300_000_000]
            ]))
            print(json.dumps(runtime_info, indent=2))

            if MAX_REQUESTED_N >= 100_000_000 and not ALLOW_100M:
                raise RuntimeError("The selected workload reaches >=100M; set ALLOW_100M=True after reviewing memory.")
            if MAX_REQUESTED_N >= 300_000_000:
                if not ALLOW_300M:
                    raise RuntimeError("300M is gated; set ALLOW_300M=True explicitly.")
                if gpu_total < 30 * 2**30 or host_available < 20 * 2**30:
                    raise RuntimeError("300M requires >=30 GiB GPU and >=20 GiB currently available host RAM.")
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## 4. Drive manifest 校验与单份 master staging

            `drive_manifest.json` 由 `colab_drive_pack.py` 生成。开发用 300M Synthetic（noise=.02）和 Winnebago raw master 只属于新的 `controlled_master_prefix` protocol，不能冒充论文 archived Synthetic（noise=.3）或原 spatial-stratified exact artifacts。

            对 ZIP_STORED master，controlled runner 的 `subset_mode='prefix'` 会直接 memory-map 所需的 10M/30M/100M 前缀；不会先加载全部 300M。10M 正式输入登记到 `controlled_10m`；旧 named routes 的现有输入登记到 `legacy_named_route_inputs`，但 notebook 仍会强制检查完整 noise=.3 Synthetic `_ntrainN`，不会接受 low-noise fallback。
            """
        )
    )
    cells.append(
        _code(
            r"""
            from efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack import (
                compare_nested_prefix,
                inspect_stored_npz,
                verify_catalog,
            )

            DATA_MANIFEST = DRIVE_DATA_ROOT / "drive_manifest.json"
            if not DATA_MANIFEST.is_file():
                raise FileNotFoundError(
                    f"Missing {DATA_MANIFEST}. Build/upload the catalog using COLAB_DRIVE_DATA.md first."
                )
            catalog = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
            available_bundles = sorted(catalog.get("bundles", {}))
            print("Available bundles:", available_bundles)
            missing_bundles = [name for name in DATA_BUNDLES if name not in catalog.get("bundles", {})]
            if missing_bundles:
                raise KeyError(f"Unknown DATA_BUNDLES={missing_bundles}; available={available_bundles}")

            selected_names = sorted({
                artifact_name
                for bundle in DATA_BUNDLES
                for artifact_name in catalog["bundles"][bundle]
            })
            artifacts_by_name = {row["name"]: row for row in catalog.get("artifacts", [])}
            selected_artifacts = [artifacts_by_name[name] for name in selected_names]

            if VERIFY_FULL_SHA256:
                print("Full SHA-256 verification of the complete catalog...")
                verify_catalog(DATA_MANIFEST)

            selected_bytes = sum(int(row["size_bytes"]) for row in selected_artifacts if row["role"] != "metadata_json")
            if CACHE_DATA_LOCALLY and selected_bytes > shutil.disk_usage("/content").free:
                raise RuntimeError("Selected bundles do not fit on the current Colab local disk.")

            staged = {}
            for row in selected_artifacts:
                source = DRIVE_DATA_ROOT / row["relative_path"]
                if not source.is_file():
                    raise FileNotFoundError(source)
                if source.stat().st_size != int(row["size_bytes"]):
                    raise ValueError(f"Byte-size mismatch: {source}")
                destination = LOCAL_DATA_DIR / source.name
                if CACHE_DATA_LOCALLY:
                    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                        print("Copying to local SSD:", source.name)
                        shutil.copy2(source, destination)
                else:
                    destination = source
                staged[row["name"]] = destination
                if row["role"] == "master_npz":
                    inspect_stored_npz(destination)

            # Runner 要求 metadata 与 NPZ 同 stem；若 catalog 保存的是非规范文件名，建立本地规范别名。
            for dataset_id, dataset in catalog.get("datasets", {}).items():
                names = set(dataset.get("artifact_names", []))
                if not names.intersection(selected_names):
                    continue
                master_name = f"{dataset_id}:master"
                metadata_name = f"{dataset_id}:metadata"
                if master_name in staged and metadata_name in staged:
                    canonical_json = LOCAL_DATA_DIR / (staged[master_name].stem + ".json")
                    if not canonical_json.exists():
                        shutil.copy2(staged[metadata_name], canonical_json)

            REPO_PROCESSED = LOCAL_REPO / "efgp_eigenpro_py/gpu/benchmark_dataset/processed"
            REPO_PROCESSED.mkdir(parents=True, exist_ok=True)
            for local_file in LOCAL_DATA_DIR.iterdir():
                if local_file.suffix.lower() not in {".npz", ".json"}:
                    continue
                link = REPO_PROCESSED / local_file.name
                if not link.exists():
                    link.symlink_to(local_file)

            os.environ["BTAB_PROCESSED_DIR"] = str(LOCAL_DATA_DIR)
            display(pd.DataFrame([
                {"artifact": row["name"], "role": row["role"], "GiB": int(row["size_bytes"])/2**30}
                for row in selected_artifacts
            ]))
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## 5. 可选：补建缺失的 archived Synthetic / Manitowoc master

            archived Synthetic 必须是 `_ntrainN`、noise=0.3、train/test seeds 20260421/1、generation chunk=5M。现有 `_nN` 300M master 是 noise=.02 development artifact，不能重命名代替。

            Manitowoc 300M 必须保持冻结 AOI、hash split、浅层优先顺序。LOD 8 只是起点；若精确容量不足 300M/75M，应提高到 LOD 9，不能用密度估算替代扫描结果。
            """
        )
    )
    cells.append(
        _code(
            r"""
            generated_before = {path.resolve() for path in LOCAL_DATA_DIR.iterdir()}
            if GENERATE_ARCHIVED_SYNTHETIC_SIZES:
                sizes_arg = ",".join(str(int(n)) for n in GENERATE_ARCHIVED_SYNTHETIC_SIZES)
                run_cmd([
                    sys.executable, "-m",
                    "efgp_eigenpro_py.gpu.benchmark_dataset.preprocess_synthetic_true_func_2d_size_sweep",
                    "--n-train-list", sizes_arg,
                    "--noise", "0.3",
                    "--seed-train", "20260421",
                    "--seed-test", "1",
                    "--chunk-rows", "5000000",
                    "--size-token", "ntrain",
                    "--output-dir", str(LOCAL_DATA_DIR),
                ], cwd=LOCAL_REPO)

            if GENERATE_MANITOWOC_300M:
                run_cmd([
                    sys.executable, "-m",
                    "efgp_eigenpro_py.gpu.benchmark_dataset.preprocess_usgs_ept_wi_2county",
                    "--ept-json", "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/WI_2County_1_B23/ept.json",
                    "--n-train-list", "300000000", "--allow-large-output",
                    "--aoi-center-x", "437100", "--aoi-center-y", "4884400", "--aoi-side-m", "52000",
                    "--aoi-crs", "EPSG:6345", "--max-lod-depth", str(MANITOWOC_START_LOD),
                    "--source-project", "WI_2County_1_B23",
                    "--official-mean-ground-density", "11.37",
                    "--official-work-unit-area-km2", "1660.1823787253759",
                    "--cache-dir", str(DRIVE_PROJECT_ROOT / "ept_cache/WI_2County_1_B23"),
                    "--temporary-dir", str(LOCAL_DATA_DIR / "_manitowoc_300m_work"),
                    "--output-dir", str(LOCAL_DATA_DIR),
                    "--dataset-stem-prefix", "USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation",
                ], cwd=LOCAL_REPO)
                manitowoc_master = LOCAL_DATA_DIR / "USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n300000000.npz"
                manitowoc_frozen_10m = LOCAL_DATA_DIR / "USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000.npz"
                if not manitowoc_master.is_file():
                    raise RuntimeError("EPT build did not produce the requested 300M master.")
                prefix_report = compare_nested_prefix(
                    larger_npz=manitowoc_master,
                    prefix_npz=manitowoc_frozen_10m,
                    chunk_rows=1_000_000,
                )
                print("Frozen Manitowoc 10M prefix verified:", prefix_report)

            # Make newly generated artifacts visible to the legacy hard-coded processed path.
            for generated_file in LOCAL_DATA_DIR.iterdir():
                if generated_file.suffix.lower() not in {".npz", ".json"}:
                    continue
                link = REPO_PROCESSED / generated_file.name
                if not link.exists():
                    link.symlink_to(generated_file)

            if SYNC_GENERATED_DATA_TO_DRIVE:
                generated_drive_dir = DRIVE_DATA_ROOT / "generated_pending_catalog"
                generated_drive_dir.mkdir(parents=True, exist_ok=True)
                generated_now = [
                    path for path in LOCAL_DATA_DIR.iterdir()
                    if path.resolve() not in generated_before
                ]
                for generated_file in generated_now:
                    if generated_file.suffix.lower() not in {".npz", ".json"}:
                        continue
                    target = generated_drive_dir / generated_file.name
                    if not target.exists() or target.stat().st_size != generated_file.stat().st_size:
                        print("Syncing generated artifact to Drive:", generated_file.name)
                        shutil.copy2(generated_file, target)
                print("Register verified generated files with colab_drive_pack before formal runs.")
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## 6. 小规模 GPU plumbing smoke

            这不是论文结果。它只验证 cuFINUFFT、GPU eigensolver、固定系统哈希和输出写入；controlled protocol 仍保持 1 warm-up + 5 measured repeats。
            """
        )
    )
    cells.append(
        _code(
            r"""
            if RUN_PLUMBING_SMOKE:
                smoke_stem = "USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000"
                smoke_out = DRIVE_RUN_ROOT / "smoke_cufinufft"
                run_cmd([
                    sys.executable, "-m",
                    "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark",
                    "--dataset-stem", smoke_stem, "--dataset-dir", str(LOCAL_DATA_DIR),
                    "--n-train", "5000", "--subset-mode", "prefix",
                    "--kernel", "matern", "--lengthscale", "0.1", "--nu", "1.5",
                    "--lambda", "0.1", "--fourier-eps", "1e-3", "--tol", "1e-7",
                    "--maxiter", "6000", "--methods", "cg,default",
                    "--box-budget", "1024", "--rank", "32",
                    "--warmup-repeats", "1", "--measured-repeats", "5",
                    "--nufft-backend", "cufinufft", "--strict-gpu-eig",
                    "--output-dir", str(smoke_out),
                ], cwd=LOCAL_REPO)
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            # A. 原有 archived complete-pipeline 实验

            这里复用参考 notebook 的 direct-CG / binned-C1 precompute policy。`group_a/b/c` 是 exploratory candidate scans：不同 setup route、单次配置，且并不保证所有方法来自完全相同的 (A,b)。结果必须标记 `archived_complete_pipeline`，不能与 controlled paired speedup 排在同一列。

            Legacy runner 没有 case 内 resume。本 notebook 将每个 group 放在独立输出目录；只有最终原子写入 `_SUCCESS.json` 的 group 才会跳过，中断时即使已有部分 CSV 也会重跑该 group。
            """
        )
    )
    cells.append(
        _code(
            r"""
            from dataclasses import replace
            import contextlib, importlib
            from efgp_eigenpro_py.gpu.backends import BackendConfig
            from efgp_eigenpro_py.gpu.box_toeplitz_active_block.config import BTABExperimentConfig

            PACKAGE = "efgp_eigenpro_py.gpu.box_toeplitz_active_block"
            MODULE_NAMES = [
                "efgp_eigenpro_py.gpu.versions",
                f"{PACKAGE}.config",
                f"{PACKAGE}.run_experiments",
            ]

            def reload_legacy_modules():
                modules = {}
                for name in MODULE_NAMES:
                    modules[name] = importlib.reload(sys.modules[name]) if name in sys.modules else importlib.import_module(name)
                return modules

            def validate_archived_synthetic_inputs(required_sizes):
                failures = []
                for n_train in sorted({int(value) for value in required_sizes}):
                    stem = f"synthetic_true_func_2d_ntrain{n_train}"
                    npz_path = LOCAL_DATA_DIR / f"{stem}.npz"
                    json_path = LOCAL_DATA_DIR / f"{stem}.json"
                    if not npz_path.is_file() or not json_path.is_file():
                        failures.append(f"{stem}: missing NPZ/JSON")
                        continue
                    metadata = json.loads(json_path.read_text(encoding="utf-8"))
                    generation = metadata.get("generation", {})
                    expected = {
                        "noise_std": 0.3,
                        "seed_train": 20260421,
                        "seed_test": 1,
                        "chunk_rows": 5_000_000,
                    }
                    mismatches = {
                        key: (generation.get(key), value)
                        for key, value in expected.items()
                        if generation.get(key) != value
                    }
                    if mismatches:
                        failures.append(f"{stem}: archived generation mismatch {mismatches}")
                if failures:
                    raise RuntimeError(
                        "Archived Synthetic inputs are incomplete or incompatible. "
                        "Use GENERATE_ARCHIVED_SYNTHETIC_SIZES first:\n- "
                        + "\n- ".join(failures)
                    )
            """
        )
    )
    cells.append(_code(legacy_policy))
    cells.append(
        _code(
            r"""
            LEGACY_OUTPUT_ROOT = DRIVE_RUN_ROOT / "legacy_archived_pipeline"
            LEGACY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            legacy_group_outputs = []

            legacy_sizes = {
                "group_a": {1_000_000, 3_000_000, 10_000_000, 30_000_000},
                "group_b": {1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000, 300_000_000},
                "group_c": {100_000_000, 300_000_000},
            }
            requested_legacy_sizes = set().union(*(
                legacy_sizes.get(group, set()) for group in RUN_LEGACY_GROUPS
            )) if RUN_LEGACY_GROUPS else set()
            if requested_legacy_sizes:
                validate_archived_synthetic_inputs(requested_legacy_sizes)

            for group_name in RUN_LEGACY_GROUPS:
                if group_name not in {"group_a", "group_b", "group_c"}:
                    raise ValueError(f"Unknown legacy group: {group_name}")
                group_out = LEGACY_OUTPUT_ROOT / group_name
                success_marker = group_out / "_SUCCESS.json"
                if success_marker.is_file():
                    marker = json.loads(success_marker.read_text(encoding="utf-8"))
                    current_data_manifest_sha = hashlib.sha256(DATA_MANIFEST.read_bytes()).hexdigest()
                    if marker.get("git_sha") != GIT_SHA or marker.get("data_manifest_sha256") != current_data_manifest_sha:
                        raise RuntimeError(
                            f"{success_marker} belongs to different code/data; change RUN_TAG instead of mixing runs."
                        )
                    print("Legacy group already complete; skipping:", group_name)
                    legacy_group_outputs.append(group_out)
                    continue

                mods = reload_legacy_modules()
                install_notebook_precompute_policy(mods)
                run_experiments = mods[f"{PACKAGE}.run_experiments"].run_experiments
                cfg = BTABExperimentConfig(
                    btab_experiment_route=group_name,
                    btab_experiment_routes=[],
                    output_dir=str(group_out),
                    run_tag=f"colab_{group_name}",
                    tol=1e-7,
                    maxiter=80_000,
                    non_v1_maxiter=3_000,
                    backend=BackendConfig(xp="cupy", fft="cupy", nufft="cufinufft", linalg="cupy"),
                )
                result = run_experiments(cfg)
                completed_output = Path(result["output_dir"])
                required_outputs = [
                    completed_output / "master_summary.csv",
                    completed_output / "aggregate_summary.csv",
                    completed_output / "experiment_config.json",
                ]
                missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
                if missing_outputs:
                    raise RuntimeError(f"Legacy group finished without required outputs: {missing_outputs}")
                marker_payload = {
                    "protocol_family": "archived_complete_pipeline",
                    "group": group_name,
                    "git_sha": GIT_SHA,
                    "data_manifest_sha256": hashlib.sha256(DATA_MANIFEST.read_bytes()).hexdigest(),
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "required_outputs": [path.name for path in required_outputs],
                }
                marker_tmp = success_marker.with_suffix(".json.partial")
                marker_tmp.write_text(json.dumps(marker_payload, indent=2), encoding="utf-8")
                marker_tmp.replace(success_marker)
                legacy_group_outputs.append(completed_output)
            print("Legacy outputs:", legacy_group_outputs)
            """
        )
    )
    cells.append(
        _code(
            r"""
            legacy_frames = []
            for output in legacy_group_outputs:
                for candidate in [output / "master_summary.csv", output / "aggregate_summary.csv"]:
                    if candidate.is_file():
                        frame = pd.read_csv(candidate)
                        frame["protocol_family"] = "archived_complete_pipeline"
                        frame["evidence_role"] = "exploratory_scale_map"
                        frame["timing_scope"] = "complete pipeline; direct CG and binned candidates use different setup routes"
                        frame["legacy_group"] = output.name
                        legacy_frames.append(frame)
                        break
            legacy_all = pd.concat(legacy_frames, ignore_index=True, sort=False) if legacy_frames else pd.DataFrame()
            if not legacy_all.empty:
                legacy_all.to_csv(LEGACY_OUTPUT_ROOT / "legacy_all_groups.csv", index=False)
                display(legacy_all.head(50))
            else:
                print("No legacy groups selected or completed.")
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            # B. 新增 controlled fixed-system 实验

            每个 case 在单一 invocation 中构造一个不可变系统，所有方法共享系统哈希；每次从零初值开始，1 次预热后做 5 次随机顺序配对。Scale master 使用 `subset_mode='prefix'`：10M/30M/100M/300M 是同一 300M master 的严格行前缀。

            - `screen_10m`：CG-only difficulty gate，与正式表分开。
            - `paper_10m`：archived Synthetic、Winnebago、Manitowoc 三个 10M q256 center。
            - `scale_archived_exact`：原数据定义的 10M/30M/100M/300M exact artifacts；不同 N 不宣称嵌套。
            - `scale_development_masters`：low-noise Synthetic 与 Winnebago raw-prefix 的新规模实验，不冒充 archived artifacts。
            - `scale_manitowoc_master`：需先准备 Manitowoc 300M master。
            - `winnebago_oat_n10m`：只改变一个 λ 或 ℓ，并配对比较 CG/default/full-eig。
            """
        )
    )
    cells.append(
        _code(
            r"""
            import copy
            CONTROLLED_DIR = LOCAL_REPO / "efgp_eigenpro_py/gpu/box_toeplitz_active_block/controlled"
            SUITE_TEMPLATE = CONTROLLED_DIR / "colab_all_experiments_suite.json"
            CONTROLLED_OUTPUT_ROOT = DRIVE_RUN_ROOT / "controlled_fixed_system"
            RUNTIME_CONFIG_ROOT = DRIVE_RUN_ROOT / "runtime_configs"
            CONTROLLED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUNTIME_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
            expected_controlled_case_count = 0

            selected_profiles = []
            if RUN_CG_SCREEN_10M: selected_profiles.append("screen_10m")
            if RUN_Q256_CENTER_10M: selected_profiles.append("paper_10m")
            if RUN_ARCHIVED_EXACT_SCALE: selected_profiles.append("scale_archived_exact")
            if RUN_DEVELOPMENT_MASTER_SCALE: selected_profiles.append("scale_development_masters")
            if RUN_MANITOWOC_SCALE: selected_profiles.append("scale_manitowoc_master")
            if RUN_WINNEBAGO_OAT_10M: selected_profiles.append("winnebago_oat_n10m")

            base_suite = json.loads(SUITE_TEMPLATE.read_text(encoding="utf-8"))
            controlled_profile_outputs = []
            for profile_name in selected_profiles:
                profile = copy.deepcopy(base_suite["profiles"][profile_name])
                cases = [
                    case for case in profile["cases"]
                    if int(case["expected_n_train"]) in {int(n) for n in ACTIVE_SIZES}
                ]
                if not cases:
                    raise RuntimeError(
                        f"{profile_name} has no case matching ACTIVE_SIZES={ACTIVE_SIZES}; "
                        "refusing to mark an empty selected workload complete."
                    )
                if profile_name == "scale_archived_exact":
                    validate_archived_synthetic_inputs(
                        case["expected_n_train"] for case in cases
                        if case["id"].startswith("synthetic_")
                    )
                profile["cases"] = cases
                expected_controlled_case_count += len(cases)
                runtime_suite = {
                    "base": copy.deepcopy(base_suite["base"]),
                    "profiles": {profile_name: profile},
                }
                runtime_path = RUNTIME_CONFIG_ROOT / f"{profile_name}.json"
                runtime_path.write_text(json.dumps(runtime_suite, indent=2), encoding="utf-8")
                output_root = CONTROLLED_OUTPUT_ROOT / profile_name
                run_cmd([
                    sys.executable, "-m",
                    "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.suite",
                    "--config", str(runtime_path), "--profile", profile_name,
                    "--dataset-dir", str(LOCAL_DATA_DIR),
                    "--output-root", str(output_root),
                    "--nufft-backend", "cufinufft", "--strict-gpu-eig",
                    "--execute", "--resume",
                ], cwd=LOCAL_REPO)
                controlled_profile_outputs.append(output_root)
            print("Controlled profile outputs:", controlled_profile_outputs)
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## B.1 q128 bridge 与 SE full-inverse control（可选）

            两个历史 suite 的 Synthetic stem 默认指向 low-noise `_n10000000`。本格在运行时改成 archived `_ntrain10000000`，不修改原模板。两种 control 仍各自保持同一 case 内固定 (A,b)。
            """
        )
    )
    cells.append(
        _code(
            r"""
            extra_suites = []
            if RUN_Q128_BRIDGE:
                extra_suites.append(("q128_bridge", "original_data_n10m_matched_bridge_suite.json", "bridge"))
            if RUN_SE_FULL_INVERSE_CONTROL:
                extra_suites.append(("se_full_inverse", "original_data_n10m_se_full_inverse_suite.json", "se_control"))

            for label, filename, profile_name in extra_suites:
                payload = json.loads((CONTROLLED_DIR / filename).read_text(encoding="utf-8"))
                for profile in payload["profiles"].values():
                    for case in profile["cases"]:
                        if case.get("dataset_stem") == "synthetic_true_func_2d_n10000000":
                            case["dataset_stem"] = "synthetic_true_func_2d_ntrain10000000"
                runtime_path = RUNTIME_CONFIG_ROOT / f"{label}.json"
                runtime_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                output_root = CONTROLLED_OUTPUT_ROOT / label
                expected_controlled_case_count += len(payload["profiles"][profile_name]["cases"])
                run_cmd([
                    sys.executable, "-m",
                    "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.suite",
                    "--config", str(runtime_path), "--profile", profile_name,
                    "--dataset-dir", str(LOCAL_DATA_DIR), "--output-root", str(output_root),
                    "--nufft-backend", "cufinufft", "--strict-gpu-eig",
                    "--execute", "--resume",
                ], cwd=LOCAL_REPO)
                controlled_profile_outputs.append(output_root)
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## B.2 正式 controlled artifact 审计

            cuFINUFFT adapter 可能在运行期失败后回退 CPU，因此不能只检查请求参数；必须同时检查 manifest 的 `nufft_backend_resolved` 和 `nufft_stage`。下格也检查 fp64、system unchanged、strict GPU eig 和完整 `run_complete.json`。
            """
        )
    )
    cells.append(
        _code(
            r"""
            controlled_manifests = sorted(CONTROLLED_OUTPUT_ROOT.rglob("system_manifest.json"))
            audit_rows = []
            for manifest_path in controlled_manifests:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                config_path = manifest_path.with_name("experiment_config.json")
                complete_path = manifest_path.with_name("run_complete.json")
                summary_path = manifest_path.with_name("matched_summary.csv")
                config = json.loads(config_path.read_text(encoding="utf-8"))
                problems = []
                if not manifest.get("system_unchanged"): problems.append("system changed")
                if manifest.get("nufft_backend_resolved") != "cufinufft": problems.append("backend fallback")
                if manifest.get("nufft_stage") != "cufinufft": problems.append("NUFFT stage is not GPU")
                if manifest.get("precision_mode") != "fp64": problems.append("not fp64")
                if not config.get("strict_gpu_eig"): problems.append("strict_gpu_eig false")
                if not complete_path.is_file(): problems.append("missing run_complete.json")
                if not summary_path.is_file():
                    problems.append("missing matched_summary.csv")
                else:
                    matched = pd.read_csv(summary_path)
                    required_columns = {
                        "method", "measured_repeats", "converged_repeats",
                        "performance_claim_eligible", "true_relres_max",
                    }
                    if not required_columns.issubset(matched.columns):
                        problems.append("matched summary lacks eligibility columns")
                    else:
                        eligible = matched["performance_claim_eligible"].astype(str).str.lower().eq("true")
                        expected_repeats = int(config.get("measured_repeats", 5))
                        five_of_five = (
                            matched["measured_repeats"].astype(int).eq(expected_repeats)
                            & matched["converged_repeats"].astype(int).eq(expected_repeats)
                        )
                        if not bool((eligible & five_of_five).all()):
                            bad_methods = matched.loc[~(eligible & five_of_five), "method"].astype(str).tolist()
                            problems.append(f"ineligible/nonconverged methods: {bad_methods}")
                audit_rows.append({
                    "case": manifest_path.parent.name,
                    "N": manifest.get("n_train"),
                    "system_id": manifest.get("system_id"),
                    "status": "PASS" if not problems else "FAIL: " + "; ".join(problems),
                })
            controlled_artifact_audit = pd.DataFrame(audit_rows)
            display(controlled_artifact_audit)
            if expected_controlled_case_count and len(controlled_artifact_audit) < expected_controlled_case_count:
                raise RuntimeError(
                    f"Expected at least {expected_controlled_case_count} controlled cases, "
                    f"but found {len(controlled_artifact_audit)} manifests."
                )
            if not controlled_artifact_audit.empty and not controlled_artifact_audit["status"].eq("PASS").all():
                raise RuntimeError("Controlled artifact audit failed; do not use these timings in the paper.")
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            # C. Prediction-equivalence audit（单独、非计时）

            Audit 会从每个 controlled config 重新构造自己的 fixed system，求解后按 GPU chunk 预测。其 solve/prediction 时间明确排除在 speedup claim 外。对于 master-prefix protocol，正式逻辑测试集为对应训练规模的前 (N/4) 行。
            """
        )
    )
    cells.append(
        _code(
            r"""
            prediction_outputs = []
            if RUN_PREDICTION_AUDIT:
                for config_path in sorted(CONTROLLED_OUTPUT_ROOT.rglob("experiment_config.json")):
                    manifest_path = config_path.with_name("system_manifest.json")
                    if not manifest_path.is_file():
                        continue
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    n_train = int(manifest["n_train"])
                    if n_train > int(PREDICTION_AUDIT_MAX_TRAIN_N):
                        continue
                    audit_out = config_path.parent / "prediction_audit"
                    if (audit_out / "prediction_audit.json").is_file():
                        print("Prediction audit already exists; skipping", config_path.parent.name)
                        prediction_outputs.append(audit_out)
                        continue
                    run_cmd([
                        sys.executable, "-m",
                        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.prediction_audit",
                        "--config", str(config_path),
                        "--prediction-chunk-size", "100000",
                        "--warmup-solves", "1",
                        "--max-test", str(n_train // 4),
                        "--output-dir", str(audit_out),
                    ], cwd=LOCAL_REPO)
                    prediction_outputs.append(audit_out)
            print("Prediction audit outputs:", prediction_outputs)
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            # D. 统一结果索引与图

            统一索引只负责查找和展示，不跨 protocol 计算 speedup。Controlled 表的 `cold_speedup_median` 是 selection/build+solve、排除公共 Fourier setup；`shared_fourier_setup_plus_method_speedup_median` 才是把公共 setup 加回两边后的比值。
            """
        )
    )
    cells.append(
        _code(
            r"""
            result_frames = []
            for summary_path in sorted(CONTROLLED_OUTPUT_ROOT.rglob("matched_summary.csv")):
                frame = pd.read_csv(summary_path)
                manifest = json.loads(summary_path.with_name("system_manifest.json").read_text(encoding="utf-8"))
                frame["protocol_family"] = "controlled_fixed_system"
                frame["evidence_role"] = "paired_scale_or_replication"
                frame["timing_scope"] = "selection/build + solve; shared Fourier setup excluded from cold columns"
                frame["case"] = summary_path.parent.name
                frame["dataset_stem"] = manifest.get("dataset_stem")
                frame["N"] = manifest.get("n_train")
                frame["system_id"] = manifest.get("system_id")
                result_frames.append(frame)

            for audit_path in sorted(CONTROLLED_OUTPUT_ROOT.rglob("prediction_audit.csv")):
                frame = pd.read_csv(audit_path)
                frame["protocol_family"] = "prediction_audit"
                frame["evidence_role"] = "accuracy_only"
                frame["timing_scope"] = "excluded from all speed claims"
                frame["case"] = audit_path.parent.parent.name
                result_frames.append(frame)

            if not legacy_all.empty:
                result_frames.append(legacy_all)
            all_experiments = pd.concat(result_frames, ignore_index=True, sort=False) if result_frames else pd.DataFrame()
            INDEX_PATH = DRIVE_RUN_ROOT / "all_experiments_index.csv"
            all_experiments.to_csv(INDEX_PATH, index=False)
            print("Unified index:", INDEX_PATH)
            display(all_experiments.head(100))
            """
        )
    )
    cells.append(
        _code(
            r"""
            import matplotlib.pyplot as plt
            controlled_plot = all_experiments[
                all_experiments.get("protocol_family", pd.Series(dtype=str)).eq("controlled_fixed_system")
            ].copy() if not all_experiments.empty else pd.DataFrame()
            if not controlled_plot.empty and {"N", "cold_speedup_median", "method"}.issubset(controlled_plot.columns):
                fig, ax = plt.subplots(figsize=(9, 5))
                for (dataset, method), group in controlled_plot.dropna(subset=["cold_speedup_median"]).groupby(["dataset_stem", "method"]):
                    group = group.sort_values("N")
                    ax.plot(group["N"], group["cold_speedup_median"], marker="o", label=f"{dataset} | {method}")
                ax.set_xscale("log")
                ax.axhline(1.0, color="black", lw=1)
                ax.set_xlabel("training rows N")
                ax.set_ylabel("paired cold speedup over CG")
                ax.grid(True, alpha=.25)
                ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
                fig.tight_layout()
                plot_path = DRIVE_RUN_ROOT / "controlled_scale_speedup.png"
                fig.savefig(plot_path, dpi=180, bbox_inches="tight")
                plt.show()
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            # E. 最终 checkpoint、Drive 校验与可选断开

            结果目录只包含配置、manifest、CSV/JSON 和图，不复制大数据。自动断开默认关闭；只有下格确认所有期望 case 均有 `run_complete.json` 且统一索引已写入 Drive 后才允许开启。
            """
        )
    )
    cells.append(
        _code(
            r"""
            final_manifest = {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_sha": GIT_SHA,
                "runtime": runtime_info,
                "data_manifest": str(DATA_MANIFEST),
                "data_manifest_sha256": hashlib.sha256(DATA_MANIFEST.read_bytes()).hexdigest(),
                "selected_data_bundles": DATA_BUNDLES,
                "run_tag": RUN_TAG,
                "active_sizes": [int(n) for n in ACTIVE_SIZES],
                "legacy_groups": list(RUN_LEGACY_GROUPS),
                "controlled_profiles": selected_profiles,
                "controlled_case_count": int(len(controlled_artifact_audit)),
                "all_controlled_artifacts_pass": (
                    None if expected_controlled_case_count == 0 else bool(
                        len(controlled_artifact_audit) >= expected_controlled_case_count
                        and controlled_artifact_audit["status"].eq("PASS").all()
                    )
                ),
                "unified_index": str(INDEX_PATH),
            }
            legacy_complete = all(
                (DRIVE_RUN_ROOT / "legacy_archived_pipeline" / group / "_SUCCESS.json").is_file()
                for group in RUN_LEGACY_GROUPS
            )
            controlled_complete = (
                expected_controlled_case_count == 0
                or (
                    len(controlled_artifact_audit) >= expected_controlled_case_count
                    and controlled_artifact_audit["status"].eq("PASS").all()
                )
            )
            prediction_complete = (not RUN_PREDICTION_AUDIT) or bool(prediction_outputs)
            workload_requested = bool(
                RUN_LEGACY_GROUPS or selected_profiles or extra_suites or RUN_PREDICTION_AUDIT
            )
            run_verified = bool(
                workload_requested and legacy_complete and controlled_complete
                and prediction_complete and INDEX_PATH.is_file()
            )
            final_manifest["run_verified"] = run_verified
            FINAL_MANIFEST_PATH = DRIVE_RUN_ROOT / "colab_run_manifest.json"
            FINAL_MANIFEST_PATH.write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
            print(json.dumps(final_manifest, indent=2))

            if DISCONNECT_RUNTIME_WHEN_VERIFIED:
                if not run_verified or not FINAL_MANIFEST_PATH.is_file():
                    raise RuntimeError("Selected workload is not fully verified; refusing to disconnect.")
                if IS_COLAB:
                    from google.colab import runtime
                    runtime.unassign()
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## 运行后检查清单

            1. 每个 controlled case 的 `system_manifest.json`：`system_unchanged=true`、`nufft_stage=cufinufft`。
            2. 每个正式方法：5/5 repeats 满足 independently recomputed residual (<10^{-7})。
            3. 300M 若只跑 core methods，表中明确列出方法集合；不要与另一次 invocation 的随机方法拼成 paired table。
            4. Legacy、controlled、prediction audit 保持不同 `protocol_family`。
            5. Synthetic 表中明确区分 archived noise=.3 `_ntrainN` 与 development noise=.02 `_nN`。
            6. 记录 Colab GPU 型号、Git SHA、数据 SHA、实际 (M)、box size/rank 和完整 timing scope。
            """
        )
    )

    for index, cell in enumerate(cells):
        cell["id"] = f"efgp-{index:03d}"
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_NOTEBOOK} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
