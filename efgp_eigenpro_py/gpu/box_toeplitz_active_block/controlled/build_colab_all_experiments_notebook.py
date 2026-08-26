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

            使用方式：选择 Colab 的 **A100 GPU + High-RAM** runtime，然后点击 **运行时 → 全部运行**。默认的 `RUN_ALL_FORMAL_EXPERIMENTS=True` 会自动完成 smoke、10M 主实验、prediction audit、OAT、box-budget 消融、Synthetic nested-prefix 10M–300M，以及 Winnebago archived-exact 10M–300M。每个规模是独立 job；失败会结构化记录并隔离，不会用 `CalledProcessError` 阻断后续实验。

            参考并保留了 `boxeig_inverse_diagnostics_experiment.ipynb` 中最关键的 direct/binned precompute policy；统一结果区重新整理 legacy、controlled 和 audit 三种 schema。
            """
        )
    )
    cells.append(
        _code(
            r"""
            # ==================== 一键正式实验：通常无需修改 ====================
            from pathlib import Path

            REPO_URL = "https://github.com/Yifiwifi/EFGP-Eigenpro.git"
            REPO_REF = "codex/colab-all-experiments"
            LOCAL_REPO = Path("/content/EFGP-Eigenpro")

            DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/EFGP_Colab")
            DRIVE_DATA_ROOT = DRIVE_PROJECT_ROOT / "data_bundle"
            RUN_TAG_PREFIX = "paper_one_click"
            # checkout 后自动使用 paper_one_click_<git sha>，同一代码自动 resume，
            # 新代码自动进入新目录，不会覆盖旧证据。
            RUN_TAG = None
            DRIVE_RUN_ROOT = None
            LOCAL_DATA_DIR = Path("/content/efgp_data")

            RUN_ALL_FORMAL_EXPERIMENTS = True
            FORMAL_SCALE_SIZES = [10_000_000, 30_000_000, 100_000_000, 300_000_000]

            # 从 drive_manifest.json 自动选择正式 campaign 所需数据。
            DATA_BUNDLES = []
            CACHE_DATA_LOCALLY = True       # 正式计时推荐 True，避免 Drive FUSE 进入 timing
            VERIFY_FULL_SHA256 = False      # 完整 catalog 仅在上传变更后单独校验一次

            # 一键正式 campaign 不重跑不可直接混表的 legacy exploratory groups。
            RUN_LEGACY_GROUPS = []

            # 下列变量由一键模式统一设定；False 时仍可作为 advanced/manual 模式使用。
            RUN_PLUMBING_SMOKE = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_CG_SCREEN_10M = False
            RUN_Q256_CENTER_10M = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_BOX_BUDGET_ABLATION = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_ARCHIVED_EXACT_SCALE = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_DEVELOPMENT_MASTER_SCALE = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_MANITOWOC_SCALE = False
            RUN_WINNEBAGO_OAT_10M = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_Q128_BRIDGE = False
            RUN_SE_FULL_INVERSE_CONTROL = False
            RUN_PREDICTION_AUDIT = RUN_ALL_FORMAL_EXPERIMENTS

            ACTIVE_SIZES = list(FORMAL_SCALE_SIZES) if RUN_ALL_FORMAL_EXPERIMENTS else [10_000_000]
            ALLOW_100M = RUN_ALL_FORMAL_EXPERIMENTS
            ALLOW_300M = RUN_ALL_FORMAL_EXPERIMENTS
            PREDICTION_AUDIT_MAX_TRAIN_N = 10_000_000
            PREDICTION_AUDIT_PROFILES = ["paper_10m"]

            # 一键模式中的 profile 级数据族过滤。尤其禁止把已知失败的
            # Winnebago raw-prefix development cases 混入 Synthetic scale job。
            PROFILE_DATASET_FAMILIES = {
                "scale_development_masters": ["Synthetic"],
                "scale_archived_exact": ["Winnebago"],
            } if RUN_ALL_FORMAL_EXPERIMENTS else {}
            ACTIVE_CASE_IDS = []

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
                RUN_CG_SCREEN_10M, RUN_Q256_CENTER_10M, RUN_BOX_BUDGET_ABLATION,
                RUN_WINNEBAGO_OAT_10M,
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
            def run_cmd(args, *, cwd=None, env=None, check=True):
                args = [str(x) for x in args]
                print("+", " ".join(args))
                return subprocess.run(args, cwd=cwd, env=env, check=check)

            if not LOCAL_REPO.exists():
                run_cmd(["git", "clone", REPO_URL, str(LOCAL_REPO)])
            run_cmd(["git", "fetch", "--all", "--tags"], cwd=LOCAL_REPO)
            repo_ref_text = str(REPO_REF).strip()
            is_full_sha = len(repo_ref_text) == 40 and all(
                char in "0123456789abcdefABCDEF" for char in repo_ref_text
            )
            checkout_ref = repo_ref_text if is_full_sha else f"origin/{repo_ref_text}"
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
            RUN_TAG = f"{RUN_TAG_PREFIX}_{GIT_SHA[:12]}"
            DRIVE_RUN_ROOT = DRIVE_PROJECT_ROOT / "runs" / RUN_TAG
            DRIVE_RUN_ROOT.mkdir(parents=True, exist_ok=True)
            print("Pinned Git SHA:", GIT_SHA)
            print("Automatic run tag:", RUN_TAG)
            print("Run output root:", DRIVE_RUN_ROOT)
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

            CAN_RUN_300M = bool(
                gpu_total >= 30 * 2**30 and host_available >= 20 * 2**30
            )
            if MAX_REQUESTED_N >= 100_000_000 and not ALLOW_100M:
                raise RuntimeError("The selected workload reaches >=100M; set ALLOW_100M=True after reviewing memory.")
            if MAX_REQUESTED_N >= 300_000_000:
                if not ALLOW_300M:
                    raise RuntimeError("300M is gated; set ALLOW_300M=True explicitly.")
                if not CAN_RUN_300M and not RUN_ALL_FORMAL_EXPERIMENTS:
                    raise RuntimeError("300M requires >=30 GiB GPU and >=20 GiB currently available host RAM.")
                if not CAN_RUN_300M:
                    print(
                        "WARNING: 300M jobs will be marked SKIPPED_HARDWARE; "
                        "10M/30M/100M jobs will still run."
                    )
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
            SMOKE_OK = True
            SMOKE_RETURN_CODE = None
            if RUN_PLUMBING_SMOKE:
                smoke_stem = "USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000"
                smoke_out = DRIVE_RUN_ROOT / "smoke_cufinufft"
                smoke_result = run_cmd([
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
                ], cwd=LOCAL_REPO, check=False)
                SMOKE_RETURN_CODE = int(smoke_result.returncode)
                SMOKE_OK = SMOKE_RETURN_CODE == 0
                if not SMOKE_OK:
                    print(
                        f"SMOKE FAILED (return code {SMOKE_RETURN_CODE}). "
                        "Heavy formal jobs will be recorded as SKIPPED_SMOKE_FAILED."
                    )
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
            - `winnebago_box_budget_n10m`：同一 10M Winnebago 系统上的 4096/8192/16384 memory-budget 消融。
            - `scale_archived_exact`：一键模式只选择 Winnebago 原数据定义的 exact artifacts；不同 N 不宣称嵌套。
            - `scale_development_masters`：一键模式只选择 low-noise Synthetic 的同一 300M master 前缀；已知失败的 Winnebago raw-prefix 不进入正式规模任务。
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
            selected_case_records = []
            base_suite = json.loads(SUITE_TEMPLATE.read_text(encoding="utf-8"))
            controlled_profile_outputs = []

            def select_profile_cases(profile, *, sizes, families=(), case_ids=()):
                size_set = {int(n) for n in sizes}
                family_set = {str(name) for name in families}
                id_set = {str(case_id) for case_id in case_ids}
                selected = []
                for case in profile["cases"]:
                    if int(case["expected_n_train"]) not in size_set:
                        continue
                    if family_set and str(case.get("dataset_family", "")) not in family_set:
                        continue
                    if id_set and str(case["id"]) not in id_set:
                        continue
                    selected.append(case)
                return selected

            formal_jobs = []
            if RUN_ALL_FORMAL_EXPERIMENTS:
                formal_jobs.extend([
                    {"job_id": "paper_10m", "profile": "paper_10m", "sizes": [10_000_000], "mandatory": True},
                    {"job_id": "winnebago_oat_n10m", "profile": "winnebago_oat_n10m", "sizes": [10_000_000], "mandatory": True},
                    {"job_id": "winnebago_box_budget_n10m", "profile": "winnebago_box_budget_n10m", "sizes": [10_000_000], "mandatory": True},
                ])
                for family, profile_name, series in [
                    ("Synthetic", "scale_development_masters", "synthetic_nested"),
                    ("Winnebago", "scale_archived_exact", "winnebago_exact"),
                ]:
                    for n_train in FORMAL_SCALE_SIZES:
                        formal_jobs.append({
                            "job_id": f"{series}_{n_train // 1_000_000}m",
                            "profile": profile_name,
                            "sizes": [int(n_train)],
                            "families": [family],
                            "mandatory": True,
                            "gate_series": series,
                            "requires_gate_n": (
                                30_000_000 if n_train == 100_000_000
                                else 100_000_000 if n_train == 300_000_000
                                else None
                            ),
                        })
            else:
                manual_profiles = []
                if RUN_CG_SCREEN_10M: manual_profiles.append("screen_10m")
                if RUN_Q256_CENTER_10M: manual_profiles.append("paper_10m")
                if RUN_BOX_BUDGET_ABLATION: manual_profiles.append("winnebago_box_budget_n10m")
                if RUN_WINNEBAGO_OAT_10M: manual_profiles.append("winnebago_oat_n10m")
                if RUN_DEVELOPMENT_MASTER_SCALE: manual_profiles.append("scale_development_masters")
                if RUN_ARCHIVED_EXACT_SCALE: manual_profiles.append("scale_archived_exact")
                if RUN_MANITOWOC_SCALE: manual_profiles.append("scale_manitowoc_master")
                formal_jobs.extend({
                    "job_id": profile_name,
                    "profile": profile_name,
                    "sizes": list(ACTIVE_SIZES),
                    "families": list(PROFILE_DATASET_FAMILIES.get(profile_name, [])),
                    "case_ids": list(ACTIVE_CASE_IDS),
                    "mandatory": True,
                } for profile_name in manual_profiles)

            selected_profiles = list(dict.fromkeys(job["profile"] for job in formal_jobs))
            campaign_job_rows = []
            if RUN_PLUMBING_SMOKE:
                campaign_job_rows.append({
                    "job_id": "plumbing_smoke",
                    "profile": "benchmark_smoke",
                    "dataset_family": "Manitowoc",
                    "n_train": 5000,
                    "mandatory": True,
                    "return_code": SMOKE_RETURN_CODE,
                    "status": "PASS" if SMOKE_OK else "EXECUTION_ERROR",
                    "reason": "" if SMOKE_OK else "smoke command returned non-zero",
                    "case_count": 1,
                    "elapsed_seconds": None,
                })
            CAMPAIGN_JOBS_CSV = DRIVE_RUN_ROOT / "campaign_jobs.csv"
            CAMPAIGN_JOBS_JSON = DRIVE_RUN_ROOT / "campaign_jobs.json"

            def write_campaign_checkpoint():
                payload = json.dumps(campaign_job_rows, indent=2)
                partial_json = CAMPAIGN_JOBS_JSON.with_suffix(".json.partial")
                partial_json.write_text(payload, encoding="utf-8")
                partial_json.replace(CAMPAIGN_JOBS_JSON)
                frame = pd.DataFrame(campaign_job_rows)
                partial_csv = CAMPAIGN_JOBS_CSV.with_suffix(".csv.partial")
                frame.to_csv(partial_csv, index=False)
                partial_csv.replace(CAMPAIGN_JOBS_CSV)

            def classify_suite_invocation(return_code, status_rows, expected_case_count):
                terminal = [str(row.get("status", "")) for row in status_rows]
                complete_status = (
                    len(status_rows) == int(expected_case_count)
                    and all(status not in {"", "running"} for status in terminal)
                )
                hard_error = any(status == "error" for status in terminal)
                scientific_failure = any(
                    row.get("ineligible_methods") or row.get("diagnostic_errors")
                    for row in status_rows
                )
                if not complete_status or hard_error:
                    return "EXECUTION_ERROR", "missing/nonterminal case status or case-level error"
                if scientific_failure or int(return_code) == 2:
                    return "SCIENTIFIC_FAIL", "complete artifacts contain ineligible methods or diagnostic errors"
                if int(return_code) == 0:
                    return "PASS", ""
                # Backward compatibility with older suite.py, which returned 1 for a
                # complete scientific failure.
                if int(return_code) == 1 and scientific_failure:
                    return "SCIENTIFIC_FAIL", "legacy scientific-failure exit code"
                return "EXECUTION_ERROR", f"unexpected suite return code {return_code}"

            def scale_core_pass(case_records):
                required = {"cg", "default", "full-eig"}
                for record in case_records:
                    summary_path = Path(record["run_dir"]) / "matched_summary.csv"
                    config_path = Path(record["run_dir"]) / "experiment_config.json"
                    if not summary_path.is_file() or not config_path.is_file():
                        return False
                    summary = pd.read_csv(summary_path)
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    core = summary.loc[summary["method"].astype(str).isin(required)].copy()
                    if set(core["method"].astype(str)) != required:
                        return False
                    expected_repeats = int(config.get("measured_repeats", 5))
                    eligible = core["performance_claim_eligible"].astype(str).str.lower().eq("true")
                    converged = pd.to_numeric(core["converged_repeats"], errors="coerce").eq(expected_repeats)
                    residual_ok = pd.to_numeric(core["true_relres_max"], errors="coerce").le(
                        float(config.get("tol", 1e-7))
                    )
                    if not bool((eligible & converged & residual_ok).all()):
                        return False
                return True

            gate_results = {}
            for job in formal_jobs:
                job_id = str(job["job_id"])
                profile_name = str(job["profile"])
                sizes = [int(n) for n in job["sizes"]]
                families = list(job.get("families", PROFILE_DATASET_FAMILIES.get(profile_name, [])))
                started = time.perf_counter()
                base_row = {
                    "job_id": job_id,
                    "profile": profile_name,
                    "dataset_family": ",".join(families),
                    "n_train": ",".join(str(n) for n in sizes),
                    "mandatory": bool(job.get("mandatory", True)),
                }

                skip_reason = ""
                if not SMOKE_OK:
                    skip_reason = "SKIPPED_SMOKE_FAILED"
                elif 300_000_000 in sizes and not CAN_RUN_300M:
                    skip_reason = "SKIPPED_HARDWARE"
                required_gate_n = job.get("requires_gate_n")
                if required_gate_n is not None:
                    gate_key = (str(job.get("gate_series")), int(required_gate_n))
                    if gate_results.get(gate_key) is not True:
                        skip_reason = "SKIPPED_UPSTREAM_GATE"
                if skip_reason:
                    campaign_job_rows.append({
                        **base_row, "return_code": None, "status": skip_reason,
                        "reason": skip_reason, "case_count": 0,
                        "elapsed_seconds": time.perf_counter() - started,
                    })
                    write_campaign_checkpoint()
                    print(f"[{job_id}] {skip_reason}")
                    continue

                profile = copy.deepcopy(base_suite["profiles"][profile_name])
                cases = select_profile_cases(
                    profile,
                    sizes=sizes,
                    families=families,
                    case_ids=job.get("case_ids", ()),
                )
                if not cases:
                    campaign_job_rows.append({
                        **base_row, "return_code": None, "status": "CONFIG_ERROR",
                        "reason": "profile filter selected zero cases", "case_count": 0,
                        "elapsed_seconds": time.perf_counter() - started,
                    })
                    write_campaign_checkpoint()
                    print(f"[{job_id}] CONFIG_ERROR: profile filter selected zero cases")
                    if job.get("gate_series"):
                        gate_results[(str(job["gate_series"]), sizes[0])] = False
                    continue

                if profile_name == "scale_archived_exact":
                    synthetic_sizes = [
                        case["expected_n_train"] for case in cases
                        if case.get("dataset_family") == "Synthetic"
                    ]
                    if synthetic_sizes:
                        validate_archived_synthetic_inputs(synthetic_sizes)
                profile["cases"] = cases
                runtime_suite = {
                    "base": copy.deepcopy(base_suite["base"]),
                    "profiles": {profile_name: profile},
                }
                runtime_path = RUNTIME_CONFIG_ROOT / f"{job_id}.json"
                runtime_path.write_text(json.dumps(runtime_suite, indent=2), encoding="utf-8")
                output_root = CONTROLLED_OUTPUT_ROOT / profile_name / "_jobs" / job_id
                output_root.mkdir(parents=True, exist_ok=True)
                job_case_records = []
                for case in cases:
                    record = {
                        "output_group": profile_name,
                        "suite_profile": profile_name,
                        "job_id": job_id,
                        "case_id": case["id"],
                        "dataset_family": case.get("dataset_family", ""),
                        "run_dir": output_root / case["id"],
                        "scale_role": case.get("scale_role"),
                        "mandatory": bool(job.get("mandatory", True)),
                    }
                    selected_case_records.append(record)
                    job_case_records.append(record)
                expected_controlled_case_count += len(cases)

                completed = run_cmd([
                    sys.executable, "-m",
                    "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.suite",
                    "--config", str(runtime_path), "--profile", profile_name,
                    "--dataset-dir", str(LOCAL_DATA_DIR),
                    "--output-root", str(output_root),
                    "--nufft-backend", "cufinufft", "--strict-gpu-eig",
                    "--execute", "--resume",
                ], cwd=LOCAL_REPO, check=False)
                status_path = output_root / "suite_status.json"
                try:
                    status_rows = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    status_rows = []
                job_status, reason = classify_suite_invocation(
                    completed.returncode, status_rows, len(cases)
                )
                campaign_job_rows.append({
                    **base_row,
                    "return_code": int(completed.returncode),
                    "status": job_status,
                    "reason": reason,
                    "case_count": len(cases),
                    "suite_status_path": str(status_path),
                    "elapsed_seconds": time.perf_counter() - started,
                })
                controlled_profile_outputs.append(output_root)
                if job.get("gate_series"):
                    gate_results[(str(job["gate_series"]), sizes[0])] = scale_core_pass(job_case_records)
                write_campaign_checkpoint()
                print(f"[{job_id}] {job_status}: {reason or 'all selected cases passed'}")

            print("Controlled job outputs:", [str(path) for path in controlled_profile_outputs])
            display(pd.DataFrame(campaign_job_rows))
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
                extra_cases = payload["profiles"][profile_name]["cases"]
                expected_controlled_case_count += len(extra_cases)
                for case in extra_cases:
                    selected_case_records.append({
                        "output_group": label,
                        "suite_profile": profile_name,
                        "case_id": case["id"],
                        "run_dir": output_root / case["id"],
                        "scale_role": case.get("scale_role"),
                    })
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
            expected_controlled_case_count = len(selected_case_records)
            selected_run_dirs = {
                Path(record["run_dir"]).resolve() for record in selected_case_records
            }
            discovered_run_dirs = {
                path.parent.resolve()
                for path in CONTROLLED_OUTPUT_ROOT.rglob("system_manifest.json")
            }
            ignored_stale_dirs = sorted(discovered_run_dirs - selected_run_dirs, key=str)
            audit_rows = []
            for record in selected_case_records:
                run_dir = Path(record["run_dir"]).resolve()
                manifest_path = run_dir / "system_manifest.json"
                config_path = run_dir / "experiment_config.json"
                complete_path = run_dir / "run_complete.json"
                summary_path = run_dir / "matched_summary.csv"
                problems = []
                warnings = []
                manifest = {}
                config = {}
                if not manifest_path.is_file():
                    problems.append("missing system_manifest.json")
                else:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if not manifest.get("system_unchanged"): problems.append("system changed")
                    if manifest.get("nufft_backend_resolved") != "cufinufft": problems.append("backend fallback")
                    if manifest.get("nufft_stage") != "cufinufft": problems.append("NUFFT stage is not GPU")
                    if manifest.get("precision_mode") != "fp64": problems.append("not fp64")
                    missing_component_hashes = [
                        field for field in ("weights_sha256", "gf_sha256", "rhs_sha256")
                        if not manifest.get(field)
                    ]
                    if missing_component_hashes:
                        warnings.append(
                            "legacy manifest lacks component hashes: "
                            + ", ".join(missing_component_hashes)
                        )
                if not config_path.is_file():
                    problems.append("missing experiment_config.json")
                else:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    if not config.get("strict_gpu_eig"): problems.append("strict_gpu_eig false")
                if not complete_path.is_file(): problems.append("missing run_complete.json")
                if not summary_path.is_file():
                    problems.append("missing matched_summary.csv")
                else:
                    matched = pd.read_csv(summary_path)
                    required_columns = {
                        "method", "measured_repeats", "converged_repeats",
                        "performance_claim_eligible", "true_relres_max", "cold_speedup_median",
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
                        true_relres_ok = pd.to_numeric(
                            matched["true_relres_max"], errors="coerce"
                        ).le(float(config.get("tol", 1e-7)))
                        claim_ok = eligible & five_of_five & true_relres_ok
                        if not bool(claim_ok.all()):
                            bad_methods = matched.loc[~claim_ok, "method"].astype(str).tolist()
                            problems.append(f"ineligible/nonconverged methods: {bad_methods}")
                        cg = matched.loc[matched["method"].astype(str).eq("cg")]
                        cg_speed = pd.to_numeric(cg.get("cold_speedup_median"), errors="coerce")
                        if len(cg) != 1 or not bool((cg_speed - 1.0).abs().le(1e-12).all()):
                            problems.append("CG cold speedup is not exactly one")
                        if {"build_seconds_median", "build_seconds_max"}.issubset(matched.columns):
                            build_median = pd.to_numeric(matched["build_seconds_median"], errors="coerce")
                            build_max = pd.to_numeric(matched["build_seconds_max"], errors="coerce")
                            jitter = build_median.gt(0.01) & build_max.gt(1.5 * build_median)
                            if bool(jitter.any()):
                                warnings.append(
                                    "build-time spikes: "
                                    + ", ".join(matched.loc[jitter, "method"].astype(str).tolist())
                                )
                audit_rows.append({
                    "output_group": record["output_group"],
                    "suite_profile": record["suite_profile"],
                    "job_id": record.get("job_id", record["suite_profile"]),
                    "mandatory": bool(record.get("mandatory", True)),
                    "case": record["case_id"],
                    "N": manifest.get("n_train"),
                    "system_id": manifest.get("system_id"),
                    "weights_sha256": manifest.get("weights_sha256"),
                    "gf_sha256": manifest.get("gf_sha256"),
                    "rhs_sha256": manifest.get("rhs_sha256"),
                    "warning": "; ".join(warnings),
                    "status": "PASS" if not problems else "FAIL: " + "; ".join(problems),
                })
            controlled_artifact_audit = pd.DataFrame(
                audit_rows,
                columns=[
                    "output_group", "suite_profile", "job_id", "mandatory", "case", "N", "system_id",
                    "weights_sha256", "gf_sha256", "rhs_sha256", "warning", "status",
                ],
            )
            CONTROLLED_AUDIT_PATH = DRIVE_RUN_ROOT / "controlled_artifact_audit.csv"
            controlled_artifact_audit.to_csv(CONTROLLED_AUDIT_PATH, index=False)
            ignored_controlled_artifacts = pd.DataFrame({
                "ignored_stale_run_dir": [str(path) for path in ignored_stale_dirs]
            })
            IGNORED_ARTIFACTS_PATH = DRIVE_RUN_ROOT / "ignored_stale_controlled_artifacts.csv"
            ignored_controlled_artifacts.to_csv(IGNORED_ARTIFACTS_PATH, index=False)
            display(controlled_artifact_audit)
            if ignored_stale_dirs:
                print(f"Ignored {len(ignored_stale_dirs)} stale/non-selected controlled run directories.")
            if len(controlled_artifact_audit) != expected_controlled_case_count:
                print(
                    "CONTROLLED AUDIT COUNT MISMATCH: "
                    f"Expected exactly {expected_controlled_case_count} controlled cases, "
                    f"but found {len(controlled_artifact_audit)} manifests."
                )
            if not controlled_artifact_audit.empty and not controlled_artifact_audit["status"].eq("PASS").all():
                print(
                    "CONTROLLED ARTIFACT AUDIT CONTAINS FAILURES. "
                    "Failed rows remain in the diagnostic tables and are excluded from performance plots."
                )
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
            expected_prediction_case_count = 0
            if RUN_PREDICTION_AUDIT:
                prediction_records = [
                    record for record in selected_case_records
                    if record.get("suite_profile") in set(PREDICTION_AUDIT_PROFILES)
                ]
                expected_prediction_case_count = len(prediction_records)
                prediction_targets = []
                prediction_errors = []
                for record in prediction_records:
                    config_path = Path(record["run_dir"]) / "experiment_config.json"
                    manifest_path = Path(record["run_dir"]) / "system_manifest.json"
                    if not manifest_path.is_file() or not config_path.is_file():
                        prediction_errors.append(
                            f"prediction target lacks config/manifest: {record['run_dir']}"
                        )
                        continue
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    n_train = int(manifest["n_train"])
                    if n_train > int(PREDICTION_AUDIT_MAX_TRAIN_N):
                        continue
                    prediction_targets.append((record, config_path, manifest))
                for record, config_path, manifest in prediction_targets:
                    audit_out = config_path.parent / "prediction_audit"
                    if (audit_out / "prediction_audit.json").is_file():
                        existing = json.loads(
                            (audit_out / "prediction_audit.json").read_text(encoding="utf-8")
                        )
                        current_config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
                        if existing.get("config_source_sha256") != current_config_sha:
                            prediction_errors.append(f"stale prediction audit: {record['case_id']}")
                            continue
                        if existing.get("dataset_content_index_sha256") != manifest.get("dataset_content_index_sha256"):
                            prediction_errors.append(f"prediction data mismatch: {record['case_id']}")
                            continue
                        audit_source_sha = existing.get("source_bundle_sha256")
                        timing_source_sha = manifest.get("source_bundle_sha256")
                        if (
                            audit_source_sha is not None
                            and timing_source_sha is not None
                            and audit_source_sha != timing_source_sha
                        ):
                            prediction_errors.append(f"prediction source mismatch: {record['case_id']}")
                            continue
                        print("Prediction audit already exists; skipping", config_path.parent.name)
                        prediction_outputs.append(audit_out)
                        continue
                    completed = run_cmd([
                        sys.executable, "-m",
                        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.prediction_audit",
                        "--config", str(config_path),
                        "--prediction-chunk-size", "100000",
                        "--warmup-solves", "1",
                        "--max-test", str(n_train // 4),
                        "--output-dir", str(audit_out),
                    ], cwd=LOCAL_REPO, check=False)
                    if completed.returncode == 0 and (audit_out / "prediction_audit.json").is_file():
                        prediction_outputs.append(audit_out)
                    else:
                        prediction_errors.append(
                            f"prediction command failed for {record['case_id']} rc={completed.returncode}"
                        )
                prediction_status = (
                    "PASS" if expected_prediction_case_count > 0
                    and len(prediction_outputs) == expected_prediction_case_count
                    and not prediction_errors else "EXECUTION_ERROR"
                )
                campaign_job_rows.append({
                    "job_id": "paper_10m_prediction_audit",
                    "profile": "prediction_audit",
                    "dataset_family": "Synthetic,Winnebago,Manitowoc",
                    "n_train": "10000000",
                    "mandatory": True,
                    "return_code": 0 if prediction_status == "PASS" else 1,
                    "status": prediction_status,
                    "reason": "; ".join(prediction_errors),
                    "case_count": expected_prediction_case_count,
                    "elapsed_seconds": None,
                })
                write_campaign_checkpoint()
            print("Prediction audit outputs:", prediction_outputs)
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            # D. 统一结果索引与图

            统一索引只负责查找和展示，不跨 protocol 计算 speedup。每行保留 `output_group/case_id`、科学配置哈希、数据与源码哈希；历史 profile 可以共存在 catalog 中，但作图必须按 profile 隔离。Controlled 表的 `cold_speedup_median` 是 selection/build+solve、排除公共 Fourier setup；`shared_fourier_setup_plus_method_speedup_median` 才是把公共 setup 加回两边后的比值。

            `paper_10m` 是固定规模的方法比较，单独画带范围的分组图和速度–内存 Pareto 图；三种 `scale_*` protocol 各自分图。旧的 `controlled_scale_speedup.png` 会被标记为 deprecated，不再作为论文证据。
            """
        )
    )
    cells.append(
        _code(
            r"""
            import re

            def dataset_family_label(stem):
                text = str(stem or "")
                lower = text.lower()
                if "synthetic_true_func_2d" in lower:
                    return "Synthetic"
                if "winnebago" in lower:
                    return "Winnebago"
                if "usgs_ept_wi_2county_1_b23" in lower:
                    return "Manitowoc"
                return text

            CONFIG_INDEX_FIELDS = (
                "kernel_family", "lengthscale", "nu", "variance", "reg_lambda",
                "fourier_eps", "nufft_tol", "l2_scaled", "tol", "maxiter",
                "precision", "subset_mode", "subset_seed", "score_tau", "box_budget",
                "inverse_max_size", "rank", "nystrom_rank", "rpcholesky_rank",
                "eig_tol", "eig_maxiter", "measured_repeats", "warmup_repeats",
                "method_order_seed", "eig_seed", "nystrom_seed", "rpcholesky_seed",
                "precompute_chunk_size", "strict_gpu_eig",
            )
            SYSTEM_COMPONENT_FIELDS = ("weights_sha256", "gf_sha256", "rhs_sha256")

            def dataset_series_id(stem):
                return re.sub(r"_n(?:train)?\d+$", "", str(stem or ""), flags=re.IGNORECASE)
            selected_record_by_dir = {
                str(Path(record["run_dir"]).resolve()): record
                for record in selected_case_records
            }
            result_frames = []
            controlled_frames = []
            for summary_path in sorted(CONTROLLED_OUTPUT_ROOT.rglob("matched_summary.csv")):
                frame = pd.read_csv(summary_path)
                run_dir = summary_path.parent.resolve()
                manifest = json.loads(summary_path.with_name("system_manifest.json").read_text(encoding="utf-8"))
                config_path = summary_path.with_name("experiment_config.json")
                config = json.loads(config_path.read_text(encoding="utf-8"))
                relative_parts = run_dir.relative_to(CONTROLLED_OUTPUT_ROOT.resolve()).parts
                output_group = relative_parts[0] if relative_parts else "unknown"
                selected_record = selected_record_by_dir.get(str(run_dir))
                frame["protocol_family"] = "controlled_fixed_system"
                frame["evidence_role"] = "paired_scale_or_replication"
                frame["timing_scope"] = "selection/build + solve; shared Fourier setup excluded from cold columns"
                frame["output_group"] = output_group
                frame["suite_profile"] = (
                    selected_record.get("suite_profile") if selected_record else output_group
                )
                frame["case"] = run_dir.name
                frame["case_id"] = run_dir.name
                frame["run_dir"] = str(run_dir)
                frame["selected_in_this_invocation"] = selected_record is not None
                frame["scale_role"] = selected_record.get("scale_role") if selected_record else None
                frame["dataset_stem"] = manifest.get("dataset_stem")
                frame["dataset_family"] = dataset_family_label(manifest.get("dataset_stem"))
                frame["dataset_series_id"] = dataset_series_id(manifest.get("dataset_stem"))
                frame["N"] = manifest.get("n_train")
                frame["system_id"] = manifest.get("system_id")
                for field in SYSTEM_COMPONENT_FIELDS:
                    frame[field] = manifest.get(field)
                frame["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
                frame["source_bundle_sha256"] = manifest.get("source_bundle_sha256")
                frame["dataset_content_index_sha256"] = manifest.get("dataset_content_index_sha256")
                frame["dataset_metadata_sha256"] = manifest.get("dataset_metadata_sha256")
                frame["nufft_backend_resolved"] = manifest.get("nufft_backend_resolved")
                frame["nufft_stage"] = manifest.get("nufft_stage")
                frame["precision_mode"] = manifest.get("precision_mode")
                for field in CONFIG_INDEX_FIELDS:
                    frame[f"cfg_{field}"] = config.get(field)
                scientific_config = {field: config.get(field) for field in CONFIG_INDEX_FIELDS}
                frame["scientific_config_id"] = hashlib.sha256(
                    json.dumps(scientific_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                expected_repeats = int(config.get("measured_repeats", 5))
                eligible = frame["performance_claim_eligible"].astype(str).str.lower().eq("true")
                artifact_eligible = bool(
                    manifest.get("system_unchanged")
                    and manifest.get("nufft_backend_resolved") == "cufinufft"
                    and manifest.get("nufft_stage") == "cufinufft"
                    and manifest.get("precision_mode") == "fp64"
                    and config.get("strict_gpu_eig")
                    and summary_path.with_name("run_complete.json").is_file()
                )
                frame["artifact_eligible"] = artifact_eligible
                frame["claim_eligible"] = (
                    eligible
                    & artifact_eligible
                    & pd.to_numeric(frame["measured_repeats"], errors="coerce").eq(expected_repeats)
                    & pd.to_numeric(frame["converged_repeats"], errors="coerce").eq(expected_repeats)
                    & pd.to_numeric(frame["true_relres_max"], errors="coerce").le(float(config.get("tol", 1e-7)))
                )
                controlled_frames.append(frame)
                result_frames.append(frame)

            for audit_path in sorted(CONTROLLED_OUTPUT_ROOT.rglob("prediction_audit.csv")):
                frame = pd.read_csv(audit_path)
                run_dir = audit_path.parent.parent.resolve()
                manifest = json.loads((run_dir / "system_manifest.json").read_text(encoding="utf-8"))
                audit_payload_path = audit_path.with_suffix(".json")
                audit_payload = (
                    json.loads(audit_payload_path.read_text(encoding="utf-8"))
                    if audit_payload_path.is_file() else {}
                )
                relative_parts = run_dir.relative_to(CONTROLLED_OUTPUT_ROOT.resolve()).parts
                output_group = relative_parts[0] if relative_parts else "unknown"
                selected_record = selected_record_by_dir.get(str(run_dir))
                frame["audit_system_id"] = frame["system_id"]
                frame["timing_system_id"] = manifest.get("system_id")
                for field in SYSTEM_COMPONENT_FIELDS:
                    frame[f"audit_{field}"] = audit_payload.get(field)
                    frame[f"timing_{field}"] = manifest.get(field)
                frame["audit_rebuilt_system"] = True
                frame["protocol_family"] = "prediction_audit"
                frame["evidence_role"] = "accuracy_only"
                frame["timing_scope"] = "excluded from all speed claims"
                frame["output_group"] = output_group
                frame["suite_profile"] = (
                    selected_record.get("suite_profile") if selected_record else output_group
                )
                frame["case"] = run_dir.name
                frame["case_id"] = run_dir.name
                frame["run_dir"] = str(run_dir)
                frame["selected_in_this_invocation"] = selected_record is not None
                frame["dataset_stem"] = manifest.get("dataset_stem")
                frame["dataset_family"] = dataset_family_label(manifest.get("dataset_stem"))
                frame["dataset_series_id"] = dataset_series_id(manifest.get("dataset_stem"))
                frame["N"] = manifest.get("n_train")
                result_frames.append(frame)

            controlled_catalog = (
                pd.concat(controlled_frames, ignore_index=True, sort=False)
                if controlled_frames else pd.DataFrame()
            )
            if not controlled_catalog.empty:
                duplicate_key = ["output_group", "case_id", "method"]
                selected_for_duplicate_check = controlled_catalog.loc[
                    controlled_catalog["selected_in_this_invocation"]
                ]
                duplicated = selected_for_duplicate_check.duplicated(duplicate_key, keep=False)
                if bool(duplicated.any()):
                    raise RuntimeError(
                        "Duplicate controlled summary rows:\n"
                        + selected_for_duplicate_check.loc[duplicated, duplicate_key].to_string(index=False)
                    )
            SELECTED_CONTROLLED_INDEX_PATH = DRIVE_RUN_ROOT / "selected_controlled_index.csv"
            selected_controlled = (
                controlled_catalog.loc[controlled_catalog["selected_in_this_invocation"]].copy()
                if not controlled_catalog.empty else pd.DataFrame()
            )
            selected_controlled.to_csv(SELECTED_CONTROLLED_INDEX_PATH, index=False)
            INELIGIBLE_INDEX_PATH = DRIVE_RUN_ROOT / "controlled_ineligible_rows.csv"
            ineligible_controlled = (
                controlled_catalog.loc[
                    controlled_catalog["selected_in_this_invocation"]
                    & ~controlled_catalog["claim_eligible"]
                ].copy()
                if not controlled_catalog.empty else pd.DataFrame()
            )
            ineligible_controlled.to_csv(INELIGIBLE_INDEX_PATH, index=False)

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
            import numpy as np
            from matplotlib.lines import Line2D

            METHOD_ORDER = ["cg", "jacobi", "default", "full-eig", "nystrom", "rpcholesky"]
            METHOD_COLORS = {
                "cg": "#4C78A8",
                "jacobi": "#9D9D9D",
                "default": "#E45756",
                "full-eig": "#72B7B2",
                "nystrom": "#54A24B",
                "rpcholesky": "#B279A2",
            }
            DATASET_ORDER = ["Manitowoc", "Winnebago", "Synthetic"]
            GENERATED_PLOT_PATHS = []

            def assert_cg_reference_one(frame, *, context):
                cg = frame.loc[frame["method"].astype(str).eq("cg")]
                values = pd.to_numeric(cg["cold_speedup_median"], errors="coerce").to_numpy(float)
                if values.size == 0 or not np.isfinite(values).all() or not np.allclose(
                    values, 1.0, rtol=0.0, atol=1e-12
                ):
                    raise RuntimeError(f"{context}: CG cold speedup must be exactly one; got {values}")

            controlled_plot = controlled_catalog.copy()
            if not controlled_plot.empty:
                controlled_plot = controlled_plot.loc[controlled_plot["claim_eligible"]].copy()

            # paper_10m is a method comparison at one N, not a scale curve.
            paper_plot = (
                controlled_plot.loc[
                    controlled_plot["output_group"].eq("paper_10m")
                    & pd.to_numeric(controlled_plot["N"], errors="coerce").eq(10_000_000)
                ].copy()
                if not controlled_plot.empty else pd.DataFrame()
            )
            if not paper_plot.empty:
                assert_cg_reference_one(paper_plot, context="paper_10m")
                paper_key = ["dataset_family", "method"]
                if bool(paper_plot.duplicated(paper_key, keep=False).any()):
                    raise RuntimeError("paper_10m has duplicate dataset/method rows; refusing to aggregate them.")
                dataset_order = [name for name in DATASET_ORDER if name in set(paper_plot["dataset_family"])]
                dataset_order += sorted(set(paper_plot["dataset_family"]) - set(dataset_order))
                x = np.arange(len(dataset_order), dtype=float)
                width = 0.13
                fig, ax = plt.subplots(figsize=(11.5, 6.2))
                for method_index, method in enumerate(METHOD_ORDER):
                    rows = paper_plot.loc[paper_plot["method"].eq(method)].set_index("dataset_family")
                    if rows.empty:
                        continue
                    values = np.asarray([
                        float(rows.loc[name, "cold_speedup_median"]) if name in rows.index else np.nan
                        for name in dataset_order
                    ])
                    mins = np.asarray([
                        float(rows.loc[name, "cold_speedup_min"]) if name in rows.index else np.nan
                        for name in dataset_order
                    ])
                    maxs = np.asarray([
                        float(rows.loc[name, "cold_speedup_max"]) if name in rows.index else np.nan
                        for name in dataset_order
                    ])
                    positions = x + (method_index - (len(METHOD_ORDER) - 1) / 2) * width
                    yerr = np.vstack([
                        np.nan_to_num(np.maximum(values - mins, 0.0)),
                        np.nan_to_num(np.maximum(maxs - values, 0.0)),
                    ])
                    bars = ax.bar(
                        positions, values, width=width, color=METHOD_COLORS[method], label=method,
                        yerr=yerr, capsize=2, linewidth=0.4, edgecolor="white",
                    )
                    for bar, value in zip(bars, values):
                        if np.isfinite(value):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                                f"{value:.2f}", ha="center", va="bottom", fontsize=7, rotation=90,
                            )
                ax.axhline(1.0, color="black", lw=1, ls="--")
                ax.set_xticks(x, dataset_order)
                ax.set_ylabel("paired cold speedup over CG")
                ax.set_title("Controlled 10M: selection/build + solve (median; whiskers=min–max)")
                ax.grid(axis="y", alpha=.25)
                ax.legend(ncol=3, frameon=False, loc="upper left")
                fig.tight_layout()
                paper_plot_path = DRIVE_RUN_ROOT / "controlled_10m_method_speedup.png"
                fig.savefig(paper_plot_path, dpi=180, bbox_inches="tight")
                GENERATED_PLOT_PATHS.append(paper_plot_path)
                plt.show()
                PAPER_10M_SUMMARY_PATH = DRIVE_RUN_ROOT / "controlled_10m_method_summary.csv"
                paper_plot.sort_values(["dataset_family", "method"]).to_csv(
                    PAPER_10M_SUMMARY_PATH, index=False
                )
                display(paper_plot[[
                    "dataset_family", "method", "cold_speedup_median", "cold_speedup_min",
                    "cold_speedup_max", "shared_fourier_setup_plus_method_speedup_median",
                    "iterations_median", "build_plus_solve_seconds_median",
                    "preconditioner_storage_bytes", "claim_eligible",
                ]].sort_values(["dataset_family", "method"]))

                # The active-box method is memory-capped; show the speed-memory trade-off explicitly.
                pareto = paper_plot.loc[
                    paper_plot["method"].ne("cg")
                    & pd.to_numeric(paper_plot["preconditioner_storage_bytes"], errors="coerce").gt(0)
                ].copy()
                if not pareto.empty:
                    fig, axes = plt.subplots(
                        1, len(dataset_order), figsize=(5.0 * len(dataset_order), 4.5), squeeze=False,
                        sharey=True,
                    )
                    for ax, dataset in zip(axes[0], dataset_order):
                        subset = pareto.loc[pareto["dataset_family"].eq(dataset)]
                        for _, row in subset.iterrows():
                            memory_mib = float(row["preconditioner_storage_bytes"]) / 2**20
                            speedup = float(row["cold_speedup_median"])
                            method = str(row["method"])
                            ax.scatter(memory_mib, speedup, s=60, color=METHOD_COLORS.get(method, "black"))
                            ax.annotate(method, (memory_mib, speedup), xytext=(4, 4),
                                        textcoords="offset points", fontsize=8)
                        ax.axhline(1.0, color="black", lw=1, ls="--")
                        ax.set_xscale("log")
                        ax.set_title(dataset)
                        ax.set_xlabel("preconditioner storage (MiB, log scale)")
                        ax.grid(True, alpha=.25)
                    axes[0][0].set_ylabel("paired cold speedup over CG")
                    fig.suptitle("Controlled 10M speed-memory trade-off", y=1.02)
                    fig.tight_layout()
                    pareto_path = DRIVE_RUN_ROOT / "controlled_10m_speed_memory_pareto.png"
                    fig.savefig(pareto_path, dpi=180, bbox_inches="tight")
                    GENERATED_PLOT_PATHS.append(pareto_path)
                    plt.show()

            # One-at-a-time robustness scan: lambda varies at ell=0.1, and ell varies at lambda=0.1.
            oat_plot = (
                controlled_plot.loc[controlled_plot["output_group"].eq("winnebago_oat_n10m")].copy()
                if not controlled_plot.empty else pd.DataFrame()
            )
            if not oat_plot.empty:
                assert_cg_reference_one(oat_plot, context="winnebago_oat_n10m")
                oat_key = ["case_id", "method"]
                if bool(oat_plot.duplicated(oat_key, keep=False).any()):
                    raise RuntimeError("winnebago_oat_n10m has duplicate case/method rows.")
                OAT_SUMMARY_PATH = DRIVE_RUN_ROOT / "winnebago_oat_10m_summary.csv"
                oat_plot.sort_values(["cfg_reg_lambda", "cfg_lengthscale", "method"]).to_csv(
                    OAT_SUMMARY_PATH, index=False
                )
                oat_methods = [method for method in ("cg", "default", "full-eig")
                               if method in set(oat_plot["method"])]
                sweep_specs = [
                    ("cfg_reg_lambda", "regularization λ", "cfg_lengthscale", 0.1),
                    ("cfg_lengthscale", "lengthscale ℓ", "cfg_reg_lambda", 0.1),
                ]
                fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), squeeze=False)
                for column, (x_field, x_label, fixed_field, fixed_value) in enumerate(sweep_specs):
                    sweep = oat_plot.loc[
                        np.isclose(
                            pd.to_numeric(oat_plot[fixed_field], errors="coerce"),
                            fixed_value, rtol=0.0, atol=1e-15,
                        )
                    ].copy()
                    ax_speed = axes[0][column]
                    ax_memory = axes[1][column]
                    for method in oat_methods:
                        group = sweep.loc[sweep["method"].eq(method)].sort_values(x_field)
                        if group.empty:
                            continue
                        x_values = pd.to_numeric(group[x_field], errors="raise").to_numpy(float)
                        y_values = pd.to_numeric(group["cold_speedup_median"], errors="raise").to_numpy(float)
                        lower = np.maximum(
                            y_values - pd.to_numeric(group["cold_speedup_min"], errors="raise").to_numpy(float),
                            0.0,
                        )
                        upper = np.maximum(
                            pd.to_numeric(group["cold_speedup_max"], errors="raise").to_numpy(float) - y_values,
                            0.0,
                        )
                        ax_speed.errorbar(
                            x_values, y_values, yerr=np.vstack([lower, upper]), marker="o", capsize=3,
                            color=METHOD_COLORS[method], label=method,
                        )
                        memory = pd.to_numeric(
                            group["preconditioner_storage_bytes"], errors="coerce"
                        ).to_numpy(float) / 2**20
                        finite_memory = np.isfinite(memory) & (memory > 0)
                        if finite_memory.any():
                            ax_memory.plot(
                                x_values[finite_memory], memory[finite_memory], marker="o",
                                color=METHOD_COLORS[method], label=method,
                            )
                    for ax in (ax_speed, ax_memory):
                        ax.set_xscale("log")
                        ax.set_xlabel(x_label)
                        ax.grid(True, alpha=.25)
                    ax_speed.axhline(1.0, color="black", lw=1, ls="--")
                    ax_speed.set_ylabel("paired cold speedup over CG")
                    ax_speed.set_title(f"fixed {fixed_field.replace('cfg_', '')}={fixed_value:g}")
                    ax_memory.set_yscale("log")
                    ax_memory.set_ylabel("preconditioner storage (MiB, log scale)")
                axes[0][0].legend(frameon=False)
                axes[1][0].legend(frameon=False)
                fig.suptitle("Winnebago 10M one-at-a-time robustness", y=1.01)
                fig.tight_layout()
                oat_plot_path = DRIVE_RUN_ROOT / "winnebago_oat_10m_speed_memory.png"
                fig.savefig(oat_plot_path, dpi=180, bbox_inches="tight")
                GENERATED_PLOT_PATHS.append(oat_plot_path)
                plt.show()

            # Fixed-system active-box memory-budget ablation.
            BOX_BUDGET_SYSTEM_MATCH = None
            budget_plot = (
                controlled_plot.loc[
                    controlled_plot["output_group"].eq("winnebago_box_budget_n10m")
                ].copy()
                if not controlled_plot.empty else pd.DataFrame()
            )
            if not budget_plot.empty:
                case_systems = budget_plot[[
                    "case_id", "system_id", "weights_sha256", "gf_sha256", "rhs_sha256"
                ]].drop_duplicates()
                BOX_BUDGET_SYSTEM_MATCH = bool(
                    len(case_systems) == 3
                    and all(case_systems[field].notna().all() and case_systems[field].nunique() == 1
                            for field in ("system_id", "weights_sha256", "gf_sha256", "rhs_sha256"))
                )
                if not BOX_BUDGET_SYSTEM_MATCH:
                    print(
                        "BOX-BUDGET AUDIT FAILED: the three budget cases do not share "
                        "identical fixed-system component hashes."
                    )
                budget_key = ["case_id", "method"]
                if bool(budget_plot.duplicated(budget_key, keep=False).any()):
                    print("BOX-BUDGET AUDIT FAILED: duplicate case/method rows.")
                    BOX_BUDGET_SYSTEM_MATCH = False
                BOX_BUDGET_SUMMARY_PATH = DRIVE_RUN_ROOT / "winnebago_box_budget_10m_summary.csv"
                budget_plot.sort_values(["cfg_box_budget", "method"]).to_csv(
                    BOX_BUDGET_SUMMARY_PATH, index=False
                )
                default_budget = budget_plot.loc[budget_plot["method"].eq("default")].sort_values(
                    "cfg_box_budget"
                )
                if not default_budget.empty:
                    x_values = pd.to_numeric(default_budget["cfg_box_budget"], errors="raise").to_numpy(float)
                    actual_sizes = pd.to_numeric(default_budget["box_size"], errors="coerce").to_numpy(float)
                    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
                    axes[0].plot(
                        x_values,
                        pd.to_numeric(default_budget["cold_speedup_median"], errors="raise"),
                        marker="o", color=METHOD_COLORS["default"],
                    )
                    axes[0].axhline(1.0, color="black", lw=1, ls="--")
                    axes[0].set_ylabel("paired cold speedup over CG")
                    axes[1].plot(
                        x_values,
                        pd.to_numeric(default_budget["iterations_median"], errors="raise"),
                        marker="o", color=METHOD_COLORS["default"],
                    )
                    axes[1].set_ylabel("median PCG iterations")
                    axes[2].plot(
                        x_values,
                        pd.to_numeric(default_budget["preconditioner_storage_bytes"], errors="raise") / 2**20,
                        marker="o", color=METHOD_COLORS["default"],
                    )
                    axes[2].set_ylabel("preconditioner storage (MiB)")
                    for ax in axes:
                        ax.set_xscale("log", base=2)
                        ax.set_xlabel("nominal box budget")
                        ax.grid(True, alpha=.25)
                        ax.set_xticks(x_values, [f"{int(x):,}" for x in x_values])
                    for x_value, actual_size in zip(x_values, actual_sizes):
                        if np.isfinite(actual_size):
                            axes[0].annotate(
                                f"actual {int(actual_size):,}",
                                (x_value, float(default_budget.loc[
                                    pd.to_numeric(default_budget["cfg_box_budget"], errors="coerce").eq(x_value),
                                    "cold_speedup_median",
                                ].iloc[0])),
                                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8,
                            )
                    fig.suptitle("Winnebago 10M fixed-system active-box budget ablation")
                    fig.tight_layout()
                    budget_plot_path = DRIVE_RUN_ROOT / "winnebago_box_budget_10m.png"
                    fig.savefig(budget_plot_path, dpi=180, bbox_inches="tight")
                    GENERATED_PLOT_PATHS.append(budget_plot_path)
                    plt.show()

            # Scale protocols remain separate: never connect archived exact, development master,
            # Manitowoc master, OAT, or paper_10m rows into one curve.
            SCALE_OUTPUT_GROUPS = {
                "scale_archived_exact", "scale_development_masters", "scale_manitowoc_master"
            }
            scale_plot = (
                controlled_plot.loc[controlled_plot["output_group"].isin(SCALE_OUTPUT_GROUPS)].copy()
                if not controlled_plot.empty else pd.DataFrame()
            )
            scale_profiles = (
                scale_plot.groupby("output_group", sort=True)
                if not scale_plot.empty else ()
            )
            for output_group, profile_frame in scale_profiles:
                assert_cg_reference_one(profile_frame, context=output_group)
                profile_sources = profile_frame["source_bundle_sha256"].dropna().astype(str)
                if len(profile_sources) != len(profile_frame) or profile_sources.nunique() != 1:
                    raise RuntimeError(f"{output_group} mixes or lacks source bundles.")
                families = [name for name in DATASET_ORDER if name in set(profile_frame["dataset_family"])]
                families += sorted(set(profile_frame["dataset_family"]) - set(families))
                fig, axes = plt.subplots(
                    1, len(families), figsize=(5.2 * len(families), 4.6), squeeze=False, sharey=True,
                )
                for ax, family in zip(axes[0], families):
                    family_frame = profile_frame.loc[profile_frame["dataset_family"].eq(family)]
                    series_values = family_frame["dataset_series_id"].dropna().astype(str)
                    if (
                        len(series_values) != len(family_frame)
                        or not bool(series_values.str.len().gt(0).all())
                        or series_values.nunique() != 1
                    ):
                        raise RuntimeError(
                            f"{output_group}/{family} mixes or lacks dataset series."
                        )
                    data_fields = [
                        "N", "dataset_stem", "dataset_content_index_sha256",
                        "dataset_metadata_sha256",
                    ]
                    case_data = family_frame[data_fields].drop_duplicates()
                    if bool(case_data[data_fields].isna().any().any()):
                        raise RuntimeError(f"{output_group}/{family} lacks dataset provenance.")
                    if bool(pd.to_numeric(case_data["N"], errors="coerce").duplicated().any()):
                        raise RuntimeError(
                            f"{output_group}/{family} has multiple data artifacts for one N."
                        )
                    if output_group in {"scale_development_masters", "scale_manitowoc_master"}:
                        for field in (
                            "dataset_stem", "dataset_content_index_sha256", "dataset_metadata_sha256"
                        ):
                            if case_data[field].astype(str).nunique() != 1:
                                raise RuntimeError(
                                    f"{output_group}/{family} does not reuse one master {field}."
                                )
                    for method in METHOD_ORDER:
                        group = family_frame.loc[family_frame["method"].eq(method)].sort_values("N")
                        if group.empty:
                            continue
                        if group["scientific_config_id"].nunique(dropna=False) != 1:
                            raise RuntimeError(
                                f"{output_group}/{family}/{method} mixes scientific configurations."
                            )
                        if bool(pd.to_numeric(group["N"], errors="coerce").duplicated().any()):
                            raise RuntimeError(f"{output_group}/{family}/{method} has duplicate N values.")
                        x_values = pd.to_numeric(group["N"], errors="raise").to_numpy(float)
                        y_values = pd.to_numeric(group["cold_speedup_median"], errors="raise").to_numpy(float)
                        linestyle = "-" if np.unique(x_values).size >= 2 else "None"
                        ax.plot(
                            x_values, y_values, marker="o", ls=linestyle,
                            color=METHOD_COLORS.get(method), label=method,
                        )
                    ax.set_xscale("log")
                    ax.axhline(1.0, color="black", lw=1, ls="--")
                    ax.set_title(family)
                    ax.set_xlabel("training rows N")
                    ax.grid(True, alpha=.25)
                axes[0][0].set_ylabel("paired cold speedup over CG")
                handles = [
                    Line2D([0], [0], marker="o", color=METHOD_COLORS[m], label=m)
                    for m in METHOD_ORDER if m in set(profile_frame["method"])
                ]
                fig.legend(handles=handles, ncol=min(3, len(handles)), frameon=False,
                           loc="upper center", bbox_to_anchor=(0.5, 1.04))
                fig.suptitle(f"Controlled scale: {output_group}", y=1.11)
                fig.tight_layout()
                scale_plot_path = DRIVE_RUN_ROOT / f"{output_group}_cold_speedup.png"
                fig.savefig(scale_plot_path, dpi=180, bbox_inches="tight")
                GENERATED_PLOT_PATHS.append(scale_plot_path)
                plt.show()

            deprecated_plot = DRIVE_RUN_ROOT / "controlled_scale_speedup.png"
            if deprecated_plot.is_file():
                print("Deprecated ambiguous plot remains on Drive but is excluded:", deprecated_plot)
            print("Generated plot paths:", [str(path) for path in GENERATED_PLOT_PATHS])
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
                "campaign_jobs_csv": str(CAMPAIGN_JOBS_CSV),
                "campaign_jobs_json": str(CAMPAIGN_JOBS_JSON),
                "campaign_jobs": campaign_job_rows,
                "controlled_case_count": int(len(controlled_artifact_audit)),
                "expected_controlled_case_count": int(expected_controlled_case_count),
                "selected_controlled_cases": [
                    {
                        **{key: value for key, value in record.items() if key != "run_dir"},
                        "run_dir": str(record["run_dir"]),
                    }
                    for record in selected_case_records
                ],
                "ignored_stale_controlled_run_count": int(len(ignored_stale_dirs)),
                "all_controlled_artifacts_pass": (
                    None if expected_controlled_case_count == 0 else bool(
                        len(controlled_artifact_audit) == expected_controlled_case_count
                        and controlled_artifact_audit["status"].eq("PASS").all()
                    )
                ),
                "unified_index": str(INDEX_PATH),
                "selected_controlled_index": str(SELECTED_CONTROLLED_INDEX_PATH),
                "controlled_artifact_audit": str(CONTROLLED_AUDIT_PATH),
                "controlled_ineligible_rows": str(INELIGIBLE_INDEX_PATH),
                "generated_plots": [str(path) for path in GENERATED_PLOT_PATHS],
                "expected_prediction_case_count": int(expected_prediction_case_count),
                "box_budget_fixed_system_match": BOX_BUDGET_SYSTEM_MATCH,
            }
            legacy_complete = all(
                (DRIVE_RUN_ROOT / "legacy_archived_pipeline" / group / "_SUCCESS.json").is_file()
                for group in RUN_LEGACY_GROUPS
            )
            controlled_complete = (
                expected_controlled_case_count == 0
                or (
                    len(controlled_artifact_audit) == expected_controlled_case_count
                    and controlled_artifact_audit["status"].eq("PASS").all()
                )
            )
            prediction_complete = (
                not RUN_PREDICTION_AUDIT
                or (
                    expected_prediction_case_count > 0
                    and len(prediction_outputs) == expected_prediction_case_count
                    and all((Path(path) / "prediction_audit.json").is_file() for path in prediction_outputs)
                )
            )
            workload_requested = bool(
                RUN_LEGACY_GROUPS or selected_profiles or extra_suites or RUN_PREDICTION_AUDIT
            )
            mandatory_jobs = [row for row in campaign_job_rows if bool(row.get("mandatory", True))]
            campaign_complete = bool(
                mandatory_jobs and all(str(row.get("status")) == "PASS" for row in mandatory_jobs)
            )
            run_verified = bool(
                workload_requested and legacy_complete and controlled_complete
                and prediction_complete and campaign_complete and INDEX_PATH.is_file()
                and (BOX_BUDGET_SYSTEM_MATCH is not False)
            )
            final_manifest["campaign_complete"] = campaign_complete
            final_manifest["run_verified"] = run_verified
            FINAL_MANIFEST_PATH = DRIVE_RUN_ROOT / "colab_run_manifest.json"
            FINAL_MANIFEST_PATH.write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
            print(json.dumps(final_manifest, indent=2))
            if run_verified:
                print("ONE-CLICK CAMPAIGN VERIFIED: all mandatory jobs passed.")
            else:
                print(
                    "ONE-CLICK CAMPAIGN COMPLETED WITH FAILURES/SKIPS. "
                    "See campaign_jobs.csv and controlled_artifact_audit.csv; completed results remain usable."
                )

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
