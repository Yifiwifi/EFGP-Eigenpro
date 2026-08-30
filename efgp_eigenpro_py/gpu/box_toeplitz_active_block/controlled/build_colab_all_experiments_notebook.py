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

            本 notebook 的正式一键流程严格分成两个阶段；历史实验只保留为可选入口，**禁止跨阶段混表或混图**：

            | protocol | 内容 | 可以说明什么 |
            |---|---|---|
            | `end_to_end_krr`（Stage 1） | 主报告显式区分 EFGP-CG、inverse 族、active-box EigenPro 族（含 full-grid）；Nyström/RPCholesky/Jacobi 放在次级完整 KRR matrix | 完整 KRR 的精度、setup、solving 和 train-total；10M–300M 两族规模、robustness、SE/Matérn 与次级算法比较 |
            | `controlled_fixed_system`（Stage 2） | Stage 1 冻结选出的 target 上，所有 solver/preconditioner 共享同一个哈希 (A\beta=b) | solver-total = selection + preconditioner build + solve 的严格配对比较 |
            | `archived_complete_pipeline` | 原 `group_a/group_b/group_c`，direct CG 与 binned-C1 candidates，含 setup/solve/prediction | 原论文探索性规模图与候选筛选 |
            | `prediction_audit` | 读取 timed system 与保存的 canonical timed \(\beta\)，仅计算 test RMSE | 同一计时解的预测等价性；其中 prediction 时间不进入 speedup claim |

            使用方式：选择 Colab 的 **A100 GPU + High-RAM** runtime，然后点击 **运行时 → 全部运行**。默认的 `RUN_ALL_FORMAL_EXPERIMENTS=True` 先运行两族 10M–300M replay 和 SE/Matérn profile，再运行含 Nyström/RPCholesky/Jacobi 的次级完整 KRR matrix，并按预注册规则冻结 Stage 2 target；随后分别生成两族 robustness 与 fixed-\(A,b\) solver comparison。族内只在 `ours-binned-active-eig` 和 `efgp-standard-full-eig` 之间按当前 median total 取最小；inverse 不与 eigenpair 合并，RMSE/R² 不作为删除成功 timing 的门槛。

            `nystrom-krr` 与 `rpcholesky-krr` 只出现在 Stage 1；Fourier-space randomized preconditioner adaptations 不得以 Nyström/RPCholesky KRR 的名字进入正式图。参考 notebook 的 legacy 路线默认关闭。
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

            # 正式证据链：Stage 1 完整 KRR，Stage 2 固定 A,b solver/preconditioner。
            RUN_STAGE1_END_TO_END_KRR = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_STAGE1_ROBUSTNESS = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_STAGE1_FAMILY_SCALE = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_STAGE1_FAMILY_ROBUSTNESS = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_STAGE1_FAMILY_KERNEL = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_STAGE2_FIXED_AB_SOLVERS = RUN_ALL_FORMAL_EXPERIMENTS
            STAGE1_SCALE_PROFILE = "scale_10m_300m"
            STAGE1_FAMILY_SCALE_PROFILE = "family_scale_10m_300m"
            STAGE1_FAMILY_KERNEL_PROFILE = "family_kernel_at_30m"
            STAGE1_METHODS = [
                "nystrom-krr", "rpcholesky-krr", "efgp-standard-cg",
                "efgp-standard-jacobi", "efgp-standard-full-eig",
                "ours-binned-default",
            ]
            STAGE1_FAMILY_METHODS = [
                "efgp-standard-cg", "efgp-standard-full-eig",
                "ours-binned-inverse", "ours-binned-active-eig",
            ]
            STAGE2_METHODS = [
                "cg", "jacobi", "default", "active-inverse", "active-eig", "full-eig",
            ]
            STAGE2_MANDATORY_METHODS = [
                "cg", "jacobi", "default", "active-eig", "full-eig",
            ]

            # 从 drive_manifest.json 自动选择正式 campaign 所需数据。
            DATA_BUNDLES = []
            CACHE_DATA_LOCALLY = True       # 正式计时推荐 True，避免 Drive FUSE 进入 timing
            VERIFY_FULL_SHA256 = False      # 只跳过无关 bundle；本次选中 artifact 仍逐个验 SHA-256

            # 一键正式 campaign 不重跑不可直接混表的 legacy exploratory groups。
            RUN_LEGACY_GROUPS = []

            # 下列变量由一键模式统一设定；False 时仍可作为 advanced/manual 模式使用。
            RUN_PLUMBING_SMOKE = RUN_ALL_FORMAL_EXPERIMENTS
            RUN_CG_SCREEN_10M = False
            RUN_Q256_CENTER_10M = False
            RUN_BOX_BUDGET_ABLATION = False
            RUN_ARCHIVED_EXACT_SCALE = False
            RUN_DEVELOPMENT_MASTER_SCALE = False
            RUN_MANITOWOC_SCALE = False
            RUN_WINNEBAGO_OAT_10M = False
            RUN_Q128_BRIDGE = False
            RUN_SE_FULL_INVERSE_CONTROL = False
            RUN_PREDICTION_AUDIT = RUN_ALL_FORMAL_EXPERIMENTS

            ACTIVE_SIZES = list(FORMAL_SCALE_SIZES) if RUN_ALL_FORMAL_EXPERIMENTS else [10_000_000]
            ALLOW_100M = RUN_ALL_FORMAL_EXPERIMENTS
            ALLOW_300M = RUN_ALL_FORMAL_EXPERIMENTS
            PREDICTION_AUDIT_MAX_TEST_N = 2_500_000
            PREDICTION_AUDIT_PROFILES = ["fixed_ab_selected_target"]

            # 正式 Stage 1 自己声明 exact per-N dataset stems。
            PROFILE_DATASET_FAMILIES = {}
            ACTIVE_CASE_IDS = []

            # Match the archived group_a/b/c notebook: formal Synthetic is
            # generated locally per N when absent, with _ntrainN/noise=.3.
            GENERATE_FORMAL_SYNTHETIC_IF_MISSING = RUN_STAGE1_END_TO_END_KRR
            GENERATE_ARCHIVED_SYNTHETIC_SIZES = []  # noise=.3 / chunk=5M / _ntrainN
            GENERATE_MANITOWOC_300M = False
            MANITOWOC_START_LOD = 8  # 只是起点；容量不足时必须提高并重新精确扫描
            SYNC_GENERATED_DATA_TO_DRIVE = False

            required_bundles = []
            if RUN_STAGE1_END_TO_END_KRR:
                # Winnebago exact artifacts come from the catalog. Synthetic
                # exact per-N files are generated locally if not already selected.
                required_bundles.append("archived_exact_available")
            if RUN_LEGACY_GROUPS:
                required_bundles.append("legacy_named_route_inputs")
            if RUN_ARCHIVED_EXACT_SCALE:
                required_bundles.append("archived_exact_available")
            if RUN_DEVELOPMENT_MASTER_SCALE:
                required_bundles.append("development_scale_masters")
            if RUN_MANITOWOC_SCALE:
                required_bundles.append("manitowoc_10m")
            if any([
                RUN_PLUMBING_SMOKE,
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
                requested_size_hints.extend([10_000_000, 100_000_000, 300_000_000])
            MAX_REQUESTED_N = max(requested_size_hints or [0])

            # Release the paid Colab accelerator after every mandatory artifact
            # has been persisted and the final campaign manifest is verified.
            DISCONNECT_RUNTIME_WHEN_VERIFIED = True
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
            NOTEBOOK_STARTED_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            NOTEBOOK_STARTED_PERF_COUNTER = time.perf_counter()
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

            `drive_manifest.json` 由 `colab_drive_pack.py` 生成。正式 Synthetic 与指定的 group_a/b/c notebook 一致：每个 N 使用 `synthetic_true_func_2d_ntrainN`；本地没有时，以 noise=.3、固定 seeds、5M chunk 和 `N/4` test 直接生成。

            已存在且 metadata 完全匹配的 `_ntrainN` 文件可以复用；缺失规模才生成。Winnebago 仍取 `archived_exact_available`。所有正式 Synthetic 文件在 Stage 1 前验证 metadata 并记录 SHA-256。
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
            DATA_MANIFEST_SHA256 = hashlib.sha256(DATA_MANIFEST.read_bytes()).hexdigest()
            DATA_MANIFEST_SNAPSHOT = DRIVE_RUN_ROOT / "data_manifest_snapshot.json"
            if DATA_MANIFEST_SNAPSHOT.is_file():
                snapshot_sha256 = hashlib.sha256(DATA_MANIFEST_SNAPSHOT.read_bytes()).hexdigest()
                if snapshot_sha256 != DATA_MANIFEST_SHA256:
                    raise RuntimeError(
                        "The run directory already contains a different data manifest snapshot; "
                        "use a new RUN_TAG instead of mixing data catalogs."
                    )
            else:
                snapshot_partial = DATA_MANIFEST_SNAPSHOT.with_suffix(".json.partial")
                shutil.copy2(DATA_MANIFEST, snapshot_partial)
                snapshot_partial.replace(DATA_MANIFEST_SNAPSHOT)
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

            def streaming_sha256(path, chunk_bytes=16 * 2**20):
                digest = hashlib.sha256()
                with Path(path).open("rb") as handle:
                    while True:
                        block = handle.read(int(chunk_bytes))
                        if not block:
                            break
                        digest.update(block)
                return digest.hexdigest()

            # Fail before copying if two selected catalog entries would collide
            # at the flat local-cache basename with different bytes.
            selected_by_basename = {}
            for row in selected_artifacts:
                source = DRIVE_DATA_ROOT / row["relative_path"]
                basename = source.name
                prior = selected_by_basename.get(basename)
                if prior is not None and prior["sha256"] != row["sha256"]:
                    raise RuntimeError(
                        f"Selected artifacts collide at {basename!r} with different SHA-256 values."
                    )
                selected_by_basename[basename] = row

            cache_valid_by_basename = {}
            additional_cache_bytes = 0
            for basename, row in selected_by_basename.items():
                source = DRIVE_DATA_ROOT / row["relative_path"]
                if not source.is_file():
                    raise FileNotFoundError(source)
                expected_size = int(row["size_bytes"])
                if source.stat().st_size != expected_size:
                    raise ValueError(f"Byte-size mismatch: {source}")
                destination = LOCAL_DATA_DIR / basename
                cache_valid = False
                if CACHE_DATA_LOCALLY and destination.is_file() and destination.stat().st_size == expected_size:
                    print("Verifying selected local cache SHA-256:", basename)
                    cache_valid = streaming_sha256(destination) == row["sha256"]
                cache_valid_by_basename[basename] = cache_valid
                if CACHE_DATA_LOCALLY and not cache_valid:
                    existing_size = destination.stat().st_size if destination.is_file() else 0
                    additional_cache_bytes += max(expected_size - int(existing_size), 0)
            if CACHE_DATA_LOCALLY and additional_cache_bytes > shutil.disk_usage("/content").free:
                raise RuntimeError(
                    "Selected uncached bytes do not fit on the current Colab local disk."
                )

            staged = {}
            inspected_masters = set()
            for row in selected_artifacts:
                source = DRIVE_DATA_ROOT / row["relative_path"]
                destination = LOCAL_DATA_DIR / source.name
                if CACHE_DATA_LOCALLY:
                    if not cache_valid_by_basename[source.name]:
                        print("Copying and SHA-verifying selected artifact:", source.name)
                        digest = hashlib.sha256()
                        with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
                            while True:
                                block = source_handle.read(16 * 2**20)
                                if not block:
                                    break
                                digest.update(block)
                                destination_handle.write(block)
                        if digest.hexdigest() != row["sha256"]:
                            raise ValueError(f"Catalog SHA-256 mismatch while copying: {source}")
                        if destination.stat().st_size != int(row["size_bytes"]):
                            raise ValueError(f"Copied byte-size mismatch: {destination}")
                        cache_valid_by_basename[source.name] = True
                else:
                    destination = source
                    print("SHA-verifying selected Drive artifact:", source.name)
                    if streaming_sha256(destination) != row["sha256"]:
                        raise ValueError(f"Catalog SHA-256 mismatch: {source}")
                    # The runners always receive LOCAL_DATA_DIR.  In no-cache
                    # mode expose the verified Drive file there through a
                    # symlink instead of silently leaving the dataset absent.
                    local_link = LOCAL_DATA_DIR / source.name
                    if not (
                        local_link.is_symlink()
                        and local_link.resolve() == source.resolve()
                    ):
                        if local_link.exists() or local_link.is_symlink():
                            local_link.unlink()
                        local_link.symlink_to(source)
                    destination = local_link
                staged[row["name"]] = destination
                if row["role"] == "master_npz" and destination not in inspected_masters:
                    inspect_stored_npz(destination)
                    inspected_masters.add(destination)

            # Runner 要求 metadata 与 NPZ 同 stem；若 catalog 保存的是非规范文件名，建立本地规范别名。
            for dataset_id, dataset in catalog.get("datasets", {}).items():
                names = set(dataset.get("artifact_names", []))
                if not names.intersection(selected_names):
                    continue
                master_name = f"{dataset_id}:master"
                metadata_name = f"{dataset_id}:metadata"
                if master_name in staged and metadata_name in staged:
                    canonical_json = LOCAL_DATA_DIR / (staged[master_name].stem + ".json")
                    metadata_sha256 = streaming_sha256(staged[metadata_name])
                    if (
                        not canonical_json.is_file()
                        or streaming_sha256(canonical_json) != metadata_sha256
                    ):
                        shutil.copy2(staged[metadata_name], canonical_json)

            # Selected catalog artifacts are already staged above. Synthetic
            # files absent there are generated below, matching the archived notebook.
            direct_imported_by_basename = {}

            def find_unique_drive_cached_file(filename):
                mydrive_root = DRIVE_PROJECT_ROOT.parent
                candidate_dirs = [
                    mydrive_root,
                    DRIVE_DATA_ROOT,
                    DRIVE_PROJECT_ROOT / "benchmark_dataset_cache",
                    mydrive_root / "EFGP_Eigenpro" / "benchmark_dataset_cache",
                    mydrive_root / "Colab_Experiments/EFGP_Eigenpro/benchmark_dataset_cache",
                    mydrive_root / "benchmark_dataset_cache",
                ]
                for base in candidate_dirs:
                    direct = Path(base) / filename
                    if direct.is_file():
                        return direct
                matches = sorted(
                    path for path in mydrive_root.rglob(filename) if path.is_file()
                )
                if len(matches) > 1:
                    raise RuntimeError(
                        f"Ambiguous Drive cache basename {filename!r}: "
                        + ", ".join(str(path) for path in matches)
                    )
                return matches[0] if matches else None

            def copy_and_hash_drive_artifact(source, destination):
                expected_size = int(source.stat().st_size)
                source_sha = None
                if destination.is_file() and destination.stat().st_size == expected_size:
                    source_sha = streaming_sha256(source)
                    if streaming_sha256(destination) == source_sha:
                        return source_sha
                existing_size = destination.stat().st_size if destination.is_file() else 0
                additional_bytes = max(expected_size - int(existing_size), 0)
                if additional_bytes > shutil.disk_usage("/content").free:
                    raise RuntimeError(
                        f"Direct-import artifact does not fit local disk: {source}"
                    )
                digest = hashlib.sha256()
                with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
                    while True:
                        block = source_handle.read(16 * 2**20)
                        if not block:
                            break
                        digest.update(block)
                        destination_handle.write(block)
                if destination.stat().st_size != expected_size:
                    raise RuntimeError(f"Direct-import byte-size mismatch: {destination}")
                copied_sha = digest.hexdigest()
                if source_sha is not None and copied_sha != source_sha:
                    raise RuntimeError(f"Direct-import SHA changed while copying: {source}")
                return copied_sha

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
            ## 5. 可选补救：补建缺失的 archived Synthetic / Manitowoc master

            正式流程与指定 archived notebook 一致：Synthetic 使用 `_ntrainN`，本地缺失时生成 noise=.3 数据。`GENERATE_ARCHIVED_SYNTHETIC_SIZES` 仍可为可选 legacy 路线额外请求 1M/3M 等规模。

            Manitowoc 300M 必须保持冻结 AOI、hash split、浅层优先顺序。LOD 8 只是起点；若精确容量不足 300M/75M，应提高到 LOD 9，不能用密度估算替代扫描结果。
            """
        )
    )
    cells.append(
        _code(
            r"""
            generated_before = {path.resolve() for path in LOCAL_DATA_DIR.iterdir()}
            formal_synthetic_missing_sizes = []
            if GENERATE_FORMAL_SYNTHETIC_IF_MISSING:
                for n_train in FORMAL_SCALE_SIZES:
                    stem = f"synthetic_true_func_2d_ntrain{int(n_train)}"
                    if not all(
                        (LOCAL_DATA_DIR / f"{stem}{suffix}").is_file()
                        for suffix in (".npz", ".json")
                    ):
                        formal_synthetic_missing_sizes.append(int(n_train))
            synthetic_sizes_to_generate = sorted({
                *formal_synthetic_missing_sizes,
                *(int(n) for n in GENERATE_ARCHIVED_SYNTHETIC_SIZES),
            })
            if synthetic_sizes_to_generate:
                sizes_arg = ",".join(str(int(n)) for n in synthetic_sizes_to_generate)
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
            SMOKE_ELAPSED_SECONDS = None
            if RUN_PLUMBING_SMOKE:
                smoke_started = time.perf_counter()
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
                SMOKE_ELAPSED_SECONDS = time.perf_counter() - smoke_started
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
                validated = []
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
                        "target_function": "true_func_2d",
                        "dim": 2,
                        "n_train": n_train,
                        "n_test": n_train // 4,
                    }
                    mismatches = {
                        key: (generation.get(key), value)
                        for key, value in expected.items()
                        if generation.get(key) != value
                    }
                    expected_shapes = {
                        "n_train": n_train,
                        "n_test": n_train // 4,
                        "dim": 2,
                    }
                    shapes = metadata.get("shapes", {})
                    shape_mismatches = {
                        key: (shapes.get(key), value)
                        for key, value in expected_shapes.items()
                        if shapes.get(key) != value
                    }
                    if metadata.get("dataset_name") != stem:
                        mismatches["dataset_name"] = (
                            metadata.get("dataset_name"), stem
                        )
                    if metadata.get("y_transform", {}).get("noise_std") != 0.3:
                        mismatches["y_transform.noise_std"] = (
                            metadata.get("y_transform", {}).get("noise_std"), 0.3
                        )
                    if shape_mismatches:
                        mismatches["shapes"] = shape_mismatches
                    if mismatches:
                        failures.append(f"{stem}: archived generation mismatch {mismatches}")
                    else:
                        def source_record(path):
                            record = selected_by_basename.get(
                                path.name, direct_imported_by_basename.get(path.name)
                            )
                            if record is not None:
                                return record
                            return {
                                "sha256": streaming_sha256(path),
                                "source_kind": "generated_local_if_missing",
                                "source": str(path),
                            }

                        npz_record = source_record(npz_path)
                        metadata_record = source_record(json_path)
                        validated.append({
                            "dataset_stem": stem,
                            "n_train": n_train,
                            "data_family": "true_func_2d_uniform_gaussian_noise_0.3",
                            "noise_std": generation["noise_std"],
                            "seed_train": generation["seed_train"],
                            "seed_test": generation["seed_test"],
                            "chunk_rows": generation["chunk_rows"],
                            "npz_sha256": npz_record["sha256"],
                            "metadata_sha256": metadata_record["sha256"],
                            "npz_source_kind": npz_record.get(
                                "source_kind", "drive_catalog"
                            ),
                            "metadata_source_kind": metadata_record.get(
                                "source_kind", "drive_catalog"
                            ),
                            "npz_source_reference": npz_record.get(
                                "source", npz_record.get("relative_path")
                            ),
                            "metadata_source_reference": metadata_record.get(
                                "source", metadata_record.get("relative_path")
                            ),
                        })
                if failures:
                    raise RuntimeError(
                        "Archived Synthetic inputs are incomplete or incompatible. "
                        "Generate the requested _ntrainN/noise=0.3 artifacts with the "
                        "frozen protocol before Stage 1:\n- "
                        + "\n- ".join(failures)
                    )
                return validated
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
            # Stage 1 — true end-to-end KRR（两族主报告 + 次级 broad matrix）

            本阶段比较的是**完整 KRR pipeline**，不是同一 Fourier 系统上的 preconditioner。主报告 profile 显式运行 `efgp-standard-cg`、`ours-binned-inverse`、`ours-binned-active-eig`、`efgp-standard-full-eig`；inverse 始终单列，后两者同属 active-box EigenPro 族并按当前 median train-total 选族内最快。次级 broad matrix 才加入 `nystrom-krr`、`rpcholesky-krr` 与 `efgp-standard-jacobi`。每条 pipeline 各自承担自己的 setup 与 solve。正式计时字段固定为：

            - `setup_seconds`：数据空间 landmarks / RPCholesky factor 或 EFGP Fourier/precompute setup；
            - `solving_phase_seconds`：score selection（若使用）+ preconditioner build + iterative solve；
            - `train_total_seconds = setup_seconds + solving_phase_seconds`；
            - prediction 单列，用于报告 RMSE/R² 和 usability，不并入训练加速。

            两族配置不在当前 timing 中扫参。每个 dataset/N 的 inverse top-k/box 与 active-eig top-k/box/rank 分别取自旧诊断 notebook 的 operation-level winners；full-grid EigenPro 也是 active-box EigenPro 族的候选。旧 timing 只导出到 appendix audit，正式 replay 全部按当前 1 次预热 + 5 次 measured 协议重新测量。Synthetic 数据引入严格匹配该 notebook：每个规模为 `_ntrainN`，缺失时使用 noise=.3、seeds 20260421/1、5M chunk、`N/4` test 本地生成。

            robustness 中冻结的 proposed top-k 是上界。若改变 lengthscale/dataset 后同一 score prefix 的中心闭包超过仍然冻结的 box budget，只有 robustness 配置会显式授权按同一确定性 score 顺序缩短到可容纳的最大 prefix；这不是扫参，不读取时间、迭代数、标签或精度。`configured_active_topk`、`effective_active_topk`、`effective_active_box_size`、`active_selection_rule` 与 `capacity_adapted` 都进入 canonical 表，必须披露实际运行配置。

            首先运行 `end_to_end_suite.json::scale_10m_300m`。target 选择规则在看结果前冻结：六方法行必须全部存在，Nyström/EFGP 必须成功；RPCholesky 的预声明显存 `resource_limit` 可以作为真实 scalability outcome 保留，但不能获得 speedup，也不阻断后续 solver target。ours/full-eig 必须落在每个 repeat 的宽松绝对 RMSE/R² usability 范围；1% relative equivalence 只作描述。standard EFGP-CG median iterations 位于 3000–6000，随后取其中最大的 N；同 N 按 suite 中预先声明的 `dataset_priority` 决定。没有合格 case 时 fail closed，Stage 1 robustness 与 Stage 2 均不启动。
            """
        )
    )
    cells.append(
        _code(
            r"""
            from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
                end_to_end_suite as stage1_suite,
            )
            from dataclasses import asdict

            STAGE1_SUITE_CONFIG = (
                LOCAL_REPO
                / "efgp_eigenpro_py/gpu/box_toeplitz_active_block/controlled/end_to_end_suite.json"
            )
            STAGE1_OUTPUT_ROOT = DRIVE_RUN_ROOT / "stage1_end_to_end_krr"
            STAGE1_RUNTIME_CONFIG_ROOT = DRIVE_RUN_ROOT / "runtime_configs" / "stage1"
            STAGE1_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            STAGE1_RUNTIME_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
            STAGE1_TARGET_PATH = STAGE1_OUTPUT_ROOT / "selected_target_regime.json"
            STAGE1_TARGET_REJECTIONS_PATH = (
                STAGE1_OUTPUT_ROOT / "target_regime_rejections.json"
            )
            stage1_config = stage1_suite.load_suite_config(STAGE1_SUITE_CONFIG)
            if list(stage1_config["base"].get("methods", [])) != list(STAGE1_METHODS):
                raise RuntimeError(
                    "Stage-1 suite method order differs from the frozen true-KRR method list."
                )
            synthetic_family_manifest_path = (
                STAGE1_OUTPUT_ROOT / "synthetic_data_family_manifest.json"
            )
            stage1_synthetic_inputs = []
            if RUN_STAGE1_END_TO_END_KRR:
                stage1_synthetic_inputs = validate_archived_synthetic_inputs(FORMAL_SCALE_SIZES)
                if synthetic_family_manifest_path.is_file():
                    prior_synthetic_inputs = json.loads(
                        synthetic_family_manifest_path.read_text(encoding="utf-8")
                    )
                    if prior_synthetic_inputs != stage1_synthetic_inputs:
                        raise RuntimeError(
                            "This run directory is already bound to different Synthetic "
                            "artifact hashes; change RUN_TAG_PREFIX instead of mixing data."
                        )
                else:
                    synthetic_family_manifest_partial = (
                        synthetic_family_manifest_path.with_suffix(".json.partial")
                    )
                    synthetic_family_manifest_partial.write_text(
                        json.dumps(stage1_synthetic_inputs, indent=2), encoding="utf-8"
                    )
                    synthetic_family_manifest_partial.replace(
                        synthetic_family_manifest_path
                    )

            stage1_scale_plan = stage1_suite.build_profile_plan(
                stage1_config,
                STAGE1_SCALE_PROFILE,
                dataset_dir=str(LOCAL_DATA_DIR),
                output_root=STAGE1_OUTPUT_ROOT,
            )
            planned_scale_sizes = sorted({int(item["config"].n_train) for item in stage1_scale_plan})
            if planned_scale_sizes != list(FORMAL_SCALE_SIZES):
                raise RuntimeError(
                    f"Stage-1 scale sizes {planned_scale_sizes} != frozen {FORMAL_SCALE_SIZES}."
                )
            stage1_family_scale_plan = stage1_suite.build_profile_plan(
                stage1_config,
                STAGE1_FAMILY_SCALE_PROFILE,
                dataset_dir=str(LOCAL_DATA_DIR),
                output_root=STAGE1_OUTPUT_ROOT,
            )
            stage1_family_kernel_plan = stage1_suite.build_profile_plan(
                stage1_config,
                STAGE1_FAMILY_KERNEL_PROFILE,
                dataset_dir=str(LOCAL_DATA_DIR),
                output_root=STAGE1_OUTPUT_ROOT,
            )
            if any(
                list(item["config"].methods) != list(STAGE1_FAMILY_METHODS)
                for item in [*stage1_family_scale_plan, *stage1_family_kernel_plan]
            ):
                raise RuntimeError("Family profiles differ from STAGE1_FAMILY_METHODS.")

            stage1_campaign_rows = []
            stage1_selected_case_records = []

            def normalize_stage1_config_value(value):
                if isinstance(value, Path):
                    return str(value)
                if isinstance(value, dict):
                    return {
                        str(key): normalize_stage1_config_value(item)
                        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                    }
                if isinstance(value, (list, tuple)):
                    return [normalize_stage1_config_value(item) for item in value]
                return value

            def declared_stage1_dataset_family(item):
                declared = item.get("dataset_family")
                if declared:
                    return str(declared)
                stem = str(item["config"].dataset_stem)
                lower = stem.lower()
                if "synthetic_true_func_2d" in lower:
                    return "Synthetic"
                if "winnebago" in lower:
                    return "Winnebago"
                if "usgs_ept_wi_2county_1_b23" in lower:
                    return "Manitowoc"
                raise RuntimeError(
                    f"Stage-1 dataset family is undeclared for {stem!r}; refusing inference-free reporting."
                )

            def load_matching_stage1_artifact(item):
                run_dir = Path(item["config"].output_dir)
                completion_path = run_dir / "run_complete.json"
                summary_path = run_dir / "pipeline_summary.csv"
                config_path = run_dir / "experiment_config.json"
                runs_path = run_dir / "pipeline_runs.csv"
                if not (
                    completion_path.is_file()
                    and summary_path.is_file()
                    and config_path.is_file()
                    and runs_path.is_file()
                ):
                    return None
                try:
                    completion = json.loads(completion_path.read_text(encoding="utf-8"))
                    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
                    summary = pd.read_csv(summary_path)
                except (OSError, json.JSONDecodeError, ValueError, pd.errors.ParserError):
                    return None
                expected_config = normalize_stage1_config_value(asdict(item["config"]))
                observed_config = normalize_stage1_config_value(saved_config)
                expected_methods = list(item["config"].methods)
                required_summary_columns = {
                    *stage1_suite.STAGE2_SYSTEM_CONFIG_FIELDS,
                    *stage1_suite.DATASET_PROVENANCE_CONFIG_FIELDS,
                    "observed_dataset_noise_std",
                    "observed_dataset_seed_train",
                    "observed_dataset_seed_test",
                    "observed_dataset_generation_chunk_rows",
                    "observed_dataset_target_function",
                    "dataset_content_index_sha256",
                    "dataset_metadata_sha256",
                    "accuracy_relative_tolerance",
                    "accuracy_max_rmse",
                    "accuracy_min_r2",
                    "expected_measured_repeats",
                    "accuracy_evaluated_repeats",
                    "accuracy_passed_repeats",
                    "execution_eligible",
                    "usability_evaluated_repeats",
                    "usability_passed_repeats",
                    "usability_eligible",
                    "reference_evaluated_repeats",
                    "reference_equivalent_repeats",
                    "reference_equivalent",
                    "quality_qualified_performance_eligible",
                    "configured_active_rank",
                    "configured_full_eig_rank",
                    "configured_active_topk",
                    "configured_expected_active_box_size",
                    "parameter_selection_policy",
                    "parameter_source",
                    "setup_seconds_at_median_total",
                    "solving_phase_seconds_at_median_total",
                }
                artifact_matches = bool(
                    completion.get("protocol_family") == "end_to_end_krr"
                    and completion.get("timing_scope") == stage1_suite.TIMING_SCOPE
                    and completion.get("artifact_complete") is True
                    and completion.get("all_rows_present") is True
                    and list(completion.get("methods", [])) == expected_methods
                    and list(saved_config.get("methods", [])) == expected_methods
                    and observed_config == expected_config
                    and required_summary_columns.issubset(summary.columns)
                    and set(summary.get("method", pd.Series(dtype=str)).astype(str))
                    == set(expected_methods)
                )
                return completion if artifact_matches else None

            def stage1_completion_is_reusable(item):
                completion = load_matching_stage1_artifact(item)
                return bool(
                    completion is not None
                    and not completion.get("error_methods")
                    and completion.get("formal_result_status") in {
                        "claim_eligible_complete",
                        "complete_with_resource_limits",
                        "complete_with_usability_ineligible_methods",
                    }
                    and "proposed_performance_claim_eligible" in completion
                )

            def run_stage1_items(items, *, profile_label):
                completed_items = []
                for item in items:
                    started = time.perf_counter()
                    run_dir = Path(item["config"].output_dir)
                    if (
                        int(item["config"].n_train) >= 300_000_000
                        and not bool(globals().get("CAN_RUN_300M", False))
                    ):
                        record = {
                            "profile": profile_label,
                            "case_id": item["case_id"],
                            "dataset_family": declared_stage1_dataset_family(item),
                            "robustness_axes": item.get("robustness_axes", []),
                            "run_dir": run_dir,
                        }
                        stage1_selected_case_records.append(record)
                        stage1_campaign_rows.append({
                            "job_id": f"stage1_{profile_label}_{item['case_id']}",
                            "profile": profile_label,
                            "dataset_family": declared_stage1_dataset_family(item),
                            "n_train": int(item["config"].n_train),
                            "mandatory": True,
                            "status": "SKIPPED_HARDWARE",
                            "reason": "300M hardware preflight failed",
                            "artifact_complete": False,
                            "scientific_eligible": False,
                            "ineligible_methods": "",
                            "resource_limit_methods": "",
                            "error_methods": "",
                            "case_count": 0,
                            "elapsed_seconds": time.perf_counter() - started,
                            "invocation_mode": "skipped",
                            "resumed_case_count": 0,
                            "executed_case_count": 0,
                        })
                        print(f"[Stage 1/{item['case_id']}] SKIPPED_HARDWARE")
                        continue
                    reused = stage1_completion_is_reusable(item)
                    status = "execution_error"
                    reason = ""
                    artifact_complete = False
                    scientific_eligible = False
                    ineligible_methods = []
                    resource_limit_methods = []
                    error_methods = []
                    try:
                        if not reused:
                            runtime_base = asdict(item["config"])
                            runtime_base.pop("dataset_dir", None)
                            runtime_base.pop("output_dir", None)
                            runtime_payload = {
                                "schema_version": stage1_config.get("schema_version", 1),
                                "protocol_family": "end_to_end_krr",
                                "base": runtime_base,
                                "profiles": {
                                    profile_label: {
                                        "cases": [{
                                            "id": item["case_id"],
                                            "dataset_family": declared_stage1_dataset_family(item),
                                        }]
                                    }
                                },
                            }
                            runtime_path = (
                                STAGE1_RUNTIME_CONFIG_ROOT
                                / f"{profile_label}_{item['case_id']}.json"
                            )
                            runtime_path.write_text(
                                json.dumps(runtime_payload, indent=2), encoding="utf-8"
                            )
                            completed = run_cmd([
                                sys.executable, "-m",
                                "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end_suite",
                                "--suite-config", str(runtime_path),
                                "--profile", profile_label,
                                "--dataset-dir", str(LOCAL_DATA_DIR),
                                "--output-root", str(STAGE1_OUTPUT_ROOT),
                                "--no-resume",
                            ], cwd=LOCAL_REPO, check=False)
                            if int(completed.returncode) != 0:
                                raise RuntimeError(
                                    f"end_to_end_suite exited with code {completed.returncode}"
                                )
                        completion_payload = load_matching_stage1_artifact(item)
                        if completion_payload is None:
                            raise RuntimeError("complete Stage-1 artifact set was not produced")
                        artifact_complete = True
                        completed_items.append(item)
                        status = str(completion_payload["formal_result_status"])
                        resource_limit_methods = [
                            str(method)
                            for method in completion_payload.get("resource_limit_methods", [])
                        ]
                        error_methods = [
                            str(method)
                            for method in completion_payload.get("error_methods", [])
                        ]
                        ineligible_methods = [
                            str(method)
                            for method in completion_payload.get(
                                "performance_ineligible_methods", []
                            )
                        ]
                        scientific_eligible = bool(
                            not error_methods
                            and status in {
                                "claim_eligible_complete",
                                "complete_with_resource_limits",
                                "complete_with_usability_ineligible_methods",
                            }
                        )
                        if resource_limit_methods:
                            reason = "declared resource-limit methods: " + ",".join(
                                resource_limit_methods
                            )
                        elif ineligible_methods:
                            reason = "performance-ineligible methods: " + ",".join(
                                ineligible_methods
                            )
                    except Exception as exc:
                        status = "execution_error"
                        reason = f"{type(exc).__name__}: {exc}"
                    elapsed = time.perf_counter() - started
                    record = {
                        "profile": profile_label,
                        "case_id": item["case_id"],
                        "dataset_family": declared_stage1_dataset_family(item),
                        "robustness_axes": item.get("robustness_axes", []),
                        "run_dir": run_dir,
                    }
                    stage1_selected_case_records.append(record)
                    stage1_campaign_rows.append({
                        "job_id": f"stage1_{profile_label}_{item['case_id']}",
                        "profile": profile_label,
                        "dataset_family": declared_stage1_dataset_family(item),
                        "n_train": int(item["config"].n_train),
                        "mandatory": True,
                        "status": status,
                        "reason": reason,
                        "artifact_complete": artifact_complete,
                        "scientific_eligible": scientific_eligible,
                        "ineligible_methods": ",".join(ineligible_methods),
                        "resource_limit_methods": ",".join(resource_limit_methods),
                        "error_methods": ",".join(error_methods),
                        "case_count": 1,
                        "elapsed_seconds": elapsed,
                        "invocation_mode": "resumed_existing" if reused else "executed",
                        "resumed_case_count": int(reused),
                        "executed_case_count": int(not reused),
                    })
                    print(f"[Stage 1/{item['case_id']}] {status}: {reason or ('resumed' if reused else 'executed')}")
                return completed_items

            completed_stage1_family_scale_items = []
            if RUN_STAGE1_FAMILY_SCALE:
                completed_stage1_family_scale_items = run_stage1_items(
                    stage1_family_scale_plan,
                    profile_label=STAGE1_FAMILY_SCALE_PROFILE,
                )

            completed_stage1_family_kernel_items = []
            if RUN_STAGE1_FAMILY_KERNEL:
                completed_stage1_family_kernel_items = run_stage1_items(
                    stage1_family_kernel_plan,
                    profile_label=STAGE1_FAMILY_KERNEL_PROFILE,
                )

            completed_stage1_scale_items = []
            if RUN_STAGE1_END_TO_END_KRR:
                completed_stage1_scale_items = run_stage1_items(
                    stage1_scale_plan, profile_label=STAGE1_SCALE_PROFILE
                )
            else:
                print("Stage 1 scale disabled in manual mode.")

            def collect_stage1_summaries(items):
                frames = []
                for item in items:
                    run_dir = Path(item["config"].output_dir)
                    path = run_dir / "pipeline_summary.csv"
                    if not path.is_file():
                        continue
                    frame = pd.read_csv(path)
                    cfg = item["config"]
                    frame["suite_profile"] = item["profile"]
                    frame["case_id"] = item["case_id"]
                    frame["declared_dataset_family"] = declared_stage1_dataset_family(item)
                    frame["dataset_family"] = declared_stage1_dataset_family(item)
                    robustness_axes = list(item.get("robustness_axes", []))
                    frame["robustness_axes"] = json.dumps(
                        robustness_axes, ensure_ascii=False
                    )
                    # Enriched campaign identity is authoritative for reporting;
                    # raw per-case summaries do not carry all profile/target fields.
                    frame["dataset_stem"] = cfg.dataset_stem
                    frame["n_train"] = int(cfg.n_train)
                    frame["kernel_family"] = cfg.kernel_family
                    frame["nu"] = float(cfg.nu)
                    frame["reg_lambda"] = float(cfg.reg_lambda)
                    frame["lengthscale"] = float(cfg.lengthscale)
                    frame["fourier_eps"] = float(cfg.fourier_eps)
                    frame["box_budget"] = int(cfg.box_budget)
                    frame["configured_active_rank"] = int(cfg.rank)
                    frame["configured_full_eig_rank"] = int(cfg.full_eig_rank or cfg.rank)
                    frame["configured_active_topk"] = cfg.active_topk
                    frame["configured_expected_active_box_size"] = (
                        cfg.expected_active_box_size
                    )
                    frame["configured_inverse_active_topk"] = cfg.inverse_active_topk
                    frame["configured_inverse_expected_active_box_size"] = (
                        cfg.inverse_expected_active_box_size
                    )
                    frame["configured_active_eig_topk"] = cfg.active_eig_topk
                    frame["configured_active_eig_expected_active_box_size"] = (
                        cfg.active_eig_expected_active_box_size
                    )
                    frame["configured_active_eig_rank"] = cfg.active_eig_rank
                    frame["parameter_selection_policy"] = str(
                        cfg.parameter_selection_policy
                    )
                    frame["parameter_source"] = str(cfg.parameter_source)
                    frame["run_dir"] = str(run_dir)
                    frames.append(frame)
                return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

            def select_two_family_rows(frame):
                # Select one explicit inverse row and one eigenpair-family row
                # per case using the current median method-owned train total.
                # Accuracy is never a binary selector; RMSE/R2 remain adjacent.
                if frame.empty:
                    return pd.DataFrame()
                selected = []
                for case_id, case in frame.groupby("case_id", sort=False):
                    case = case.copy()
                    case["train_total_seconds_median"] = pd.to_numeric(
                        case["train_total_seconds_median"], errors="coerce"
                    )
                    family_specs = (
                        ("EFGP-CG", ["efgp-standard-cg"]),
                        ("inverse", ["ours-binned-inverse"]),
                        (
                            "active-box-EigenPro",
                            ["ours-binned-active-eig", "efgp-standard-full-eig"],
                        ),
                    )
                    cg_rows = case.loc[
                        case["method"].astype(str).eq("efgp-standard-cg")
                    ]
                    cg_total = (
                        float(cg_rows.iloc[0]["train_total_seconds_median"])
                        if len(cg_rows) == 1
                        else np.nan
                    )
                    for family, candidates in family_specs:
                        pool = case.loc[
                            case["method"].astype(str).isin(candidates)
                            & case["status"].astype(str).eq("ok")
                            & case["train_total_seconds_median"].notna()
                        ].sort_values(
                            ["train_total_seconds_median", "method"],
                            kind="mergesort",
                        )
                        if pool.empty:
                            selected.append({
                                "case_id": case_id,
                                "reporting_family": family,
                                "status": "no_successful_candidate",
                                "family_candidate_methods": ",".join(candidates),
                                "selection_rule": "minimum current median train total among status=ok rows; no accuracy cutoff",
                            })
                            continue
                        row = pool.iloc[0].to_dict()
                        total = float(row["train_total_seconds_median"])
                        row.update({
                            "reporting_family": family,
                            "selected_method": str(row["method"]),
                            "family_candidate_methods": ",".join(candidates),
                            "family_candidate_count": int(len(pool)),
                            "selection_rule": "minimum current median train total among status=ok rows; no accuracy cutoff",
                            "speedup_vs_efgp_cg_ratio_of_medians": (
                                cg_total / total
                                if np.isfinite(cg_total) and total > 0
                                else np.nan
                            ),
                        })
                        selected.append(row)
                return pd.DataFrame(selected)

            stage1_family_scale_all = collect_stage1_summaries(
                completed_stage1_family_scale_items
            )
            stage1_family_scale_selected = select_two_family_rows(
                stage1_family_scale_all
            )
            STAGE1_FAMILY_SCALE_ALL_PATH = (
                STAGE1_OUTPUT_ROOT / "stage1_family_scale_all_methods.csv"
            )
            STAGE1_FAMILY_SCALE_SELECTED_PATH = (
                STAGE1_OUTPUT_ROOT / "stage1_family_scale_selected.csv"
            )
            stage1_family_scale_all.to_csv(STAGE1_FAMILY_SCALE_ALL_PATH, index=False)
            stage1_family_scale_selected.to_csv(
                STAGE1_FAMILY_SCALE_SELECTED_PATH, index=False
            )

            stage1_family_kernel_all = collect_stage1_summaries(
                completed_stage1_family_kernel_items
            )
            stage1_family_kernel_selected = select_two_family_rows(
                stage1_family_kernel_all
            )
            STAGE1_FAMILY_KERNEL_ALL_PATH = (
                STAGE1_OUTPUT_ROOT / "stage1_family_kernel_all_methods.csv"
            )
            STAGE1_FAMILY_KERNEL_SELECTED_PATH = (
                STAGE1_OUTPUT_ROOT / "stage1_family_kernel_selected.csv"
            )
            stage1_family_kernel_all.to_csv(STAGE1_FAMILY_KERNEL_ALL_PATH, index=False)
            stage1_family_kernel_selected.to_csv(
                STAGE1_FAMILY_KERNEL_SELECTED_PATH, index=False
            )

            # Appendix evidence: retain the original operation-level selection,
            # the complete candidate rows, and a transparent two-family collapse.
            archived_matern_tables = (
                LOCAL_REPO
                / "efgp_eigenpro_py/gpu/box_toeplitz_active_block/outputs"
                / "btab_group_a_group_b_group_c_20260703_053504_export_bundle"
                / "btab_group_a_group_b_group_c_20260703_053504"
                / "diagnostic_tables"
            )
            archived_se_tables = (
                LOCAL_REPO
                / "efgp_eigenpro_py/gpu/box_toeplitz_active_block/outputs"
                / "btab_group_a_group_b_group_c_20260703_053504_export_bundle"
                / "btab_group_b_20260704_081848_export_bundle"
                / "btab_group_b_20260704_081848"
                / "diagnostic_tables"
            )
            archived_table_specs = (
                (archived_matern_tables, "matern"),
                (archived_se_tables, "se"),
            )
            archived_candidate_frames = []
            archived_selected_frames = []
            for table_root, kernel_key in archived_table_specs:
                candidate_path = table_root / "paper_table1_candidates.csv"
                selected_path = table_root / "paper_table1_selected.csv"
                if not candidate_path.is_file() or not selected_path.is_file():
                    raise FileNotFoundError(
                        f"Tracked archived table is missing: {table_root}"
                    )
                candidates = pd.read_csv(candidate_path)
                selected = pd.read_csv(selected_path)
                if kernel_key == "matern":
                    candidates = candidates.loc[
                        candidates["kernel_family"].astype(str).str.lower().eq("matern")
                    ]
                    selected = selected.loc[
                        selected["kernel_family"].astype(str).str.lower().eq("matern")
                    ]
                else:
                    candidates = candidates.loc[
                        candidates["kernel_family"].astype(str).str.lower().eq("se")
                    ]
                    selected = selected.loc[
                        selected["kernel_family"].astype(str).str.lower().eq("se")
                    ]
                candidates["archived_source_file"] = str(candidate_path)
                selected["archived_source_file"] = str(selected_path)
                archived_candidate_frames.append(candidates)
                archived_selected_frames.append(selected)

            archived_full_results = pd.concat(
                archived_candidate_frames, ignore_index=True, sort=False
            )
            archived_operation_selected = pd.concat(
                archived_selected_frames, ignore_index=True, sort=False
            )

            def archived_reporting_family(method_family):
                label = str(method_family)
                if label == "EFGP-CG":
                    return "EFGP-CG"
                if label == "Active inverse":
                    return "inverse"
                if label in {"Box-EigenPro", "EigenPro-PCG"}:
                    return "active-box-EigenPro"
                return "not_reported"

            archived_operation_selected["reporting_family"] = (
                archived_operation_selected["method_family"].map(
                    archived_reporting_family
                )
            )
            archived_operation_selected["time_total"] = pd.to_numeric(
                archived_operation_selected["time_total"], errors="coerce"
            )
            archived_regime_map_two_family = (
                archived_operation_selected.loc[
                    archived_operation_selected["reporting_family"].ne("not_reported")
                    & archived_operation_selected["time_total"].notna()
                ]
                .sort_values(
                    [
                        "dataset_stem", "kernel_family", "n_train",
                        "reporting_family", "time_total", "method",
                    ],
                    kind="mergesort",
                )
                .drop_duplicates(
                    ["dataset_stem", "kernel_family", "n_train", "reporting_family"],
                    keep="first",
                )
                .copy()
            )
            archived_regime_map_two_family["family_collapse_rule"] = (
                "after the original operation-level screen, choose minimum time_total; "
                "Box-EigenPro and full-grid EigenPro-PCG form one family"
            )
            ARCHIVED_FULL_RESULTS_PATH = (
                DRIVE_RUN_ROOT / "archived_original_full_results.csv"
            )
            ARCHIVED_OPERATION_SELECTED_PATH = (
                DRIVE_RUN_ROOT / "archived_original_operation_selected.csv"
            )
            ARCHIVED_REGIME_MAP_PATH = (
                DRIVE_RUN_ROOT / "archived_regime_map_two_family.csv"
            )
            archived_full_results.to_csv(ARCHIVED_FULL_RESULTS_PATH, index=False)
            archived_operation_selected.to_csv(
                ARCHIVED_OPERATION_SELECTED_PATH, index=False
            )
            archived_regime_map_two_family.to_csv(
                ARCHIVED_REGIME_MAP_PATH, index=False
            )
            (DRIVE_RUN_ROOT / "archived_selection_protocol.json").write_text(
                json.dumps({
                    "operation_level_rule": (
                        "For each dataset/kernel/N, retain converged candidates with "
                        "train RMSE <= 1.10 times the EFGP-CG train RMSE, then choose "
                        "minimum time_total separately for Active inverse, Box-EigenPro, "
                        "and full-grid EigenPro-PCG. Test RMSE is not a selector."
                    ),
                    "family_collapse_rule": (
                        "Keep the Active inverse winner as the inverse family; choose "
                        "the lower-time winner of Box-EigenPro and full-grid "
                        "EigenPro-PCG as the active-box-EigenPro family."
                    ),
                    "formal_replay_rule": (
                        "Archived timings are never copied into current five-repeat "
                        "formal tables; only the frozen configurations are transferred."
                    ),
                    "full_results_csv": str(ARCHIVED_FULL_RESULTS_PATH),
                    "operation_selected_csv": str(ARCHIVED_OPERATION_SELECTED_PATH),
                    "regime_map_csv": str(ARCHIVED_REGIME_MAP_PATH),
                }, indent=2),
                encoding="utf-8",
            )

            stage1_scale_summary = collect_stage1_summaries(completed_stage1_scale_items)
            STAGE1_SCALE_SUMMARY_PATH = STAGE1_OUTPUT_ROOT / "stage1_scale_summary.csv"
            stage1_scale_summary.to_csv(STAGE1_SCALE_SUMMARY_PATH, index=False)

            # Fail closed before target selection.  Presence of a summary file is
            # not evidence completeness: the canonical loader replays every raw
            # repeat, checks exact six-method/1+5 coverage, and recomputes all
            # timing and accuracy eligibility fields.
            from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled import (
                two_stage_reporting as canonical_reporting,
            )
            stage1_scale_verified_rows = []
            stage1_scale_verified = pd.DataFrame()
            if RUN_STAGE1_END_TO_END_KRR:
                if len(completed_stage1_scale_items) != len(stage1_scale_plan):
                    missing_scale_cases = sorted(
                        {item["case_id"] for item in stage1_scale_plan}
                        - {item["case_id"] for item in completed_stage1_scale_items}
                    )
                    raise RuntimeError(
                        "Stage-1 scale campaign is incomplete; refusing target "
                        f"selection and Stage 2. Missing={missing_scale_cases}"
                    )
                stage1_scale_verified_rows = canonical_reporting.load_stage1_summaries(
                    (STAGE1_SCALE_SUMMARY_PATH,)
                )
                stage1_scale_verified = pd.DataFrame(stage1_scale_verified_rows)

            END_TO_END_TARGET = None
            target_selection_error = ""
            if RUN_STAGE1_END_TO_END_KRR and stage1_scale_verified_rows:
                selection = stage1_config["target_selection"]
                try:
                    END_TO_END_TARGET = stage1_suite.select_target_regime(
                        stage1_scale_verified_rows,
                        required_methods=STAGE1_METHODS,
                        cg_iteration_min=int(selection["cg_iteration_min"]),
                        cg_iteration_max=int(selection["cg_iteration_max"]),
                        dataset_priority=selection.get("dataset_priority", ()),
                        allowed_resource_limit_methods=selection.get(
                            "allowed_resource_limit_methods", ("rpcholesky-krr",)
                        ),
                    )
                except RuntimeError as exc:
                    target_selection_error = str(exc)
                    rejection_payload = {
                        "status": "NO_ELIGIBLE_TARGET_FAIL_CLOSED",
                        "selection_rule": selection["rule"],
                        "error": target_selection_error,
                        "rejections": getattr(exc, "rejections", []),
                        "scale_summary_csv": str(STAGE1_SCALE_SUMMARY_PATH),
                    }
                    STAGE1_TARGET_REJECTIONS_PATH.write_text(
                        json.dumps(rejection_payload, indent=2), encoding="utf-8"
                    )
                    print(target_selection_error)
                else:
                    canonical_reporting._validate_stage1_scale_design(
                        stage1_scale_verified_rows,
                        END_TO_END_TARGET,
                        stage1_config,
                    )
                    STAGE1_TARGET_PATH.write_text(
                        json.dumps(END_TO_END_TARGET, indent=2), encoding="utf-8"
                    )
                    print("Frozen Stage-1 target:", END_TO_END_TARGET)

            stage1_robustness_plan = []
            completed_stage1_robustness_items = []
            if RUN_STAGE1_ROBUSTNESS and END_TO_END_TARGET is not None:
                stage1_robustness_plan = stage1_suite.materialize_robustness_plan(
                    stage1_config,
                    END_TO_END_TARGET,
                    dataset_dir=str(LOCAL_DATA_DIR),
                    output_root=STAGE1_OUTPUT_ROOT,
                )
                completed_stage1_robustness_items = run_stage1_items(
                    stage1_robustness_plan,
                    profile_label="robustness_at_selected_target",
                )
            elif RUN_STAGE1_ROBUSTNESS:
                print("Stage 1 robustness skipped: frozen target is unavailable.")

            stage1_family_robustness_plan = []
            completed_stage1_family_robustness_items = []
            if RUN_STAGE1_FAMILY_ROBUSTNESS and END_TO_END_TARGET is not None:
                stage1_family_robustness_plan = (
                    stage1_suite.materialize_family_robustness_plan(
                        stage1_config,
                        END_TO_END_TARGET,
                        dataset_dir=str(LOCAL_DATA_DIR),
                        output_root=STAGE1_OUTPUT_ROOT,
                    )
                )
                completed_stage1_family_robustness_items = run_stage1_items(
                    stage1_family_robustness_plan,
                    profile_label="family_robustness_at_selected_target",
                )
            elif RUN_STAGE1_FAMILY_ROBUSTNESS:
                print("Stage 1 family robustness skipped: frozen target is unavailable.")

            stage1_family_robustness_all = collect_stage1_summaries(
                completed_stage1_family_robustness_items
            )
            stage1_family_robustness_selected = select_two_family_rows(
                stage1_family_robustness_all
            )
            STAGE1_FAMILY_ROBUSTNESS_ALL_PATH = (
                STAGE1_OUTPUT_ROOT / "stage1_family_robustness_all_methods.csv"
            )
            STAGE1_FAMILY_ROBUSTNESS_SELECTED_PATH = (
                STAGE1_OUTPUT_ROOT / "stage1_family_robustness_selected.csv"
            )
            stage1_family_robustness_all.to_csv(
                STAGE1_FAMILY_ROBUSTNESS_ALL_PATH, index=False
            )
            stage1_family_robustness_selected.to_csv(
                STAGE1_FAMILY_ROBUSTNESS_SELECTED_PATH, index=False
            )

            stage1_robustness_summary = collect_stage1_summaries(
                completed_stage1_robustness_items
            )
            STAGE1_ROBUSTNESS_SUMMARY_PATH = (
                STAGE1_OUTPUT_ROOT / "stage1_robustness_summary.csv"
            )
            stage1_robustness_summary.to_csv(
                STAGE1_ROBUSTNESS_SUMMARY_PATH, index=False
            )
            stage1_robustness_verified_rows = []
            stage1_robustness_verified = pd.DataFrame()
            if RUN_STAGE1_ROBUSTNESS and END_TO_END_TARGET is not None:
                if len(completed_stage1_robustness_items) != len(stage1_robustness_plan):
                    missing_robustness_cases = sorted(
                        {item["case_id"] for item in stage1_robustness_plan}
                        - {item["case_id"] for item in completed_stage1_robustness_items}
                    )
                    raise RuntimeError(
                        "Stage-1 robustness campaign is incomplete; refusing Stage 2. "
                        f"Missing={missing_robustness_cases}"
                    )
                stage1_robustness_verified_rows = (
                    canonical_reporting.load_stage1_summaries(
                        (STAGE1_ROBUSTNESS_SUMMARY_PATH,)
                    )
                )
                stage1_robustness_verified = pd.DataFrame(
                    stage1_robustness_verified_rows
                )
                _, stage1_robustness_design_claims = (
                    canonical_reporting.build_stage1_robustness(
                        stage1_robustness_verified_rows,
                        END_TO_END_TARGET,
                        stage1_config,
                    )
                )
                design_claim = next(
                    claim for claim in stage1_robustness_design_claims
                    if claim["claim_id"] == "stage1_robustness_oat_design_complete"
                )
                if design_claim["status"] != "supported":
                    raise RuntimeError(
                        "Stage-1 robustness OAT design failed canonical validation; "
                        "refusing Stage 2."
                    )
            stage1_case_index = pd.DataFrame([
                {
                    **{key: value for key, value in record.items() if key != "run_dir"},
                    "run_dir": str(record["run_dir"]),
                }
                for record in stage1_selected_case_records
            ])
            STAGE1_CASE_INDEX_PATH = STAGE1_OUTPUT_ROOT / "stage1_case_index.csv"
            stage1_case_index.to_csv(STAGE1_CASE_INDEX_PATH, index=False)
            STAGE1_CASE_INDEX_PATH.with_suffix(".json").write_text(
                json.dumps(stage1_case_index.to_dict(orient="records"), indent=2),
                encoding="utf-8",
            )
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## Stage 1 报告：inverse 与 active-box EigenPro 分族、time–quality、setup / solving

            `stage1_family_scale_selected.csv`、`stage1_family_robustness_selected.csv` 和 `stage1_family_kernel_selected.csv` 是论文前三组表的直接输入：每个 case 固定输出 EFGP-CG、inverse 族、active-box EigenPro 族；后一族在 localized/full-grid 两条成功 current timing 中取 median total 最小者。所有成功的完整 KRR 行都保留时间与精度，RMSE/R² 不充当 speed gate。次级 broad matrix 和 Stage 2 fixed-system 数据均保持独立。
            """
        )
    )
    cells.append(
        _code(
            r"""
            import matplotlib.pyplot as plt
            import numpy as np

            STAGE1_METHOD_ORDER = list(STAGE1_METHODS)
            STAGE1_GENERATED_PLOT_PATHS = []

            # Primary paper view: keep the inverse and active-box-EigenPro
            # branches separate.  The latter includes the full-grid limit.
            if not stage1_family_scale_selected.empty:
                family_scale_plot = stage1_family_scale_selected.copy()
                family_scale_plot["N"] = pd.to_numeric(
                    family_scale_plot["n_train"], errors="coerce"
                )
                family_scale_plot["train_total_seconds_median"] = pd.to_numeric(
                    family_scale_plot["train_total_seconds_median"], errors="coerce"
                )
                families = list(dict.fromkeys(
                    family_scale_plot["dataset_family"].dropna().astype(str)
                ))
                fig, axes = plt.subplots(
                    1, max(len(families), 1),
                    figsize=(6.5 * max(len(families), 1), 4.8),
                    squeeze=False,
                )
                for axis, dataset_family in zip(axes.ravel(), families):
                    subset = family_scale_plot.loc[
                        family_scale_plot["dataset_family"].astype(str).eq(dataset_family)
                    ]
                    for reporting_family in (
                        "EFGP-CG", "inverse", "active-box-EigenPro"
                    ):
                        rows = subset.loc[
                            subset["reporting_family"].eq(reporting_family)
                        ].sort_values("N")
                        if rows.empty:
                            continue
                        axis.plot(
                            rows["N"], rows["train_total_seconds_median"],
                            marker="o", label=reporting_family,
                        )
                    axis.set_xscale("log")
                    axis.set_yscale("log")
                    axis.set_title(f"{dataset_family}: two-family KRR comparison")
                    axis.set_xlabel("training samples N")
                    axis.set_ylabel("setup + solving, seconds")
                    axis.grid(True, which="both", alpha=0.25)
                if families:
                    axes.ravel()[0].legend(frameon=False)
                fig.tight_layout()
                primary_family_plot = (
                    DRIVE_RUN_ROOT / "stage1_two_family_scale_10m_300m.png"
                )
                fig.savefig(primary_family_plot, dpi=180, bbox_inches="tight")
                STAGE1_GENERATED_PLOT_PATHS.append(primary_family_plot)
                plt.show()
                plt.close(fig)
                display(stage1_family_scale_selected[[
                    "dataset_family", "n_train", "reporting_family",
                    "selected_method", "configured_inverse_active_topk",
                    "configured_active_eig_topk", "configured_active_eig_rank",
                    "effective_active_box_size", "effective_active_rank",
                    "setup_seconds_at_median_total",
                    "solving_phase_seconds_at_median_total",
                    "train_total_seconds_median", "test_rmse_median",
                    "test_r2_median", "speedup_vs_efgp_cg_ratio_of_medians",
                ]])

            if not stage1_family_kernel_selected.empty:
                family_kernel_plot = stage1_family_kernel_selected.loc[
                    stage1_family_kernel_selected["reporting_family"].isin(
                        ["inverse", "active-box-EigenPro"]
                    )
                ].copy()
                family_kernel_plot["train_total_seconds_median"] = pd.to_numeric(
                    family_kernel_plot["train_total_seconds_median"], errors="coerce"
                )
                display(family_kernel_plot[[
                    "dataset_family", "kernel_family", "reporting_family",
                    "selected_method", "effective_active_box_size",
                    "effective_active_rank", "train_total_seconds_median",
                    "test_rmse_median", "test_r2_median",
                    "speedup_vs_efgp_cg_ratio_of_medians",
                ]].sort_values([
                    "dataset_family", "kernel_family", "reporting_family"
                ]))

            if not stage1_family_robustness_selected.empty:
                family_robustness_plot = stage1_family_robustness_selected.loc[
                    stage1_family_robustness_selected["reporting_family"].isin(
                        ["inverse", "active-box-EigenPro"]
                    )
                ].copy()
                family_robustness_plot.to_csv(
                    DRIVE_RUN_ROOT / "stage1_two_family_robustness.csv", index=False
                )
                display(family_robustness_plot[[
                    "case_id", "robustness_axes", "dataset_family",
                    "reg_lambda", "lengthscale", "box_budget",
                    "reporting_family", "selected_method",
                    "effective_active_box_size", "effective_active_rank",
                    "train_total_seconds_median", "test_rmse_median",
                    "test_r2_median", "speedup_vs_efgp_cg_ratio_of_medians",
                ]])

            if not stage1_scale_verified.empty:
                observed_protocols = set(stage1_scale_verified["protocol_family"].astype(str))
                observed_methods = set(stage1_scale_verified["method"].astype(str))
                if observed_protocols != {"end_to_end_krr"}:
                    raise RuntimeError(f"Stage-1 report saw mixed protocols: {observed_protocols}")
                if not observed_methods.issubset(set(STAGE1_METHOD_ORDER)):
                    raise RuntimeError(
                        f"Stage-1 report saw non-KRR methods: {sorted(observed_methods - set(STAGE1_METHOD_ORDER))}"
                    )

                # Plot only canonical values recomputed from pipeline_runs.csv;
                # never trust persisted derived speedup/reference columns.
                scale = stage1_scale_verified.copy()
                scale["dataset_family"] = scale["dataset"].astype(str)
                scale["N"] = pd.to_numeric(scale["n_train"], errors="coerce")
                for column in (
                    "train_total_seconds", "setup_seconds",
                    "solving_phase_seconds", "test_rmse", "test_r2",
                    "speedup_vs_ours",
                ):
                    scale[column] = pd.to_numeric(scale[column], errors="coerce")
                complete_mask = scale["status"].astype(str).eq("ok")
                complete_scale = scale.loc[complete_mask].copy()

                families = list(dict.fromkeys(scale["dataset_family"].dropna().astype(str)))
                fig, axes = plt.subplots(
                    1, max(len(families), 1), figsize=(7 * max(len(families), 1), 5),
                    squeeze=False,
                )
                for axis, family in zip(axes.ravel(), families):
                    family_rows = complete_scale.loc[complete_scale["dataset_family"].eq(family)]
                    for method in STAGE1_METHOD_ORDER:
                        rows = family_rows.loc[family_rows["method"].eq(method)].sort_values("N")
                        if rows.empty:
                            continue
                        axis.plot(
                            rows["N"], rows["train_total_seconds"],
                            marker="o", label=method,
                        )
                    axis.set_xscale("log")
                    axis.set_yscale("log")
                    axis.set_title(f"{family}: all successful complete KRR pipelines")
                    axis.set_xlabel("training samples N")
                    axis.set_ylabel("train total (setup + solving), seconds")
                    axis.grid(True, which="both", alpha=0.25)
                if families:
                    axes.ravel()[0].legend(fontsize=8)
                fig.tight_layout()
                total_plot = DRIVE_RUN_ROOT / "stage1_krr_train_total_10m_300m.png"
                fig.savefig(total_plot, dpi=180, bbox_inches="tight")
                STAGE1_GENERATED_PLOT_PATHS.append(total_plot)
                plt.show()
                plt.close(fig)

                reference_rmse = scale.loc[
                    scale["method"].eq("efgp-standard-full-eig"),
                    ["case_id", "test_rmse"],
                ].rename(columns={"test_rmse": "recomputed_reference_rmse"})
                if reference_rmse["case_id"].duplicated().any():
                    raise RuntimeError("duplicate full-eig accuracy references")
                scale = scale.merge(reference_rmse, on="case_id", how="left", validate="many_to_one")
                scale["rmse_ratio_to_full_eig"] = (
                    scale["test_rmse"] / scale["recomputed_reference_rmse"]
                )
                ours_components = scale.loc[
                    scale["method"].eq("ours-binned-default"),
                    ["case_id", "setup_seconds", "solving_phase_seconds", "train_total_seconds"],
                ].rename(columns={
                    "setup_seconds": "ours_setup_seconds",
                    "solving_phase_seconds": "ours_solving_phase_seconds",
                    "train_total_seconds": "ours_train_total_seconds",
                })
                if ours_components["case_id"].duplicated().any():
                    raise RuntimeError("duplicate proposed-pipeline timing references")
                scale = scale.merge(
                    ours_components, on="case_id", how="left", validate="many_to_one"
                )
                fig, ax = plt.subplots(figsize=(9, 5))
                for (family, method), rows in scale.groupby(["dataset_family", "method"], dropna=False):
                    rows = rows.sort_values("N")
                    ax.plot(
                        rows["N"], rows["rmse_ratio_to_full_eig"], marker="o",
                        label=f"{family} / {method}",
                    )
                ax.axhline(
                    1.0 + float(stage1_config["base"]["accuracy_relative_tolerance"]),
                    color="black", linestyle="--",
                    label="1% reference-equivalence guide (descriptive)",
                )
                ax.set_xscale("log")
                ax.set_xlabel("training samples N")
                ax.set_ylabel("test RMSE / standard EFGP full-eig RMSE")
                ax.set_title("Stage 1 RMSE ratio: descriptive quality trade-off, not a speed gate")
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(fontsize=7, ncol=2)
                fig.tight_layout()
                accuracy_plot = DRIVE_RUN_ROOT / "stage1_krr_accuracy_tradeoff.png"
                fig.savefig(accuracy_plot, dpi=180, bbox_inches="tight")
                STAGE1_GENERATED_PLOT_PATHS.append(accuracy_plot)
                plt.show()
                plt.close(fig)

                if END_TO_END_TARGET is not None:
                    target_rows = complete_scale.loc[
                        complete_scale["dataset_stem"].astype(str).eq(str(END_TO_END_TARGET["dataset_stem"]))
                        & complete_scale["N"].eq(int(END_TO_END_TARGET["n_train"]))
                    ].copy()
                    target_rows["method_order"] = target_rows["method"].map(
                        {method: index for index, method in enumerate(STAGE1_METHOD_ORDER)}
                    )
                    target_rows = target_rows.sort_values("method_order")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    x = np.arange(len(target_rows))
                    setup = target_rows["setup_seconds"].to_numpy(dtype=float)
                    solving = target_rows["solving_phase_seconds"].to_numpy(dtype=float)
                    ax.bar(x, setup, label="setup")
                    ax.bar(x, solving, bottom=setup, label="solving phase")
                    ax.set_xticks(x, target_rows["method"], rotation=30, ha="right")
                    ax.set_ylabel("seconds")
                    ax.set_title("Stage 1 selected target: complete-KRR time decomposition")
                    ax.legend()
                    ax.grid(True, axis="y", alpha=0.25)
                    fig.tight_layout()
                    split_plot = DRIVE_RUN_ROOT / "stage1_krr_setup_solving_breakdown.png"
                    fig.savefig(split_plot, dpi=180, bbox_inches="tight")
                    STAGE1_GENERATED_PLOT_PATHS.append(split_plot)
                    plt.show()
                    plt.close(fig)

                stage1_accuracy_audit = scale[[
                    "protocol_family", "suite_profile", "case_id", "dataset_family", "dataset_stem", "N",
                    "method", "status", "execution_eligible", "usability_eligible",
                    "reference_equivalent", "quality_qualified_performance_eligible",
                    "speedup_claim_eligible", "speedup_complete_pairing", "pareto_nondominated",
                    "test_rmse", "test_r2", "rmse_relative_delta_to_reference",
                    "recomputed_reference_rmse", "rmse_ratio_to_full_eig",
                    "setup_seconds", "solving_phase_seconds", "train_total_seconds",
                    "setup_speedup_vs_ours", "solving_speedup_vs_ours", "speedup_vs_ours",
                ]].copy()
                stage1_accuracy_audit.to_csv(
                    DRIVE_RUN_ROOT / "stage1_krr_accuracy_timing_audit.csv", index=False
                )
                display(stage1_accuracy_audit)
            else:
                print("No completed Stage-1 scale rows; formal KRR plots skipped.")

            if not stage1_robustness_verified.empty:
                robustness = stage1_robustness_verified.copy()
                robustness["speedup_vs_ours"] = pd.to_numeric(
                    robustness["speedup_vs_ours"], errors="coerce"
                )
                robustness.to_csv(
                    DRIVE_RUN_ROOT / "stage1_krr_robustness_all_axes.csv", index=False
                )
                case_order = list(dict.fromkeys(robustness["case_id"].astype(str)))
                fig, ax = plt.subplots(figsize=(10, 4.5))
                for method in STAGE1_METHOD_ORDER:
                    if method == "ours-binned-default":
                        continue
                    rows = robustness.loc[robustness["method"].eq(method)].copy()
                    rows = rows.loc[
                        rows["speedup_complete_pairing"].astype(str).str.lower().eq("true")
                        & rows["speedup_vs_ours"].notna()
                    ]
                    rows["case_order"] = rows["case_id"].astype(str).map(
                        {case_id: index for index, case_id in enumerate(case_order)}
                    )
                    rows = rows.sort_values("case_order")
                    ax.plot(
                        rows["case_order"], rows["speedup_vs_ours"],
                        marker="o", label=method,
                    )
                    outside = rows.loc[
                        ~rows["speedup_claim_eligible"].astype(str).str.lower().eq("true")
                    ]
                    if not outside.empty:
                        ax.scatter(
                            outside["case_order"], outside["speedup_vs_ours"],
                            marker="x", color="black", s=45, zorder=4,
                        )
                ax.axhline(1.0, color="black", linestyle="--")
                ax.set_xticks(np.arange(len(case_order)), case_order, rotation=45, ha="right")
                ax.set_ylabel("baseline train total / ours train total")
                ax.set_title(
                    "Stage 1 robustness: all paired speedups (x = outside broad usable range)"
                )
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(fontsize=8, ncol=2)
                fig.tight_layout()
                robustness_plot = DRIVE_RUN_ROOT / "stage1_krr_robustness.png"
                fig.savefig(robustness_plot, dpi=180, bbox_inches="tight")
                STAGE1_GENERATED_PLOT_PATHS.append(robustness_plot)
                plt.show()
                plt.close(fig)
            print("Stage-1 plot paths:", [str(path) for path in STAGE1_GENERATED_PLOT_PATHS])
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            # Stage 2 — fixed identical hashed \(A,b\) solver/preconditioner comparison

            只有 Stage 1 target 成功冻结后，本阶段才在同一 invocation 中构造一个不可变 Fourier 系统；所有方法共享 weights/Gf/RHS/λ 哈希，每次从零初值开始，1 次预热后做 5 次随机顺序配对。正式 headline 是 `solver_total_seconds = score selection + preconditioner construction + CG/PCG solve`。

            正式 mandatory 方法为 CG、Jacobi、proposed default、active-eig 与 full-eig。`active-inverse` 只在 Stage 2 计时开始前，根据冻结 target 的预声明 active-box upper bound 是否不超过显式 inverse cap 判为可行时才执行；决定与 benchmark 的 15 个 system-defining 字段和冻结方法配置一并写入 `stage2_feasibility.json`，不得依据 timing outcome 回填。`default` 是部署路由标签，并继续使用 Stage 1 冻结的较小 `default_inverse_max_size`，因此不会因为放宽显式 inverse 的可行性 cap 而改变算法；图表同时显示实际 `method_kind`。若显式 active route 与 default 的 kind 相同，它只是独立重复的 route-control，不是另一种算法。`fourier-nystrom-precond` / `fourier-rpcholesky-precond` 是可选 exploratory Fourier adaptations，默认不运行，也绝不能标成 Stage 1 的 Nyström/RPCholesky KRR。旧 `paper_10m`、OAT、fixed-system scale profiles 仍可在 `RUN_ALL_FORMAL_EXPERIMENTS=False` 时手动启用。
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
            STAGE2_FEASIBILITY_PATH = DRIVE_RUN_ROOT / "stage2_feasibility.json"
            CONTROLLED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            RUNTIME_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
            expected_controlled_case_count = 0
            selected_case_records = []
            base_suite = json.loads(SUITE_TEMPLATE.read_text(encoding="utf-8"))
            controlled_profile_outputs = []
            STAGE2_FEASIBILITY = None
            STAGE2_FEASIBLE_METHODS = []
            STAGE2_SYSTEM_CONFIG_FIELDS = (
                "dataset_stem", "n_train", "subset_seed", "subset_mode",
                "kernel_family", "lengthscale", "nu", "variance",
                "reg_lambda", "fourier_eps", "nufft_tol", "l2_scaled",
                "precision", "nufft_backend", "precompute_chunk_size",
            )
            STAGE2_METHOD_CONFIG_FIELDS = (
                "rank", "full_eig_rank", "active_topk",
                "expected_active_box_size",
                "allow_frozen_topk_capacity_adaptation",
                "box_budget", "inverse_max_size",
                "default_inverse_max_size",
                "parameter_selection_policy", "parameter_source",
            )

            def build_prospective_stage2_feasibility(
                frozen_target, *, box_budget, inverse_max_size,
                default_inverse_max_size, active_box_upper_bound,
            ):
                box_budget = int(box_budget)
                inverse_max_size = int(inverse_max_size)
                default_inverse_max_size = int(default_inverse_max_size)
                active_box_upper_bound = int(active_box_upper_bound)
                if (
                    box_budget <= 0
                    or inverse_max_size <= 0
                    or default_inverse_max_size <= 0
                ):
                    raise RuntimeError(
                        "Stage-2 declared box/inverse caps must be positive."
                    )
                candidates = list(STAGE2_METHODS)
                mandatory = list(STAGE2_MANDATORY_METHODS)
                if set(candidates) != set([*mandatory, "active-inverse"]):
                    raise RuntimeError(
                        "Frozen Stage-2 candidates must be the five mandatory methods "
                        "plus conditional active-inverse."
                    )
                inverse_feasible = active_box_upper_bound <= inverse_max_size
                methods = {
                    method: {
                        "feasible": True,
                        "reason": "prospectively mandatory fixed-A,b method",
                    }
                    for method in mandatory
                }
                methods["active-inverse"] = {
                    "feasible": bool(inverse_feasible),
                    "reason": (
                        "declared active-box upper bound is within inverse_max_size"
                        if inverse_feasible else
                        "declared active-box upper bound exceeds inverse_max_size; "
                        "exact inverse is prospectively omitted"
                    ),
                }
                missing = [
                    field for field in STAGE2_SYSTEM_CONFIG_FIELDS
                    if frozen_target.get(field) is None
                ]
                missing.extend(
                    field for field in STAGE2_METHOD_CONFIG_FIELDS
                    if field not in {"inverse_max_size", "default_inverse_max_size"}
                    and frozen_target.get(field) is None
                )
                if missing:
                    raise RuntimeError(
                        "Frozen target is missing Stage-2 feasibility fields: "
                        + ",".join(missing)
                    )
                return {
                    "schema_version": 1,
                    "protocol_family": "controlled_fixed_system",
                    "decision_basis": (
                        "prospective declared active-box upper bound before timing"
                    ),
                    **{
                        field: frozen_target[field]
                        for field in STAGE2_SYSTEM_CONFIG_FIELDS
                    },
                    **{
                        field: frozen_target[field]
                        for field in STAGE2_METHOD_CONFIG_FIELDS
                        if field not in {
                            "box_budget", "inverse_max_size",
                            "default_inverse_max_size",
                        }
                    },
                    "box_budget": box_budget,
                    "active_box_upper_bound": active_box_upper_bound,
                    "inverse_max_size": inverse_max_size,
                    "default_inverse_max_size": default_inverse_max_size,
                    "default_resolved_kind": (
                        "active-inverse"
                        if active_box_upper_bound <= default_inverse_max_size
                        else "active-eig"
                    ),
                    "methods": methods,
                }

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
                frozen_target = globals().get("END_TO_END_TARGET")
                if RUN_STAGE2_FIXED_AB_SOLVERS and frozen_target is not None:
                    target_n = int(frozen_target["n_train"])
                    target_stem = str(frozen_target["dataset_stem"])
                    frozen_stage1_suite = globals().get("stage1_config")
                    if not isinstance(frozen_stage1_suite, dict):
                        raise RuntimeError(
                            "Stage-2 feasibility requires the frozen Stage-1 suite config."
                        )
                    declared_scale_cases = frozen_stage1_suite["profiles"][
                        STAGE1_SCALE_PROFILE
                    ]["cases"]
                    matching_declared_cases = [
                        case for case in declared_scale_cases
                        if str(case.get("dataset_stem", frozen_stage1_suite["base"].get("dataset_stem")))
                        == target_stem
                        and int(case.get("n_train", frozen_stage1_suite["base"].get("n_train")))
                        == target_n
                    ]
                    if len(matching_declared_cases) != 1:
                        raise RuntimeError(
                            "Frozen target must match exactly one predeclared Stage-1 scale case."
                        )
                    declared_target_config = dict(frozen_stage1_suite["base"])
                    declared_target_config.update(matching_declared_cases[0])
                    target_box_budget = int(declared_target_config["box_budget"])
                    stage2_policy = frozen_stage1_suite.get("stage2_fixed_ab", {})
                    if not isinstance(stage2_policy, dict):
                        raise RuntimeError("stage2_fixed_ab policy must be an object.")
                    target_inverse_max_size = int(
                        stage2_policy.get(
                            "inverse_max_size",
                            declared_target_config["inverse_max_size"],
                        )
                    )
                    target_default_inverse_max_size = int(
                        stage2_policy.get(
                            "default_inverse_max_size",
                            declared_target_config["inverse_max_size"],
                        )
                    )
                    target_active_box_upper_bound = int(
                        declared_target_config.get("expected_active_box_size")
                        or declared_target_config["box_budget"]
                    )
                    STAGE2_FEASIBILITY = build_prospective_stage2_feasibility(
                        frozen_target,
                        box_budget=target_box_budget,
                        inverse_max_size=target_inverse_max_size,
                        default_inverse_max_size=target_default_inverse_max_size,
                        active_box_upper_bound=target_active_box_upper_bound,
                    )
                    STAGE2_FEASIBILITY_PATH.write_text(
                        json.dumps(STAGE2_FEASIBILITY, indent=2), encoding="utf-8"
                    )
                    STAGE2_FEASIBLE_METHODS = [
                        method for method in STAGE2_METHODS
                        if STAGE2_FEASIBILITY["methods"][method]["feasible"]
                    ]
                    if not set(STAGE2_MANDATORY_METHODS).issubset(
                        STAGE2_FEASIBLE_METHODS
                    ):
                        raise RuntimeError(
                            "Stage-2 prospective feasibility removed a mandatory method."
                        )
                    print(
                        "Prospective Stage-2 feasibility:", STAGE2_FEASIBILITY_PATH,
                        STAGE2_FEASIBLE_METHODS,
                    )
                    target_family_rows = globals().get(
                        "stage1_scale_summary", pd.DataFrame()
                    )
                    target_lower = target_stem.lower()
                    target_family = (
                        "Synthetic" if "synthetic_true_func_2d" in target_lower
                        else "Winnebago" if "winnebago" in target_lower
                        else "Manitowoc" if "usgs_ept_wi_2county_1_b23" in target_lower
                        else target_stem
                    )
                    if not target_family_rows.empty:
                        family_match = target_family_rows.loc[
                            target_family_rows["dataset_stem"].astype(str).eq(target_stem)
                            & pd.to_numeric(target_family_rows["n_train"], errors="coerce").eq(target_n)
                        ]
                        if not family_match.empty:
                            target_family = str(family_match.iloc[0]["dataset_family"])
                    target_profile_name = "fixed_ab_selected_target"
                    base_suite["profiles"][target_profile_name] = {
                        "description": (
                            "Stage 2 fixed identical hashed A,b at the prospectively selected "
                            "Stage-1 target; solver total includes selection/build+solve."
                        ),
                        "overrides": {
                            "methods": list(STAGE2_FEASIBLE_METHODS),
                            **{
                                field: frozen_target[field]
                                for field in STAGE2_SYSTEM_CONFIG_FIELDS
                                if field not in {"dataset_stem", "n_train"}
                            },
                            "box_budget": target_box_budget,
                            "inverse_max_size": target_inverse_max_size,
                            "default_inverse_max_size": (
                                target_default_inverse_max_size
                            ),
                            "rank": int(declared_target_config["rank"]),
                            "full_eig_rank": int(
                                declared_target_config.get("full_eig_rank")
                                or declared_target_config["rank"]
                            ),
                            "active_topk": declared_target_config.get("active_topk"),
                            "expected_active_box_size": declared_target_config.get(
                                "expected_active_box_size"
                            ),
                            "allow_frozen_topk_capacity_adaptation": bool(
                                declared_target_config.get(
                                    "allow_frozen_topk_capacity_adaptation", False
                                )
                            ),
                            "parameter_selection_policy": declared_target_config.get(
                                "parameter_selection_policy", ""
                            ),
                            "parameter_source": declared_target_config.get(
                                "parameter_source", ""
                            ),
                        },
                        "cases": [{
                            "id": f"fixed_ab_target_n{target_n}",
                            "dataset_family": target_family,
                            "dataset_stem": target_stem,
                            "n_train": target_n,
                            "subset_mode": frozen_target["subset_mode"],
                            "expected_n_train": target_n,
                            "scale_role": "Stage-2 solver comparison at frozen Stage-1 target",
                        }],
                    }
                    formal_jobs.append({
                        "job_id": "fixed_ab_selected_target",
                        "profile": target_profile_name,
                        "sizes": [target_n],
                        "families": [target_family],
                        "mandatory": True,
                        "planned_methods": list(STAGE2_FEASIBLE_METHODS),
                        "stage2_feasibility_path": str(STAGE2_FEASIBILITY_PATH),
                    })
                elif RUN_STAGE2_FIXED_AB_SOLVERS:
                    print("Stage 2 skipped: Stage 1 did not yield an eligible frozen target.")
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
            CAMPAIGN_EXECUTION_STARTED_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            CAMPAIGN_EXECUTION_STARTED_PERF_COUNTER = time.perf_counter()
            CAMPAIGN_JOBS_CSV = DRIVE_RUN_ROOT / "campaign_jobs.csv"
            CAMPAIGN_JOBS_JSON = DRIVE_RUN_ROOT / "campaign_jobs.json"
            previous_campaign_rows_by_job = {}
            if CAMPAIGN_JOBS_JSON.is_file():
                try:
                    previous_rows = json.loads(CAMPAIGN_JOBS_JSON.read_text(encoding="utf-8"))
                    if not isinstance(previous_rows, list) or not all(
                        isinstance(row, dict) for row in previous_rows
                    ):
                        raise TypeError("campaign ledger is not a list of objects")
                    previous_campaign_rows_by_job = {
                        str(row["job_id"]): row for row in previous_rows if row.get("job_id")
                    }
                except (OSError, json.JSONDecodeError, TypeError):
                    print("WARNING: previous campaign ledger is unreadable; first-run timing cannot be carried forward.")
            campaign_job_rows = list(globals().get("stage1_campaign_rows", []))
            planned_campaign_job_ids = (
                [str(row["job_id"]) for row in globals().get("stage1_campaign_rows", [])]
                +
                (["plumbing_smoke"] if RUN_PLUMBING_SMOKE else [])
                + [str(job["job_id"]) for job in formal_jobs]
                + (
                    ["prediction_accuracy_audit"]
                    if globals().get("RUN_PREDICTION_AUDIT", False) else []
                )
            )
            if RUN_PLUMBING_SMOKE:
                smoke_elapsed_seconds = globals().get("SMOKE_ELAPSED_SECONDS")
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
                    "elapsed_seconds": smoke_elapsed_seconds,
                    "current_invocation_elapsed_seconds": smoke_elapsed_seconds,
                    "elapsed_seconds_scope": "current smoke invocation wall time; never method timing",
                    "invocation_mode": "executed",
                    "resumed_case_count": 0,
                    "executed_case_count": 1,
                    "first_run_elapsed_seconds": (
                        smoke_elapsed_seconds if SMOKE_OK else None
                    ),
                    "first_run_elapsed_seconds_source": (
                        "current successful invocation"
                        if SMOKE_OK else "unavailable: smoke failed"
                    ),
                })

            def write_campaign_checkpoint():
                # Preserve first-run provenance for planned jobs that this
                # resumed invocation has not replayed yet. Otherwise a second
                # interruption after the first job would truncate the old
                # ledger to the current prefix.
                current_by_job = {
                    str(row["job_id"]): row
                    for row in campaign_job_rows if row.get("job_id")
                }
                checkpoint_rows = [
                    current_by_job.get(job_id)
                    or previous_campaign_rows_by_job.get(job_id)
                    for job_id in planned_campaign_job_ids
                    if (
                        job_id in current_by_job
                        or job_id in previous_campaign_rows_by_job
                    )
                ]
                checkpoint_rows.extend(
                    row for row in campaign_job_rows
                    if str(row.get("job_id")) not in set(planned_campaign_job_ids)
                )
                payload = json.dumps(checkpoint_rows, indent=2)
                partial_json = CAMPAIGN_JOBS_JSON.with_suffix(".json.partial")
                partial_json.write_text(payload, encoding="utf-8")
                partial_json.replace(CAMPAIGN_JOBS_JSON)
                frame = pd.DataFrame(checkpoint_rows)
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

            def execution_metadata_from_counts(
                *, case_count, resumed_count, elapsed_seconds,
                invocation_successful, previous_row=None,
            ):
                case_count = int(case_count)
                resumed_count = int(resumed_count)
                executed_count = max(case_count - resumed_count, 0)
                if case_count <= 0:
                    invocation_mode = "no_cases"
                elif resumed_count == case_count:
                    invocation_mode = "resumed_existing"
                elif resumed_count:
                    invocation_mode = "mixed_execute_and_resume"
                else:
                    invocation_mode = "executed"
                previous_row = previous_row if isinstance(previous_row, dict) else {}
                previous_first_run = (
                    previous_row.get("first_run_elapsed_seconds")
                    if previous_row.get("status") == "PASS" else None
                )
                if invocation_mode == "executed" and invocation_successful:
                    first_run_elapsed = float(elapsed_seconds)
                    first_run_source = "current successful invocation"
                elif previous_first_run is not None:
                    first_run_elapsed = float(previous_first_run)
                    first_run_source = "preserved successful campaign checkpoint"
                else:
                    first_run_elapsed = None
                    first_run_source = (
                        "unavailable: current/previous fresh invocation was not successful"
                    )
                return {
                    "elapsed_seconds": float(elapsed_seconds),
                    "current_invocation_elapsed_seconds": float(elapsed_seconds),
                    "elapsed_seconds_scope": (
                        "current suite invocation wall time; resume validation is not first-run "
                        "experiment time and neither value is method timing"
                    ),
                    "invocation_mode": invocation_mode,
                    "resumed_case_count": resumed_count,
                    "executed_case_count": executed_count,
                    "first_run_elapsed_seconds": first_run_elapsed,
                    "first_run_elapsed_seconds_source": first_run_source,
                }

            def suite_execution_metadata(
                status_rows, elapsed_seconds, job_status, previous_row=None,
            ):
                statuses = [str(row.get("status", "")) for row in status_rows]
                metadata = execution_metadata_from_counts(
                    case_count=len(statuses),
                    resumed_count=sum(
                        status.startswith("resumed_existing") for status in statuses
                    ),
                    elapsed_seconds=elapsed_seconds,
                    invocation_successful=(job_status == "PASS"),
                    previous_row=previous_row,
                )
                metadata["suite_case_statuses"] = statuses
                return metadata

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
                    "planned_methods": ",".join(job.get("planned_methods", [])),
                    "stage2_feasibility_path": job.get(
                        "stage2_feasibility_path", ""
                    ),
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
                    elapsed = time.perf_counter() - started
                    campaign_job_rows.append({
                        **base_row, "return_code": None, "status": skip_reason,
                        "reason": skip_reason, "case_count": 0,
                        "elapsed_seconds": elapsed,
                        "current_invocation_elapsed_seconds": elapsed,
                        "elapsed_seconds_scope": "gate evaluation only; no experiment executed",
                        "invocation_mode": "skipped",
                        "resumed_case_count": 0,
                        "executed_case_count": 0,
                        "first_run_elapsed_seconds": None,
                        "first_run_elapsed_seconds_source": "not executed",
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
                    elapsed = time.perf_counter() - started
                    campaign_job_rows.append({
                        **base_row, "return_code": None, "status": "CONFIG_ERROR",
                        "reason": "profile filter selected zero cases", "case_count": 0,
                        "elapsed_seconds": elapsed,
                        "current_invocation_elapsed_seconds": elapsed,
                        "elapsed_seconds_scope": "configuration validation only; no experiment executed",
                        "invocation_mode": "no_cases",
                        "resumed_case_count": 0,
                        "executed_case_count": 0,
                        "first_run_elapsed_seconds": None,
                        "first_run_elapsed_seconds_source": "not executed",
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
                    if not isinstance(status_rows, list) or not all(
                        isinstance(row, dict) for row in status_rows
                    ):
                        raise TypeError("suite status is not a list of objects")
                except (OSError, json.JSONDecodeError, TypeError):
                    status_rows = []
                job_status, reason = classify_suite_invocation(
                    completed.returncode, status_rows, len(cases)
                )
                elapsed = time.perf_counter() - started
                execution_metadata = suite_execution_metadata(
                    status_rows,
                    elapsed,
                    job_status,
                    previous_campaign_rows_by_job.get(job_id),
                )
                campaign_job_rows.append({
                    **base_row,
                    "return_code": int(completed.returncode),
                    "status": job_status,
                    "reason": reason,
                    "case_count": len(cases),
                    "suite_status_path": str(status_path),
                    **execution_metadata,
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

            cuFINUFFT adapter 可能在运行期失败后回退 CPU，因此不能只检查请求参数；必须同时检查 manifest 的 `nufft_backend_resolved` 和 `nufft_stage`。下格先检查 fp64、system unchanged、strict GPU eig 和完整 `run_complete.json`；随后 canonical reporter 还会强制读取并重哈希 timing NPZ 中的 weights/Gf/storage-RHS/solve-RHS，核对共同 tolerance、maxiter 和零初值。任一检查失败都会在正式图生成前终止。
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
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        manifest = {}
                        problems.append("unreadable system_manifest.json")
                    if not isinstance(manifest, dict):
                        manifest = {}
                        problems.append("system_manifest.json is not an object")
                    if not manifest.get("system_unchanged"): problems.append("system changed")
                    if manifest.get("nufft_backend_resolved") != "cufinufft": problems.append("backend fallback")
                    if manifest.get("nufft_stage") != "cufinufft": problems.append("NUFFT stage is not GPU")
                    if manifest.get("precision_mode") != "fp64": problems.append("not fp64")
                    if not manifest.get("device_name"): problems.append("missing timing device_name")
                    if not manifest.get("compute_capability"): problems.append("missing compute_capability")
                    if not manifest.get("timing_runtime_sha256"): problems.append("missing timing_runtime_sha256")
                    missing_component_hashes = [
                        field for field in (
                            "weights_sha256", "gf_sha256", "rhs_sha256",
                            "rhs_storage_sha256",
                        )
                        if not manifest.get(field)
                    ]
                    if missing_component_hashes:
                        problems.append(
                            "manifest lacks exact component hashes: "
                            + ", ".join(missing_component_hashes)
                        )
                if not config_path.is_file():
                    problems.append("missing experiment_config.json")
                else:
                    try:
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        config = {}
                        problems.append("unreadable experiment_config.json")
                    if not isinstance(config, dict):
                        config = {}
                        problems.append("experiment_config.json is not an object")
                    if not config.get("strict_gpu_eig"): problems.append("strict_gpu_eig false")
                if not complete_path.is_file(): problems.append("missing run_complete.json")
                if not summary_path.is_file():
                    problems.append("missing matched_summary.csv")
                else:
                    try:
                        matched = pd.read_csv(summary_path)
                    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                        matched = pd.DataFrame()
                        problems.append(f"unreadable matched_summary.csv: {type(exc).__name__}")
                    required_columns = {
                        "method", "measured_repeats", "converged_repeats",
                        "performance_claim_eligible", "true_relres_max", "cold_speedup_median",
                    }
                    if matched.empty:
                        problems.append("matched summary is empty")
                    elif not required_columns.issubset(matched.columns):
                        problems.append("matched summary lacks eligibility columns")
                    else:
                        expected_methods = config.get("methods")
                        observed_methods = matched["method"].astype(str).tolist()
                        if (
                            not isinstance(expected_methods, list)
                            or not expected_methods
                            or len({str(method) for method in expected_methods})
                            != len(expected_methods)
                            or sorted(observed_methods)
                            != sorted(str(method) for method in expected_methods)
                        ):
                            problems.append(
                                "matched summary method coverage differs from experiment_config.methods"
                            )
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
                    "run_dir": str(run_dir),
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
                    "rhs_storage_sha256": manifest.get("rhs_storage_sha256"),
                    "device_name": manifest.get("device_name"),
                    "compute_capability": manifest.get("compute_capability"),
                    "timing_runtime_sha256": manifest.get("timing_runtime_sha256"),
                    "warning": "; ".join(warnings),
                    "status": "PASS" if not problems else "FAIL: " + "; ".join(problems),
                })
            controlled_artifact_audit = pd.DataFrame(
                audit_rows,
                columns=[
                    "run_dir", "output_group", "suite_profile", "job_id", "mandatory", "case", "N", "system_id",
                    "weights_sha256", "gf_sha256", "rhs_sha256", "rhs_storage_sha256",
                    "device_name", "compute_capability", "timing_runtime_sha256",
                    "warning", "status",
                ],
            )
            if not controlled_artifact_audit.empty:
                for output_group, group_index in controlled_artifact_audit.groupby(
                    "output_group", sort=False
                ).groups.items():
                    group = controlled_artifact_audit.loc[group_index]
                    if (
                        group["device_name"].dropna().astype(str).nunique() != 1
                        or group["compute_capability"].dropna().astype(str).nunique() != 1
                        or group["timing_runtime_sha256"].dropna().astype(str).nunique() != 1
                    ):
                        controlled_artifact_audit.loc[group_index, "status"] = (
                            "FAIL: selected profile mixes or lacks GPU runtime identity"
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
                raise RuntimeError(
                    "CONTROLLED AUDIT COUNT MISMATCH: "
                    f"Expected exactly {expected_controlled_case_count} controlled cases, "
                    f"but found {len(controlled_artifact_audit)} manifests."
                )
            if not controlled_artifact_audit.empty and not controlled_artifact_audit["status"].eq("PASS").all():
                failed_controlled = controlled_artifact_audit.loc[
                    ~controlled_artifact_audit["status"].eq("PASS"),
                    ["case", "status"],
                ].to_dict(orient="records")
                raise RuntimeError(
                    "CONTROLLED ARTIFACT AUDIT FAILED; refusing every formal "
                    f"Stage-2 table/plot: {failed_controlled}"
                )
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            ## Stage 2 正式结果：solver total（selection/build + solve）

            本表和图只接收 `fixed_ab_selected_target`，并显式拒绝 Stage 1 KRR 方法名及 Fourier Nyström/RPCholesky adaptations。柱高是实际 `solver_total_seconds_median`；shared Fourier setup 和 prediction 不进入本阶段 headline。是否“更快”由该 total 直接决定，不会从 iteration count 推断。
            """
        )
    )
    cells.append(
        _code(
            r"""
            import matplotlib.pyplot as plt
            import numpy as np

            STAGE2_FORMAL_METHODS = list(
                globals().get("STAGE2_FEASIBLE_METHODS", STAGE2_METHODS)
            )
            STAGE2_FORBIDDEN_METHODS = {
                "nystrom", "rpcholesky", "fourier-nystrom-precond",
                "fourier-rpcholesky-precond", "nystrom-krr", "rpcholesky-krr",
            }
            STAGE2_GENERATED_PLOT_PATHS = []
            stage2_frames = []
            stage2_system_ids = set()
            for record in selected_case_records:
                if record.get("suite_profile") != "fixed_ab_selected_target":
                    continue
                run_dir = Path(record["run_dir"])
                summary_path = run_dir / "matched_summary.csv"
                manifest_path = run_dir / "system_manifest.json"
                if not (summary_path.is_file() and manifest_path.is_file()):
                    continue
                frame = pd.read_csv(summary_path)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                frame["case_id"] = record["case_id"]
                frame["run_dir"] = str(run_dir)
                frame["system_id"] = manifest.get("system_id")
                frame["N"] = manifest.get("n_train")
                frame["protocol_family"] = "controlled_fixed_system"
                frame["headline_timing"] = (
                    "solver_total_seconds = selection + preconditioner build + solve"
                )
                stage2_system_ids.add(manifest.get("system_id"))
                stage2_frames.append(frame)
            stage2_solver_summary = (
                pd.concat(stage2_frames, ignore_index=True, sort=False)
                if stage2_frames else pd.DataFrame()
            )
            if not stage2_solver_summary.empty:
                raw_stage2_methods = set(stage2_solver_summary["method"].astype(str))
                forbidden_raw_methods = raw_stage2_methods.intersection(
                    STAGE2_FORBIDDEN_METHODS
                )
                if forbidden_raw_methods:
                    raise RuntimeError(
                        "Stage-2 formal table contains forbidden KRR/adaptation "
                        f"labels: {sorted(forbidden_raw_methods)}"
                    )
            # Keep the direct matched-summary aggregation only as an explicitly
            # non-authoritative diagnostic.  The public summary path is written
            # below *after* the canonical reporter has recomputed every timing
            # component and speedup from verified repeat rows.
            RAW_STAGE2_DIAGNOSTIC_PATH = (
                DRIVE_RUN_ROOT / "stage2_fixed_ab_raw_diagnostic.csv"
            )
            stage2_solver_summary.to_csv(RAW_STAGE2_DIAGNOSTIC_PATH, index=False)
            STAGE2_SOLVER_SUMMARY_PATH = (
                DRIVE_RUN_ROOT / "stage2_fixed_ab_solver_summary.csv"
            )

            # Run the authoritative cross-artifact audit *before* constructing
            # a formal table or figure.  The chart below consumes only its
            # repeat-recomputed output, never derived fields from matched_summary.
            from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.two_stage_reporting import (
                TwoStageReportConfig,
                build_two_stage_report,
            )
            TWO_STAGE_FORMAL_REPORT_ROOT = DRIVE_RUN_ROOT / "two_stage_formal_report"
            TWO_STAGE_FORMAL_REPORT_RESULT = None
            TWO_STAGE_GENERATED_PLOT_PATHS = []
            stage1_formal_input_paths = [
                str(path)
                for path, frame in (
                    (
                        globals().get("STAGE1_SCALE_SUMMARY_PATH"),
                        globals().get("stage1_scale_verified", pd.DataFrame()),
                    ),
                    (
                        globals().get("STAGE1_ROBUSTNESS_SUMMARY_PATH"),
                        globals().get("stage1_robustness_verified", pd.DataFrame()),
                    ),
                )
                if path is not None and not frame.empty and Path(path).is_file()
            ]
            stage2_formal_input_paths = [
                str(Path(record["run_dir"]) / "matched_summary.csv")
                for record in selected_case_records
                if record.get("suite_profile") == "fixed_ab_selected_target"
                and (Path(record["run_dir"]) / "matched_summary.csv").is_file()
            ]
            formal_artifacts_ready = bool(
                stage1_formal_input_paths
                and stage2_formal_input_paths
                and STAGE1_TARGET_PATH.is_file()
                and STAGE2_FEASIBILITY_PATH.is_file()
            )
            if formal_artifacts_ready:
                TWO_STAGE_FORMAL_REPORT_RESULT = build_two_stage_report(
                    TwoStageReportConfig(
                        stage1_paths=tuple(stage1_formal_input_paths),
                        stage2_paths=tuple(stage2_formal_input_paths),
                        output_dir=str(TWO_STAGE_FORMAL_REPORT_ROOT),
                        selected_target_path=str(STAGE1_TARGET_PATH),
                        stage1_suite_path=str(STAGE1_SUITE_CONFIG),
                        stage2_feasibility_path=str(STAGE2_FEASIBILITY_PATH),
                        include_fourier_adaptations_in_formal_stage2=False,
                        make_plots=True,
                    )
                )
                TWO_STAGE_GENERATED_PLOT_PATHS = [
                    TWO_STAGE_FORMAL_REPORT_ROOT / name
                    for name in TWO_STAGE_FORMAL_REPORT_RESULT["manifest"]["artifacts"]
                    if str(name).lower().endswith(".png")
                ]
                canonical_stage2_path = (
                    TWO_STAGE_FORMAL_REPORT_ROOT / "stage2_formal_solver_totals.csv"
                )
                stage2_solver_summary = pd.read_csv(canonical_stage2_path)
                stage2_solver_summary["solver_total_seconds_median"] = pd.to_numeric(
                    stage2_solver_summary["solver_total_seconds"], errors="coerce"
                )
                stage2_solver_summary["selection_seconds_median"] = pd.to_numeric(
                    stage2_solver_summary["selection_seconds"], errors="coerce"
                )
                stage2_solver_summary["preconditioner_build_seconds_median"] = pd.to_numeric(
                    stage2_solver_summary["preconditioner_build_seconds"], errors="coerce"
                )
                stage2_solver_summary["solve_seconds_median"] = pd.to_numeric(
                    stage2_solver_summary["solve_seconds"], errors="coerce"
                )
                stage2_solver_summary["protocol_family"] = "controlled_fixed_system"
                stage2_solver_summary["headline_timing"] = (
                    "solver_total_seconds = selection + preconditioner build + solve"
                )
                print(
                    "Canonical two-stage claim audit:",
                    TWO_STAGE_FORMAL_REPORT_ROOT / "claim_audit.csv",
                )
            elif RUN_ALL_FORMAL_EXPERIMENTS and END_TO_END_TARGET is not None:
                raise RuntimeError(
                    "Formal target exists but canonical Stage-1/Stage-2 artifacts are "
                    "incomplete; refusing headline output."
                )
            else:
                stage2_solver_summary = pd.DataFrame()

            if not stage2_solver_summary.empty:
                observed = set(stage2_solver_summary["method"].astype(str))
                forbidden = observed.intersection(STAGE2_FORBIDDEN_METHODS)
                if forbidden:
                    raise RuntimeError(
                        f"Stage-2 formal table contains forbidden KRR/adaptation labels: {sorted(forbidden)}"
                    )
                if observed != set(STAGE2_FORMAL_METHODS):
                    raise RuntimeError(
                        f"Stage-2 method coverage {sorted(observed)} != frozen {STAGE2_FORMAL_METHODS}"
                    )
                if len(stage2_system_ids) != 1 or None in stage2_system_ids:
                    raise RuntimeError(
                        f"Stage-2 methods do not share one verified system_id: {stage2_system_ids}"
                    )
                stage2_solver_summary["solver_total_seconds_median"] = pd.to_numeric(
                    stage2_solver_summary["solver_total_seconds_median"], errors="coerce"
                )
                stage2_solver_summary["selection_seconds_median"] = pd.to_numeric(
                    stage2_solver_summary["selection_seconds_median"], errors="coerce"
                )
                stage2_solver_summary["preconditioner_build_seconds_median"] = pd.to_numeric(
                    stage2_solver_summary["preconditioner_build_seconds_median"], errors="coerce"
                )
                stage2_solver_summary["solve_seconds_median"] = pd.to_numeric(
                    stage2_solver_summary["solve_seconds_median"], errors="coerce"
                )
                canonical_speedup_fields = {
                    "method_kind",
                    "result_role",
                    "measured_repeats",
                    "solver_total_speedup_over_cg_median",
                    "paired_comparisons",
                    "solver_total_speedup_source",
                }
                missing_speedup_fields = canonical_speedup_fields.difference(
                    stage2_solver_summary.columns
                )
                if missing_speedup_fields:
                    raise RuntimeError(
                        "canonical Stage-2 output lacks matched-repeat speedup fields: "
                        f"{sorted(missing_speedup_fields)}"
                    )
                for numeric_field in (
                    "measured_repeats",
                    "solver_total_speedup_over_cg_median",
                    "paired_comparisons",
                ):
                    stage2_solver_summary[numeric_field] = pd.to_numeric(
                        stage2_solver_summary[numeric_field], errors="coerce"
                    )
                eligible_mask = (
                    stage2_solver_summary["performance_claim_eligible"]
                    .astype(str).str.lower().eq("true")
                )
                eligible_rows = stage2_solver_summary.loc[eligible_mask]
                if (
                    eligible_rows.empty
                    or not np.isfinite(
                        eligible_rows["solver_total_speedup_over_cg_median"].to_numpy(
                            dtype=float
                        )
                    ).all()
                    or not (
                        eligible_rows["paired_comparisons"]
                        == eligible_rows["measured_repeats"]
                    ).all()
                ):
                    raise RuntimeError(
                        "canonical Stage-2 speedup must use every eligible matched repeat"
                    )
                cg_speedup = eligible_rows.loc[
                    eligible_rows["method"].astype(str).eq("cg"),
                    "solver_total_speedup_over_cg_median",
                ]
                if len(cg_speedup) != 1 or not np.isclose(
                    float(cg_speedup.iloc[0]), 1.0, rtol=1e-12, atol=1e-12
                ):
                    raise RuntimeError(
                        "canonical Stage-2 matched-repeat CG self-speedup must equal 1"
                    )
                stage2_solver_summary.to_csv(
                    STAGE2_SOLVER_SUMMARY_PATH, index=False
                )
                order = {method: index for index, method in enumerate(STAGE2_FORMAL_METHODS)}
                stage2_solver_summary["method_order"] = stage2_solver_summary["method"].map(order)
                plot_rows = stage2_solver_summary.loc[
                    stage2_solver_summary["performance_claim_eligible"].astype(str).str.lower().eq("true")
                ].sort_values("method_order").copy()
                plot_rows["method_display"] = plot_rows.apply(
                    lambda row: (
                        f"{row['method']} [{row['method_kind']}]"
                        if str(row["method"]) == "default"
                        else (
                            f"{row['method']} [explicit route-control]"
                            if str(row["method"]) in {"active-inverse", "active-eig"}
                            and bool(
                                (
                                    stage2_solver_summary["method"].astype(str).eq("default")
                                    & stage2_solver_summary["method_kind"].astype(str).eq(
                                        str(row["method_kind"])
                                    )
                                ).any()
                            )
                            else str(row["method"])
                        )
                    ),
                    axis=1,
                )
                fig, (ax_total, ax_speedup) = plt.subplots(1, 2, figsize=(13, 5))
                x = np.arange(len(plot_rows))
                canonical_total = plot_rows["solver_total_seconds_median"].to_numpy(dtype=float)
                ax_total.bar(x, canonical_total)
                ax_total.set_xticks(x, plot_rows["method_display"], rotation=30, ha="right")
                ax_total.set_ylabel("solver total, seconds")
                ax_total.set_title(
                    "Stage 2: canonical median solver total on identical hashed A,b"
                )
                ax_total.grid(True, axis="y", alpha=0.25)

                speedup = pd.to_numeric(
                    plot_rows["solver_total_speedup_over_cg_median"], errors="coerce"
                )
                ax_speedup.bar(x, speedup)
                ax_speedup.axhline(1.0, color="black", linestyle="--")
                ax_speedup.set_xticks(x, plot_rows["method_display"], rotation=30, ha="right")
                ax_speedup.set_ylabel("median matched CG_i / method_i total")
                ax_speedup.set_title("Paired total speedup (>1 is faster than CG)")
                ax_speedup.grid(True, axis="y", alpha=0.25)
                fig.tight_layout()
                stage2_plot = DRIVE_RUN_ROOT / "stage2_fixed_ab_solver_total.png"
                fig.savefig(stage2_plot, dpi=180, bbox_inches="tight")
                STAGE2_GENERATED_PLOT_PATHS.append(stage2_plot)
                plt.show()
                plt.close(fig)
                display(stage2_solver_summary[[
                    "method", "method_kind", "result_role", "system_id",
                    "selection_seconds_median",
                    "preconditioner_build_seconds_median", "solve_seconds_median",
                    "solver_total_seconds_median", "solver_total_speedup_over_cg_median",
                    "paired_comparisons", "performance_claim_eligible",
                ]])
            else:
                print("No Stage-2 fixed-target rows; solver-total plot skipped.")
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            # Stage 2 prediction-equivalence audit（复用 timed system/solution；单独、非计时）

            Audit 必须读取 Stage 2 timing case 保存的 exact system 与 canonical timed \(\beta\)，不得重建系统或重解。它只按 GPU chunk 计算预测，prediction 时间明确排除在 solver-total claim 外；每个 case 最多使用测试集的前 2.5M 行。Stage 1 已经在各完整 KRR pipeline 内独立报告 test RMSE/R²、宽松 usability 与描述性的 reference equivalence，因此两种 accuracy 证据不混用。
            """
        )
    )
    cells.append(
        _code(
            r"""
            prediction_outputs = []
            prediction_validation_rows = []
            expected_prediction_case_count = 0
            if RUN_PREDICTION_AUDIT:
                from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.prediction_audit import (
                    PREDICTION_AUDIT_COMPLETION_FILENAME,
                    PREDICTION_AUDIT_CSV_FILENAME,
                    PREDICTION_AUDIT_JSON_FILENAME,
                    prediction_source_manifest,
                )
                CURRENT_PREDICTION_SOURCE_SHA256 = prediction_source_manifest()[
                    "prediction_source_bundle_sha256"
                ]
                prediction_started = time.perf_counter()
                prediction_resumed_count = 0
                prediction_records = [
                    record for record in selected_case_records
                    if record.get("suite_profile") in set(PREDICTION_AUDIT_PROFILES)
                ]
                expected_prediction_case_count = len(prediction_records)
                prediction_targets = []
                prediction_errors = []
                prediction_scientific_failures = []

                def validate_prediction_payload(payload, *, audit_out, config_path, timing_manifest, required_methods, expected_n_test):
                    problems = []
                    if not isinstance(payload, dict):
                        return ["prediction audit payload is not a JSON object"]
                    current_config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
                    if payload.get("config_source_sha256") != current_config_sha:
                        problems.append("config_source_sha256 mismatch")
                    if payload.get("dataset_content_index_sha256") != timing_manifest.get("dataset_content_index_sha256"):
                        problems.append("dataset_content_index_sha256 mismatch")
                    if payload.get("source_bundle_sha256") != timing_manifest.get("source_bundle_sha256"):
                        problems.append("source_bundle_sha256 mismatch")
                    if payload.get("prediction_source_bundle_sha256") != CURRENT_PREDICTION_SOURCE_SHA256:
                        problems.append("prediction_source_bundle_sha256 mismatch")
                    if payload.get("test_dataset_content_index_verified") is not True:
                        problems.append("test dataset content index was not verified")
                    if payload.get("test_dataset_metadata_verified") is not True:
                        problems.append("test dataset metadata was not verified")
                    if payload.get("strict_prediction_nufft") is not True:
                        problems.append("strict_prediction_nufft must be true")
                    if payload.get("observed_prediction_nufft_stages") != ["cufinufft"]:
                        problems.append("prediction did not exclusively use cufinufft")
                    if payload.get("audit_rebuilt_system") is not False:
                        problems.append("audit_rebuilt_system must be false")
                    if payload.get("timing_system_reused") is not True:
                        problems.append("timing_system_reused must be true")
                    if payload.get("timing_solutions_reused") is not True:
                        problems.append("timing_solutions_reused must be true")
                    if payload.get("timing_solution_hashes_verified") is not True:
                        problems.append("timing_solution_hashes_verified must be true")
                    if payload.get("timing_system_hashes_exact") is not True:
                        problems.append("timing_system_hashes_exact must be true")
                    if payload.get("audit_solve_count") != 0:
                        problems.append("audit_solve_count must be zero")
                    if payload.get("audit_solves_per_method") != 0:
                        problems.append("audit_solves_per_method must be zero")
                    timing_completion_path = config_path.parent / "run_complete.json"
                    if not timing_completion_path.is_file():
                        problems.append("timing run_complete.json is missing")
                    elif payload.get("timing_run_complete_sha256") != hashlib.sha256(
                        timing_completion_path.read_bytes()
                    ).hexdigest():
                        problems.append("timing run completion checksum mismatch")
                    if payload.get("audit_pass") is not True:
                        problems.append("audit_pass is not true")
                    for field in (
                        "system_id", "weights_sha256", "gf_sha256", "rhs_sha256",
                        "rhs_storage_sha256", "reg_lambda",
                    ):
                        if not timing_manifest.get(field) or payload.get(field) != timing_manifest.get(field):
                            problems.append(f"{field} does not exactly match timing artifact")
                    try:
                        observed_n_test = int(payload.get("evaluated_n_test", -1))
                    except (TypeError, ValueError):
                        observed_n_test = -1
                    if observed_n_test != int(expected_n_test):
                        problems.append(
                            f"evaluated_n_test={payload.get('evaluated_n_test')} expected={expected_n_test}"
                        )
                    rows = payload.get("rows", [])
                    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                        problems.append("prediction rows are not a list of objects")
                        rows = []
                    observed_methods = {str(row.get("method")) for row in rows}
                    if observed_methods != set(required_methods):
                        problems.append(
                            f"prediction methods={sorted(observed_methods)} expected={sorted(required_methods)}"
                        )
                    if len(rows) != len(required_methods):
                        problems.append(
                            "prediction JSON must contain exactly one row per required method"
                        )
                    non_equivalent = sorted(
                        str(row.get("method")) for row in rows
                        if str(row.get("method")) != "cg"
                        and row.get("prediction_equivalent_to_cg") is not True
                    )
                    if non_equivalent:
                        problems.append(f"prediction equivalence failed: {non_equivalent}")

                    json_path = Path(audit_out) / PREDICTION_AUDIT_JSON_FILENAME
                    csv_path = Path(audit_out) / PREDICTION_AUDIT_CSV_FILENAME
                    completion_path = Path(audit_out) / PREDICTION_AUDIT_COMPLETION_FILENAME
                    if not (json_path.is_file() and csv_path.is_file() and completion_path.is_file()):
                        problems.append("prediction JSON/CSV/completion artifact set is incomplete")
                    else:
                        try:
                            completion = json.loads(completion_path.read_text(encoding="utf-8"))
                            csv_frame = pd.read_csv(csv_path)
                        except (OSError, json.JSONDecodeError, ValueError) as exc:
                            problems.append(f"prediction completion/CSV unreadable: {type(exc).__name__}")
                        else:
                            if not isinstance(completion, dict):
                                problems.append("prediction completion is not a JSON object")
                            else:
                                if completion.get("prediction_audit_json_sha256") != hashlib.sha256(json_path.read_bytes()).hexdigest():
                                    problems.append("prediction JSON checksum mismatch")
                                if completion.get("prediction_audit_csv_sha256") != hashlib.sha256(csv_path.read_bytes()).hexdigest():
                                    problems.append("prediction CSV checksum mismatch")
                                if completion.get("system_id") != payload.get("system_id"):
                                    problems.append("prediction completion system_id mismatch")
                                if completion.get("methods") != list(required_methods):
                                    problems.append("prediction completion methods mismatch")
                                try:
                                    completion_row_count = int(completion.get("row_count", -1))
                                except (TypeError, ValueError, OverflowError):
                                    completion_row_count = -1
                                if completion_row_count != len(rows):
                                    problems.append("prediction completion row_count mismatch")
                                if completion.get("audit_pass") != payload.get("audit_pass"):
                                    problems.append("prediction completion audit_pass mismatch")
                                if completion.get("prediction_source_bundle_sha256") != CURRENT_PREDICTION_SOURCE_SHA256:
                                    problems.append("prediction completion source mismatch")
                            if len(csv_frame) != len(rows) or set(csv_frame.get("method", [])) != set(required_methods):
                                problems.append("prediction CSV rows/methods differ from JSON")
                    return problems

                for record in prediction_records:
                    config_path = Path(record["run_dir"]) / "experiment_config.json"
                    manifest_path = Path(record["run_dir"]) / "system_manifest.json"
                    if not manifest_path.is_file() or not config_path.is_file():
                        message = f"prediction target lacks config/manifest: {record['run_dir']}"
                        prediction_errors.append(message)
                        prediction_validation_rows.append({
                            "case_id": record["case_id"], "status": "EXECUTION_ERROR",
                            "audit_pass": False,
                            "exact_timing_system_match": False,
                            "timing_solutions_reused": False,
                            "timing_solution_hashes_verified": False,
                            "zero_audit_solves": False,
                            "problems": message,
                        })
                        continue
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                        n_train = int(manifest["n_train"])
                    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        message = (
                            f"prediction target has unreadable config/manifest: "
                            f"{record['run_dir']} ({type(exc).__name__})"
                        )
                        prediction_errors.append(message)
                        prediction_validation_rows.append({
                            "case_id": record["case_id"], "status": "EXECUTION_ERROR",
                            "audit_pass": False,
                            "exact_timing_system_match": False,
                            "timing_solutions_reused": False,
                            "timing_solution_hashes_verified": False,
                            "zero_audit_solves": False,
                            "problems": message,
                        })
                        continue
                    required_methods = [str(method) for method in config.get("methods", [])]
                    expected_n_test = min(int(PREDICTION_AUDIT_MAX_TEST_N), n_train // 4)
                    prediction_targets.append(
                        (record, config_path, manifest, required_methods, expected_n_test)
                    )
                for record, config_path, manifest, required_methods, expected_n_test in prediction_targets:
                    audit_out = config_path.parent / "prediction_audit"
                    existing_is_valid = False
                    if (audit_out / PREDICTION_AUDIT_JSON_FILENAME).is_file():
                        try:
                            existing = json.loads(
                                (audit_out / PREDICTION_AUDIT_JSON_FILENAME).read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError) as exc:
                            existing = {}
                            existing_problems = [
                                f"existing prediction JSON unreadable: {type(exc).__name__}"
                            ]
                        else:
                            existing_problems = validate_prediction_payload(
                                existing,
                                audit_out=audit_out,
                                config_path=config_path,
                                timing_manifest=manifest,
                                required_methods=required_methods,
                                expected_n_test=expected_n_test,
                            )
                        if not existing_problems:
                            print("Valid prediction audit already exists; reusing", config_path.parent.name)
                            prediction_outputs.append(audit_out)
                            prediction_resumed_count += 1
                            prediction_validation_rows.append({
                                "case_id": record["case_id"],
                                "suite_profile": record.get("suite_profile"),
                                "dataset_family": record.get("dataset_family"),
                                "n_train": int(manifest["n_train"]),
                                "required_methods": ",".join(required_methods),
                                "evaluated_n_test": expected_n_test,
                                "status": "PASS_RESUMED",
                                "audit_pass": True,
                                "exact_timing_system_match": True,
                                "timing_solutions_reused": True,
                                "timing_solution_hashes_verified": True,
                                "zero_audit_solves": True,
                                "problems": "",
                                "audit_path": str(audit_out / "prediction_audit.json"),
                            })
                            existing_is_valid = True
                        else:
                            print(
                                "Existing prediction audit is stale/ineligible; rerunning",
                                record["case_id"], existing_problems,
                            )
                    if existing_is_valid:
                        continue
                    command = [
                        sys.executable, "-m",
                        "efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.prediction_audit",
                        "--config", str(config_path),
                        "--prediction-chunk-size", "100000",
                        "--strict-prediction-nufft",
                        "--max-test", str(expected_n_test),
                        "--output-dir", str(audit_out),
                    ]
                    command.extend(["--methods", ",".join(required_methods)])
                    completed = run_cmd(command, cwd=LOCAL_REPO, check=False)
                    final_payload = {}
                    final_problems = ["prediction audit JSON missing"]
                    if (audit_out / PREDICTION_AUDIT_JSON_FILENAME).is_file():
                        try:
                            final_payload = json.loads(
                                (audit_out / PREDICTION_AUDIT_JSON_FILENAME).read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError) as exc:
                            final_problems = [
                                f"prediction JSON unreadable after execution: {type(exc).__name__}"
                            ]
                        else:
                            final_problems = validate_prediction_payload(
                                final_payload,
                                audit_out=audit_out,
                                config_path=config_path,
                                timing_manifest=manifest,
                                required_methods=required_methods,
                                expected_n_test=expected_n_test,
                            )
                    if completed.returncode == 0 and not final_problems:
                        prediction_outputs.append(audit_out)
                        validation_status = "PASS_EXECUTED"
                    else:
                        message = (
                            f"prediction audit failed for {record['case_id']} rc={completed.returncode}: "
                            + "; ".join(final_problems)
                        )
                        prediction_errors.append(message)
                        if completed.returncode == 2 or final_payload.get("audit_pass") is False:
                            prediction_scientific_failures.append(message)
                        validation_status = (
                            "SCIENTIFIC_FAIL" if message in prediction_scientific_failures
                            else "EXECUTION_ERROR"
                        )
                    prediction_validation_rows.append({
                        "case_id": record["case_id"],
                        "suite_profile": record.get("suite_profile"),
                        "dataset_family": record.get("dataset_family"),
                        "n_train": int(manifest["n_train"]),
                        "required_methods": ",".join(required_methods),
                        "evaluated_n_test": expected_n_test,
                        "status": validation_status,
                        "audit_pass": final_payload.get("audit_pass") is True,
                        "exact_timing_system_match": all(
                            final_payload.get(field) == manifest.get(field)
                            and manifest.get(field) is not None
                            for field in (
                                "system_id", "weights_sha256", "gf_sha256",
                                "rhs_sha256", "rhs_storage_sha256", "reg_lambda",
                            )
                        ),
                        "timing_solutions_reused": final_payload.get("timing_solutions_reused") is True,
                        "timing_solution_hashes_verified": (
                            final_payload.get("timing_solution_hashes_verified") is True
                        ),
                        "zero_audit_solves": (
                            final_payload.get("audit_solve_count") == 0
                            and final_payload.get("audit_solves_per_method") == 0
                        ),
                        "problems": "; ".join(final_problems),
                        "audit_path": str(audit_out / "prediction_audit.json"),
                    })
                prediction_status = (
                    "PASS" if expected_prediction_case_count > 0
                    and len(prediction_outputs) == expected_prediction_case_count
                    and not prediction_errors
                    else ("SCIENTIFIC_FAIL" if prediction_scientific_failures else "EXECUTION_ERROR")
                )
                prediction_elapsed = time.perf_counter() - prediction_started
                prediction_execution_metadata = execution_metadata_from_counts(
                    case_count=expected_prediction_case_count,
                    resumed_count=prediction_resumed_count,
                    elapsed_seconds=prediction_elapsed,
                    invocation_successful=(prediction_status == "PASS"),
                    previous_row=previous_campaign_rows_by_job.get("prediction_accuracy_audit"),
                )
                prediction_families = sorted({
                    str(record.get("dataset_family"))
                    for record, *_ in prediction_targets
                    if record.get("dataset_family")
                })
                prediction_sizes = sorted({
                    int(manifest["n_train"])
                    for _, _, manifest, _, _ in prediction_targets
                })
                campaign_job_rows.append({
                    "job_id": "prediction_accuracy_audit",
                    "profile": ",".join(PREDICTION_AUDIT_PROFILES),
                    "dataset_family": ",".join(prediction_families),
                    "n_train": ",".join(str(value) for value in prediction_sizes),
                    "mandatory": True,
                    "return_code": 0 if prediction_status == "PASS" else (2 if prediction_status == "SCIENTIFIC_FAIL" else 1),
                    "status": prediction_status,
                    "reason": "; ".join(prediction_errors),
                    "case_count": expected_prediction_case_count,
                    **prediction_execution_metadata,
                })
                write_campaign_checkpoint()
            PREDICTION_ARTIFACT_AUDIT_PATH = DRIVE_RUN_ROOT / "prediction_artifact_audit.csv"
            pd.DataFrame(prediction_validation_rows).to_csv(PREDICTION_ARTIFACT_AUDIT_PATH, index=False)
            print("Prediction audit outputs:", prediction_outputs)
            """
        )
    )
    cells.append(
        _markdown(
            r"""
            # Legacy / diagnostic 统一索引与图（不进入两阶段正式图）

            统一索引只负责查找和展示，不跨 protocol 计算 speedup。正式结论应使用前面的 `stage1_krr_*` 和 `stage2_fixed_ab_*` artifacts；本节保留历史 profile 的诊断/复现图。每行保留 `output_group/case_id`、科学配置哈希、数据与源码哈希，且作图必须按 profile 隔离。

            `paper_10m` 是固定规模的方法比较，单独画带范围的分组图和速度–内存 Pareto 图；三种 `scale_*` protocol 各自分图。旧的 `controlled_scale_speedup.png` 会被标记为 deprecated，不再作为论文证据。
            """
        )
    )
    cells.append(
        _code(
            r"""
            import re
            from efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.prediction_audit import (
                PREDICTION_AUDIT_COMPLETION_FILENAME,
                PREDICTION_AUDIT_CSV_FILENAME,
                PREDICTION_AUDIT_JSON_FILENAME,
            )

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
            SYSTEM_COMPONENT_FIELDS = (
                "weights_sha256", "gf_sha256", "rhs_sha256", "rhs_storage_sha256"
            )

            def dataset_series_id(stem):
                return re.sub(r"_n(?:train)?\d+$", "", str(stem or ""), flags=re.IGNORECASE)
            selected_record_by_dir = {
                str(Path(record["run_dir"]).resolve()): record
                for record in selected_case_records
            }
            prediction_outputs_declared = "prediction_outputs" in globals()
            selected_prediction_audit_dirs = {
                str(Path(path).resolve())
                for path in globals().get("prediction_outputs", [])
            }
            result_frames = []
            controlled_frames = []
            prediction_frames = []
            report_ingest_warnings = []
            for summary_path in sorted(CONTROLLED_OUTPUT_ROOT.rglob("matched_summary.csv")):
                run_dir = summary_path.parent.resolve()
                config_path = summary_path.with_name("experiment_config.json")
                try:
                    frame = pd.read_csv(summary_path)
                    manifest = json.loads(
                        summary_path.with_name("system_manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    if not isinstance(manifest, dict) or not isinstance(config, dict):
                        raise ValueError("manifest/config is not a JSON object")
                    required_report_columns = {
                        "method", "performance_claim_eligible", "measured_repeats",
                        "converged_repeats", "true_relres_max", "cold_speedup_median",
                    }
                    if frame.empty or not required_report_columns.issubset(frame.columns):
                        raise ValueError(
                            "controlled summary is empty or lacks required report columns"
                        )
                except (
                    OSError, json.JSONDecodeError, ValueError,
                    pd.errors.ParserError, pd.errors.EmptyDataError,
                ) as exc:
                    report_ingest_warnings.append({
                        "artifact": str(summary_path),
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    continue
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
                frame["device_name"] = manifest.get("device_name")
                frame["compute_capability"] = manifest.get("compute_capability")
                frame["timing_runtime_sha256"] = manifest.get("timing_runtime_sha256")
                frame["prepared_system_loaded_from_artifact"] = manifest.get(
                    "prepared_system_loaded_from_artifact", False
                )
                frame["setup_inclusive_timing_eligible"] = manifest.get(
                    "setup_inclusive_timing_eligible",
                    not bool(manifest.get("prepared_system_loaded_from_artifact", False)),
                )
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
                run_dir = audit_path.parent.parent.resolve()
                audit_payload_path = audit_path.with_suffix(".json")
                try:
                    frame = pd.read_csv(audit_path)
                    manifest = json.loads(
                        (run_dir / "system_manifest.json").read_text(encoding="utf-8")
                    )
                    audit_payload = (
                        json.loads(audit_payload_path.read_text(encoding="utf-8"))
                        if audit_payload_path.is_file() else {}
                    )
                    if not isinstance(manifest, dict) or not isinstance(audit_payload, dict):
                        raise ValueError("prediction/system manifest is not a JSON object")
                    if frame.empty or "method" not in frame.columns:
                        raise ValueError("prediction audit is empty or lacks method column")
                except (
                    OSError, json.JSONDecodeError, ValueError,
                    pd.errors.ParserError, pd.errors.EmptyDataError,
                ) as exc:
                    report_ingest_warnings.append({
                        "artifact": str(audit_path),
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                completion_path = audit_path.parent / PREDICTION_AUDIT_COMPLETION_FILENAME
                completion_valid = False
                if completion_path.is_file() and audit_payload_path.is_file():
                    try:
                        completion_payload = json.loads(
                            completion_path.read_text(encoding="utf-8")
                        )
                        completion_valid = bool(
                            isinstance(completion_payload, dict)
                            and completion_payload.get("prediction_audit_json_sha256")
                            == hashlib.sha256(audit_payload_path.read_bytes()).hexdigest()
                            and completion_payload.get("prediction_audit_csv_sha256")
                            == hashlib.sha256(audit_path.read_bytes()).hexdigest()
                        )
                    except (OSError, json.JSONDecodeError):
                        completion_valid = False
                relative_parts = run_dir.relative_to(CONTROLLED_OUTPUT_ROOT.resolve()).parts
                output_group = relative_parts[0] if relative_parts else "unknown"
                selected_record = selected_record_by_dir.get(str(run_dir))
                csv_system_id = (
                    frame["system_id"].iloc[0]
                    if "system_id" in frame.columns and not frame.empty else None
                )
                frame["audit_system_id"] = audit_payload.get("system_id", csv_system_id)
                frame["timing_system_id"] = manifest.get("system_id")
                for field in SYSTEM_COMPONENT_FIELDS:
                    frame[f"audit_{field}"] = audit_payload.get(field)
                    frame[f"timing_{field}"] = manifest.get(field)
                exact_system_match = all(
                    audit_payload.get(field) is not None
                    and audit_payload.get(field) == manifest.get(field)
                    for field in ("system_id", *SYSTEM_COMPONENT_FIELDS, "reg_lambda")
                )
                frame["audit_rebuilt_system"] = audit_payload.get("audit_rebuilt_system")
                frame["timing_system_reused"] = audit_payload.get("timing_system_reused")
                frame["timing_solutions_reused"] = audit_payload.get("timing_solutions_reused")
                frame["timing_solution_hashes_verified"] = audit_payload.get(
                    "timing_solution_hashes_verified"
                )
                frame["audit_solve_count"] = audit_payload.get("audit_solve_count")
                frame["audit_pass"] = audit_payload.get("audit_pass")
                frame["prediction_completion_valid"] = completion_valid
                frame["exact_timing_system_match"] = exact_system_match
                row_equivalent = frame["method"].astype(str).eq("cg")
                if "prediction_equivalent_to_cg" in frame.columns:
                    row_equivalent = row_equivalent | frame[
                        "prediction_equivalent_to_cg"
                    ].astype(str).str.lower().eq("true")
                frame["prediction_claim_eligible"] = bool(
                    exact_system_match
                    and audit_payload.get("audit_rebuilt_system") is False
                    and audit_payload.get("timing_system_reused") is True
                    and audit_payload.get("timing_solutions_reused") is True
                    and audit_payload.get("timing_solution_hashes_verified") is True
                    and audit_payload.get("audit_solve_count") == 0
                    and audit_payload.get("audit_solves_per_method") == 0
                    and audit_payload.get("audit_pass") is True
                    and completion_valid
                ) & row_equivalent
                frame["protocol_family"] = "prediction_audit"
                frame["evidence_role"] = "accuracy_only"
                frame["timing_scope"] = "timed beta reused; prediction-only audit excluded from all speed claims"
                frame["output_group"] = output_group
                frame["suite_profile"] = (
                    selected_record.get("suite_profile") if selected_record else output_group
                )
                frame["case"] = run_dir.name
                frame["case_id"] = run_dir.name
                frame["run_dir"] = str(run_dir)
                frame["selected_in_this_invocation"] = bool(
                    selected_record is not None
                    and (
                        not prediction_outputs_declared
                        or str(audit_path.parent.resolve())
                        in selected_prediction_audit_dirs
                    )
                )
                frame["dataset_stem"] = manifest.get("dataset_stem")
                frame["dataset_family"] = dataset_family_label(manifest.get("dataset_stem"))
                frame["dataset_series_id"] = dataset_series_id(manifest.get("dataset_stem"))
                frame["N"] = manifest.get("n_train")
                if "test_rmse_ratio_vs_cg" in frame.columns:
                    frame["test_rmse_difference_ppm_vs_cg"] = 1e6 * (
                        pd.to_numeric(frame["test_rmse_ratio_vs_cg"], errors="coerce") - 1.0
                    )
                    frame["test_rmse_absolute_relative_difference_vs_cg"] = (
                        pd.to_numeric(frame["test_rmse_ratio_vs_cg"], errors="coerce") - 1.0
                    ).abs()
                prediction_frames.append(frame)
                result_frames.append(frame)

            controlled_catalog = (
                pd.concat(controlled_frames, ignore_index=True, sort=False)
                if controlled_frames else pd.DataFrame()
            )
            prediction_catalog = (
                pd.concat(prediction_frames, ignore_index=True, sort=False)
                if prediction_frames else pd.DataFrame()
            )
            selected_prediction = (
                prediction_catalog.loc[prediction_catalog["selected_in_this_invocation"]].copy()
                if not prediction_catalog.empty else pd.DataFrame()
            )
            PREDICTION_ACCURACY_SUMMARY_PATH = DRIVE_RUN_ROOT / "prediction_accuracy_summary.csv"
            selected_prediction.to_csv(PREDICTION_ACCURACY_SUMMARY_PATH, index=False)
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
            REPORT_INGEST_WARNINGS_PATH = DRIVE_RUN_ROOT / "report_ingest_warnings.csv"
            pd.DataFrame(report_ingest_warnings).to_csv(
                REPORT_INGEST_WARNINGS_PATH, index=False
            )
            if report_ingest_warnings:
                print(
                    f"Skipped {len(report_ingest_warnings)} unreadable stale/partial "
                    f"report artifacts; see {REPORT_INGEST_WARNINGS_PATH}."
                )
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

            # Legacy plot allow-list deliberately excludes true KRR method names and
            # exploratory Fourier Nyström/RPCholesky adaptations. Formal Stage 2 uses
            # STAGE2_FORMAL_METHODS and solver_total_seconds in the dedicated cell above.
            METHOD_ORDER = [
                "cg", "jacobi", "default", "active-inverse", "full-inverse",
                "active-eig", "full-eig",
            ]
            METHOD_COLORS = {
                "cg": "#4C78A8",
                "jacobi": "#9D9D9D",
                "default": "#E45756",
                "full-eig": "#72B7B2",
                "active-inverse": "#B279A2",
                "full-inverse": "#F58518",
                "active-eig": "#54A24B",
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

            def minmax_error(frame, median_field, min_field, max_field):
                median = pd.to_numeric(frame[median_field], errors="raise").to_numpy(float)
                minimum = (
                    pd.to_numeric(frame[min_field], errors="coerce").to_numpy(float)
                    if min_field in frame.columns else median.copy()
                )
                maximum = (
                    pd.to_numeric(frame[max_field], errors="coerce").to_numpy(float)
                    if max_field in frame.columns else median.copy()
                )
                minimum = np.where(np.isfinite(minimum), minimum, median)
                maximum = np.where(np.isfinite(maximum), maximum, median)
                return median, np.vstack([
                    np.maximum(median - minimum, 0.0),
                    np.maximum(maximum - median, 0.0),
                ])

            controlled_plot = selected_controlled.copy()
            if not controlled_plot.empty:
                controlled_plot = controlled_plot.loc[controlled_plot["claim_eligible"]].copy()
                if "controlled_artifact_audit" in globals():
                    passed_run_dirs = set(
                        controlled_artifact_audit.loc[
                            controlled_artifact_audit["status"].eq("PASS"), "run_dir"
                        ].astype(str)
                    )
                    controlled_plot = controlled_plot.loc[
                        controlled_plot["run_dir"].astype(str).isin(passed_run_dirs)
                    ].copy()

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

            # Accuracy-only evidence from exact timed systems and canonical timed solutions.
            prediction_plot = (
                selected_prediction.loc[selected_prediction["prediction_claim_eligible"]].copy()
                if not selected_prediction.empty else pd.DataFrame()
            )
            if not prediction_plot.empty:
                prediction_plot["audit_case_label"] = prediction_plot.apply(
                    lambda row: f"{row['dataset_family']} {int(row['N']) / 1e6:g}M",
                    axis=1,
                )
                prediction_key = ["case_id", "method"]
                if bool(prediction_plot.duplicated(prediction_key, keep=False).any()):
                    raise RuntimeError("Prediction audit has duplicate case/method rows.")
                audit_cases = list(dict.fromkeys(prediction_plot["audit_case_label"].astype(str)))
                audit_methods = [
                    method for method in METHOD_ORDER
                    if method != "cg" and method in set(prediction_plot["method"])
                ]
                x = np.arange(len(audit_cases), dtype=float)
                width = min(0.16, 0.75 / max(len(audit_methods), 1))
                fig, ax = plt.subplots(figsize=(max(10.5, 1.8 * len(audit_cases)), 5.3))
                for method_index, method in enumerate(audit_methods):
                    rows = prediction_plot.loc[
                        prediction_plot["method"].eq(method)
                    ].set_index("audit_case_label")
                    values = np.asarray([
                        float(rows.loc[label, "test_rmse_difference_ppm_vs_cg"])
                        if label in rows.index else np.nan
                        for label in audit_cases
                    ])
                    positions = x + (method_index - (len(audit_methods) - 1) / 2) * width
                    ax.bar(
                        positions, values, width=width,
                        color=METHOD_COLORS.get(method), label=method,
                    )
                ax.axhline(0.0, color="black", lw=1)
                ax.set_xticks(x, audit_cases, rotation=20, ha="right")
                ax.set_ylabel("test RMSE difference vs CG (ppm)")
                ax.set_title(
                    "Prediction equivalence using canonical timed solutions "
                    "(accuracy only; lower absolute difference is better)"
                )
                ax.grid(axis="y", alpha=.25)
                ax.legend(ncol=min(3, max(len(audit_methods), 1)), frameon=False)
                fig.tight_layout()
                prediction_plot_path = DRIVE_RUN_ROOT / "prediction_accuracy_vs_cg.png"
                fig.savefig(prediction_plot_path, dpi=180, bbox_inches="tight")
                GENERATED_PLOT_PATHS.append(prediction_plot_path)
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
                    "case_id", "system_id", "weights_sha256", "gf_sha256", "rhs_sha256",
                    "rhs_storage_sha256", "cfg_reg_lambda", "device_name", "compute_capability",
                    "timing_runtime_sha256",
                ]].drop_duplicates()
                BOX_BUDGET_SYSTEM_MATCH = bool(
                    len(case_systems) == 3
                    and all(case_systems[field].notna().all() and case_systems[field].nunique() == 1
                            for field in (
                                "system_id", "weights_sha256", "gf_sha256", "rhs_sha256",
                                "rhs_storage_sha256", "cfg_reg_lambda", "device_name",
                                "compute_capability", "timing_runtime_sha256",
                            ))
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
                    speed_values, speed_yerr = minmax_error(
                        default_budget, "cold_speedup_median", "cold_speedup_min", "cold_speedup_max"
                    )
                    iteration_values, iteration_yerr = minmax_error(
                        default_budget, "iterations_median", "iterations_min", "iterations_max"
                    )
                    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
                    axes[0].errorbar(
                        x_values, speed_values, yerr=speed_yerr,
                        marker="o", capsize=3, color=METHOD_COLORS["default"],
                    )
                    axes[0].axhline(1.0, color="black", lw=1, ls="--")
                    axes[0].set_ylabel("paired cold speedup over CG")
                    axes[1].errorbar(
                        x_values, iteration_values, yerr=iteration_yerr,
                        marker="o", capsize=3, color=METHOD_COLORS["default"],
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
                    budget_audit_label = (
                        "fixed-system" if BOX_BUDGET_SYSTEM_MATCH
                        else "INVALID fixed-system audit: systems differ"
                    )
                    fig.suptitle(
                        f"Winnebago 10M {budget_audit_label} active-box budget ablation "
                        "(whiskers=min–max)"
                    )
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
            scale_method_availability_rows = []
            if not scale_plot.empty:
                for (output_group, family), family_frame in scale_plot.groupby(
                    ["output_group", "dataset_family"], sort=True
                ):
                    union_methods = {
                        str(method) for method in family_frame["method"].dropna().astype(str)
                    }
                    for n_train, n_frame in family_frame.groupby("N", sort=True):
                        available_methods = {
                            str(method) for method in n_frame["method"].dropna().astype(str)
                        }
                        ordered_methods = [
                            method for method in METHOD_ORDER if method in available_methods
                        ] + sorted(available_methods - set(METHOD_ORDER))
                        scale_method_availability_rows.append({
                            "output_group": output_group,
                            "dataset_family": family,
                            "N": int(n_train),
                            "methods": ",".join(ordered_methods),
                            "method_count": len(available_methods),
                            "is_subset_vs_profile_union": available_methods != union_methods,
                        })
            SCALE_METHOD_AVAILABILITY_PATH = DRIVE_RUN_ROOT / "scale_method_availability.csv"
            pd.DataFrame(scale_method_availability_rows).to_csv(
                SCALE_METHOD_AVAILABILITY_PATH, index=False
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
                    for runtime_field in (
                        "device_name", "compute_capability", "timing_runtime_sha256",
                    ):
                        runtime_values = family_frame[runtime_field].dropna().astype(str)
                        if len(runtime_values) != len(family_frame) or runtime_values.nunique() != 1:
                            raise RuntimeError(
                                f"{output_group}/{family} mixes or lacks {runtime_field}; "
                                "do not join scale points measured on different GPU runtimes."
                            )
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
                    family_union_methods = set(family_frame["method"].dropna().astype(str))
                    method_subset_notes = []
                    for n_train, n_frame in family_frame.groupby("N", sort=True):
                        available_methods = set(n_frame["method"].dropna().astype(str))
                        if available_methods != family_union_methods:
                            ordered_available = [
                                method for method in METHOD_ORDER if method in available_methods
                            ] + sorted(available_methods - set(METHOD_ORDER))
                            method_subset_notes.append(
                                f"{int(n_train) / 1e6:g}M: {', '.join(ordered_available)}"
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
                        y_values, yerr = minmax_error(
                            group, "cold_speedup_median", "cold_speedup_min", "cold_speedup_max"
                        )
                        linestyle = "-" if np.unique(x_values).size >= 2 else "None"
                        ax.errorbar(
                            x_values, y_values, yerr=yerr, marker="o", ls=linestyle, capsize=3,
                            color=METHOD_COLORS.get(method), label=method,
                        )
                    ax.set_xscale("log")
                    ax.axhline(1.0, color="black", lw=1, ls="--")
                    ax.set_title(family)
                    ax.set_xlabel("training rows N")
                    ax.grid(True, alpha=.25)
                    if method_subset_notes:
                        ax.text(
                            0.02, 0.02,
                            "method subsets at large N\n" + "\n".join(method_subset_notes),
                            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
                            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
                        )
                axes[0][0].set_ylabel("paired cold speedup over CG")
                handles = [
                    Line2D([0], [0], marker="o", color=METHOD_COLORS[m], label=m)
                    for m in METHOD_ORDER if m in set(profile_frame["method"])
                ]
                fig.legend(handles=handles, ncol=min(3, len(handles)), frameon=False,
                           loc="upper center", bbox_to_anchor=(0.5, 1.04))
                fig.suptitle(
                    f"Controlled scale: {output_group} (median; whiskers=min–max)", y=1.11
                )
                fig.tight_layout()
                scale_plot_path = DRIVE_RUN_ROOT / f"{output_group}_cold_speedup.png"
                fig.savefig(scale_plot_path, dpi=180, bbox_inches="tight")
                GENERATED_PLOT_PATHS.append(scale_plot_path)
                plt.show()

                setup_profile = profile_frame.loc[
                    profile_frame["setup_inclusive_timing_eligible"].astype(str).str.lower().eq("true")
                ].copy()
                if not setup_profile.empty:
                    fig_setup, setup_axes = plt.subplots(
                        1, len(families), figsize=(5.2 * len(families), 4.6),
                        squeeze=False, sharey=True,
                    )
                    for ax_setup, family in zip(setup_axes[0], families):
                        family_setup = setup_profile.loc[
                            setup_profile["dataset_family"].eq(family)
                        ]
                        for method in METHOD_ORDER:
                            group = family_setup.loc[
                                family_setup["method"].eq(method)
                            ].sort_values("N")
                            if group.empty:
                                continue
                            x_values = pd.to_numeric(
                                group["N"], errors="raise"
                            ).to_numpy(float)
                            y_values = pd.to_numeric(
                                group["shared_fourier_setup_plus_method_speedup_median"],
                                errors="raise",
                            ).to_numpy(float)
                            ax_setup.plot(
                                x_values, y_values, marker="o",
                                ls="-" if np.unique(x_values).size >= 2 else "None",
                                color=METHOD_COLORS.get(method), label=method,
                            )
                        ax_setup.set_xscale("log")
                        ax_setup.axhline(1.0, color="black", lw=1, ls="--")
                        ax_setup.set_title(family)
                        ax_setup.set_xlabel("training rows N")
                        ax_setup.grid(True, alpha=.25)
                    setup_axes[0][0].set_ylabel(
                        "setup-inclusive median speedup over CG"
                    )
                    fig_setup.legend(
                        handles=handles, ncol=min(3, len(handles)), frameon=False,
                        loc="upper center", bbox_to_anchor=(0.5, 1.04),
                    )
                    fig_setup.suptitle(
                        f"Controlled scale: {output_group} (shared Fourier setup included; "
                        "artifact-reused setup timings excluded)", y=1.11,
                    )
                    fig_setup.tight_layout()
                    setup_plot_path = (
                        DRIVE_RUN_ROOT / f"{output_group}_setup_inclusive_speedup.png"
                    )
                    fig_setup.savefig(setup_plot_path, dpi=180, bbox_inches="tight")
                    GENERATED_PLOT_PATHS.append(setup_plot_path)
                    plt.show()
                else:
                    print(
                        f"No eligible setup-inclusive timing rows for {output_group}; "
                        "solver-only scale plot remains valid."
                    )

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

            结果目录只包含配置、数据 manifest 快照、CSV/JSON 和图，不复制大数据。自动断开默认关闭；只有下格确认所有期望 case 均有 `run_complete.json`、prediction audit 严格复用 timing system/solutions，且统一索引已写入 Drive 后才允许开启。
            """
        )
    )
    cells.append(
        _code(
            r"""
            CAMPAIGN_EXECUTION_FINISHED_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            current_campaign_invocation_elapsed_seconds = (
                time.perf_counter() - CAMPAIGN_EXECUTION_STARTED_PERF_COUNTER
            )
            current_notebook_invocation_elapsed_seconds = (
                time.perf_counter() - NOTEBOOK_STARTED_PERF_COUNTER
            )
            prior_final_manifest = {}
            prior_manifest_path = DRIVE_RUN_ROOT / "colab_run_manifest.json"
            if prior_manifest_path.is_file():
                try:
                    prior_final_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
                    if not isinstance(prior_final_manifest, dict):
                        raise TypeError("prior final manifest is not an object")
                except (OSError, json.JSONDecodeError, TypeError):
                    prior_final_manifest = {}
                    print("WARNING: prior final manifest is unreadable; first-run total cannot be carried forward.")
            non_smoke_jobs = [row for row in campaign_job_rows if row.get("job_id") != "plumbing_smoke"]

            def formal_campaign_job_passed(row):
                if str(row.get("profile")) in {
                    STAGE1_SCALE_PROFILE,
                    "robustness_at_selected_target",
                    STAGE1_FAMILY_SCALE_PROFILE,
                    STAGE1_FAMILY_KERNEL_PROFILE,
                    "family_robustness_at_selected_target",
                }:
                    return bool(
                        row.get("artifact_complete")
                        and row.get("scientific_eligible")
                        and str(row.get("status")) in {
                            "claim_eligible_complete",
                            "complete_with_resource_limits",
                            "complete_with_usability_ineligible_methods",
                        }
                    )
                return str(row.get("status")) == "PASS"

            all_campaign_jobs_fresh = bool(
                non_smoke_jobs
                and all(
                    row.get("invocation_mode") == "executed"
                    and formal_campaign_job_passed(row)
                    for row in non_smoke_jobs
                )
            )
            if all_campaign_jobs_fresh:
                first_run_campaign_elapsed_seconds = current_campaign_invocation_elapsed_seconds
                first_run_campaign_elapsed_source = "current invocation"
            elif (
                prior_final_manifest.get("run_verified") is True
                and prior_final_manifest.get("first_run_campaign_elapsed_seconds") is not None
            ):
                first_run_campaign_elapsed_seconds = float(
                    prior_final_manifest["first_run_campaign_elapsed_seconds"]
                )
                first_run_campaign_elapsed_source = "preserved prior final manifest"
            else:
                first_run_campaign_elapsed_seconds = None
                first_run_campaign_elapsed_source = (
                    "unavailable: one or more artifacts predated the timing ledger"
                )
            resume_summary = {
                "executed_job_count": sum(
                    row.get("invocation_mode") == "executed" for row in campaign_job_rows
                ),
                "resumed_job_count": sum(
                    row.get("invocation_mode") == "resumed_existing" for row in campaign_job_rows
                ),
                "mixed_job_count": sum(
                    row.get("invocation_mode") == "mixed_execute_and_resume" for row in campaign_job_rows
                ),
                "resumed_case_count": sum(
                    int(row.get("resumed_case_count", 0) or 0) for row in campaign_job_rows
                ),
                "executed_case_count": sum(
                    int(row.get("executed_case_count", 0) or 0) for row in campaign_job_rows
                ),
            }
            final_manifest = {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "notebook_invocation_started_utc": NOTEBOOK_STARTED_UTC,
                "campaign_execution_started_utc": CAMPAIGN_EXECUTION_STARTED_UTC,
                "campaign_execution_finished_utc": CAMPAIGN_EXECUTION_FINISHED_UTC,
                "current_notebook_invocation_elapsed_seconds": current_notebook_invocation_elapsed_seconds,
                "current_campaign_invocation_elapsed_seconds": current_campaign_invocation_elapsed_seconds,
                "first_run_campaign_elapsed_seconds": first_run_campaign_elapsed_seconds,
                "first_run_campaign_elapsed_seconds_source": first_run_campaign_elapsed_source,
                "resume_summary": resume_summary,
                "elapsed_time_semantics": {
                    "campaign_jobs.elapsed_seconds": (
                        "wall time of the current invocation; for resumed_existing it is only "
                        "resume validation and must not be reported as first-run experiment time"
                    ),
                    "campaign_jobs.first_run_elapsed_seconds": (
                        "exact fresh suite wall time when observed, otherwise preserved or null; "
                        "never a per-method timing claim"
                    ),
                    "paper_method_timing": (
                        "use the canonical Stage-1 verified summaries and "
                        "stage2_fixed_ab_solver_summary.csv matched-repeat paired totals; "
                        "never use raw matched_summary.csv or campaign wall time"
                    ),
                },
                "git_sha": GIT_SHA,
                "runtime": runtime_info,
                "data_manifest": str(DATA_MANIFEST),
                "data_manifest_sha256": DATA_MANIFEST_SHA256,
                "data_manifest_snapshot": str(DATA_MANIFEST_SNAPSHOT),
                "data_manifest_snapshot_sha256": hashlib.sha256(
                    DATA_MANIFEST_SNAPSHOT.read_bytes()
                ).hexdigest(),
                "selected_data_bundles": DATA_BUNDLES,
                "run_tag": RUN_TAG,
                "active_sizes": [int(n) for n in ACTIVE_SIZES],
                "independent_manitowoc_300m_status": (
                    "requested" if RUN_MANITOWOC_SCALE and 300_000_000 in ACTIVE_SIZES
                    else "pending_data_generation_and_prefix_verification"
                ),
                "real_data_300m_route": (
                    "Winnebago archived-exact; independent Manitowoc is currently 10M only"
                ),
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
                "report_ingest_warnings": str(REPORT_INGEST_WARNINGS_PATH),
                "prediction_artifact_audit": str(PREDICTION_ARTIFACT_AUDIT_PATH),
                "prediction_accuracy_summary": str(PREDICTION_ACCURACY_SUMMARY_PATH),
                "prediction_accuracy_plot": str(DRIVE_RUN_ROOT / "prediction_accuracy_vs_cg.png"),
                "scale_method_availability": str(SCALE_METHOD_AVAILABILITY_PATH),
                "stage1_protocol_family": "end_to_end_krr",
                "stage1_suite_config": str(STAGE1_SUITE_CONFIG),
                "stage1_synthetic_data_family_manifest": (
                    str(synthetic_family_manifest_path)
                    if synthetic_family_manifest_path.is_file() else None
                ),
                "stage1_synthetic_data_family_manifest_sha256": (
                    hashlib.sha256(
                        synthetic_family_manifest_path.read_bytes()
                    ).hexdigest()
                    if synthetic_family_manifest_path.is_file() else None
                ),
                "stage1_case_index": str(STAGE1_CASE_INDEX_PATH),
                "stage1_scale_summary": str(STAGE1_SCALE_SUMMARY_PATH),
                "stage1_robustness_summary": str(STAGE1_ROBUSTNESS_SUMMARY_PATH),
                "stage1_family_methods": list(STAGE1_FAMILY_METHODS),
                "stage1_family_scale_all_methods": str(STAGE1_FAMILY_SCALE_ALL_PATH),
                "stage1_family_scale_selected": str(STAGE1_FAMILY_SCALE_SELECTED_PATH),
                "stage1_family_robustness_all_methods": str(STAGE1_FAMILY_ROBUSTNESS_ALL_PATH),
                "stage1_family_robustness_selected": str(STAGE1_FAMILY_ROBUSTNESS_SELECTED_PATH),
                "stage1_family_kernel_all_methods": str(STAGE1_FAMILY_KERNEL_ALL_PATH),
                "stage1_family_kernel_selected": str(STAGE1_FAMILY_KERNEL_SELECTED_PATH),
                "archived_original_full_results": str(ARCHIVED_FULL_RESULTS_PATH),
                "archived_original_operation_selected": str(ARCHIVED_OPERATION_SELECTED_PATH),
                "archived_regime_map_two_family": str(ARCHIVED_REGIME_MAP_PATH),
                "stage1_target_regime": (
                    str(STAGE1_TARGET_PATH) if END_TO_END_TARGET is not None else None
                ),
                "stage1_target_selection_error": target_selection_error,
                "stage1_methods": list(STAGE1_METHOD_ORDER),
                "stage2_protocol_family": "controlled_fixed_system",
                "stage2_headline_timing": (
                    "solver_total_seconds = selection + preconditioner build + solve"
                ),
                "stage2_method_candidates": list(STAGE2_METHODS),
                "stage2_methods": list(STAGE2_FORMAL_METHODS),
                "stage2_feasibility": (
                    str(STAGE2_FEASIBILITY_PATH)
                    if STAGE2_FEASIBILITY is not None else None
                ),
                "stage2_feasibility_decision": STAGE2_FEASIBILITY,
                "stage2_solver_summary": (
                    str(STAGE2_SOLVER_SUMMARY_PATH)
                    if TWO_STAGE_FORMAL_REPORT_RESULT is not None
                    and STAGE2_SOLVER_SUMMARY_PATH.is_file()
                    else None
                ),
                "two_stage_formal_report": (
                    str(TWO_STAGE_FORMAL_REPORT_ROOT / "report_manifest.json")
                    if TWO_STAGE_FORMAL_REPORT_RESULT is not None else None
                ),
                "two_stage_claim_audit": (
                    str(TWO_STAGE_FORMAL_REPORT_ROOT / "claim_audit.csv")
                    if TWO_STAGE_FORMAL_REPORT_RESULT is not None else None
                ),
                "generated_plots": [
                    str(path) for path in [
                        *STAGE1_GENERATED_PLOT_PATHS,
                        *STAGE2_GENERATED_PLOT_PATHS,
                        *TWO_STAGE_GENERATED_PLOT_PATHS,
                        *GENERATED_PLOT_PATHS,
                    ]
                ],
                "expected_prediction_case_count": int(expected_prediction_case_count),
                "box_budget_fixed_system_match": BOX_BUDGET_SYSTEM_MATCH,
            }
            legacy_complete = all(
                (DRIVE_RUN_ROOT / "legacy_archived_pipeline" / group / "_SUCCESS.json").is_file()
                for group in RUN_LEGACY_GROUPS
            )
            expected_selected_controlled_pairs = set()
            selected_config_method_schema_valid = True
            for record in selected_case_records:
                config_path = Path(record["run_dir"]) / "experiment_config.json"
                try:
                    selected_config = json.loads(config_path.read_text(encoding="utf-8"))
                    if not isinstance(selected_config, dict):
                        raise TypeError("experiment_config is not an object")
                    methods = selected_config.get("methods")
                    if (
                        not isinstance(methods, list)
                        or not methods
                        or len({str(method) for method in methods}) != len(methods)
                    ):
                        raise ValueError("invalid experiment_config.methods")
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    selected_config_method_schema_valid = False
                    continue
                run_dir_key = str(Path(record["run_dir"]).resolve())
                expected_selected_controlled_pairs.update(
                    (run_dir_key, str(method)) for method in methods
                )
            observed_selected_controlled_pairs = (
                set(
                    zip(
                        selected_controlled["run_dir"].astype(str),
                        selected_controlled["method"].astype(str),
                    )
                )
                if not selected_controlled.empty else set()
            )
            selected_controlled_summary_complete = bool(
                selected_config_method_schema_valid
                and observed_selected_controlled_pairs
                == expected_selected_controlled_pairs
                and len(selected_controlled) == len(expected_selected_controlled_pairs)
                and (
                    selected_controlled.empty
                    or selected_controlled["claim_eligible"].eq(True).all()
                )
            )
            final_manifest["selected_controlled_summary_complete"] = (
                selected_controlled_summary_complete
            )
            final_manifest["expected_selected_controlled_method_row_count"] = int(
                len(expected_selected_controlled_pairs)
            )
            controlled_complete = (
                expected_controlled_case_count == 0
                or (
                    len(controlled_artifact_audit) == expected_controlled_case_count
                    and controlled_artifact_audit["status"].eq("PASS").all()
                    and selected_controlled_summary_complete
                )
            )
            prediction_validation_frame = pd.DataFrame(prediction_validation_rows)
            expected_prediction_summary_pairs = {
                (str(row.get("case_id")), method)
                for row in prediction_validation_rows
                for method in str(row.get("required_methods", "")).split(",")
                if method
            }
            observed_prediction_summary_pairs = (
                set(
                    zip(
                        selected_prediction["case_id"].astype(str),
                        selected_prediction["method"].astype(str),
                    )
                )
                if not selected_prediction.empty else set()
            )
            prediction_summary_complete = bool(
                not RUN_PREDICTION_AUDIT
                or (
                    expected_prediction_case_count > 0
                    and expected_prediction_summary_pairs
                    and observed_prediction_summary_pairs
                    == expected_prediction_summary_pairs
                    and len(selected_prediction)
                    == len(expected_prediction_summary_pairs)
                    and selected_prediction["prediction_claim_eligible"].eq(True).all()
                )
            )
            prediction_artifacts_pass = bool(
                len(prediction_validation_frame) == expected_prediction_case_count
                and not prediction_validation_frame.empty
                and prediction_validation_frame["status"].astype(str).str.startswith("PASS").all()
                and prediction_validation_frame["audit_pass"].eq(True).all()
                and prediction_validation_frame["exact_timing_system_match"].eq(True).all()
                and prediction_validation_frame["timing_solutions_reused"].eq(True).all()
                and prediction_validation_frame["timing_solution_hashes_verified"].eq(True).all()
                and prediction_validation_frame["zero_audit_solves"].eq(True).all()
                and prediction_summary_complete
            )
            final_manifest["prediction_summary_complete"] = prediction_summary_complete
            final_manifest["prediction_artifacts_pass"] = prediction_artifacts_pass
            final_manifest["prediction_audits_verified"] = prediction_artifacts_pass
            prediction_complete = (
                not RUN_PREDICTION_AUDIT
                or (
                    expected_prediction_case_count > 0
                    and len(prediction_outputs) == expected_prediction_case_count
                    and all(
                        (Path(path) / PREDICTION_AUDIT_JSON_FILENAME).is_file()
                        and (Path(path) / PREDICTION_AUDIT_CSV_FILENAME).is_file()
                        and (Path(path) / PREDICTION_AUDIT_COMPLETION_FILENAME).is_file()
                        for path in prediction_outputs
                    )
                    and prediction_artifacts_pass
                )
            )
            selected_output_groups = (
                set(selected_controlled["output_group"].astype(str))
                if not selected_controlled.empty else set()
            )
            expected_plot_paths = []
            if RUN_STAGE1_FAMILY_SCALE and not stage1_family_scale_selected.empty:
                expected_plot_paths.append(
                    DRIVE_RUN_ROOT / "stage1_two_family_scale_10m_300m.png"
                )
            if RUN_STAGE1_END_TO_END_KRR and not stage1_scale_summary.empty:
                expected_plot_paths.extend([
                    DRIVE_RUN_ROOT / "stage1_krr_train_total_10m_300m.png",
                    DRIVE_RUN_ROOT / "stage1_krr_accuracy_tradeoff.png",
                ])
            if END_TO_END_TARGET is not None:
                expected_plot_paths.append(
                    DRIVE_RUN_ROOT / "stage1_krr_setup_solving_breakdown.png"
                )
            if RUN_STAGE1_ROBUSTNESS and not stage1_robustness_summary.empty:
                expected_plot_paths.append(
                    DRIVE_RUN_ROOT / "stage1_krr_robustness.png"
                )
            if not stage2_solver_summary.empty:
                expected_plot_paths.append(
                    DRIVE_RUN_ROOT / "stage2_fixed_ab_solver_total.png"
                )
            expected_plot_paths.extend(TWO_STAGE_GENERATED_PLOT_PATHS)
            if "paper_10m" in selected_output_groups:
                expected_plot_paths.extend([
                    DRIVE_RUN_ROOT / "controlled_10m_method_speedup.png",
                    DRIVE_RUN_ROOT / "controlled_10m_speed_memory_pareto.png",
                ])
            if "winnebago_oat_n10m" in selected_output_groups:
                expected_plot_paths.append(
                    DRIVE_RUN_ROOT / "winnebago_oat_10m_speed_memory.png"
                )
            if "winnebago_box_budget_n10m" in selected_output_groups:
                expected_plot_paths.append(
                    DRIVE_RUN_ROOT / "winnebago_box_budget_10m.png"
                )
            selected_scale_groups = sorted(
                selected_output_groups.intersection(SCALE_OUTPUT_GROUPS)
            )
            setup_timing_coverage_complete = True
            for output_group in selected_scale_groups:
                expected_plot_paths.extend([
                    DRIVE_RUN_ROOT / f"{output_group}_cold_speedup.png",
                    DRIVE_RUN_ROOT / f"{output_group}_setup_inclusive_speedup.png",
                ])
                eligible_setup_rows = controlled_plot.loc[
                    controlled_plot["output_group"].astype(str).eq(output_group)
                    & controlled_plot["setup_inclusive_timing_eligible"].astype(str).str.lower().eq("true")
                ]
                expected_setup_run_dirs = {
                    str(Path(record["run_dir"]).resolve())
                    for record in selected_case_records
                    if str(record["output_group"]) == output_group
                }
                observed_setup_run_dirs = set(
                    eligible_setup_rows["run_dir"].astype(str)
                )
                if observed_setup_run_dirs != expected_setup_run_dirs:
                    setup_timing_coverage_complete = False
            if RUN_PREDICTION_AUDIT:
                expected_plot_paths.append(
                    DRIVE_RUN_ROOT / "prediction_accuracy_vs_cg.png"
                )
            generated_plot_keys = {
                str(Path(path).resolve())
                for path in [
                    *STAGE1_GENERATED_PLOT_PATHS,
                    *STAGE2_GENERATED_PLOT_PATHS,
                    *TWO_STAGE_GENERATED_PLOT_PATHS,
                    *GENERATED_PLOT_PATHS,
                ]
            }
            plot_artifacts_complete = bool(
                setup_timing_coverage_complete
                and all(
                    path.is_file() and str(path.resolve()) in generated_plot_keys
                    for path in expected_plot_paths
                )
            )
            final_manifest["expected_plot_artifacts"] = [
                str(path) for path in expected_plot_paths
            ]
            final_manifest["setup_timing_coverage_complete"] = (
                setup_timing_coverage_complete
            )
            final_manifest["plot_artifacts_complete"] = plot_artifacts_complete
            stage1_scale_job_rows = [
                row for row in stage1_campaign_rows
                if row.get("profile") == STAGE1_SCALE_PROFILE
            ]
            stage1_scale_artifacts_complete = bool(
                not RUN_STAGE1_END_TO_END_KRR
                or (
                    len(completed_stage1_scale_items) == len(stage1_scale_plan)
                    and len(stage1_scale_summary)
                    == len(stage1_scale_plan) * len(STAGE1_METHOD_ORDER)
                    and len(stage1_scale_job_rows) == len(stage1_scale_plan)
                    and all(bool(row.get("artifact_complete")) for row in stage1_scale_job_rows)
                )
            )
            stage1_scale_scientifically_eligible = bool(
                not RUN_STAGE1_END_TO_END_KRR
                or (
                    stage1_scale_job_rows
                    and all(
                        formal_campaign_job_passed(row)
                        for row in stage1_scale_job_rows
                    )
                )
            )
            stage1_scale_complete = bool(
                stage1_scale_artifacts_complete
                and stage1_scale_scientifically_eligible
            )
            stage1_robustness_job_rows = [
                row for row in stage1_campaign_rows
                if row.get("profile") == "robustness_at_selected_target"
            ]
            stage1_robustness_artifacts_complete = bool(
                not RUN_STAGE1_ROBUSTNESS
                or (
                    END_TO_END_TARGET is not None
                    and len(completed_stage1_robustness_items) == len(stage1_robustness_plan)
                    and len(stage1_robustness_summary)
                    == len(stage1_robustness_plan) * len(STAGE1_METHOD_ORDER)
                    and len(stage1_robustness_job_rows) == len(stage1_robustness_plan)
                    and all(
                        bool(row.get("artifact_complete"))
                        for row in stage1_robustness_job_rows
                    )
                )
            )
            stage1_robustness_scientifically_eligible = bool(
                not RUN_STAGE1_ROBUSTNESS
                or (
                    stage1_robustness_job_rows
                    and all(
                        formal_campaign_job_passed(row)
                        for row in stage1_robustness_job_rows
                    )
                )
            )
            stage1_robustness_complete = bool(
                stage1_robustness_artifacts_complete
                and stage1_robustness_scientifically_eligible
            )
            family_profile_specs = (
                (
                    RUN_STAGE1_FAMILY_SCALE,
                    STAGE1_FAMILY_SCALE_PROFILE,
                    stage1_family_scale_plan,
                    completed_stage1_family_scale_items,
                    stage1_family_scale_all,
                ),
                (
                    RUN_STAGE1_FAMILY_KERNEL,
                    STAGE1_FAMILY_KERNEL_PROFILE,
                    stage1_family_kernel_plan,
                    completed_stage1_family_kernel_items,
                    stage1_family_kernel_all,
                ),
                (
                    RUN_STAGE1_FAMILY_ROBUSTNESS,
                    "family_robustness_at_selected_target",
                    stage1_family_robustness_plan,
                    completed_stage1_family_robustness_items,
                    stage1_family_robustness_all,
                ),
            )
            stage1_family_profiles_complete = True
            stage1_family_profile_audit = {}
            for enabled, profile_name, plan, completed, summary in family_profile_specs:
                jobs = [
                    row for row in stage1_campaign_rows
                    if row.get("profile") == profile_name
                ]
                profile_complete = bool(
                    not enabled
                    or (
                        (
                            profile_name != "family_robustness_at_selected_target"
                            or END_TO_END_TARGET is not None
                        )
                        and len(plan) > 0
                        and len(completed) == len(plan)
                        and len(summary) == len(plan) * len(STAGE1_FAMILY_METHODS)
                        and len(jobs) == len(plan)
                        and all(formal_campaign_job_passed(row) for row in jobs)
                    )
                )
                stage1_family_profile_audit[profile_name] = {
                    "enabled": bool(enabled),
                    "planned_cases": len(plan),
                    "completed_cases": len(completed),
                    "summary_rows": len(summary),
                    "complete": profile_complete,
                }
                stage1_family_profiles_complete &= profile_complete
            stage2_formal_complete = bool(
                not RUN_STAGE2_FIXED_AB_SOLVERS
                or (
                    END_TO_END_TARGET is not None
                    and STAGE2_FEASIBILITY is not None
                    and STAGE2_FEASIBILITY_PATH.is_file()
                    and not stage2_solver_summary.empty
                    and set(stage2_solver_summary["method"].astype(str))
                    == set(STAGE2_FORMAL_METHODS)
                )
            )
            two_stage_report_complete = bool(
                not RUN_STAGE2_FIXED_AB_SOLVERS
                or (
                    TWO_STAGE_FORMAL_REPORT_RESULT is not None
                    and (TWO_STAGE_FORMAL_REPORT_ROOT / "report_manifest.json").is_file()
                    and (TWO_STAGE_FORMAL_REPORT_ROOT / "claim_audit.csv").is_file()
                )
            )
            final_manifest["stage1_scale_complete"] = stage1_scale_complete
            final_manifest["stage1_scale_artifacts_complete"] = stage1_scale_artifacts_complete
            final_manifest["stage1_scale_scientifically_eligible"] = stage1_scale_scientifically_eligible
            final_manifest["stage1_robustness_complete"] = stage1_robustness_complete
            final_manifest["stage1_robustness_artifacts_complete"] = stage1_robustness_artifacts_complete
            final_manifest["stage1_robustness_scientifically_eligible"] = stage1_robustness_scientifically_eligible
            final_manifest["stage1_family_profiles_complete"] = stage1_family_profiles_complete
            final_manifest["stage1_family_profile_audit"] = stage1_family_profile_audit
            final_manifest["stage2_formal_complete"] = stage2_formal_complete
            final_manifest["two_stage_report_complete"] = two_stage_report_complete
            workload_requested = bool(
                RUN_STAGE1_END_TO_END_KRR or RUN_STAGE2_FIXED_AB_SOLVERS
                or RUN_LEGACY_GROUPS or selected_profiles or extra_suites
                or RUN_PREDICTION_AUDIT
            )
            mandatory_jobs = [row for row in campaign_job_rows if bool(row.get("mandatory", True))]
            campaign_complete = bool(
                mandatory_jobs and all(formal_campaign_job_passed(row) for row in mandatory_jobs)
            )
            run_verified = bool(
                workload_requested and stage1_scale_complete
                and stage1_robustness_complete and stage1_family_profiles_complete
                and stage2_formal_complete
                and two_stage_report_complete
                and legacy_complete and controlled_complete
                and prediction_complete and campaign_complete and INDEX_PATH.is_file()
                and plot_artifacts_complete
                and (BOX_BUDGET_SYSTEM_MATCH is not False)
            )
            if not run_verified and first_run_campaign_elapsed_source == "current invocation":
                first_run_campaign_elapsed_seconds = None
                first_run_campaign_elapsed_source = (
                    "unavailable: fresh campaign did not pass final verification"
                )
                final_manifest["first_run_campaign_elapsed_seconds"] = None
                final_manifest["first_run_campaign_elapsed_seconds_source"] = (
                    first_run_campaign_elapsed_source
                )
            final_manifest["campaign_complete"] = campaign_complete
            final_manifest["run_verified"] = run_verified
            FINAL_MANIFEST_PATH = DRIVE_RUN_ROOT / "colab_run_manifest.json"
            final_manifest_partial = FINAL_MANIFEST_PATH.with_suffix(".json.partial")
            final_manifest_partial.write_text(
                json.dumps(final_manifest, indent=2), encoding="utf-8"
            )
            final_manifest_partial.replace(FINAL_MANIFEST_PATH)
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

            1. 两族 profiles 的每个 `pipeline_summary.csv` 必须含 EFGP-CG、standard full-eig、显式 binned inverse、显式 binned active-eig 四条；`stage1_family_*_selected.csv` 每个 case 必须留下 EFGP-CG、inverse、active-box-EigenPro 三个 reporting families。次级 broad matrix 仍严格含原六个 true end-to-end KRR 方法。
            2. 10M/30M/100M/300M 的失败/resource-limit 行原样保留，不以 pilot 或子采样算法冒充大规模结果。
            3. `selected_target_regime.json` 必须来自冻结规则；没有合格 target 时检查 `target_regime_rejections.json`，不得事后换点。
            4. \(\lambda\)、\(\ell\)、box budget、dataset robustness 只在 target 冻结后 materialize；两族 profile 与次级六-pipeline profile 分目录运行，不能用 default 自动路由代替两族结果。
            5. Stage 2 每个 case 的 `system_manifest.json` 必须有 `system_unchanged=true` 和完整 weights/Gf/RHS/λ 哈希；方法仅为 solver/preconditioner family。
            6. `stage2_feasibility.json` 必须在 Stage 2 timing 前生成；mandatory 五方法恒为 feasible，`active-inverse` 仅在冻结 active-box upper bound 不超过 `inverse_max_size` 时运行。
            7. Stage 2 headline 必须读取 `solver_total_seconds`（selection + preconditioner build + solve），不能用 iteration-only 或排除 build 的 solve-only 数字代替。
            8. `nystrom-krr` / `rpcholesky-krr` 只属于 Stage 1；Fourier adaptations 即使手动运行，也必须使用 `fourier-*-precond` 标签并排除在正式两阶段图外。
            9. Prediction audit 同时有 JSON、CSV 和 completion manifest，严格复用 timed system/solutions，audit solve count=0。
            10. Appendix audit 必须同时包含 `archived_original_full_results.csv`、`archived_original_operation_selected.csv`、`archived_regime_map_two_family.csv` 与 `archived_selection_protocol.json`；旧 timing 只用于历史 regime map，不进入 current replay median。
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
