# GPU 迁移：分步对照与进度控制

本文件与仓库根目录下的 `efgp_eigenpro_py/algorithm_mapping.md` 同一思路：**按算法步骤拆解任务**，并标明**算法上在算什么**、**CPU 侧参照实现落在哪些模块**、**GPU 侧计划落在本目录哪些文件**，便于你逐步核对与打勾完成。

> 约定：`efgp_eigenpro_py/gpu/` 为隔离区；CPU 主路径（`efgp_solver.py` 等）在未验收前尽量不改行为，只在 GPU 子包内迭代。

---

## 总览：三版目标与完成标准

| 版本 | 算法闭环 | 完成标准（建议） |
|------|----------|------------------|
| **V1** | `top_q=0`：`precompute → apply_A → plain CG → predict` 全在 GPU 数据上跑通 | 与 CPU：`mtot` 一致；`iters` / `relres` 接近；预测 `rms` 接近；hard case **wall-clock** 明显下降 |
| **V2** | V1 + **仅** `P(v)` 在 GPU；`U, θ, μ` 仍在 CPU 估好再上传 | `q∈{32,48,64}` hard regime 下 solve 常数进一步降；数值与 CPU PCG 对齐 |
| **V3** | V2 + **top-q eigenspace** 在 GPU（block matvec + 子空间迭代） | `eig` 墙钟明显下降；一次算 `q_max` 再切片复用，避免重复 eig 成本 |

**本目录文件状态**

| 文件 | 职责 |
|------|------|
| `backends.py` | `xp / fft / linalg / nufft` 后端选择与组装 |
| `contexts.py` | GPU 常驻数组与工作区（避免反复分配） |
| `v1_ops.py` | V1：`gpu_precompute_v1`（统一 NUFFT adapter：`cufinufft` 优先，回退 CPU `nufft_ops`，支持 chunked）、`apply_A_v1`（GPU Toeplitz+FFT）、`solve_beta_plain_cg_v1`（GPU CG，支持 `t_matvec_avg` 统计）、`predict_v1`（统一 type-2 adapter，输出实值）；剩余工作主要是 benchmark 验收与进一步性能优化 |
| `v2_preconditioner.py` | V2：`P(v)` 与预条件数据上传 |
| `v3_eigenspace.py` | V3：GPU 上 top-q 子空间估计 |
| `versions.py` | 按版本串联整条链的 orchestration（Python 侧调度） |

---

## 全局前置：后端抽象（对应你规划里的 Step 0）

**算法上做什么**  
不改变数学，只把「数组在哪、FFT/NUFFT/BLAS 调谁」抽成可切换后端，使同一套 orchestration 能在 `numpy` 与 `cupy` 之间切换。

**CPU 参照模块**

- 现状：`efgp_solver.py`、`toeplitz.py`、`nufft_ops.py` 大量直接用 `numpy` / `scipy.fft` / `finufft`。

**GPU 目标模块**

- [x] `backends.py`：`BackendConfig`、`build_gpu_backend_bundle` 导出 `xp`、`fft`（`CuPyFFTOps` 薄封装）、`linalg`（`CuPyLinalgOps`）、`nufft`（可选原始模块）、`device_name` / `has_nufft` / `supports_fp64` / `supports_complex128` 等 capability；构建时做 **CUDA 设备可见性 + 最小分配** 预检
- [x] 约定：`xp.asarray` / `xp.zeros`；`bundle.fft.fftn` / `bundle.fft.ifftn`；`bundle.linalg.norm` / `bundle.linalg.vdot`；NUFFT 须在统一 adapter 中封装（勿直接假设 `bundle.nufft.nufft2d1` 存在）

**验收核对**

- [ ] 无 GPU 环境 import 不崩溃（或给出清晰 `RuntimeError` 说明缺 CuPy）
- [x] `nufft` 后端名可日志打印（便于实验记录）

---

## Step A（建议先做）：`apply_A` 工作区复用 —— 与 CPU 对齐后再迁 GPU

**算法上做什么**  
对任意权重向量 \(u\in\mathbb{C}^M\)，计算  
\(A u = D(F^*F)(Du)+\lambda u\)（见 `algorithm_mapping.md` 步骤 4）。GPU 上最怕每步 `malloc`；CPU 侧若已是 **persistent workspace**，迁移成本最低。

**CPU 参照模块**

- `efgp_solver.py`：`_apply_A`、`_apply_A_hot_1d` / `_apply_A_hot_2d`、`_apply_A_block`
- `toeplitz.py`：`ToeplitzWorkspace`、`apply_toeplitz_fft_nd`、`_ifft_of_fft_times_Gf`（1D/2D 热路径）

**GPU 目标模块**

- [x] `contexts.py`：`GPUOperatorContext` 填满 `pad / fft_buf / ifft_buf / work_vec` 等字段
- [x] `v1_ops.py`：`apply_A_v1` 实现与 CPU 相同的数值路径（先 correctness 再优化）

**验收核对**

- [ ] 随机向量 \(v\)：`||A_{\mathrm{GPU}}v-A_{\mathrm{CPU}}v||/||A_{\mathrm{CPU}}v||\) 在设定容差内
- [x] 记录 `t_matvec_total`、`t_matvec_avg`（与 `benchmark.py` 里统计思路一致）

---

## V1 — Step 1：`precompute` 上 GPU（两次 type-1 + 嵌入 FFT）

**算法上做什么**（对应 `algorithm_mapping.md` 步骤 3）

1. 在 \(J_{2m}\) 上累计 \(v_\ell=\sum_n e^{2\pi i h\langle \ell,x_n\rangle}\)（type-1 NUFFT）  
2. \(\widehat v=\mathrm{FFT}(v)\)（嵌入尺寸与 CPU 一致）  
3. \(b_j=\sum_n \overline{\phi_j(x_n)}y_n\)（type-1 NUFFT）再乘对角权重 \(D\) 的元素

**CPU 参照模块**

- `discretization.py`：`choose_grid_params`（**建议保留 CPU**）、`basis_weights`
- `efgp_solver.py`：`precompute` / `precompute_streaming`（chunk 逻辑参考）
- `nufft_ops.py`：`nufft*1` / `nufftnd1`

**GPU 目标模块**

- [x] `v1_ops.py`：`gpu_precompute_v1`：chunk 上传 `x,y`，在 GPU 上累加 `XtXcol_acc`、`rhs_acc`
- [x] `backends.py`：接入 GPU NUFFT（`cufinufft` 等）或文档化 fallback
- [x] `contexts.py`：`GPUDataContext` 持有 `gf_gpu`、`rhs_gpu`、`weights_gpu_nd` / `weights_gpu_flat`、`x_center_gpu`，`meta` 含 `mtot`、`dim`（= weights 维数）、`h`、`weight_shape`、`gf_shape`、`rhs_shape`

**验收核对**

- [ ] `Gf` 与 `rhs` 与 CPU `PrecomputeState` 逐元素相对误差达标
- [ ] 大 \(N\)：chunk 前后结果与一次性全量（若可跑）一致

---

## V1 — Step 2：`apply_A` 全 GPU 常驻

**算法上做什么**  
同 Step A，但所有中间量留在 GPU，不在迭代中回传主机。

**CPU 参照模块**

- 同 Step A

**GPU 目标模块**

- [x] `v1_ops.py`：`apply_A_v1` 完整实现 + workspace 复用


---

## V1 — Step 3：plain CG 在 GPU（不自洽前不接 `scipy.sparse.linalg.cg`）

**算法上做什么**  
解 \(A\beta=b\)。标准 CG：残差、步长、`Ap`、向量内积、`axpy` 均在 GPU；**迭代循环在 Python** 便于计时与接预条件器。

**CPU 参照模块**

- `linear_solvers.py`：`pcg`（SciPy 包装，作行为参考）
- `benchmark.py`：`cg_basic` / `pcg_solve`（手写 CG 参考与 matvec 计时）

**GPU 目标模块**

- [x] `v1_ops.py` 或新建 `cg_gpu.py`（若你希望单文件职责更清晰）：`cg_plain_gpu`
- [x] `versions.py`：`run_v1_pure_efgp` 用真实 CG 替换占位

**验收核对**


---

## V1 — Step 4：`predict`（type-2 NUFFT）上 GPU

**算法上做什么**（对应 `algorithm_mapping.md` 步骤 8）  
\(\tilde f(x^*)=\sum_{j\in J_m}\beta_j\phi_j(x^*)\)，实现为对 \(\beta\) 加权后的 **type-2 NUFFT**。

**CPU 参照模块**

- `efgp_solver.py`：`predict`
- `nufft_ops.py`：`nufft*2` / `nufftnd2`

**GPU 目标模块**

- [x] `v1_ops.py`：`predict_v1`
- [x] `backends.py`：GPU type-2 路径

**验收核对**

- [ ] 同一 `beta`：`yhat_gpu` 与 `solver.predict` 相对误差达标

---

## V2 — Step 5：预条件器 `P(v)` 上 GPU（eigenspace 仍在 CPU）

**算法上做什么**（对应 `algorithm_mapping.md` 步骤 6–7 中的 \(P_q\)）  
\[
P(v)=v-U_q\bigl((1-\mu/\theta_i))\odot(U_q^*v)\bigr)
\]  
（与 `eigenpro_precond.EigenProPreconditioner.apply` 一致。）

**CPU 参照模块**

- `eigenspace.py`：`estimate_top_eigenspace`（仍在 CPU）
- `eigenpro_precond.py`：`EigenProPreconditioner`（`scale`、`eigvecs_h`、buffer 复用思路）
- `efgp_solver.py` / `model.py`：组装 `precond` 的位置（参考调用关系）

**GPU 目标模块**

- [x] `v2_preconditioner.py`：`build_gpu_preconditioner_data`、`apply_preconditioner_v2`  
- [x] `U` contiguous；`proj/scaled` buffer 复用；优先 `gemv`/`gemm`

**验收核对**

- [ ] 固定 `v`：`||P_{\mathrm{GPU}}(v)-P_{\mathrm{CPU}}(v)||\) 达标
- [ ] PCG 整体与 CPU `model.py` / `linear_solvers.pcg` 路径对齐 hard case

---

## V2 — Step 6：GPU PCG + `P`（仍可选 SciPy 仅作对照，主路径手写）

**算法上做什么**  
解 \(A\beta=b\)，左预条件 `M^{-1}\approx P`（EigenPro）。迭代中每步调用 `apply_A` 与 `apply_P`。

**CPU 参照模块**

- `model.py`：`_pcg_history`、`EFGPRegressor.fit` 中 precond 接入方式
- `linear_solvers.py`：`pcg`

**GPU 目标模块**

- [x] `versions.py`：`run_v2_with_preconditioner_apply` 接入真实 PCG
- [x] `v1_ops.py` + `v2_preconditioner.py`：组合调用

**验收核对**

- [ ] `q` 扫描下迭代次数与 `relres` 与 CPU 可比
- [ ] 记录 `t_precond_total`（若沿用 `benchmark.pcg_solve` 统计风格）

---

## V3 — Step 7：GPU `apply_A_block(V)` + top-q 子空间估计

**算法上做什么**（对应 `algorithm_mapping.md` 步骤 5 + 扩展）  
对块 \(V\in\mathbb{C}^{M\times b}\) 执行 \(AV\)，在一次 FFT/嵌入中批处理多列；在此之上做 **block Lanczos** 或 **randomized subspace iteration**，一次估 \(q_{\max}\)，再切片出 \(q\in\{32,48,64\}\)。

**CPU 参照模块**

- `efgp_solver.py`：`_apply_A_block`
- `toeplitz.py`：`apply_toeplitz_fft_nd_block`
- `eigenspace.py`：`estimate_top_eigenspace`（`subspace_iter` 逻辑参考）

**GPU 目标模块**

- [ ] `v1_ops.py`（或拆分 `toeplitz_gpu.py`）：`apply_A_block_gpu`
- [x] `v3_eigenspace.py`：`estimate_top_eigenspace_v3` 替换占位实现
- [ ] `backends.py`：`linalg` QR / SVD / eigh 调用策略（cuSOLVER）

**验收核对**

- [ ] 与 CPU `eigsh` / dense 真值（小规模 \(M\)）对比特征值/子空间误差
- [ ] 一次 `q_max` 计算后切片复用，墙钟符合预期

---

## 建议的「每周核对」清单（可直接复制到你的笔记）

**数值**

- [ ] `rhs`、`Gf`、`apply_A`、`predict` 四项相对误差
- [ ] `beta` 或 `yhat` 端到端与 CPU 对齐

**性能**

- [ ] `t_precompute`、`t_matvec_avg`、`t_solve`、`t_predict`
- [ ] 确认无意外 D2H/H2D（可用简单计次或 profiler）

**工程**

- [ ] `gpu/` 内不出现散落 `np.`（除显式 `numpy`↔`cupy` 边界转换）
- [ ] 大 \(N\) chunk 路径与全量路径一致性

---

## 与 `algorithm_mapping.md` 的步骤索引对照

| `algorithm_mapping.md` | GPU 本 README 章节 |
|--------------------------|-------------------|
| 步骤 1（选 \(h,m\)） | 全局：网格仍在 CPU → `discretization.py` |
| 步骤 3（precompute） | V1 Step 1 |
| 步骤 4（`apply_A`） | Step A + V1 Step 2 |
| 步骤 5（eigenspace） | V3 Step 7（V2 仍用 CPU `eigenspace.py`） |
| 步骤 6（\(P_q\)） | V2 Step 5 |
| 步骤 7（PCG/Richardson） | V1 Step 3 + V2 Step 6 |
| 步骤 8（predict） | V1 Step 4 |

当你完成某一节时，把该节下的 `- [ ]` 改成 `- [x]` 即可在 Git diff 里直观看到进度（若你希望保持 README 不含勾选状态，也可把清单移到单独 `gpu/TODO.md`，需要时我可以再拆一份）。
