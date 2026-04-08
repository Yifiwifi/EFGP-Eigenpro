# 算法原文与开发任务对照（EFGP + EigenPro 权重空间）
---
For simplicity, we construct a version for Gaussian Kernel first.

## 输入与输出

Input. Training data $\{(x_i,y_i)\}_{i=1}^N$ with $x_i\in\mathbb{R}^d$, a translation-invariant kernel $k$, regularization $\lambda>0$, Fourier tolerance $\varepsilon$, top-eigenspace rank $q$, and stopping tolerance $\rho$.

Output. Approximate predictor

$$
\widetilde{f}(x)=\sum_{j\in J_m}\beta_j \phi_j(x).
$$


- 实现统一的输入数据校验与类型转换（`numpy`/`complex`）。
- 确定 `kernel` 与 `khat` 的归一化约定并写入文档。

---

## 步骤 1

Choose Fourier discretization parameters $(h,m)$.  
Use the error formulas from [barnett2023uniform] to choose the grid spacing $h>0$ and the frequency cutoff $m\in\mathbb{N}$. Define

$$
J_m:=\{-m,-m+1,\dots,m\}^d,
\qquad
M:=|J_m|=(2m+1)^d.
$$



- 在 `discretization.choose_grid_params` 中补上 Barnett2023 统一误差公式。
- 统一空间尺度 $L$ 的定义并文档化。
- 若未来需要支持 $d>3$，补充通用网格生成方法。

---

## 步骤 2

Greengard alg6.2

Construct the Fourier basis and weight-space system.  
Define the basis functions

$$
\phi_j(x)=\sqrt{h^d\widehat{k}(hj)}\,e^{2\pi i h\langle j,x\rangle},
\qquad j\in J_m.
$$

Let the design matrix (but we do not compute them here)

$$
\Phi\in\mathbb{C}^{N\times M}
\qquad
\Phi=FD,
\qquad
\Phi_{n,j}=\phi_j(x_n), F_{n,l}=, D_{l,l}=
$$

where rows are indexed by $n=1,\dots,N$ and columns by $j\in J_m$.  
The weight-space unknown is

$$
\beta\in\mathbb{C}^{M}.
$$

Define the system matrix and right-hand side

$$
A:=\Phi^*\Phi+\lambda I_M \in \mathbb{C}^{M\times M},
\qquad
b:=\Phi^*y\in\mathbb{C}^{M},
$$

where $y=(y_1,\dots,y_N)^\top\in\mathbb{R}^N$ and $I_M\in\mathbb{R}^{M\times M}$.

**拆解任务**

- 在 `kernels.py` 中实现 $\widehat{k}$（SE / Matern），并统一归一化。
- 在 `discretization.basis_weights` 中确保 $h^d\widehat{k}(hj)$ 与实现一致。
- 保持显式矩阵 $\Phi$ 不构造，改用矩阵自由 `A` 乘法。

---
## 步骤3

We complete Greengard alg6.3 now. We have rhs of linear system (8)[Greengard] and v \hat{v}.


Precompute the Toeplitz kernel and right-hand side.  
Define

$$
J_{2m}:=\{-2m,-2m+1,\dots,2m\}^d,
\qquad
M_{2}:=|J_{2m}|=(4m+1)^d.
$$

Compute the Toeplitz kernel vector

$$
v\in\mathbb{C}^{M_2},
\qquad
v_\ell=\sum_{n=1}^N e^{2\pi i h\langle \ell,x_n\rangle},
\qquad \ell\in J_{2m},
$$

and the right-hand side

$$
b\in\mathbb{C}^M,
\qquad
b_j=\sum_{n=1}^N \overline{\phi_j(x_n)}\,y_n,
\qquad j\in J_m,
$$

using type-1 NUFFT.  
Precompute

$$
\widehat v \in \mathbb{C}^{M_2},
$$

the $d$-dimensional FFT of the zero-padded Toeplitz kernel vector $v$.

**拆解任务**

- `efgp_solver.precompute` 中严格对齐 $J_{2m}$ 与 FFT 尺寸。
- `toeplitz.py` 的切片规则需要与 $v$ 的嵌入方式一致。
- 如需输出 `v` 或 `\widehat v`，新增缓存字段。

---

## 步骤 4

**原文**

Build a fast matrix-vector oracle.  
Define the diagonal matrix

$$
D\in\mathbb{R}^{M\times M},
\qquad
D_{jj}=\sqrt{h^d\widehat{k}(hj)},
$$

and the Fourier feature matrix

$$
F\in\mathbb{C}^{N\times M},
\qquad
F_{n,j}=e^{2\pi i h\langle j,x_n\rangle},
$$

so that $\Phi=FD$.  
For any vector

$$
u\in\mathbb{C}^M,
$$

compute

$$
Au = D(F^*F)(Du)+\lambda u,
$$

where

$$
F^*F\in\mathbb{C}^{M\times M}.
$$

This multiplication is carried out using two FFTs and diagonal multiplications, in $O(M\log M)$.

**拆解任务**

- 在 `efgp_solver._apply_A` 中验证矩阵自由乘法的等价性。
- 检查维度展开顺序，确保与 `mtot` 的网格一致。
- 对小规模问题做显式矩阵对比验证。

Check apply_toeplitz_fft_1d/estimate_top_eigenspace/richardson/pcg；
Check solve in efgp_solver
---

## 步骤 5

**原文**

Estimate the top eigenspace of $A$.  
Obtain approximate top eigenpairs

$$
(\theta_i,u_i),\qquad i=1,\dots,q,
$$

where

$$
\theta_i\in\mathbb{R},\qquad u_i\in\mathbb{C}^M,\qquad \|u_i\|_2=1.
$$

Stack them into

$$
U_q:=[u_1,\dots,u_q]\in\mathbb{C}^{M\times q},
\qquad
\Theta_q:=\mathrm{diag}(\theta_1,\dots,\theta_q)\in\mathbb{R}^{q\times q}.
$$

**拆解任务**

- 使用 `eigenspace.estimate_top_eigenspace` 的矩阵自由 `eigsh`。
- 明确 `q` 的选择与计算成本。
- 验证 `A` 的 Hermitian 性与数值稳定性。

Consider randomized SVD / operator Nyström later.

---

## 步骤 6

**原文**

Construct the EigenPro-style preconditioner.  
Set

$$
P_q
:=
I_M-U_q\Bigl(I_q-\mu\Theta_q^{-1}\Bigr)U_q^*
\in\mathbb{C}^{M\times M},
\qquad
\mu=\theta_{q+1}\ \text{or an approximation to it},
$$

where $I_q\in\mathbb{R}^{q\times q}$ is the identity matrix.  
Then, ideally,

$$
\mathrm{spec}(P_qA)=\{\mu,\dots,\mu,\theta_{q+1},\dots,\theta_M\},
$$

i.e. the first $q$ eigenvalues are flattened to $\mu$.

**拆解任务**

- 在 `eigenpro_precond.py` 中实现 $U_q^*$ 的共轭转置。
- 明确 $\mu$ 的估计策略并写成可切换选项。

---

## 步骤 7
preconditioning before solving the system.  
Two options are natural:

Preconditioned Richardson / gradient iteration:

$$
\beta_{t+1}=\beta_t-\eta P_q(A\beta_t-b),
\qquad
\beta_t\in\mathbb{C}^M.
$$

Preconditioned conjugate gradient: use $P_q\in\mathbb{C}^{M\times M}$ as a spectral preconditioner within PCG applied to

$$
A\beta=b,
\qquad
A\in\mathbb{C}^{M\times M},\ \beta,b\in\mathbb{C}^M.
$$

Stop when the relative residual satisfies conditions:

**拆解任务**

- `linear_solvers.richardson` 与 `linear_solvers.pcg` 的接口统一。
- 对齐论文的相对残差停止准则 $\rho$。
- 实验性对比 Richardson 与 PCG 的迭代次数与稳定性。
We do the CPU part and linear solving first. Later introduce SGD and minibatch with optimization problem later

-考虑自己重写CG整合scipy的PCG

---


## 步骤 8

**原文**

Evaluate the predictor.  
For target points $\{x_\ell^*\}_{\ell=1}^{N_{\mathrm{test}}}$, compute

$$
\widetilde{f}(x_\ell^*)=\sum_{j\in J_m}\beta_j\phi_j(x_\ell^*),
\qquad \ell=1,\dots,N_{\mathrm{test}},
$$

where

$$
\beta\in\mathbb{C}^{M}.
$$

This batched evaluation is performed using type-2 NUFFT.

**拆解任务**

- `efgp_solver.predict` 中校验坐标中心化与 $2\pi h$ 缩放。
- 如需要方差，扩展新的评估函数。

---

## 关键实现模块定位

- `discretization.py`：离散参数与网格
- `kernels.py`：$\widehat{k}$ 的实现
- `nufft_ops.py`：NUFFT 封装
- `toeplitz.py`：Toeplitz FFT 嵌入
- `eigenspace.py`：顶特征估计
- `eigenpro_precond.py`：谱预条件器
- `linear_solvers.py`：PCG / Richardson
- `efgp_solver.py`：主流程
