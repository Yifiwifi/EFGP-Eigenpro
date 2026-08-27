# Two-stage KRR and fixed-system experiments

The experiment code deliberately separates two scientific questions.

## Stage 1: complete KRR pipelines

`end_to_end.py` compares data-space `nystrom-krr`, data-space
`rpcholesky-krr`, complete standard-setup EFGP-CG/Jacobi/full-grid-EigenPro
pipelines, and the proposed binned-setup pipeline. Every row pays for its own
model setup and solve. The primary total is
`train_total_seconds = setup_seconds + solving_phase_seconds`; prediction is
reported separately. This is the method-owned algorithmic training total, not
process wall clock: common dataset I/O, backend creation, and host-to-device
staging are disclosed exclusions. For score-selected EFGP methods, the solving
phase includes score selection, preconditioner construction, and CG/PCG solve.
Every successful matched-repeat pair retains
its raw setup, solving, and training-total speedups, including methods that trade
accuracy for time. Dataset-specific absolute RMSE/R2 bounds define only a broad
usable-quality range. The 1% full-eig-relative label is descriptive and never
removes a timing or time-quality result.

`end_to_end_suite.json` declares the 10M, 30M, 100M, and 300M scale matrix on
Synthetic and Winnebago data. Exact RPCholesky keeps its published rank-by-N
factor requirement: if that factor cannot fit, the row remains visible as
`resource_limit` and is never replaced by a pilot-set approximation. The suite
uses a frozen CG-iteration window and declared dataset tie priority to select
the Stage-2 target. An RPCholesky `resource_limit` is a predeclared large-scale
scalability outcome: it receives no timing/accuracy speedup, but it does not
prevent selecting a fixed-Fourier solver target when all Nyström/EFGP rows and
the proposed/full-eig rows remain inside the declared broad usable-quality
range. Any other failed method still
blocks selection. Only after `selected_target_regime.json` is written does it
materialize the declared lambda, lengthscale, box-budget, and dataset checks,
using the exact per-N Winnebago artifact. The command fails closed if no target
satisfies that rule. It also refuses to select from a partial scale matrix:
every declared case must exist, and its raw repeat file is revalidated before
target selection or any downstream run begins.

The formal rerun does not rescan full-eig or proposed parameters on current
timings. It freezes the archived `paper_table1_selected.csv` winners: separate
full-eig rank `q`, proposed active top-k, and proposed eigenspace rank `q` for
each dataset/N case. Only the configuration is transferred; all setup and solve
times are measured again under the 1-warm-up + 5-measured-repeat protocol. The
Synthetic source used noise 0.3 while the current nested master uses noise
0.02, so those rows are explicitly labelled historical transfers rather than
current-data optima. Lambda, lengthscale, and dataset robustness freeze the
selected target configuration. The box-budget axis is separately labelled
budget-adaptive because a frozen enclosing box cannot fit every smaller cap.
If a lengthscale change produces fewer Fourier modes than the frozen top-k, the
runner deterministically clips top-k to all available modes and records
`frozen_score_topk_clamped_to_grid`; it never launches a replacement scan.

```bash
python -m efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.end_to_end_suite \
  --dataset-dir /content/efgp_data \
  --output-root /content/drive/MyDrive/EFGP_Colab/end_to_end_krr \
  --run-robustness-after-selection
```

Do not append a shared Fourier setup time to the fixed-system results below
and call the result an end-to-end Nystrom/RPCholesky KRR comparison. They are
different algorithms and different protocols.

## Stage 2: matched fixed-system experiments

This folder answers one narrow question: how much does a preconditioner reduce
the cost of solving one already constructed Fourier system? It does not compare
different Fourier setup routes.

## Controlled protocol

`benchmark.py` constructs one set of Fourier weights, one Toeplitz generator,
and one right-hand side. It separately hashes the stored precompute RHS and the
possibly precision-cast RHS actually passed to CG/PCG; the latter, together
with the weights, generator, and regularization value, defines `system_id`.
Every method receives that same immutable system, tolerance, iteration limit,
and zero initialization.

The timed method unit is

```
score selection (when used) + preconditioner construction + CG/PCG solve
```

This quantity is recorded canonically as `solver_total_seconds`; in outputs
from the corrected runner, the older `build_plus_solve_seconds` field is an
identical compatibility alias. The
common Fourier setup is reported once and excluded from the primary
fixed-system speedup. Every score-selected method (`default`,
`active-inverse`, and `active-eig`) reruns and pays for its own score-box
selection exactly once in every warm-up and measured invocation. The resolved
active set is passed directly into the preconditioner builder, so construction
does not repeat the score sort or enclosing-box calculation. Each repeated selection
must reproduce the prospectively frozen box hash, rule, and effective rank.
The runner performs one warm-up by default, shuffles the method order
independently in every round, synchronizes the whole CUDA device at all timing
boundaries, and uses at least five measured repeats. The runner does not compute
prediction or RMSE inside the timing harness; the separate
`prediction_audit.py` performs an explicitly untimed, chunked test-RMSE audit.
True-residual recomputation, hashing, and post diagnostics are also outside the
timed unit.

The formal selected-target campaign writes `stage2_feasibility.json` before
timing. CG, Jacobi, default, active-eig, and full-eig are mandatory.
`active-inverse` is executed only when the frozen Stage-1 active-box upper bound
is no larger than the prospectively declared `inverse_max_size`; otherwise it
is recorded as infeasible with a reason and is not allowed to abort the whole
fixed-system case. The artifact is bound to all 15 system-building fields and
the frozen method configuration, not just dataset, N, and kernel parameters.
The formal suite declares a Stage-2 explicit-inverse cap of 16,384 while
retaining a separate deployment/default threshold of 1,024. Thus Stage 2's
primary `default` remains the same frozen active-eig route used by the Stage-1
proposed pipeline; relaxing the explicit inverse feasibility cap cannot change
that primary algorithm. On the expected 30M target, the archived active box has
size 10,609, so the explicit inverse comparison is prospectively feasible.

The canonical reporter does not trust summary eligibility flags. It recomputes
Stage-1 accuracy and timing eligibility from each case's `pipeline_runs.csv`,
and Stage-2 convergence and totals from `matched_runs.csv`. For Stage 2 it also
requires equal initial/final/per-repeat system IDs, verifies the saved timing
system artifact and SHA-256, requires and rehashes its materialized
weights/Gf/storage-RHS/solve-RHS arrays, and matches all embedded, nested, and
external component hashes plus the canonical system-config hash to the frozen
target. Every warm-up and measured row must also match the common configured
tolerance and iteration limit and declare a zero initial vector. Formal plots
are generated only from this repeat-recomputed output; an audit failure aborts
before a headline artifact is written.

Portable prepared-system artifacts preserve both the original system-build
runtime and the current timing runtime. If a Colab resume loads an artifact on
a different GPU, solver-only timing remains eligible after the exact hashes and
current device identity pass, but the reused historical setup time is excluded
from setup-inclusive claims. A canonical timing-runtime hash covers the device,
compute capability, CuPy/CUDA versions, and resolved NUFFT backend so partial
resume cannot silently mix software runtimes inside a paired ablation.

The raw output records both the recursive residual and a fresh, complex128-audited
`norm(b - A beta) / norm(b)`, plus the relative coefficient difference from a
converged CG solution. It also records the actual storage, solve, and
preconditioner dtypes. `fp64` is the publication path. `mixed32` is explicitly
named mixed precision because the Fourier arrays remain in double precision;
the runner rejects a tolerance near machine epsilon unless the override flag is
given.

## Methods

- `cg`: no preconditioner.
- `jacobi`: inverse diagonal of the same Fourier matrix.
- `default`: score threshold `rho > tau`, followed only by a declared box-memory
  cap; use the box inverse if it fits the inverse cap, otherwise use the fixed
  spectral rank. This rule never sees timing, iterations, the right-hand side,
  test labels, or RMSE. The threshold-to-cap reduction sorts the scores once.
  If the cap removes threshold-selected modes, the result is called a
  score-ranked capped box, and the manifest reports the actual induced tail
  threshold `max(rho[T])`; the original `tau` corollary no longer applies.
- `active-inverse`: the inverse operation on the fixed strict box.
- `full-inverse`: the same exact inverse operation with the box fixed to the
  complete Fourier grid. All full-grid construction and factorization work is
  charged to build time. This small-`M` control isolates whether localization
  lowers cold build+solve time even when the full inverse reaches the solution
  in one PCG iteration.
- `active-eig`: the EigenPro preconditioner formula restricted to the fixed box.
- `full-eig`: the same spectral formula with the box equal to the full Fourier
  grid. This is the direct ablation for strict localization.
- `fourier-nystrom-precond`: an optional Gaussian randomized Nyström
  preconditioner for the unregularized
  part `A - lambda I`, following the fixed-system access model of
  [Frangella--Tropp--Udell](https://tropp.caltech.edu/papers/FTU23-Randomized-Nystrom-SIMAX.pdf).
  All sketch products and factorization are charged to
  construction. This implementation is a fixed-rank, complex-Hermitian
  Fourier-system adaptation of the real-symmetric construction. Timing repeats
  hold the sketch seed fixed; a publication stability check should repeat the
  experiment over at least five sketch seeds.
- `fourier-rpcholesky-precond`: an optional simple randomized pivoted Cholesky
  preconditioner (block size one) applied to
  the same Fourier-space positive-semidefinite part `A - lambda I`. It reads
  weighted-Toeplitz columns directly, stores a rank-`r` factor, and applies
  `(L L* + lambda I)^{-1}`. All pivoting, column gathers, factor updates, and
  the small inverse are charged to construction. This is a complex-Hermitian
  Fourier-system adaptation of RPCholesky, not the published data-space KRR
  solver. Timing repeats hold the pivot seed fixed; random-seed stability is a
  separate experiment.

`full-eig` is deliberately not called a circulant preconditioner. Although the
Fourier Gram matrix is Toeplitz, `A = D G D + lambda I` is generally not
Toeplitz when `D` is nonconstant. A circulant baseline needs a separate SPD
construction before it is a fair comparison.

The two `fourier-*-precond` methods are exploratory Fourier-space
preconditioners, not complete KRR algorithms and not formal defaults. The
published [RPCholesky KRR method](https://arxiv.org/abs/2304.12465) acts on the
data-space kernel matrix and belongs in the separate end-to-end KRR comparison.
Archived outputs may retain the old ambiguous labels `nystrom` and
`rpcholesky`; artifact readers can still audit those outputs, but new
fixed-system configurations reject those names.

The legacy base suite defaults to `cg,jacobi,default,full-eig`.  The corrected
selected-target Stage-2 campaign instead requires
`cg,jacobi,default,active-eig,full-eig`; it adds `active-inverse` only when the
prospective box-budget feasibility rule permits it.  The older
`fixed_system_inverse_control_n10m` profile remains an optional small-grid
control and is not the formal selected-target result.

The runner never selects the fastest measured configuration. Rows produced by
`default` are marked `deployable_default`; explicitly requested active-box rows
are marked `sensitivity_candidate`. If a separate candidate grid is used to
define an oracle, its best row is only a ceiling and the sum of all candidate
construction/solve times is the search cost. That oracle must not replace the
frozen default in a headline speedup.

A timing ratio is emitted as a usable performance claim only when both methods
reach the requested independently recomputed residual in the paired repeat.
Failed or max-iteration rows remain in the raw and summary files but cannot
produce a speedup.

## Frozen GeoLife task

The scalable real task uses the official Microsoft
[GeoLife GPS Trajectories 1.3](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
archive. Inputs are longitude and latitude in the fixed Beijing box
`[116.10, 116.70] x [39.70, 40.13]`; the target is GPS altitude. Altitude is
converted from feet to metres, and only finite values in `[-100, 1000] m` are
kept. Same-trajectory, same-timestamp repeats are removed. No speed-jump filter
is used.

Coordinates are projected to EPSG:32650 and shared-scaled from the fixed crop.
A trajectory hash assigns each complete PLT file to train or test, so adjacent
points from one trajectory cannot cross the split. An independent source-record
hash defines the nested order. Sampling is without replacement; unique means
unique source records, not unique coordinates. Standardization is fitted once
on all retained training-bucket targets and reused across sizes.

The audited source contains 17,969,317 retained records: 14,321,097 train and
3,648,220 test. This supports exact 100k/25k, 1m/250k, and 10m/2.5m artifacts
without replication. The main GeoLife cases use the squared-exponential kernel
with `lengthscale=0.02`, fixed before timing. Synthetic and USGS retain the base
Matérn setting.

The three GeoLife aliases reject a sidecar unless its archive hash, crop,
altitude cleaning, audit counts, trajectory split, nested without-replacement
order, projection, fixed target transform, and exact train/test sizes match the
frozen metadata. They fail before loading arrays if either the NPZ/JSON pair is
missing or any checked value differs.

GeoLife is licensed for non-commercial use only. The suite references local
files and does not package or distribute the archive, NPZ files, or sidecars.

Run the small plumbing configuration:

```powershell
python -m efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark `
  --dataset-stem GeoLife_Beijing_GPS_altitude_regression_ntrain100000 `
  --n-train 5000 --kernel se --lengthscale 0.02 `
  --lambda 0.1 --fourier-eps 1e-3 --tol 1e-7 --maxiter 6000 `
  --methods cg,jacobi,default,active-eig,full-eig `
  --score-tau 1 --box-budget 4096 --inverse-max-size 256 `
  --rank 32 --warmup-repeats 1 `
  --measured-repeats 5 --post-diagnostic-mode full `
  --diagnostic-topk 64,256 --nufft-backend none
```

`--nufft-backend none` makes the local missing-cuFINUFFT fallback explicit; the
manifest reports `cpu_finufft`. In Colab, use `--nufft-backend cufinufft` so a
missing GPU library fails instead of silently changing setup route, and add
`--strict-gpu-eig`.

Use `--n-train 0` for every row in the selected exact-size artifact; omitting
the flag deliberately uses the 20,000-row benchmark default.

## Three-dataset suite and scale

`suite.py` validates and runs the same matched protocol on GeoLife, synthetic,
and USGS data. The supplied `three_dataset_suite.json` has four profiles:

- `demo`: a fixed-seed, without-replacement 5,000-row subset of the 100,000-row
  GeoLife artifact and 5,000 rows from each existing control. This is only a
  plumbing check;
- `local_1m`: exact one-million-row GeoLife, synthetic, and USGS artifacts;
- `scale_10m`: exact ten-million-row GeoLife, synthetic, and USGS artifacts,
  using one-million-row NUFFT chunks;
- `scale_100m`: exact one-hundred-million-row synthetic and USGS artifacts for
  a high-memory A100/Colab run. GeoLife is absent because the source has fewer
  than one hundred million eligible records.

Validate the local files without loading the large arrays:

```powershell
python -m efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.suite `
  --profile local_1m --output-root controlled_local_check
```

Run the publication route on a mounted Google Drive:

```bash
python -m efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.suite \
  --profile scale_10m --dataset-dir /content/drive/MyDrive/processed \
  --output-root /content/drive/MyDrive/controlled_scale_10m \
  --nufft-backend cufinufft --strict-gpu-eig --execute
```

The exact 100k, 1m, and 10m GeoLife artifacts have been materialized locally.
The suite never repeats smaller data to satisfy a larger profile. The 10m
artifact has passed metadata and file validation but has not yet been used for
solver timing. The development-suite synthetic sidecars define `true_func_2d` as
`sin(2*pi*x1) + 0.5*cos(2*pi*x2) + 0.2*sin(2*pi*(x1+x2))`, with training noise
standard deviation 0.02, seeds 20260421 and 1, and a noise-free test set of
size \(N/4\). These low-noise `_nN` artifacts are not used in the paper
supplement below; that supplement restores the archived `_ntrainN` generation
protocol with noise standard deviation 0.3.

The current loader keeps all training `x,y` arrays resident in float64. NUFFT
chunking bounds setup temporaries but does not remove that resident \(O(Nd)\)
storage. The ten-million-row synthetic and USGS artifacts have now been run
with the matched solver protocol described below. The ten-million-row GeoLife
artifact remains locally validated but not solver-timed.
The `scale_100m` profile is explicit about requiring full resident arrays; a robust
out-of-core \(10^8+\) route still needs host-to-device streaming rather than a
full-array upload.

Older road-network output folders only verify that the harness and diagnostics
run end to end; they are not evidence for the frozen GeoLife task. A GeoLife
result is eligible only after the suite metadata checks pass and the matched
run completes.
Any local run using `cpu_finufft` remains a solver-only/hybrid demo rather than
a final all-GPU end-to-end paper number.

## Separate MUR negative control

The road-network task was removed from the scale suite because it contains only
347,899 training rows. MUR SST remains a separate negative control and is not a
dataset alias or case in the main three-dataset profiles. Its fixed North
Atlantic pilots did not pass the strict-box cold-time gate. Each ratio below is
the paired median `reference solver_total_seconds / default
solver_total_seconds`, so a value above one favors the strict default; common
Fourier setup is excluded.

| frozen run | CG/default | full-eig/default | decision |
|---|---:|---:|---|
| MUR stride-10 pilot, 100k, rank 64 | 0.531 | 0.862 | fail |
| MUR stride-10 pilot, 100k, rank 256 | 0.524 | 1.155 | fail |
| MUR stride-4 pilot, 1m, rank 256 | 0.824 | 1.068 | fail |

These runs use one shared system per case, fp64 arithmetic, a recomputed
relative-residual tolerance of `1e-7`, and five paired measured repeats. Their
strict score boxes are uncapped and capture more than 99.99% of the same-rank
full-grid dominant-subspace leverage.

The frozen GeoLife 1m run is also not a strict-box success: CG/default is 1.596,
but full-eig/default is 0.803. Score localization and acceleration over CG are
not sufficient to beat full-grid spectral correction after construction is
charged. The acceptance report requires at least 1.25x over both references,
with at least four wins in five pairs.

Do not relax this gate or select one of these ranks after seeing the timings.
Unless a future held-out task passes the same frozen rule, the paper should
describe strict boxes as a regime-dependent build/storage option and keep the
main contribution at fixed-system spectral preconditioning.

## Section 4.2 diagnostics

With `--post-diagnostic-mode full`, the runner measures the complement error
`epsilon_T` and active--tail coupling `eta_inv` or `eta_eig`. For inverse boxes
it also evaluates the proposition's plug-in lower/upper endpoints. A
nonpositive lower endpoint is written as `bound_status=not_informative`; such a
row is retained rather than filtered.

Hermitian norms are estimated from four complex-Gaussian starts by power
iteration on the squared operator, which avoids cancellation between positive
and negative spectral ends. The requested diagnostic tolerance is used as a
relative stabilization test. Iteration counts, cross-start ranges, and
stabilization flags are saved. These are heuristic estimates, not certified
norms: a positive plug-in lower endpoint is labelled
`estimated_informative_not_certified`, and an unstabilized estimate makes the
bound row ineligible.

`--diagnostic-topk 64,256,...` adds nested score-ranked inverse boxes outside
the timing table and records their PCG iterations and true residuals. If an
expanded box exceeds `inverse_max_size`, its row is retained as `skipped` with
the reason.

When `full-eig` is present, post diagnostics also save
`score_leverage_arrays.npz`. It contains the dominant-subspace leverage map,
the score map, and the score-selected box. The JSON row reports what fraction
of full-grid leading-eigenspace leverage is captured by that strict box. This
tests whether the score localizes difficult modes; cumulative score mass alone
does not.

## Parameter stability

Start from the full 100,000-row `example_spatial_strict_box_config.json` and run
a one-factor-at-a-time sweep:

```powershell
python -m efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.sweep `
  --config efgp_eigenpro_py/gpu/box_toeplitz_active_block/controlled/example_spatial_strict_box_config.json `
  --lambdas 0.01,0.1,1.0 --lengthscales 0.05,0.1,0.2
```

The sweep does not form a Cartesian product. Every parameter setting defines a
new Fourier system, but all methods within that setting still share exactly one
matrix and right-hand side. `sweep_index.csv` reports every method's speedup,
matrix dimension, and system hash; the default--oracle gap is then a direct
within-setting comparison only if a separate, declared candidate grid was run.
The supplied sweep does not choose or aggregate an oracle by itself.

## Original-task integrated supplement (2026-08-23)

Synthetic and USGS, rather than a newly selected favorable data set, are the
main line of the controlled supplement. The USGS artifact is reused directly.
The archived Synthetic NPZ was not retained, so it is deterministically
reconstructed from the saved notebook settings: noise standard deviation 0.3,
five-million-row generation chunks, train/test seeds 20260421 and 1, and
`N/4` test points. The archived output contains no NPZ hash, so byte identity
cannot be certified; the reconstructed CG test RMSE is `0.0082486384` versus
`0.0082486503` in the archived row. The archived six-size paper tables remain
an exploratory complete-pipeline regime map. The runs here are confirmatory
solver experiments: every method in one case shares one hashed Fourier matrix
and right-hand side, uses zero initialization and a common tolerance/iteration
cap, and is measured after one warm-up in five shuffled, paired repeats.

The CG-only screen was frozen before preconditioner timing at
`N=10,000,000`, Matérn-3/2, `ell=lambda=0.1`, with an operational target
interval of 3,000--6,000 iterations. The restored Synthetic task requires
4,246 iterations and USGS requires 4,845. Both pass, and USGS remains the
parameter-sensitivity case. Screen rows are never combined with a method row
from another invocation, because CPU FINUFFT construction need not be bitwise
identical across processes.

The configuration bridge combines the restored Synthetic runs under
`outputs/original_synthetic_n10m_matched_bridge_q128_matern` and
`outputs/original_synthetic_n10m_se_full_inverse_control` with the USGS cases
under `outputs/original_data_n10m_matched_bridge_q128` and
`outputs/original_data_n10m_se_full_inverse_control`. Matérn uses rank `q=128`
and a fixed `|B|<=2601` cap; the separate SE full-inverse control uses the
score-threshold box `|B|=841` on the `M=1225` grid. Entries below are legacy
median `build+solve` seconds followed by paired cold speedup over CG; all listed
preconditioned rows converged in all five repeats and won all five CG pairs.
The explicit active rows predate the corrected rule that charges each active
method for its own score-selection time, so this table is archival context
rather than a formal `solver_total_seconds` comparison and must be rerun for a
Stage-2 claim.

| data/kernel | CG iter / solve | active inverse | active eig | full eig | Fourier Nyström precond | Fourier RPCholesky precond |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic Matérn | 4246 / 8.657 | 2.514 / 3.464x | 2.465 / 3.522x | 3.144 / 2.754x | 4.515 / 1.931x | 4.480 / 1.957x |
| USGS Matérn | 4905 / 10.158 | 5.667 / 1.793x | 4.622 / 2.198x | 1.846 / 5.528x | 2.380 / 4.278x | 2.148 / 4.714x |
| Synthetic SE | 2054 / 3.279 | 0.159 / 20.677x | 0.705 / 4.643x | 0.586 / 5.600x | 0.585 / 5.560x | 0.747 / 4.391x |
| USGS SE | 1528 / 2.458 | 0.162 / 15.069x | 0.373 / 6.630x | 0.359 / 6.803x | 0.091 / 27.004x | 0.289 / 8.263x |

### SE full-inverse control on the two restored/original 10m tasks

The follow-up fixed-system control is stored under
`outputs/original_synthetic_n10m_se_full_inverse_control` for Synthetic and
`outputs/original_data_n10m_se_full_inverse_control` for USGS. Both cases use
the archived task protocol, the squared-exponential
kernel with `ell=lambda=0.1`, `M=1225`, fp64 arithmetic, one warm-up, and five
shuffled measured repeats. The table reports medians from each case's
`matched_summary.csv`; these archived cold times are preconditioner build plus
CG/PCG solve and exclude both the one shared Fourier setup and the explicit
active method's score-selection charge. Reruns use `solver_total_seconds` and
include that selection charge.

| data | method | box size | iterations | build (s) | solve (s) | cold build+solve (s) | cold speedup over CG |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic SE | CG | -- | 2054 | 0 | 3.2786978 | 3.2786978 | 1.000000x |
| Synthetic SE | active-inverse | 841 | 6 | 0.1441978 | 0.0156614 | 0.1585640 | 20.677441x |
| Synthetic SE | full-inverse | 1225 | 1 | 0.2850381 | 0.0038683 | 0.2887243 | 11.355808x |
| USGS SE | CG | -- | 1528 | 0 | 2.4580098 | 2.4580098 | 1.000000x |
| USGS SE | active-inverse | 841 | 12 | 0.1351900 | 0.0271348 | 0.1621026 | 15.068851x |
| USGS SE | full-inverse | 1225 | 1 | 0.3100209 | 0.0042155 | 0.3142364 | 7.853297x |

The full inverse needs only one iteration, compared with 6 and 12 for the
localized active inverse, but iteration count is not the cold-start objective.
Its larger full-grid build cost makes it slower overall. From the summary
medians, active-inverse takes 0.1585640 versus 0.2887243 seconds on Synthetic
and 0.1621026 versus 0.3142364 seconds on USGS, making it 1.821x and 1.939x
faster, respectively. Thus the strict inverse box is the faster cold-start
inverse control on both original SE systems despite requiring more iterations.

This bridge changes the interpretation of the archive without discarding it.
On Synthetic Matérn, active inverse uses fewer iterations (169 versus 733),
but active eig has the lower cold time. On USGS Matérn, the archive's
full-grid spectral preference survives the exact-system comparison. On the
small SE grid, active inverse is the fastest proposed operation, but the fixed
Fourier-Nyström-preconditioner seed is faster on USGS; this is why the archived
supplement reports strong exploratory adapters instead of only the proposed
candidates. Jacobi is also retained:
it is slower than CG on both USGS systems, essentially tied on Synthetic SE,
and 2.111x faster on Synthetic Matérn.

The frozen deployable Matérn configuration uses the score-ranked
memory-capped box `|B|=8099`, `q=256`. Its six-method center runs are
`outputs/integrated_synthetic_archived_ntrain10000000_matern_q256_control` and
`outputs/integrated_usgs_n10000000_matern_q256_primary`. Default cold
speedup is 3.881x on Synthetic and 3.238x on USGS, with 5/5 paired wins in
both. It beats the same-rank full-grid correction on Synthetic, but not on
USGS; the Fourier-RPCholesky preconditioner is the fastest archived USGS
cold-start row at 3.827x. The capped box
stores 64.6 MiB versus 282.3 MiB for full eig. It captures 99.9991% and
98.6404% of same-rank full-grid leverage on Synthetic and USGS, respectively.

The one-factor-at-a-time USGS scan is frozen in
`usgs_n10m_matern_oat_primary_q256_suite.json` and written under
`outputs/usgs_n10m_matern_oat_primary_q256`. It changes one of
`lambda={0.01,0.1,1}` or `ell={0.05,0.1,0.2}` at a time and keeps the same
default rule, rank, memory cap, tolerance, and repeat protocol. All five
systems converged, achieved 5/5 paired wins, and gave median cold speedups
from 2.310x to 3.684x. The corresponding CG/default iterations are
12966/2166, 4896/726, 1811/233, 5768/1447, and 3076/367 for
`(lambda,ell)=(0.01,0.1),(0.1,0.1),(1,0.1),(0.1,0.05),(0.1,0.2)`.

The current formal `prediction_audit.py` loads the exact persisted timing
system and one canonical measured beta per method. It never rebuilds `A,b`,
never performs a warm-up, and never solves again. The one-click campaign audits
all timed methods for the three `paper_10m` cases and `cg/default` for the
Synthetic development-master 100M and 300M cases, using at most the first 2.5
million test points in chunks of 100,000. A prediction artifact is eligible only
when `audit_pass=true`, `system_id` and the weights/Gf/rhs hashes exactly match
the timing manifest, timed-solution hashes verify, and the audit solve count is
zero. Prediction seconds are accuracy-only and are not used in any speed claim.
Older v1 audit directories that rebuilt a system do not satisfy this gate and
must be regenerated from timing artifacts containing canonical solutions.

All these local supplement rows use CPU FINUFFT for the shared setup and an
RTX 3050 Laptop GPU for the solves. They are matched solver-only evidence,
not replacements for the archived A100 end-to-end timings. The partial
`usgs_n10m_matern_oat_q128` and `usgs_n10m_matern_oat_q64` directories are
resource/calibration runs and are not paper OAT results. GeoLife and MUR are
kept out of the main line: the original USGS task already passes the frozen
difficulty screen, while their existing runs serve only as held-out or
negative controls.

## Output files

- `experiment_config.json`: requested protocol.
- `system_manifest.json`: system hash, precision, actual dtypes, setup route,
  GPU/CUDA versions, fixed box rule, source-file hashes, dataset member checksums,
  and scoped git status.
- `matched_runs.csv/json`: every warm-up and measured row.
- `matched_summary.csv/json`: median/min/max timing and paired speedups.
- `matched_comparisons.csv/json`: direct paired comparisons, including
  strict/default versus full-grid spectral when both are present.
- `post_diagnostics.csv/json`: measured theoretical quantities and diagnostic
  PCG runs.
- `score_leverage_arrays.npz`: score/leverage data when full-grid spectral
  diagnostics are requested.
- `run_complete.json`: written atomically only after all required result files;
  `--resume` also checks the effective config, source/data fingerprints, current
  timing-runtime hash, the exact JSON/CSV method set, and repeat identifiers
  before reusing a case. A Colab reconnect on a different GPU/runtime therefore
  reruns singleton paper/scale cases instead of indefinitely preserving a mixed
  profile.
- `timing_system.npz`: byte-exact portable weights, generator, storage/solve
  RHS, prediction center, and frozen system metadata used by the timed case.
- `timing_solutions.npz` and `timing_solutions_manifest.json`: one canonical
  measured beta per method with array and timing-row checksums. Prediction
  audits consume these arrays and do not solve again.
- `prediction_audit/prediction_audit.csv/json` (when run): exact timing-system
  and timed-solution provenance, test RMSE and ratio to CG, equivalence gate,
  chunk size, zero audit solves, and accuracy-only prediction seconds.
- `prediction_audit/prediction_audit_complete.json`: written last and
  atomically; anchors the prediction JSON and CSV checksums, exact method set,
  source hash, row count, and audit decision for safe resume.

The controlled-suite CLI uses distinct terminal exit codes:

- `0`: every selected case and method is performance-claim eligible.
- `2`: all selected cases wrote complete artifacts, but at least one method is
  ineligible or a post diagnostic failed. This is a scientific result; inspect
  `suite_status.json` and `controlled_ineligible_rows.csv`.
- `1`: configuration, data validation, or case execution failed. Case-level
  tracebacks remain in `suite_status.json` when the suite reached execution.

The Colab campaign accepts exit code `2`, records it, and continues independent
jobs. Exit code `1` is also isolated at the job level so later independent jobs
can finish, but the final campaign manifest is not verified.

`preconditioner_storage_bytes` sums arrays retained by the built operation. It
is not a CUDA allocator peak; temporary eigensolver/SVD workspaces require a
separate peak-memory profiler if that quantity is needed in a paper table.

The main cold speedup is paired by repeat:

```
CG solver total / candidate solver total,
where solver total = score selection + preconditioner build + CG/PCG solve.
```

The reuse speedup excludes construction. `break_even_rhs` reports how many
right-hand sides are required to amortize construction when reuse is possible.
The strict-versus-full comparison additionally reports the cold-to-reuse RHS
crossover when the strict box builds faster but has a slower repeated solve.
