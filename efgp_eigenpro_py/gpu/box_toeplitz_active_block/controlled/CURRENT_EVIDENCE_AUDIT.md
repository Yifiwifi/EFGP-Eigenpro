# Current evidence audit for the two-stage campaign

This note records what the saved artifacts can and cannot establish before the
new Stage 1 and corrected-total Stage 2 campaigns are rerun. It is deliberately
more conservative than a paper claim.

## Stage 1: complete KRR pipelines

No saved artifact currently contains one common-protocol comparison of all of
the following complete KRR pipelines:

- data-space Nystrom KRR;
- data-space RPCholesky KRR;
- standard-setup EFGP-CG;
- standard-setup EFGP+Jacobi;
- standard-setup EFGP+full-grid EigenPro;
- the proposed binned-setup EFGP pipeline.

The archived complete-pipeline bundle at
`outputs/btab_group_a_group_b_group_c_20260703_053504_export_bundle/`
contains an EFGP-only regime map. It has one measured run, no warm-up, no
shared-system hashes, different exact/binned setup routes, and post-hoc row
selection. It is useful for prospective target selection but is not the
missing cross-model comparison.

The strongest development signal is Winnebago--Matern. Relative to archived
EFGP-CG, the selected binned pipeline shows both setup and solve-scope gains:

| N | setup speedup | solve-scope speedup | training-total speedup | test-RMSE ratio |
|---:|---:|---:|---:|---:|
| 10M | 1.77x | 3.35x | 3.27x | 0.999278 |
| 30M | 2.11x | 3.80x | 3.68x | 0.999505 |
| 100M | 1.99x | 13.23x | 9.65x | 0.999653 |
| 300M | 2.21x | 18.02x | 9.56x | 0.999750 |

These values are a target-regime signal only. In particular, archived
Synthetic--SE test-RMSE ratios deteriorate to 1.0365, 1.0885, 1.2028, and
1.6561 at the four target sizes. The saved results therefore refute an
unqualified claim that accuracy is independent of dataset or kernel.

The replacement Stage 1 protocol is implemented in `end_to_end.py` and
`end_to_end_suite.py`. It reports each pipeline's own setup, solving phase,
method-owned algorithmic training total, prediction accuracy, resource-limit
status, and prospective robustness grid. Common dataset I/O, backend creation,
and host-to-device staging are deliberately outside that total and are stated
as exclusions rather than described as end-to-end wall time.

At rank 256 in FP64, the RPCholesky factor alone requires about 57.2, 190.7,
and 572.2 GiB at 30M, 100M, and 300M samples. A prospectively declared
`resource_limit` row is therefore retained as a genuine scalability outcome;
it is never replaced by a smaller pilot and never receives a speedup. The
target-selection rule may ignore only this declared RPCholesky resource limit,
while still requiring all six method rows and successful proposed/full-eig rows
inside the declared broad absolute usable-quality range. The 1% reference
equivalence label is descriptive and does not discard time-quality results.

## Stage 2: one identical Fourier A,b

The canonical saved bundle is `colab_result/paper_one_click_49076d35b4e8`.
Its manifests verify one fixed Fourier system per case, and its archived
`build_plus_solve_seconds` already includes score selection inside
`build_seconds`. Those values remain archival because they lack the corrected
canonical naming and authoritative raw-repeat revalidation. New runs expose
the timing definition directly as

`solver_total_seconds = selection + preconditioner construction + solve`.

The old ambiguous `nystrom` and `rpcholesky` rows are Fourier-system
preconditioners, not data-space KRR. New configurations reject those names;
the explicit exploratory names are `fourier-nystrom-precond` and
`fourier-rpcholesky-precond`.

The latest saved scale rows do not support the claim that the proposed default
beats every formal solver baseline in total time. The ratio below is archived default
build-plus-solve divided by full-grid EigenPro build-plus-solve; values above
one mean the default is slower.

| dataset | 10M | 30M | 100M | 300M |
|---|---:|---:|---:|---:|
| Synthetic | 1.09 | 1.13 | 1.03 | 1.27 |
| Winnebago | 2.31 | 2.94 | 3.86 | 4.19 |

The saved data support acceleration over CG/Jacobi in selected cases and a
memory advantage over the full-grid rank-256 correction, but not universal
total-time superiority. The corrected formal Stage 2 campaign must include CG,
Jacobi, default, active-eig, and full-eig. It includes active-inverse only when
the prospective active-box upper bound is no larger than `inverse_max_size`.
The rerun suite uses a separate Stage-2 inverse cap of 16,384, making the
expected 30M target box of size 10,609 feasible for the explicit inverse row.
The primary `default` retains its independent 1,024 threshold and therefore
remains the frozen Stage-1 active-eig method.
Full-inverse is not part of that formal matrix;
`fixed_system_inverse_control_n10m` supplies it only as a separate legacy
small-grid control.

## Claim rule

The reporting module `two_stage_reporting.py` fails closed:

- Stage 1 retains raw paired setup, solving, and training-total speedups for
  every successful matched-repeat pair. Broad absolute RMSE/R2 bounds mark
  usability, while full-eig-relative equivalence is descriptive only;
- partial or failed scale campaigns stop before target selection; the selected
  Stage 1 target is recomputed from complete raw-repeat-verified scale evidence,
  and Stage 2 must match all 15 system-building configuration fields (including
  subset, variance, precision, and NUFFT settings), not merely its dataset, N,
  kernel, nu, lambda, lengthscale, and Fourier tolerance;
- robustness requires the complete predeclared one-at-a-time grid around that
  exact selected target, with all non-varied controls held fixed;
- Stage 2 headline claims require a verified fixed system and canonical
  per-repeat `solver_total_seconds = selection + construction + solve`,
  recomputed from `matched_runs.csv`, rather than the old alias or a sum of
  independently summarized medians; speedup is then the median of matched
  repeat ratios `CG_i / method_i`, never the ratio of two method medians;
- Stage 1 accuracy/timing eligibility is likewise recomputed from each case's
  `pipeline_runs.csv`; summary booleans are checked, not trusted;
- initial, final, per-repeat, and embedded-artifact system IDs must agree; the
  timing NPZ must contain materialized weights, Gf, storage RHS, and solve RHS;
  their recomputed component/system hashes must agree across the embedded,
  nested, and external manifests; every timing row must use the common
  tolerance, iteration limit, and zero initial vector;
- the formal Stage 2 matrix requires CG, Jacobi, default, active-eig, and
  full-eig; active-inverse is also required whenever the prospectively
  declared active-box upper bound is within the declared inverse-size cap, and is
  otherwise recorded as explicitly infeasible before timing; missing feasible
  methods or unknown rows block the headline;
- shared Fourier setup is reported separately;
- unsupported and missing claims remain `not_supported` or `not_evaluable`.
