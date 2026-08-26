# Colab all-experiments notebook

The single entry point is
`../colab_all_experiments_10m_300m.ipynb`.  It combines the original named
experiment groups with the new fixed-system, scale, OAT, and prediction-audit
routes while preserving their different timing semantics.

## Before opening Colab

1. Push the current branch.  The notebook defaults to
   `REPO_REF="codex/colab-all-experiments"`, checks it out detached, records the
   resolved commit SHA, and automatically writes to
   `paper_one_click_<sha-prefix>`.  A local unpushed notebook cannot make the
   new runner available to Colab.
2. Upload the **contents** of `D:\NU\ML\colab_drive_upload_ready` to
   `MyDrive/EFGP_Colab/data_bundle`.  Preserve the directory structure and the
   two top-level files `drive_manifest.json` and `checksums.sha256`.
3. Select an A100 High-RAM runtime and choose **Runtime -> Run all**.  The
   one-click formal campaign enables the 100M/300M gates automatically.  If the
   runtime has less than 30 GiB GPU memory or 20 GiB currently available host
   RAM, only the 300M jobs are marked `SKIPPED_HARDWARE`; smaller jobs continue.
4. Full SHA verification is intentionally outside the routine timed campaign.
   Run it once after changing the uploaded catalog; routine runs keep
   `VERIFY_FULL_SHA256=False` and still validate registered byte sizes and
   per-case data fingerprints.

`RUN_ALL_FORMAL_EXPERIMENTS=True` creates this fixed job order automatically:

1. cuFINUFFT/GPU smoke.
2. `paper_10m`: Synthetic, Winnebago, and Manitowoc, six methods.
3. Winnebago 10M lambda/lengthscale OAT.
4. Winnebago 10M box-budget ablation at 4096/8192/16384.
5. Synthetic development-master prefixes at 10M/30M/100M/300M.
6. Winnebago archived-exact artifacts at 10M/30M/100M/300M.
7. Prediction audit for the three `paper_10m` cases only.
8. Artifact audit, eligible-only plots, campaign ledger, and final manifest.

The development scale is filtered to `dataset_family="Synthetic"`; the known
failing Winnebago raw-prefix route is not launched.  The archived scale is
filtered to `dataset_family="Winnebago"`; unavailable archived Synthetic high-N
files are not requested.  Scale 10M and 30M run independently.  A 100M job
requires its 30M core methods (`cg/default/full-eig`) to pass, and 300M requires
the 100M core gate plus the hardware gate.

The one-click formal campaign intentionally excludes redundant CG screening,
legacy exploratory reruns, q128/SE optional controls, Winnebago raw-prefix
scale, and Manitowoc high-N generation.  They remain available in advanced
manual mode but are not required for the final controlled paper package.

## Scale routes and data meaning

| switch | data definition | status |
|---|---|---|
| `RUN_ARCHIVED_EXACT_SCALE` | archived/exact artifacts; one-click selects Winnebago only and does not claim cross-N nesting | Winnebago ready through 300M |
| `RUN_DEVELOPMENT_MASTER_SCALE` | exact prefixes of one 300M master; one-click selects noise-0.02 Synthetic only | ready at 10M/30M/100M/300M |
| `RUN_MANITOWOC_SCALE` | independent 2023 USGS 3DEP acquisition, nested EPT master | 10M ready; 300M master must still be generated and prefix-verified |

Advanced/manual mode can use
`GENERATE_ARCHIVED_SYNTHETIC_SIZES=[30_000_000,100_000_000,300_000_000]`
to create the missing archived Synthetic files.  Add 1M and 3M before running
the complete legacy groups.  `GENERATE_MANITOWOC_300M=True` starts the frozen
EPT build at LOD 8; if its exact eligible-row report is insufficient, raise the
depth to 9.  Capacity estimates are not accepted as a substitute for the exact
scan.

The three output protocol families must never share a speedup column:

- `archived_complete_pipeline`: original direct-CG versus binned-candidate
  complete pipeline; setup routes and linear systems are not matched.
- `controlled_fixed_system`: one hashed system per case, 1 warm-up and 5 paired
  repeats; primary timing is selection/build plus solve.
- `prediction_audit`: rebuilt system used only for RMSE equivalence; its timing
  is excluded from performance claims.

Each scale size is an independent suite job under
`controlled_fixed_system/<profile>/_jobs/<job_id>/`.  The suite returns `0` for
all-pass, `2` for complete artifacts with scientific failures, and `1` for
execution/config/data errors.  The notebook records all three in
`campaign_jobs.csv/json` and continues independent jobs; it never converts a
complete negative result into a top-level `CalledProcessError`.

Regenerate the notebook after changing its source template with:

```powershell
python efgp_eigenpro_py\gpu\box_toeplitz_active_block\controlled\build_colab_all_experiments_notebook.py
```
