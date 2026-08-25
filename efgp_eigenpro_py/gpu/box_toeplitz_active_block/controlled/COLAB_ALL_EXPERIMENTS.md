# Colab all-experiments notebook

The single entry point is
`../colab_all_experiments_10m_300m.ipynb`.  It combines the original named
experiment groups with the new fixed-system, scale, OAT, and prediction-audit
routes while preserving their different timing semantics.

## Before opening Colab

1. Push the current code to a commit and set `REPO_REF` in the first notebook
   cell to that commit SHA.  A local unpushed notebook cannot make the new
   runners available to a fresh Colab clone.
2. Upload the **contents** of `D:\NU\ML\colab_drive_upload_ready` to
   `MyDrive/EFGP_Colab/data_bundle`.  Preserve the directory structure and the
   two top-level files `drive_manifest.json` and `checksums.sha256`.
3. Select an A100 High-RAM runtime for 300M.  The notebook intentionally blocks
   100M/300M until `ALLOW_100M`/`ALLOW_300M` are explicitly enabled.
4. On the first upload, set `VERIFY_FULL_SHA256=True`; subsequent runs can use
   the byte-size checks unless the Drive data changed.

All heavy switches default to false.  A practical order is:

1. `RUN_PLUMBING_SMOKE=True`.
2. `RUN_CG_SCREEN_10M=True`.
3. `RUN_Q256_CENTER_10M=True`, then `RUN_WINNEBAGO_OAT_10M=True`.
4. Set the requested `ACTIVE_SIZES` and enable exactly one scale family.
5. Run `RUN_PREDICTION_AUDIT=True` after the corresponding controlled cases.
6. Run legacy groups separately, for example
   `RUN_LEGACY_GROUPS=["group_a"]`; a group is skipped only after its final
   `_SUCCESS.json` has been written.

## Scale routes and data meaning

| switch | data definition | status |
|---|---|---|
| `RUN_ARCHIVED_EXACT_SCALE` | archived/exact Synthetic and Winnebago artifacts; each N is an independent file | Winnebago ready through 300M; Synthetic noise-0.3 ready only at 10M |
| `RUN_DEVELOPMENT_MASTER_SCALE` | noise-0.02 Synthetic and raw-prefix Winnebago, exact nested prefixes of one 300M master | ready at 10M/30M/100M/300M |
| `RUN_MANITOWOC_SCALE` | independent 2023 USGS 3DEP acquisition, nested EPT master | 10M ready; 300M master must still be generated and prefix-verified |

Use `GENERATE_ARCHIVED_SYNTHETIC_SIZES=[30_000_000,100_000_000,300_000_000]`
for the controlled 10M--300M archived-data scale.  Add 1M and 3M to that list
before running the complete legacy groups, because only the archived 10M
Synthetic file currently exists.  The notebook constructs every missing file
with the frozen noise/seeds/chunk protocol.  `GENERATE_MANITOWOC_300M=True` starts the frozen
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

Regenerate the notebook after changing its source template with:

```powershell
python efgp_eigenpro_py\gpu\box_toeplitz_active_block\controlled\build_colab_all_experiments_notebook.py
```
