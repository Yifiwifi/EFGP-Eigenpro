# Colab / Google Drive benchmark data bundle

## One master per compatible scale sweep, not four copies

The controlled 10M, 30M, 100M, and 300M scale sweeps use exact row prefixes of
one largest NPZ per compatible data definition.  Upload that **master once**.
This rule must not be applied across different sampling or noise definitions.

The masters were written with `numpy.savez`, so every NPY member is
`ZIP_STORED`.  `colab_drive_pack.py` can memory-map an array at its byte offset
inside the NPZ and return `[0:n]` without extracting the archive or allocating
the other 300M rows.  The same rule is applied to the fixed test set: a 10M
training prefix uses the corresponding first 2.5M test rows.

The two complete 300M development/controlled masters already in the workspace
are:

| Drive dataset id | Upload exactly once | Size | Ready prefixes |
|---|---|---:|---|
| `synthetic-development-v1` | `synthetic_true_func_2d_n300000000.npz` | 6.90 GB | 10M, 30M, 100M, 300M |
| `winnebago-controlled-prefix-v1` | `USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain300000000.npz` | 4.50 GB | 10M, 30M, 100M, 300M |
| `manitowoc-v1` | currently `USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000.npz` | 150 MB | 10M; larger prefixes planned |

Sizes above are decimal bytes.  The 300M synthetic master uses noise standard
deviation 0.02 and is a development scale route; the archived paper synthetic
system uses noise standard deviation 0.3 and remains a separate exact NPZ.
The 300M Winnebago master exposes raw row prefixes; it does not reproduce the
spatial-stratified indexes in the archived paper artifacts.  Neither 300M
master may therefore be presented as a reproduction of the original paper
experiment.  The synthetic archive contains two additional
300M training-prefix arrays (`y_train_true` and `train_noise`), which explains
its larger size.  The data files, their JSON metadata, a catalog, and SHA-256
checksums are the complete upload set; raw LAZ/EPT cache files are not needed
in Colab.

## Create a zero-copy staging directory on Windows

Run from `D:\NU\ML`.  Hard links consume no additional data blocks, while
presenting one clean directory that can be selected for Google Drive upload.

```powershell
$stage = "D:\NU\ML\colab_drive_upload_ready"

python -m efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack prepare `
  --source-npz efgp_eigenpro_py\gpu\benchmark_dataset\processed\synthetic_true_func_2d_n300000000.npz `
  --source-metadata efgp_eigenpro_py\gpu\benchmark_dataset\processed\synthetic_true_func_2d_n300000000.json `
  --dataset-id synthetic-development-v1 --output-dir $stage --link-mode hardlink `
  --bundle controlled_master_prefix --bundle development_scale_masters

python -m efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack prepare `
  --source-npz efgp_eigenpro_py\gpu\benchmark_dataset\processed\USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain300000000.npz `
  --source-metadata efgp_eigenpro_py\gpu\benchmark_dataset\processed\USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain300000000.json `
  --dataset-id winnebago-controlled-prefix-v1 --output-dir $stage --link-mode hardlink `
  --bundle controlled_master_prefix --bundle development_scale_masters `
  --bundle archived_exact_available --bundle legacy_named_route_inputs

python -m efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack prepare `
  --source-npz efgp_eigenpro_py\gpu\benchmark_dataset\processed\USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000.npz `
  --source-metadata efgp_eigenpro_py\gpu\benchmark_dataset\processed\USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000.json `
  --dataset-id manitowoc-v1 --output-dir $stage --link-mode hardlink `
  --bundle controlled_10m --bundle independent_replication_10m --bundle manitowoc_10m

```

Register each archived paper input as an exact artifact instead of assigning
it master-prefix semantics.  For example, these are the distinct 10M
synthetic-noise-0.3 and spatial-stratified Winnebago inputs:

```powershell
python -m efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack add-artifact `
  --source efgp_eigenpro_py\gpu\benchmark_dataset\processed\synthetic_true_func_2d_ntrain10000000.npz `
  --name synthetic-paper-noise03-10m:npz --dataset-family synthetic-paper-noise03 `
  --role exact_npz --output-dir $stage --link-mode hardlink `
  --bundle controlled_10m --bundle original_data_10m `
  --bundle archived_exact_available --bundle legacy_named_route_inputs

python -m efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack add-artifact `
  --source efgp_eigenpro_py\gpu\benchmark_dataset\processed\synthetic_true_func_2d_ntrain10000000.json `
  --name synthetic-paper-noise03-10m:metadata --dataset-family synthetic-paper-noise03 `
  --role metadata_json --output-dir $stage --link-mode hardlink `
  --bundle controlled_10m --bundle original_data_10m `
  --bundle archived_exact_available --bundle legacy_named_route_inputs

python -m efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack add-artifact `
  --source efgp_eigenpro_py\gpu\benchmark_dataset\processed\USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain10000000.npz `
  --name winnebago-paper-spatial-10m:npz --dataset-family winnebago-paper-spatial `
  --role exact_npz --output-dir $stage --link-mode hardlink `
  --bundle controlled_10m --bundle original_data_10m `
  --bundle archived_exact_available --bundle legacy_named_route_inputs

python -m efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack add-artifact `
  --source efgp_eigenpro_py\gpu\benchmark_dataset\processed\USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain10000000.json `
  --name winnebago-paper-spatial-10m:metadata --dataset-family winnebago-paper-spatial `
  --role metadata_json --output-dir $stage --link-mode hardlink `
  --bundle controlled_10m --bundle original_data_10m `
  --bundle archived_exact_available --bundle legacy_named_route_inputs
```

Apply the same `add-artifact` operation to every other exact NPZ/JSON referenced
by an archived experiment configuration.  A compressed exact NPZ is valid in
this route; it is deliberately not exposed through the zero-copy prefix API.
The prepared workspace catalog registers archived Synthetic only at 10M, and
Winnebago exact pairs at 1M, 3M, 10M, 30M, and 100M plus the historical 300M
file.  The Winnebago sidecars document that sampling differs across some N, so
these are independent exact systems rather than one nested sweep.  Archived
Synthetic noise-0.3 files are still missing at 1M/3M/30M/100M/300M; the Colab
notebook refuses to run the corresponding legacy group until the required
`_ntrainN` artifacts have been regenerated with the frozen protocol.

After all registrations, verify every data artifact:

```powershell
python -m efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack verify `
  --manifest $stage\drive_manifest.json
```

The ready upload is 13,847,768,160 bytes (12.90 GiB), below a fresh 15GB Drive
quota.  `checksums.sha256` covers every NPZ/JSON artifact listed in the catalog;
the notebook separately records the SHA-256 of `drive_manifest.json` in each
run manifest.

`prepare` computes a streaming SHA-256, so the first pass over each multi-GB
file takes some time but does not load it into RAM.  It is safe to rerun: an
existing hard link to the same source is reused.  If the staging directory is
on another volume, use `--link-mode manifest-only`; the generated manifest then
lists the exact source and Drive destination for manual upload.

The resulting layout is:

```text
colab_drive_upload_ready/
  drive_manifest.json
  checksums.sha256
  data/
    synthetic-development-v1/
      dataset_manifest.json
      synthetic_true_func_2d_n300000000.npz
      synthetic_true_func_2d_n300000000.json
    winnebago-controlled-prefix-v1/
      dataset_manifest.json
      USGS_LPC_..._ntrain300000000.npz
      USGS_LPC_..._ntrain300000000.json
    manitowoc-v1/
      dataset_manifest.json
      USGS_EPT_..._n10000000.npz
      USGS_EPT_..._n10000000.json
    synthetic-paper-noise03/
      synthetic_true_func_2d_ntrain10000000.npz
      synthetic_true_func_2d_ntrain10000000.json
    winnebago-paper-spatial/
      USGS_LPC_..._ntrain{1000000,3000000,10000000,30000000,100000000}.{npz,json}
```

Google Drive does not retain hard-link identity, but there is only one linked
entry for each master in this staging tree, so each master is uploaded once.

### Manifest schema used by the Colab notebook

`drive_manifest.json` has three top-level indexes:

- `artifacts` is a list of uploadable files.  Every item has `name`, absolute
  local `source_path`, Drive `relative_path`, `size_bytes`, `sha256`, `role`,
  and `dataset_family`.  A `master_npz` item additionally has `storage` and the
  complete `array_schema`, including mmap byte offsets.
- `bundles` maps a semantic bundle name to artifact names.  The commands above
  create `controlled_10m`, `original_data_10m`,
  `independent_replication_10m`, `archived_exact_available`,
  `legacy_named_route_inputs`, `controlled_master_prefix`,
  `development_scale_masters`, and `manitowoc_10m`;
  they also create one bundle named after each dataset id.
- `datasets` maps each dataset id to its master, metadata, and
  `logical_prefixes`.  Every logical prefix records train/test slices and is
  explicitly marked `ready` or `planned`.

Notebook code should select artifact names through `bundles`, resolve them via
`artifacts`, and read prefix semantics from `datasets`.  This avoids embedding
machine-specific filenames in experiment cells.

## Verify and open a prefix in Colab

Mount Drive, verify it, and preferably copy the one master needed for the
current experiment to Colab's local SSD.  This is one temporary runtime copy;
it does not create four persistent Drive copies and avoids slow random reads
through the Drive FUSE mount.

```python
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack import (
    open_dataset_prefix,
    verify_catalog,
)

drive_root = Path("/content/drive/MyDrive/efgp_colab_data")
verify_catalog(drive_root / "drive_manifest.json")

# Optional but recommended before a solver run:
# !cp "$drive_root/data/winnebago-controlled-prefix-v1/USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain300000000.npz" /content/

master = drive_root / "data/winnebago-controlled-prefix-v1/USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain300000000.npz"
data = open_dataset_prefix(master, 30_000_000)
x_train = data["x_train"]  # read-only memmap view, shape (30_000_000, 2)
y_train = data["y_train"]  # read-only memmap view, shape (30_000_000,)
x_test = data["x_test"]    # corresponding fixed prefix, shape (7_500_000, 2)
y_test = data["y_test"]
```

`open_dataset_prefix` does not by itself make a 300M solver resident-memory
safe.  The experiment notebook must consume these views in chunks and avoid
unconditional `np.asarray(..., dtype=float64)` on the full dataset.

## Generate the future 300M Manitowoc master

Keep the frozen AOI, hash split, and ordering.  Start at EPT depth 8; if the
preprocessor reports fewer than exactly 300M train / 75M test eligible rows,
rerun with depth 9.  Downloads in the explicit cache directory resume.

```powershell
python -m efgp_eigenpro_py.gpu.benchmark_dataset.preprocess_usgs_ept_wi_2county `
  --ept-json https://s3-us-west-2.amazonaws.com/usgs-lidar-public/WI_2County_1_B23/ept.json `
  --n-train-list 300000000 --allow-large-output `
  --aoi-center-x 437100 --aoi-center-y 4884400 --aoi-side-m 52000 `
  --aoi-crs EPSG:6345 --max-lod-depth 8 `
  --source-project WI_2County_1_B23 `
  --official-mean-ground-density 11.37 `
  --official-work-unit-area-km2 1660.1823787253759 `
  --cache-dir efgp_eigenpro_py\gpu\benchmark_dataset\raw\WI_2County_1_B23_ept_cache `
  --temporary-dir efgp_eigenpro_py\gpu\benchmark_dataset\processed\_manitowoc_300m_work `
  --dataset-stem-prefix USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation
```

Before changing the Drive catalog, prove that the frozen 10M data are exactly
the first 10M/2.5M rows of the new master:

```powershell
python -m efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack compare-prefix `
  --larger-npz efgp_eigenpro_py\gpu\benchmark_dataset\processed\USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n300000000.npz `
  --prefix-npz efgp_eigenpro_py\gpu\benchmark_dataset\processed\USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000.npz
```

Then rerun `prepare` with the 300M NPZ and JSON under the same
`manitowoc-v1` id.  The catalog will mark all four prefixes ready.  The old
10M staging entry is deliberately not deleted automatically; after verifying
and uploading the 300M replacement, it can be removed from Drive because it is
already contained in the 300M master.
