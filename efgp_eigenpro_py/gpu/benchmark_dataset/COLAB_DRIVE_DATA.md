# Colab / Google Drive benchmark data bundle

## Prefix masters only for compatible development sweeps

Compatible development sweeps may use exact row prefixes of one largest NPZ and
upload that **master once**.  This rule must not be applied across different
sampling or noise definitions, and it does not apply to the forward formal
Stage-1 Synthetic route.

Formal Stage-1 Synthetic uses four exact per-N files:
`synthetic_true_func_2d_ntrain{10000000,30000000,100000000,300000000}`.  They
follow the frozen original-task generation definition (noise standard deviation
0.3, train/test seeds 20260421/1, five-million-row generation chunks, and an
`N/4` noise-free test set).  They are independent exact artifacts rather than
declared prefixes of the noise-0.02 development master.  The latter must never
be substituted when an exact formal artifact is missing.

The masters were written with `numpy.savez`, so every NPY member is
`ZIP_STORED`.  `colab_drive_pack.py` can memory-map an array at its byte offset
inside the NPZ and return `[0:n]` without extracting the archive or allocating
the other 300M rows.  The same rule is applied to the fixed test set: a 10M
training prefix uses the corresponding first 2.5M test rows.

The development/controlled masters already in the workspace are:

| Drive dataset id | Upload exactly once | Size | Ready prefixes |
|---|---|---:|---|
| `synthetic-development-v1` | `synthetic_true_func_2d_n300000000.npz` | 6.90 GB | 10M, 30M, 100M, 300M |
| `winnebago-controlled-prefix-v1` | `USGS_LPC_IL_Winnebago_2018_ground_elevation_regression_ntrain300000000.npz` | 4.50 GB | 10M, 30M, 100M, 300M |
| `manitowoc-v1` | currently `USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000.npz` | 150 MB | 10M; larger prefixes planned |

Sizes above are decimal bytes.  The 300M synthetic master uses noise standard
deviation 0.02 and remains a development-only scale route.  It is not a formal
Stage-1 input.  The exact noise-0.3 Synthetic status is:

| Formal Stage-1 artifact | Current status |
|---|---|
| `_ntrain10000000` NPZ/JSON | import from the existing Google Drive exact-data collection |
| `_ntrain30000000` NPZ/JSON | import from the existing Google Drive exact-data collection |
| `_ntrain100000000` NPZ/JSON | import from the existing Google Drive exact-data collection |
| `_ntrain300000000` NPZ/JSON | import from the existing Google Drive exact-data collection |

The formal claim is that these files belong to the same frozen generated-data
family, not that they are byte-for-byte copies of a particular historical NPZ.
The imported copies receive catalog hashes for this run. Archived experiment
artifacts contribute only frozen selected configurations to the new run, and
their old timings are excluded.

The 300M Winnebago master exposes raw row prefixes; it does not reproduce the
spatial-stratified indexes in the archived paper artifacts.  Neither of the two
300M development masters may therefore be presented as a reproduction of the
original paper experiment.  The synthetic archive contains two additional
300M training-prefix arrays (`y_train_true` and `train_noise`), which explains
its larger size.  The data files, their JSON metadata, a catalog, and SHA-256
checksums are the complete upload set; raw LAZ/EPT cache files are not needed
in Colab.

## Create a zero-copy staging directory on Windows

Run from `D:\NU\ML`.  Hard links consume no additional data blocks, while
presenting one clean directory that can be selected for Google Drive upload.
The `synthetic-development-v1` command below stages the optional low-noise
development master only; its presence does not satisfy formal Stage-1 input
validation.

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

The four files already stored in Google Drive should be copied into the formal
bundle and registered directly. The following generator is only a recovery
path if an exact size is genuinely missing; it is not part of the default
all-experiment run:

```powershell
python -m efgp_eigenpro_py.gpu.benchmark_dataset.preprocess_synthetic_true_func_2d_size_sweep `
  --n-train-list 30000000,100000000,300000000 `
  --noise 0.3 --seed-train 20260421 --seed-test 1 `
  --chunk-rows 5000000 --size-token ntrain `
  --output-dir efgp_eigenpro_py\gpu\benchmark_dataset\processed
```

Register each exact paper-definition input instead of assigning it
master-prefix semantics.  Repeat the Synthetic NPZ and metadata operations for
all four formal sizes; each file must appear in the bundle selected by Stage 1,
in `drive_manifest.json`, and in `checksums.sha256`.  For example, these are the
distinct 10M synthetic-noise-0.3 and spatial-stratified Winnebago inputs:

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
by a formal or archived experiment configuration.  A compressed exact NPZ is
valid in this route; it is deliberately not exposed through the zero-copy
prefix API. Registering all four existing noise-0.3 Synthetic pairs in
`archived_exact_available` is preferred. For compatibility with the original
experiment, the formal notebook can also locate an unregistered pair in the
standard MyDrive cache directories by exact basename, copy it locally, compute
a fresh SHA-256, and bind that hash to the run. Add 1M and 3M only when the
optional complete legacy groups are also requested.

Winnebago exact pairs are registered at 1M, 3M, 10M, 30M, and 100M plus the
historical 300M file.  Their sidecars document that sampling differs across
some N, so these are independent exact systems rather than one nested sweep.

After all registrations, verify every data artifact:

```powershell
python -m efgp_eigenpro_py.gpu.benchmark_dataset.colab_drive_pack verify `
  --manifest $stage\drive_manifest.json
```

After the catalog contains the four existing files, launch the formal campaign.
Do not resume or append to a run directory whose Stage-1 Synthetic rows used
the noise-0.02 development master.

The previously prepared development/legacy upload is 13,847,768,160 bytes
(12.90 GiB), but that size alone is not evidence that the forward formal bundle
contains all four exact Synthetic pairs. After importing and registering them,
recompute the catalog size and checksums. `checksums.sha256` must cover every NPZ/JSON artifact listed in the
catalog; the notebook separately records the SHA-256 of `drive_manifest.json`
in each run manifest.

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
      synthetic_true_func_2d_ntrain{10000000,30000000,100000000,300000000}.npz
      synthetic_true_func_2d_ntrain{10000000,30000000,100000000,300000000}.json
    winnebago-paper-spatial/
      USGS_LPC_..._ntrain{1000000,3000000,10000000,30000000,100000000}.{npz,json}
```

Google Drive does not retain hard-link identity, but there is only one linked
entry for each master in this staging tree, so each master is uploaded once.
Each formal Synthetic `_ntrainN` pair is an exact artifact and is uploaded
separately; it is not deduplicated into the development master.

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
