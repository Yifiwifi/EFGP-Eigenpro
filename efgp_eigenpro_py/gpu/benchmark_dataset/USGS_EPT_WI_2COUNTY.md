# USGS WI 2County EPT ground-elevation benchmarks

This route defines reproducible two-dimensional regression benchmarks from
two USGS 3DEP work units.  Horizontal position predicts the NAVD88 GEOID18
classification-2 ground elevation.  The second work unit used for independent
screening is Manitowoc County, `WI_2County_1_B23` (work unit 300411).

## Official Manitowoc source and frozen support

- EPT: `https://s3-us-west-2.amazonaws.com/usgs-lidar-public/WI_2County_1_B23/ept.json`
- official EPT point count: 24,781,638,330;
- official EPT `boundsConforming` in EPSG:3857:
  `[-9803688, 5447592, 161, -9740631, 5517414, 407]`;
- native work-unit CRS: NAD83(2011) / UTM zone 16N, EPSG:6345;
- transformed conforming horizontal extent: easting
  `[414205.75, 460013.37]` m and northing `[4859171.46, 4909623.85]` m;
- frozen full-work-unit bounding square in EPSG:6345: center
  `(437100, 4884400)` m, side 52,000 m.  It contains the complete conforming
  extent; empty portions outside the irregular work-unit boundary add no rows.

NOAA's [official InPort record](https://www.fisheries.noaa.gov/inport/item/72914)
identifies Manitowoc as work unit 300411, links the EPT source, and records that
EPT reprojection changed only the horizontal CRS to EPSG:3857.  The
[USGS project report](https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/metadata/WI_2County_B23/USGS_WI_2County_B23_Project_Report.pdf)
records EPSG:6345, EPSG:5703, Quality Level 1, and the collection dates.  The
[USGS work-unit report](https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/metadata/WI_2County_B23/WI_2County_1_B23/reports/WI_2County_B23_WU300411_Report.pdf)
reports 12.98 first returns/m2 and 11.37 classified-ground returns/m2.

The official 641 square-mile work-unit area is 1,660.1823787 km2.  Thus the
report-based full-density planning estimate is 18,876,273,646 class-2 points,
or 15,101,018,917 expected training-bucket points under the fixed 80/20 hash
split.  This is far above the 375 million distinct rows required for a
300M-train/75M-test artifact.  It is a planning estimate: a formal large
artifact must still scan the necessary EPT levels and verify exact bucket
capacity before writing output.

## Deterministic pipeline

1. Read only AOI-intersecting additive EPT nodes, shallower levels first.
2. Transform every point from EPSG:3857 to EPSG:6345, apply the exact frozen
   square test, retain finite LAS classification-2 points, and preserve NAVD88
   GEOID18 elevation in meters.
3. Form a stable source ID from EPT node rank and within-node point ordinal.
   Split with SplitMix64 modulo 5 (residue 0 is test), independently of target.
4. Order by EPT depth and then an independent SplitMix64 hash.  Requested sizes
   are therefore exact prefixes without replacement.
5. Scale both input coordinates by the fixed 52 km square.  Standardize targets
   with the first one million rows of the frozen training order and reuse that
   transform for every size and future deeper levels.

The checked LOD-6 build has exact class-2 hash-bucket capacity
40,787,587 train / 10,197,897 test and generated exact 100k, 1M, and 10M
training prefixes.  Pairwise array checks confirm that `100k` is an exact
prefix of `1M`, which is an exact prefix of `10M`; all values are finite, and
the 10M artifact contains
10,000,000 train plus 2,500,000 disjoint test rows.

Generate the checked size sweep with:

```powershell
python -m efgp_eigenpro_py.gpu.benchmark_dataset.preprocess_usgs_ept_wi_2county `
  --ept-json https://s3-us-west-2.amazonaws.com/usgs-lidar-public/WI_2County_1_B23/ept.json `
  --n-train-list 100000,1000000,10000000 `
  --aoi-center-x 437100 --aoi-center-y 4884400 --aoi-side-m 52000 `
  --aoi-crs EPSG:6345 --max-lod-depth 6 `
  --source-project WI_2County_1_B23 `
  --official-mean-ground-density 11.37 `
  --official-work-unit-area-km2 1660.1823787253759 `
  --dataset-stem-prefix USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation
```

For a 300M disk artifact, request `300000000`, pass `--allow-large-output`, and
increase the depth cap until the exact scan satisfies 300M/75M.  That artifact
is a preprocessing/scalability result and is not resident-runnable on a 4 GB
GPU.

## Matched 10M solver experiment

The primary controlled run uses one shared Matérn-3/2 Fourier system with
`lengthscale=0.1`, `lambda=0.1`, fp64 arithmetic, tolerance `1e-7`, rank 256,
one warm-up, and five independently shuffled measured repeats:

```powershell
python -m efgp_eigenpro_py.gpu.box_toeplitz_active_block.controlled.benchmark `
  --dataset-stem USGS_EPT_WI_2County_1_B23_full_workunit_ground_elevation_n10000000 `
  --n-train 0 --kernel matern --lengthscale 0.1 --nu 1.5 --variance 1 `
  --lambda 0.1 --fourier-eps 1e-5 --nufft-tol 1e-10 `
  --tol 1e-7 --maxiter 6000 --precision fp64 `
  --methods cg,jacobi,default,full-eig,nystrom,rpcholesky `
  --score-tau 1 --box-budget 8192 --inverse-max-size 1024 `
  --rank 256 --nystrom-rank 256 --rpcholesky-rank 256 `
  --eig-tol 1e-3 --eig-maxiter 1280 `
  --warmup-repeats 1 --measured-repeats 5 `
  --nufft-backend none --precompute-chunk-size 1000000 `
  --post-diagnostic-mode cheap --strict-gpu-eig `
  --output-dir efgp_eigenpro_py/gpu/box_toeplitz_active_block/controlled/outputs/integrated_manitowoc_n10000000_matern_q256_primary
```

Here `--nufft-backend none` resolves to the CPU FINUFFT construction route;
it does not mean that the Fourier setup is omitted.  The run records system ID
`6b333ce0ebe7...`, `M=34,225`, 2.146 s shared setup, and an unchanged system
fingerprint after all methods.  Median cold times are 11.021 s for CG and
3.367 s for the deployable default, a paired median speedup of 3.175x with
five wins in five repeats.  The default uses 67,749,888 bytes of stored
preconditioner data, versus 283,649,072 bytes for full-eig.

A separate accuracy-only audit evaluates all 2.5M test rows.  CG has
standardized test RMSE 0.086899118 and the default has 0.086899125
(ratio 1.000000079).  Its solve and prediction timings are audit-only and are
not used for the performance claim.
