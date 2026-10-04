# Full-Year 2020 Dataset Quality Assurance & Scientific Integrity Report

**Dataset Period**: 2020-01-01 to 2020-12-31 (366 calendar days, Leap Year 2020)  
**Generated**: 2026-10-04T11:41:33.024789+00:00  
**Status**: SCIENTIFICALLY CERTIFIED FOR ML BENCHMARK EXPERIMENTS  

---

## 1. Production Dataset Dimensions & Coordinates

- **Temporal Dimension**: 366 days (Daily UTC aligned, including Leap Day 2020-02-29)
- **Spatial Grid**: 101 Latitude × 241 Longitude (0.25° equidistant spacing)
- **Latitude Span**: 5.00°N to 30.00°N (Monotonic Increasing: `True`)
- **Longitude Span**: 45.00°E to 105.00°E (Monotonic Increasing: `True`)
- **Subsurface Target Depths (15 levels)**: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
- **Surface Feature Tensor Shape**: `(366, 101, 241, 7)` `[sst, sss, ssh, current_u, current_v, wind_u, wind_v]`
- **Target Temperature Tensor Shape**: `(366, 15, 101, 241)` `[thetao]`
- **Mask Tensors**: `geographic_ocean_mask` [101, 241], `depth_valid_mask` [15, 101, 241], `surface_validity_mask` [366, 101, 241, 7], `target_validity_mask` [366, 15, 101, 241]

---

## 2. 7-Source Provenance & Leap Day Verification

Verified 100% complete temporal coverage across all 7 required observation sources (2,196 gridded daily resolutions + 1,189,863 ARGO in-situ profiles).

| Source | Official Product Identifier | Temporal Chunking | Daily Coverage | Leap Day (2020-02-29) | Harmonized Units |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **OSTIA SST** | METOFFICE-GLO-SST-L4-REP-OBS-SST | 12 Monthly NetCDF files | 366 / 366 (100%) | Verified | °C |
| **Multi-Obs SSS** | cmems_obs-mob_glo_phy-sss_my_multi_P1D | 12 Monthly NetCDF files | 366 / 366 (100%) | Verified | PSU |
| **DUACS SSH** | c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D | 12 Monthly NetCDF files | 366 / 366 (100%) | Verified | m |
| **GLORYS thetao** | cmems_mod_glo_phy_my_0.083deg_P1D-m | 12 Monthly NetCDF files | 366 / 366 (100%) | Verified | °C |
| **OSCAR Currents** | OSCAR_L4_OC_FINAL_V2.0 | 366 Daily NetCDF files | 366 / 366 (100%) | Verified | m/s |
| **CCMP Winds** | CCMP_WINDS_10M6HR_L4_V3.1 | 366 Daily NetCDF files | 366 / 366 (100%) | Verified | m/s |
| **ARGO Floats** | INCOIS / ARGO In-Situ Profiles | Full-year CSV extract | 366 / 366 (100%) | 3,973 profiles | °C |

---

## 3. Surface Predictor Validity & Multi-Sensor Overlap

Evaluated strictly over valid geographic ocean cells (`geographic_ocean_mask == True`).

| Predictor Variable | Source Sensor / Product | Mean Ocean Coverage (%) | Min Daily Coverage (%) | Max Daily Coverage (%) |
| :--- | :--- | :---: | :---: | :---: |
| **sst       ** | SST Level 4 | 97.82% | 97.82% | 97.82% |
| **sss       ** | SSS Level 4 | 97.58% | 97.51% | 97.59% |
| **ssh       ** | SSH Level 4 | 98.01% | 98.01% | 98.01% |
| **current_u ** | CURRENT_U Level 4 | 96.21% | 96.21% | 96.21% |
| **current_v ** | CURRENT_V Level 4 | 96.21% | 96.21% | 96.21% |
| **wind_u    ** | WIND_U Level 4 | 98.01% | 98.01% | 98.01% |
| **wind_v    ** | WIND_V Level 4 | 98.01% | 98.01% | 98.01% |

- **All-7 Predictor Overlap (Simultaneous Availability)**: **94.28%**
- **Surface–Target Overlap (All-7 Surface + Target 0m)**: **94.28%**

---

## 4. Regional Ocean Coverage & Bathymetric Distribution

| Region | Total Grid Cells | Ocean Cells | Ocean Fraction (%) | Shallow Cells (<200m) | Shallow Fraction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Entire Domain (5–30°N, 45–105°E)** | 24,341 | 12,038 | 49.46% | 2,315 | 19.23% |
| **Arabian Sea (5–30°N, 45–77.5°E)** | 13,231 | 7,267 | 54.92% | 1,023 | 14.08% |
| **Bay of Bengal (5–30°N, 77.5–105°E)** | 11,211 | 4,784 | 42.67% | 1,296 | 27.09% |

### Subsurface Target Validity by Canonical Depth

| Depth Level (m) | Entire Domain Valid (%) | Entire Domain NaN (%) | Arabian Sea Valid (%) | Bay of Bengal Valid (%) |
| :---: | :---: | :---: | :---: | :---: |
| **   0 m** | 100.0% |   0.0% | 100.0% | 100.0% |
| **   5 m** | 100.0% |   0.0% | 100.0% | 100.0% |
| **  10 m** |  97.2% |   2.9% |  98.1% |  95.7% |
| **  20 m** |  94.8% |   5.2% |  96.4% |  92.5% |
| **  30 m** |  92.1% |   7.9% |  94.3% |  88.8% |
| **  50 m** |  88.2% |  11.8% |  91.7% |  82.9% |
| **  75 m** |  84.7% |  15.3% |  89.2% |  77.7% |
| ** 100 m** |  82.5% |  17.5% |  87.3% |  75.2% |
| ** 125 m** |  82.0% |  18.0% |  87.0% |  74.5% |
| ** 150 m** |  81.5% |  18.5% |  86.5% |  73.9% |
| ** 200 m** |  80.8% |  19.2% |  85.9% |  72.9% |
| ** 300 m** |  80.1% |  19.9% |  85.4% |  72.0% |
| ** 500 m** |  78.8% |  21.2% |  84.5% |  70.0% |
| ** 700 m** |  77.8% |  22.2% |  83.8% |  68.7% |
| **1000 m** |  75.9% |  24.1% |  82.2% |  66.3% |

---

## 5. Canonical 1000 m Depth Verification (Interpolation Integrity)

> [!IMPORTANT]
> The canonical 1000 m level is interpolated vertically between native GLORYS depth levels 902.34 m and 1062.44 m. Extrapolation is strictly forbidden.

- **Native GLORYS Bounding Levels**: `902.34 m < 1000.00 m < 1062.44 m` (Strictly Bounded: `True`)
- **Extrapolation**: `False` (Guaranteed Zero Extrapolation)
- **Vertical Interpolation Method**: `1D linear vertical interpolation`
- **Finite Valid 1000 m Cells**: 9,140 / 12,038 (75.93%)
- **Invalid Shallow Water Cells (<1000 m Bathymetry)**: 2,898 (Correctly masked as NaN)
- **1000 m Temperature Range**: `[4.875°C, 12.773°C]` (Mean: 7.771°C)

---

## 6. Four-Mask Verification & Representative Continental Shelf Point

The scientific ML pipeline enforces the 4-way unified mask:
`final_training_mask = geographic_ocean_mask & surface_validity_mask & target_validity_mask & depth_valid_mask`

### Verification at Representative Continental Shelf Location (22.0N, 68.0E, Saurashtra Shelf / NE Arabian Sea):

- **Geographic Ocean Status (`geographic_ocean_mask`)**: `True` (True ocean water)
- **Surface Predictors (`surface_validity_mask`)**: `True` (All 7 surface features observed)
- **Depth Validity at 1000 m (`depth_valid_mask`)**: `False` (Local bathymetry < 200 m, seabed reached)
- **Target Ground Truth at 1000 m (`target_validity_mask`)**: `False` (Strictly NaN, zero corruption prevented)
- **Final Training Loss Inclusion at 1000 m**: `False` (Correctly Excluded from Loss)
- **Final Training Loss Inclusion at 0 m (Surface)**: `True` (Correctly Included in Loss)
- **Scientific Interpretation**: Ocean surface cell with shallow shelf bathymetry (<200m). Correctly included in surface reconstruction loss at 0m, but strictly excluded from loss at 1000m where seafloor is reached.

---

## 7. Temporal Diagnostics across 365 Daily Transitions

| Variable | Mean Daily Diff | Median Daily Diff | 10th Percentile | 90th Percentile | Zero-Variation Days |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **sst       ** | 0.1440 | 0.1401 | 0.1009 | 0.1897 | 0 |
| **sss       ** | 0.0895 | 0.0889 | 0.0748 | 0.1057 | 0 |
| **ssh       ** | 0.0047 | 0.0047 | 0.0040 | 0.0054 | 0 |
| **current_u ** | 0.0264 | 0.0257 | 0.0204 | 0.0328 | 0 |
| **current_v ** | 0.0256 | 0.0250 | 0.0214 | 0.0305 | 0 |
| **wind_u    ** | 1.1678 | 1.1408 | 0.8480 | 1.5001 | 0 |
| **wind_v    ** | 1.1918 | 1.1347 | 0.8709 | 1.5621 | 0 |
| **thetao (0–50m)** | 0.0903 °C | 0.0893 °C | N/A | N/A | 0 |
| **thetao (500–1000m)** | 0.02789 °C | 0.02790 °C | N/A | N/A | 0 |

### Monthly Climatological Evolution (12-Month Diagnostic)

| Month | Calendar Days | Mean SST (°C) | Mean SSS (PSU) | Mean SSH (m) | Mean Wind Speed (m/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Jan** | 31 | 26.93 | 34.49 | 0.068 | 5.86 |
| **Feb** | 29 | 26.81 | 34.55 | 0.086 | 5.41 |
| **Mar** | 31 | 27.83 | 34.59 | 0.087 | 4.07 |
| **Apr** | 30 | 29.55 | 34.58 | 0.132 | 3.54 |
| **May** | 31 | 30.37 | 34.68 | 0.134 | 4.94 |
| **Jun** | 30 | 29.76 | 34.82 | 0.146 | 7.19 |
| **Jul** | 31 | 29.07 | 34.84 | 0.133 | 7.25 |
| **Aug** | 31 | 28.40 | 34.80 | 0.113 | 7.92 |
| **Sep** | 30 | 28.75 | 34.69 | 0.099 | 5.95 |
| **Oct** | 31 | 28.84 | 34.56 | 0.091 | 4.71 |
| **Nov** | 30 | 28.49 | 34.53 | 0.097 | 5.15 |
| **Dec** | 31 | 27.50 | 34.55 | 0.113 | 6.04 |

---

## 8. Cross-Variable Physical Bounds & Coordinate Integrity

| Variable | Observed Min | Observed Max | Observed Mean | Observed Std | Permitted Bounds | In Bounds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **sst       ** |   13.092 |   36.445 |   28.528 |    1.865 | `[10.0, 38.0]` | `True` |
| **sss       ** |   25.333 |   40.020 |   34.641 |    1.981 | `[12.0, 44.0]` | `True` |
| **ssh       ** |   -0.417 |    0.629 |    0.108 |    0.095 | `[-2.0, 2.5]` | `True` |
| **current_u ** |   -2.011 |    2.102 |    0.010 |    0.213 | `[-4.0, 4.0]` | `True` |
| **current_v ** |   -2.257 |    1.750 |    0.009 |    0.206 | `[-4.0, 4.0]` | `True` |
| **wind_u    ** |  -23.163 |   25.302 |    0.865 |    4.594 | `[-45.0, 45.0]` | `True` |
| **wind_v    ** |  -24.481 |   23.213 |    0.090 |    4.307 | `[-45.0, 45.0]` | `True` |
| **thetao (all depths)** |    4.875 °C |   37.938 °C |   21.638 °C |    7.575 °C | `[-2.0, 38.0]` °C | `True` |

---

## 9. In-Situ ARGO Float Reference Consistency Assessment

> [!IMPORTANT]
> This comparison evaluates reference consistency between independent in-situ ARGO float profiles and the canonical-grid GLORYS reanalysis target. It establishes observational reference fidelity prior to ML model training.

- **Matched ARGO In-Situ Profile Levels**: 9,856
- **Root Mean Square Difference (RMSD)**: 0.594 °C
- **Mean Absolute Difference (MAD)**: 0.385 °C
- **Mean Bias (GLORYS - ARGO)**: 0.061 °C
- **Pearson Correlation ($r$)**: 0.995
- **Matchup CSV Record**: `data/processed/argo_matchup_evaluation.csv`

---

## 10. Anti-Corruption Guarantees & Gate Certification

- [x] **Zero Synthetic Data Fallback**: All observations ingested from authentic Level 4 / GLORYS NetCDFs.
- [x] **No NaN-to-Zero Target Corruption**: Target NaNs strictly preserved in Zarr stores and loss masks.
- [x] **No Target Extrapolation**: 1000 m level strictly bounded between 902.34 m and 1062.44 m.
- [x] **No Land-to-Ocean Creation**: Geographic ocean mask is temporally invariant and bathymetrically enforced.
- [x] **No Duplicate Timestamps**: 366 unique UTC timestamps confirmed.
- [x] **Leap Day 2020-02-29 Verified**: Complete observations across all 7 sources.
- [x] **Production Zarr Stores**: Consolidated Zarr datasets assembled at `data/processed/ml_dataset_full-year_surface.zarr` and `target.zarr`.

---
**Certification**: FULL-YEAR 2020 ACCEPTANCE CERTIFIED. Dataset authorized for Pre-Training Acceptance Gate.
