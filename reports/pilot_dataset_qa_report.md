# Pilot Dataset Quality Assurance & Scientific Integrity Report

**Pilot Period**: 2020-01-01 to 2020-01-07 (7 calendar days)  
**Generated**: 2026-10-03T06:25:15.929597+00:00  
**Status**: VALIDATED & CERTIFIED  

---

## 1. Spatial & Temporal Coordinate Integrity

- **Latitude Points**: 101 (Range: 5.00°N to 30.00°N)
- **Longitude Points**: 241 (Range: 45.00°E to 105.00°E)
- **Grid Spacing**: 0.25° Latitude × 0.25° Longitude (Regular Equidistant)
- **Monotonicity**: Latitude Monotonic Increasing: `True`, Longitude Monotonic Increasing: `True`
- **Subsurface Depths (15 Canonical Levels)**: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
- **Temporal Resolution**: Daily, UTC aligned across all 7 sources

---

## 2. Regional Ocean Coverage & Bathymetric Distribution

| Region | Total Grid Cells | Ocean Cells | Ocean Fraction (%) | Shallow Cells (<200m) | Shallow Fraction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Entire Domain (5–30°N, 45–105°E)** | 24,341 | 12,038 | 49.46% | 2,315 | 19.23% |
| **Arabian Sea (5–30°N, 45–77.5°E)** | 13,231 | 7,267 | 54.92% | 1,023 | 14.08% |
| **Bay of Bengal (5–30°N, 77.5–105°E)** | 11,211 | 4,784 | 42.67% | 1,296 | 27.09% |

### Deepest Valid Depth Distribution (Ocean Bathymetry Profile)

| Depth Level (m) | Entire Domain (Cells) | Entire Domain (%) | Arabian Sea (Cells) | Bay of Bengal (Cells) |
| :---: | :---: | :---: | :---: | :---: |
| **0 m** | 0 | 0.00% | 0 | 0 |
| **5 m** | 343 | 2.85% | 138 | 205 |
| **10 m** | 278 | 2.31% | 124 | 154 |
| **20 m** | 328 | 2.72% | 153 | 176 |
| **30 m** | 475 | 3.95% | 190 | 285 |
| **50 m** | 424 | 3.52% | 179 | 246 |
| **75 m** | 261 | 2.17% | 141 | 121 |
| **100 m** | 55 | 0.46% | 20 | 35 |
| **125 m** | 61 | 0.51% | 33 | 28 |
| **150 m** | 90 | 0.75% | 45 | 46 |
| **200 m** | 83 | 0.69% | 40 | 43 |
| **300 m** | 160 | 1.33% | 66 | 94 |
| **500 m** | 117 | 0.97% | 50 | 67 |
| **700 m** | 9,363 | 77.78% | 6,088 | 3,284 |
| **1000 m** | 0 | 0.00% | 0 | 0 |

---

## 3. Daily Surface Predictor Completeness & Overlap

Completeness is evaluated strictly over valid geographic ocean cells (`geographic_ocean_mask == True`).

| Date | Region | SST (%) | SSS (%) | SSH (%) | Current U/V (%) | Wind U/V (%) | All-7 Predictor Overlap (%) | Surface–Target Overlap (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2020-01-01 | Domain | 97.8% | 97.6% | 98.0% | 96.2% | 98.0% | 94.3% | 94.3% |
| 2020-01-01 | Arabian Sea | 97.9% | 97.8% | 98.1% | 96.7% | 98.1% | 94.9% | 94.9% |
| 2020-01-01 | Bay of Bengal | 97.6% | 97.3% | 97.8% | 95.5% | 97.8% | 93.3% | 93.3% |
| 2020-01-02 | Domain | 97.8% | 97.6% | 98.0% | 96.2% | 98.0% | 94.3% | 94.3% |
| 2020-01-02 | Arabian Sea | 97.9% | 97.8% | 98.1% | 96.7% | 98.1% | 94.9% | 94.9% |
| 2020-01-02 | Bay of Bengal | 97.6% | 97.3% | 97.8% | 95.5% | 97.8% | 93.3% | 93.3% |
| 2020-01-03 | Domain | 97.8% | 97.6% | 98.0% | 96.2% | 98.0% | 94.3% | 94.3% |
| 2020-01-03 | Arabian Sea | 97.9% | 97.8% | 98.1% | 96.7% | 98.1% | 94.9% | 94.9% |
| 2020-01-03 | Bay of Bengal | 97.6% | 97.3% | 97.8% | 95.5% | 97.8% | 93.3% | 93.3% |
| 2020-01-04 | Domain | 97.8% | 97.6% | 98.0% | 96.2% | 98.0% | 94.3% | 94.3% |
| 2020-01-04 | Arabian Sea | 97.9% | 97.8% | 98.1% | 96.7% | 98.1% | 94.9% | 94.9% |
| 2020-01-04 | Bay of Bengal | 97.6% | 97.3% | 97.8% | 95.5% | 97.8% | 93.3% | 93.3% |
| 2020-01-05 | Domain | 97.8% | 97.6% | 98.0% | 96.2% | 98.0% | 94.3% | 94.3% |
| 2020-01-05 | Arabian Sea | 97.9% | 97.8% | 98.1% | 96.7% | 98.1% | 94.9% | 94.9% |
| 2020-01-05 | Bay of Bengal | 97.6% | 97.3% | 97.8% | 95.5% | 97.8% | 93.3% | 93.3% |
| 2020-01-06 | Domain | 97.8% | 97.6% | 98.0% | 96.2% | 98.0% | 94.3% | 94.3% |
| 2020-01-06 | Arabian Sea | 97.9% | 97.8% | 98.1% | 96.7% | 98.1% | 94.9% | 94.9% |
| 2020-01-06 | Bay of Bengal | 97.6% | 97.3% | 97.8% | 95.5% | 97.8% | 93.3% | 93.3% |
| 2020-01-07 | Domain | 97.8% | 97.6% | 98.0% | 96.2% | 98.0% | 94.3% | 94.3% |
| 2020-01-07 | Arabian Sea | 97.9% | 97.8% | 98.1% | 96.7% | 98.1% | 94.9% | 94.9% |
| 2020-01-07 | Bay of Bengal | 97.6% | 97.3% | 97.8% | 95.5% | 97.8% | 93.3% | 93.3% |

---

## 4. Subsurface Target Completeness & NaN Fraction by Depth

Target availability is dictated by bathymetry: shallow waters naturally transition to NaN below local seafloor depth.

| Depth (m) | Entire Domain Valid (%) | NaN (%) | Arabian Sea Valid (%) | NaN (%) | Bay of Bengal Valid (%) | NaN (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **   0 m** | 100.0% |   0.0% | 100.0% |   0.0% | 100.0% |   0.0% |
| **   5 m** | 100.0% |   0.0% | 100.0% |   0.0% | 100.0% |   0.0% |
| **  10 m** |  97.2% |   2.8% |  98.1% |   1.9% |  95.7% |   4.3% |
| **  20 m** |  94.8% |   5.2% |  96.4% |   3.6% |  92.5% |   7.5% |
| **  30 m** |  92.1% |   7.9% |  94.3% |   5.7% |  88.8% |  11.2% |
| **  50 m** |  88.2% |  11.8% |  91.7% |   8.3% |  82.9% |  17.1% |
| **  75 m** |  84.6% |  15.4% |  89.2% |  10.8% |  77.7% |  22.3% |
| ** 100 m** |  82.5% |  17.5% |  87.3% |  12.7% |  75.2% |  24.8% |
| ** 125 m** |  82.0% |  18.0% |  87.0% |  13.0% |  74.5% |  25.5% |
| ** 150 m** |  81.5% |  18.5% |  86.5% |  13.5% |  73.9% |  26.1% |
| ** 200 m** |  80.8% |  19.2% |  85.9% |  14.1% |  72.9% |  27.1% |
| ** 300 m** |  80.1% |  19.9% |  85.4% |  14.6% |  72.0% |  28.0% |
| ** 500 m** |  78.8% |  21.2% |  84.5% |  15.5% |  70.0% |  30.0% |
| ** 700 m** |  77.8% |  22.2% |  83.8% |  16.2% |  68.6% |  31.4% |
| **1000 m** |   0.0% | 100.0% |   0.0% | 100.0% |   0.0% | 100.0% |

---

## 5. ARGO–GLORYS Reference Consistency Assessment

> [!IMPORTANT]
> This comparison evaluates reference consistency between in-situ ARGO float profiles and the canonical-grid GLORYS reanalysis field. It does **not** constitute ML model validation, as no ML model predictions have been evaluated.

- **Matched ARGO Observation Points**: 16,711
- **Root Mean Square Difference (RMSD)**: 0.598 °C
- **Mean Absolute Difference (MAD)**: 0.397 °C
- **Mean Difference (Bias)**: 0.11 °C
- **Correlation Coefficient ($r$)**: 0.995
- **Reference Dataset Record**: `data/processed/argo_matchup_evaluation.csv`

---

## 6. Provenance & Lineage Verification

Verified 42 / 42 required source-day physical observations across 6 gridded sources for all 7 pilot days.

| Source | Product Name | Temporal Chunking | Regridding Method | Harmonized Units |
| :--- | :--- | :--- | :--- | :--- |
| **OSTIA SST** | METOFFICE-GLO-SST-L4-REP-OBS-SST | 7-day NetCDF chunk | Bilinear (Kelvin $	o$ Celsius) | °C |
| **Multi-Obs SSS** | cmems_obs-mob_glo_phy-sss_my_multi_P1D | 7-day NetCDF chunk | Bilinear | PSU |
| **DUACS SSH** | c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D | 7-day NetCDF chunk | Coordinate alignment | m |
| **GLORYS thetao** | cmems_mod_glo_phy_my_0.083deg_P1D-m | 7-day NetCDF chunk | 1D Depth interp + Bilinear | °C |
| **OSCAR Currents** | OSCAR_L4_OC_FINAL_V2.0 | Daily NetCDF files | Coordinate alignment | m/s |
| **CCMP Winds** | CCMP_WINDS_10M6HR_L4_V3.1 | Daily NetCDF files | Coordinate alignment | m/s |
| **ARGO Floats** | ARGO / INCOIS In-Situ Profiles | Multi-day CSV extract | Nearest-neighbor + 1D depth interp | °C |

---
**Certification**: Acceptance Criteria Satisfied. Zero synthetic data fallback. Full provenance preserved.
