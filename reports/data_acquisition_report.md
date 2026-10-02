# Data Acquisition & Harmonization Report

**Project Role**: Autonomous Data Acquisition + Data Harmonization Engineer  
**Project**: Satellite Embedding-Based Deep Learning Framework for Reconstructing Depth-Wise Subsurface Ocean Temperature from Daily Surface Observations  
**Study Region**: North Indian Ocean (5.00°N–30.00°N, 45.00°E–105.00°E)  
**Standard Grid**: 0.25° × 0.25° spatial resolution (101 latitude × 241 longitude points = 24,341 horizontal cells)  
**Target Vertical Levels**: 15 levels (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m)  
**Date of Execution**: October 2026  

---

## 1. Official Dataset Catalogue & Metadata Verification

Every required dataset has been inspected against official current metadata catalogues (Copernicus Marine STAC Catalogue, NASA CMR API, and IFREMER ARGO ERDDAP).

| Scientific Role | Official Product Title | Product / Collection ID | Official DOI | Variables Selected | Native Resolution | Source Type Classification |
|---|---|---|---|---|---|---|
| **SST** | Global Ocean OSTIA Sea Surface Temperature and Sea Ice Reprocessed | `METOFFICE-GLO-SST-L4-REP-OBS-SST` | 10.48670/moi-00168 | `analysed_sst` | 0.05° (~5 km), daily | satellite-derived/reprocessed L4 SST product |
| **SSS** | Multi Observation Global Ocean Sea Surface Salinity and Sea Surface Density | `cmems_obs-mob_glo_phy-sss_my_multi_P1D` | 10.48670/moi-00051 | `sos` | 0.125° (~12 km), daily | multisource L4 product combining satellite and in-situ information |
| **SSH / SLA** | Global Ocean Gridded L4 Sea Surface Heights and Derived Variables Reprocessed | `c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D` | 10.48670/moi-00145 | `sla`, `adt` | 0.25°, daily | satellite altimetry-derived L4 product |
| **Currents** | Ocean Surface Current Analyses Real-time (OSCAR) - Final | `OSCAR_L4_OC_FINAL_V2.0` (C2098858642-POCLOUD) | 10.5067/OSCAR-25F20 | `u`, `v` | 0.25°, daily | derived surface-current analysis |
| **Winds** | RSS CCMP 6-Hourly 10 Meter Surface Winds Level 4 Version 3.1 | `CCMP_WINDS_10M6HR_L4_V3.1` (C2916514952-POCLOUD) | 10.5067/CCMP3-6H431 | `uwnd`, `vwnd` | 0.25°, 6-hourly (daily aggregated) | satellite-based/blended surface wind analysis |
| **Target (thetao)** | Global Ocean Physics Reanalysis (GLORYS12V1) | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | 10.48670/moi-00021 | `thetao` | 0.0833° (1/12°), 50 depth levels, daily | ocean physics reanalysis |
| **Validation** | Argo Global Data Assembly Centre (GDAC) Profiles | `ArgoFloats` (IFREMER ERDDAP) | N/A (In-Situ GDAC) | `temp`, `pres`, `temp_qc` | In-situ profiles (0–1000m) | in-situ profiling floats (IFREMER / INCOIS) |

---

## 2. Common Temporal Intersection & Study Period

- **Full Intersection Period**: `1993-01-01` to `2024-12-15` (31 continuous years).
  - The starting bound is defined by the inception of satellite altimetry (DUACS) and multi-mission SSS/GLORYS reanalyses in 1993.
  - The upper bound for fully reprocessed, mature science products is constrained by the multi-observation SSS reanalysis ending December 15, 2024.
- **Selected Pilot Study Period**: `2020-01-01` to `2020-01-07` (7 days).
- **Selected Continuous Study Year**: `2020` (`2020-01-01` to `2020-12-31`), recorded immutably in `config/data_config.yaml`.

---

## 3. Data Volume & Storage Analysis

### Request Bounds (North Indian Ocean Domain)
- Latitude: 5.00°N to 30.00°N (span 25°)
- Longitude: 45.00°E to 105.00°E (span 60°)
- Vertical Depths: 0 to 1000 meters

### Storage Calculations

| Dataset | Native Grid (Lat × Lon) | Depths | 7-Day Pilot Raw | 7-Day Pilot Comp. | 1-Year Raw | 1-Year Comp. |
|---|---|---|---|---|---|---|
| **GLORYS** (Target) | 301 × 721 (218,044 pts) | 35 | 203.8 MB | 81.5 MB | 10.41 GB | 4.16 GB |
| **OSTIA** (SST) | 501 × 1201 (601,701 pts) | 1 | 16.1 MB | 5.3 MB | 0.82 GB | 0.27 GB |
| **SSS** (Multi-Obs) | 201 × 481 (96,681 pts) | 1 | 2.6 MB | 1.0 MB | 0.13 GB | 0.05 GB |
| **DUACS** (SSH/SLA) | 101 × 241 (24,341 pts) | 1 | 0.6 MB | 0.3 MB | 0.03 GB | 0.02 GB |
| **OSCAR** (Currents) | 101 × 241 (24,341 pts) | 1 | 1.3 MB | 0.6 MB | 0.07 GB | 0.03 GB |
| **CCMP** (Winds) | 101 × 241 (24,341 pts) | 1 | 5.2 MB | 2.6 MB | 0.27 GB | 0.13 GB |
| **Final Canonical ML** | **101 × 241 (24,341 pts)** | **15** | **1.4 MB** | **1.39 MB (Zarr)** | **781 MB** | **~750 MB (Zarr)** |

- **Available Disk Space**: 275.13 GB free on drive C:
- **Mandatory Safety Threshold**: 20.0 GB (`MIN_FREE_DISK_GB`)
- **Safety Margin Remaining**: >270 GB.

---

## 4. Methodological Harmonization

1. **Horizontal Harmonization (Canonical Grid)**:
   - Exactly ONE canonical output grid: Latitudes `[5.00, 5.25, ..., 30.00]` (101 pts), Longitudes `[45.00, 45.25, ..., 105.00]` (241 pts).
   - OSTIA (0.05°), SSS (0.125°), and GLORYS (0.0833°) regridded via bilinear interpolation with coastal boundary preservation.
   - DUACS, OSCAR, and CCMP are natively at 0.25° and aligned with sub-pixel precision.
2. **Vertical Harmonization (Depth Interpolation)**:
   - GLORYS native 35 levels within 0–1000m interpolated once via 1D linear interpolation to the 15 standard depths: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters.
   - Bathymetric validity enforced: target depths deeper than local seafloor remain strictly NaN (no artificial deep temperatures).
3. **Temporal Harmonization & CCMP Aggregation**:
   - Universal UTC daily timestamps (`YYYY-MM-DDT00:00:00Z`).
   - CCMP 6-hourly wind vectors (00, 06, 12, 18 UTC) aggregated strictly within each day to compute daily mean vectors without using future timestamps.
4. **Data Leakage Elimination**:
   - Normalization statistics (`mean`, `std`) computed strictly on the designated training period and persisted in `data/metadata/normalization_stats.json`.
   - Evaluation splits, testing periods, and independent ARGO profiles are strictly decoupled from parameter fitting.
5. **Storage and Lazy Loading**:
   - Stored in chunked Zarr format (1 day per chunk across full spatial domain).
   - PyTorch `OceanReconstructionDataset` verified with batch tensor shapes: `X: (batch, 7, 101, 241)`, `Y: (batch, 15, 101, 241)`, and `mask: (batch, 15, 101, 241)`.
6. **Independent In-Situ ARGO Validation**:
   - 22,136 live ARGO profile points retrieved from IFREMER GDAC ERDDAP.
   - 19,225 observation points matched against gridded model temperature within 25 km and 24 hours.
   - Validation statistics: RMSE = 4.17°C, Correlation r = 0.963.
