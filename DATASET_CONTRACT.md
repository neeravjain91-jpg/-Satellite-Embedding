# Dataset Specification & Scientific Contract

## Project
**Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature**  
**Document**: Canonical Dataset Contract  
**Version**: 2.0 (Hardened Multi-Mask Observational Contract)  
**Target Domain**: North Indian Ocean  

---

## 1. Spatial Grid Specification

| Coordinate | Canonical Name | Bounds | Step / Resolution | Number of Points | Standard Name | Units |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Latitude** | `latitude` | `[5.00°N, 30.00°N]` | `0.25°` | `101` | `latitude` | `degrees_north` |
| **Longitude** | `longitude` | `[45.00°E, 105.00°E]` | `0.25°` | `241` | `longitude` | `degrees_east` |

- **Grid Alignment**: Regular equidistant rectangular grid, cell-center aligned at quarter-degree multiples (`5.00, 5.25, ..., 30.00` and `45.00, 45.25, ..., 105.00`).
- **Domain Coverage**: Arabian Sea, Bay of Bengal, Andaman Sea, Persian Gulf, Gulf of Oman, and northern equatorial Indian Ocean.
- **Sub-Regional Partitions**:
  - **Entire Domain**: Lat `[5.00, 30.00]`, Lon `[45.00, 105.00]`
  - **Arabian Sea**: Lat `[5.00, 30.00]`, Lon `[45.00, 77.50]`
  - **Bay of Bengal**: Lat `[5.00, 30.00]`, Lon `[77.50, 105.00]`

---

## 2. Vertical Coordinate Specification

The vertical subsurface coordinate has exactly **15 canonical depth levels**:

$$\mathcal{D} = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] \text{ meters}$$

- **Dimension Name**: `depth`
- **Units**: `m` (meters)
- **Direction**: Positive downwards (`positive: "down"`)
- **Vertical Resolution Strategy**: Dense sampling (0–150 m) to resolve the mixed layer and main thermocline dynamics, transitioning to wider spacing (200–1000 m) for deep water masses.

---

## 3. Temporal Coordinate Specification

- **Temporal Resolution**: Daily (`1D`)
- **Timestamp Standard**: ISO 8601 UTC representation (`YYYY-MM-DDT00:00:00Z`), indexed in xarray as `datetime64[ns]`.
- **Temporal Alignment**:
  - Daily aggregated/centered products (`OSTIA`, `GLORYS`, `DUACS`, `SSS`, `OSCAR`, `CCMP`) are synchronized to the identical UTC calendar day.
  - Multi-day NetCDF files (e.g. 7-day chunks) are indexed by exact datetime matching using `find_time_index_in_dataset()` to extract the target calendar day.

---

## 4. Four-Mask Architecture & Scientific Semantics

To prevent any conflation of coastlines, bathymetry, sensor dropouts, and subsurface limits, the dataset contract defines and enforces **four strictly separate masks**:

### 4.1. `geographic_ocean_mask`
- **Shape**: `(latitude: 101, longitude: 241)`
- **Type**: Boolean
- **Definition**: True if the grid coordinate is geographically part of the ocean, False if it is permanent landmass (e.g. Indian peninsula, Arabian peninsula, Indochina).
- **Invariance**: Static across time. It does **not** change if satellite data or GLORYS $\theta_o$ is temporarily missing at that cell.

### 4.2. `target_validity_mask`
- **Shape**: `(time: T, depth: 15, latitude: 101, longitude: 241)`
- **Type**: Boolean
- **Definition**: True if GLORYS $\theta_o$ ground-truth temperature is physically valid at that specific date, depth, and spatial coordinate. False if land, below local seabed, or unobserved.
- **Contract Rule**: All ML loss functions and evaluation metrics **must strictly compute loss over `target_validity_mask == True`**. Under no circumstances may target NaNs be converted to zero.

### 4.3. `depth_valid_mask` & `deepest_valid_depth_m`
- **Shape**:
  - `depth_valid_mask`: `(depth: 15, latitude: 101, longitude: 241)` Boolean
  - `deepest_valid_depth_m`: `(latitude: 101, longitude: 241)` Float32
- **Definition**: True where the local ocean water column depth is greater than or equal to the target depth level.
- **Contract Rule**: Preserves shallow-water physical limitations. In shallow regions (e.g. Persian Gulf or shelf seas where bathymetry is 60 m), `depth_valid_mask` is True at $0, 5, 10, 20, 30, 50$ m, and False at $75, 100, \dots, 1000$ m. Subsurface temperatures are **never** extrapolated below the seafloor.

### 4.4. `surface_validity_mask`
- **Shape**: `(time: T, latitude: 101, longitude: 241, feature: 7)`
- **Type**: Boolean
- **Definition**: True if the respective satellite surface predictor is valid and observed at that time and coordinate.
- **Missingness vs. Land Distinction**:
  - Geographic land: `geographic_ocean_mask == False` (and `surface_validity_mask == False`).
  - Missing satellite observation over ocean: `geographic_ocean_mask == True` AND `surface_validity_mask == False`.

---

## 5. Input Variables Specification

| Variable | Feature Index | Physical Name | Source Product | Native Res | Regridding Method | Units | Valid Physical Range |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| `sst` | 0 | Sea Surface Temperature | OSTIA L4 REP (`METOFFICE-GLO-SST-L4-REP-OBS-SST`) | 0.05° daily | Bilinear interpolation with ocean-boundary re-masking; Kelvin to Celsius conversion | °C | `[10.0, 35.0]` |
| `sss` | 1 | Sea Surface Salinity | Copernicus Multi-Obs (`cmems_obs-mob_glo_phy-sss_my_multi_P1D`) | 0.125° daily | Bilinear interpolation | PSU | `[12.0, 42.0]` |
| `ssh` | 2 | Sea Surface Height (SLA) | DUACS Two-Sat L4 (`c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D`) | 0.25° daily | Coordinate alignment | m | `[-1.5, 1.5]` |
| `current_u` | 3 | Ocean Current Zonal Velocity | OSCAR L4 OC Final V2.0 (`OSCAR_L4_OC_FINAL_V2.0`) | 0.25° daily | Coordinate alignment / Schema-aware subsetting | m/s | `[-3.0, 3.0]` |
| `current_v` | 4 | Ocean Current Meridional Velocity | OSCAR L4 OC Final V2.0 (`OSCAR_L4_OC_FINAL_V2.0`) | 0.25° daily | Coordinate alignment / Schema-aware subsetting | m/s | `[-3.0, 3.0]` |
| `wind_u` | 5 | Surface Wind Zonal Velocity (10m) | CCMP V3.1 6-hourly (`CCMP_WINDS_10M6HR_L4_V3.1`) | 0.25° 6-hr | Vector daily averaging; coordinate alignment | m/s | `[-35.0, 35.0]` |
| `wind_v` | 6 | Surface Wind Meridional Velocity (10m) | CCMP V3.1 6-hourly (`CCMP_WINDS_10M6HR_L4_V3.1`) | 0.25° 6-hr | Vector daily averaging; coordinate alignment | m/s | `[-35.0, 35.0]` |

---

## 6. Target Variable Specification & Scientific Scope

| Variable | Name | Source Product | Native Res | Interpolation & Regridding | Units | Valid Physical Range |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `thetao` | Sea Water Potential Temperature | GLORYS12V1 (`cmems_mod_glo_phy_my_0.083deg_P1D-m`) | 0.0833° horizontal, 50 native vertical levels, daily | 1. Vertical 1D linear interpolation to 15 canonical depths with seafloor cutoff.<br>2. Horizontal bilinear interpolation to canonical 0.25° grid. | °C | `[-2.0, 35.0]` |

### 6.1. Scientific Target Semantics & Learning Task
- **Reanalysis / Reference Target Distinction**: GLORYS $\theta_o$ is a numerical ocean reanalysis reference field, combining dynamical ocean physics (NEMO) with multi-sensor satellite and in-situ data assimilation. It is **explicitly distinguished from direct observational truth**.
- **ML Task Definition**: The ML framework is learning a surface-to-subsurface mapping with GLORYS $\theta_o$ as the training/reference field.
- **Evaluation Requirements**: Final observational validation requires an evaluation set whose relationship to GLORYS assimilation is explicitly established (differentiating between reanalysis-assimilated profiles and genuinely withheld observations).

---

## 7. Storage Formats & Dataset Containers

1. **Zarr Stores (Primary ML Store)**:
   - `real_ml_dataset_{mode}_surface.zarr`:
     - Arrays: `surface_features` $(T, 101, 241, 7)$, `surface_validity_mask` $(T, 101, 241, 7)$, `geographic_ocean_mask` $(101, 241)$
     - Chunks: `{"time": 1, "latitude": 101, "longitude": 241, "feature": 7}`
   - `real_ml_dataset_{mode}_target.zarr`:
     - Arrays: `thetao` $(T, 15, 101, 241)$, `target_validity_mask` $(T, 15, 101, 241)$, `depth_valid_mask` $(15, 101, 241)$, `deepest_valid_depth_m` $(101, 241)$, `geographic_ocean_mask` $(101, 241)$
     - Chunks: `{"time": 1, "depth": 15, "latitude": 101, "longitude": 241}`
2. **Canonical Ocean Mask Store**:
   - `data/processed/canonical_ocean_mask.nc`:
     - Variables: `geographic_ocean_mask`, `depth_valid_mask`, `deepest_valid_depth_m`.
3. **In-Situ Consistency Store**:
   - `data/processed/argo_matchup_evaluation.csv`:
     - Columns: `argo_id, time, latitude, longitude, depth, observed_temperature, model_temperature, spatial_distance, temporal_distance, quality_flag`.
     - Designation: **ARGO–GLORYS Reference Consistency Assessment** (solely for baseline assessment, not ML validation).
