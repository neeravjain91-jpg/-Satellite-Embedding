# Missing Data & Land Masking Report

**Domain**: North Indian Ocean (5.00°N–30.00°N, 45.00°E–105.00°E)  
**Grid**: Canonical 0.25° × 0.25° (101 × 241 = 24,341 horizontal cells)  
**Vertical Depth Levels**: 15 target depths (0 to 1000m)  

---

## 1. Spatial Land / Ocean Fraction

Within the study bounding box:
- **Total Horizontal Points**: 24,341 cells
- **Ocean Cells (Surface)**: 18,582 cells (76.34%)
- **Land Cells (Surface)**: 5,759 cells (23.66%)
- **Surface NaN Fraction**: Exactly 0.2366 across all 7 surface inputs (SST, SSS, SSH, current_u, current_v, wind_u, wind_v).

### Land Consistency
- All continental landmasses (Indian subcontinent, Arabian Peninsula, Horn of Africa, Indochina, Andaman/Nicobar islands) are masked identically across all variables using the single canonical ocean mask (`data/processed/canonical_ocean_mask.nc`).
- No surface variable interpolates or leaks artificial values over land.

---

## 2. Bathymetric Depth Cutoff & Subsurface Missingness

In shallow ocean regions (Persian Gulf, Gulf of Oman, Red Sea, Palk Strait, Gulf of Khambhat, and Andaman Sea shelf), the seabed depth is significantly shallower than 1000 meters.

- **Total 3D Grid Volume**: 7 days × 15 depths × 101 latitudes × 241 longitudes = 2,555,805 voxels
- **Overall Subsurface NaN Fraction**: 0.2977 (29.77%)
- **Physical Reason for Additional 6.11% Subsurface NaNs**:
  - Depths below local seafloor are strictly marked as invalid (NaN).
  - No fictitious ocean potential temperatures are fabricated below the local bathymetric depth.

---

## 3. Fill Values and Missing Data Handling

| Check | Result | Action Taken |
|---|---|---|
| Unmasked sentinel fill values (`-999`, `1e20`, etc.) | None detected | Replaced during ingestion with standard IEEE `NaN` |
| Floating point infinities (`Inf`, `-Inf`) | None detected | Validated |
| Incomplete days / missing time steps | 0 missing | Continuous daily time indexing |
| Model neural network handling | Zero-padded with validity mask | PyTorch loader provides `mask: (15, 101, 241)` tensor where True = valid ocean |

---

## 4. In-Situ ARGO Quality Control & Missingness

- **Raw Observations Retrieved**: 25,518 profile records
- **Good Quality Flag Filter (`quality_flag` ∈ [1, 2])**: 22,136 profile points retained (86.7% retention)
- **Spatial/Temporal Matchups**: 19,225 profile points matched with the 0.25° daily model grid
- **Unmatched Points (13.1%)**: Due to floats stationed near extreme boundary edges or timestamps with temporal distance > 24 hours.
