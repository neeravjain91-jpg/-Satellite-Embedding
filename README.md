# Satellite Embedding-Based Subsurface Ocean Temperature Reconstruction

**Domain**: North Indian Ocean (5.00°N–30.00°N, 45.00°E–105.00°E)  
**Standard Grid**: Canonical 0.25° × 0.25° spatial resolution (101 latitude × 241 longitude points = 24,341 horizontal cells)  
**Temporal Resolution**: Daily  
**Target Depths (15 levels)**: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 meters  

---

## Scientific Overview

Reconstructing depth-resolved subsurface ocean temperature ($\theta_o$) from daily surface satellite and blended observation observations over the North Indian Ocean:
1. **Sea Surface Temperature (SST)**: OSTIA L4 Reprocessed (DOI: [10.48670/moi-00168](https://doi.org/10.48670/moi-00168))
2. **Sea Surface Salinity (SSS)**: Copernicus Multi-Observation SSS (DOI: [10.48670/moi-00051](https://doi.org/10.48670/moi-00051))
3. **Sea Level Anomaly (SSH/SLA)**: DUACS Two-Satellite Gridded SLA/ADT (DOI: [10.48670/moi-00145](https://doi.org/10.48670/moi-00145))
4. **Surface Currents ($u, v$)**: OSCAR L4 OC Final 0.25° (DOI: [10.5067/OSCAR-25F20](https://doi.org/10.5067/OSCAR-25F20))
5. **Surface Winds ($u, v$)**: RSS CCMP 6-Hourly 10m Wind Analysis V3.1 (DOI: [10.5067/CCMP3-6H431](https://doi.org/10.5067/CCMP3-6H431))
6. **Subsurface Target ($\theta_o$)**: GLORYS Global Ocean Physics Reanalysis 1/12° (DOI: [10.48670/moi-00021](https://doi.org/10.48670/moi-00021))
7. **Independent In-Situ Validation**: Global ARGO / INCOIS profiling floats

---

## Architecture & Codebase Structure

```
├── config/
│   └── data_config.yaml                  # Immutable configuration and temporal boundaries
├── data/
│   ├── raw/                              # Downloaded raw datasets
│   ├── interim/                          # Staging and test datasets
│   ├── processed/                        # Final standardized Zarr and NetCDF datasets
│   │   ├── canonical_ocean_mask.nc       # 2D surface and 3D bathymetric ocean masks
│   │   ├── ml_dataset_full-year_surface.zarr
│   │   ├── ml_dataset_full-year_target.zarr
│   │   └── argo_matchup_evaluation.csv   # Validation matchup table
│   ├── manifests/
│   │   ├── download_manifest.json        # Resumable download ledger
│   │   └── download_manifest.csv
│   ├── metadata/
│   │   ├── copernicus_detailed_metadata.json
│   │   └── normalization_stats.json      # Training-split-only fitted statistics
│   └── checksums.csv                     # SHA-256 integrity checks
├── preprocessing/
│   ├── canonical_grid.py                 # Single source of truth for coordinates
│   ├── depth_interpolation.py            # Vertical interpolation with seafloor cutoff
│   ├── ocean_mask.py                     # Canonical land and bathymetry masking
│   ├── regrid.py                         # Bilinear and sub-pixel coordinate regridding
│   ├── temporal_align.py                 # Daily UTC alignment & CCMP 6-hourly aggregation
│   ├── quality_control.py                # Automated QC suite
│   ├── fit_transform.py                  # Strict training split normalizer (zero leakage)
│   ├── apply_transform.py                # Transform and inverse transform application
│   ├── build_dataset.py                  # ML dataset assembly & PyTorch lazy DataLoader
│   └── argo_matchup.py                   # In-situ ARGO matchup and statistical evaluation
├── scripts/
│   ├── inspect_products.py               # STAC & CMR catalogue metadata inspector
│   ├── estimate_sizes.py                 # Pre-flight size estimation & disk safety
│   ├── manifest_manager.py               # Resumable checkpointing with retry backoff
│   ├── download_all.py                   # Master acquisition orchestrator
│   ├── download_glorys.py                # GLORYS thetao subsetting
│   ├── download_ostia.py                 # OSTIA SST subsetting
│   ├── download_sss.py                   # Multi-Obs SSS subsetting
│   ├── download_duacs.py                 # DUACS SSH/SLA subsetting
│   ├── download_oscar.py                 # OSCAR currents acquisition
│   ├── download_ccmp.py                  # CCMP 6-hourly winds acquisition
│   ├── download_argo.py                  # IFREMER GDAC in-situ float downloader
│   └── harmonize_and_validate.py         # End-to-end harmonization & validation suite
└── reports/
    ├── data_acquisition_report.md        # Provenance, volumes, and methodology
    ├── data_quality_report.csv           # Automated QC checks table
    ├── data_quality_report.md            # Detailed QC evaluation
    └── missing_data_report.md            # Missingness and bathymetry cutoff analysis
```

---

## Quickstart

### 1. Pre-flight Size Estimation & Disk Safety
```bash
python scripts/estimate_sizes.py
```

### 2. Run Data Acquisition (7-Day Pilot or Full-Year)
```bash
# Pilot mode
python scripts/download_all.py --pilot

# Full-year mode
python scripts/download_all.py --full-year
```

### 3. Run Harmonization, QC, and PyTorch Verification
```bash
python scripts/harmonize_and_validate.py
```

---

## Machine Learning Integration

The PyTorch dataset loader lazily streams chunks directly from Zarr format:

```python
from preprocessing.build_dataset import OceanReconstructionDataset
from torch.utils.data import DataLoader

dataset = OceanReconstructionDataset(
    surface_zarr_path="data/processed/ml_dataset_full-year_surface.zarr",
    target_zarr_path="data/processed/ml_dataset_full-year_target.zarr"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)
for batch in loader:
    x = batch["x"]      # (batch, 7_features, 101_lat, 241_lon)
    y = batch["y"]      # (batch, 15_depths, 101_lat, 241_lon)
    mask = batch["mask"]# (batch, 15_depths, 101_lat, 241_lon)
    # Train neural network...
```

---

## Quality Control & Validation Metrics

- **Automated QC Suite**: 8 variables evaluated across 2.5+ million grid points; 100% passed coordinate monotonicity, physical bounds, and fill value checks.
- **In-Situ ARGO Matchup**: 19,225 in-situ float observation points matched against the model grid with correlation $r = 0.963$.
