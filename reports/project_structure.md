# Repository Architecture & Directory Structure

**Project**: Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature  
**Repository**: `https://github.com/neeravjain91-jpg/-Satellite-Embedding`

---

## High-Level Layout

```
code/
├── data/                  # Raw, interim, and processed ocean datasets (Zarr stores & manifests)
├── models/                # PyTorch neural architectures, scikit-learn baselines, checkpoints
├── preprocessing/         # Regridding, canonical grid definitions, masking, tabular extraction
├── scripts/               # Production pipelines, training routines, benchmark compilation
├── tests/                 # 97 automated pytest unit and integration regression tests
├── results/               # Authoritative B0–B8 evaluation JSON manifests & checksums
├── reports/               # Formal scientific reports, audit logs, and QA summaries
├── src/                   # React 18 + TypeScript + Tailwind frontend interactive prototype
├── submission/            # Final presentation (PPTX), PDF document, and demo scripts
├── index.html             # Single-page application entry point
├── package.json           # Node / Vite build scripts and frontend dependencies
├── tsconfig.json          # Strict TypeScript compiler options
├── vite.config.ts         # Vite bundler configuration (port 3000)
├── vercel.json            # Vercel SPA routing rewrite rules
└── README.md              # Project overview, installation, benchmark summary, and evaluation
```

---

## Detailed Directory Breakdown

### 1. `data/`
- `raw/`: Raw downloaded NetCDF files from Earthdata, Copernicus Marine, and RSS (OSTIA, SMAP, DUACS, OSCAR, CCMP, GLORYS).
- `interim/`: Temporarily chunked and validated regridded arrays during harmonization.
- `processed/`: Production Zarr v3 datasets (`full_year_2020_surface.zarr`, `full_year_2020_target.zarr`, `scalers.json`) certified across 366 days.

### 2. `models/`
- `base.py`: Abstract baseline interfaces, masked loss functions (`masked_mse_loss`), and metric calculation utilities.
- `baselines.py`: Implementations for B0 (Persistence), B0b (Day-252 Persistence), B1 (Climatology), B2 (Multi-Output Ridge), B5 (Pointwise MLP), B6 (Spatial CNN), B7 (Temporal GRU), and B8 (Spatiotemporal Embedding Model).
- `checkpoints/`: Instantiated PyTorch weights (`.pt`) and hyperparameter tuning records.

### 3. `preprocessing/`
- `canonical_grid.py`: Authoritative domain grid definitions ($5^\circ-30^\circ\text{N}$, $45^\circ-105^\circ\text{E}$ at 0.25°), 7 surface feature names, and 15 canonical ocean depths.
- `ocean_mask.py`: Four-way composite mask construction (geographic ocean, surface observation validity, GLORYS target validity, and GEBCO bathymetry cutoff).
- `regrid.py`: Normalized bilinear interpolation with coastal zero-bleed prevention.
- `tabular_dataset.py`: Temporal partition extraction, purge buffer enforcement (6-day gaps), and train-only z-score normalization.

### 4. `scripts/`
- `harmonize_and_validate.py`: Full-year 2020 production pipeline with chunked spatial-temporal execution.
- `compile_master_benchmark.py`: Aggregates B0–B8 result manifests into `results/master_benchmark_summary.json` and markdown summaries.
- `serve_dashboard.py`: Lightweight Python local server utility.

### 5. `tests/` (97 Automated Tests Passing)
- `test_b6_b7_b8_genuine_context.py`: Validates non-degenerate spatial patches ($3 \times 3$), temporal sequences ($T=5$), and cubes ($5 \times 3 \times 3$).
- `test_ml_protocol_splits_and_leakage.py`: Enforces zero purge leakage across Train (0–252), Purge 1 (253–258), Val (259–306), Purge 2 (307–312), and Test (313–365).
- `test_ml_baselines_real_models.py`: Instantiates actual neural models and verifies gradients and parameter counts.
- `test_scientific_masks_and_integrity.py`: Verifies bathymetric target NaN preservation and masked loss integrity.
- `test_regrid_coastal_no_zero_bleeding.py`: Ensures normalized regridding never introduces fake zero values into ocean cells.

### 6. `results/`
- `B0.json` through `B8.json`: Authoritative evaluation metrics, depth-wise errors, regional and seasonal breakdowns, and paired 95% bootstrap confidence intervals.
- `master_benchmark_summary.json`: Unified machine-readable JSON containing the full model hierarchy and SHA-256 artifact checksums.

### 7. `reports/`
- `final_project_report.md`: Complete 31-section comprehensive scientific report.
- `final_results_table.md`: Master tabular comparison across all 10 models with instantiated parameter counts.
- `final_methodology.md`: In-depth specification of dataflow, tensors, architecture, and validation protocol.
- `final_limitations.md`: Scientific disclosure of reference target semantics and operational constraints.

### 8. `src/` (Frontend Prototype)
- `components/layout/`: Navigation header with persistent `PROTOTYPE • LOCAL SIMULATION` status badge and sidebar.
- `components/views/`: 8 distinct views: Overview Dashboard, Ocean Surface Explorer, Reconstruction Pipeline Simulation, Vertical Depth Profile Inspector, 128-D Latent Space Studio, Benchmark Hierarchy Comparison, Scientific Validation & Error Breakdown, and Methodology Specification.
- `components/shared/`: Custom interactive SVG visualizations for the North Indian Ocean basin, inverted vertical depth profile charts (0–1000m), and 2D latent scatter plots.
- `mock/`: Fully self-contained, deterministic physics-guided simulation data and exact certified benchmark constants.

### 9. `submission/`
- `Final_Project_Presentation.pptx`: Slide deck for project defense and examiner evaluation.
- `Final_Project_Submission.pdf`: Clean submission document formatted for academic submission.
- `demo_flow.md`: 10-step scripted walkthrough for live examiner demonstration.
- `FINAL_SUBMISSION_CHECKLIST.md`: Verification audit checklist.
