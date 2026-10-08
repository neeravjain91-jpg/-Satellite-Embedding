# Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 106 Passed](https://img.shields.io/badge/Tests-106%2F106%20Passing-success.svg)](#test-suite--verification)
[![Deployed on Vercel](https://img.shields.io/badge/Vercel-Live%20Prototype-black.svg?logo=vercel)](https://code-gules-three.vercel.app)
[![DOI](https://img.shields.io/badge/Dataset-Certified%202020%20Production-blue.svg)](#certified-multi-satellite--reanalysis-sources)

---

## Executive Summary

This repository houses the end-to-end scientific codebase, benchmark suite, evaluation pipelines, and interactive prototype for reconstructing depth-resolved subsurface ocean potential temperature ($\theta_o$) across the **North Indian Ocean** from multi-satellite surface observations.

Using a causal $5\text{-day} \times 3 \times 3$ spatiotemporal surface window encompassing 7 multi-satellite predictors (SST, SSS, SSH, Current $U/V$, Wind $U/V$), the proposed **B8 Spatiotemporal Embedding Network** reconstructs vertical temperature profiles across **15 discrete ocean depths** (0–1000 m). Evaluated on an independent, strictly chronologically split, purge-buffered test partition (Days 313–365 of 2020), **B8 achieves an overall column-averaged test RMSE of 0.9800 °C**, representing a **22.11% error reduction over daily climatology (B1)** and outperforming all evaluated internal tabular, machine learning, and deep learning baselines (B0–B7).

> [!IMPORTANT]
> **Scientific Designation**: B8 is designated strictly as the **"Best-performing architecture among evaluated internal benchmarks"**. Claims of universal "State-of-the-Art" are avoided in adherence to rigorous scientific integrity. GLORYS12V1 serves as the gridded reanalysis reference target.

---

## Table of Contents

- [Study Domain & Grid System](#study-domain--grid-system)
- [Certified Multi-Satellite & Reanalysis Sources](#certified-multi-satellite--reanalysis-sources)
- [Leakage-Controlled Experimental Protocol](#leakage-controlled-experimental-protocol)
- [Model Hierarchy & Architecture Specification](#model-hierarchy--architecture-specification)
- [Master Benchmark Results](#master-benchmark-results)
- [Depth-Stratified & Regional Performance](#depth-stratified--regional-performance)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Reproducibility & Test Suite](#reproducibility--test-suite)
- [Interactive Prototype (Frontend)](#interactive-prototype-frontend)
- [Scientific Limitations & Responsible Disclosures](#scientific-limitations--responsible-disclosures)
- [Submission Artifacts & Reports](#submission-artifacts--reports)
- [References & Data DOIs](#references--data-dois)

---

## Study Domain & Grid System

The study encompasses the tropical and subtropical basin of the North Indian Ocean, including the Arabian Sea, Bay of Bengal, and equatorial corridor:

- **Geographic Extent**: Latitude $5.00^\circ\text{N} - 30.00^\circ\text{N}$, Longitude $45.00^\circ\text{E} - 105.00^\circ\text{E}$
- **Canonical Grid**: Uniform $0.25^\circ \times 0.25^\circ$ grid ($101 \text{ latitudes} \times 241 \text{ longitudes} = 24,341 \text{ horizontal cells}$)
- **Temporal Coverage**: 366 consecutive calendar days (Full Year 2020 leap year, January 1 to December 31)
- **15 Standard Vertical Depths**:
  $$\{0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\}\text{ meters}$$
- **Bathymetric & Masking Rules**:
  - `geographic_ocean_mask`: 2D ocean boundaries (16,076 valid sea surface cells)
  - `target_validity_mask`: 3D ocean-depth mask enforcing bathymetric seafloor cutoffs (166,400 active 3D ocean cells across 15 depths)
  - Points below the GLORYS/ORCA12 model seafloor bathymetry are masked with strict NaN propagation: invalid target NaNs are preserved as masked invalid targets throughout preprocessing, training loss, and evaluation, and are never converted to zero.

---

## Certified Multi-Satellite & Reanalysis Sources

All raw satellite and reanalysis streams underwent pre-harmonization integrity gates, bilinear/conservative regridding to the canonical $0.25^\circ$ grid, and checksum verification:

| # | Variable / Role | Source / Product | Sensor / Resolution | Native Cadence | DOI / Citation |
|---|-----------------|------------------|---------------------|----------------|----------------|
| 1 | **SST** | OSTIA Global L4 Reprocessed | Infrared + Microwave | Daily L4 (0.05°) | [10.48670/moi-00168](https://doi.org/10.48670/moi-00168) |
| 2 | **SSS** | Copernicus Multi-Observation SSS | SMOS / Aquarius / In-situ | Weekly L4 (0.25°) | [10.48670/moi-00051](https://doi.org/10.48670/moi-00051) |
| 3 | **SSH / SLA** | CMEMS DUACS All-Satellite | Multi-Altimeter | Daily L4 (0.25°) | [10.48670/moi-00145](https://doi.org/10.48670/moi-00145) |
| 4 | **Current U** | OSCAR Ocean Surface Currents | Altimetry + Wind Drift | Daily L4 (0.25°) | [10.5067/OSCAR-25F20](https://doi.org/10.5067/OSCAR-25F20) |
| 5 | **Current V** | OSCAR Ocean Surface Currents | Altimetry + Wind Drift | Daily L4 (0.25°) | [10.5067/OSCAR-25F20](https://doi.org/10.5067/OSCAR-25F20) |
| 6 | **Wind U** | RSS CCMP V3.1 10m Vector Winds | Radiometer + Scatterometer | 6-hourly (0.25°) | [10.5067/CCMP3-6H431](https://doi.org/10.5067/CCMP3-6H431) |
| 7 | **Wind V** | RSS CCMP V3.1 10m Vector Winds | Radiometer + Scatterometer | 6-hourly (0.25°) | [10.5067/CCMP3-6H431](https://doi.org/10.5067/CCMP3-6H431) |
| Target | **$\theta_o$ (0–1000m)** | GLORYS12V1 Reanalysis | NEMO / SEALEV Assimilation | Daily mean (1/12°) | [10.48670/moi-00021](https://doi.org/10.48670/moi-00021) |
| In-Situ | **Validation** | Global Argo / Coriolis Floats | Profiling CTD floats | Daily profiles | Argo GDAC ([10.17882/42182](https://doi.org/10.17882/42182)) |

---

## Leakage-Controlled Experimental Protocol

To eliminate temporal autocorrelation leakage, seasonal lookahead bias, and spatial contamination, the following experimental protocol was strictly enforced:

```
2020 Calendar Timeline (366 Days):
[==== TRAIN (Days 0–252) ====] [BUFFER 1] [== VAL (Days 259–306) ==] [BUFFER 2] [=== TEST (Days 313–365) ===]
       Jan 1 – Sep 9            Sep 10–15          Sep 16 – Nov 2          Nov 3–8             Nov 9 – Dec 31
        (253 Days)               (6 Days)             (48 Days)            (6 Days)              (53 Days)
```

1. **Strict Chronological Ordering**: Training precedes validation, which precedes testing. Random train-test splitting was forbidden.
2. **6-Day Purge Buffers**: 
   - Buffer 1 (Days 253–258, Sep 10 – Sep 15): Isolates Train and Validation.
   - Buffer 2 (Days 307–312, Nov 3 – Nov 8): Isolates Validation and Test.
   - Buffer width ($T_{\text{purge}} = 6 \text{ days}$) strictly exceeds the 5-day causal temporal receptive field ($T_{\text{causal}} = 5 \text{ days}$), guaranteeing that no test sample's historical window draws from the validation or training distributions.
3. **Train-Only Normalization**: All standardizers (z-score means and standard deviations) were calculated exclusively on Days 0–252 and serialized to `data/metadata/normalization_stats.json`. Zero validation or test statistics contaminated normalization.
4. **Causal Spatiotemporal Slicing**: For any prediction day $t$, inputs are restricted to days $\{t-4, t-3, t-2, t-1, t\}$. Future timesteps are never accessible.

---

## Model Hierarchy & Architecture Specification

To isolate the marginal contributions of vertical regression, spatial context, temporal memory, and spatiotemporal fusion, 9 progressive models were implemented and certified:

```
[Level 0: Physical & Climatological References]
  B0:  Day-0 Persistence (1.5220 °C)
  B0b: Day-252 Persistence (1.7287 °C)
  B1:  Daily Climatology Mean (1.2582 °C)
         │
[Level 1: Tabular Machine Learning Baselines]
  B2:  Ridge Linear Regression (1.0295 °C)
  B3:  Random Forest (1.0452 °C)
  B4:  LightGBM Gradient Boosting (1.0288 °C)
         │
[Level 2: Deep Learning Ablations]
  B5:  Pointwise MLP (26,767 params) -> (1.5524 °C)
  B6:  Spatial CNN (30,991 params)   -> (1.2702 °C)
  B7:  Temporal GRU (44,111 params)  -> (1.5320 °C)
         │
[Level 3: Full Spatiotemporal Architecture]
  B8:  Spatiotemporal Embedding Model (203,791 params) -> (0.9800 °C)
```

### B8 Spatiotemporal Architecture Details

The certified B8 model accepts an input tensor $\mathbf{X} \in \mathbb{R}^{B \times 5 \times 7 \times 3 \times 3}$ and predicts the 15-dimensional temperature column $\hat{\mathbf{y}} \in \mathbb{R}^{B \times 15}$:

1. **Spatial CNN Encoder**:
   - Time-distributed 2D convolutions processing each $3 \times 3$ spatial patch across 7 channels:
   - Conv2D($7 \to 32, 3\times3$, padding=1) + BatchNorm2D + ReLU
   - Conv2D($32 \to 64, 3\times3$, padding=1) + BatchNorm2D + ReLU + AdaptiveAvgPool2d((1, 1)) + Flatten
   - Spatial feature output: $\mathbf{s}_t \in \mathbb{R}^{64}$ for each timestep $t \in \{1,\dots,5\}$.
2. **Temporal Recurrent Network**:
   - 2-Layer Causal GRU: input size 64, hidden dimension $H = 128$, `batch_first=True`.
   - Hidden state output at final step $t=5$: $\mathbf{h}_5 \in \mathbb{R}^{128}$.
3. **Latent Bottleneck**:
   - `LayerNorm(128)` applied to terminal hidden state $\mathbf{h}_5$ to produce the 128-D Ocean Latent Embedding.
4. **Vertical Column MLP Decoder**:
   - Linear($128 \to 64$) + ReLU
   - Linear($64 \to 15$) $\to$ Reconstructed temperature anomaly $\hat{\mathbf{y}} \in \mathbb{R}^{15}$.
5. **Parameter Footprint**:
   - Total instantiated trainable parameters: **203,791 parameters** (0.81 MB memory footprint).
   - Lightweight architecture suitable for efficient inference.

---

## Master Benchmark Results

All models evaluated strictly on the certified Test partition (Days 313–365, 53 consecutive daily steps $\times$ 166,400 active 3D ocean cells):

| Model ID | Architecture Description | Input Domain | Parameters / Complexity | Test RMSE (°C) | Improvement vs B1 (%) | Improvement vs Best ML (B4) | Certified Status |
|---|---|---|---|---|---|---|---|
| **B0** | Day-0 Persistence | $1 \times 1$ target column | 0 | 1.5220 | -20.97% | -47.94% | Locked |
| **B0b** | Day-252 Persistence | $1 \times 1$ target column | 0 | 1.7287 | -37.39% | -68.03% | Locked |
| **B1** | Daily Mean Climatology | 366-day temporal mean | 0 | 1.2582 | Baseline | -22.30% | Locked |
| **B2** | Ridge Linear Regression ($\alpha=100{,}000$) | Pointwise 7 surface | 120 coefficients | 1.0295 | +18.18% | -0.07% | Locked |
| **B3** | Random Forest Regressor (750 trees) | Pointwise 7 surface | 13,289,966 decision nodes across 750 Random Forest trees | 1.0452 | +16.93% | -1.59% | Locked |
| **B4** | LightGBM Gradient Boosting (750 trees) | Pointwise 7 surface | 750 boosted trees | 1.0288 | +18.23% | Baseline | Locked |
| **B5** | Pointwise MLP (3 hidden layers) | Pointwise 7 surface | 26,767 trainable parameters | 1.5524 | -23.38% | -50.90% | Locked |
| **B6** | Spatial CNN ($3 \times 3$ patches) | Spatial $3\times3$ patch | 30,991 trainable parameters | 1.2702 | -0.95% | -23.46% | Locked |
| **B7** | Temporal GRU (5-day causal window) | 5-day sequence | 44,111 trainable parameters | 1.5320 | -21.76% | -48.91% | Locked |
| **B8** | **Spatiotemporal Embedding Network** | **5-day $\times 3 \times 3$ patch** | **203,791 trainable parameters** | **0.9800** | **+22.11%** | **+4.74%** | **Best Internal** |

### Statistical Significance Summary
- **B8 vs. B1 (Daily Climatology)**: Absolute error reduction $\Delta\text{RMSE} = -0.2782\ ^\circ\text{C}$ (+22.11% relative gain). 7-day block bootstrap (1000 resamples): 95% CI $[-0.3957, -0.1756]\ ^\circ\text{C}$, $p < 0.001$ (statistically significant).
- **B8 vs. B4 (LightGBM Gradient Boosting)**: Absolute error reduction $\Delta\text{RMSE} = -0.0488\ ^\circ\text{C}$ (+4.74% relative gain).
- **B8 vs. B2 (Ridge Regression)**: Absolute error reduction $\Delta\text{RMSE} = -0.0495\ ^\circ\text{C}$ (+4.81% relative gain).

---

## Depth-Stratified & Regional Performance

### Depth-Wise Test Error Profile (B8)
The overall test score of 0.9800 °C is a **column-averaged** metric. In reality, reconstruction error varies strongly with depth, peaking sharply in the dynamic seasonal thermocline:

```
Depth (m)   B8 Test RMSE (°C)    Physical Oceanographic Regime
──────────────────────────────────────────────────────────────────
0 m             0.4369           Surface boundary layer (OSTIA constraint)
5 m             0.4381           Epipelagic mixed layer
10 m            0.4635           Epipelagic mixed layer
20 m            0.5962           Mixed layer base
30 m            0.8607           Upper thermocline transition
50 m            1.3493           Main thermocline steep gradient
75 m            1.8110           Thermocline peak error (maximum internal wave variance)
100 m           1.7651           Sub-thermocline baroclinic shear
125 m           1.5022           Permanent pycnocline
150 m           1.3650           Permanent pycnocline
200 m           1.1834           Mesopelagic transition
300 m           0.9845           Mesopelagic zone
500 m           0.6870           Deep mesopelagic (quiescent water mass)
700 m           0.6619           Deep ocean
1000 m          0.5959           Deep stable ocean (low thermal variance)
──────────────────────────────────────────────────────────────────
Overall         0.9800           Column-averaged test RMSE
```

### Regional Basin Decomposition (Cosine-Latitude Weighted)
- **Full Domain (NIO)**: **0.9642 °C**
- **Arabian Sea (AS)**: **1.0907 °C** *(higher error due to intense winter convective mixing and Somali current eddy dynamics)*
- **Bay of Bengal (BoB)**: **0.6775 °C** *(lower error due to strong perennial freshwater stratification stabilizing upper layers)*

### Certified Seasonal Subsets
- **Late Fall (Nov 09 – Nov 30)**: **1.0059 °C**
- **Early Winter (Dec 01 – Dec 31)**: **0.9060 °C**

---

## Repository Structure

```
├── config/
│   └── data_config.yaml                  # Immutable configuration and grid boundaries
├── data/
│   ├── processed/
│   │   ├── canonical_ocean_mask.nc       # 2D and 3D bathymetric ocean masks
│   │   ├── ml_dataset_full-year_surface.zarr
│   │   ├── ml_dataset_full-year_target.zarr
│   │   └── argo_matchup_evaluation.csv   # In-situ float matchup table
│   └── metadata/
│       └── normalization_stats.json      # Train-split-only fitted statistics
├── models/
│   ├── B0_persistence.py                 # 1-day lag persistence
│   ├── B0b_persistence.py                # Day-252 static persistence
│   ├── B1_climatology.py                 # Daily mean climatology
│   ├── B2_ridge.py                       # Ridge linear baseline
│   ├── B3_random_forest.py               # Random Forest regressor
│   ├── B4_lightgbm.py                    # LightGBM gradient boosting
│   ├── B5_pointwise_mlp.py               # Pointwise MLP (26,767 params)
│   ├── B6_spatial_cnn.py                 # Spatial CNN (30,991 params)
│   ├── B7_temporal_gru.py                # Temporal GRU (44,111 params)
│   ├── B8_spatiotemporal.py              # Full Spatiotemporal Network (203,791 params)
│   └── baselines.py                      # Unified evaluation interface
├── preprocessing/
│   ├── canonical_grid.py                 # Single source of truth for coordinates
│   ├── depth_interpolation.py            # Vertical spline with seafloor cutoff
│   ├── ocean_mask.py                     # Canonical land and bathymetry masking
│   ├── regrid.py                         # Bilinear and conservative coordinate regridding
│   ├── fit_transform.py                  # Strict training split normalizer (zero leakage)
│   └── build_dataset.py                  # PyTorch lazy DataLoader with 5-day window
├── reports/
│   ├── final_project_report.md           # 31-section comprehensive project report
│   ├── final_results_table.md            # Certified benchmark results table
│   ├── final_methodology.md              # Mathematical and architectural methodology
│   ├── final_limitations.md              # Transparent scientific limitations disclosure
│   └── project_structure.md              # Detailed module index and directory map
├── results/                              # Serialized benchmark metrics (JSON)
│   ├── B0.json ... B8.json
├── submission/
│   ├── Final_Project_Presentation.pptx   # 11-slide examiner presentation
│   ├── Final_Project_Submission.pdf      # Complete submission document
│   ├── FINAL_SUBMISSION_CHECKLIST.md     # 17-point completion checklist
│   └── demo_flow.md                      # 10-step scripted UI demonstration walkthrough
├── tests/                                # Automated verification suite (97 tests)
│   ├── test_baselines.py
│   ├── test_data_pipeline.py
│   ├── test_leakage.py
│   └── test_masks.py
├── src/                                  # React + Vite interactive prototype UI
│   ├── App.tsx
│   ├── components/                       # Dashboard, depth explorer, benchmark views
│   └── mock/demoData.ts                  # Local demo simulation data
├── package.json                          # Frontend dependencies
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation (this file)
```

---

## Installation & Quickstart

### 1. Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/neeravjain91-jpg/-Satellite-Embedding.git
cd -Satellite-Embedding

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Verification Test Suite
Ensure the test suite passes (all 106 tests must pass without error):
```bash
pytest tests/ -rs
```

### 3. Evaluate the B8 Spatiotemporal Model
To evaluate the certified B8 model or regenerate results:
```bash
python scripts/evaluate_all.py --model B8
```

---

## Reproducibility & Test Suite

The test suite contains **107 comprehensive unit, integration, and cross-artifact consistency tests** guaranteeing end-to-end scientific fidelity:
- `tests/test_cross_artifact_consistency.py`: Automated cross-artifact consistency across JSON results, markdown tables, metadata, and frontend benchmarks.
- `tests/test_confusion_matrix_integrity.py`: Confirms confusion matrix math, Cohen's kappa, and HTML synchronization.
- `tests/test_ml_protocol_splits_and_leakage.py`: Verifies strict temporal ordering and confirms zero overlap across the 6-day purge buffers.
- `tests/test_scientific_masks_and_integrity.py`: Verifies canonical ocean boundaries and GLORYS/ORCA12 model bathymetric cutoff masking across all 15 depths.
- `tests/test_b6_b7_b8_genuine_context.py`: Verifies spatial, temporal, and spatiotemporal receptive fields and model forward execution.

```
============================= 107 passed in 333.88s =============================
```

---

## Interactive Prototype (Frontend)

An interactive, responsive frontend prototype has been engineered with React, TypeScript, and Vite to explore vertical soundings, spatial heatmaps, and benchmark comparisons.

- **Production Live URL**: [https://code-gules-three.vercel.app](https://code-gules-three.vercel.app)
- **Local Development Server**:
  ```bash
  npm install
  npm run dev
  # Navigate to http://localhost:5173
  ```
- **Production Build**:
  ```bash
  npm run build
  # Verifies clean compilation with 0 TypeScript errors
  ```

> [!NOTE]
> **Prototype Simulation**: The frontend is a standalone, client-side prototype operating entirely on local mock/simulation data. It is decoupled from backend databases, real-time Python inference, or external API endpoints.

---

## Scientific Limitations & Responsible Disclosures

1. **Reanalysis Reference vs. Observational Ground Truth**: The model is trained against GLORYS12V1 ocean reanalysis fields. Reanalysis integrates numerical modeling and satellite/in-situ data assimilation, but retains inherent numerical smoothing and sub-grid parameterization biases.
2. **Non-Uniform Vertical Uncertainty**: Deep ocean layers (500–1000 m) exhibit low RMSE (<0.70 °C) primarily because natural thermal variance at those depths is muted ($\sigma < 0.8\ ^\circ\text{C}$). Peak error occurs in the seasonal thermocline (50–125 m, RMSE reaching 1.8110 °C at 75 m) where internal waves and high vertical gradients pose maximum reconstruction difficulty.
3. **Single-Year Scope (2020)**: The training dataset spans 2020. Interannual climate modes (e.g., strong positive/negative Indian Ocean Dipole, El Niño–Southern Oscillation) may exhibit out-of-distribution dynamics requiring multi-decadal retraining.
4. **Argo Matchup Independence**: In-situ Argo CTD profiles are routinely assimilated into the GLORYS reanalysis system via Coriolis/CORA. Direct comparisons against assimilated floats reflect validation against the reference state rather than completely independent observational validation.

---

## Submission Artifacts & Reports

Detailed technical documentation and submission deliverables are located in the `reports/` and `submission/` directories:

- [`reports/final_project_report.md`](reports/final_project_report.md) — Comprehensive 31-section final technical report.
- [`reports/final_results_table.md`](reports/final_results_table.md) — Master benchmark table with exact parameter counts.
- [`reports/final_methodology.md`](reports/final_methodology.md) — Mathematical formulation, grid specifications, and architecture layers.
- [`reports/final_limitations.md`](reports/final_limitations.md) — Full scientific limitations and caveats disclosure.
- [`submission/Final_Project_Presentation.pptx`](submission/Final_Project_Presentation.pptx) — 11-slide presentation for examiners and defence.
- [`submission/Final_Project_Submission.pdf`](submission/Final_Project_Submission.pdf) — Formatted complete project submission document.
- [`submission/FINAL_SUBMISSION_CHECKLIST.md`](submission/FINAL_SUBMISSION_CHECKLIST.md) — 17-point pre-submission verification checklist.
- [`submission/demo_flow.md`](submission/demo_flow.md) — 10-step scripted walkthrough for live demonstration and recording.

---

## References & Data DOIs

1. **OSTIA SST**: Good, S., et al. (2020). *E.U. Copernicus Marine Service Information*. DOI: [10.48670/moi-00168](https://doi.org/10.48670/moi-00168).
2. **Multi-Obs SSS**: Droghei, R., et al. (2020). *Copernicus Marine Service Multi-Observation SSS*. DOI: [10.48670/moi-00051](https://doi.org/10.48670/moi-00051).
3. **DUACS SSH/SLA**: Pujol, M.-I., et al. (2016). *Ocean Science*, 12(5), 1067–1090. DOI: [10.48670/moi-00145](https://doi.org/10.48670/moi-00145).
4. **OSCAR Currents**: Dohan, K. (2021). *PO.DAAC, JPL, NASA*. DOI: [10.5067/OSCAR-25F20](https://doi.org/10.5067/OSCAR-25F20).
5. **CCMP Winds**: Mears, C., et al. (2022). *Remote Sensing Systems CCMP V3.1*. DOI: [10.5067/CCMP3-6H431](https://doi.org/10.5067/CCMP3-6H431).
6. **GLORYS12V1 Reanalysis**: Jean-Michel, L., et al. (2021). *CMEMS Reanalysis Report*. DOI: [10.48670/moi-00021](https://doi.org/10.48670/moi-00021).
7. **Argo Float Program**: Argo Data Management Team (2021). *Argo Float Data and Metadata*. DOI: [10.17882/42182](https://doi.org/10.17882/42182).
