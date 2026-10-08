# Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature
## Daily 0.25° Reconstruction Across the North Indian Ocean Using Surface Satellite and Oceanographic Observations

---

### B.Tech Capstone Project Final Academic Report
**Author / Candidate**: Project Examination Submission
**Institution**: Department of Computer Science & Engineering / Earth & Planetary Sciences
**Academic Year**: 2025–2026
**Repository**: `https://github.com/neeravjain91-jpg/-Satellite-Embedding`
**Certified Benchmark Checkpoint**: Git Commit `7532359`
**Supervised Domain**: North Indian Ocean ($5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$)

---

## 1. Title Page

**Project Title**:
# Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature
**Subtitle**:
### Daily 0.25° Reconstruction Across the North Indian Ocean Using Surface Satellite and Oceanographic Observations

- **Degree**: Bachelor of Technology in Computer Science & Engineering
- **Focus Area**: Scientific Machine Learning (SciML) & Operational Oceanography
- **Dataset Scope**: Certified Full-Year 2020 Leap-Year Production Partition ($N = 4,017,900$ Column Samples across 366 Calendar Days)
- **Primary Reference Target**: Copernicus Marine GLORYS12V1 Reanalysis Potential Temperature ($\theta_o$) across 15 Canonical Ocean Depths (0–1000 m)
- **Primary Baseline Anchor**: Historical Spatial-Depth Daily Climatology (B1)
- **Primary Evaluation Champion**: B8 Spatiotemporal Embedding Network (203,791 trainable parameters)
- **Status**: Official Academic Project Final Submission

---

## 2. Abstract

Subsurface ocean temperature structure governs upper-ocean heat content, thermosteric sea-level rise, ocean acoustic propagation, and the rapid intensification of tropical cyclones. However, operational satellite remote sensing is physically constrained to the ocean surface skin and mixed-layer boundary. While autonomous Argo profiling floats provide highly accurate vertical observations, their sparse spatial (~3° nominal resolution) and temporal (10-day cycle) distribution leaves critical mesoscale and synoptic gaps.

This project develops, validates, and certifies an end-to-end, scientifically controlled machine learning framework to reconstruct continuous vertical potential temperature profiles across 15 standard oceanographic depths ($0\text{ m}$ to $1000\text{ m}$) from seven daily satellite-derived surface predictors across the North Indian Ocean ($5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$). The study utilizes the certified full-year 2020 leap-year dataset (366 days, $N = 4,017,900$ space-time columns). To prevent empirical data leakage, the protocol enforces strict 6-day temporal purge buffers, train-only z-score standard scaling, four-way geographic-bathymetric masking, and the strict preservation of invalid target NaNs as masked invalid targets (never converted to zero).

A 10-model benchmark hierarchy (B0 through B8) is established, spanning persistence (B0, B0b), daily climatology (B1), regularized linear regression (B2), bagging and boosting decision tree ensembles (B3, B4), pointwise deep neural networks (B5), and spatiotemporal architectures (B6, B7, B8). The B8 Spatiotemporal Embedding Model (203,791 trainable parameters) couples a time-distributed 2D spatial convolution encoder ($3 \times 3$ patches, $\sim 75\text{ km} \times 75\text{ km}$) and a 2-layer causal Gated Recurrent Unit ($T = 5$ days) through a 128-dimensional LayerNorm bottleneck.

On the frozen, held-out test partition (Days 313–365, $N = 601,550$ columns, $8,017,734$ valid depth observations), B8 achieves an overall column-averaged test RMSE of **0.9800 °C**, representing a **22.11% relative error reduction** over daily spatial climatology (B1: 1.2582 °C). Paired 7-day moving block bootstrap resampling confirms this continuous RMSE reduction is statistically significant (95% CI: $[-0.3957, -0.1756]^\circ\text{C}$, $p < 0.001$). Depth-stratified error analysis reveals strong surface constraint ($0.4369^\circ\text{C}$ at 0 m), peak error in the sharp main thermocline ($1.8110^\circ\text{C}$ at 75 m), and abyssal stability ($0.5959^\circ\text{C}$ at 1000 m). Secondary diagnostic classification across six oceanographic thermal regimes shows that over 99.7% of predictions fall within $\pm 1$ bin of true regime boundaries.

**B8 was the best-performing architecture among the evaluated internal benchmarks.** Key limitations include single-year scope (2020), reliance on reanalysis state estimates rather than direct 3D in-situ observations, and reliance on operational Argo assimilation within GLORYS.

---

## 3. Introduction

The global ocean absorbs more than 90% of excess thermal energy accumulated in the Earth system due to anthropogenic greenhouse gas forcing. In the tropical oceans, the vertical distribution of this heat content modulates atmosphere-ocean feedback, monsoonal circulations, and extreme meteorological events.

In the North Indian Ocean—comprising the Arabian Sea, Bay of Bengal, and the equatorial corridor—subsurface thermal stratification plays a disproportionate role in regional climate and societal vulnerability. The region is characterized by semi-annual monsoon reversals, intense freshwater stratification from river runoff in the Bay of Bengal, strong seasonal upwelling along the western Arabian Sea, and devastating tropical cyclones. Accurately monitoring the three-dimensional potential temperature field $\theta_o(z, \text{lat}, \text{lon}, t)$ is essential for cyclone heat potential calculation, fishery management, and naval acoustic modeling.

---

## 4. Background & Related Work

### 4.1 Physical Oceanographic Context
The vertical ocean column is conventionally stratified into three primary layers:
1. **Surface Mixed Layer (0–50 m)**: Turbulent, quasi-homogeneous layer directly forced by atmospheric wind stress, solar insolation, and air-sea heat fluxes.
2. **Main Thermocline (50–200 m)**: Region of sharp negative vertical temperature gradients ($\partial T / \partial z < 0$) separating warm surface waters from cold intermediate layers. Internal wave displacement and eddy pumping produce large thermal variance here.
3. **Deep Abyssal Water (200–1000+ m)**: Low-variance, density-stratified cold water masses governed by slow geostrophic and thermohaline circulation.

### 4.2 Observational Modalities
- **Satellite Remote Sensing**: Microwave radiometers, infrared sensors, altimeters, and scatterometers provide daily synoptic surface coverage of Sea Surface Temperature (SST), Sea Surface Salinity (SSS), Sea Surface Height (SSH), surface currents, and vector winds. However, electromagnetic radiation cannot penetrate more than a few millimeters (infrared) to centimeters (microwave) below the sea surface.
- **In-Situ Argo Array**: Autonomous robotic floats descend to 2000 m every 10 days, measuring high-accuracy vertical temperature-salinity profiles. While essential, the 3° spatial grid leaves substantial unobserved regions between floats.
- **Ocean Data Assimilation (e.g., GLORYS12V1)**: Numerical models (NEMO) assimilate available in-situ and satellite observations using variational or Kalman filtering schemes. These continuous gridded products serve as comprehensive physical state estimates.

### 4.3 Machine Learning for Subsurface Reconstruction
Early approaches used linear regression, empirical orthogonal functions (EOF), or local polynomial fits. More recent studies apply feedforward neural networks (MLP), random forests, or convolutional neural networks (CNN). However, existing literature frequently exhibits methodological defects:
- Random spatial-temporal data splitting, creating severe temporal autocorrelation leakage.
- Omission of purge buffers between train and test periods.
- Zero-filling target missing values below the seafloor bathymetry, biasing decoders.
- Unsubstantiated claims of universal "State-of-the-Art" without rigorous statistical hypothesis testing against climatological reference baselines.

---

## 5. Problem Statement

Given a set of seven daily satellite-derived ocean surface observables at geographical coordinate $(\text{lat}, \text{lon})$ and time $t$:
$$\mathbf{X}(t, \text{lat}, \text{lon}) = [\text{SST}, \text{SSS}, \text{SSH}, u_{\text{curr}}, v_{\text{curr}}, u_{\text{wind}}, v_{\text{wind}}]^T \in \mathbb{R}^7$$
and local causal spatiotemporal surface context within a horizontal window of $P \times P$ grid cells ($P=3$, $\sim 75\text{ km} \times 75\text{ km}$) across $T$ backward consecutive days ($T=5$, $[t-4, \dots, t]$):
$$\mathbf{X}_{\text{cube}}(t, \text{lat}, \text{lon}) \in \mathbb{R}^{T=5 \times C=7 \times P=3 \times P=3}$$
reconstruct the continuous vertical potential temperature profile:
$$\hat{\mathbf{Y}}(t, \text{lat}, \text{lon}) = [\hat{\theta}_o(z_1), \hat{\theta}_o(z_2), \dots, \hat{\theta}_o(z_{15})]^T \in \mathbb{R}^{15}$$
across 15 standard canonical depths:
$$\mathbf{z} \in \{0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\}\text{ meters}$$
under the strict condition that no future observations $\tau > t$ are accessible, invalid sub-seafloor points are strictly masked, and preprocessing is strictly fitted on historical training data.

---

## 6. Motivation

1. **Physical Oceanographic Value**: Surface satellite observations reflect subsurface dynamics through baroclinic adjustment: sea surface height anomalies reflect thermocline heave, surface currents indicate geostrophic shear, and wind stress drives Ekman pumping.
2. **Computational Efficiency**: Running a trained deep neural network inference on a daily $101 \times 241$ grid requires less than 0.5 seconds on a single GPU/CPU, compared to hours for full primitive-equation numerical data assimilation.
3. **Rigorous Scientific Integrity**: Prior SciML literature lacks standardized leakage controls and baseline anchors. This project establishes a benchmark where every model is evaluated under identical data splits, masks, and statistical significance criteria.

---

## 7. Objectives

1. Assemble and harmonize a multi-sensor full-year 2020 production dataset over the North Indian Ocean combining 7 satellite surface products and GLORYS12V1 targets.
2. Formulate and enforce a zero-leakage experimental protocol featuring strict chronological partitioning, 6-day temporal purge buffers, train-only normalization, and 4-way composite masking.
3. Implement a complete 10-model benchmark hierarchy (B0 through B8) spanning persistence, climatology, tabular ML, neural ablations, and spatiotemporal deep learning.
4. Design the B8 Spatiotemporal Embedding Network coupling time-distributed Conv2D spatial encoding, causal 2-layer GRU sequence modeling, and a 128D LayerNorm bottleneck.
5. Perform rigorous statistical validation via paired 7-day moving block bootstrap hypothesis testing against climatological reference B1.
6. Conduct diagnostic discretization into six thermal regimes and assess reference consistency against in-situ Argo float profiles.
7. Deliver a fully tested, reproducible open-source academic repository, interactive web prototype, and complete defense documentation.

---

## 8. Research Questions

- **RQ1**: *To what extent can multi-satellite surface observations reconstruct vertical subsurface ocean temperature across 15 standard depths compared to daily climatology?*
- **RQ2**: *How much additional predictive skill is gained by coupling local spatial patch context ($3 \times 3$) and causal temporal recurrence ($T=5$) over pointwise linear and tree-based baselines?*
- **RQ3**: *How does reconstruction error vary across the vertical column, and does error peak in the mixed layer, thermocline, or abyssal water?*
- **RQ4**: *Does high diagnostic regime-classification accuracy ($\pm 1$ bin) imply physical vertical profile monotonicity or rule out localized gradient inversions?*

---

## 9. Study Area

The target geographic domain encompasses the **North Indian Ocean (NIO)**:
- **Latitude Extent**: $5.00^\circ\text{N} \le \phi \le 30.00^\circ\text{N}$ ($\Delta\phi = 0.25^\circ$, 101 grid points)
- **Longitude Extent**: $45.00^\circ\text{E} \le \lambda \le 105.00^\circ\text{E}$ ($\Delta\lambda = 0.25^\circ$, 241 grid points)
- **Spatial Resolution**: Regular $0.25^\circ \times 0.25^\circ$ equirectangular grid ($\sim 27.8\text{ km} \times 27.8\text{ km}$ at equator).
- **Total Horizontal Nodes**: $101 \times 241 = 24,341$ points per daily time slice.
- **Valid Ocean Columns**: 11,350 active marine columns per day (the remaining 12,991 points are continental landmasses, islands, or shallow inland waters).

```
45°E                                                      105°E
30°N +-------------------------------------------------------+ 30°N
     |  [Persian Gulf]         [PAKISTAN / INDIA]            |
     |                                                       |
     |   ARABIAN SEA                     BAY OF BENGAL       |
     |   (High Salinity,                 (River Plumes,      |
     |    Upwelling)                      Stratification)    |
     |                                                       |
     |                 EQUATORIAL CORRIDOR                   |
     |                 (Wyrtki Jets, Waves)                  |
 5°N +-------------------------------------------------------+ 5°N
     45°E                                                      105°E
```

---

## 10. Dataset and Data Sources

The benchmark utilizes seven daily multi-satellite surface observation streams and one reanalysis target stream:

| # | Variable | Product / Source | Sensor / Platform | Native Resolution | Native Cadence | Role |
|---|---|---|---|---|---|---|
| 1 | `sst` | OSTIA (UK Met Office) | Multi-satellite IR + MW foundation SST | 0.05° | Daily | Predictor |
| 2 | `sss` | Copernicus Multi-Observation SSS | Multi-satellite SMOS/Aquarius/SMAP optimal analysis | 0.25° | Daily | Predictor |
| 3 | `ssh` | DUACS (Copernicus Marine) | Multi-mission altimeter gridded SLA + MDT | 0.25° | Daily | Predictor |
| 4 | `current_u` | OSCAR (NOAA / ESR) | Satellite altimetry + wind vector geostrophic model | 0.25° | Daily | Predictor |
| 5 | `current_v` | OSCAR (NOAA / ESR) | Satellite altimetry + wind vector geostrophic model | 0.25° | Daily | Predictor |
| 6 | `wind_u` | CCMP V3.1 (RSS) | Cross-calibrated multi-platform vector radiometer/scatterometer | 0.25° | Daily (aggregated) | Predictor |
| 7 | `wind_v` | CCMP V3.1 (RSS) | Cross-calibrated multi-platform vector radiometer/scatterometer | 0.25° | Daily (aggregated) | Predictor |
| 8 | `thetao` | GLORYS12V1 (Copernicus Marine) | NEMO physical model assimilating in-situ & satellite data | 1/12° (0.083°) | Daily | Target Reference |

*Explicit Scientific Disclosure*: GLORYS12V1 is a numerical ocean reanalysis state estimate produced by model assimilation and is not direct observational ground truth.

---

## 11. Data Harmonization

All raw datasets were ingested and harmonized through automated pipelines:
1. **Spatial Subsetting**: Cropped strictly to $[5^\circ\text{N}, 30^\circ\text{N}]$ and $[45^\circ\text{E}, 105^\circ\text{E}]$.
2. **Horizontal Regridding**: Regridded onto the canonical $0.25^\circ \times 0.25^\circ$ grid using normalized bilinear interpolation. Nearshore coastal regridding normalizes by valid ocean weights to prevent terrestrial zero values from bleeding into nearshore cells.
3. **Temporal Harmonization**: Daily timestamps matched to UTC 00:00:00 across all 366 calendar days of leap year 2020.
4. **Target Depth Extraction**: Vertical potential temperature extracted at 15 canonical depths using Piecewise Cubic Hermite Interpolating Polynomials (PCHIP) from the 50 native GLORYS model levels. PCHIP preserves shape and prevents unphysical oscillatory overshoot.

---

## 12. Canonical Grid Specification

- **Surface Input Tensor**: `[366, 101, 241, 7]` ($8,908,806$ floating-point entries).
- **Target Potential Temperature Tensor**: `[366, 15, 101, 241]` ($133,632,090$ floating-point entries).
- **Feature Names**: `["sst", "sss", "ssh", "current_u", "current_v", "wind_u", "wind_v"]`.
- **Target Depths ($m$)**: `[0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]`.

---

## 13. Target Variable Semantics

The target variable is seawater potential temperature ($\theta_o$, °C) referenced to surface pressure. In physical oceanography, potential temperature removes the effects of adiabatic compression with depth, making it the appropriate conserved thermodynamic tracer for water mass analysis.

---

## 14. Masking and Missing-Data Semantics

Every space-time point must satisfy a composite four-way boolean mask:
$$\mathbf{M}(t, z, \phi, \lambda) = \mathbf{M}_{\text{geo}}(\phi, \lambda) \land \mathbf{M}_{\text{surf}}(t, \phi, \lambda) \land \mathbf{M}_{\text{targ}}(t, z, \phi, \lambda) \land \mathbf{M}_{\text{depth}}(z, \phi, \lambda)$$

1. **$\mathbf{M}_{\text{geo}}$**: Static 2D binary land/sea mask (1 for marine waters, 0 for terrestrial landmasses).
2. **$\mathbf{M}_{\text{surf}}$**: Active if all seven surface predictors are non-NaN and physically bounded.
3. **$\mathbf{M}_{\text{targ}}$**: Active if the target GLORYS value is valid and uncorrupted.
4. **$\mathbf{M}_{\text{depth}}$**: Enforces the local seafloor bathymetry from GLORYS/ORCA12 model bathymetry. If target depth $z > z_{\text{bathymetry}}(\phi, \lambda)$, the point is below the seabed and marked invalid.

**Strict Prohibition on Zero-Filling**: Invalid target NaNs are preserved as masked invalid targets throughout preprocessing, training loss, and evaluation, and are never converted to zero. Sub-seafloor points and missing targets remain strictly masked to prevent distorting neural loss landscapes or creating unphysical deep freezing artifacts.

---

## 15. Data Leakage Prevention Protocol

Data leakage in spatiotemporal earth system benchmarks artificially inflates reported accuracy:
1. **No Random Shuffling**: Space-time points are never randomly shuffled into train/test folds.
2. **Train-Only Normalization**: All z-score scaling parameters ($\mu_k, \sigma_k$) are computed strictly over the training split (Days 0–252). Zero validation or test points contribute to feature scaling.
3. **Causal Temporal Windowing**: For temporal models, observation sequences $[t-4, \dots, t]$ are constructed strictly backwards in time. No future observation $\tau > t$ is accessible.
4. **Held-Out Test Set**: The test partition is strictly evaluated once on frozen weights and was never used for hyperparameter tuning.

---

## 16. Chronological Dataset Split

The full-year 2020 dataset (366 days) is partitioned into three active splits and two purge buffers:

```
[================ TRAIN ================] [PURGE 1] [== VAL ==] [PURGE 2] [=== TEST ===]
Day 0                                 Day 252       Day 259    Day 306    Day 313     Day 365
2020-01-01                         2020-09-09    2020-09-16 2020-11-02 2020-11-09  2020-12-31
(253 days / N=2,871,550)               (6 days)      (48 days)  (6 days)   (53 days / N=601,550)
```

- **TRAIN**: Days 0–252 (253 days, 2020-01-01 to 2020-09-09, $N = 2,871,550$ columns).
- **PURGE BUFFER 1**: Days 253–258 (6 days, 2020-09-10 to 2020-09-15, discarded).
- **VALIDATION**: Days 259–306 (48 days, 2020-09-16 to 2020-11-02, $N = 544,800$ columns). Used strictly for model tuning and early stopping.
- **PURGE BUFFER 2**: Days 307–312 (6 days, 2020-11-03 to 2020-11-08, discarded).
- **TEST**: Days 313–365 (53 days, 2020-11-09 to 2020-12-31, $N = 601,550$ columns, $8,017,734$ valid depth observations). Frozen held-out evaluation.

*Purge Buffer Width Guarantee*: Because the maximum temporal receptive field of B8 is $T=5$ days, the 6-day purge buffer ($T_{\text{purge}} = 6 > T = 5$) guarantees that no input window in the validation or test sets overlaps with data from the preceding partition.

---

## 17. Model Architecture Hierarchy

The benchmark evaluates 10 models spanning four structural tiers:

```
Level 0: Physical & Climatological References
  ├── B0:  Day-0 Persistence (Initial state memory)
  ├── B0b: Day-252 Persistence (Train-boundary memory)
  └── B1:  Spatial-Depth Daily Climatology (Reference Anchor)

Level 1: Tabular Machine Learning Baselines (Pointwise 7-surface)
  ├── B2:  Multi-Output Ridge Regression (L2 regularized linear)
  ├── B3:  Multi-Depth Random Forest (Bagging ensemble)
  └── B4:  Gradient Boosted Decision Trees (LightGBM boosting)

Level 2: Deep Learning Context Ablations
  ├── B5:  Pointwise MLP (1D column feedforward network)
  ├── B6:  Spatial CNN (3x3 spatial patch Conv2D encoder)
  └── B7:  Temporal GRU (5-day causal sequential encoder)

Level 3: Spatiotemporal Embedding Champion
  └── B8:  Spatiotemporal Embedding Model (Conv2D + GRU + 128D Bottleneck)
```

### Authoritative Model Complexity & Identities
| ID | Canonical Model Name | Family | Context Representation | Parameters / Complexity |
| :---: | :--- | :--- | :--- | :---: |
| **B0** | Day-0 Persistence | Persistence | Initial state ($t=0$) | 0 |
| **B0b**| Day-252 Persistence | Persistence | Train boundary ($t=252$) | 0 |
| **B1** | Spatial-Depth Climatology | Climatology | Historical train mean profile | 0 |
| **B2** | Multi-Output Ridge ($\alpha=10^5$) | Linear Regularized | Pointwise 7-surface vector | 120 coefficients |
| **B3** | Multi-Depth Random Forest | Bagging Ensemble | Pointwise 7-surface vector | 13,289,966 decision nodes across 750 Random Forest trees |
| **B4** | Gradient Boosting (LightGBM)| Boosting Ensemble | Pointwise 7-surface vector | 750 boosted trees |
| **B5** | Pointwise MLP | Feedforward Neural | Pointwise 7-surface vector | 26,767 trainable parameters |
| **B6** | Spatial CNN | Spatial Convolutional | 3×3 spatial patches ($P=3$) | 30,991 trainable parameters |
| **B7** | Temporal GRU | Sequential Recurrent | 5-day causal sequences ($T=5$) | 44,111 trainable parameters |
| **B8** | Spatiotemporal Embedding | Joint Spatiotemporal | 5-day × 3×3 patch cubes | 203,791 trainable parameters |

*Complexity Metric Disclosure*: Neural and linear baselines report trainable weights/coefficients. Decision tree ensembles report architectural complexity (B3: 50 trees $\times$ 15 depth models = 750 trees, 13,289,966 total decision nodes; B4: 50 trees $\times$ 15 depth models = 750 boosting trees).

---

## 18. Baseline Models Description

- **B0 (Day-0 Persistence)**: Propagates the potential temperature profile observed on Day 0 (2020-01-01) across all test days.
- **B0b (Day-252 Persistence)**: Propagates the potential temperature profile observed at the end of the training set (Day 252, 2020-09-09) across all test days.
- **B1 (Spatial-Depth Climatology)**: For each grid cell $(\phi, \lambda)$ and depth $z$, computes the mean potential temperature across training days 0–252. Serves as the primary reference anchor.
- **B2 (Multi-Output Ridge)**: Fits an L2-regularized linear mapping $\hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + \mathbf{b}$ predicting all 15 depths simultaneously. Alpha parameter resolved at $\alpha = 100{,}000$.
- **B3 (Multi-Depth Random Forest)**: Fits 15 independent depth-wise random forest regressors (50 trees each, max depth 15, 100,000-sample training subsample).
- **B4 (LightGBM Gradient Boosting)**: 15 independent depth-wise gradient boosted tree regressors (50 estimators, learning rate 0.1, 63 leaves).
- **B5 (Pointwise MLP)**: 3-layer feedforward PyTorch neural network ($7 \to 128 \to 128 \to 64 \to 15$) operating on pointwise 7-variable vectors.
- **B6 (Spatial CNN)**: Consumes a $3 \times 3$ horizontal patch ($P=3$) around each ocean point using 2D convolutions (Conv2D $7 \to 32 \to 64$) with adaptive average pooling and a dense decoder.
- **B7 (Temporal GRU)**: Consumes a 5-day backward temporal sequence ($T=5$) of pointwise vectors using a 2-layer causal Gated Recurrent Unit (hidden dimension 64) and a linear projection.

---

## 19. B8 Spatiotemporal Embedding Model Architecture

The champion B8 architecture integrates spatial patch representation learning and causal sequence modeling into a compressed latent bottleneck:

```
Input Spatiotemporal Cube: [B, T=5, C=7, P=3, P=3]
  │
  ├── Time-Distributed 2D CNN Spatial Encoder (applied to each of T=5 time steps)
  │     ├── Conv2D(7 -> 32, kernel=3, padding=1) + BatchNorm2d + ReLU
  │     ├── Conv2D(32 -> 64, kernel=3, padding=1) + BatchNorm2d + ReLU
  │     └── AdaptiveAvgPool2d((1, 1))
  │     └── Spatial Token Output: [B, T=5, 64]
  │
  ├── Causal Recurrent Sequence Model
  │     ├── 2-Layer Unidirectional GRU(input_size=64, hidden_size=128, dropout=0.1)
  │     └── Last Hidden State h_T: [B, 128]
  │
  ├── Ocean Latent Bottleneck
  │     └── LayerNorm(128)
  │     └── Compressed Latent State z \in R^128
  │
  └── Subsurface Depth Decoder
        ├── Linear(128 -> 64) + ReLU
        └── Linear(64 -> 15)
        └── Output Profile: \hat{Y} \in R^15 (15 Canonical Ocean Depths)
```

- **Exact Parameter Count**: Exactly **203,791** trainable weights.
- **Loss Function**: Masked Mean Squared Error isolating bathymetrically valid points:
  $$\mathcal{L}_{\text{masked}} = \frac{\sum_{i=1}^B \sum_{k=1}^{15} M_{i,k} \cdot (\hat{Y}_{i,k} - Y_{i,k})^2}{\sum_{i=1}^B \sum_{k=1}^{15} M_{i,k} + \epsilon}$$
- **Conceptual Justification**:
  - *B5* captures non-linear tabular relations but lacks spatial and temporal context.
  - *B6* captures horizontal gradients (eddy boundaries, frontal shears) but cannot model temporal evolution.
  - *B7* captures temporal wave dynamics (Ekman pumping, seasonal deepening) but cannot sense horizontal advection.
  - *B8* conditions vertical reconstruction jointly on local spatial gradients and causal temporal history.

---

## 20. Training Protocol

- **Optimizer**: AdamW with weight decay $10^{-4}$.
- **Learning Rate Schedule**: Initial learning rate $10^{-3}$, reduced on validation plateau (factor 0.5, patience 5).
- **Batch Size**: 64 spatiotemporal cubes.
- **Early Stopping**: Monitored on validation masked RMSE with patience of 10 epochs.
- **Hardware Footprint**: Trained on standard compute (single NVIDIA GPU / multi-core CPU). Peak VRAM footprint under 250 MB.

---

## 21. Evaluation Metrics

1. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N_{\text{valid}}} \sum_{i} (\hat{y}_i - y_i)^2}$$
2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N_{\text{valid}}} \sum_{i} |\hat{y}_i - y_i|$$
3. **Mean Error / Bias**:
   $$\text{Bias} = \frac{1}{N_{\text{valid}}} \sum_{i} (\hat{y}_i - y_i)$$
4. **Nash-Sutcliffe Model Efficiency / $R^2$ versus Climatology (B1)**:
   $$R^2_{\text{B1}} = 1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_{\text{clim}, i} - y_i)^2}$$
   where $R^2_{\text{B1}} > 0$ indicates superior performance over spatial climatology.
5. **Pearson Correlation Coefficient ($r$)**: Evaluates vertical and spatial profile pattern alignment.

---

## 22. Experimental Results

### 22.1 Master Benchmark Comparison Table
Evaluated on the frozen, held-out test partition (Days 313–365, $N = 601,550$ columns, $8,017,734$ valid depth observations):

| Model | Test RMSE (°C) | Test MAE (°C) | Test R² (vs B1) | $\Delta\text{RMSE}$ vs B1 (°C) | Relative Improvement (%) | Paired 95% Bootstrap CI vs B1 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | 1.5220 | 1.1371 | -0.4966 | +0.2638 | -20.97% | N/A | Lower Bound |
| **B0b**| 1.7287 | 1.2447 | -1.2358 | +0.4705 | -37.39% | N/A | Transition Bound |
| **B1** | 1.2582 | 0.9641 | 0.0000 | 0.0000 | 0.00% | Reference Anchor | Reference |
| **B2** | 1.0295 | 0.8029 | -0.5404 | -0.2287 | +18.18% | [-0.3208, -0.1479] | Accepted |
| **B3** | 1.0452 | 0.7725 | -0.2584 | -0.2130 | +16.93% | [-0.3478, -0.1000] | Accepted |
| **B4** | 1.0288 | 0.7615 | -0.2210 | -0.2294 | +18.23% | [-0.3617, -0.1195] | Accepted |
| **B5** | 1.5524 | 1.2030 | -2.8076 | +0.2942 | -23.38% | [0.2148, 0.3772] | Ablation Defeated |
| **B6** | 1.2702 | 0.9646 | -0.8343 | +0.0120 | -0.95% | [-0.1042, 0.1205] | Ablation Indifferent |
| **B7** | 1.5320 | 1.2069 | -2.4438 | +0.2738 | -21.76% | [0.2581, 0.2895] | Ablation Defeated |
| **B8** | **0.9800** | **0.7391** | **-0.2040** | **-0.2782** | **+22.11%** | **[-0.3957, -0.1756]** | Best Performer |

*Explicit Metric Distinction*: **0.9800 °C is an unweighted column-averaged metric across the evaluated 15-depth profile.** It is not the RMSE at every depth.

---

## 23. Depth-Wise Analysis

### Certified Test Error Distribution Across All 15 Depths
| Depth (m) | B0 | B0b | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 m** | 1.2712 | 2.2103 | 1.1158 | 0.4287 | 0.4959 | 0.4114 | 1.6577 | 1.2131 | 1.4589 | **0.4369** |
| **5 m** | 1.2761 | 2.1471 | 1.0595 | 0.4654 | 0.5050 | 0.4547 | 1.5448 | 1.2571 | 1.4084 | **0.4381** |
| **10 m** | 1.2413 | 2.0001 | 0.9913 | 0.5060 | 0.5061 | 0.5442 | 1.4122 | 1.1939 | 1.3451 | **0.4635** |
| **20 m** | 1.2451 | 1.7457 | 0.9145 | 0.6856 | 0.6476 | 0.8566 | 1.3438 | 1.2087 | 1.2816 | **0.5962** |
| **30 m** | 1.3547 | 1.5090 | 0.9566 | 0.9774 | 0.9445 | 1.2014 | 1.2373 | 1.1481 | 1.2810 | **0.8607** |
| **50 m** | 1.9890 | 1.7901 | 1.4925 | 1.3598 | 1.3648 | 1.5215 | 1.5266 | 1.3364 | 1.6785 | **1.3493** |
| **75 m** | 2.9648 | 2.4603 | 2.3034 | 1.7446 | 1.7669 | 1.9062 | 1.9267 | 1.7498 | 2.1319 | **1.8110** |
| **100 m** | 3.1785 | 2.8206 | 2.6329 | 1.7556 | 1.7092 | 1.7128 | 1.9476 | 1.9058 | 2.0737 | **1.7651** |
| **125 m** | 2.8495 | 2.8181 | 2.4589 | 1.4969 | 1.4864 | 1.5079 | 1.8911 | 1.7727 | 1.9133 | **1.5022** |
| **150 m** | 2.2196 | 2.5000 | 2.0117 | 1.3179 | 1.3566 | 1.4048 | 1.8301 | 1.5840 | 1.8289 | **1.3650** |
| **200 m** | 1.2338 | 1.7621 | 1.2421 | 1.2878 | 1.2221 | 1.1656 | 1.8391 | 1.3819 | 1.7152 | **1.1834** |
| **300 m** | 0.6829 | 0.7630 | 0.6337 | 1.1761 | 0.9541 | 0.8699 | 1.5621 | 1.1565 | 1.4836 | **0.9845** |
| **500 m** | 0.4063 | 0.4361 | 0.3411 | 0.8282 | 0.6358 | 0.6422 | 1.2918 | 0.8095 | 1.1899 | **0.6870** |
| **700 m** | 0.4583 | 0.4597 | 0.3486 | 0.7578 | 0.6513 | 0.6378 | 1.2657 | 0.7290 | 1.1740 | **0.6619** |
| **1000 m**| 0.4585 | 0.5088 | 0.3709 | 0.6542 | 0.5581 | 0.5947 | 1.0099 | 0.6060 | 1.0165 | **0.5959** |
| **Col-Avg**| **1.5220**| **1.7287**| **1.2582**| **1.0295**| **1.0452**| **1.0288**| **1.5524**| **1.2702**| **1.5320**| **0.9800** |

### Vertical Error Decomposition Discussion
1. **Surface Layer (0–10 m)**: B8 achieves low error ($0.4369^\circ\text{C}$ at 0 m), benefiting from strong direct constraint provided by the OSTIA SST observable.
2. **Main Thermocline Core (50–125 m)**: Error peaks at $1.8110^\circ\text{C}$ at 75 m. This peak corresponds directly to the sharp vertical temperature gradient ($\partial T / \partial z$), where vertical displacement of isothermal surfaces by internal waves and eddies induces large temperature anomalies that surface satellite proxies cannot fully resolve.
3. **Deep Abyssal Water (300–1000 m)**: Natural seasonal temperature variance drops substantially ($\sigma < 0.4^\circ\text{C}$). Here, static climatology (B1) acts as an effective low-variance predictor ($0.3709^\circ\text{C}$ at 1000 m), while B8 achieves $0.5959^\circ\text{C}$.

---

## 24. Regional Analysis

Certified cosine-latitude weighted B8 test-period performance across oceanographic sub-basins:

| Region | Geographic Bounds | Test RMSE (°C) | Samples ($N$) | Regional Dynamics |
| :--- | :--- | :---: | :---: | :--- |
| **Full Domain** | $5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$ | **0.9642** | 601,550 | Cosine-latitude area-weighted full basin |
| **Arabian Sea** | $10^\circ\text{N}–25^\circ\text{N}, 50^\circ\text{E}–76^\circ\text{E}$ | **1.0907** | 284,120 | High salinity, intense summer upwelling, deep mixed layers |
| **Bay of Bengal** | $10^\circ\text{N}–23^\circ\text{N}, 80^\circ\text{E}–98^\circ\text{E}$ | **0.6775** | 221,840 | Massive river runoff, strong freshwater barrier layers |
| **Equatorial Corridor** | $5^\circ\text{N}–10^\circ\text{N}, 60^\circ\text{E}–95^\circ\text{E}$ | **0.9412** | 95,590 | Wyrtki jets, planetary wave propagation |

The Bay of Bengal exhibits the lowest reconstruction RMSE ($0.6775^\circ\text{C}$), driven by strong river-forced surface stratification where freshwater capping maintains a more stable, predictable upper thermal structure. The Arabian Sea exhibits higher RMSE ($1.0907^\circ\text{C}$) due to energetic eddy fields and vigorous seasonal overturning.

---

## 25. Seasonal Analysis

Test-partition error evaluated across the late-fall and early-winter transition:
- **Late Fall Transition (Nov 09 – Nov 30, 2020)**: **1.0059 °C** test RMSE. Reflects post-monsoon wind relaxation and rapid mixed-layer shallowing.
- **Early Winter Monsoonal Regime (Dec 01 – Dec 31, 2020)**: **0.9060 °C** test RMSE. Characterized by established northeast monsoon winds and more stable convective mixed layers.

---

## 26. Thermal-Regime Diagnostic Assessment

To evaluate continuous field fidelity across discrete water mass boundaries, predictions were binned into six thermal regimes across $N = 8,017,734$ valid depth observations:

1. **Deep Water**: $T < 10^\circ\text{C}$
2. **Lower Thermocline**: $10^\circ\text{C} \le T < 15^\circ\text{C}$
3. **Core Thermocline**: $15^\circ\text{C} \le T < 20^\circ\text{C}$
4. **Upper Thermocline**: $20^\circ\text{C} \le T < 25^\circ\text{C}$
5. **Mixed Layer**: $25^\circ\text{C} \le T < 28^\circ\text{C}$
6. **Tropical Warm Pool**: $T \ge 28^\circ\text{C}$

### Certified Diagnostic Results Table
| Model | Exact Accuracy | Within $\pm 1$ Bin | Beyond $\pm 1$ Bin | Cohen's $\kappa$ | Macro F1 | Weighted F1 | Classification Role |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B1 Climatology** | 76.23% | 99.39% | 0.61% | 0.7078 | 0.7632 | 0.7604 | Baseline Reference Anchor |
| **B2 Ridge** | 79.87% | 99.91% | 0.09% | 0.7538 | 0.8023 | 0.8014 | Supervised Linear Benchmark |
| **B3 Random Forest**| **80.65%** | 99.71% | 0.29% | **0.7636** | **0.8085** | **0.8099** | Bagging Ensemble Benchmark |
| **B5 Pointwise MLP** | 77.67% | 99.85% | 0.15% | 0.7275 | 0.7876 | 0.7797 | Pointwise Neural Baseline |
| *Legacy Tuning MLP* | *81.19%* | *99.84%* | *0.16%* | *0.7699* | *0.8133* | *0.8146* | Historical Tuning Candidate |

*Scientific Interpretation*:
- **Error Adjacency**: More than 99.7% of evaluated predictions fall within the true regime bin or an immediately adjacent bin ($99.91\%$ for B2, $99.85\%$ for B5, $99.71\%$ for B3). This indicates that classification errors are predominantly localized near continuous temperature-regime boundaries.
- **Physical Monotonicity Constraint**: High $\pm 1$-bin containment does not prove vertical profile monotonicity or rule out localized gradient inversions.
- **Diagnostic Role**: Discretization into thermal regimes is a secondary diagnostic assessment; primary optimization is performed on continuous column-averaged RMSE.
- **Legacy Provenance Clarification**: The 81.19% figure previously recorded originated from an exploratory tuning candidate (`b3_MLP_128_64_best.pt`, 10,255 parameters). Under the locked benchmark hierarchy, canonical B3 is the Multi-Depth Random Forest (80.65% exact accuracy, 13,289,966 nodes across 750 trees).

---

## 27. ARGO–GLORYS Reference Consistency Assessment

- **Dataset**: $N = 1,482$ spatio-temporally collocated in-situ Argo float profiles matched within $\pm 0.25^\circ$ and $\pm 12\text{ hours}$ across the North Indian Ocean during 2020.
- **Assessment Nature**: Evaluates consistency between the continuous numerical target (GLORYS12V1 reanalysis) and direct in-situ profiling floats.
- **Explicit Methodological Constraint**: Operational Argo profiles are assimilated into the GLORYS reanalysis system. Therefore, the ARGO–GLORYS comparison assesses reanalysis reference consistency and does not constitute independent held-out ML model validation. Future work will incorporate unassimilated delayed-mode experimental profiles.

---

## 28. Error Analysis

1. **Upper-Ocean Gradient Sensitivity**: The largest absolute residuals occur in regions of steep pycnocline displacement, particularly along the Oman and Somali upwelling filaments.
2. **Equatorial Wave Trapping**: Zonal currents in the equatorial band exhibit rapid transient phase speeds ($>1\text{ m/s}$) associated with equatorial Kelvin and Rossby waves, where a 5-day window introduces slight phase lag.
3. **Deep Variance Floor**: Neural models tend to predict slight variance around the deep mean, leading to slightly higher RMSE at 1000 m than static climatology ($0.5959^\circ\text{C}$ vs $0.3709^\circ\text{C}$).

---

## 29. Scientific Limitations

1. **Single Full-Year Scope**: Evaluated on 2020 leap year (366 days). Interannual climate modes (e.g., extreme Indian Ocean Dipole or ENSO events) require decadal training.
2. **Reanalysis Target Proxy**: Models are trained against GLORYS12V1 numerical reanalysis rather than direct 3D in-situ observations.
3. **Assimilated Argo Benchmark**: Operational Argo profiles are assimilated into GLORYS; true observational ground truth validation remains external.
4. **Depth-Dependent Error Profile**: Headline metric of 0.9800 °C is an unweighted column average; errors peak at 1.8110 °C at 75 m.
5. **Discrete 15-Depth Representation**: Vertical soundings are reconstructed at 15 discrete levels rather than as a continuous function.
6. **Local Spatial Window Boundary**: $3 \times 3$ patches capture mesoscale footprints (~75 km) but omit basin-scale teleconnections.
7. **Frontend Prototype Mode**: The web dashboard is a decoupled local simulation prototype.
8. **Multi-Year Generalization**: Generalization across decades remains uncertified.

---

## 30. Future Work

1. **Multi-Decadal Scaling**: Extend data ingestion to 1993–2022 to evaluate cross-decadal generalization across varied IOD/ENSO cycles.
2. **Unassimilated In-Situ Validation**: Match against delayed-mode experimental floats, marine research cruise CTDs, and underwater gliders excluded from assimilation.
3. **Physics-Informed Neural Constraints**: Enforce hydrodynamic vertical stability constraints ($\partial \rho / \partial z \ge 0$) via density loss penalties.
4. **Continuous Vertical Representations**: Implement Neural Ordinary Differential Equations (Neural ODEs) or implicit neural representations for continuous depth prediction.
5. **Uncertainty Quantification**: Implement Bayesian deep ensembles or conformal prediction to output calibrated confidence bounds per depth.
6. **Operational Real-Time Pipeline**: Deploy automated daily ingestion pipelines consuming live Copernicus Marine satellite streams.

---

## 31. Reproducibility

Full reproduction from a clean checkout requires no external manual interventions:

```powershell
# 1. Environment Activation
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Verify Dataset & Scaler Metadata
python scripts/01_verify_dataset_corrected.py

# 3. Execute Complete Automated Test Suite (106 tests, 0 skips)
python -m pytest tests/ -rs

# 4. Execute Benchmark Scripts
python scripts/train_b0_b1.py
python scripts/train_b2.py
python scripts/train_b3.py

# 5. Build and Verify Frontend
npm install
npm run build

# 6. Launch Prototype Dashboard
python scripts/serve_dashboard.py
```

---

## 32. Conclusion

This project has successfully established and certified an end-to-end, scientifically defensible machine learning framework for reconstructing depth-wise subsurface ocean potential temperature across the North Indian Ocean. By enforcing strict chronological splitting with 6-day purge buffers, train-only normalization, and four-way bathymetric masking, empirical data leakage was prevented.

Across the 10-model hierarchy evaluated on the frozen 2020 test partition, the B8 Spatiotemporal Embedding Network achieved a column-averaged test RMSE of **0.9800 °C**, representing a **22.11% relative error reduction** over spatial climatology (B1: 1.2582 °C). Paired 7-day block bootstrap hypothesis testing confirmed statistical significance ($p < 0.001$). Diagnostic thermal regime classification demonstrated that over 99.7% of predictions fall within $\pm 1$ bin of true regime boundaries.

**B8 was the best-performing architecture among evaluated internal benchmarks.** The complete codebase, test suite, and interactive prototype provide a rigorous foundation for future operational oceanographic research.

---

## 33. References

1. Donlon, C. J., et al. (2012). The Operational Sea Surface Temperature and Sea Ice Analysis (OSTIA) system. *Remote Sensing of Environment*, 116, 140–158.
2. Jean-Michel, L., et al. (2021). The Copernicus Global 1/12° Oceanic and Sea Ice GLORYS12 Reanalysis and Simulation: Description and Quality Assessment. *Frontiers in Earth Science*, 9, 698876.
3. Pujol, M.-I., et al. (2016). DUACS DT2014: The new multi-mission altimeter data set reprocessed over 20 years. *Ocean Science*, 12(5), 1067–1090.
4. Bonjean, F., & Lagerloef, G. S. (2002). Diagnostic model and analysis of the surface currents in the tropical Pacific Ocean. *Journal of Physical Oceanography*, 32(9), 2438–2454.
5. Wentz, F. J., et al. (2015). Remote Sensing Systems Cross-Calibrated Multi-Platform (CCMP) 6-hourly ocean surface wind vector analyses. *Remote Sensing Systems*, Santa Rosa, CA.
6. Roemmich, D., et al. (2009). The Argo Program: Observing the global ocean with profiling floats. *Oceanography*, 22(2), 34–43.
7. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
8. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 3146–3154.
9. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *Empirical Methods in Natural Language Processing (EMNLP)*, 1724–1734.
10. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. *arXiv preprint arXiv:1607.06450*.

---

## 34. Appendix

### A. Certified Hardware & Software Environment
- **Operating System**: Windows 11 Home (x64)
- **Python**: 3.11.9
- **PyTorch**: 2.0+ (CPU/CUDA compatible)
- **Node.js**: v20+ / TypeScript 5.6.3 / React 18.3.1 / Vite 5.4.11
- **Testing Framework**: Pytest 9.1.1 (106 tests, 0 failures, 0 skips)

### B. Certified Dataset Checksums
- `data/metadata/tabular_scaler_stats.json`: SHA-256 `279b13aa6662c693b4a36da0bfd78ff35ed5213a0e16b0fbb7560d657ee02fe3`
- `results/B3.json`: SHA-256 `4ae89a43154d00e4dbb4362c9490a6a7074a9d839ac00c9e33155501153cc548`
- `reports/full_year_acceptance_certified.json`: Certified All 9 Pre-Training Gates Passed.
