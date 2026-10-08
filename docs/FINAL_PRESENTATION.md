# Final Academic Presentation Slide Deck
## Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature
**Daily 0.25° Reconstruction Across the North Indian Ocean Using Surface Satellite and Oceanographic Observations**

---

### Slide 1: Title Slide
# Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature
### Daily 0.25° Reconstruction Across the North Indian Ocean Using Surface Satellite and Oceanographic Observations

- **Candidate**: B.Tech Capstone Project Defense
- **Institution**: Department of Computer Science & Engineering / Earth & Planetary Sciences
- **Academic Year**: 2025–2026
- **Domain**: North Indian Ocean (5°N–30°N, 45°E–105°E) | **Dataset**: Certified Full-Year 2020 Leap Year (366 Days)
- **Git Checkpoint**: `7532359` | **Live Web Prototype**: `https://code-gules-three.vercel.app`

---

### Slide 2: Scientific Problem
## The Observation Dilemma in 3D Oceanography
- **The Core Constraint**:
  - Satellites observe only the surface skin and mixed layer (top millimeters to centimeters).
  - Subsurface ocean state remains physically invisible to infrared, microwave, and altimetric sensors.
- **In-Situ Observational Gaps**:
  - Autonomous Argo profiling floats sample sparsely (~3° horizontal spacing, 10-day cycle).
  - Leaves critical mesoscale eddies (50–200 km) and rapid synoptic events unobserved.
- **The SciML Objective**:
  - Invert seven daily multi-satellite surface observables to reconstruct continuous 3D vertical potential temperature profiles across 15 standard ocean depths ($0\text{ m}$ to $1000\text{ m}$).

---

### Slide 3: Motivation & Oceanographic Relevance
## Why Subsurface Temperature Matters
- **Tropical Cyclone Intensification**:
  - Cyclones feed on Ocean Heat Content (OHC) integrated down to the 26 °C isotherm ($D_{26}$), not surface SST alone.
- **Marine Heatwaves & Ecological Collapse**:
  - Subsurface thermal anomalies cause severe coral bleaching and fishery collapse.
- **Underwater Acoustic Propagation**:
  - Vertical sound speed profiles determine SOFAR acoustic duct propagation.
- **Computational Opportunity**:
  - Traditional numerical data assimilation (e.g., GLORYS) requires hours of HPC run time; our trained neural network reconstructs the entire basin in <0.5 seconds.

---

### Slide 4: Research Objectives
## Project Goals & Deliverables
1. **Curate Certified Production Dataset**: Harmonize 7 multi-satellite daily surface streams and GLORYS12V1 targets on a unified 0.25° grid across full-year 2020.
2. **Enforce Leakage Controls**: Eliminate empirical data leakage through chronological splitting, 6-day purge buffers, and train-only normalization.
3. **Build 10-Model Benchmark Hierarchy**: Establish systematic baselines from persistence (B0, B0b) and climatology (B1) to tabular ML (B2–B4) and deep neural models (B5–B8).
4. **Design B8 Champion Network**: Integrate local spatial patch convolutions and causal recurrent memory into a 128D latent bottleneck.
5. **Verify Statistical Significance**: Evaluate via paired 7-day moving block bootstrap hypothesis testing against climatological reference B1.

---

### Slide 5: Study Area & Oceanographic Domain
## The North Indian Ocean (5°N–30°N, 45°E–105°E)
- **Spatial Resolution**: Uniform $0.25^\circ \times 0.25^\circ$ grid ($101 \times 241 = 24,341$ points/day).
- **Valid Marine Cells**: 11,350 ocean columns per daily time slice.
- **Contrasting Basin Dynamics**:
  - **Arabian Sea**: High salinity (>36 PSU), intense summer coastal upwelling (Somali Current), energetic mesoscale eddies.
  - **Bay of Bengal**: Massive river runoff (Ganges-Brahmaputra), low surface salinity (<32 PSU), strong freshwater barrier layers.
  - **Equatorial Corridor**: Cross-equatorial jets, Wyrtki jets, planetary Kelvin/Rossby waves.

---

### Slide 6: Dataset & Data Sources
## Multi-Satellite Predictors & Reference Target
| # | Variable | Product / Source | Sensor / Platform | Role |
|---|---|---|---|---|
| 1 | `sst` | OSTIA (UK Met Office) | Multi-satellite IR + MW foundation SST | Surface Predictor |
| 2 | `sss` | Copernicus Multi-Observation SSS | SMOS / Aquarius / SMAP optimal interpolation | Surface Predictor |
| 3 | `ssh` | DUACS (Copernicus Marine) | Multi-mission gridded sea level anomaly | Surface Predictor |
| 4 | `current_u` | OSCAR (NOAA / ESR) | Altimetry + wind-driven surface geostrophic velocity | Surface Predictor |
| 5 | `current_v` | OSCAR (NOAA / ESR) | Altimetry + wind-driven surface geostrophic velocity | Surface Predictor |
| 6 | `wind_u` | CCMP V3.1 (RSS) | Cross-calibrated radiometer/scatterometer vector wind | Surface Predictor |
| 7 | `wind_v` | CCMP V3.1 (RSS) | Cross-calibrated radiometer/scatterometer vector wind | Surface Predictor |
| 8 | `thetao` | GLORYS12V1 (CMEMS) | NEMO physical model assimilating satellite & in-situ | Reanalysis Target |

*Note*: GLORYS12V1 is a numerical ocean reanalysis state estimate, not direct observational ground truth.

---

### Slide 7: Data Harmonization & Masking
## Grid Unification & Seafloor Bathymetry Semantics
- **Interpolation Protocol**:
  - Normalized bilinear regridding for surface variables with coastline preservation.
  - PCHIP (Piecewise Cubic Hermite) monotonic splining for vertical target depths (0–1000 m).
- **Four-Way Composite Mask**:
  $$\mathbf{M} = \mathbf{M}_{\text{geo}} \land \mathbf{M}_{\text{surf}} \land \mathbf{M}_{\text{targ}} \land \mathbf{M}_{\text{depth}}$$
- **Bathymetric Masking Rule**:
  - Points below the seafloor according to GLORYS/ORCA12 model bathymetry are strictly preserved as `NaN`.
  - **Zero-Imputation Prohibition**: Invalid target NaNs are preserved as masked invalid targets and are never converted to zero.

---

### Slide 8: Leakage-Control Protocol
## Chronological Splitting & Purge Buffers
```
[================ TRAIN ================] [PURGE 1] [== VAL ==] [PURGE 2] [=== TEST ===]
Day 0                                 Day 252       Day 259    Day 306    Day 313     Day 365
2020-01-01                         2020-09-09    2020-09-16 2020-11-02 2020-11-09  2020-12-31
(253 days / N=2,871,550)               (6 days)      (48 days)  (6 days)   (53 days / N=601,550)
```
- **Strict Chronological Ordering**: No random shuffling across time.
- **6-Day Purge Buffers**: Exceeds temporal autocorrelation memory and protects $T=5$ causal models ($T_{\text{purge}} = 6 > T = 5$).
- **Train-Only Normalization**: Z-score parameters derived strictly on Days 0–252. Zero test contamination.
- **Causal Sequences**: Backward windows $[t-4, \dots, t]$ strictly forbid forward-looking observations.

---

### Slide 9: Model Hierarchy
## 10 Systematic Benchmarks (B0 through B8)
| ID | Model Name | Architecture Family | Context Representation | Parameters / Complexity |
| :---: | :--- | :--- | :--- | :---: |
| **B0** | Day-0 Persistence | Persistence | Initial state ($t=0$) | 0 |
| **B0b**| Day-252 Persistence | Persistence | Train boundary ($t=252$) | 0 |
| **B1** | Spatial-Depth Climatology | Climatology | Historical train mean profile | 0 |
| **B2** | Multi-Output Ridge | Linear Regularized | Pointwise 7-surface vector | 120 coefficients |
| **B3** | Multi-Depth Random Forest | Bagging Ensemble | Pointwise 7-surface vector | 13,289,966 decision nodes across 750 Random Forest trees |
| **B4** | Gradient Boosting (LightGBM)| Boosting Ensemble | Pointwise 7-surface vector | 750 boosted trees |
| **B5** | Pointwise MLP | Feedforward Neural | Pointwise 7-surface vector | 26,767 trainable parameters |
| **B6** | Spatial CNN | Spatial Convolutional | 3×3 spatial patches ($P=3$) | 30,991 trainable parameters |
| **B7** | Temporal GRU | Sequential Recurrent | 5-day causal sequences ($T=5$) | 44,111 trainable parameters |
| **B8** | Spatiotemporal Embedding | Joint Spatiotemporal | 5-day × 3×3 patch cubes | 203,791 trainable parameters |

---

### Slide 10: Champion Architecture (B8 Network)
## Joint Spatiotemporal Latent Bottleneck
```
Input Spatiotemporal Cube: [B, T=5, C=7, P=3, P=3]
  │
  ├── Time-Distributed 2D CNN Encoder (per timestep)
  │     ├── Conv2D(7 -> 32, 3x3, pad 1) + BatchNorm2d + ReLU
  │     ├── Conv2D(32 -> 64, 3x3, pad 1) + BatchNorm2d + ReLU
  │     └── AdaptiveAvgPool2d((1, 1)) ──> [B, T=5, 64]
  │
  ├── Causal Recurrent Sequence Model
  │     ├── 2-Layer Unidirectional GRU(input=64, hidden=128, dropout=0.1)
  │     └── Last Hidden State h_T: [B, 128]
  │
  ├── Ocean Latent Bottleneck
  │     └── LayerNorm(128) ──> z \in R^128 (Compressed Ocean State)
  │
  └── Subsurface Depth Decoder
        ├── Linear(128 -> 64) + ReLU
        └── Linear(64 -> 15) ──> \hat{Y} \in R^15 (15 Canonical Depths)
```
- **Total Parameters**: Exactly **203,791** trainable weights.
- **Physical Rationale**: Couples horizontal baroclinic shear ($3 \times 3$ patch) with propagating wave memory ($T=5$ GRU).

---

### Slide 11: Training Protocol & Loss Formulation
## Masked Optimization & Resource Footprint
- **Masked MSE Loss**:
  $$\mathcal{L}_{\text{masked}} = \frac{\sum_{i, k} M_{i,k} \cdot (\hat{Y}_{i,k} - Y_{i,k})^2}{\sum_{i, k} M_{i,k} + \epsilon}$$
  Isolates valid marine depths; invalid seafloor NaNs never contaminate gradients.
- **Optimization Strategy**:
  - AdamW optimizer (learning rate $10^{-3}$, weight decay $10^{-4}$).
  - ReduceLROnPlateau (factor 0.5, patience 5).
  - Early stopping on validation masked RMSE (patience 10).
- **Execution Efficiency**:
  - Full-field batch memory < 250 MB VRAM.
  - Trainable on consumer hardware; inference < 0.5s per basin time step.

---

### Slide 12: Certified Benchmark Results
## Master Test Set Evaluation (Days 313–365)
| Model | Test RMSE (°C) | Test MAE (°C) | $\Delta\text{RMSE}$ vs B1 (°C) | Rel. Improvement (%) | Paired 95% Bootstrap CI vs B1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | 1.5220 | 1.1371 | +0.2638 | -20.97% | N/A |
| **B0b**| 1.7287 | 1.2447 | +0.4705 | -37.39% | N/A |
| **B1** | 1.2582 | 0.9641 | 0.0000 | 0.00% | Reference Anchor |
| **B2** | 1.0295 | 0.8029 | -0.2287 | +18.18% | [-0.3208, -0.1479] |
| **B3** | 1.0452 | 0.7725 | -0.2130 | +16.93% | [-0.3478, -0.1000] |
| **B4** | 1.0288 | 0.7615 | -0.2294 | +18.23% | [-0.3617, -0.1195] |
| **B5** | 1.5524 | 1.2030 | +0.2942 | -23.38% | [0.2148, 0.3772] |
| **B6** | 1.2702 | 0.9646 | +0.0120 | -0.95% | [-0.1042, 0.1205] |
| **B7** | 1.5320 | 1.2069 | +0.2738 | -21.76% | [0.2581, 0.2895] |
| **B8** | **0.9800** | **0.7391** | **-0.2782** | **+22.11%** | **[-0.3957, -0.1756]** |

*Certified Outcome*: B8 is the **best-performing architecture among evaluated internal benchmarks**. Paired bootstrap test vs B1: $p < 0.001$.

---

### Slide 13: Depth-Stratified Performance
## Vertical Error Decomposition Across 15 Depths
```
Depth (m)   B8 Test RMSE (°C)    Physical Oceanographic Regime
──────────────────────────────────────────────────────────────────
   0 m           0.4369          Sea Surface / Skin Layer
   5 m           0.4381          Upper Mixed Layer
  10 m           0.4635          Upper Mixed Layer
  20 m           0.5962          Mixed Layer Base
  30 m           0.8607          Upper Pycnocline Transition
  50 m           1.3493          Seasonal Thermocline
  75 m           1.8110 <---     Main Thermocline Core (Peak Gradient Error)
 100 m           1.7651          Main Thermocline Core
 125 m           1.5022          Lower Thermocline
 150 m           1.3650          Lower Thermocline Base
 200 m           1.1834          Sub-Thermocline Transition
 300 m           0.9845          Intermediate Water
 500 m           0.6870          Deep Intermediate Layer
 700 m           0.6619          Deep Ocean Boundary
1000 m           0.5959          Abyssal Reference Level
──────────────────────────────────────────────────────────────────
Col-Avg          0.9800 °C       Unweighted 15-Depth Mean
```
- **Surface**: Strongly constrained by OSTIA SST ($0.4369^\circ\text{C}$).
- **Thermocline Peak (75 m)**: Steep temperature gradient ($\partial T/\partial z$) amplifies isotherm heave error ($1.8110^\circ\text{C}$).
- **Abyssal**: Low physical temperature variance ($\sigma < 0.4^\circ\text{C}$); static climatology acts as an effective anchor.

---

### Slide 14: Regional & Seasonal Generalization
## Basin-Scale & Monsoonal Evaluation
- **Cosine-Latitude Weighted Regional Test RMSE**:
  - **Full Domain**: **0.9642 °C** ($N = 601,550$)
  - **Bay of Bengal**: **0.6775 °C** ($N = 221,840$)
    *Mechanism*: Massive river freshwater capping maintains stable, predictable upper stratification.
  - **Arabian Sea**: **1.0907 °C** ($N = 284,120$)
    *Mechanism*: High evaporation, seasonal upwelling, and vigorous mesoscale eddy fields increase thermal variance.
  - **Equatorial Corridor**: **0.9412 °C** ($N = 95,590$)
    *Mechanism*: Rapid Wyrtki jets and planetary wave propagation.
- **Seasonal Subsets**:
  - **Late Fall Transition (Nov 09 – Nov 30)**: **1.0059 °C** (Monsoon retreat, mixed-layer shallowing).
  - **Early Winter Regime (Dec 01 – Dec 31)**: **0.9060 °C** (Established NE monsoon convection).

---

### Slide 15: Thermal-Regime Diagnostic Evaluation
## Discrete Water Mass Boundary Assessment
Evaluated across $N = 8,017,734$ valid depth observations into 6 thermal regimes:
| Model | Exact Accuracy | Within $\pm 1$ Bin | Beyond $\pm 1$ Bin | Cohen's $\kappa$ | Macro F1 | Weighted F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1 Climatology** | 76.23% | 99.39% | 0.61% | 0.7078 | 0.7632 | 0.7604 |
| **B2 Ridge** | 79.87% | 99.91% | 0.09% | 0.7538 | 0.8023 | 0.8014 |
| **B3 Random Forest**| **80.65%** | 99.71% | 0.29% | **0.7636** | **0.8085** | **0.8099** |
| **B5 Pointwise MLP** | 77.67% | 99.85% | 0.15% | 0.7275 | 0.7876 | 0.7797 |

- **Local Error Adjacency**: Over 99.7% of predictions fall within $\pm 1$ bin across all supervised ML models.
- **Scientific Interpretation**: Classification errors are predominantly localized near continuous temperature-regime boundaries.
- **Constraint**: High $\pm 1$-bin containment does not prove vertical profile monotonicity or rule out localized gradient inversions.

---

### Slide 16: ARGO–GLORYS Consistency Assessment
## In-Situ Float Comparison Role & Scope
- **Assessment Scope**:
  - Matched $N = 1,482$ in-situ Argo float profiles within $\pm 0.25^\circ$ and $\pm 12\text{ hours}$ across the North Indian Ocean in 2020.
- **Methodological Distinction**:
  - Operational Argo profiles are assimilated into GLORYS12V1 during its numerical data assimilation cycle.
  - This assessment evaluates the reference consistency of GLORYS against direct in-situ floats; it does **not** constitute independent held-out ground-truth validation of the ML model.
- **Future Pathway**:
  - Independent validation requires unassimilated delayed-mode experimental floats, research cruise CTDs, and underwater gliders.

---

### Slide 17: Scientific Limitations & Future Work
## Responsible Disclosures & Engineering Horizons
- **Identified Limitations**:
  1. Single-year benchmark (2020): Interannual climate modes (IOD/ENSO) uncertified.
  2. GLORYS target is a numerical reanalysis state estimate, not raw observational ground truth.
  3. Non-uniform vertical error: 0.9800 °C is a column average; thermocline error peaks at 1.8110 °C.
  4. Local horizontal context ($3 \times 3$) omits basin-wide planetary wave teleconnections.
- **Future Research Directions**:
  1. Multi-decadal scaling across 1993–2022.
  2. Physics-informed vertical density stability penalties ($\partial \rho / \partial z \ge 0$).
  3. Continuous vertical representations (Neural ODEs / Implicit Neural Fields).
  4. Calibrated uncertainty estimation (conformal prediction bounds).

---

### Slide 18: Conclusion & Summary
## Key Project Takeaways
- **Rigorous SciML Framework**: Built a zero-leakage, purge-buffered, bathymetrically masked benchmark on full-year 2020 ocean data.
- **Benchmark Performance**:
  - B8 Spatiotemporal Embedding Network achieves **0.9800 °C** column-averaged test RMSE.
  - **+22.11% relative error reduction** over daily spatial climatology (B1: 1.2582 °C).
  - Paired 7-day block bootstrap confirms statistical significance ($p < 0.001$, 95% CI: $[-0.3957, -0.1756]^\circ\text{C}$).
- **Architectural Insight**: Joint spatiotemporal conditioning ($3 \times 3$ patches + causal GRU) outperforms spatial-only (B6) and temporal-only (B7) neural ablations.
- **Quality & Reproducibility**: 106 automated tests passing with zero skips; production interactive prototype live on Vercel.

**Certified Outcome**: Complete, verified, and submission-ready.
