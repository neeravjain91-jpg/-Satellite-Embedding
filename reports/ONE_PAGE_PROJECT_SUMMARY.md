# Executive One-Page Project Summary
## Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature
**Daily 0.25° Reconstruction Across the North Indian Ocean Using Surface Satellite and Oceanographic Observations**

---

### 1. Problem Statement & Motivation
Satellite sensors observe only the ocean surface skin and mixed-layer boundary, leaving three-dimensional subsurface thermal stratification unobserved. While sparse autonomous Argo floats provide vertical measurements, their 3-degree resolution leaves major mesoscale and synoptic gaps. Subsurface ocean thermal structure governs tropical cyclone intensification, marine heatwaves, and acoustic propagation. This project develops a scientifically controlled machine learning framework to reconstruct vertical potential temperature profiles across 15 standard oceanographic depths ($0\text{ m}$ to $1000\text{ m}$) from daily satellite surface observations across the North Indian Ocean ($5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$).

### 2. Dataset & Zero-Leakage Protocol
- **Temporal Domain**: Certified full-year 2020 leap year (366 consecutive days, Jan 1 – Dec 31, 2020).
- **Spatial Grid**: Regular $0.25^\circ \times 0.25^\circ$ grid ($101 \times 241 = 24,341$ points per time step, 11,350 valid ocean columns per day).
- **Surface Predictors ($C=7$)**: OSTIA SST, Copernicus Multi-Observation SSS, DUACS SSH, OSCAR zonal/meridional currents, CCMP zonal/meridional vector winds.
- **Reference Target**: Copernicus Marine GLORYS12V1 daily potential temperature ($\theta_o$) across 15 canonical depths: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m`. *(Note: GLORYS is a numerical ocean reanalysis state estimate, not direct observational ground truth).*
- **Chronological Split**: TRAIN (Days 0–252, 253d, $N=2,871,550$), PURGE 1 (Days 253–258, 6d), VAL (Days 259–306, 48d, $N=544,800$), PURGE 2 (Days 307–312, 6d), TEST (Days 313–365, 53d, $N=601,550$ columns, $8,017,734$ valid depth observations).
- **Leakage Controls**: All normalization scalers fitted strictly on training data; 6-day purge buffers exceed temporal autocorrelation memory; 4-way composite mask preserves seafloor bathymetry; invalid target NaNs are preserved as masked invalid targets and are never converted to zero.

### 3. Model Hierarchy & Champion B8 Architecture
We established a locked 10-model hierarchy: persistence (B0: 1.5220 °C, B0b: 1.7287 °C, -37.39%), daily climatology (B1: 1.2582 °C), regularized linear regression (B2: 1.0295 °C, 120 coefficients), random forest (B3: 1.0452 °C, 13,289,966 decision nodes across 750 Random Forest trees), LightGBM (B4: 1.0288 °C, 750 boosted trees), pointwise MLP (B5: 1.5524 °C, 26,767 trainable parameters), spatial CNN (B6: 1.2702 °C, 3×3 spatial patch, P=3, 30,991 trainable parameters), temporal GRU (B7: 1.5320 °C, 5-day causal temporal history, T=5, 44,111 trainable parameters), and the champion B8 Spatiotemporal Embedding Network (203,791 trainable parameters).

**Canonical B8 Architecture (203,791 trainable parameters)**:
- **Input**: $T=5, C=7, P=3$ spatiotemporal cubes $[B, T=5, C=7, P=3, P=3]$.
- **Per-Timestep Spatial Encoder**: Conv2D ($7 \to 32 \to 64$, $3 \times 3$ spatial patch) with BatchNorm2d and AdaptiveAvgPool2d into a 64-dimensional latent token.
- **Temporal Encoder**: 2-layer causal GRU ($\text{input} = 64, \text{hidden} = 128$).
- **Latent Bottleneck**: $\text{LayerNorm}(128)$ yielding compressed ocean state vector $z \in \mathbb{R}^{128}$.
- **Decoder**: $128 \to 64 \to 15$ linear decoder mapping to 15 canonical depths.
- **Total**: Exactly **203,791** trainable parameters.

### 4. Primary Results & Statistical Validation
- **Primary Test RMSE**: B8 achieves an overall column-averaged test RMSE of **0.9800 °C** (*unweighted 15-depth column average, not error at every depth*), representing a **+22.11% relative error reduction** over spatial climatology (B1: 1.2582 °C). **B8 was the best-performing architecture among evaluated internal benchmarks.**
- **Statistical Significance**: Paired 7-day moving block bootstrap resampling ($B=1000$) confirms significance: difference $-0.2782^\circ\text{C}$, 95% CI: $[-0.3957, -0.1756]^\circ\text{C}$, $p < 0.001$.
- **Depth-Wise Error Profile**: 0 m ($0.4369^\circ\text{C}$), 5 m ($0.4381^\circ\text{C}$), 10 m ($0.4635^\circ\text{C}$), 20 m ($0.5962^\circ\text{C}$), 30 m ($0.8607^\circ\text{C}$), 50 m ($1.3493^\circ\text{C}$), 75 m ($1.8110^\circ\text{C}$, thermocline gradient peak), 100 m ($1.7651^\circ\text{C}$), 125 m ($1.5022^\circ\text{C}$), 150 m ($1.3650^\circ\text{C}$), 200 m ($1.1834^\circ\text{C}$), 300 m ($0.9845^\circ\text{C}$), 500 m ($0.6870^\circ\text{C}$), 700 m ($0.6619^\circ\text{C}$), 1000 m ($0.5959^\circ\text{C}$).
- **Regional Performance**: Bay of Bengal ($0.6775^\circ\text{C}$, river-stratified), Equatorial corridor ($0.9412^\circ\text{C}$), Arabian Sea ($1.0907^\circ\text{C}$, upwelling/eddy-rich), Full domain cosine-weighted ($0.9642^\circ\text{C}$).

### 5. Diagnostic Thermal Regime Evaluation
Discretizing into 6 oceanographic regimes (<10 °C, 10–15 °C, 15–20 °C, 20–25 °C, 25–28 °C, $\ge 28$ °C) across $N=8,017,734$ points:
B3 Random Forest: 80.65% exact accuracy ($\kappa = 0.7636$); B2 Ridge: 79.87% ($\kappa = 0.7538$); B5 MLP: 77.67% ($\kappa = 0.7275$); B1 Climatology: 76.23% ($\kappa = 0.7078$). Over 99.7% of predictions fall within $\pm 1$ bin across all supervised ML models ($99.91\%$ for B2, $99.71\%$ for B3, $99.85\%$ for B5), indicating that classification errors are predominantly localized near continuous temperature-regime boundaries. *(Constraint: $\pm 1$-bin containment does not prove vertical profile monotonicity or rule out localized gradient inversions).*

### 6. In-Situ ARGO Consistency Assessment
Collocation against $N=1,482$ in-situ Argo profiles ($\pm 0.25^\circ, \pm 12\text{h}$) assesses reanalysis reference consistency between GLORYS and direct floats. Because GLORYS assimilates operational Argo floats, this does not constitute independent held-out ML validation.

### 7. Limitations & Future Horizons
- **Limitations**: Single-year scope (2020); reanalysis target proxy; thermocline error peak ($1.8110^\circ\text{C}$); frontend is a decoupled prototype simulation.
- **Future Work**: Multi-decadal scaling (1993–2022); validation against unassimilated delayed-mode experimental floats; physics-informed vertical density penalties ($\partial \rho/\partial z \ge 0$); continuous Neural ODEs.

### 8. Reproducibility & Engineering Artifacts
- **Repository**: `https://github.com/neeravjain91-jpg/-Satellite-Embedding` (Certified Release Commit `88f7dbc99cae223a2a66539cddbb5073ca4360c1`).
- **Test Suite**: 107 automated pytest unit/integration/consistency tests (107 passed, 0 failed, 0 skipped).
- **Web Prototype**: Interactive React + Vite frontend live at `https://code-gules-three.vercel.app` (0 build errors).
- **Reproduction**: `pytest tests/ -rs`, `python scripts/01_verify_dataset_corrected.py`, `python scripts/serve_dashboard.py`.
