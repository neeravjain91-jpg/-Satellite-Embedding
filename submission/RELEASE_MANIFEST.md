# Final Academic Submission Release Manifest
## Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature

---

### 1. Release Overview
- **Project Title**: **Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature**
- **Repository**: `https://github.com/neeravjain91-jpg/-Satellite-Embedding`
- **Certified Release Commit**: `88f7dbc99cae223a2a66539cddbb5073ca4360c1`
- **Frontend Production URL**: [https://code-gules-three.vercel.app](https://code-gules-three.vercel.app)
- **Vercel Deployment URL**: [https://code-jry5vs2ze-neeravjain91-6032s-projects.vercel.app](https://code-jry5vs2ze-neeravjain91-6032s-projects.vercel.app)
- **Vercel Deployment ID**: `dpl_8rqrbnpzFUPRAJRGCWfBM7BvJjrM`
- **Release Date**: October 2026

---

### 2. Dataset & Geographical Domain
- **Temporal Scope**: Full-year 2020 leap-year partition (366 days, 2020-01-01 to 2020-12-31).
- **Geographic Domain**: North Indian Ocean ($5.00^\circ\text{N}–30.00^\circ\text{N}, 45.00^\circ\text{E}–105.00^\circ\text{E}$).
- **Spatial Resolution**: $0.25^\circ \times 0.25^\circ$ regular equirectangular grid ($101 \times 241 = 24,341$ points).
- **Active Marine Grid Columns**: 11,350 valid ocean columns per day ($N = 4,017,900$ total space-time columns).
- **Surface Predictors ($C=7$)**:
  1. OSTIA Sea Surface Temperature (SST, METOFFICE-GLO-SST-L4-REP-OBS-SST)
  2. Copernicus Multi-Observation Sea Surface Salinity (SSS, cmems_obs-mob_glo_phy-sss_my_multi_P1D)
  3. DUACS Sea Level Anomaly / SSH (c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D)
  4. OSCAR Zonal Surface Geostrophic Current U (c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D)
  5. OSCAR Meridional Surface Geostrophic Current V
  6. CCMP V3.1 Zonal Surface Wind Vector U (CCMP_RT_Wind_Analysis_19930101_20231231_V03.1)
  7. CCMP V3.1 Meridional Surface Wind Vector V
- **Target Field**: Copernicus Marine GLORYS12V1 daily potential temperature ($\theta_o$) across 15 canonical depths:
  `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] meters`.
- **Target Characterization**: Explicitly identified as the **GLORYS numerical ocean reanalysis reference** state estimate (not raw observational ground truth).

---

### 3. Chronological Partitions & Leakage Controls
- **TRAIN Split**: Days 0–252 (253 calendar days, Jan 01 – Sep 09, 2020; $N = 2,871,550$ profiles / 69.1%).
- **PURGE BUFFER 1**: Days 253–258 (6 calendar days, Sep 10 – Sep 15, 2020; discarded to prevent causal overlap).
- **VAL Split**: Days 259–306 (48 calendar days, Sep 16 – Nov 02, 2020; $N = 544,800$ profiles / 13.1%).
- **PURGE BUFFER 2**: Days 307–312 (6 calendar days, Nov 03 – Nov 08, 2020; discarded to prevent causal overlap).
- **TEST Split**: Days 313–365 (53 calendar days, Nov 09 – Dec 31, 2020; $N = 601,550$ profiles / 14.5%, $8,017,734$ valid depth observations).
- **Leakage Protection Protocol**:
  - Buffer width guarantee: $T_{\text{purge}} = 6\text{ days} > T_{\text{causal}} = 5\text{ days}$.
  - Normalization parameters fitted strictly on Days 0–252 training data (SHA-256: `279b13aa6662c693b4a36da0bfd78ff35ed5213a0e16b0fbb7560d657ee02fe3`). Zero test statistics leaked into preprocessing.
  - Causal receptive field: strictly backwards-looking $[t-4, t-3, t-2, t-1, t]$.
  - Composite 4-way mask: strictly preserves GLORYS/ORCA12 model bathymetric cutoffs; invalid target NaNs are preserved as masked invalid targets throughout preprocessing, training loss, and evaluation, and are never converted to zero.

---

### 4. Evaluated Model Hierarchy
| Level | Model ID | Canonical Architecture Designation | Context Scope | Parameters / Complexity | Certified Test RMSE | Relative Error Reduction (vs B1) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **0** | **B0** | Day-0 Persistence | Boundary $t=0$ state | 0 | $1.5220\ ^\circ\text{C}$ | -20.97% |
| **0** | **B0b** | Day-252 Persistence | Train boundary $t=252$ state | 0 | $1.7287\ ^\circ\text{C}$ | -37.39% |
| **0** | **B1** | Spatial-Depth Climatology Reference | Spatial-depth historical mean | 0 | $1.2582\ ^\circ\text{C}$ | **0.00% (Anchor)** |
| **1** | **B2** | Multi-Output Ridge ($\alpha=100{,}000$) | Pointwise (7 variables) | 120 coefficients | $1.0295\ ^\circ\text{C}$ | +18.18% |
| **1** | **B3** | Multi-Depth Random Forest (750 trees) | Pointwise (7 variables) | 13,289,966 decision nodes across 750 Random Forest trees | $1.0452\ ^\circ\text{C}$ | +16.93% |
| **1** | **B4** | Gradient Boosted Trees (LightGBM) | Pointwise (7 variables) | 750 boosted trees | $1.0288\ ^\circ\text{C}$ | +18.23% |
| **2** | **B5** | Pointwise MLP (128-128-64) | Pointwise (7 variables) | 26,767 trainable parameters | $1.5524\ ^\circ\text{C}$ | -23.38% |
| **2** | **B6** | Spatial CNN ($P=3$, $3 \times 3$ patch) | Spatial patch ($3 \times 3$, 7 channels) | 30,991 trainable parameters | $1.2702\ ^\circ\text{C}$ | -0.95% |
| **2** | **B7** | Temporal GRU ($T=5$ causal window) | Causal temporal sequence ($T=5$) | 44,111 trainable parameters | $1.5320\ ^\circ\text{C}$ | -21.76% |
| **2** | **B8** | Spatiotemporal Embedding Model | Spatiotemporal cube ($T=5, C=7, P=3$) | **203,791 trainable parameters** | **0.9800 °C** | **+22.11% (Champion)** |

---

### 5. Primary Champion (B8) Results
- **Primary Metric**: Column-averaged test Root Mean Squared Error (RMSE) across 15 canonical depths: **0.9800 °C**.
- **Error Reduction vs Climatology Anchor (B1)**: **+22.11%** ($\Delta\text{RMSE} = -0.2782^\circ\text{C}$).
- **Scientific Designation**: **"B8 was the best-performing architecture among the evaluated internal benchmarks."** (Zero unsupported "State-of-the-Art" claims).
- **Exact Depth-Wise Test RMSE**:
  - $0\text{ m}$: $0.4369^\circ\text{C}$ ($\text{MAE} = 0.3213^\circ\text{C}$)
  - $5\text{ m}$: $0.4381^\circ\text{C}$ ($\text{MAE} = 0.3316^\circ\text{C}$)
  - $10\text{ m}$: $0.4635^\circ\text{C}$ ($\text{MAE} = 0.3479^\circ\text{C}$)
  - $20\text{ m}$: $0.5962^\circ\text{C}$ ($\text{MAE} = 0.4277^\circ\text{C}$)
  - $30\text{ m}$: $0.8607^\circ\text{C}$ ($\text{MAE} = 0.6312^\circ\text{C}$)
  - $50\text{ m}$: $1.3493^\circ\text{C}$ ($\text{MAE} = 0.9611^\circ\text{C}$)
  - $75\text{ m}$: $1.8110^\circ\text{C}$ ($\text{MAE} = 1.3867^\circ\text{C}$, Thermocline shear peak error)
  - $100\text{ m}$: $1.7651^\circ\text{C}$ ($\text{MAE} = 1.3604^\circ\text{C}$)
  - $125\text{ m}$: $1.5022^\circ\text{C}$ ($\text{MAE} = 1.1748^\circ\text{C}$)
  - $150\text{ m}$: $1.3650^\circ\text{C}$ ($\text{MAE} = 1.0873^\circ\text{C}$)
  - $200\text{ m}$: $1.1834^\circ\text{C}$ ($\text{MAE} = 0.8985^\circ\text{C}$)
  - $300\text{ m}$: $0.9845^\circ\text{C}$ ($\text{MAE} = 0.7284^\circ\text{C}$)
  - $500\text{ m}$: $0.6870^\circ\text{C}$ ($\text{MAE} = 0.5115^\circ\text{C}$)
  - $700\text{ m}$: $0.6619^\circ\text{C}$ ($\text{MAE} = 0.4735^\circ\text{C}$)
  - $1000\text{ m}$: $0.5959^\circ\text{C}$ ($\text{MAE} = 0.4442^\circ\text{C}$)
- **Regional Basin Validation (Cosine-Latitude Area-Weighted)**:
  - Full Domain: $0.9642^\circ\text{C}$
  - Bay of Bengal: $0.6775^\circ\text{C}$
  - Arabian Sea: $1.0907^\circ\text{C}$
- **Seasonal Partition Validation**:
  - Late Fall (Days 313–334, Nov 09 – Nov 30): $1.0059^\circ\text{C}$
  - Early Winter (Days 335–365, Dec 01 – Dec 31): $0.9060^\circ\text{C}$

---

### 6. Statistical Hypothesis Testing
- **Statistical Framework**: Paired 7-day moving block bootstrap resampling ($B = 1,000$ iterations) accounting for temporal autocorrelation in ocean dynamics.
- **B8 vs. B1 Continuous RMSE Gain**:
  - Point estimate: $\Delta\text{RMSE} = -0.2782^\circ\text{C}$
  - Paired 95% Confidence Interval: $[-0.3957, -0.1756]^\circ\text{C}$
  - Hypothesis test: Two-sided $p < 0.001$ (statistically significant at $\alpha = 0.01$).
- **Tabular Baselines Paired 95% CIs vs. B1**:
  - B2 Ridge: $[-0.3208, -0.1479]^\circ\text{C}$
  - B3 Random Forest: $[-0.3478, -0.1000]^\circ\text{C}$
  - B4 LightGBM: $[-0.3617, -0.1195]^\circ\text{C}$

---

### 7. Diagnostic Thermal Regime Evaluation
Discretization across six oceanographic thermal regimes confirms that deep learning and ensemble models preserve vertical regime structure with $>99.7\%$ within-$\pm 1$-bin accuracy:
- **B1 Climatology**: Accuracy = $76.23\%$, Within-$\pm 1$-Bin = $99.39\%$, $\kappa = 0.7078$, Macro F1 = $0.7632$, Weighted F1 = $0.7604$
- **B2 Ridge**: Accuracy = $79.87\%$, Within-$\pm 1$-Bin = $99.91\%$, $\kappa = 0.7538$, Macro F1 = $0.8023$, Weighted F1 = $0.8014$
- **B3 Random Forest**: Accuracy = $80.65\%$, Within-$\pm 1$-Bin = $99.71\%$, $\kappa = 0.7636$, Macro F1 = $0.8085$, Weighted F1 = $0.8099$
- **B5 Pointwise MLP**: Accuracy = $77.67\%$, Within-$\pm 1$-Bin = $99.85\%$, $\kappa = 0.7275$, Macro F1 = $0.7876$, Weighted F1 = $0.7797$
- **Legacy Exploratory MLP (Historical Non-Canonical)**: Accuracy = $81.19\%$, Within-$\pm 1$-Bin = $99.84\%$, $\kappa = 0.7699$, Macro F1 = $0.8133$, Weighted F1 = $0.8146$ (properly archived as historical baseline).

---

### 8. Engineering Verification, Test Suite & Build Status
- **Test Suite**: Pytest 9.1.1 on Python 3.11.9.
  - **107 tests collected, 107 passed, 0 failed, 0 skipped** (100% pass rate in 333.88s).
  - Validates model architecture instantiation, temporal causality, purge buffers, data normalization, NaN masking, and cross-artifact consistency across JSON, Markdown, and TypeScript.
- **Frontend Build**: React 18 + Vite 5 + TypeScript production build (`npm run build`).
  - **0 TypeScript errors, 0 build errors**.
  - Production bundles: `dist/assets/index-qmrJetGv.js` (287.25 kB) and `dist/assets/index-DpXea7zc.css` (35.89 kB).
- **Vercel Cloud Production**:
  - Live HTTP status: **200 OK**.
  - Deployment status: `READY` (Aliased to `https://code-gules-three.vercel.app`, Deployment ID: `dpl_8rqrbnpzFUPRAJRGCWfBM7BvJjrM`).

---

### 9. Explicit Scientific Disclosures & Known Limitations
1. **Single-Year Benchmark Scope**: Evaluated exclusively on full-year 2020 (366 days). Multi-decadal variability and major climate indices (IOD/ENSO) remain external future work.
2. **Reanalysis Reference Semantics**: Copernicus Marine GLORYS12V1 is a numerical ocean reanalysis model assimilation state estimate, not raw in-situ observational ground truth.
3. **ARGO Matchup Independence**: Operational in-situ Argo profiling floats are assimilated into the GLORYS reanalysis system; therefore, the ARGO–GLORYS comparison ($N=1,482$ profiles) represents an **ARGO–GLORYS Reference Consistency Assessment** evaluating reanalysis fidelity rather than serving as independent validation of the machine learning model.
4. **Depth-Varying Error Profile**: Column-averaged RMSE ($0.9800^\circ\text{C}$) is an unweighted average; errors peak in the high-shear thermocline at 75 m ($1.8110^\circ\text{C}$) and decrease to $0.4369^\circ\text{C}$ at the surface and $0.5959^\circ\text{C}$ at 1000 m.
5. **Interactive Prototype Scope**: The web frontend is a client-side simulation prototype featuring local mock soundings and interactive spatial controls (`PROTOTYPE • LOCAL SIMULATION`), decoupled from live Python backend inference.

---

### 10. Release Certification Signoff
All 16 phase requirements, model locks, leakage protections, and automated test gates have been fully verified and locked.

**RELEASE STATUS: COMPLETE — SUBMISSION READY**
