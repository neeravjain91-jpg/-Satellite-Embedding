# Comprehensive Viva & Defense Preparation Guide
## Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature

**Repository**: `neeravjain91-jpg/-Satellite-Embedding`
**Certified Benchmark Checkpoint**: Git Commit `7532359`
**Target Audience**: B.Tech Project Viva Voce Examiners, Technical Evaluators, and Oceanographic Reviewers

---

## Part 1: Core Pitch Explanations

### A. 25-Second Elevator Pitch
"Satellites can only see the ocean surface, yet subsurface temperatures drive cyclones and marine heatwaves. We built an end-to-end deep learning framework that reconstructs 3D temperature profiles across 15 depths down to 1000 meters in the North Indian Ocean using 7 daily surface satellite variables. Evaluated under strict zero-leakage protocols with 6-day purge buffers on full-year 2020 data, our spatiotemporal model B8 achieved a 0.9800 °C test RMSE—a 22.11% improvement over daily climatology, proven significant with paired block bootstrapping."

### B. 60-Second Technical Overview
"Reconstructing subsurface ocean temperature from satellite data is an ill-posed inverse problem. While Argo floats measure vertical profiles, their 3-degree resolution leaves vast unobserved gaps. We addressed this using daily multi-satellite observations—SST, sea surface salinity, sea surface height, surface currents, and vector winds—on a 0.25-degree grid over the North Indian Ocean for the entire year of 2020.

To prevent data leakage, we enforced chronological splitting, 6-day purge buffers, train-only normalization, and four-way bathymetric masking. Across a 10-model hierarchy from persistence to deep learning, our B8 Spatiotemporal Embedding Network—combining a 2D Conv2D patch encoder and a 2-layer causal GRU into a 128-dimensional bottleneck—achieved a column-averaged test RMSE of 0.9800 °C, beating daily climatology at 1.2582 °C by 22.11%. B8 was the best-performing architecture among evaluated internal benchmarks."

### C. 3-Minute Comprehensive Technical Defense
"Subsurface temperature stratification governs oceanic heat storage and cyclone intensification. While operational numerical reanalyses like GLORYS12V1 provide state estimates by assimilating observations into primitive-equation models, they are computationally expensive.

We formulated subsurface reconstruction as a supervised spatiotemporal representation learning problem. Using full-year 2020 leap-year data across the North Indian Ocean (5°N–30°N, 45°E–105°E), we consumed seven daily satellite predictors: OSTIA SST, Copernicus Multi-Observation SSS, DUACS SSH, OSCAR currents, and CCMP winds, predicting GLORYS potential temperature at 15 canonical depths from 0 to 1000 meters.

We addressed widespread data leakage vulnerabilities in published literature:
1. We used strict chronological splitting: 253 days train, 48 days validation, and 53 days held-out test.
2. We placed 6-day purge buffers between splits to eliminate ocean memory autocorrelation.
3. We derived all z-score scalers strictly on the training partition.
4. We enforced four-way masking to preserve bathymetric seafloor cutoffs: invalid target NaNs are preserved as masked invalid targets throughout preprocessing, training loss, and evaluation, and are never converted to zero.

We evaluated a 10-model benchmark hierarchy:
- Reference baselines: Day-0 persistence (1.5220 °C), Day-252 persistence (1.7287 °C), and spatial climatology (1.2582 °C).
- Tabular ML: Multi-Output Ridge (1.0295 °C), Random Forest (1.0452 °C), and LightGBM (1.0288 °C).
- Ablations: Pointwise MLP (1.5524 °C), Spatial CNN (1.2702 °C), and Temporal GRU (1.5320 °C).
- Champion B8: Spatiotemporal Embedding Network with 203,791 parameters, coupling a time-distributed Conv2D on 3x3 patches and a 2-layer causal GRU over 5-day sequences.

B8 achieved 0.9800 °C column-averaged test RMSE (+22.11% over climatology). Paired 7-day block bootstrap testing confirmed significance with a 95% CI of [-0.3957, -0.1756] °C and p < 0.001. Error decomposition showed expected surface accuracy (0.4369 °C at 0 m) and thermocline gradient error (1.8110 °C at 75 m). A secondary diagnostic classification across six thermal regimes confirmed that over 99.7% of predictions fall within ±1 bin of true boundaries. B8 is certified as the best-performing architecture among our evaluated internal benchmarks."

---

## Part 2: Fundamental Concepts & Architectural Justifications

### D. Problem Statement
Given daily satellite surface variables $\mathbf{X}(t, \phi, \lambda) \in \mathbb{R}^7$ and local causal spatiotemporal context $[T=5, C=7, P=3, P=3]$, reconstruct continuous vertical potential temperature $\hat{\mathbf{Y}} \in \mathbb{R}^{15}$ across 15 depths down to 1000 m without future temporal leakage or land/seafloor corruption.

### E. Why Subsurface Ocean Temperature Matters
1. **Tropical Cyclogenesis**: Cyclone intensification depends on Ocean Heat Content (OHC) integrated from the surface to the 26 °C isotherm ($D_{26}$), not surface SST alone.
2. **Acoustic Waveguides**: The speed of underwater sound depends on temperature, salinity, and pressure, forming the SOFAR acoustic channel.
3. **Marine Heatwaves**: Deep-reaching marine heatwaves decimate benthic coral reef ecosystems.
4. **Steric Sea-Level Rise**: Thermal expansion of water accounts for 30–40% of observed global sea-level rise.

### F. Why Surface Variables Can Inform Subsurface State
Through physical dynamic coupling:
- **SSH (Sea Surface Height)**: Integrates vertical density anomaly; high SSH indicates depressed, warm thermoclines; low SSH indicates cold upwelled water.
- **SSS (Sea Surface Salinity)**: Governs upper-ocean buoyancy and barrier-layer formation, especially in the freshwater-stratified Bay of Bengal.
- **Surface Currents**: Reveal geostrophic eddies and shear strain that displace isopycnals.
- **Wind Vectors**: Drive Ekman divergence, coastal upwelling (e.g., Somali jet), and turbulent mixed-layer deepening.

### G. Why GLORYS12V1 Was Used
GLORYS12V1 is a state-of-the-art global reanalysis produced by Mercator Ocean. It assimilates satellite SST, SLA, and in-situ Argo profiles into the NEMO physical ocean model at 1/12° resolution, providing continuous, dynamically balanced 3D gridded fields across 366 consecutive days.

### H. Why 15 Canonical Depths
Depths `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] meters` reflect the standard World Ocean Atlas (WOA) and CMEMS vertical grid:
- Dense sampling in the upper 150 m (8 levels) resolves the mixed layer and main thermocline.
- Moderate sampling from 200 to 500 m (3 levels) captures the sub-thermocline intermediate layer.
- Sparse sampling at 700 and 1000 m (2 levels) anchors deep abyssal water.

### I. Why Full-Year 2020
2020 was a leap year (366 days), providing a complete seasonal cycle encompassing the Northeast monsoon, Spring inter-monsoon, Southwest monsoon, and Fall transition over the North Indian Ocean.

### J. Why Chronological Split
Random shuffling violates the temporal independence assumption. Daily ocean fields have strong temporal autocorrelation. Random splitting causes test frames to interpolate between adjacent training days, resulting in artificial memorization and drastically underestimated error.

### K. Why Purge Buffers
Eulerian ocean anomaly decorrelation timescales are 5–10 days. Placing 6-day purge buffers between train/val (Days 253–258) and val/test (Days 307–312) prevents boundary leakage and ensures that causal sequence windows ($T=5$) never cross partition boundaries.

### L. Why $T=5$ Days
$T=5$ days captures high-frequency baroclinic wave adjustment and transient wind-forcing responses without exceeding the purge buffer width or incurring prohibitive memory overhead.

### M. Why B5, B6, B7, and B8 are Conceptually Different
- **B5 (Pointwise MLP)**: Only considers local 1D column predictors at $(t, \phi, \lambda)$. Blind to neighboring horizontal physics and temporal history.
- **B6 (Spatial CNN)**: Consumes a $3 \times 3$ horizontal patch ($\sim 75 \times 75$ km) at time $t$. Captures horizontal eddy gradients and fronts, but lacks temporal evolution.
- **B7 (Temporal GRU)**: Consumes a 5-day sequence at a single point. Models temporal wave propagation, but is blind to horizontal advection.
- **B8 (Spatiotemporal Embedding)**: Jointly encodes $3 \times 3$ spatial patches across 5 consecutive days, capturing advection, wave propagation, and local buoyancy simultaneously.

### N. Why B8 Performs Best Among Evaluated Models
Ocean stratification is governed by 3D fluid dynamics (the Navier-Stokes equations with rotation and stratification). Neither spatial context nor temporal history alone is sufficient. Coupling spatial patch representations and causal recurrence matches the physical reality of propagating, advecting ocean eddies.

### O. Why B8 is NOT Called "State-of-the-Art" (SOTA)
In adherence to scientific integrity, B8 is designated strictly as the **"Best-performing architecture among evaluated internal benchmarks"**. Claims of universal SOTA require exhaustive comparisons against all published external algorithms across multi-decadal benchmarks, which was outside this single-year study.

### P. Meaning of RMSE
Root Mean Squared Error measures the sample standard deviation of differences between predicted and reference values. Penalizes large outliers heavily:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum (\hat{y}_i - y_i)^2}$$

### Q. Meaning of MAE
Mean Absolute Error measures the average magnitude of absolute errors, providing a linear, less outlier-sensitive error scale:
$$\text{MAE} = \frac{1}{N} \sum |\hat{y}_i - y_i|$$

### R. Meaning of $R^2$ vs Climatology
In oceanography, $R^2$ is referenced against climatology (the historical mean profile, B1):
$$R^2 = 1 - \frac{\sum (\hat{y}_i - y_i)^2}{\sum (y_{\text{clim}, i} - y_i)^2}$$
A positive $R^2$ indicates the model outperforms the seasonal climatological mean.

### S. Meaning of Pearson Correlation ($r$)
Measures the linear association and spatial/vertical pattern similarity between predicted and reference temperature profiles:
$$r = \frac{\sum (\hat{y}_i - \bar{\hat{y}})(y_i - \bar{y})}{\sqrt{\sum (\hat{y}_i - \bar{\hat{y}})^2 \sum (y_i - \bar{y})^2}}$$

### T. Meaning of Cohen's Kappa ($\kappa$)
Measures inter-rater agreement for categorical regime classification above chance agreement:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
where $p_o$ is observed accuracy and $p_e$ is chance-expected accuracy. $\kappa > 0.70$ signifies substantial agreement.

### U. Meaning of $\pm 1$-Bin Accuracy
The percentage of predictions that fall within the true discrete thermal regime or an immediately adjacent regime bin. Demonstrates whether errors are localized near continuous boundaries.

### V. Role of ARGO Floats
Argo floats provide direct in-situ vertical CTD observations. In this study, collocated Argo profiles ($N=1,482$) were used for an **ARGO–GLORYS Reference Consistency Assessment**, confirming consistency between the numerical reanalysis reference and in-situ observations.

### W. Main Limitations
1. Evaluated on a single year (2020).
2. GLORYS reanalysis target is not direct observational ground truth.
3. Operational Argo is assimilated into GLORYS, so it is not an independent ML validation.
4. Non-uniform error profile across depth (peaks at 1.8110 °C at 75 m).
5. Discretized into 15 depths rather than continuous vertical functions.
6. Local spatial window ($3 \times 3$) omits basin-scale teleconnections.
7. Frontend is a decoupled prototype simulation.
8. Decadal climate shifts (IOD/ENSO) remain uncertified.

### X. Future Work
1. Multi-decadal training across 1993–2022.
2. Independent validation against unassimilated delayed-mode experimental floats and gliders.
3. Physics-informed vertical density stability constraints ($\partial \rho / \partial z \ge 0$).
4. Neural Ordinary Differential Equations (Neural ODEs) for continuous vertical profiles.
5. Calibrated uncertainty estimation (conformal prediction).

---

## Part 3: 30 Likely Viva Questions & Model Answers

#### Q1: What is the core problem your project addresses?
**Answer**: "We reconstruct 3D vertical potential temperature profiles across 15 depths down to 1000 meters in the North Indian Ocean using seven daily satellite-derived surface observations, addressing the spatial and temporal sparsity of in-situ Argo floats."

#### Q2: Why can't satellite remote sensing observe subsurface ocean temperatures directly?
**Answer**: "Water strongly absorbs electromagnetic radiation. Infrared sensors only penetrate the top few micrometers (skin layer), and microwave radiometers penetrate roughly one millimeter. Neither can measure subsurface layers directly."

#### Q3: Which seven surface predictors are used, and why?
**Answer**: "OSTIA SST, Copernicus Multi-Observation SSS, DUACS SSH, OSCAR zonal and meridional currents, and CCMP zonal and meridional winds. They capture thermal forcing, freshwater buoyancy, steric thermocline displacement, geostrophic shears, and wind-driven Ekman pumping."

#### Q4: What is GLORYS12V1, and why is it your target instead of raw Argo float measurements?
**Answer**: "GLORYS12V1 is a CMEMS global reanalysis at 1/12° resolution assimilating satellite and in-situ data into the NEMO physical ocean model. It provides continuous daily 3D gridded fields across 366 days ($133\text{M}$ data points), whereas sparse Argo floats only provide ~1,500 profiles across the entire year, which is insufficient for dense spatiotemporal training."

#### Q5: Is GLORYS direct ground truth?
**Answer**: "No. GLORYS is a numerical ocean reanalysis state estimate. It incorporates model physics, assimilation approximations, and forcing errors. Our model learns to reconstruct the GLORYS reanalysis state estimate."

#### Q6: How do you handle points below the ocean floor in shallow waters?
**Answer**: "We enforce GLORYS/ORCA12 model bathymetry. If a canonical depth exceeds the seafloor depth at that coordinate, it is masked as `NaN`. Invalid target NaNs are preserved as masked invalid targets throughout preprocessing, training loss, and evaluation, and are never converted to zero."

#### Q7: What are the exact dates for your train, validation, and test splits?
**Answer**: "Train: Days 0–252 (Jan 1 – Sep 9, 253 days). Purge Buffer 1: Days 253–258 (Sep 10 – Sep 15, 6 days). Validation: Days 259–306 (Sep 16 – Nov 2, 48 days). Purge Buffer 2: Days 307–312 (Nov 3 – Nov 8, 6 days). Test: Days 313–365 (Nov 9 – Dec 31, 53 days)."

#### Q8: Why did you insert 6-day purge buffers between splits?
**Answer**: "Ocean anomalies have an autocorrelation memory of 5–10 days. The 6-day purge buffers break temporal autocorrelation and ensure that our 5-day causal temporal input windows never cross between training, validation, and test distributions."

#### Q9: How was normalization performed to prevent leakage?
**Answer**: "Z-score standard scaling parameters (mean and standard deviation for 7 features and 15 depths) were fitted exclusively on the 253-day training split ($N = 2,871,550$). No validation or test data contaminated normalization."

#### Q10: What is baseline B1, and why is it essential?
**Answer**: "B1 is spatial-depth daily climatology, computed as the historical mean potential temperature per grid cell and depth over the training split. It represents the standard oceanographic reference anchor. Any machine learning model must outperform B1 to prove true predictive skill."

#### Q11: What were the test RMSE results of the reference baselines?
**Answer**: "Day-0 persistence (B0) achieved 1.5220 °C; Day-252 persistence (B0b) achieved 1.7287 °C; and spatial climatology (B1) achieved 1.2582 °C."

#### Q12: How did regularized linear regression (B2) perform?
**Answer**: "Multi-Output Ridge Regression achieved 1.0295 °C test RMSE (18.18% improvement over B1). Its regularization parameter was resolved at $\alpha = 100{,}000$."

#### Q13: What were the results of the decision tree baselines B3 and B4?
**Answer**: "B3 (Multi-Depth Random Forest) achieved 1.0452 °C test RMSE with 13,289,966 decision nodes across 750 trees. B4 (LightGBM) achieved 1.0288 °C test RMSE with 750 boosting trees."

#### Q14: Why did Pointwise MLP (B5) perform worse than linear and tree baselines?
**Answer**: "B5 achieved 1.5524 °C test RMSE. Without spatial patch gradients or temporal recurrence, a unregularized multi-layer feedforward network overfits high-frequency pointwise surface noise across 2.87 million training rows."

#### Q15: What did your ablation studies B6 and B7 demonstrate?
**Answer**: "B6 (Spatial CNN only) achieved 1.2702 °C (-0.95% vs B1), showing spatial context alone without time is indifferent to climatology. B7 (Temporal GRU only) achieved 1.5320 °C (-21.76% vs B1), showing temporal sequence alone without spatial advection fails. Only joint spatiotemporal conditioning in B8 succeeds."

#### Q16: Walk through the exact architecture of your champion model, B8.
**Answer**: "Input: $[B, T=5, C=7, P=3, P=3]$. Time-distributed Conv2D ($7 \to 32 \to 64$) with BatchNorm and AdaptiveAvgPool yields a 64D spatial token per time step. A 2-layer causal GRU (hidden size 128) processes the sequence of 5 tokens. The last hidden state passes through a 128D LayerNorm bottleneck, and a 2-layer MLP decoder ($128 \to 64 \to 15$) outputs the 15-depth temperature profile. Total trainable parameters: 203,791."

#### Q17: What was the primary test result of B8?
**Answer**: "B8 achieved a column-averaged test RMSE of **0.9800 °C**, representing a **22.11% relative error reduction** over daily climatology (B1: 1.2582 °C)."

#### Q18: Is 0.9800 °C the RMSE at every depth?
**Answer**: "No. 0.9800 °C is the unweighted column-averaged mean across all 15 depths. Errors vary substantially with depth: 0.4369 °C at 0 m, peaking at 1.8110 °C at 75 m in the thermocline, and decreasing to 0.5959 °C at 1000 m."

#### Q19: Why does reconstruction error peak in the thermocline (75–100 m)?
**Answer**: "The main thermocline has steep vertical temperature gradients ($\partial T/\partial z$ up to $0.15\ ^\circ\text{C/m}$). Internal waves and eddy displacement cause large vertical isotherm heave. A small vertical prediction displacement produces a large absolute temperature residual."

#### Q20: Why does climatology B1 beat neural models at 1000 m depth?
**Answer**: "At 1000 m, physical temperature variance is extremely low ($\sigma < 0.4\ ^\circ\text{C}$). The static training mean has virtually zero variance, achieving 0.3709 °C RMSE. Neural networks retain small stochastic variance, resulting in 0.5959 °C RMSE."

#### Q21: How did you prove that B8's improvement over B1 is statistically significant?
**Answer**: "We performed a paired 7-day moving block bootstrap with 1,000 resamples over the 53-day test partition. The paired difference $\Delta\text{RMSE}$ yielded a 95% confidence interval of $[-0.3957, -0.1756]\ ^\circ\text{C}$ with $p < 0.001$, proving significant superiority over climatology."

#### Q22: Can you claim B8 is statistically superior to Ridge (B2) or LightGBM (B4)?
**Answer**: "No. While B8 achieves a numerically lower RMSE (0.9800 °C vs 1.0288 °C for B4 and 1.0295 °C for B2), we did not certify a formal paired bootstrap significance test against B2/B4. We strictly report B8 as the numerically best-performing internal architecture."

#### Q23: Why did you perform a thermal-regime diagnostic classification?
**Answer**: "To evaluate how well continuous temperature predictions capture distinct physical water mass regimes (<10 °C, 10–15 °C, 15–20 °C, 20–25 °C, 25–28 °C, $\ge 28$ °C). It serves as a secondary structural diagnostic, not the primary regression training target."

#### Q24: What were the diagnostic classification results?
**Answer**: "B3 Random Forest achieved 80.65% exact accuracy ($\kappa = 0.7636$). B2 Ridge achieved 79.87% ($\kappa = 0.7538$). B5 Pointwise MLP achieved 77.67% ($\kappa = 0.7275$). Climatology B1 achieved 76.23% ($\kappa = 0.7078$)."

#### Q25: What does the '$\pm 1$-bin accuracy' metric tell us?
**Answer**: "Across all supervised ML models, over 99.7% of predictions fell within $\pm 1$ bin of the true regime (99.91% for B2, 99.71% for B3, 99.85% for B5). This indicates that classification errors are predominantly localized near continuous temperature-regime boundaries."

#### Q26: Does $\pm 1$-bin containment prove vertical monotonicity or rule out inversions?
**Answer**: "No. High $\pm 1$-bin containment does not prove vertical profile monotonicity or rule out localized gradient inversions. Vertical stability is formally evaluated using continuous profile gradients."

#### Q27: What was the origin of the 81.19% classification accuracy figure in earlier notes?
**Answer**: "It came from an exploratory hyperparameter tuning candidate (`b3_MLP_128_64_best.pt`, 10,255 parameters). Under the locked canonical hierarchy, B3 is Random Forest (80.65% accuracy) and B5 is Pointwise MLP (77.67% accuracy). The legacy artifact is preserved purely for provenance disclosure."

#### Q28: How does performance differ between the Arabian Sea and Bay of Bengal?
**Answer**: "B8 achieves 0.6775 °C test RMSE in the Bay of Bengal, compared to 1.0907 °C in the Arabian Sea. The Bay of Bengal has massive river runoff creating stable freshwater capping, making upper stratification more predictable. The Arabian Sea has high salinity, intense monsoon upwelling, and vigorous eddies."

#### Q29: What was the purpose and outcome of the ARGO–GLORYS assessment?
**Answer**: "We collocated 1,482 in-situ Argo float profiles with GLORYS within $\pm 0.25^\circ$ and $\pm 12\text{ hours}$. Because GLORYS assimilates operational Argo floats, this assesses reanalysis reference consistency, not independent ground-truth validation of our ML model."

#### Q30: What is the single biggest limitation of your project, and how would you address it in the future?
**Answer**: "The single-year temporal boundary (2020). While it captures a complete seasonal cycle, it cannot certify generalization across multi-year climate modes like the Indian Ocean Dipole or ENSO. Future work would scale training across 1993–2022 using distributed GPU clusters and validate against unassimilated delayed-mode research floats."

---

## Part 4: Key Numerical Checksheet for Viva

| Metric / Parameter | Certified Value | Context / Meaning |
|---|---|---|
| Domain Bounds | 5°N–30°N, 45°E–105°E | North Indian Ocean basin |
| Spatial Grid Size | 101 × 241 = 24,341 points | 0.25° equirectangular resolution |
| Valid Ocean Columns / Day | 11,350 columns | Masked from 24,341 total grid nodes |
| Total Days in Dataset | 366 days | Full-year 2020 leap year |
| Train Partition | Days 0–252 (253 days) | $N = 2,871,550$ training samples |
| Purge Buffer Width | 6 days (Days 253–258 & 307–312) | Exceeds $T=5$ temporal window |
| Validation Partition | Days 259–306 (48 days) | $N = 544,800$ validation samples |
| Test Partition | Days 313–365 (53 days) | $N = 601,550$ columns ($8,017,734$ valid depth pts) |
| Canonical Depths | 15 levels (0 to 1000 m) | Standard oceanographic depth grid |
| Surface Predictors | 7 satellite variables | SST, SSS, SSH, Current U/V, Wind U/V |
| B1 Climatology Test RMSE | 1.2582 °C | Reference baseline anchor |
| B2 Ridge Test RMSE | 1.0295 °C | 120 coefficients ($\alpha=100{,}000$) |
| B3 Random Forest Test RMSE | 1.0452 °C | 13,289,966 decision nodes across 750 Random Forest trees |
| B4 LightGBM Test RMSE | 1.0288 °C | 750 boosted trees |
| B5 Pointwise MLP Test RMSE | 1.5524 °C | 26,767 trainable parameters |
| B6 Spatial CNN Test RMSE | 1.2702 °C | 30,991 trainable parameters |
| B7 Temporal GRU Test RMSE | 1.5320 °C | 44,111 trainable parameters |
| **B8 Spatiotemporal Test RMSE** | **0.9800 °C** | **Champion model (203,791 trainable parameters)** |
| B8 vs B1 Improvement | **+22.11%** ($-0.2782^\circ\text{C}$) | Paired 95% CI: $[-0.3957, -0.1756]^\circ\text{C}$ ($p < 0.001$) |
| B8 Depth 0 m Test RMSE | 0.4369 °C | Strong direct SST constraint |
| B8 Depth 75 m Test RMSE | 1.8110 °C | Main thermocline gradient peak error |
| B8 Depth 1000 m Test RMSE | 0.5959 °C | Abyssal low-variance water |
| Arabian Sea Test RMSE | 1.0907 °C | High-salinity upwelling basin |
| Bay of Bengal Test RMSE | 0.6775 °C | Stratified freshwater plume basin |
| Full Domain Cosine RMSE | 0.9642 °C | Cosine-latitude area-weighted |
| B3 Diagnostic Accuracy | 80.65% ($\kappa = 0.7636$) | Highest discrete regime accuracy |
| B2 Diagnostic Accuracy | 79.87% ($\kappa = 0.7538$) | Highest $\pm 1$-bin containment (99.91%) |
| Automated Test Suite | 107 passed, 0 failed, 0 skipped | 100% pass rate in pytest |
