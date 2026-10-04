# Final Methodology Document: Scientific Protocol & Architecture Specification

**Project**: Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature  
**Domain**: North Indian Ocean (5°N–30°N, 45°E–105°E)  
**Dataset**: Full-Year 2020 Certified Production Partition  

---

## 1. Input Space: Seven Surface Predictors
The framework consumes seven multi-satellite daily ocean surface observations:
1. **Sea Surface Temperature (SST)**: UK Met Office OSTIA operational foundation temperature (°C).
2. **Sea Surface Salinity (SSS)**: NASA SMAP Level 3 sea surface salinity (Practical Salinity Units, PSU).
3. **Sea Surface Height (SSH)**: Copernicus Marine DUACS multi-mission gridded sea level anomaly and absolute dynamic topography (m).
4. **Zonal Surface Current (Current U)**: NOAA OSCAR geostrophic and wind-driven surface velocity component (m/s).
5. **Meridional Surface Current (Current V)**: NOAA OSCAR meridional surface velocity component (m/s).
6. **Zonal 10m Wind Stress (Wind U)**: Remote Sensing Systems CCMP V3.1 cross-calibrated surface wind vector (m/s).
7. **Meridional 10m Wind Stress (Wind V)**: Remote Sensing Systems CCMP V3.1 cross-calibrated surface wind vector (m/s).

All surface fields are regridded onto a uniform 0.25° grid across $101 \times 241$ coordinates. Coastal regridding uses normalized bilinear interpolation to prevent terrestrial zero values from bleeding into nearshore waters.

---

## 2. Target Space: 15 Canonical Oceanographic Depths
The target field is Copernicus Marine GLORYS12V1 daily potential temperature ($\theta_o$) interpolated from native model levels to 15 canonical oceanographic depths:
$$z \in \{0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\}\text{ meters}$$
Interpolation is performed using Piecewise Cubic Hermite Interpolating Polynomials (PCHIP) to guarantee monotonic stratification and prevent artificial temperature inversions. Sub-seafloor points are strictly masked as invalid (NaN) using GEBCO 2024 bathymetry; they are never converted to physical 0 °C.

---

## 3. Four-Way Unified Masking Protocol
Every sample $(t, z, \text{lat}, \text{lon})$ must satisfy a 4-way composite boolean validity mask:
$$\mathbf{M}(t, z, \text{lat}, \text{lon}) = \mathbf{M}_{\text{geo}} \land \mathbf{M}_{\text{surf}} \land \mathbf{M}_{\text{targ}} \land \mathbf{M}_{\text{depth}}$$
1. **$\mathbf{M}_{\text{geo}}$ (Geographic Ocean Mask)**: $1$ over open ocean, $0$ over continental landmasses and islands.
2. **$\mathbf{M}_{\text{surf}}$ (Surface Observation Mask)**: $1$ if all 7 surface predictor values are non-NaN and physically valid.
3. **$\mathbf{M}_{\text{targ}}$ (Target Validity Mask)**: $1$ if GLORYS target field is present and uncorrupted.
4. **$\mathbf{M}_{\text{depth}}$ (Bathymetric Depth Mask)**: $1$ if target depth $z \le \text{depth}_{\text{GEBCO}}(\text{lat}, \text{lon})$, $0$ if below the ocean floor.

Invalid target entries are excluded from loss computation and metrics:
$$\mathcal{L}_{\text{masked}} = \frac{\sum_{i, k} M_{i,k} \cdot (\hat{Y}_{i,k} - Y_{i,k})^2}{\sum_{i, k} M_{i,k} + \epsilon}$$

---

## 4. Train-Only Normalization & Checksum Verification
To prevent normalization leakage from validation or test splits:
- Scaler statistics (mean $\mu_c$ and standard deviation $\sigma_c$) are computed **strictly from the training partition** (Days 0–252, $N = 2,871,550$).
- Feature standardization:
  $$x_c^{\text{norm}} = \frac{x_c - \mu_c^{\text{train}}}{\sigma_c^{\text{train}}}$$
- Scaler parameters are persisted with a deterministic SHA-256 checksum to guarantee reproducible deployment.

---

## 5. Chronological Partitioning & Purge Buffer Logic
The 366 days of 2020 are partitioned chronologically:
- **TRAIN**: Days 0–252 (253 days: Jan 01 – Sep 09, 2020; $N = 2,871,550$)
- **PURGE 1**: Days 253–258 (6 days: Sep 10 – Sep 15, 2020; completely discarded)
- **VAL**: Days 259–306 (48 days: Sep 16 – Nov 02, 2020; $N = 544,800$)
- **PURGE 2**: Days 307–312 (6 days: Nov 03 – Nov 08, 2020; completely discarded)
- **TEST**: Days 313–365 (53 days: Nov 09 – Dec 31, 2020; $N = 601,550$; frozen)

**Purge Buffer Rationale**: Ocean thermodynamic autocorrelation exhibits an e-folding decorrelation scale of 4–5 days. The 6-day purge buffer ensures that no transient physical eddy state observed at the end of training or validation leaks into the subsequent evaluation partition.

---

## 6. Spatiotemporal Context Construction
- **Temporal Context**: A 5-day causal temporal window:
  $$\mathbf{X}_{\text{time}} = [t-4, t-3, t-2, t-1, t]$$
  Windows strictly look backward. At partition boundaries (e.g., Day 259 or Day 313), historical inputs are clamped at the partition start to prevent crossing purge intervals.
- **Spatial Context**: A $3 \times 3$ grid cell neighborhood centered on each evaluation point:
  $$P = 3 \implies 3 \times 3 \text{ patch} \approx 75\text{ km} \times 75\text{ km footprint}$$
  Captures mesoscale horizontal curvature, eddy vorticity, and thermal fronts.
- **Composite Tensor**: Formed into a 5D cube $\mathbf{X}_{\text{cube}} \in \mathbb{R}^{B \times 5 \times 7 \times 3 \times 3}$.

---

## 7. Champion Architecture: B8 Spatiotemporal Embedding Model

```
Input Tensor: [Batch, T=5, C=7, P=3, P=3]
  │
  ├──► [Spatial Conv2D Encoder] (Time-Distributed)
  │      Conv2D(7 -> 32, kernel=3, padding=1)
  │      BatchNorm2d(32) + ReLU
  │      Conv2D(32 -> 64, kernel=3, padding=1)
  │      BatchNorm2d(64) + ReLU
  │      AdaptiveAvgPool2d((1, 1))
  │      Output: [Batch, T=5, 64]
  │
  ├──► [Temporal GRU Encoder]
  │      2-Layer Causal GRU(input_size=64, hidden_size=128, batch_first=True)
  │      Output hidden states: h_1, h_2, h_3, h_4, h_5
  │      Terminal state: h_5 ∈ [Batch, 128]
  │
  ├──► [Latent Representation Bottleneck]
  │      LayerNorm(128)
  │      Output: Latent Vector z ∈ [Batch, 128]
  │
  └──► [Depth-Wise Decoder MLP]
         Linear(128 -> 64) + ReLU + Dropout(0.1)
         Linear(64 -> 15)
         Output: Reconstructed Profile Ŷ ∈ [Batch, 15]
```

### Parameter Breakdown
- **Spatial Encoder (Conv2D + BN)**: 20,864 parameters
- **Temporal GRU (2 layers, 64→128, 128→128)**: 173,056 parameters
- **Latent LayerNorm (128)**: 256 parameters
- **Decoder MLP (128→64→15)**: 9,615 parameters
- **Total Parameters**: **203,791** (verified by direct parameter tensor summation).

---

## 8. Evaluation & Bootstrap Protocol
1. **Primary Metric**: Column-averaged test RMSE across all 15 canonical depths:
   $$\text{RMSE}_{\text{col}} = \frac{1}{15} \sum_{k=1}^{15} \text{RMSE}_k = 0.9800\ ^\circ\text{C}$$
2. **Regional Area Weighting**: Cosine-latitude weighting accounts for meridional cell area convergence:
   $$w_i = \cos(\text{lat}_i)$$
3. **Statistical Resampling (7-Day Block Bootstrap)**:
   - Evaluates statistical significance while preserving temporal autocorrelation.
   - Number of resamples: $B = 1,000$.
   - Resampling unit: 7-day contiguous temporal blocks.
   - Paired difference test: $\Delta\text{RMSE}^{(b)} = \text{RMSE}_{\text{B8}}^{(b)} - \text{RMSE}_{\text{B1}}^{(b)}$.
   - 95% Confidence Interval: $[q_{0.025}, q_{0.975}] = [-0.3957, -0.1756]\ ^\circ\text{C}$ (statistically significant at $p < 0.001$).
