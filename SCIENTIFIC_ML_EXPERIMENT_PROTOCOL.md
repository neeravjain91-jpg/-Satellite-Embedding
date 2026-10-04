# SCIENTIFIC ML EXPERIMENT PROTOCOL

**Project:** Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature from Daily Surface Observations  
**Domain:** North Indian Ocean (5–30°N, 45–105°E)  
**Year:** 2020  
**Author:** Independent Scientific ML Architect  
**Status:** Pre-implementation design — do not execute  

---

## 1. FINAL TARGET SEMANTICS

### 1.1 The Target Reality

GLORYS12V1 thetao is a **reanalysis product**. It is the output of a NEMO model configuration constrained by data assimilation (SEEK filter, 7-day cycle) of satellite SST, satellite SSH, sea ice concentration, and in-situ T/S profiles (Argo, XBT, CTD, moorings). It is NOT a direct measurement of ocean temperature.

### 1.2 Required Wording — Abstract/Introduction

> *"The training target is the daily potential temperature field from the GLORYS12V1 reanalysis (CMEMS GLOBAL_REANALYSIS_PHY_001_030). GLORYS provides global, gap-free, daily 3D temperature estimates at 1/12° resolution on 50 vertical levels, obtained by assimilating satellite and in-situ observations into a NEMO-based ocean model. We use GLORYS as a **reference field** — a physically consistent, observationally constrained estimate of the subsurface state — recognizing that it is a model product with documented uncertainties (QUID RMSE vs in-situ: ~0.6–1.0°C in the upper 300 m at this latitude). The objective is to approximate this reference field from surface satellite fields alone. Independent evaluation against withheld Argo profiles (Section X) provides an upper bound on the method's ability to reproduce the observed subsurface state."*

### 1.3 Required Wording — Results/Discussion

> *"All RMSE, MAE, and R² values reported in this study quantify the discrepancy between model predictions and the GLORYS reanalysis target. These metrics reflect the model's ability to **emulate the reanalysis** from surface fields, not its ability to reconstruct in-situ temperature. Argo-based evaluation (Section X) provides the latter estimate and should be consulted before making physical interpretations."*

### 1.4 Prohibited Wording

| Do NOT Say | Instead Say |
|------------|-------------|
| "observed temperature" | "GLORYS reference temperature" |
| "ground truth" | "reanalysis estimate" |
| "reconstruction" (without qualification) | "GLORYS emulation" or "reanalysis approximation" |
| "validated against Argo" | "compared with withheld Argo profiles (note: GLORYS assimilates Argo)" |
| "predicted ocean temperature" | "predicted temperature relative to GLORYS" |

---

## 2. TEMPORAL SPLIT — DESIGN

### 2.1 Available Timeline

Year 2020 = 366 days (leap year).

| Partition | Days | Indices (0-based) | Calendar (approx) |
|-----------|------|-------------------|-------------------|
| Train | ~253 | 0–252 | Jan 1 – Sep 9 |
| Purg | ~6 | 253–258 | Sep 10–15 |
| Validation | ~48 | 259–306 | Sep 16 – Nov 2 |
| Purg | ~6 | 307–312 | Nov 3–8 |
| Test | ~53 | 313–365 | Nov 9 – Dec 31 |

### 2.2 Purge Evaluation

**Constraint:** GLORYS assimilation cycle is 7 days. Each analysis at day T "sees" observations from approximately [T−3, T+3]. The effective contamination window is ±3 days at each partition boundary.

**7-day purge (recommended):** Remove 3 days from end of train + 3 days from start of validation → 6 days gap. Add 1 extra day for safety → 7 days. Same at val/test boundary. Total lost: 14 days (7 at each boundary).

**14-day purge (alternative):** Remove 7 days from each side of each boundary. Total lost: 28 days.

| Factor | 7-Day Purge | 14-Day Purge |
|--------|-------------|--------------|
| Assimilation window | ±3 days covered with margin | ±3 days covered with large margin |
| Training days lost | 6 (3+3) | 14 (7+7) |
| Validation days lost | 6 (3+3) | 14 (7+7) |
| Test days lost | 6 (3+3) | 14 (7+7) |
| Remaining training | 253 | 245 |
| Remaining validation | 48 | 40 |
| Remaining test | 53 | 45 |
| Autocorrelation at depth | Still present (>7 days) | Still present (>14 days at 300m) |
| **Verdict** | **Sufficient for assimilation leakage** | Overkill — loses data without eliminating physical autocorrelation |

**Recommendation: 7-day purge, documented with caveat.**

> *"A 7-day buffer is applied between partitions to prevent direct contamination from the GLORYS 7-day assimilation cycle. Thermal decorrelation timescales at depth (20–40 days at 300 m) exceed this buffer; temporal autocorrelation is a property of the physical system, not a leakage artifact, and is handled by the baselines appropriately. The effective sample size for uncertainty estimation is addressed via block bootstrapping (Section 7)."*

### 2.3 Exact Split Logic

```python
# 0-based indices for 366-day year (2020)
# GLORYS assimilation window: ±3 days = 6 days + 1 safety = 7-day purge

train_indices    = range(0,     253)       # days 0–252   (253 days)
# purge: days 253–258 (6 days: 3+1+2? use 256-258 = 3 + 3 from val side)
# Exact: train ends at 252, val starts at 259 → gap = 253–258 (6 days)
val_indices      = range(259,   307)       # days 259–306 (48 days)
# purge: days 307–312 (6 days)
test_indices     = range(313,   366)       # days 313–365 (53 days)
```

### 2.4 Audit Requirement

Compute the maximum correlation between:
- `thetao[t, d, lat, lon]` at the last training day and `thetao[t+lag, d, lat, lon]` at the first validation day, for lag = 0..14 days.

Report this correlation in the supplement. If ρ(3) > 0.05 for any (d, lat, lon) AFTER the purge, the purge must be extended.

---

## 3. TEMPORAL AUTOCORRELATION

### 3.1 Purpose

Quantify how quickly temperature at each depth becomes statistically independent of its past. This:
- Justifies the purge choice
- Determines effective sample size for uncertainty estimation
- Identifies depths where temporal persistence dominates (high autocorrelation → naive persistence baseline is strong)

### 3.2 Design

**Data:** GLORYS thetao on the training partition (days 0–252), at representative locations per region.

**Locations (one per region, plus one equatorial):**

| Label | Region | Lat | Lon | Water depth (m) |
|-------|--------|-----|-----|-----------------|
| AS1 | Arabian Sea (central) | 18°N | 64°E | >3000 |
| AS2 | Arabian Sea (west) | 22°N | 68°E | >2000 |
| AS3 | Arabian Sea (south) | 12°N | 72°E | >3000 |
| BB1 | Bay of Bengal (central) | 15°N | 88°E | >3000 |
| BB2 | Bay of Bengal (north) | 20°N | 90°E | ~1500 |
| BB3 | Bay of Bengal (south) | 8°N | 85°E | >3000 |

**Depths:** 50 m, 100 m, 300 m (spanning mixed layer, thermocline, deep).

**Statistic:** Lag autocorrelation function (ACF)

```
ACF(lag) = corr( thetao[t], thetao[t+lag] )   over t in training set
```

**Lag range:** 0 to 60 days (step = 1 day).

**Thresholds:**
- `τ_0.2` = lag at which ACF first drops below 0.2
- `τ_e` = lag at which ACF first drops below 1/e ≈ 0.37
- `τ_0.05` = lag at which ACF first drops below 0.05

### 3.3 Expected Output Table

| Region | Depth (m) | τ_0.2 (days) | τ_e (days) | τ_0.05 (days) |
|--------|-----------|--------------|------------|---------------|
| Arabian Sea | 50 | — | — | — |
| Arabian Sea | 100 | — | — | — |
| Arabian Sea | 300 | — | — | — |
| Bay of Bengal | 50 | — | — | — |
| Bay of Bengal | 100 | — | — | — |
| Bay of Bengal | 300 | — | — | — |

### 3.4 Impact on Purge Decision

| If τ_0.2 at 300 m is ... | Then ... |
|--------------------------|----------|
| ≤ 7 days | 7-day purge achieves both assimilation and autocorrelation independence |
| 8–14 days | 7-day purge removes assimilation leakage only; physical autocorrelation remains across partition boundary. Document: "Statistical independence is not achieved at depth; block bootstrapping is used for confidence intervals." |
| > 14 days | The system has strong memory. Consider using a gap or acknowledging that train and validation decorrelate slowly. 14-day purge still insufficient — cannot practically eliminate. Accept and document. |

**Regardless of outcome:** The 7-day purge is retained. No practical purge can eliminate 30-day ocean memory. The 7-day purge addresses the **assimilation cycle contamination**, which is the only true leakage.

---

## 4. ARGO EVALUATION

### 4.1 The Fundamental Problem

GLORYS12V1 assimilates Argo T/S profiles. Any Argo profile used in the 7-day assimilation window around the GLORYS analysis date has directly influenced the target. Evaluating an ML model (trained on GLORYS) against those same Argo profiles is **circular**.

### 4.2 Eligibility Criteria

An Argo profile is **eligible** for ML evaluation if and only if:

1. **Temporal withholding:** The profile date falls in the **test partition** (days 313–365, i.e., November 9 – December 31, 2020). GLORYS values at test time are model forecasts (the assimilation cycle has run forward), so the GLORYS value at a test-day grid cell is **not directly influenced** by test-day Argo profiles. However, GLORYS's background state at test time is the result of the previous assimilation cycle, which did use earlier Argo profiles.

2. **Assimilation non-overlap:** We cannot identify exact per-profile assimilation status. Instead, we use a **conservative buffer**: any Argo profile within 7 days of a GLORYS analysis is considered "potentially assimilated." Since our test partition begins 7+ days after the last training/validation analysis, the **direct** influence is mitigated.

3. **Result:** Test-period Argo profiles offer **partial independence** — GLORYS did not assimilate them in the analysis for those days, but the model's prior state was conditioned on earlier (non-test) Argo profiles.

### 4.3 Protocol

#### 4.3.1 Data Source
- Argo delayed-mode (DM) profiles from the Argo Global Data Assembly Centre (GDAC)
- Quality-controlled (QC flags = 1 or 2)
- Preferred: Roemmich-Gilson monthly gridded Argo product as a secondary comparison

#### 4.3.2 Temporal Matching
- Match Argo profile date to nearest GLORYS daily field
- Tolerance: ±1 day
- If >1 day offset, discard
- Only use profiles with date in test partition (days 313–365)

#### 4.3.3 Spatial Matching
- Nearest-neighbor to GLORYS 1/4° grid cell (maximum distance 0.25°)
- If the same GLORYS cell has >1 Argo profile on the same day, average the profiles before computing metrics
- Exclude profiles in water depth < 1000 m (coastal Argo is rare but quality degrades)

#### 4.3.4 Vertical Interpolation
- Argo profiles: native pressure levels → convert to depth using standard pressure-to-depth conversion
- GLORYS: native 50 z* levels → linear interpolation to Argo depths
- Maximum extrapolation: none — discard Argo measurements shallower than GLORYS layer 1 or deeper than layer 50

#### 4.3.5 Quality Control (Argo DM)
- Retain only profiles with QC flag = 1 (good data)
- Remove profiles where the temperature gradient between consecutive levels exceeds 5°C / 10 dbar (spike test)
- Remove profiles in the top 10 m if they show suspicious diurnal warming (T > 35°C in this latitude band)
- Remove profiles with fewer than 5 valid levels

#### 4.3.6 Metrics

For each matched (Argo_profile, GLORYS_cell, day, depth):

```
error_GLORYS = GLORYS_thetao[cell, day, depth] - Argo_T[profile, depth]
error_ML     = ML_prediction[cell, day, depth] - Argo_T[profile, depth]
```

Compute:

| Metric | Definition | Interpretation |
|--------|------------|---------------|
| RMSE_GLORYS | sqrt(mean(error_GLORYS²)) | Baseline — GLORYS quality vs Argo |
| RMSE_ML | sqrt(mean(error_ML²)) | ML model quality vs Argo |
| Bias_GLORYS | mean(error_GLORYS) | GLORYS systematic offset |
| Bias_ML | mean(error_ML) | ML systematic offset |
| σ_GLORYS | std(error_GLORYS) | GLORYS random error |
| σ_ML | std(error_ML) | ML random error |
| ΔRMSE | RMSE_ML − RMSE_GLORYS | Positive = model degrades; negative = model improves |

Report per depth bin (0–50, 50–200, 200–500, 500–1000 m) and per region (Arabian Sea, Bay of Bengal).

#### 4.3.7 Uncertainty

- Bootstrap 95% CI on all Argo metrics (1000 resamples of profiles with replacement)
- Report: `RMSE_ML = X°C [95% CI: X_low, X_high]`
- Report effective degrees of freedom accounting for spatial autocorrelation (if >1 profile per grid cell, the profiles are not independent)

### 4.4 Prohibited Claims About ARGO

| Do NOT Claim | Instead Say |
|-------------|-------------|
| "Validated against independent Argo observations" | "Compared with test-period Argo profiles, which were not directly assimilated into the GLORYS analysis for those days but may reflect the background state that incorporated earlier Argo data." |
| "Argo validation demonstrates reconstruction skill" | "The ML–Argo discrepancy is comparable to the GLORYS–Argo discrepancy, indicating that the model does not substantially degrade the reanalysis quality relative to in-situ observations." |
| "RMSE against Argo of X°C" (isolated) | "RMSE against Argo is X°C (GLORYS RMSE against Argo is Y°C)." Always paired. |

---

## 5. BASELINE EXPERIMENT DESIGN

### 5.1 Hierarchy

| Level | Model | Input | Output | Parameters | Role |
|-------|-------|-------|--------|------------|------|
| **B0** | Persistence (day 0) | None | thetao all days | 0 | Upper bound — measures memory of initial state |
| **B0b** | Persistence (day 252) | None | thetao test days | 0 | Measures memory across gap |
| **B1** | Climatology | None | per-depth mean | 0 per point | Lower bound — no temporal skill |
| **B2** | Ridge Regression | 7 features | thetao per depth | 8 per depth | Linear mapping baseline |
| **B3** | Random Forest | 7 features | thetao per depth | Ensemble | Nonlinear, no extrapolation |
| **B4** | LightGBM / XGB | 7 features | thetao per depth | Boosting | Gradient boosting |
| **B5** | Pointwise MLP | 7 features | thetao per depth | ~2.6K per depth | Neural, no spatial context |
| **B6** | Spatial CNN | 7×P×P patch | thetao at center | ~K×P²×7 | Adds spatial context locally |
| **B7** | Temporal model | 7×T_window | thetao last step | depends | Adds temporal context |
| **B8** | Embedding model | 7×H×W×T | thetao full field | depends | Full spatiotemporal (this project) |

### 5.2 Fixed Across All Models (Non-Negotiable)

| Component | Fixed Value | Rationale |
|-----------|-------------|-----------|
| Training partition | days 0–252 | Identical for all models |
| Validation partition | days 259–306 | Hyperparameter selection only |
| Test partition | days 313–365 | Touched exactly once |
| Surface features | sst, sss, ssh, current_u, current_v, wind_u, wind_v | All 7, exactly as regridded and masked |
| Normalization | Training-set μ/σ per feature, per depth | Same transformation applied to all |
| Evaluation mask | Unified intersection of all 4 masks | Same valid points for all models |
| Grid resolution | 1/4° (101×241) | No model uses a different grid |
| Metric computation | Single shared Python module | Eliminates implementation differences |
| Random seed | 42 for all stochastic processes | Reproducibility |
| Depth strategy | Independent per-depth models (Option A) | No depth feature shared across depths |

### 5.3 Allowed to Vary

| Component | Variation | Documentation required |
|-----------|-----------|----------------------|
| Hyperparameters | Search space per model | Full trial log (Section 6.1.8) |
| Architecture | By definition | Full architecture description |
| Training hardware | CPU/GPU/TPU | Wall time report |
| Training iterations | As needed | Convergence curves |

### 5.4 Prohibited Advantages

- No model may use **future information** relative to its prediction time (temporal leakage)
- No model may use **target information** during feature construction (target leakage)
- No model may train on a different set of valid points than other models (mask leakage)
- No model may access the test partition during training, validation, or checkpoint selection
- No model may use additional features beyond the specified 7 (except the embedding model's spatial/temporal structure, which uses the same 7 features arranged as a field)

### 5.5 Persistence Baseline Design (B0)

**B0 (day 0 persistence):**

```
prediction_B0[t, d, lat, lon] = GLORYS_thetao[0, d, lat, lon]
```

for all t in test partition. This is the temperature field at Jan 1, 2020, propagated unchanged.

**B0b (end-of-training persistence):**

```
prediction_B0b[t, d, lat, lon] = GLORYS_thetao[252, d, lat, lon]
```

for all t in test partition. Measures how much information from the last training day survives the validation+purge gap (~60 days) into the test period.

### 5.6 Climatology Baseline Design (B1)

```
climatology[d, lat, lon] = mean_t( GLORYS_thetao[t, d, lat, lon] over t in train_indices )
```

A single number per (depth, location). All R² calculations use this as the reference.

---

## 6. LEAKAGE AUDIT FOR ML

Each check must be automated as an assertion in the pipeline. If any fails, the experiment is invalid and must be restarted from the point of failure.

### 6.1 Mandatory Checks

#### 6.1.1 Feature Normalization

- **Check:** μ_feat, σ_feat computed from training partition only
- **Assert:** `all(t in train_indices for t in stats_computation_indices)`
- **Also:** No feature has NaN or Inf after normalization
- **Also:** Validation/test features transformed using training μ/σ, never re-estimated

#### 6.1.2 Target Normalization

- **Check:** μ_target[d], σ_target[d] computed from training partition only, using only valid (unmasked) points
- **Assert:** `count_valid(training_targets_at_depth_d) > 0` for each depth
- **Traceability:** Save target normalization stats to JSON with SHA256 checksum

#### 6.1.3 Temporal Leakage

- **Check:** No model trains on any day index ≥ 253 (the start of the purge window)
- **Assert:** `max(training_time_indices) == 252`
- **Also:** Any lagged features constructed by temporal models must not reach into the validation or test partitions
- **Also:** OSCAR interpolation buffers must be verified to not use future data (if OSCAR is 5-day and interpolated to daily, ensure the interpolation at day t uses data from [t−2, t+2] which must all fall within the same partition)

#### 6.1.4 Spatial Leakage

- **Check:** No station-based or region-based split
- **Document:** All models see all locations in training. This is intentional — spatial leakage across locations within the same time step is only relevant for spatial models (CNN, embedding). For pointwise models, it is irrelevant.

#### 6.1.5 Overlap of Repeated Fields

- **Check:** No feature at time t contains information from time > t
- **Special case — SSS daily product:** If the L4 SSS OI product has a temporal smoothing window (e.g., 3-day), verify that the window for day t does not include validation or test days

#### 6.1.6 Persistence Baselines

- **Check:** B0 uses thetao from day 0 (within training). B0b uses day 252 (last training day).
- **Assert:** Both source days ≤ 252

#### 6.1.7 Climatology Construction

- **Check:** Climatology computed from training partition only
- **Assert:** `climatology_source_indices ⊆ train_indices`

#### 6.1.8 Hyperparameter Selection

- **Check:** Hyperparameters chosen based on validation metric — never test
- **Assert:** Test data is NOT loaded during any hyperparameter search
- **Report:** Full trial log: for each hyperparameter configuration, the validation RMSE per depth. Minimum 30 trials per model type.

#### 6.1.9 Model Checkpoint Selection

- **Check:** For MLP and neural models, the best epoch is selected by minimum validation loss
- **Assert:** `best_epoch = argmin(val_loss_epochs)` — test loss never computed during training
- **Also:** Early stopping patience = 20 epochs, with a maximum of 200 epochs

---

## 7. METRIC DESIGN

### 7.1 Primary Metrics

Let:
- `ŷ[t, d, lat, lon]` = model prediction (denormalized, physical units °C)
- `y[t, d, lat, lon]` = GLORYS target (physical units °C)
- `M[t, d, lat, lon]` = evaluation mask (1 = valid for evaluation)
- `N_d` = number of valid points at depth `d` = sum(M over t, lat, lon)

#### RMSE_d (Depth-Specific RMSE)

```
RMSE_d = sqrt( (1/N_d) * Σ_{M=1} (ŷ − y)² )
```

Units: °C. Report to 3 decimal places.

#### MAE_d (Depth-Specific Mean Absolute Error)

```
MAE_d = (1/N_d) * Σ_{M=1} |ŷ − y|
```

Units: °C. Less sensitive to outliers than RMSE.

#### Bias_d (Depth-Specific Mean Bias)

```
Bias_d = (1/N_d) * Σ_{M=1} (ŷ − y)
```

Units: °C. Positive = model overestimates. Report even if small — a consistently signed bias indicates a systematic error.

#### Pearson r_d (Depth-Specific Correlation)

```
r_d = corr( ŷ[M=1], y[M=1] )
```

Dimensionless, [−1, 1]. Measures pattern agreement, not magnitude.

#### R²_d (Depth-Specific Coefficient of Determination, Relative to Climatology)

```
SS_res  = Σ_{M=1} (ŷ − y)²
SS_clim = Σ_{M=1} (y_clim[d,lat,lon] − y)²
R²_d = 1 − SS_res / SS_clim
```

Where `y_clim[d, lat, lon]` = training-mean climatology (B1).

**Interpretation:**
- R²_d = 1.0 → perfect
- R²_d = 0.0 → equivalent to climatology
- R²_d < 0.0 → **worse than climatology** → this model at this depth is harmful

**Critical rule:** R² is NEVER computed relative to the global mean. Only the per-location-per-depth climatology.

### 7.2 Aggregation Across Depths

#### Unweighted Mean

```
RMSE_mean = (1/15) * Σ_{d=0}^{14} RMSE_d
MAE_mean  = (1/15) * Σ_{d=0}^{14} MAE_d
```

#### Weighted Mean (by Sample Count)

```
RMSE_weighted = Σ_d (N_d * RMSE_d) / Σ_d N_d
MAE_weighted  = Σ_d (N_d * MAE_d) / Σ_d N_d
```

**Mandatory:** Report BOTH weighted and unweighted. If they differ by > 10%, shallow depths (which have more data) dominate the weighted metrics, and the unweighted may better represent performance across the water column.

### 7.3 Regional Metrics

Compute all per-depth metrics separately for:

| Region | Lat Range | Lon Range | Area (approx) | Notes |
|--------|-----------|-----------|---------------|-------|
| Full Domain | 5–30°N | 45–105°E | full grid | Primary evaluation |
| Arabian Sea | 5–25°N | 45–77°E | ~10⁶ km² | High salinity, seasonal upwelling |
| Bay of Bengal | 5–25°N | 77–100°E | ~10⁶ km² | Low salinity, barrier layer |

**Critical rule:** The Arabian Sea and Bay of Bengal regions must be evaluated with area weighting, NOT simple averaging across grid cells. Use cosine(latitude) weights.

### 7.4 Area Weighting

Grid cells at latitude `lat` have area proportional to `cos(lat * π/180)`. All regional spatial averages must use this weighting:

```
w(lat) = cos(lat × π/180) / Σ cos(lat × π/180)
```

For a regional mean metric at depth `d`:

```
Metric_region[d] = Σ_{lat,lon in region} w(lat) × metric_point[lat, lon, d]
```

This ensures the northern Arabian Sea (at 25°N) does not dominate the average relative to the southern region (at 5°N).

### 7.5 Seasonal Analysis

Compute test-partition metrics in two seasonal bins (since test runs Nov–Dec, this is limited):

| Season | Months | Days in test | Notes |
|--------|--------|-------------|-------|
| Late fall | Nov | 313–342 (~30 days) | Post-monsoon, well-mixed |
| Early winter | Dec | 343–365 (~23 days) | Cooling, deepening mixed layer |

Compute per-depth RMSE for each sub-period. If the model performs much better in one season, this reveals a seasonal bias.

### 7.6 Uncertainty on All Metrics

- **Bootstrap:** Resample time steps (with replacement, 1000 iterations) within the test partition. Compute metrics on each resample. Report 95% percentile CI.
- **Block bootstrap:** Because of temporal autocorrelation (Section 3), use block length = τ_0.2 (the decorrelation lag from Section 3) for the resampling block size.
- **Report format:** `RMSE_d = X.XXX°C [95% CI: X.XXX, X.XXX]`

---

## 8. ABLATION DESIGN

### 8.1 Purpose

Determine which features and architectural components contribute to performance. The goal is to answer:
- Is the embedding model learning useful spatiotemporal structure, or is it dominated by a single feature (SST)?
- How much does each surface variable contribute?

### 8.2 Feature Ablations

Apply to the **pointwise MLP (B5)** and the **embedding model (B8)**. For each ablation, retrain from scratch.

| Ablation | Removed features | Expected effect | Interpretation |
|----------|------------------|-----------------|----------------|
| **A1** | SST | Remove sst | Should increase RMSE at all depths, especially surface. If RMSE unchanged, SST is informationally redundant. |
| **A2** | SSH | Remove ssh | Should increase RMSE at depth (SSH correlates with thermocline depth). If unchanged, SSH provides no subsurface information. |
| **A3** | SSS | Remove sss | Should increase RMSE in Bay of Bengal at 20–100 m (barrier layer). If unchanged, SSS resolution is too coarse. |
| **A4** | Currents | Remove current_u, current_v | Should increase RMSE in eddy-rich regions. |
| **A5** | Wind | Remove wind_u, wind_v | Should increase RMSE in mixed layer (wind drives mixing). |
| **A6** | All except SST | Only sst | Measures how much information the other 6 features add beyond SST alone. If RMSE increases by < 10%, SST dominates. |
| **A7** | All except SSH | Only ssh | Measures SSH-only information content for subsurface. |
| **A8** | All surface | None | Full feature set — baseline for comparison. |

### 8.3 Context Ablations

Apply to the **embedding model (B8)** and **spatial CNN (B6)**.

| Ablation | Change | Interpretation |
|----------|--------|----------------|
| **C1** | Reduce spatial patch from full field to 3×3 | If performance drops significantly, large-scale spatial context matters. |
| **C2** | Reduce spatial patch to 1×1 (pointwise) | If this matches B5 performance, the embedding adds no spatial value. |
| **C3** | Remove temporal context (predict day t from surface fields at day t only) | If performance drops, temporal context (surface evolution) helps. |
| **C4** | Reduce temporal window from N days to 1 day | Measures value of temporal memory. |
| **C5** | Reduce embedding dimension by 50% | Tests whether the embedding is overparameterized. |

### 8.4 Execution Order

1. Train and evaluate B5 (pointwise MLP) with all 7 features (A8) — this is the reference.
2. Train and evaluate B5 with each feature ablation A1–A7.
3. For the embedding model B8, train with full features and full context — reference.
4. Train B8 with context ablations C1–C5.
5. **Do not** train all combinations — only one factor changed at a time.

### 8.5 Reporting Ablation Results

For each ablation, produce a table:

| Ablation | RMSE_50m | RMSE_100m | RMSE_300m | RMSE_mean | Δ from reference |
|----------|----------|-----------|-----------|-----------|------------------|
| A8 (full) | — | — | — | — | 0.00 (reference) |
| A1 (no SST) | — | — | — | — | +X.XX |
| A2 (no SSH) | — | — | — | — | +X.XX |
| ... | — | — | — | — | — |

If `Δ < +0.05°C` for any ablation, that feature contributes negligibly at those depths.

---

## 9. SCIENTIFIC CLAIM BOUNDARIES

### 9.1 Claims We MAY Make

With evidence level required:

| Claim | Evidence Required | Example Wording |
|-------|-------------------|-----------------|
| "The embedding model achieves lower RMSE against GLORYS than all pointwise baselines at all depths." | Table showing RMSE_d for all models; bootstrap CIs show non-overlap at ≥4 depths. | "Our model achieves a mean RMSE of X°C across depths, outperforming the best pointwise baseline (Y°C) by Z°C (p < 0.01 via paired bootstrap)." |
| "Surface features alone can reconstruct GLORYS subsurface temperature within X°C at depths ≤ 200 m." | RMSE_d at ≤ 200 m is lower than climatology (R² > 0). | "At 100 m, the model achieves RMSE = X°C (R² = Y), indicating that surface fields contain sufficient information to estimate thermocline temperature within the reanalysis uncertainty." |
| "SSS contributes critically to Bay of Bengal performance at 20–100 m." | Ablation A3 (no SSS) shows a ΔRMSE > 0.1°C in the Bay of Bengal at those depths, with non-overlapping CIs. | "Removing SSS increased Bay of Bengal RMSE at 50 m from X to Y°C (+Z%), confirming the role of salinity stratification in this basin." |
| "The model generalizes to the test period without temporal overfitting." | Test RMSE is within 10% of validation RMSE for all depths. | "The generalization gap (validation vs test RMSE) is < 10% at all depths, indicating no substantial overfitting to the training period." |
| "The embedding architecture adds value beyond pointwise processing." | Ablation C2 (1×1 patch) degrades RMSE by > 0.1°C at any depth compared to full B8. | "Reducing spatial context to a single pixel increased mean RMSE by X%, demonstrating that spatial structure in surface fields contributes to subsurface prediction." |

### 9.2 Claims We MUST NOT Make

| Prohibited Claim | Reason |
|------------------|--------|
| "We reconstruct observed subsurface ocean temperature." | The target is GLORYS, not observations. |
| "The model is validated against independent Argo data." | ARGO profiles are not independent of GLORYS (GLORYS assimilates Argo). |
| "Our method can be used for operational oceanography without reanalysis." | Untested in an operational setting; would require real-time surface data and evaluation against real-time in situ profiles. |
| "The embedding learns physically meaningful features." | Without a physical interpretation analysis (e.g., latent space analysis, sensitivity maps), we cannot claim physical interpretability. |
| "Our method outperforms all prior work on ocean temperature reconstruction." | Requires comparison against published methods on identical data, which does not exist for this specific task. |
| "The model can predict the impact of climate change on subsurface temperature." | The 366-day period does not capture interannual or decadal variability. |

### 9.3 Claims Requiring Additional Evidence

These claims are permissible IF accompanied by the specified evidence:

| Conditional Claim | Evidence Must Include |
|-------------------|----------------------|
| "The model generalizes across years." | Train on 2020, evaluate on 2021 data (must acquire) — not part of Phase 1. |
| "The embedding captures mesoscale eddy-driven subsurface variability." | Show that RMSE spatial patterns correlate with eddy kinetic energy from AVISO. |
| "Surface features are sufficient for thermocline reconstruction." | Show R² > 0.5 at depths 50–200 m for the embedding model AND the Ridge baseline, confirming the linear signal is present. |

---

## 10. FINAL ML ACCEPTANCE GATE

### 10.1 Pre-Training Gate

All conditions must be satisfied BEFORE the first ML model (including climatology B1) is trained.

| # | Condition | Verification Method | Severity If Failed |
|---|-----------|-------------------|-------------------|
| 1 | GLORYS target semantics documented in writing | Check protocol.md Section 1 wording | BLOCKER — no experiment valid |
| 2 | 7-day temporal purge implemented between train/val/test | Assert indices; log partition file | BLOCKER — temporal leakage |
| 3 | Depth_valid_mask includes bathymetric constraint (GEBCO) | Plot mask; assert no train point below seafloor | BLOCKER — bathymetric contamination |
| 4 | GLORYS level 50 (~1062.8 m) acquired; 1000 m interpolated | Verify interpolation error < 0.2°C | BLOCKER — target corruption at 1000 m |
| 5 | All 4 masks computed and intersected; mask fractions logged | Report % valid per depth per partition | BLOCKER — mask semantics must be checkable |
| 6 | All 7 features regridded via CONSERVATIVE method to 1/4° | Log CDO/xesmf command; verify global mean SST preserved | MAJOR — regridding artifacts |
| 7 | Normalization computed on TRAINING partition only | Assert index range; save and checksum JSON | BLOCKER — normalization leakage |
| 8 | All product DOIs, versions, download dates, SHA256 checksums logged | Visible in repo provenance file | BLOCKER — irreproducible |
| 9 | Temporal autocorrelation experiment run (Section 3) | ACF table populated | MAJOR — needed for purge justification |
| 10 | ARGO evaluation protocol written and approved | Section 4 of this protocol saved | MAJOR — must be designed before test predictions exist |

### 10.2 Pre-Embedding Gate

Conditions that must be satisfied BEFORE the embedding model (B8) is trained.

| # | Condition | Verification |
|---|-----------|-------------|
| 1 | All baselines B0–B5 produce complete predictions on test set | No NaN predictions; RMSE_d computed for all depths |
| 2 | At least one learned baseline (B2–B5) beats climatology (B1) at every depth (R²_d > 0) | Table of R²_d for B2–B5 vs B1 |
| 3 | Ridge R²_d > 0 at ALL depths | If Ridge fails at depth d, no linear signal exists — embedding must justify nonlinearity |
| 4 | The pointwise MLP (B5) is within 10% of the best baseline (B2–B4) | If B5 is worse, the neural approach needs justification |
| 5 | Ablation A8 (full features) reference for B5 computed | Ablation table base row exists |
| 6 | Spatial CNN (B6) or temporal model (B7) at least run on validation | Shows that adding one context dimension helps before adding both |
| 7 | Hyperparameter search logs for all baselines in repo | Full logs, not just best config |
| 8 | All leakage checks (Section 6) automated and passing | CI/CD check or pre-training assertion script |

### 10.3 Pre-Publication Gate

Conditions for any public dissemination (paper, preprint, talk).

| # | Condition |
|---|-----------|
| 1 | All results include bootstrap confidence intervals |
| 2 | All metric tables include per-depth and per-region breakdowns |
| 3 | GLORYS–Argo discrepancy reported alongside ML–Argo discrepancy |
| 4 | Climatology performance reported alongside all learned models |
| 5 | Ablation results for at least A1–A8 included |
| 6 | Code, data provenance, and pre-trained models archived with DOI |
| 7 | Seed_log.txt with all random seeds included |
| 8 | A one-page "scientific interpretation" document written explaining which depths/regions are well-predicted and why |
| 9 | The term "reconstruction" not used unqualified |

### 10.4 Breach Protocol

If ANY leakage check (Section 6) fails after an experiment has run:

1. **Stop immediately.** Do not report results.
2. Document the failure: what leaked, how much, the impact estimate.
3. Fix the root cause.
4. Re-run from the earliest affected step.
5. If the fix changes results by > 5% of any metric, the previous results are VOID.
6. If data contamination is unrecoverable (e.g., test data seen during normalization), discard the entire experiment and restart from scratch.

---

## APPENDIX: QUICK REFERENCE TABLES

### A. Partition Summary

| Partition | Start | End | Days | Indices (0-based) | Calendar 2020 |
|-----------|-------|-----|------|-------------------|---------------|
| Train | Jan 1 | Sep 9 | 253 | 0–252 | Winter–Summer |
| Purge | Sep 10 | Sep 15 | 6 | 253–258 | — |
| Validation | Sep 16 | Nov 2 | 48 | 259–306 | Fall (post-monsoon) |
| Purge | Nov 3 | Nov 8 | 6 | 307–312 | — |
| Test | Nov 9 | Dec 31 | 53 | 313–365 | Fall–Winter |

### B. Depth Specifications

| Index | Depth (m) | Expected GLORYS native levels bracketing | Layer |
|-------|-----------|------------------------------------------|-------|
| 0 | 0 | Surface (~0.5 m) | Surface |
| 1 | 5 | ~2.5 m, ~7.5 m | Near-surface |
| 2 | 10 | ~7.5 m, ~12.5 m | Near-surface |
| 3 | 20 | ~17.5 m, ~22.5 m | Mixed layer |
| 4 | 30 | ~27.5 m, ~32.5 m | Mixed layer |
| 5 | 50 | ~47.5 m, ~55 m | Thermocline top |
| 6 | 75 | ~72 m, ~82 m | Thermocline |
| 7 | 100 | ~95 m, ~110 m | Thermocline |
| 8 | 125 | ~118 m, ~135 m | Thermocline |
| 9 | 150 | ~142 m, ~162 m | Thermocline |
| 10 | 200 | ~190 m, ~215 m | Below thermocline |
| 11 | 300 | ~285 m, ~320 m | Deep |
| 12 | 500 | ~480 m, ~530 m | Deep |
| 13 | 700 | ~680 m, ~740 m | Deep |
| 14 | 1000 | ~902 m, ~1062 m | Deep (requires level 50) |

### C. Mask Summary

| Mask | Dimensions | Source | Definition |
|------|------------|--------|------------|
| geographic_ocean_mask | (H, W) | GEBCO bathymetry | 1 if water depth > 0 m |
| surface_validity_mask | (T, H, W) | All 7 products | 1 if all 7 features valid at (t, lat, lon) |
| target_validity_mask | (T, D, H, W) | GLORYS thetao | 1 if GLORYS value not NaN |
| depth_valid_mask | (D, H, W) | GEBCO bathymetry | 1 if bathymetry[lat,lon] >= target_depth[d] |
| **training_mask** | (T, D, H, W) | All 4 ANDed | 1 only if all four masks = 1 |

### D. Metric Report Template

| Model | Depth | RMSE | MAE | Bias | r | R² |
|-------|-------|------|-----|------|---|----|
| B1 Climatology | 0 | — | — | — | — | — |
| B1 Climatology | 5 | — | — | — | — | — |
| ... | ... | — | — | — | — | — |
| B2 Ridge | 0 | — | — | — | — | — |
| ... | ... | — | — | — | — | — |

Repeated for: Full Domain, Arabian Sea, Bay of Bengal.

---

*End of Scientific ML Experiment Protocol. Version 1.0. No code implementation — design document only.*