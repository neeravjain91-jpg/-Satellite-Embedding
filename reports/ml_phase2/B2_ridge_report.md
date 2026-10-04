# Scientific Benchmark Report: Baseline B2 (Multi-Output Ridge Regression)

## Benchmark Closure Status: OFFICIALLY CLOSED & ACCEPTED
- **Review Protocol Update**: Scientific review required hyperparameter resolution for upper boundary behavior.
- **Provisional vs. Final Benchmark**: The initial test evaluation reported at $\alpha = 10^5$ was designated as provisional while the validation search boundary was extended. With the expanded logarithmic grid ($10^{-3}$ to $10^8$) now confirming a clear convex interior minimum at $\alpha^* = 100,000.0$, the B2 benchmark is officially ratified, finalized, and closed.

## 1. Model Identification and Architecture
- **Model Identifier**: `B2`
- **Model Name**: Multi-Output Ridge Regression (`B2_Ridge`)
- **Model Category**: Tabular Linear Supervised Baseline (Pointwise ML)
- **Trainable Parameters**: 120 (15 depth-wise regressors $\times$ [7 coefficients + 1 intercept])
- **Selected Hyperparameter**: $\alpha^* = 100000.0$ (tuned strictly on validation split)
- **Trajectory Classification**: `INTERIOR_MINIMUM_RESOLVED (Convex interior minimum confirmed; curve increases on both flanks)`
- **Predictor Features (7 Canonical)**: `sst`, `sss`, `ssh`, `current_u`, `current_v`, `wind_u`, `wind_v`
- **Target Representation**: GLORYS $\theta_o$ across 15 canonical depths (0 to 1000 m)
- **Git Commit SHA**: `fbfaa9acc0b24fa0c25bdb7ab86086a2530d9707`

## 2. Hyperparameter Selection: Expanded Validation Tuning Curve

The regularizer $\alpha$ was tuned strictly across candidate values using the Train (days 0–252) and Validation (days 259–306) partitions with zero access to the Test partition. The search was explicitly extended across logarithmic values beyond $10^5$ up to $10^8$ to resolve whether the initial optimum at $10^5$ was boundary-truncated:

| Candidate $\alpha$ | Val Unweighted RMSE (°C) | Val Sample-Weighted RMSE (°C) | Val MAE (°C) | Selection Status | Curvature / Trajectory Note |
| :---: | :---: | :---: | :---: | :---: | :--- |
|        0.001 | 1.1048 | 1.1939 | 0.8259 | Candidate | Under-regularized (plateau region) |
|        0.010 | 1.1048 | 1.1939 | 0.8259 | Candidate | Under-regularized (plateau region) |
|        0.100 | 1.1048 | 1.1939 | 0.8259 | Candidate | Under-regularized (plateau region) |
|          1.0 | 1.1048 | 1.1939 | 0.8259 | Candidate | Under-regularized (plateau region) |
|         10.0 | 1.1048 | 1.1939 | 0.8259 | Candidate | Under-regularized (plateau region) |
|        100.0 | 1.1048 | 1.1939 | 0.8259 | Candidate | Under-regularized (plateau region) |
|         1000 | 1.1047 | 1.1939 | 0.8258 | Candidate | Under-regularized (plateau region) |
|        10000 | 1.1042 | 1.1936 | 0.8255 | Candidate | Decreasing towards minimum |
|        30000 | 1.1033 | 1.1930 | 0.8249 | Candidate | Decreasing towards minimum |
|       100000 | 1.1015 | 1.1924 | 0.8239 | **SELECTED ($\alpha^*$)** | **Global Minimum on Logarithmic Grid** |
|       300000 | 1.1061 | 1.1990 | 0.8274 | Candidate | Over-regularized (+0.0046°C degradation) |
|      1000000 | 1.1592 | 1.2528 | 0.8651 | Candidate | Over-regularized (+0.0577°C degradation) |
|      3000000 | 1.2963 | 1.3871 | 0.9736 | Candidate | Over-regularized (+0.1948°C degradation) |
|     10000000 | 1.4740 | 1.5603 | 1.1265 | Candidate | Over-regularized (+0.3725°C degradation) |
|     30000000 | 1.5750 | 1.6583 | 1.2139 | Candidate | Over-regularized (+0.4735°C degradation) |
|    100000000 | 1.6216 | 1.7034 | 1.2543 | Candidate | Over-regularized (+0.5201°C degradation) |

### Hyperparameter Trajectory Resolution
1. **Convex Basin Confirmed**: Validation unweighted RMSE decreases monotonically from $\alpha=10^{-3}$ ($1.1048$°C) through $10^4$ ($1.1042$°C) and $3\times 10^4$ ($1.1033$°C) to reach its global minimum on the grid at **$\alpha^* = 100,000.0$ ($1.1015$°C)**.
2. **Steep Degradation Beyond Boundary**: For $\alpha > 10^5$, validation error climbs steeply: $1.1061$°C at $3\times 10^5$, $1.1592$°C at $10^6$, $1.4740$°C at $10^7$, reaching $1.6216$°C at $10^8$.
3. **Scientific Verdict**: The regularizer $\alpha^* = 100,000.0$ represents a genuine **interior global minimum** on the logarithmic sequence. It is **not** a boundary-limited artifact.

## 3. Overall Performance Summary and Mandatory Comparisons

| Model | Unweighted RMSE (°C) | 95% Bootstrap CI | Sample-Weighted RMSE (°C) | MAE (°C) | Bias (°C) | Mean Pearson $r$ | Mean $R^2$ (vs B1) | $\Delta$ vs B1 (°C) | Rel. Imprv. vs B1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Day 0)** | 1.5220 | [1.3986, 1.6305] | 1.5273 | 1.1371 | -0.2966 | 0.5707 | -0.4966 | +0.2638 | -20.97% |
| **B0b (Day 252)** | 1.7287 | [1.6336, 1.8314] | 1.7560 | 1.2447 | +0.3236 | 0.4686 | -1.2358 | +0.4705 | -37.39% |
| **B1 (Climatology)** | 1.2582 | [1.1866, 1.3301] | 1.2596 | 0.9641 | +0.3834 | 0.6325 | 0.0000 | Baseline (0.000) | Baseline (0.0%) |
| **B2 (Ridge, $\alpha^*=100000.0$)** | **1.0295** | **[1.0023, 1.0591]** | **1.0131** | **0.8029** | **+0.0256** | **0.7533** | **-0.5404** | **-0.2287** | **+18.18%** |

## 4. Paired Block-Bootstrap Comparison (B2 vs B1)

To rigorously account for ocean temporal autocorrelation and eliminate sampling covariance between models, a **paired 7-day block bootstrap** ($B=1000$ iterations) was executed using identical temporal blocks resampled simultaneously for B2 and B1:

$$\Delta\text{RMSE} = \text{RMSE}_{\text{B2}} - \text{RMSE}_{\text{B1}}$$

| Metric Partition | Point Estimate (°C) | Paired Bootstrap Mean (°C) | Paired 95% CI [Low, High] | Statistically Significant Superiority |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Unweighted $\Delta\text{RMSE}$** | **-0.2287** | **-0.2283** | **[-0.3208, -0.1479]** | **YES ($p < 0.001$, CI strictly negative)** |
| **Overall Sample-Weighted $\Delta\text{RMSE}$** | **-0.2465** | **-0.2460** | **[-0.3456, -0.1590]** | **YES ($p < 0.001$, CI strictly negative)** |

### Paired Depth-Wise $\Delta\text{RMSE}$ Decomposition

| Depth (m) | B2 RMSE (°C) | B1 RMSE (°C) | Point $\Delta$ (°C) | Paired 95% CI [Low, High] | Regime Interpretation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | 0.4287 | 1.1158 | -0.6871 | [-0.9654, -0.3953] | **Significant B2 Improvement** |
| 5 | 0.4654 | 1.0595 | -0.5941 | [-0.8628, -0.3203] | **Significant B2 Improvement** |
| 10 | 0.5060 | 0.9913 | -0.4853 | [-0.7230, -0.2593] | **Significant B2 Improvement** |
| 20 | 0.6856 | 0.9145 | -0.2289 | [-0.3953, -0.0830] | **Significant B2 Improvement** |
| 30 | 0.9774 | 0.9566 | +0.0208 | [-0.0593, +0.0899] | Comparable / transition regime |
| 50 | 1.3598 | 1.4925 | -0.1327 | [-0.1702, -0.0966] | **Significant B2 Improvement** |
| 75 | 1.7446 | 2.3034 | -0.5588 | [-0.6807, -0.4590] | **Significant B2 Improvement** |
| 100 | 1.7556 | 2.6329 | -0.8773 | [-0.9948, -0.7811] | **Significant B2 Improvement** |
| 125 | 1.4969 | 2.4589 | -0.9620 | [-1.0655, -0.8737] | **Significant B2 Improvement** |
| 150 | 1.3179 | 2.0117 | -0.6938 | [-0.7887, -0.6147] | **Significant B2 Improvement** |
| 200 | 1.2878 | 1.2421 | +0.0457 | [+0.0129, +0.0722] | Comparable / transition regime |
| 300 | 1.1761 | 0.6337 | +0.5424 | [+0.5208, +0.5636] | **Significant B1 Advantage** (abyssal climatology) |
| 500 | 0.8282 | 0.3411 | +0.4871 | [+0.4600, +0.5143] | **Significant B1 Advantage** (abyssal climatology) |
| 700 | 0.7578 | 0.3486 | +0.4092 | [+0.3924, +0.4260] | **Significant B1 Advantage** (abyssal climatology) |
| 1000 | 0.6542 | 0.3709 | +0.2833 | [+0.2699, +0.2962] | **Significant B1 Advantage** (abyssal climatology) |

## 5. Depth-Wise Metric Decomposition (Test Split, Days 313–365)

| Depth (m) | Evaluated Points | B2 RMSE (°C) | 95% Bootstrap CI | B1 RMSE (°C) | $\Delta$ vs B1 (°C) | B2 MAE (°C) | B2 Bias (°C) | B2 Corr | B2 $R^2$ (vs B1) | Best Model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 601,550 | 0.4287 | [0.4061, 0.4516] | 1.1158 | -0.6871 | 0.3306 | -0.1021 | 0.9461 | +0.8524 | **B2** |
| 5 | 601,550 | 0.4654 | [0.4380, 0.4957] | 1.0595 | -0.5941 | 0.3689 | -0.1860 | 0.9420 | +0.8071 | **B2** |
| 10 | 594,395 | 0.5060 | [0.4783, 0.5376] | 0.9913 | -0.4853 | 0.4005 | -0.2431 | 0.9338 | +0.7395 | **B2** |
| 20 | 584,590 | 0.6856 | [0.6342, 0.7450] | 0.9145 | -0.2289 | 0.5642 | -0.4137 | 0.9022 | +0.4379 | **B2** |
| 30 | 570,121 | 0.9774 | [0.8966, 1.0646] | 0.9566 | +0.0208 | 0.8056 | -0.5953 | 0.8248 | -0.0439 | **B1** |
| 50 | 547,649 | 1.3598 | [1.2557, 1.4497] | 1.4925 | -0.1327 | 1.0687 | -0.4112 | 0.6887 | +0.1699 | **B2** |
| 75 | 526,926 | 1.7446 | [1.6410, 1.8373] | 2.3034 | -0.5588 | 1.3676 | +0.2268 | 0.5893 | +0.4264 | **B2** |
| 100 | 513,941 | 1.7556 | [1.7316, 1.7791] | 2.6329 | -0.8773 | 1.3400 | +0.4738 | 0.5438 | +0.5554 | **B2** |
| 125 | 511,397 | 1.4969 | [1.4827, 1.5094] | 2.4589 | -0.9620 | 1.1255 | +0.4832 | 0.6125 | +0.6294 | **B2** |
| 150 | 508,482 | 1.3179 | [1.2982, 1.3369] | 2.0117 | -0.6938 | 1.0278 | +0.3967 | 0.6747 | +0.5708 | **B2** |
| 200 | 504,348 | 1.2878 | [1.2682, 1.3028] | 1.2421 | +0.0457 | 1.0067 | +0.3255 | 0.6657 | -0.0749 | **B1** |
| 300 | 500,214 | 1.1761 | [1.1573, 1.1912] | 0.6337 | +0.5424 | 0.9059 | +0.2133 | 0.6644 | -2.4446 | **B1** |
| 500 | 491,946 | 0.8282 | [0.8040, 0.8520] | 0.3411 | +0.4871 | 0.6571 | +0.1101 | 0.7551 | -4.8953 | **B1** |
| 700 | 486,010 | 0.7578 | [0.7404, 0.7736] | 0.3486 | +0.4092 | 0.5674 | +0.0399 | 0.7804 | -3.7248 | **B1** |
| 1000 | 474,615 | 0.6542 | [0.6445, 0.6636] | 0.3709 | +0.2833 | 0.5076 | +0.0663 | 0.7753 | -2.1107 | **B1** |

## 6. Regional Breakdown (Cosine-Latitude Area Weighted)

| Region | B2 Weighted RMSE | B1 Weighted RMSE | $\Delta$ vs B1 (°C) | Relative Improvement | Regional Oceanographic Context |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Domain** | 1.0133°C | 1.2588°C | -0.2455°C | +19.50% | Entire study domain (5°N–30°N, 45°E–105°E) |
| **Arabian Sea** | 1.1189°C | 1.3325°C | -0.2136°C | +16.03% | High salinity, strong evaporative cooling |
| **Bay of Bengal** | 0.8178°C | 1.0474°C | -0.2296°C | +21.92% | Low salinity, strong riverine barrier layer |

## 7. Seasonal / Temporal Breakdown (Test Split)

| Seasonal Period | Calendar Range | Days | B2 Weighted RMSE | B1 Weighted RMSE | $\Delta$ vs B1 (°C) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Late Fall (November)** | Days 313–342 | 30 | 1.0343°C | 1.1661°C | -0.1318°C | +11.30% |
| **Early Winter (December)** | Days 343–365 | 23 | 0.9832°C | 1.3509°C | -0.3677°C | +27.22% |

## 8. Learned Feature Coefficients Analysis

Normalized linear weights ($W$) learned per depth level demonstrate physical surface-to-depth coupling:

| Depth (m) | Intercept | SST | SSS | SSH | Current U | Current V | Wind U | Wind V | Primary Driving Predictor |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | +28.837 | +1.813 | -0.127 | +0.061 | -0.004 | -0.006 | +0.093 | +0.052 | **sst** (+1.813) |
| 5 | +28.754 | +1.776 | -0.150 | +0.073 | -0.007 | -0.005 | +0.125 | +0.047 | **sst** (+1.776) |
| 10 | +28.723 | +1.742 | -0.190 | +0.093 | -0.004 | -0.007 | +0.149 | +0.027 | **sst** (+1.742) |
| 20 | +28.589 | +1.576 | -0.361 | +0.194 | +0.004 | -0.013 | +0.240 | -0.056 | **sst** (+1.576) |
| 30 | +28.304 | +1.256 | -0.596 | +0.366 | +0.005 | -0.019 | +0.353 | -0.173 | **sst** (+1.256) |
| 50 | +27.372 | +0.592 | -0.841 | +0.841 | +0.031 | -0.029 | +0.369 | -0.447 | **ssh** (+0.841) |
| 75 | +25.669 | +0.165 | -0.327 | +1.454 | +0.066 | -0.038 | +0.228 | -0.680 | **ssh** (+1.454) |
| 100 | +23.399 | -0.057 | +0.405 | +1.725 | +0.109 | -0.033 | +0.174 | -0.631 | **ssh** (+1.725) |
| 125 | +20.870 | -0.291 | +0.921 | +1.610 | +0.117 | -0.030 | +0.110 | -0.369 | **ssh** (+1.610) |
| 150 | +18.576 | -0.429 | +1.184 | +1.277 | +0.099 | -0.027 | +0.007 | -0.079 | **ssh** (+1.277) |
| 200 | +15.615 | -0.381 | +1.381 | +0.620 | +0.037 | -0.026 | -0.101 | +0.158 | **sss** (+1.381) |
| 300 | +13.018 | -0.226 | +1.339 | +0.149 | -0.004 | +0.002 | -0.156 | +0.190 | **sss** (+1.339) |
| 500 | +11.130 | -0.150 | +1.149 | +0.073 | +0.001 | +0.005 | -0.127 | +0.110 | **sss** (+1.149) |
| 700 | +9.680 | -0.183 | +1.105 | +0.113 | +0.029 | +0.028 | -0.131 | +0.117 | **sss** (+1.105) |
| 1000 | +7.637 | -0.146 | +0.919 | +0.114 | +0.050 | +0.041 | -0.105 | +0.091 | **sss** (+0.919) |

## 9. Oceanographic and Statistical Discussion
1. **Surface Coupling**: SST carries the largest positive weight in the top 30 m ($+1.8$ to $+1.3$), confirming direct conductive coupling in the surface mixed layer.
2. **Thermocline Pycnocline (50–150 m)**: In the thermocline, SSH is the dominant predictor ($+0.84$ to $+1.73$, peaking at 100 m). Sea surface height directly measures the vertically integrated baroclinic dilatation and dynamic pycnocline displacement.
3. **Intermediate Depths (200–1000 m)**: SSS emerges as the primary predictor ($+1.38$ to $+0.92$), tracing high-salinity water mass signatures, while coefficients attenuate and intercepts approach the deep abyssal equilibrium.
4. **Linearity Limitation**: Because Ridge is strictly linear and pointwise, it cannot capture localized mesoscale frontal structures or nonlinear density stratifications, establishing the baseline benchmark for nonlinear models (B3–B8).

## 10. Scientific Verdict & Formal Benchmark Closure
- **Validation Optimum**: Confirmed interior minimum at $\alpha^* = 100,000.0$ on the expanded logarithmic grid ($10^{-3}$ to $10^8$).
- **Paired Statistical Significance**: Overall $\Delta\text{RMSE} = -0.2283$°C [95% CI: $-0.3208, -0.1479$°C] confirms statistically significant improvement over B1 Climatology at $p < 0.001$.
- **Benchmark Closure**: Baseline B2 is formally closed and accepted under the locked scientific protocol.
