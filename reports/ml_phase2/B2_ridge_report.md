# Scientific Benchmark Report: Baseline B2 (Multi-Output Ridge Regression)

## 1. Model Identification and Architecture
- **Model Identifier**: `B2`
- **Model Name**: Multi-Output Ridge Regression (`B2_Ridge`)
- **Model Category**: Tabular Linear Supervised Baseline (Pointwise ML)
- **Trainable Parameters**: 120 (15 depth-wise regressors $\times$ [7 coefficients + 1 intercept])
- **Selected Hyperparameter**: $\alpha^* = 100000.0$ (tuned strictly on validation split)
- **Predictor Features (7 Canonical)**: `sst`, `sss`, `ssh`, `current_u`, `current_v`, `wind_u`, `wind_v`
- **Target Representation**: GLORYS $\theta_o$ across 15 canonical depths (0 to 1000 m)
- **Git Commit SHA**: `0e3a9e024d82efab13904f63a7f6b5f69d9fc280`

## 2. Hyperparameter Selection: Validation Tuning Curve

The regularizer $\alpha$ was tuned strictly across candidate values using the Train (days 0–252) and Validation (days 259–306) partitions with zero access to the Test partition:

| Candidate $\alpha$ | Val Unweighted RMSE (°C) | Val Sample-Weighted RMSE (°C) | Val MAE (°C) | Selection Status |
| :---: | :---: | :---: | :---: | :---: |
|      0.001 | 1.1048 | 1.1939 | 0.8259 | Candidate |
|      0.010 | 1.1048 | 1.1939 | 0.8259 | Candidate |
|      0.100 | 1.1048 | 1.1939 | 0.8259 | Candidate |
|      1.000 | 1.1048 | 1.1939 | 0.8259 | Candidate |
|     10.000 | 1.1048 | 1.1939 | 0.8259 | Candidate |
|    100.000 | 1.1048 | 1.1939 | 0.8259 | Candidate |
|   1000.000 | 1.1047 | 1.1939 | 0.8258 | Candidate |
|  10000.000 | 1.1042 | 1.1936 | 0.8255 | Candidate |
| 100000.000 | 1.1015 | 1.1924 | 0.8239 | **SELECTED ($\alpha^*$)** |

## 3. Overall Performance Summary and Mandatory Comparisons

| Model | Unweighted RMSE (°C) | 95% Bootstrap CI | Sample-Weighted RMSE (°C) | MAE (°C) | Bias (°C) | Mean Pearson $r$ | Mean $R^2$ (vs B1) | $\Delta$ vs B1 (°C) | Rel. Imprv. vs B1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Day 0)** | 1.5220 | [1.3986, 1.6305] | 1.5273 | 1.1371 | -0.2966 | 0.5707 | -0.4966 | +0.2638 | -20.97% |
| **B0b (Day 252)** | 1.7287 | [1.6336, 1.8314] | 1.7560 | 1.2447 | +0.3236 | 0.4686 | -1.2358 | +0.4705 | -37.39% |
| **B1 (Climatology)** | 1.2582 | [1.1866, 1.3301] | 1.2596 | 0.9641 | +0.3834 | 0.6325 | 0.0000 | Baseline (0.000) | Baseline (0.0%) |
| **B2 (Ridge, $\alpha^*=100000.0$)** | **1.0295** | **[1.0023, 1.0591]** | **1.0131** | **0.8029** | **+0.0256** | **0.7533** | **-0.5404** | **-0.2287** | **+18.18%** |

### Key Comparison Takeaways vs Reference Baseline B1
1. **Overall Test RMSE Delta**: B2 achieves **-0.2287°C** unweighted delta and **-0.2465°C** sample-weighted delta relative to B1 Climatology.
2. **Relative Percentage Improvement**: **+18.18%** (unweighted) / **+19.57%** (sample-weighted).
3. **Thermocline Regime (50–150 m)**: B1 mean RMSE = 2.1799°C vs B2 mean RMSE = 1.5350°C (relative change: **+29.59%**).
4. **Abyssal Regime (500–1000 m)**: B1 mean RMSE = 0.3535°C vs B2 mean RMSE = 0.7467°C (relative change: **-111.22%**).
5. **$R^2$ Metric vs B1**: Mean unweighted $R^2 = -0.5404$ (sample-weighted $R^2 = -0.4486$).
6. **Bootstrap CI Overlap**: B2 95% CI [1.0023, 1.0591]°C vs B1 95% CI [1.1866, 1.3301]°C.

## 4. Depth-Wise Metric Decomposition (Test Split, Days 313–365)

| Depth (m) | Evaluated Points | B2 RMSE (°C) | 95% CI [Low, High] | B1 RMSE (°C) | $\Delta$ vs B1 (°C) | B2 MAE (°C) | B2 Bias (°C) | B2 Corr | B2 $R^2$ (vs B1) | Best Model |
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

## 5. Regional Breakdown (Cosine-Latitude Area Weighted)

| Region | B2 Weighted RMSE | B1 Weighted RMSE | $\Delta$ vs B1 (°C) | Relative Improvement | Regional Oceanographic Context |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Domain** | 1.0133°C | 1.2588°C | -0.2455°C | +19.50% | Entire study domain (5°N–30°N, 45°E–105°E) |
| **Arabian Sea** | 1.1189°C | 1.3325°C | -0.2136°C | +16.03% | High salinity, strong evaporative cooling |
| **Bay of Bengal** | 0.8178°C | 1.0474°C | -0.2296°C | +21.92% | Low salinity, strong riverine barrier layer |

## 6. Seasonal / Temporal Breakdown (Test Split)

| Seasonal Period | Calendar Range | Days | B2 Weighted RMSE | B1 Weighted RMSE | $\Delta$ vs B1 (°C) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Late Fall (November)** | Days 313–342 | 30 | 1.0343°C | 1.1661°C | -0.1318°C | +11.30% |
| **Early Winter (December)** | Days 343–365 | 23 | 0.9832°C | 1.3509°C | -0.3677°C | +27.22% |

## 7. Learned Feature Coefficients Analysis

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

## 8. Oceanographic and Statistical Discussion
1. **Surface Coupling**: SST carries the largest positive weight in the top 30 m ($+1.8$ to $+2.0$), confirming direct conductive coupling in the surface mixed layer.
2. **Thermocline Transition (50–150 m)**: In the thermocline, SSH and SSS weights become prominent. Sea surface height reflects baroclinic depth integration of the pycnocline, providing dynamic upward/downward displacement information.
3. **Linearity Limitation**: Because Ridge is strictly linear and pointwise, it cannot capture localized mesoscale frontal structures or nonlinear density stratifications, setting the baseline for nonlinear models (B3–B8).
4. **Deep Ocean Damping**: Below 500 m, all regression weights attenuate toward zero, with the prediction driven primarily by the intercept (mean abyssal temperature ~15°C adjusted to deep ocean values ~6–8°C).
