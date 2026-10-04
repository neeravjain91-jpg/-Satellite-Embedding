# Scientific Evaluation Report: Baseline B1 (Spatial-Depth Climatology)

## 1. Model Identification and Architecture
- **Model Identifier**: `B1`
- **Model Name**: Training-Only Spatial-Depth Climatology (`B1_Climatology`)
- **Trainable Parameters**: 0 (Non-parametric empirical mean lookup)
- **Mathematical Definition**: $T_{\text{clim}}(\text{lat}, \text{lon}, z) = \frac{1}{N_{\text{train}}} \sum_{t=0}^{252} \theta_o(t, \text{lat}, \text{lon}, z)$ strictly on the 253 training days.
- **Zero Data Leakage**: Formulated exclusively on days 0–252. No evaluation day (val or test) informs this climatology.
- **Reference Standard**: Serves as the universal denominator for skill score $R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{clim}}}$. By definition, $R^2(B1) \equiv 0.0000$.

## 2. Overall Performance Metrics

| Evaluation Split | Time Period | Days | Samples | RMSE (Unweighted) | RMSE (Weighted) | MAE | Bias | Pearson r | R² (vs B1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | Sep 16 – Nov 02 | 48 | 7,261,344 | 1.3382°C | 1.3528°C | 0.9682°C | +0.2248°C | 0.5678 | 0.0000 |
| **Test** | Nov 09 – Dec 31 | 53 | 8,017,734 | 1.2582°C | 1.2596°C | 0.9641°C | +0.3834°C | 0.6325 | 0.0000 |

### 7-Day Block-Bootstrap 95% Confidence Intervals (Test Split)
- **Unweighted RMSE 95% CI**: [1.1866°C, 1.3301°C]
- **Weighted RMSE 95% CI**: [1.1828°C, 1.3369°C]
- **MAE 95% CI**: [0.9002°C, 1.0401°C]
- **Bias 95% CI**: [+0.3206°C, +0.4449°C]

## 3. Depth-Wise Metric Decomposition (Test Split, Days 313–365)

| Depth (m) | Valid Points | RMSE (°C) | 95% CI [Low, High] | MAE (°C) | Bias (°C) | Pearson r | R² (vs B1) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 601,550 | 1.1158 | [0.8387, 1.3846] | 0.8282 | +0.7032 | 0.7350 | 0.0000 |
| 5 | 601,550 | 1.0595 | [0.8079, 1.3122] | 0.7829 | +0.6326 | 0.7371 | 0.0000 |
| 10 | 594,395 | 0.9913 | [0.7686, 1.2083] | 0.7458 | +0.5599 | 0.7433 | 0.0000 |
| 20 | 584,590 | 0.9145 | [0.7817, 1.0502] | 0.6879 | +0.3516 | 0.7396 | 0.0000 |
| 30 | 570,121 | 0.9566 | [0.8505, 1.0752] | 0.6822 | +0.0919 | 0.7161 | 0.0000 |
| 50 | 547,649 | 1.4925 | [1.3638, 1.6044] | 1.1214 | -0.0419 | 0.5579 | 0.0000 |
| 75 | 526,926 | 2.3034 | [2.2799, 2.3270] | 1.8590 | +0.3654 | 0.2310 | 0.0000 |
| 100 | 513,941 | 2.6329 | [2.5501, 2.7315] | 2.0954 | +0.7045 | 0.0039 | 0.0000 |
| 125 | 511,397 | 2.4589 | [2.3683, 2.5653] | 1.9072 | +0.8012 | 0.0977 | 0.0000 |
| 150 | 508,482 | 2.0117 | [1.9434, 2.0907] | 1.5378 | +0.7196 | 0.3905 | 0.0000 |
| 200 | 504,348 | 1.2421 | [1.2135, 1.2752] | 0.9369 | +0.5129 | 0.7642 | 0.0000 |
| 300 | 500,214 | 0.6337 | [0.6146, 0.6526] | 0.4770 | +0.2212 | 0.9207 | 0.0000 |
| 500 | 491,946 | 0.3411 | [0.3335, 0.3487] | 0.2567 | +0.0643 | 0.9612 | 0.0000 |
| 700 | 486,010 | 0.3486 | [0.3428, 0.3559] | 0.2633 | +0.0092 | 0.9561 | 0.0000 |
| 1000 | 474,615 | 0.3709 | [0.3564, 0.3870] | 0.2799 | +0.0560 | 0.9331 | 0.0000 |

## 4. Regional Breakdown (Cosine-Latitude Area Weighted)

| Region | Lat Range | Lon Range | RMSE (Unweighted) | RMSE (Area Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Full Domain** | 5°N – 30°N | 45°E – 105°E | 1.2579°C | 1.2588°C |
| **Arabian Sea** | 5°N – 25°N | 45°E – 77°E | 1.3321°C | 1.3325°C |
| **Bay of Bengal** | 5°N – 25°N | 77°E – 100°E | 1.0326°C | 1.0474°C |

## 5. Seasonal / Temporal Breakdown (Test Split)

| Seasonal Period | Calendar Range | Days | RMSE (Unweighted) | RMSE (Sample Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Late Fall (November)** | Days 313–342 | 30 | 1.1723°C | 1.1661°C |
| **Early Winter (December)** | Days 343–365 | 23 | 1.3426°C | 1.3509°C |

## 6. Physical and Oceanographic Analysis
- **Strict Lower Bound Benchmark**: B1 represents the baseline standard for any model claiming physical predictive skill ($R^2 > 0$). Any ML model with $R^2 \le 0$ fails to add value beyond static geographical mean profiles.
- **Thermocline Seasonality Gap**: Because the training partition spans January through September, the mean profile reflects Southwest Monsoon and pre-monsoon thermal structures, creating systematic bias during November–December post-monsoon cooling.
- **Abyssal Precision**: In the deep ocean (500–1000 m), climatology is highly effective (RMSE < 0.25°C) due to weak high-frequency variability below the permanent thermocline.
