# Scientific Evaluation Report: Baseline B0b (Day 252 Persistence)

## 1. Model Identification and Architecture
- **Model Identifier**: `B0b`
- **Model Name**: End-of-Training Persistence (`B0b_PersistenceDay252`)
- **Trainable Parameters**: 0
- **Input Representation**: GLORYS thetao field from Day 252 (2020-09-09, last day of training partition) propagated statically across validation and test partitions.
- **Physical Role**: Evaluates memory surviving across the 6-day purge buffer into validation (lead time: 7–54 days) and test (lead time: 61–113 days).

## 2. Overall Performance Metrics

| Evaluation Split | Time Period | Days | Samples | RMSE (Unweighted) | RMSE (Weighted) | MAE | Bias | Pearson r | R² (vs B1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | Sep 16 – Nov 02 | 48 | 7,261,344 | 1.1046°C | 1.1095°C | 0.7669°C | +0.1649°C | 0.8156 | 0.2218 |
| **Test** | Nov 09 – Dec 31 | 53 | 8,017,734 | 1.7287°C | 1.7560°C | 1.2447°C | +0.3236°C | 0.4686 | -1.2358 |

### 7-Day Block-Bootstrap 95% Confidence Intervals (Test Split)
- **Unweighted RMSE 95% CI**: [1.6336°C, 1.8314°C]
- **Weighted RMSE 95% CI**: [1.6539°C, 1.8658°C]
- **MAE 95% CI**: [1.1771°C, 1.3175°C]
- **Bias 95% CI**: [+0.2608°C, +0.3851°C]

## 3. Depth-Wise Metric Decomposition (Test Split, Days 313–365)

| Depth (m) | Valid Points | RMSE (°C) | 95% CI [Low, High] | MAE (°C) | Bias (°C) | Pearson r | R² (vs B1) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 601,550 | 2.2103 | [1.8932, 2.5066] | 1.5044 | +0.9402 | 0.0677 | -2.9239 |
| 5 | 601,550 | 2.1471 | [1.8338, 2.4392] | 1.4592 | +0.9031 | 0.0977 | -3.1066 |
| 10 | 594,395 | 2.0001 | [1.7285, 2.2611] | 1.3759 | +0.7737 | 0.1848 | -3.0711 |
| 20 | 584,590 | 1.7457 | [1.5704, 1.9362] | 1.2172 | +0.4305 | 0.3520 | -2.6444 |
| 30 | 570,121 | 1.5090 | [1.4530, 1.5790] | 1.0771 | +0.0737 | 0.5405 | -1.4883 |
| 50 | 547,649 | 1.7901 | [1.7355, 1.8426] | 1.2079 | -0.1707 | 0.6128 | -0.4385 |
| 75 | 526,926 | 2.4603 | [2.4082, 2.5182] | 1.8649 | +0.0045 | 0.4286 | -0.1409 |
| 100 | 513,941 | 2.8206 | [2.7167, 2.9405] | 2.1847 | +0.3026 | 0.1954 | -0.1477 |
| 125 | 511,397 | 2.8181 | [2.7051, 2.9503] | 2.1558 | +0.4911 | 0.1256 | -0.3136 |
| 150 | 508,482 | 2.5000 | [2.4116, 2.6042] | 1.8464 | +0.5322 | 0.2814 | -0.5443 |
| 200 | 504,348 | 1.7621 | [1.7173, 1.8147] | 1.1986 | +0.4522 | 0.5575 | -1.0125 |
| 300 | 500,214 | 0.7630 | [0.7542, 0.7740] | 0.5675 | +0.1024 | 0.8613 | -0.4497 |
| 500 | 491,946 | 0.4361 | [0.4335, 0.4387] | 0.3206 | +0.0002 | 0.9313 | -0.6346 |
| 700 | 486,010 | 0.4597 | [0.4571, 0.4635] | 0.3288 | -0.0133 | 0.9205 | -0.7389 |
| 1000 | 474,615 | 0.5088 | [0.5012, 0.5170] | 0.3618 | +0.0318 | 0.8720 | -0.8813 |

## 4. Regional Breakdown (Cosine-Latitude Area Weighted)

| Region | Lat Range | Lon Range | RMSE (Unweighted) | RMSE (Area Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Full Domain** | 5°N – 30°N | 45°E – 105°E | 1.7211°C | 1.7470°C |
| **Arabian Sea** | 5°N – 25°N | 45°E – 77°E | 1.8134°C | 1.8225°C |
| **Bay of Bengal** | 5°N – 25°N | 77°E – 100°E | 1.2509°C | 1.2800°C |

## 5. Seasonal / Temporal Breakdown (Test Split)

| Seasonal Period | Calendar Range | Days | RMSE (Unweighted) | RMSE (Sample Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Late Fall (November)** | Days 313–342 | 30 | 1.6122°C | 1.6309°C |
| **Early Winter (December)** | Days 343–365 | 23 | 1.8618°C | 1.8985°C |

## 6. Physical and Oceanographic Analysis
- **Memory Persistence Advantage over B0**: B0b achieves substantial error reductions relative to B0 on validation (lead time ~20–50 days), retaining partial mesoscale memory.
- **Temporal Drift into Test Partition**: By November–December (lead times exceeding 2 months), B0b error increases noticeably as the Northeast Monsoon transitions the Arabian Sea and Bay of Bengal upper ocean.
- **Thermocline Dynamics**: The 75–125 m thermocline experiences the highest prediction error, confirming that dynamic vertical reconstruction from surface forcing is required.
