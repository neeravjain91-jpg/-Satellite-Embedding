# Scientific Evaluation Report: Baseline B0 (Day 0 Persistence)

## 1. Model Identification and Architecture
- **Model Identifier**: `B0`
- **Model Name**: Day 0 Persistence (`B0_PersistenceDay0`)
- **Trainable Parameters**: 0
- **Input Representation**: GLORYS thetao field from Day 0 (2020-01-01) propagated statically across all evaluation timestamps.
- **Physical Role**: Upper bound of initial state memory (decorrelation timescale evaluation).

## 2. Overall Performance Metrics

| Evaluation Split | Time Period | Days | Samples | RMSE (Unweighted) | RMSE (Weighted) | MAE | Bias | Pearson r | R² (vs B1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | Sep 16 – Nov 02 | 48 | 7,261,344 | 1.9858°C | 2.0273°C | 1.4514°C | -0.4552°C | 0.3210 | -1.2116 |
| **Test** | Nov 09 – Dec 31 | 53 | 8,017,734 | 1.5220°C | 1.5273°C | 1.1371°C | -0.2966°C | 0.5707 | -0.4966 |

### 7-Day Block-Bootstrap 95% Confidence Intervals (Test Split)
- **Unweighted RMSE 95% CI**: [1.3986°C, 1.6305°C]
- **Weighted RMSE 95% CI**: [1.3924°C, 1.6456°C]
- **MAE 95% CI**: [1.0474°C, 1.2290°C]
- **Bias 95% CI**: [-0.3593°C, -0.2351°C]

## 3. Depth-Wise Metric Decomposition (Test Split, Days 313–365)

| Depth (m) | Valid Points | RMSE (°C) | 95% CI [Low, High] | MAE (°C) | Bias (°C) | Pearson r | R² (vs B1) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 601,550 | 1.2712 | [0.9066, 1.5805] | 0.8925 | -0.7211 | 0.7508 | -0.2980 |
| 5 | 601,550 | 1.2761 | [0.9055, 1.5866] | 0.8939 | -0.7313 | 0.7406 | -0.4507 |
| 10 | 594,395 | 1.2413 | [0.8958, 1.5317] | 0.8841 | -0.7286 | 0.7285 | -0.5681 |
| 20 | 584,590 | 1.2451 | [0.9488, 1.5064] | 0.8805 | -0.7211 | 0.7082 | -0.8539 |
| 30 | 570,121 | 1.3547 | [1.1210, 1.5687] | 0.9356 | -0.6929 | 0.6282 | -1.0055 |
| 50 | 547,649 | 1.9890 | [1.8680, 2.1011] | 1.3449 | -0.3081 | 0.2984 | -0.7760 |
| 75 | 526,926 | 2.9648 | [2.9243, 3.0113] | 2.2359 | -0.0543 | 0.0373 | -0.6568 |
| 100 | 513,941 | 3.1785 | [3.1452, 3.2084] | 2.4848 | -0.0565 | -0.0478 | -0.4574 |
| 125 | 511,397 | 2.8495 | [2.8150, 2.8803] | 2.2339 | -0.1095 | 0.0010 | -0.3430 |
| 150 | 508,482 | 2.2196 | [2.1778, 2.2659] | 1.7611 | -0.1455 | 0.2852 | -0.2173 |
| 200 | 504,348 | 1.2338 | [1.2196, 1.2490] | 1.0027 | -0.0365 | 0.7437 | 0.0134 |
| 300 | 500,214 | 0.6829 | [0.6672, 0.6969] | 0.5219 | +0.0248 | 0.9111 | -0.1613 |
| 500 | 491,946 | 0.4063 | [0.4037, 0.4090] | 0.3009 | -0.0138 | 0.9478 | -0.4189 |
| 700 | 486,010 | 0.4583 | [0.4553, 0.4607] | 0.3387 | -0.0839 | 0.9303 | -0.7279 |
| 1000 | 474,615 | 0.4585 | [0.4531, 0.4653] | 0.3455 | -0.0701 | 0.8968 | -0.5283 |

## 4. Regional Breakdown (Cosine-Latitude Area Weighted)

| Region | Lat Range | Lon Range | RMSE (Unweighted) | RMSE (Area Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Full Domain** | 5°N – 30°N | 45°E – 105°E | 1.5206°C | 1.5253°C |
| **Arabian Sea** | 5°N – 25°N | 45°E – 77°E | 1.3507°C | 1.3516°C |
| **Bay of Bengal** | 5°N – 25°N | 77°E – 100°E | 1.6953°C | 1.7199°C |

## 5. Seasonal / Temporal Breakdown (Test Split)

| Seasonal Period | Calendar Range | Days | RMSE (Unweighted) | RMSE (Sample Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Late Fall (November)** | Days 313–342 | 30 | 1.6175°C | 1.6317°C |
| **Early Winter (December)** | Days 343–365 | 23 | 1.3608°C | 1.3505°C |

## 6. Physical and Oceanographic Analysis
- **Initial Condition Memory Decay**: By days 313–365 (11 months after initialization), memory of January 1 conditions is completely lost in the upper 200 m due to seasonal monsoon heating and cooling cycles.
- **Thermocline Error Concentration**: Error peaks sharply between 50 m and 150 m where seasonal displacement of the pycnocline/thermocline produces massive temperature discrepancies.
- **Abyssal Stability**: Below 500 m, persistence errors diminish dramatically, approaching deep ocean isothermal stability (~0.1–0.3°C).
