# Final Master Results Table: Baselines B0 through B8

**Project**: Satellite Embedding-Based Deep Learning Framework for Reconstruction of Depth-Wise Subsurface Ocean Temperature  
**Domain**: North Indian Ocean (5°N–30°N, 45°E–105°E)  
**Dataset**: Full-Year 2020 Certified Production Partition (366 Days, Leap Year)  
**Target**: GLORYS12V1 Potential Temperature (15 Canonical Depths: 0 m to 1000 m)  
**Test Partition**: Days 313–365 ($N = 601,550$ test samples, frozen)

---

## 1. Master Model Evaluation Hierarchy

| Model ID | Model Name | Architecture Family | Context Representation | Parameter Count | Test RMSE (°C) | Test MAE (°C) | Test R² (vs B1) | Delta vs B1 (°C) | Relative Improvement (%) | Paired 95% Bootstrap CI vs B1 |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Day-0 Persistence | Persistence | Initial state (Day 0) | 0 | 1.5220 | 1.1371 | -0.4966 | +0.2638 | -20.97% | N/A |
| **B0b** | Day-252 Persistence | Persistence | Train boundary (Day 252) | 0 | 1.7287 | 1.2447 | -1.2358 | +0.4705 | -37.39% | N/A |
| **B1** | Spatial-Depth Climatology | Climatology | Historical train mean profile | 0 | 1.2582 | 0.9641 | 0.0000 | 0.0000 | 0.00% | Reference Anchor |
| **B2** | Multi-Output Ridge ($\alpha = 100{,}000$) | Linear Regularized | Pointwise 7-surface vector | 120 | 1.0295 | 0.8029 | -0.5404 | -0.2287 | +18.18% | [-0.3208, -0.1479] |
| **B3** | Random Forest | Bagging Ensemble | Pointwise 7-surface vector | 10,255 | 1.0452 | 0.7461 | -0.1528 | -0.2130 | +16.93% | N/A |
| **B4** | Gradient Boosting (LightGBM) | Boosting Ensemble | Pointwise 7-surface vector | 750 | 1.0288 | 0.7615 | -0.2210 | -0.2294 | +18.23% | [-0.3617, -0.1195] |
| **B5** | Pointwise MLP | Feedforward Neural | Pointwise 7-surface vector | **26,767** | 1.5524 | 1.2030 | -2.8076 | +0.2942 | -23.38% | [0.2148, 0.3772] |
| **B6** | Spatial CNN | Spatial Convolutional | 3×3 spatial patches ($P=3$) | **30,991** | 1.2702 | 0.9646 | -0.8343 | +0.0120 | -0.95% | [-0.1042, 0.1205] |
| **B7** | Temporal GRU | Sequential Recurrent | 5-day causal sequences ($T=5$) | **44,111** | 1.5320 | 1.2069 | -2.4438 | +0.2738 | -21.76% | [0.2581, 0.2895] |
| **B8** | Spatiotemporal Embedding | Joint Spatiotemporal | 5-day × 3×3 patch cubes | **203,791** | **0.9800** | **0.7391** | **-0.2040** | **-0.2782** | **+22.11%** | **[-0.3957, -0.1756]** |

> **Note on Parameter Counts**: All parameter counts are derived from direct inspection of the instantiated PyTorch and Scikit-Learn models. B5 = 26,767, B6 = 30,991, B7 = 44,111, B8 = 203,791.

---

## 2. Depth-Wise Test Error Distribution (RMSE in °C)

| Depth (m) | B0 (Day 0) | B0b (Day 252) | B1 (Climatology) | B2 (Ridge) | B3 (RF) | B4 (LightGBM) | B5 (MLP) | B6 (CNN) | B7 (GRU) | B8 (Champion) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 m** | 1.2712 | 2.2103 | 1.1158 | 0.4287 | 0.4959 | 0.4114 | 1.6577 | 1.2131 | 1.4589 | **0.4369** |
| **5 m** | 1.2761 | 2.1471 | 1.0595 | 0.4654 | 0.5050 | 0.4547 | 1.5448 | 1.2571 | 1.4084 | **0.4381** |
| **10 m** | 1.2413 | 2.0001 | 0.9913 | 0.5060 | 0.5061 | 0.5442 | 1.4122 | 1.1939 | 1.3451 | **0.4635** |
| **20 m** | 1.2451 | 1.7457 | 0.9145 | 0.6856 | 0.6476 | 0.8566 | 1.3438 | 1.2087 | 1.2816 | **0.5962** |
| **30 m** | 1.3547 | 1.5090 | 0.9566 | 0.9774 | 0.9445 | 1.2014 | 1.2373 | 1.1481 | 1.2810 | **0.8607** |
| **50 m** | 1.9890 | 1.7901 | 1.4925 | 1.3598 | 1.3648 | 1.5215 | 1.5266 | 1.3364 | 1.6785 | **1.3493** |
| **75 m** | 2.9648 | 2.4603 | 2.3034 | 1.7446 | 1.7669 | 1.9062 | 1.9267 | 1.7498 | 2.1319 | **1.8110** |
| **100 m** | 3.1785 | 2.8206 | 2.6329 | 1.7556 | 1.7092 | 1.7128 | 1.9476 | 1.9058 | 2.0737 | **1.7651** |
| **125 m** | 2.8495 | 2.8181 | 2.4589 | 1.4969 | 1.4864 | 1.5079 | 1.8911 | 1.7727 | 1.9133 | **1.5022** |
| **150 m** | 2.2196 | 2.5000 | 2.0117 | 1.3179 | 1.3566 | 1.4048 | 1.8301 | 1.5840 | 1.8289 | **1.3650** |
| **200 m** | 1.2338 | 1.7621 | 1.2421 | 1.2878 | 1.2221 | 1.1656 | 1.8391 | 1.3819 | 1.7152 | **1.1834** |
| **300 m** | 0.6829 | 0.7630 | 0.6337 | 1.1761 | 0.9541 | 0.8699 | 1.5621 | 1.1565 | 1.4836 | **0.9845** |
| **500 m** | 0.4063 | 0.4361 | 0.3411 | 0.8282 | 0.6358 | 0.6422 | 1.2918 | 0.8095 | 1.1899 | **0.6870** |
| **700 m** | 0.4583 | 0.4597 | 0.3486 | 0.7578 | 0.6513 | 0.6378 | 1.2657 | 0.7290 | 1.1740 | **0.6619** |
| **1000 m** | 0.4585 | 0.5088 | 0.3709 | 0.6542 | 0.5581 | 0.5947 | 1.0099 | 0.6060 | 1.0165 | **0.5959** |
| **Column-Avg** | **1.5220** | **1.7287** | **1.2582** | **1.0295** | **1.0452** | **1.0288** | **1.5524** | **1.2702** | **1.5320** | **0.9800** |

---

## 3. Sub-Basin & Seasonal Test Performance

### Regional Cosine-Weighted Test RMSE (°C)
- **Full Domain (5°N–30°N, 45°E–105°E)**: **0.9642 °C** ($N = 601,550$)
- **Arabian Sea Basin (5°N–25°N, 45°E–77°E)**: **1.0907 °C** ($N = 284,120$)
- **Bay of Bengal Basin (5°N–25°N, 77°E–100°E)**: **0.6775 °C** ($N = 221,840$)

### Seasonal Test RMSE (°C)
- **Late Fall Transition (Nov 09 – Nov 30, 2020)**: **1.0059 °C**
- **Early Winter Cooling (Dec 01 – Dec 31, 2020)**: **0.9060 °C**

---

## 4. Pointwise Pairwise Comparisons (B8 vs Strongest Baselines)
- **B8 vs B2 (Ridge)**:
  - $\text{RMSE}_{\text{B8}} = 0.9800\ ^\circ\text{C}$, $\text{RMSE}_{\text{B2}} = 1.0295\ ^\circ\text{C}$
  - Pointwise Difference: $\Delta\text{RMSE} = -0.0495\ ^\circ\text{C}$ (4.81% relative error reduction over Ridge).
- **B8 vs B4 (LightGBM)**:
  - $\text{RMSE}_{\text{B8}} = 0.9800\ ^\circ\text{C}$, $\text{RMSE}_{\text{B4}} = 1.0288\ ^\circ\text{C}$
  - Pointwise Difference: $\Delta\text{RMSE} = -0.0488\ ^\circ\text{C}$ (4.74% relative error reduction over LightGBM).
- **B8 vs B1 (Climatology)**:
  - Pointwise Difference: $\Delta\text{RMSE} = -0.2782\ ^\circ\text{C}$ (22.11% relative error reduction, $p < 0.001$).
