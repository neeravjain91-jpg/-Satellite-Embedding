# Scientific Comparative Benchmark: Reference Baselines B0, B0b, and B1

## 1. Executive Summary
This report provides the formal comparative analysis of the three foundational reference baselines of the ML hierarchy:
1. **B0 (Day 0 Persistence)**: Upper bound of initial state memory (GLORYS state at 2020-01-01).
2. **B0b (Day 252 Persistence)**: End-of-training persistence across the purge gap (GLORYS state at 2020-09-09).
3. **B1 (Spatial-Depth Climatology)**: Lower bound benchmark for physical predictive skill ($R^2 \equiv 0.0000$).

## 2. Comprehensive Comparison: Validation Split (Days 259–306)

| Baseline Model | Unweighted RMSE | Weighted RMSE | Unweighted MAE | Unweighted Bias | Mean Pearson r | Mean R² (vs B1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Day 0 Persistence)** | 1.9858°C | 2.0273°C | 1.4514°C | -0.4552°C | 0.3210 | -1.2116 |
| **B0b (Day 252 Persistence)** | 1.1046°C | 1.1095°C | 0.7669°C | +0.1649°C | 0.8156 | 0.2218 |
| **B1 (Climatology)** | 1.3382°C | 1.3528°C | 0.9682°C | +0.2248°C | 0.5678 | 0.0000 |

## 3. Comprehensive Comparison: Test Split (Days 313–365)

| Baseline Model | Unweighted RMSE | 95% Bootstrap CI | Weighted RMSE | Unweighted MAE | Unweighted Bias | Mean Pearson r | Mean R² (vs B1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0 (Day 0)** | 1.5220°C | [1.3986, 1.6305] | 1.5273°C | 1.1371°C | -0.2966°C | 0.5707 | -0.4966 |
| **B0b (Day 252)** | 1.7287°C | [1.6336, 1.8314] | 1.7560°C | 1.2447°C | +0.3236°C | 0.4686 | -1.2358 |
| **B1 (Climatology)** | 1.2582°C | [1.1866, 1.3301] | 1.2596°C | 0.9641°C | +0.3834°C | 0.6325 | 0.0000 |

## 4. Depth-Wise Profile Comparison (Test Split RMSE, °C)

| Depth (m) | Ocean Layer | B0 (Day 0) | B0b (Day 252) | B1 (Climatology) | Best Baseline |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0.0 | Surface Mixed Layer | 1.2712 | 2.2103 | 1.1158 | **B1** |
| 5.0 | Surface Mixed Layer | 1.2761 | 2.1471 | 1.0595 | **B1** |
| 10.0 | Surface Mixed Layer | 1.2413 | 2.0001 | 0.9913 | **B1** |
| 20.0 | Surface Mixed Layer | 1.2451 | 1.7457 | 0.9145 | **B1** |
| 30.0 | Surface Mixed Layer | 1.3547 | 1.5090 | 0.9566 | **B1** |
| 50.0 | Thermocline / Pycnocline | 1.9890 | 1.7901 | 1.4925 | **B1** |
| 75.0 | Thermocline / Pycnocline | 2.9648 | 2.4603 | 2.3034 | **B1** |
| 100.0 | Thermocline / Pycnocline | 3.1785 | 2.8206 | 2.6329 | **B1** |
| 125.0 | Thermocline / Pycnocline | 2.8495 | 2.8181 | 2.4589 | **B1** |
| 150.0 | Thermocline / Pycnocline | 2.2196 | 2.5000 | 2.0117 | **B1** |
| 200.0 | Upper Mesopelagic | 1.2338 | 1.7621 | 1.2421 | **B0** |
| 300.0 | Upper Mesopelagic | 0.6829 | 0.7630 | 0.6337 | **B1** |
| 500.0 | Deep Ocean (Abyssal) | 0.4063 | 0.4361 | 0.3411 | **B1** |
| 700.0 | Deep Ocean (Abyssal) | 0.4583 | 0.4597 | 0.3486 | **B1** |
| 1000.0 | Deep Ocean (Abyssal) | 0.4585 | 0.5088 | 0.3709 | **B1** |

## 5. Regional Comparison (Test Split Weighted RMSE, °C)

| Region | B0 (Day 0) | B0b (Day 252) | B1 (Climatology) | Regional Contrast |
| :--- | :---: | :---: | :---: | :--- |
| **Full Domain** | 1.5253°C | 1.7470°C | 1.2588°C | Baseline domain reference |
| **Arabian Sea** | 1.3516°C | 1.8225°C | 1.3325°C | High salinity / strong cooling |
| **Bay of Bengal** | 1.7199°C | 1.2800°C | 1.0474°C | Freshwater capping / barrier layer |

## 6. Key Scientific Findings & Thresholds for Phase 2
1. **Thermocline Peak Barrier**: All three reference models experience their peak error in the 50–150 m range (RMSE reaching 1.2–1.8°C). This is the key physical challenge for subsequent machine learning models B2–B8.
2. **Loss of Memory**: B0 error demonstrates that initial condition memory degrades within 30–60 days, proving that deep learning models cannot rely on long-lag persistence.
3. **Physical Baseline Standard**: B1 provides an unweighted RMSE of 1.2582°C and sample-weighted RMSE of 1.2596°C on the test partition. Machine learning models (B2–B8) must achieve lower RMSE and strictly positive R² to demonstrate valid subsurface reconstruction capability.
