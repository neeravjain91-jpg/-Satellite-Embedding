# B3 Random Forest Exploratory Hyperparameter Study
## Controlled Validation Optimization and Frozen-Test Verification

---

### Protocol Disclosure
- **Purpose**: Controlled exploratory hyperparameter tuning for Baseline B3 (Multi-Depth Random Forest).
- **Leakage Prohibition**: Models trained **strictly** on the canonical TRAIN split (Days 0–252, $N=2,871,550$).
- **Selection Rule**: Configuration selection performed **strictly on the VALIDATION split** (Days 259–306, $N=544,800$).
- **Test Partition Integrity**: The frozen held-out TEST partition (Days 313–365, $N=601,550$ columns, $8,017,734$ valid depth targets) was evaluated **exactly once** for the single validation winner.
- **Certified Results Guarantee**: Certified production baseline results (`results/B3.json`, `1.0452 °C`, `80.65%`) remain **FROZEN and UNCHANGED**.

---

### 1. Evaluated Configurations
| Model ID | Configuration Parameters | Sample Train Size | Trees per Regressor | Total Trees (15 Depths) |
| :--- | :--- | :---: | :---: | :---: |
| **Certified B3** | $n=50$, $\text{max\_depth}=15$, $\text{leaf}=1$, $\text{feat}=1.0$ | 100,000 | 50 | 750 |
| **B3-A** | $n=100$, $\text{max\_depth}=15$, $\text{leaf}=1$, $\text{feat}=1.0$ | 100,000 | 100 | 1500 |
| **B3-B** | $n=100$, $\text{max\_depth}=20$, $\text{leaf}=1$, $\text{feat}=1.0$ | 100,000 | 100 | 1500 |
| **B3-C** | $n=150$, $\text{max\_depth}=18$, $\text{leaf}=1$, $\text{feat}=1.0$ | 200,000 | 150 | 2250 |

---

### 2. Validation Performance Matrix (Hyperparameter Selection)
| Rank | Model ID | Val RMSE (°C) | Val MAE (°C) | Val R² (vs B1) | Val Bias (°C) | Val Thermal Acc (%) | Val ±1-Bin (%) | Val Cohen's $\kappa$ | Val Macro F1 | Decision Nodes | Train Time (s) | Eval Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ref | **Certified B3** | 1.0467 | 0.7660 | -0.1211 | +0.0460 | 82.06% | 99.69% | 0.7776 | 0.8102 | 13,289,966 | ~55s | ~8s |
| 1 | **B3-A** *(Winner)* | **1.0437** | 0.7638 | -0.1141 | +0.0454 | 82.12% | 99.69% | 0.7783 | 0.8107 | 26,549,170 | 86.2s | 9.1s |
| 2 | **B3-C** | **1.0457** | 0.7640 | -0.1280 | +0.0497 | 82.20% | 99.70% | 0.7793 | 0.8112 | 131,541,724 | 345.7s | 25.9s |
| 3 | **B3-B** | **1.0461** | 0.7661 | -0.1172 | +0.0404 | 82.04% | 99.69% | 0.7774 | 0.8101 | 85,230,172 | 104.0s | 13.4s |

> **Validation Selection Decision**: **B3-A** is selected as the primary winner having achieved the lowest unweighted Validation RMSE (1.0437 °C).

---

### 3. Frozen-Test Evaluation of Validation Winner (B3-A)
| Model ID | Test Split Role | Test RMSE (°C) | Test MAE (°C) | Test R² (vs B1) | Test Bias (°C) | Test Thermal Acc (%) | Test ±1-Bin (%) | Test Cohen's $\kappa$ | Test Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Certified B3** | Locked Baseline | 1.0452 | 0.7725 | -0.2584 | -0.0235 | 80.65% | 99.71% | 0.7636 | 0.8085 |
| **B3-A** | Exploratory Candidate | **1.0412** | 0.7705 | -0.2518 | -0.0216 | **80.68%** | 99.71% | 0.7639 | 0.8088 |

---

### 4. Direct Quantitative Comparison Against Certified B3
| Metric | Certified B3 | Validation Winner (B3-A) | Delta / Change | Formula |
| :--- | :---: | :---: | :---: | :--- |
| **Test Column-Avg RMSE** | 1.0452 °C | 1.0412 °C | **-0.0040 °C** | `winner_test_rmse - 1.0452` |
| **Test Thermal Regime Accuracy** | 80.65% | 80.68% | **+0.03%** | `winner_test_thermal_accuracy - 80.65` |
| **Decision Tree Nodes** | 13,289,966 | 26,549,170 | **+13,259,204 nodes** | Complexity increment |

---

### 5. Depth-Wise Test Error Distribution (RMSE in °C)
| Depth (m) | Certified B3 | B3-A | Delta (°C) | Relative Gain (%) |
| :---: | :---: | :---: | :---: | :---: |
| **0 m** | 0.4294 | 0.4284 | -0.0010 | +0.23% |
| **5 m** | 0.4754 | 0.4716 | -0.0038 | +0.80% |
| **10 m** | 0.5553 | 0.5457 | -0.0096 | +1.73% |
| **20 m** | 0.8694 | 0.8606 | -0.0088 | +1.01% |
| **30 m** | 1.2633 | 1.2556 | -0.0077 | +0.61% |
| **50 m** | 1.4528 | 1.4516 | -0.0012 | +0.08% |
| **75 m** | 1.9606 | 1.9302 | -0.0304 | +1.55% |
| **100 m** | 1.7325 | 1.7362 | +0.0037 | -0.21% |
| **125 m** | 1.5489 | 1.5475 | -0.0014 | +0.09% |
| **150 m** | 1.4173 | 1.4223 | +0.0050 | -0.35% |
| **200 m** | 1.1922 | 1.1912 | -0.0010 | +0.08% |
| **300 m** | 0.8870 | 0.8840 | -0.0030 | +0.34% |
| **500 m** | 0.6661 | 0.6641 | -0.0020 | +0.30% |
| **700 m** | 0.6339 | 0.6333 | -0.0006 | +0.09% |
| **1000 m** | 0.5942 | 0.5963 | +0.0021 | -0.35% |

---

### 6. Summary & Recommendation
- **Validation Ranking**: The 3 exploratory configurations were ranked on validation RMSE as follows: `B3-A > B3-C > B3-B`.
- **Test Performance**: On the held-out test split, **B3-A** achieved a test RMSE of `1.0412 °C` (change: `-0.0040 °C`) and a thermal-regime accuracy of `80.68%` (gain: `+0.03%`).
- **Integrity Status**: Under the user-directed release freeze protocol, **Certified B3 remains locked at 1.0452 °C and 80.65%**. These exploratory results are recorded strictly for research documentation without modifying any production benchmark artifacts.