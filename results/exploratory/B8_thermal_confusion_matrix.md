# B8 Spatiotemporal Embedding Model: Thermal Regime Diagnostic Report
## Six-Regime Physical Discretization & Comparative Evaluation

---

### 1. Executive Summary
- **Evaluated Model**: Canonical B8 (Spatiotemporal Embedding Model: Conv2D + GRU bottleneck, $203,791$ parameters).
- **Evaluation Dataset**: Held-out Test Split (Days 313–365, Nov 9 – Dec 31, 2020), $8,017,734$ valid vertical target points across 15 canonical depths.
- **Model Provenance**: Faithful reproduction run using canonical configuration ($P=3$, $T=5$, $\text{embed}=128$, train-only normalization, $200\text{k}$ training samples, 4 epochs, batch size 1024, lr=0.001, seed=42). Original pre-freeze checkpoint was not preserved on disk; newly saved checkpoint (`results/exploratory/b8_checkpoint.pt`) belongs to this reproduction run.
- **Continuous Test RMSE**: Reproduction computed test RMSE = `0.9797°C` (Certified benchmark reference remains locked at `0.9800°C`).
- **Exact Thermal Accuracy**: **`81.65%`** (6,546,686 / 8,017,734 correct; vs Certified B3: `80.65%`, $\Delta = +1.00$ pp; vs B1: `76.23%`, $\Delta = +5.42$ pp).
- **Within $\pm 1$ Bin Containment**: **`99.82%`** (8,003,348 targets).
- **Beyond $\pm 1$ Bin Error Rate**: **`0.18%`** (**14,386** targets; errors are predominantly confined to the true or an adjacent temperature bin).
- **Cohen's Kappa ($\kappa$)**: **`0.7757`** (Substantial agreement).
- **Macro / Weighted F1**: **`0.8185` / `0.8192`**.

---

### 2. Physical Thermal Regime Definitions
> **Diagnostic Scope Notice**: The six temperature-based thermal regimes are diagnostic discretization intervals used to evaluate prediction fidelity on the continuous ocean temperature field. They do **not** represent static physical depth layers. Furthermore, this diagnostic classification does not establish vertical profile monotonicity or rule out localized physical temperature inversions in the continuous profile.

| Regime Class | Oceanographic Regime Name | Temperature Boundary | Diagnostic Association |
| :---: | :--- | :---: | :--- |
| **Class 0** | Deep Water | $T < 10.0^\circ\text{C}$ | Predominantly deep abyssal waters |
| **Class 1** | Lower Thermocline | $10.0 \le T < 15.0^\circ\text{C}$ | Predominantly lower thermocline base |
| **Class 2** | Core Thermocline | $15.0 \le T < 20.0^\circ\text{C}$ | Seasonal thermocline core region |
| **Class 3** | Upper Thermocline | $20.0 \le T < 25.0^\circ\text{C}$ | Dynamic upper thermocline gradient |
| **Class 4** | Subsurface Mixed Layer | $25.0 \le T < 28.0^\circ\text{C}$ | Warm subsurface mixed layer waters |
| **Class 5** | Tropical Warm Pool | $T \ge 28.0^\circ\text{C}$ | Tropical surface and near-surface warm layer |

---

### 3. Raw 6×6 Confusion Matrix (Target Sample Counts)
| True \ Pred | <10°C | 10–15°C | 15–20°C | 20–25°C | 25–28°C | ≥28°C | Total Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<10°C** | 690,806 | 106,800 | 0 | 0 | 0 | 0 | **797,606** |
| **10–15°C** | 68,920 | 1,182,856 | 124,628 | 164 | 0 | 0 | **1,376,568** |
| **15–20°C** | 0 | 53,626 | 763,744 | 161,775 | 1,277 | 0 | **980,422** |
| **20–25°C** | 0 | 0 | 82,396 | 839,780 | 156,416 | 2,048 | **1,080,640** |
| **25–28°C** | 0 | 0 | 468 | 176,864 | 1,510,260 | 51,316 | **1,738,908** |
| **≥28°C** | 0 | 0 | 0 | 10,429 | 473,921 | 1,559,240 | **2,043,590** |
| **Total Pred** | **759,726** | **1,343,282** | **971,236** | **1,189,012** | **2,141,874** | **1,612,604** | **8,017,734** |

---

### 4. Row-Normalized Confusion Matrix (% of True Regime)
| True \ Pred | <10°C | 10–15°C | 15–20°C | 20–25°C | 25–28°C | ≥28°C | Recall (Diag) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<10°C** | 86.61% | 13.39% | 0.00% | 0.00% | 0.00% | 0.00% | **86.61%** |
| **10–15°C** | 5.01% | 85.93% | 9.05% | 0.01% | 0.00% | 0.00% | **85.93%** |
| **15–20°C** | 0.00% | 5.47% | 77.90% | 16.50% | 0.13% | 0.00% | **77.90%** |
| **20–25°C** | 0.00% | 0.00% | 7.62% | 77.71% | 14.47% | 0.19% | **77.71%** |
| **25–28°C** | 0.00% | 0.00% | 0.03% | 10.17% | 86.85% | 2.95% | **86.85%** |
| **≥28°C** | 0.00% | 0.00% | 0.00% | 0.51% | 23.19% | 76.30% | **76.30%** |

---

### 5. Detailed Per-Class Diagnostic Performance
| Regime Class | Support (N) | True Positives | False Positives | False Negatives | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep Water (<10°C)** | 797,606 | 690,806 | 68,920 | 106,800 | 0.9093 | 0.8661 | **0.8872** |
| **Lower Thermocline (10–15°C)** | 1,376,568 | 1,182,856 | 160,426 | 193,712 | 0.8806 | 0.8593 | **0.8698** |
| **Core Thermocline (15–20°C)** | 980,422 | 763,744 | 207,492 | 216,678 | 0.7864 | 0.7790 | **0.7827** |
| **Upper Thermocline (20–25°C)** | 1,080,640 | 839,780 | 349,232 | 240,860 | 0.7063 | 0.7771 | **0.7400** |
| **Mixed Layer (25–28°C)** | 1,738,908 | 1,510,260 | 631,614 | 228,648 | 0.7051 | 0.8685 | **0.7783** |
| **Tropical Warm Pool (≥28°C)** | 2,043,590 | 1,559,240 | 53,364 | 484,350 | 0.9669 | 0.7630 | **0.8529** |

---

### 6. Comparative Baseline Diagnostic Ladder
| Model ID | Model Name | Test RMSE (°C) | Exact Accuracy (%) | Within ±1 Bin (%) | Beyond ±1 Bin (%) | Cohen's $\kappa$ | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Spatial-Depth Climatology | 1.2582 | 76.23% | 99.39% | 0.61% | 0.7078 | 0.7632 | 0.7604 |
| **B3** | Certified Random Forest | 1.0452 | 80.65% | 99.71% | 0.29% | 0.7636 | 0.8085 | 0.8099 |
| **B8** | Spatiotemporal Embedding | **0.9797** | **81.65%** | **99.82%** | **0.18%** | **0.7757** | **0.8185** | **0.8192** |

---

### 7. Analytical Findings & Verification
1. **Accuracy Comparison vs B3**: B8 achieved **`81.65%`**, representing a difference of **`+1.00 percentage points`** compared to Certified B3 (`80.65%`).
2. **Error Concentration**: **`99.82%`** of all predictions fall within $\pm 1$ adjacent thermal regime, confirming that errors are overwhelmingly concentrated along the sub-diagonal and super-diagonal.
3. **Extreme Outliers**: Only **`0.18%`** of test points (14,386 targets) deviate beyond $\pm 1$ regime, confirming that classification errors are predominantly confined to the true or an adjacent temperature bin.
4. **Sample Count Integrity**: Exactly `8,017,734` valid test targets evaluated; matrix row and column marginal sums strictly verify to `8,017,734`.
5. **Release Protection Notice**: This evaluation is non-canonical and exploratory. All certified production artifacts (`results/B3.json`, `results/B8.json`, `results/confusion_matrix.json`) remain strictly untouched.