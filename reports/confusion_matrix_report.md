# Diagnostic Ocean Thermal Regime Confusion Matrix Report

**Dataset Split**: Held-Out Test Partition (Days 313–365, Nov 9 – Dec 31, 2020)  
**Total Evaluated Ocean Target Observations**: **8,017,734** valid physical target points across 15 canonical depths ($0$ to $1000$ m)  
**Oceanographic Reference**: GLORYS numerical ocean reanalysis reference (assessed via ARGO–GLORYS Reference Consistency Assessment; note that this comparison assesses the reanalysis reference state and does not constitute independent validation of the ML model)<br>
**Evaluated Models**: Canonical B1 (Spatial Climatology), Canonical B2 (Multi-Output Ridge), Canonical B3 (Multi-Depth Random Forest), Canonical B5 (Pointwise MLP), and Historical Exploratory Tuning Candidate (Legacy MLP)  
**Status**: OFFICIALLY AUDITED, HARMONIZED & VERIFIED  

---

## 1. Problem Formulation and Regime Definition

Subsurface ocean potential temperature ($\theta_o$) reconstruction is fundamentally a continuous depth-wise regression task across 15 vertical levels. To evaluate the preservation of vertical water-column stratification and boundary behavior, continuous temperature predictions are discretized into **six temperature-based thermal regimes defined for diagnostic classification of the continuous temperature field**.

This discretization is strictly an oceanographic diagnostic evaluation of stratification integrity; the primary project benchmark remains continuous depth-wise regression (where the B8 Spatiotemporal Embedding model achieves the leading continuous RMSE of $0.9800^\circ\text{C}$).

### Thermal Regime Definitions (Continuous, Non-Overlapping, Zero Gaps)

| Regime Index | Regime Name | Temperature Range | Dominant Associated Depths | Oceanographic Characteristics |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **Deep Water** | $< 10.0^\circ\text{C}$ | $500 - 1000\text{ m}$ | Cold, dense, weakly stratified North Indian deep water masses. |
| **1** | **Lower Thermocline** | $10.0 \le T < 15.0^\circ\text{C}$ | $200 - 300\text{ m}$ | Base of the permanent thermocline; diminishing surface atmospheric coupling. |
| **2** | **Core Thermocline** | $15.0 \le T < 20.0^\circ\text{C}$ | $100 - 150\text{ m}$ | Sharp pycnocline/thermocline; maximum vertical temperature gradient ($\partial T / \partial z$). |
| **3** | **Upper Thermocline** | $20.0 \le T < 25.0^\circ\text{C}$ | $50 - 100\text{ m}$ | Subsurface barrier layer; seasonal shoaling and mesoscale eddy variability. |
| **4** | **Subsurface Mixed Layer** | $25.0 \le T < 28.0^\circ\text{C}$ | $10 - 50\text{ m}$ | Wind-stirred euphotic layer; modulated by seasonal monsoonal forcing. |
| **5** | **Tropical Warm Pool** | $T \ge 28.0^\circ\text{C}$ | $0 - 10\text{ m}$ | High-SST North Indian Ocean warm pool waters ($\ge 28^\circ\text{C}$). |

*Note on scientific terminology*: These six regimes represent physically motivated temperature intervals for diagnostic classification. They do not constitute an international or WMO/IOC standard classification. Warm pool temperatures ($\ge 28^\circ\text{C}$) occur across both equatorial and northern basin sectors during warm seasons, and are designated as the Tropical Warm Pool.

---

## 2. Comparative Diagnostic Benchmark Summary

All evaluations are conducted over the exact held-out test split of **$8,017,734$** valid observations ($N$ confirmed by masking target NaNs without artificial zero-filling).

| Model ID | Model Architecture | Parameters / Complexity | Exact Accuracy | Within $\pm 1$ Bin | Beyond $\pm 1$ Bin | Cohen's Kappa ($\kappa$) | Macro F1 | Weighted F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Spatial-Depth Climatology | 0 (Hist Mean) | 76.23% | 99.39% | 0.61% | 0.7078 | 0.7632 | 0.7604 |
| **B2** | Multi-Output Ridge ($\alpha^*=10^5$) | 120 coefs | 79.87% | 99.91% | 0.09% | 0.7538 | 0.8023 | 0.8014 |
| **B3** | Multi-Depth Random Forest | 13,289,966 nodes (750 trees) | **80.65%** | 99.71% | 0.29% | **0.7636** | **0.8085** | **0.8099** |
| **B5** | Pointwise MLP | 26,767 weights | 77.67% | 99.85% | 0.15% | 0.7275 | 0.7876 | 0.7797 |
| *Ref* | *Legacy Exploratory MLP (`MLP_128_64`)\** | *10,255 weights* | *81.19%* | *99.84%* | *0.16%* | *0.7699* | *0.8133* | *0.8146* |

*\*Provenance Disclosure on Legacy Artifact*: The 81.19% accuracy figure previously reported originated from an exploratory hyperparameter tuning candidate (`models/checkpoints/b3_MLP_128_64_best.pt`, 10,255 parameters). Under the locked benchmark hierarchy, B3 is canonically the Multi-Depth Random Forest ($80.65\%$ diagnostic accuracy, $1.0452^\circ\text{C}$ continuous test RMSE), and B5 is the Pointwise MLP with 26,767 parameters ($77.67\%$ diagnostic accuracy, $1.5524^\circ\text{C}$ continuous test RMSE).

---

## 3. Canonical B3: Multi-Depth Random Forest Confusion Matrix

- **Architecture**: 15 depth-wise Random Forest models (`sklearn.ensemble.RandomForestRegressor`), 50 trees each, max depth 15 ($13,289,966$ total decision nodes).
- **Evaluation Provenance**: B3 classification metrics are reproduced by refitting the canonical B3 training protocol with the locked seed (`random_state=42`) and 100,000-row training subsample, then evaluating on the frozen held-out test partition.
- **Exact Accuracy**: **80.65%** ($6,466,349$ correctly classified points out of $8,017,734$).
- **Within $\pm 1$ Bin Tolerance**: **99.71%** ($7,994,254 / 8,017,734$).
- **Beyond $\pm 1$ Bin Error**: **0.29%** ($23,480 / 8,017,734$).
- **Cohen's Kappa**: **0.7636**.

### Raw Sample Counts ($N = 8,017,734$)

| Actual \ Predicted | <10°C (Deep) | 10–15°C (Low-TC) | 15–20°C (Core-TC) | 20–25°C (Upp-TC) | 25–28°C (Mixed) | ≥28°C (Warm Pool) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<10°C (Deep)** | **687,687** | 109,919 | 0 | 0 | 0 | 0 | **797,606** |
| **10–15°C (Low-TC)** | 70,667 | **1,176,049** | 129,101 | 749 | 2 | 0 | **1,376,568** |
| **15–20°C (Core-TC)** | 0 | 53,055 | **751,149** | 173,738 | 2,480 | 0 | **980,422** |
| **20–25°C (Upp-TC)** | 0 | 0 | 88,573 | **838,458** | 151,940 | 1,669 | **1,080,640** |
| **25–28°C (Mixed)** | 0 | 0 | 719 | 229,230 | **1,444,916** | 64,043 | **1,738,908** |
| **≥28°C (Warm Pool)** | 0 | 0 | 0 | 17,861 | 457,639 | **1,568,090** | **2,043,590** |
| **Total Predicted** | **758,354** | **1,339,023** | **969,542** | **1,260,036** | **2,056,977** | **1,633,802** | **8,017,734** |

### Per-Class Performance Metrics (Canonical B3 Random Forest)

| Thermal Regime Tier | Support ($N$) | True Positives | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Water (<10°C)** | 797,606 | 687,687 | 90.68% | 86.22% | 0.8839 |
| **Lower Thermocline (10–15°C)** | 1,376,568 | 1,176,049 | 87.83% | 85.43% | 0.8661 |
| **Core Thermocline (15–20°C)** | 980,422 | 751,149 | 77.47% | 76.61% | 0.7704 |
| **Upper Thermocline (20–25°C)** | 1,080,640 | 838,458 | 66.54% | 77.59% | 0.7164 |
| **Subsurface Mixed Layer (25–28°C)** | 1,738,908 | 1,444,916 | 70.24% | 83.09% | 0.7613 |
| **Tropical Warm Pool (≥28°C)** | 2,043,590 | 1,568,090 | 95.98% | 76.73% | 0.8528 |
| **Macro Average** | — | — | **81.46%** | **80.95%** | **0.8085** |
| **Weighted Average** | **8,017,734** | **6,466,349** | **81.65%** | **80.65%** | **0.8099** |

---

## 4. Canonical B5: Pointwise Multi-Layer Perceptron Confusion Matrix

- **Architecture**: PyTorch Pointwise MLP (`Linear(7, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 15)`), 26,767 trainable parameters.
- **Exact Accuracy**: **77.67%** ($6,227,512$ correctly classified points out of $8,017,734$).
- **Within $\pm 1$ Bin Tolerance**: **99.85%** ($8,006,024 / 8,017,734$).
- **Beyond $\pm 1$ Bin Error**: **0.15%** ($11,710 / 8,017,734$).
- **Cohen's Kappa**: **0.7275**.

### Raw Sample Counts ($N = 8,017,734$)

| Actual \ Predicted | <10°C (Deep) | 10–15°C (Low-TC) | 15–20°C (Core-TC) | 20–25°C (Upp-TC) | 25–28°C (Mixed) | ≥28°C (Warm Pool) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<10°C (Deep)** | **709,463** | 88,143 | 0 | 0 | 0 | 0 | **797,606** |
| **10–15°C (Low-TC)** | 100,148 | **1,145,665** | 130,470 | 285 | 0 | 0 | **1,376,568** |
| **15–20°C (Core-TC)** | 0 | 55,776 | **747,934** | 174,469 | 2,241 | 2 | **980,422** |
| **20–25°C (Upp-TC)** | 0 | 0 | 80,690 | **831,244** | 166,712 | 1,994 | **1,080,640** |
| **25–28°C (Mixed)** | 0 | 0 | 42 | 211,206 | **1,441,628** | 86,032 | **1,738,908** |
| **≥28°C (Warm Pool)** | 0 | 0 | 0 | 7,146 | 684,866 | **1,351,578** | **2,043,590** |
| **Total Predicted** | **809,611** | **1,289,584** | **959,136** | **1,224,350** | **2,295,447** | **1,439,606** | **8,017,734** |

### Per-Class Performance Metrics (Canonical B5 Pointwise MLP)

| Thermal Regime Tier | Support ($N$) | True Positives | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Water (<10°C)** | 797,606 | 709,463 | 87.63% | 88.95% | 0.8828 |
| **Lower Thermocline (10–15°C)** | 1,376,568 | 1,145,665 | 88.84% | 83.23% | 0.8594 |
| **Core Thermocline (15–20°C)** | 980,422 | 747,934 | 77.98% | 76.29% | 0.7712 |
| **Upper Thermocline (20–25°C)** | 1,080,640 | 831,244 | 67.89% | 76.92% | 0.7213 |
| **Subsurface Mixed Layer (25–28°C)** | 1,738,908 | 1,441,628 | 62.80% | 82.90% | 0.7147 |
| **Tropical Warm Pool (≥28°C)** | 2,043,590 | 1,351,578 | 93.89% | 66.14% | 0.7761 |
| **Macro Average** | — | — | **79.84%** | **79.07%** | **0.7876** |
| **Weighted Average** | **8,017,734** | **6,227,512** | **80.08%** | **77.67%** | **0.7797** |

---

## 5. Canonical B2: Multi-Output Ridge Regression Confusion Matrix

- **Architecture**: 15 depth-wise Ridge models with $\alpha^* = 100,000$ (120 coefficients total).
- **Exact Accuracy**: **79.87%** ($6,404,011 / 8,017,734$).
- **Within $\pm 1$ Bin Tolerance**: **99.91%** ($8,010,752 / 8,017,734$).
- **Beyond $\pm 1$ Bin Error**: **0.09%** ($6,982 / 8,017,734$).
- **Cohen's Kappa**: **0.7538**.

### Raw Sample Counts ($N = 8,017,734$)

| Actual \ Predicted | <10°C (Deep) | 10–15°C (Low-TC) | 15–20°C (Core-TC) | 20–25°C (Upp-TC) | 25–28°C (Mixed) | ≥28°C (Warm Pool) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<10°C (Deep)** | **672,961** | 124,645 | 0 | 0 | 0 | 0 | **797,606** |
| **10–15°C (Low-TC)** | 71,449 | **1,161,093** | 143,897 | 129 | 0 | 0 | **1,376,568** |
| **15–20°C (Core-TC)** | 53 | 65,071 | **738,559** | 174,537 | 2,193 | 9 | **980,422** |
| **20–25°C (Upp-TC)** | 0 | 0 | 73,527 | **841,146** | 164,101 | 1,866 | **1,080,640** |
| **25–28°C (Mixed)** | 0 | 0 | 0 | 168,000 | **1,496,511** | 74,397 | **1,738,908** |
| **≥28°C (Warm Pool)** | 0 | 0 | 0 | 2,732 | 547,117 | **1,493,741** | **2,043,590** |
| **Total Predicted** | **744,463** | **1,350,809** | **955,983** | **1,186,544** | **2,209,922** | **1,570,013** | **8,017,734** |

---

## 6. Canonical B1: Spatial-Depth Climatology Confusion Matrix

- **Architecture**: Historical train-split mean field across coordinates and depths (0 parameters).
- **Exact Accuracy**: **76.23%** ($6,112,280 / 8,017,734$).
- **Within $\pm 1$ Bin Tolerance**: **99.39%** ($7,969,092 / 8,017,734$).
- **Beyond $\pm 1$ Bin Error**: **0.61%** ($48,642 / 8,017,734$).
- **Cohen's Kappa**: **0.7078**.

### Raw Sample Counts ($N = 8,017,734$)

| Actual \ Predicted | <10°C (Deep) | 10–15°C (Low-TC) | 15–20°C (Core-TC) | 20–25°C (Upp-TC) | 25–28°C (Mixed) | ≥28°C (Warm Pool) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<10°C (Deep)** | **740,665** | 56,941 | 0 | 0 | 0 | 0 | **797,606** |
| **10–15°C (Low-TC)** | 67,797 | **1,185,362** | 118,851 | 4,270 | 288 | 0 | **1,376,568** |
| **15–20°C (Core-TC)** | 358 | 51,755 | **755,302** | 163,044 | 9,963 | 0 | **980,422** |
| **20–25°C (Upp-TC)** | 0 | 17 | 120,530 | **754,233** | 197,358 | 8,502 | **1,080,640** |
| **25–28°C (Mixed)** | 0 | 0 | 2,752 | 198,118 | **1,400,240** | 137,798 | **1,738,908** |
| **≥28°C (Warm Pool)** | 0 | 0 | 0 | 18,349 | 748,763 | **1,276,478** | **2,043,590** |
| **Total Predicted** | **808,820** | **1,294,075** | **997,435** | **1,138,014** | **2,356,612** | **1,422,778** | **8,017,734** |

---

## 7. Historical Exploratory Tuning Candidate (Legacy MLP)

- **Checkpoint**: `models/checkpoints/b3_MLP_128_64_best.pt`
- **Architecture**: Pointwise MLP (`Linear(7, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 15)`), 10,255 parameters.
- **Exact Accuracy**: **81.19%** ($6,509,625 / 8,017,734$).
- **Within $\pm 1$ Bin Tolerance**: **99.84%** ($8,005,155 / 8,017,734$).
- **Beyond $\pm 1$ Bin Error**: **0.16%** ($12,579 / 8,017,734$).
- **Cohen's Kappa**: **0.7699**.

### Raw Sample Counts ($N = 8,017,734$)

| Actual \ Predicted | <10°C (Deep) | 10–15°C (Low-TC) | 15–20°C (Core-TC) | 20–25°C (Upp-TC) | 25–28°C (Mixed) | ≥28°C (Warm Pool) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<10°C (Deep)** | **695,353** | 102,253 | 0 | 0 | 0 | 0 | **797,606** |
| **10–15°C (Low-TC)** | 75,342 | **1,163,137** | 137,834 | 255 | 0 | 0 | **1,376,568** |
| **15–20°C (Core-TC)** | 0 | 47,705 | **761,959** | 168,932 | 1,821 | 5 | **980,422** |
| **20–25°C (Upp-TC)** | 0 | 0 | 87,190 | **824,379** | 167,082 | 1,989 | **1,080,640** |
| **25–28°C (Mixed)** | 0 | 0 | 1 | 180,261 | **1,457,588** | 101,058 | **1,738,908** |
| **≥28°C (Warm Pool)** | 0 | 0 | 0 | 8,508 | 427,873 | **1,607,209** | **2,043,590** |
| **Total Predicted** | **770,695** | **1,313,095** | **986,984** | **1,182,335** | **2,054,364** | **1,710,261** | **8,017,734** |

---

## 8. Physical Oceanographic Interpretation & Error Adjacency

1. **Local Error Adjacency**:
   - More than 99.7% of evaluated predictions across all supervised ML models fall within the true thermal-regime bin or an immediately adjacent bin ($99.91\%$ for B2 Ridge, $99.85\%$ for B5 Pointwise MLP, $99.71\%$ for B3 Random Forest, $99.84\%$ for Legacy MLP, and $99.39\%$ for B1 Climatology).
   - This indicates that most classification errors are local in regime space and are concentrated near continuous temperature-regime boundaries.
   - Non-adjacent misclassifications (beyond $\pm 1$ bin) range from a low of **0.09%** (B2 Ridge) to **0.29%** (B3 Random Forest). Non-adjacent errors exist but remain rare (<0.30% across all supervised ML models).
   - *Important constraint*: High $\pm 1$-bin containment does not prove vertical monotonicity across individual depth profiles, nor does discrete regime grouping preclude localized gradient inversions. Vertical thermal structure is formally evaluated via continuous profile metrics.

2. **Comparative Model Behavior**:
   - **Canonical B3 Random Forest** achieves the highest discrete regime classification score among certified baselines with **80.65%** exact accuracy and $\kappa = 0.7636$, outperforming spatial climatology (76.23%, $+4.42\%$). (Note: No formal paired significance test has been performed for discrete classification accuracy; significance tests are certified for continuous RMSE).
   - **Canonical B2 Ridge** delivers **79.87%** exact accuracy with the highest near-neighbor containment (99.91% within $\pm 1$ bin; 0.09% beyond), demonstrating that regularized linear column projections yield high local regime containment along the vertical thermal gradient.
   - **Canonical B5 Pointwise MLP** achieves **77.67%** exact accuracy ($\kappa = 0.7275$), outperforming climatology by $+1.44\%$, but exhibits lower overall skill than tree-based ensembles when spatial/temporal context is excluded.

3. **Data Source & Mask Integrity**:
   - Target reference: GLORYS numerical ocean reanalysis reference (compared against in-situ ARGO float profiles in the ARGO–GLORYS Reference Consistency Assessment; note that this comparison assesses the reanalysis reference state and does not constitute independent validation of the ML model).
   - Surface salinity: Copernicus Multi-Observation SSS.
   - Masking: Exact 4-way target masks applied; bathymetric limits rigorously maintained without artificial zero-filling.
