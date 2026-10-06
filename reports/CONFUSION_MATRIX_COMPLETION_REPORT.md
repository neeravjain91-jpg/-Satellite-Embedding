# Project Completion Report: Phase 1 Thermal-Regime Confusion Matrix Repair & Benchmark Harmonization

**Repository**: `neeravjain91-jpg/-Satellite-Embedding`  
**Execution Date**: 2026-10-05  
**Evaluation Scope**: Full-Year 2020 Certified Production Dataset, Held-Out Test Split (Days 313–365, Nov 9 – Dec 31, 2020)  
**Total Valid Physical Test Targets ($N$)**: **8,017,734** points across 15 vertical depths ($0$ to $1000$ m)  
**Status**: OFFICIALLY HARMONIZED, VERIFIED, AND ACCEPTED  

---

## 1. Executive Summary

This project phase resolved a critical architectural and metadata collision in the benchmark evaluation pipeline and finalized the thermal-regime diagnostic classification framework to achieve total scientific consistency with the canonical B0–B8 benchmark hierarchy.

### Key Actions Completed
1. **Resolved Model Identity Collision**:
   - Traced git history to commit `68dc37b48279bc41a1daa639da3a82bd2fb8d9bf` to confirm canonical B3 identity as the Multi-Depth Random Forest ($13,289,966$ nodes across 750 trees, test RMSE $1.0452^\circ\text{C}$).
   - Restored `results/B3.json` to this verified Random Forest baseline (SHA-256: `4ae89a43154d00e4dbb4362c9490a6a7074a9d839ac00c9e33155501153cc548`).
   - Confirmed canonical B5 identity as the Pointwise MLP (`Linear(7, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 15)`, 26,767 trainable parameters, continuous test RMSE $1.5524^\circ\text{C}$).
   - Synchronized `results/master_benchmark_summary.json` and `reports/master_benchmark_summary.md`.

2. **Diagnosed 81.19% Confusion Matrix Provenance (Decision Rule Case B)**:
   - Provenance analysis established that the previously reported 81.19% exact accuracy ($N=8,017,734$, Cohen's Kappa $\kappa=0.7699$) originated from an exploratory hyperparameter tuning candidate (`models/checkpoints/b3_MLP_128_64_best.pt`, 10,255 parameters, 2 hidden layers).
   - In accordance with Decision Rule Case B, this candidate was **not** rebranded as canonical B3 and was **not** promoted as canonical B5. It is documented transparently as an exploratory tuning reference.
   - Evaluated genuine diagnostic classification metrics for all canonical benchmark models (B1, B2, B3, B5) on the exact $8,017,734$ test observations.

3. **Standardized Oceanographic Terminology & Physics**:
   - Replaced all informal/standardization claims with the formal definition: *"Six temperature-based thermal regimes defined for diagnostic classification of the continuous temperature field."*
   - Purged references to "equatorial warm pool", adopting the geographically correct designation *"Tropical Warm Pool ($\ge 28^\circ\text{C}$)"*.
   - Strictly enforced reference terminology: GLORYS is designated as the *"GLORYS numerical ocean reanalysis reference"*, and ARGO matchups are designated as the *"ARGO–GLORYS Reference Consistency Assessment"* (noting that because operational ARGO observations are assimilated into GLORYS, this comparison evaluates the reanalysis reference state rather than serving as independent validation of the ML model).
   - Replaced causal and proof claims with descriptive, mathematically verified statements: more than $99.7\%$ of evaluated predictions fall within the true thermal-regime bin or an immediately adjacent bin, indicating local regime containment along continuous boundaries.

4. **Refactored Training and Evaluation Scripts**:
   - Established `scripts/train_b5.py` to train/evaluate canonical B5 Pointwise MLP (26,767 parameters).
   - Refactored `scripts/train_b3.py` to train/evaluate canonical B3 Multi-Depth Random Forest.
   - Upgraded `scripts/evaluate_confusion_matrix.py` to evaluate all canonical baselines and legacy candidates. Explicitly documented that B3 classification metrics are reproduced by refitting the canonical B3 training protocol with the locked seed (`random_state=42`) and 100,000-row training subsample, then evaluating on the frozen held-out test partition.
   - Synchronized `results/confusion_matrix.json`, `reports/confusion_matrix_report.md`, and the interactive artifact `confusion_matrix.html`.

---

## 2. Canonical Model Identities & Benchmark Hierarchy

The authoritative B0–B8 benchmark hierarchy is locked as follows:

| Model ID | Canonical Name | Architecture / Family | Trainable Parameters / Complexity | Test Column-Avg RMSE (°C) | Benchmark Role |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **B0** | Day-0 Persistence | Temporal Initial State | 0 | 1.5220 | Reference persistence baseline (initial day) |
| **B0b** | Day-252 Persistence | Train-Boundary State | 0 | 1.7287 | Boundary persistence baseline (end of training period) |
| **B1** | Spatial-Depth Climatology | Historical Train Mean Field | 0 | 1.2582 | Primary historical climatological reference |
| **B2** | Multi-Output Ridge Regression | 15 Depth Linear Regressors ($\alpha^*=100,000$) | 120 coefficients | 1.0295 | Certified linear supervised ML baseline |
| **B3** | Multi-Depth Random Forest | 15 Depth Random Forest Models ($50\times 15 = 750$ trees) | 13,289,966 nodes | 1.0452 | Bagging non-linear ensemble baseline |
| **B4** | Gradient Boosted Decision Trees | LightGBM 4.7.0 ($50\times 15 = 750$ trees) | 750 trees | 1.0288 | Boosting non-linear ensemble baseline |
| **B5** | Pointwise Multi-Layer Perceptron | PyTorch Feedforward (`128-128-64`) | 26,767 weights | 1.5524 | Pointwise neural baseline (zero spatial/temporal context) |
| **B6** | Spatial CNN | PyTorch Conv2D ($3\times 3$ patches) | 30,991 weights | 1.2702 | Spatial context neural baseline |
| **B7** | Temporal GRU | PyTorch Causal GRU ($T=5$ history) | 44,111 weights | 1.5320 | Temporal sequential neural baseline |
| **B8** | Spatiotemporal Embedding Model | Joint Conv2D + GRU Latent Bottleneck | 203,791 weights | **0.9800** | Best-performing model among evaluated benchmarks |

*Note on Model Complexity*: Neural and linear baselines report trainable weights/coefficients (B2: 120, B5: 26,767, B6: 30,991, B7: 44,111, B8: 203,791). Decision tree ensembles report architectural complexity (B3 Random Forest: 750 trees with 13,289,966 total decision nodes; B4 LightGBM: 750 boosting trees).

---

## 3. Provenance Audit of the 81.19% Confusion Matrix

The previous confusion matrix artifact reported:
- Exact Accuracy: $81.19\%$
- Within $\pm 1$ Bin: $99.84\%$
- Beyond $\pm 1$ Bin: $0.16\%$
- Cohen's Kappa: $\kappa = 0.7699$
- Total Sample Count: $N = 8,017,734$

### Provenance Determination (Decision Rule Case B)
- **Origin**: Evaluation of `models/checkpoints/b3_MLP_128_64_best.pt` on the 8,017,734 test points.
- **Model Topology**: `PointwiseMLPNet(7, [128, 64], 15)` with 10,255 trainable parameters.
- **Finding**: This was an exploratory hyperparameter tuning candidate evaluated during earlier Phase 3 experiments. It is **not** canonical B3 (which is Random Forest) and **not** canonical B5 (which has 26,767 parameters, hidden dims `[128, 128, 64]`, and achieves $77.67\%$ classification accuracy).
- **Resolution**: Under Decision Rule Case B, the legacy 10,255-parameter model is transparently documented as a historical tuning candidate. The canonical models (B1, B2, B3, B5) have been evaluated to establish genuine, reproducible classification metrics.
- **B3 Reproducibility Provenance**: B3 classification metrics are reproduced by refitting the canonical B3 training protocol with the locked seed (`random_state=42`) and 100,000-row training subsample, then evaluating on the frozen held-out test partition.

---

## 4. Physical Thermal Regime Definitions

Vertical water-column potential temperature ($\theta_o$) across the North Indian Ocean ($5^\circ\text{N} - 30^\circ\text{N}$, $45^\circ\text{E} - 105^\circ\text{E}$) is partitioned into six contiguous, non-overlapping temperature intervals:

$$\text{Regime Bins} = [-\infty, 10.0^\circ\text{C}, 15.0^\circ\text{C}, 20.0^\circ\text{C}, 25.0^\circ\text{C}, 28.0^\circ\text{C}, +\infty]$$

| Regime Index | Regime Name | Range ($^\circ\text{C}$) | Dominant Canonical Depths | Physical Water-Column Context |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **Deep Water** | $T < 10.0$ | $500 - 1000\text{ m}$ | Cold, dense, weakly stratified deep water masses below the permanent thermocline. |
| **1** | **Lower Thermocline** | $10.0 \le T < 15.0$ | $200 - 300\text{ m}$ | Base of the permanent thermocline; weak seasonal atmospheric coupling. |
| **2** | **Core Thermocline** | $15.0 \le T < 20.0$ | $100 - 150\text{ m}$ | Pycnocline/thermocline core; maximum vertical temperature gradient ($\partial T / \partial z$). |
| **3** | **Upper Thermocline** | $20.0 \le T < 25.0$ | $50 - 100\text{ m}$ | Subsurface barrier layer; intense eddy pumping and seasonal monsoonal shoaling. |
| **4** | **Subsurface Mixed Layer** | $25.0 \le T < 28.0$ | $10 - 50\text{ m}$ | Wind-stirred euphotic layer; direct atmospheric forcing. |
| **5** | **Tropical Warm Pool** | $T \ge 28.0$ | $0 - 10\text{ m}$ | High-SST North Indian Ocean warm pool waters ($\ge 28^\circ\text{C}$). |

---

## 5. Comprehensive Diagnostic Classification Results

Evaluated over the held-out test split of **$8,017,734$** valid target observations:

| Metric | B1 Climatology | B2 Ridge | B3 Random Forest | B5 Pointwise MLP | Legacy Tuning MLP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Accuracy** | **76.23%** | **79.87%** | **80.65%** | **77.67%** | **81.19%** |
| **True Positives (Diagonal)** | 6,112,280 | 6,404,011 | 6,466,349 | 6,227,512 | 6,509,625 |
| **Within $\pm 1$ Bin Tolerance** | **99.39%** | **99.91%** | **99.71%** | **99.85%** | **99.84%** |
| **Beyond $\pm 1$ Bin Error** | **0.61%** | **0.09%** | **0.29%** | **0.15%** | **0.16%** |
| **Cohen's Kappa ($\kappa$)** | **0.7078** | **0.7538** | **0.7636** | **0.7275** | **0.7699** |
| **Macro F1-Score** | 0.7632 | 0.8023 | 0.8085 | 0.7876 | 0.8133 |
| **Weighted F1-Score** | 0.7604 | 0.8014 | 0.8099 | 0.7797 | 0.8146 |
| **Model Parameters** | 0 | 120 | 13,289,966 nodes | 26,767 | 10,255 |

---

## 6. Confusion Matrices and Per-Class Breakdowns

### 6.1 Canonical B3 Multi-Depth Random Forest ($80.65\%$ Accuracy, $\kappa = 0.7636$)

#### Raw Confusion Matrix ($N = 8,017,734$)

| Actual \ Predicted | <10°C (Deep) | 10–15°C (Low-TC) | 15–20°C (Core-TC) | 20–25°C (Upp-TC) | 25–28°C (Mixed) | ≥28°C (Warm Pool) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<10°C (Deep)** | **687,687** | 109,919 | 0 | 0 | 0 | 0 | **797,606** |
| **10–15°C (Low-TC)** | 70,667 | **1,176,049** | 129,101 | 749 | 2 | 0 | **1,376,568** |
| **15–20°C (Core-TC)** | 0 | 53,055 | **751,149** | 173,738 | 2,480 | 0 | **980,422** |
| **20–25°C (Upp-TC)** | 0 | 0 | 88,573 | **838,458** | 151,940 | 1,669 | **1,080,640** |
| **25–28°C (Mixed)** | 0 | 0 | 719 | 229,230 | **1,444,916** | 64,043 | **1,738,908** |
| **≥28°C (Warm Pool)** | 0 | 0 | 0 | 17,861 | 457,639 | **1,568,090** | **2,043,590** |
| **Total Predicted** | **758,354** | **1,339,023** | **969,542** | **1,260,036** | **2,056,977** | **1,633,802** | **8,017,734** |

#### Per-Class Metrics (Canonical B3 Random Forest)
- **Deep Water (<10°C)**: Support: 797,606 | Precision: 90.68% | Recall: 86.22% | F1: 0.8839
- **Lower Thermocline (10–15°C)**: Support: 1,376,568 | Precision: 87.83% | Recall: 85.43% | F1: 0.8661
- **Core Thermocline (15–20°C)**: Support: 980,422 | Precision: 77.47% | Recall: 76.61% | F1: 0.7704
- **Upper Thermocline (20–25°C)**: Support: 1,080,640 | Precision: 66.54% | Recall: 77.59% | F1: 0.7164
- **Subsurface Mixed Layer (25–28°C)**: Support: 1,738,908 | Precision: 70.24% | Recall: 83.09% | F1: 0.7613
- **Tropical Warm Pool (≥28°C)**: Support: 2,043,590 | Precision: 95.98% | Recall: 76.73% | F1: 0.8528

---

### 6.2 Canonical B5 Pointwise MLP ($77.67\%$ Accuracy, $\kappa = 0.7275$)

#### Raw Confusion Matrix ($N = 8,017,734$)

| Actual \ Predicted | <10°C (Deep) | 10–15°C (Low-TC) | 15–20°C (Core-TC) | 20–25°C (Upp-TC) | 25–28°C (Mixed) | ≥28°C (Warm Pool) | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<10°C (Deep)** | **709,463** | 88,143 | 0 | 0 | 0 | 0 | **797,606** |
| **10–15°C (Low-TC)** | 100,148 | **1,145,665** | 130,470 | 285 | 0 | 0 | **1,376,568** |
| **15–20°C (Core-TC)** | 0 | 55,776 | **747,934** | 174,469 | 2,241 | 2 | **980,422** |
| **20–25°C (Upp-TC)** | 0 | 0 | 80,690 | **831,244** | 166,712 | 1,994 | **1,080,640** |
| **25–28°C (Mixed)** | 0 | 0 | 42 | 211,206 | **1,441,628** | 86,032 | **1,738,908** |
| **≥28°C (Warm Pool)** | 0 | 0 | 0 | 7,146 | 684,866 | **1,351,578** | **2,043,590** |
| **Total Predicted** | **809,611** | **1,289,584** | **959,136** | **1,224,350** | **2,295,447** | **1,439,606** | **8,017,734** |

#### Per-Class Metrics (Canonical B5 Pointwise MLP)
- **Deep Water (<10°C)**: Support: 797,606 | Precision: 87.63% | Recall: 88.95% | F1: 0.8828
- **Lower Thermocline (10–15°C)**: Support: 1,376,568 | Precision: 88.84% | Recall: 83.23% | F1: 0.8594
- **Core Thermocline (15–20°C)**: Support: 980,422 | Precision: 77.98% | Recall: 76.29% | F1: 0.7712
- **Upper Thermocline (20–25°C)**: Support: 1,080,640 | Precision: 67.89% | Recall: 76.92% | F1: 0.7213
- **Subsurface Mixed Layer (25–28°C)**: Support: 1,738,908 | Precision: 62.80% | Recall: 82.90% | F1: 0.7147
- **Tropical Warm Pool (≥28°C)**: Support: 2,043,590 | Precision: 93.89% | Recall: 66.14% | F1: 0.7761

---

### 6.3 Canonical B2 Multi-Output Ridge ($79.87\%$ Accuracy, $\kappa = 0.7538$)

#### Raw Confusion Matrix ($N = 8,017,734$)

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

### 6.4 Canonical B1 Spatial Climatology ($76.23\%$ Accuracy, $\kappa = 0.7078$)

#### Raw Confusion Matrix ($N = 8,017,734$)

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

### 6.5 Historical Exploratory Tuning Candidate (Legacy MLP)

#### Raw Confusion Matrix ($N = 8,017,734$)

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

## 7. Mathematical Verification of Metrics

1. **Total Sample Count Consistency**:
   $$\sum_{i=0}^5 \text{Support}_i = 797,606 + 1,376,568 + 980,422 + 1,080,640 + 1,738,908 + 2,043,590 = 8,017,734$$
   Row sums and column sums in every confusion matrix sum to exactly $8,017,734$.

2. **Diagonal Sum (True Positives) & Accuracy**:
   - B3 Random Forest: $\sum_{i=0}^5 \text{CM}[i, i] = 6,466,349 \implies \text{Accuracy} = \frac{6,466,349}{8,017,734} = 80.6506\% \to \mathbf{80.65\%}$
   - B5 Pointwise MLP: $\sum_{i=0}^5 \text{CM}[i, i] = 6,227,512 \implies \text{Accuracy} = \frac{6,227,512}{8,017,734} = 77.6717\% \to \mathbf{77.67\%}$
   - B2 Ridge: $\sum_{i=0}^5 \text{CM}[i, i] = 6,404,011 \implies \text{Accuracy} = \frac{6,404,011}{8,017,734} = 79.8731\% \to \mathbf{79.87\%}$
   - B1 Climatology: $\sum_{i=0}^5 \text{CM}[i, i] = 6,112,280 \implies \text{Accuracy} = \frac{6,112,280}{8,017,734} = 76.2345\% \to \mathbf{76.23\%}$
   - Legacy MLP: $\sum_{i=0}^5 \text{CM}[i, i] = 6,509,625 \implies \text{Accuracy} = \frac{6,509,625}{8,017,734} = 81.1903\% \to \mathbf{81.19\%}$

3. **Adjacency / Within $\pm 1$ Bin Calculation**:
   $$\text{Within }\pm 1 = \frac{1}{N} \sum_{|i - j| \le 1} \text{CM}[i, j] \times 100\%$$
   - B2 Ridge: $8,010,752 / 8,017,734 = \mathbf{99.91\%}$ (Beyond: $0.09\%$, $6,982$ points)
   - B5 Pointwise MLP: $8,006,024 / 8,017,734 = \mathbf{99.85\%}$ (Beyond: $0.15\%$, $11,710$ points)
   - B3 Random Forest: $7,994,254 / 8,017,734 = \mathbf{99.71\%}$ (Beyond: $0.29\%$, $23,480$ points)
   - Legacy MLP: $8,005,155 / 8,017,734 = \mathbf{99.84\%}$ (Beyond: $0.16\%$, $12,579$ points)
   - B1 Climatology: $7,969,092 / 8,017,734 = \mathbf{99.39\%}$ (Beyond: $0.61\%$, $48,642$ points)
   - Exact sum: $\text{Within } \pm 1 + \text{Beyond } \pm 1 = 100.00\%$ across all models.

4. **Cohen's Kappa ($\kappa$)**:
   $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
   All models exceed $\kappa > 0.70$, indicating substantial agreement beyond chance. B3 Random Forest achieves the highest certified inter-rater agreement among certified baselines at $\kappa = 0.7636$.

---

## 8. Physical Oceanographic Interpretation & Error Adjacency

1. **Local Error Adjacency**:
   More than 99.7% of evaluated predictions across all supervised ML models (B2, B3, B5) fall within the true thermal-regime bin or an immediately adjacent bin ($99.91\%$ for B2 Ridge, $99.85\%$ for B5 Pointwise MLP, $99.71\%$ for B3 Random Forest). This indicates that most classification errors are local in regime space and are concentrated near continuous temperature-regime boundaries. Non-adjacent misclassifications (beyond $\pm 1$ bin) range from a low of $0.09\%$ (B2 Ridge) to $0.29\%$ (B3 Random Forest). Non-adjacent errors exist but remain rare (<0.30% across all supervised ML models).

   *Important scientific constraint*: High $\pm 1$-bin containment does not prove vertical monotonicity across individual depth profiles, nor does discrete regime grouping preclude localized gradient inversions. Vertical thermal structure is formally evaluated via continuous profile metrics.

2. **Comparative Model Analysis**:
   - **B3 Random Forest** achieves the highest discrete regime classification score among certified baselines with **80.65%** exact accuracy and $\kappa = 0.7636$, outperforming spatial climatology (76.23%, $+4.42\%$). (Note: No formal paired significance test has been certified for discrete classification accuracy; paired bootstrap significance is certified for continuous RMSE).
   - **B2 Ridge Regression** delivers **79.87%** exact accuracy with the highest near-neighbor containment ($99.91\%$ within $\pm 1$ bin; only $0.09\%$ beyond), demonstrating that regularized linear column projections yield high local regime containment along the vertical thermal gradient.
   - **B5 Pointwise MLP** achieves $77.67\%$ exact accuracy and $99.85\%$ within $\pm 1$ bin. Without spatial patches (as in B6/B8) or temporal sequence memory (as in B7/B8), pointwise neural optimization on 1D columns exhibits lower discrete and continuous skill than tree ensembles.

---

## 9. Scientific Claims and Reference Audit

| Audit Area | Previous / Ambiguous Wording | Corrected Authoritative Designation | Scientific Rationale |
| :--- | :--- | :--- | :--- |
| **Target Field Reference** | "Observational ground truth", "measured truth" | **"GLORYS numerical ocean reanalysis reference"** | GLORYS12V1 is a numerical simulation integrating satellite/in-situ observations via data assimilation, not direct observational ground truth. |
| **ARGO Observations** | "Independent validation of ML predictions" | **"ARGO–GLORYS Reference Consistency Assessment"** | Operational ARGO float profiles are assimilated into GLORYS; this comparison assesses reanalysis reference consistency and does not constitute independent validation of the ML model. |
| **Sea Surface Salinity** | Generic / SMAP labels | **"Copernicus Multi-Observation SSS"** | Accurately identifies the operational data stream (CMEMS MULTIOBS L4 SSS). |
| **Bathymetry & Masking** | Independent GEBCO claims | **"GLORYS/ORCA12 Model Bathymetry & 4-Way Mask"** | Production masking strictly adheres to the numerical model's bathymetry and valid target envelope, preserving genuine target NaNs without artificial zero-filling. |
| **Regime Categorization** | "Classical/international 6-regime standard" | **"Six temperature-based thermal regimes defined for diagnostic classification"** | Clarifies that regimes are physically motivated diagnostic discretization intervals, not an international treaty or WMO standard. |
| **Warm Pool Boundary** | "Equatorial warm pool" | **"Tropical Warm Pool ($\ge 28^\circ\text{C}$)"** | Acknowledges that warm pool temperatures occur across both equatorial and northern basin sectors in the North Indian Ocean. |

---

## 10. Cross-Artifact Synchronization Verification

All repository artifacts are synchronized with the canonical benchmark identities:

1. `results/B3.json`:
   - Model Name: `Random Forest`
   - Parameters / Complexity: `13,289,966` decision nodes across 750 trees
   - Backend: `sklearn.ensemble.RandomForestRegressor`
   - Test RMSE: `1.0452` °C (Weighted: `1.0349` °C)
   - SHA-256: `4ae89a43154d00e4dbb4362c9490a6a7074a9d839ac00c9e33155501153cc548`

2. `results/B5.json`:
   - Model Name: `Pointwise MLP`
   - Trainable Parameters: `26,767`
   - Backend: `torch` (`[128, 128, 64]`)
   - Test RMSE: `1.5524` °C (Weighted: `1.5450` °C)
   - SHA-256: `f924e5f7b0260d8da91eb1b58998a54e1b7278a0124fa43bf14551c22ac07c56`

3. `results/master_benchmark_summary.json` & `reports/master_benchmark_summary.md`:
   - B3 listed as `Multi-Depth Random Forest` (Test RMSE $1.0452^\circ\text{C}$).
   - B5 listed as `Pointwise MLP` (Test RMSE $1.5524^\circ\text{C}$).
   - B8 confirmed as best-performing architecture ($0.9800^\circ\text{C}$).
   - Table column header updated to `Parameters / Complexity*` with explanatory note.

4. `results/confusion_matrix.json` & `reports/confusion_matrix_report.md`:
   - Contains all 5 models (B1, B2, B3, B5, and Legacy MLP reference).
   - $N = 8,017,734$ valid physical test targets.
   - Fully harmonized per-class and summary classification metrics with verified exact counts.

5. `confusion_matrix.html`:
   - Updated interactive dashboard with buttons for B3 (RF), B5 (MLP), B2 (Ridge), B1 (Clim), and Legacy MLP.
   - Dynamic metric recalculation for counts, percentages, and KPIs.
   - Scientifically hardened error adjacency explanation.

---

## 11. Script Architecture & Collision Prevention

To prevent future collisions between B3 and B5:
- `scripts/train_b3.py`: Dedicated script for B3 Multi-Depth Random Forest (`B3_RandomForest`), writing strictly to `results/B3.json`.
- `scripts/train_b5.py`: Dedicated script for B5 Pointwise MLP (`B5_PointwiseMLP`), writing strictly to `results/B5.json`.
- `scripts/train_b3_to_b8.py`: Multi-model benchmark suite respecting the exact B3 (Random Forest) and B5 (Pointwise MLP) separation.
- `scripts/evaluate_confusion_matrix.py`: Unified evaluation script testing all canonical models and recording provenance notes, explicitly noting that B3 is evaluated via deterministic refitting under the canonical training protocol with the locked seed (`random_state=42`) and 100,000-row training subsample.

---

## 12. Verification & Test Suite Execution

The repository test suite was executed to verify that all preprocessing, masking, model instantiation, metrics engine, and data integrity guarantees remain unbroken:
- **Test Command**: `python -m pytest tests/`
- **Result**: **97/97 tests passed** with zero failures, zero errors.

---

## 13. Git Status & Certification

- **Branch**: `main`
- **Verification Summary**:
  - Restored canonical B3 Random Forest from historical commit `68dc37b48279bc41a1daa639da3a82bd2fb8d9bf`.
  - Harmonized `master_benchmark_summary.json` and `master_benchmark_summary.md`.
  - Created `scripts/train_b5.py` and refactored `scripts/train_b3.py`.
  - Computed and saved `results/confusion_matrix.json` and `reports/confusion_matrix_report.md`.
  - Updated `confusion_matrix.html` interactive artifact.
  - Hardened all causal, proof, significance, and ARGO independence language.
