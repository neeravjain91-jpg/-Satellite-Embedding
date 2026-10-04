# ML IMPLEMENTATION SCIENTIFIC AUDIT

**Auditor:** Independent Scientific Reviewer  
**Date:** 2025-01  
**Scope:** Forensic code-level review of all implemented ML infrastructure  
**Against:** SCIENTIFIC_ML_EXPERIMENT_PROTOCOL.md  
**Reference:** 63/63 tests passing as of last report  

---

## AUDIT METHODOLOGY

Each component was read in full and checked against:
- The SCIENTIFIC_ML_EXPERIMENT_PROTOCOL.md specification  
- Physical oceanographic plausibility  
- Standard ML leakage detection patterns  
- Cross-component consistency  

---

## 1. BASELINE FAIRNESS AUDIT

### Finding Summary

| Model | Status | Fairness | Verdict |
|-------|--------|----------|---------|
| B0 Day-0 Persistence | Fully implemented | Fair | **PASS** |
| B0b Day-252 Persistence | Fully implemented | Fair | **PASS** |
| B1 Climatology | Fully implemented | Fair | **PASS** (note stale docs) |
| B2 Ridge | Fully implemented | Fair | **PASS** |
| B3 Random Forest | Fully implemented | Fair, with caveats | **MAJOR** |
| B4 Gradient Boosting | Fully implemented | Fair, with caveats | **MAJOR** |
| B5 Pointwise MLP | Fully implemented | Fair, with caveats | **MAJOR** |
| B6 Spatial CNN | **STUB** | N/A — not trained | **BLOCKER** |
| B7 Temporal Model | **STUB** | N/A — not trained | **BLOCKER** |
| B8 Embedding Model | **STUB** | N/A — not trained | **BLOCKER** |

### Per-Baseline Detail

**B0 / B0b Persistence — PASS**
- Mathematically correct: stores day 0 / day 252 GLORYS thetao per coordinate, propagates unchanged.
- Fallback global mean profile for coordinates not observed on source day — acceptable.
- No targets beyond source day accessed.
- No test leakage.

**B1 Climatology — PASS with MAJOR docstring issue**
- Computed strictly on training partition using 4-way mask.
- Global depth mean fallback for unobserved locations — acceptable.
- **MAJOR:** `01_climatology.py` docstrings reference the OLD split ("Days 0-255" = 256 days, "Days 310-365" = 56 days). The actual code calls `load_tabular_dataset()` which enforces the new 7-day purge split. Execution is correct; documentation is misleading and must be updated.

**B2 Ridge — PASS**
- Per-depth fitting correctly masks invalid bathymetry.
- Validation-driven alpha selection.
- No test access during tuning.
- **MINOR:** `baselines.py` B2_Ridge defaults list alpha candidates missing 1000.0; `02_ridge.py` includes it. Inconsistent.

**B3 Random Forest — MAJOR (3 issues)**
1. **NaN-to-zero target contamination:** `Y_train_sub = np.nan_to_num(Y_train_sub, nan=0.0)` fills ALL target NaNs (invalid depths below seafloor) with 0.0 °C. The multi-output RF trains to predict 0.0 at those depths. While evaluation masks exclude them, the model splits are biased by having highly predictable-but-meaningless 0.0 targets.
2. **Hyperparameter grid:** Only 3 candidates. Protocol recommends ≥ 30.
3. **Subsampling:** 100k from ~4.75M (2.1%).

**B4 Gradient Boosting — MAJOR (2 issues)**
1. **Backend inconsistency:** `baselines.py` B4 uses `HistGradientBoostingRegressor`; `04_gradient_boosting.py` uses LightGBM/XGBoost. Different algorithms.
2. **Hyperparameter grid:** Only 3 candidates.
- **Positive:** Per-depth training is correct — no NaN-to-zero contamination like RF and MLP.

**B5 Pointwise MLP — MAJOR (2 issues)**
1. **NaN-to-zero in training Y tensor:** `np.nan_to_num(Y_train_sub, nan=0.0)` fills invalid depths with 0.0 °C. The masked MSE loss excludes these correctly, but the Y_t tensor contains 0.0 at invalid positions, creating latent risk of unnoticed contamination.
2. **Early stopping:** patience=4, max_epochs=20. Very aggressive.
3. **Subsampling:** 500k from ~4.75M (10.5%).

**B6 Spatial CNN — BLOCKER**
- `fit()` builds the network skeleton only. No training loop.
- `predict()` returns constant 15.0 °C.
- Cannot be evaluated.

**B7 Temporal Model — BLOCKER**
- `fit()` builds network skeleton. No training loop.
- `predict()` returns constant 15.0 °C.
- Architecture is an MLP with flattened input — **not a temporal model** (no LSTM, GRU, TCN).

**B8 Embedding Model — BLOCKER**
- `fit()` builds network skeleton. No training loop.
- `predict()` returns constant 15.0 °C.
- Architecture is a 3-layer MLP (7→128→128→15) with LayerNorm — **not a spatiotemporal embedding**.

---

## 2. B0 / B0b PERSISTENCE AUDIT — PASS

| Check | Status | Evidence |
|-------|--------|----------|
| Uses day 0 state | PASS | `time_idx == 0` mask |
| Uses day 252 state | PASS | `time_idx == max_t` (max_t=252) |
| Never accesses test targets | PASS | Only reads from train_data |
| Fallback for unobserved locations | PASS | `fallback_profile = np.nanmean(Y_day0)` |
| Masks handled correctly | PASS | Full 15-d profile; metrics use mask |

---

## 3. CLIMATOLOGY AUDIT — PASS with MAJOR doc issue

| Check | Status |
|-------|--------|
| Computed from training only | PASS |
| Spatial coordinates preserved | PASS — per-(lat, lon) dict |
| Depth-wise (15 values) | PASS |
| Low-sample cell handling | PASS — falls back to global depth mean |
| R² = 1 − SS_res / SS_clim correct | PASS — climatology used as reference when provided |
| Stale docstrings (old 256-day split) | **MAJOR** — must update |

---

## 4. B2–B5 TABULAR FAIRNESS

| Requirement | B2 Ridge | B3 RF | B4 GBDT | B5 MLP |
|-------------|----------|-------|---------|--------|
| Same 7 predictors | PASS | PASS | PASS | PASS |
| Same normalization | PASS | PASS | PASS | PASS |
| Same target definition | PASS | **MAJOR**¹ | **PASS**² | **MAJOR**¹ |
| Same mask logic | PASS | PASS | PASS | PASS |
| Same train/val/test | PASS | PASS | PASS | PASS |
| No future values | PASS | PASS | PASS | PASS |
| No target-derived features | PASS | PASS | PASS | PASS |

¹ **RF/MLP:** NaN-to-zero fills invalid depths with 0.0 °C. RF trains on these; MLP masks loss but stores 0.0 in Y_t.  
² **GBDT:** Per-depth training on valid mask only — correct.

---

## 5. B6 SPATIAL CNN AUDIT — BLOCKER

- No patch extraction implemented
- No training loop
- No mask handling
- Outputs constant 15.0 °C
- Architecture mockup is valid (Conv2D → AdaptiveAvgPool → MLP) but untrained

---

## 6. B7 TEMPORAL MODEL AUDIT — BLOCKER

- Architecture is an MLP (7×T_window → 128 → 64 → 15) — **not temporal**
- No recurrent or convolutional temporal processing
- No training loop
- No data pipeline for temporal sequences
- Outputs constant 15.0 °C

---

## 7. B8 EMBEDDING MODEL AUDIT — BLOCKER

| Claimed | Actual | 
|---------|--------|
| Spatiotemporal embedding | 3-layer MLP (7→128→128→15) with LayerNorm |
| Spatial context | None — pointwise |
| Temporal context | None — single timestep |
| Embedding dimension | 128 (intermediate width, not an embedding) |
| Training | None — stub |
| Parameter count | ~19K (comparable to B5: 39K across 15 depths) |

**B8 as implemented tests:** "Does a 3-layer MLP with LayerNorm beat a 4-layer MLP?"  
**B8 should test:** "Does a spatiotemporal embedding beat pointwise processing?"

**This is the most important finding.** The entire project centers on embedding-based reconstruction, but the embedding model is not implemented.

---

## 8. HYPERPARAMETER / CHECKPOINT LEAKAGE — PASS for B0–B5

All implemented models use validation-based selection exclusively:
- Ridge: α chosen by minimum validation RMSE
- RF: Best of 3 candidates by validation RMSE
- GBDT: Best of 3 candidates by validation RMSE
- MLP: Minimum validation loss epoch, early stopping

Test set is never accessed during training or hyperparameter selection.

---

## 9. MASK AUDIT — PASS

The 4-way mask is correctly constructed and enforced:
| Mask | Source | Correct? |
|------|--------|----------|
| geographic_ocean_mask | GLORYS thetao (any depth valid) | PASS |
| surface_validity_mask | ~isnan over 7 features | PASS |
| target_validity_mask | ~isnan(thetao) | PASS |
| depth_valid_mask | GLORYS model bathymetry | PASS |
| **unified training mask** | geo AND depth AND surf AND targ | PASS |

**Note:** Depth mask derived from GLORYS, not independent GEBCO/ETOPO. This is an ACCEPTED limitation for the GLORYS-emulation paradigm.

---

## 10. METRIC AUDIT — PASS

| Metric | Implementation | Correct? |
|--------|---------------|----------|
| RMSE | sqrt(mean(diff²)) over mask | PASS |
| MAE | mean(|diff|) over mask | PASS |
| Bias | mean(diff) over mask | PASS |
| Pearson r | corrcoef with std>1e-6 guard | PASS |
| R² vs climatology | 1 − SS_res / SS_clim | PASS |
| Unweighted depth mean | mean(RMSE_d) | PASS |
| Weighted depth mean | Σ(N_d × RMSE_d) / Σ(N_d) | PASS |
| Cosine(lat) area weighting | cos(lat×π/180) in regional per-depth | PASS |
| Block bootstrap | 7-day blocks, 95% percentile CI | PASS |
| Seasonal breakdown | Late Fall (313–342), Early Winter (343–365) | PASS |

No mathematical errors found. The metric engine is comprehensive and correct.

---

## 11. ARGO EVALUATION AUDIT — PASS

| Check | Status | Detail |
|-------|--------|--------|
| Test-period only | PASS | 2020-11-09 to 2020-12-31 |
| QC flag == 1 | PASS | Good data only |
| Spike test | PASS | Gradient < 5°C / 10 dbar |
| Diurnal test | PASS | T < 35°C in top 10 m |
| Min 5 levels | PASS | |
| No extrapolation | PASS | bounds_error=False + manual clip |
| Shallow exclusion | PASS | Water depth >= 1000 m (documented) |
| Paired GLORYS vs ML | PASS | Both errors computed |
| Bootstrap CI | PASS | Profile-level resampling, 95% CI |
| Terminology | **EXEMPLARY** | "ARGO–GLORYS reference consistency assessment" — no false independence claim |

---

## 12. PRE-TRAINING GATE AUDIT — PASS

8 conditions, all enforceable:
1. Dataset certification artifact present
2. Exact 253/6/48/6/53 split
3. Zero purge buffer leakage
4. Normalization scaler with SHA-256
5. Mask has correct shape (N, 15)
6. No NaN targets inside valid mask
7. Exactly 15 canonical depths
8. Provenance manifest present

**No bypass flags found.** The gate cannot be circumvented through `--force`, environment variables, or optional parameters.

---

## 13. EXPERIMENT REGISTRY AUDIT — PASS (MINOR gaps)

| Recorded | Status |
|----------|--------|
| Git SHA | PASS |
| Dataset version | PASS |
| Split metadata | PASS |
| Scaler SHA-256 | PASS |
| Random seed | PASS |
| Hyperparameters | PASS |
| Model config | PASS |
| Checkpoint path | PASS |
| Metrics | PASS |
| **Missing: model weight hash** | **MINOR** — add SHA-256 of .pt file |
| **Missing: hardware info** | **MINOR** |
| **Missing: wall time** | **MINOR** |

---

## 14. SCIENTIFIC VALIDITY OF BASELINE LADDER — BLOCKER

The ladder collapses because B6–B8 are stubs. The comparative analysis cannot proceed on the premise of increasing spatiotemporal complexity because the spatial and temporal rungs are missing.

B0–B5 comparisons are valid:
- B0 → B0b: memory across time gap (valid)
- B0b → B1: value of spatial climatology (valid)
- B1 → B2: value of surface features (valid)
- B2 → B3: value of nonlinearity (valid, with NaN-to-zero caveat)
- B3 → B4: ensemble method difference (valid)
- B4 → B5: neural representation (valid)

But B5 → B6 (spatial), B6 → B7 (temporal), B7 → B8 (spatiotemporal) cannot be tested.

---

## 15. FINAL VERDICT

### Finding Table

| Component | Finding | Severity | Required Fix |
|-----------|---------|----------|-------------|
| B0/B0b | Correct | **PASS** | None |
| B1 Climatology | Correct execution; stale docstrings | **MAJOR** | Update docstrings to 253/6/48/6/53 split |
| B2 Ridge | Correct; minor alpha inconsistency | **MAJOR** | Sync alpha lists between baselines.py and 02_ridge.py |
| B3 Random Forest | NaN-to-zero contamination; tiny hyperparameter search | **MAJOR** | Use per-depth training or mask-respecting fill |
| B4 GBDT | Backend inconsistency (HGBR vs LightGBM) | **MAJOR** | Pin one backend; update baselines.py |
| B5 Pointwise MLP | NaN-to-zero in Y_t tensor; aggressive early stopping | **MAJOR** | Use NaN in Y_t, not 0.0; increase patience |
| B6 Spatial CNN | **Stub — no training** | **BLOCKER** | Implement training loop with spatial patch extraction |
| B7 Temporal Model | **Stub — MLP, not temporal; no training** | **BLOCKER** | Implement LSTM/TCN with temporal window |
| B8 Embedding Model | **Stub — MLP, not embedding; no training** | **BLOCKER** | Implement true spatiotemporal encoder |
| Hyperparameter selection | Correct — validation-based | **PASS** | None |
| Checkpoint selection | Correct — validation-based | **PASS** | None |
| 4-way mask | Correct | **PASS** | None |
| Metric engine | Comprehensive and correct | **PASS** | None |
| ARGO evaluation | Scientific and honest | **PASS** | None |
| Pre-training gate | Robust, no bypass | **PASS** | None |
| Experiment registry | All essential fields | **PASS** | Add weight hash (MINOR) |

### A. Baseline Fairness Verdict
**B0–B5: FAIR.** All use identical inputs, normalization, masks, and split. The NaN-to-zero issue in B3 and B5 should be fixed but does not fundamentally break comparability.

**B6–B8: NOT FAIR — not evaluable.**

### B. B8 Architecture Verdict
**BLOCKER.** B8 is a pointwise MLP with LayerNorm — not a spatiotemporal embedding. It tests "does LayerNorm help?" instead of "does spatiotemporal embedding help?".

### C. Leakage Verdict
**NO LEAKAGE DETECTED.** Temporal split, normalization isolation, purge buffers, mask enforcement, and test isolation are all correctly implemented and verified by tests. B6–B8 are stubs so cannot leak.

### D. Metric Correctness Verdict
**PASS.** All metrics mathematically correct. Mask handling verified. Area weighting verified. Bootstrap verified.

### E. ARGO Evaluation Verdict
**PASS.** Scientifically honest. Does not claim independence. Correctly labeled as "consistency assessment."

### F. Pre-Training Gate Verdict
**PASS.** 8 conditions enforced. No bypasses found.

### G. Required Fixes Before First Official Training

**BLOCKER conditions for B0–B5 training:**
1. ✅ Pre-training gate passes (tests confirm)
2. ✅ Temporal split correct (tests confirm)
3. ✅ Mask integrity verified (tests confirm)
4. ✅ Normalization isolation verified (tests confirm)
5. ✅ ARGO evaluation protocol implemented
6. ⬜ B3/B5 NaN-to-zero contamination fixed (should not block initial runs for metric infrastructure validation, but must be fixed before reporting results)
7. ⬜ B1/B2 docstrings updated
8. ⬜ B4 backend pinned

**BLOCKER conditions for B6–B8:**
- All three must be implemented with real training loops
- B8 must be a true spatiotemporal embedding, not an MLP
- Before B8 implementation, the spatial (B6) and temporal (B7) baselines must be demonstrated individually

### H. Issues That May Wait Until After B0–B5

1. B6–B8 implementation (separate phase)
2. Model weight hash in experiment registry
3. Hardware/environment logging in registry
4. Expanded hyperparameter search (increases runtime)

### I. Final ML Readiness Verdict

| Criterion | Status |
|-----------|--------|
| B0–B5 infrastructure | **READY** — with NaN-to-zero fix recommended |
| B6–B8 infrastructure | **NOT READY** — stubs implemented; architecture decisions deferred |
| Pre-training gate | **PASS** |
| Metric engine | **PASS** |
| ARGO evaluation | **PASS** |
| Leakage integrity | **PASS** |
| Dataset certification | Required before execution (gate enforces this) |
| Reproducibility | **PASS** (registry + SHA-256) |

**Verdict:**
B0–B5 training is **conditionally ready** after fixing:
1. NaN-to-zero contamination in B3/B5 training data
2. Backend pinning for B4
3. Docstring updates for split numbers

The protocol specifies that B6–B8 comparison is the core scientific contribution. **Until B8 is implemented as a true spatiotemporal embedding, the project cannot test its primary hypothesis.** The current infrastructure correctly supports Phase 1 baselines (B0–B5) but does not yet support the research goal (B8).

---

*End of ML Implementation Scientific Audit. No code was modified. All findings derived from reading the repository at commit state with 63/63 passed tests.*