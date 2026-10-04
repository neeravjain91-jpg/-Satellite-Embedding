"""
scripts/train_b3_to_b8.py
Comprehensive Execution Script for Baselines B3 through B8 on the Certified 2020 Production Dataset:
- B3: Multi-Depth Random Forest Regressor
- B4: Multi-Depth Gradient Boosted Decision Trees (LightGBM 4.7.0)
- B5: Pointwise Multi-Layer Perceptron (PyTorch)
- B6: Spatial CNN (Conv2D local patch encoder)
- B7: Temporal GRU (causal sequential encoder)
- B8: Spatiotemporal Embedding Model (joint Conv2D + GRU latent bottleneck)

Protocol Guarantees:
- Chronological Splits:
    Train: Days 0..252 (253 days)
    Purge 1: Days 253..258 (6 days, discarded)
    Val: Days 259..306 (48 days)
    Purge 2: Days 307..312 (6 days, discarded)
    Test: Days 313..365 (53 days)
- Zero Leakage: All normalization strictly from train partition.
- Target Integrity: Invalid bathymetric depths strictly preserved as NaN; never 0 °C.
- Metrics: Comprehensive 15 depths, area-weighted regions, seasons, 7-day block bootstrap CIs (N=1000).
- Paired Block-Bootstrap Delta RMSE evaluated against B1 Climatology.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import numpy as np
import pandas as pd
import xarray as xr

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_DEPTHS, CANONICAL_FEATURES
from preprocessing.tabular_dataset import load_tabular_dataset
from models.baselines import (
    B1_Climatology,
    B3_RandomForest,
    B4_GradientBoosting,
    B5_PointwiseMLP,
    B6_SpatialCNN,
    B7_TemporalModel,
    B8_EmbeddingModel,
)
from models.metrics_engine import (
    compute_comprehensive_metrics,
    compute_block_bootstrap_ci
)


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_paired_block_bootstrap(y_true, y_pred_m, y_pred_b1, mask, time_indices,
                                   block_length_days=7, n_bootstraps=1000,
                                   depth_levels=CANONICAL_DEPTHS, random_seed=42):
    """
    Computes 95% paired block-bootstrap confidence intervals for Delta RMSE = RMSE_Model - RMSE_B1.
    """
    np.random.seed(random_seed)
    unique_times = np.sort(np.unique(time_indices))
    n_days = len(unique_times)
    n_depths = len(depth_levels)

    if n_days < block_length_days:
        blocks = [[t] for t in unique_times]
    else:
        blocks = []
        for i in range(0, n_days, block_length_days):
            blocks.append(list(unique_times[i:i + block_length_days]))
    n_blocks = len(blocks)

    day_sse_m = np.zeros((n_days, n_depths), dtype=np.float64)
    day_sse_b1 = np.zeros((n_days, n_depths), dtype=np.float64)
    day_cnt = np.zeros((n_days, n_depths), dtype=np.int64)

    for t_idx, t_val in enumerate(unique_times):
        day_m = (time_indices == t_val)
        for d in range(n_depths):
            comb = mask[:, d] & day_m
            if np.any(comb):
                diff_m = y_pred_m[comb, d] - y_true[comb, d]
                diff_b1 = y_pred_b1[comb, d] - y_true[comb, d]
                day_sse_m[t_idx, d] = np.sum(diff_m ** 2)
                day_sse_b1[t_idx, d] = np.sum(diff_b1 ** 2)
                day_cnt[t_idx, d] = int(np.sum(comb))

    boot_delta_unw = []
    boot_delta_w = []
    boot_delta_per_depth = [[] for _ in range(n_depths)]

    for _ in range(n_bootstraps):
        sampled_block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        sampled_day_vals = []
        for b_idx in sampled_block_indices:
            sampled_day_vals.extend(blocks[b_idx])

        day_idxs = [np.where(unique_times == dv)[0][0] for dv in sampled_day_vals]

        boot_sse_m = np.sum(day_sse_m[day_idxs, :], axis=0)
        boot_sse_b1 = np.sum(day_sse_b1[day_idxs, :], axis=0)
        boot_cnt = np.sum(day_cnt[day_idxs, :], axis=0)

        depth_rmses_m = []
        depth_rmses_b1 = []
        for d in range(n_depths):
            if boot_cnt[d] > 0:
                r_m = np.sqrt(boot_sse_m[d] / boot_cnt[d])
                r_b1 = np.sqrt(boot_sse_b1[d] / boot_cnt[d])
                depth_rmses_m.append(r_m)
                depth_rmses_b1.append(r_b1)
                boot_delta_per_depth[d].append(r_m - r_b1)
            else:
                depth_rmses_m.append(np.nan)
                depth_rmses_b1.append(np.nan)

        d_unw = np.nanmean(depth_rmses_m) - np.nanmean(depth_rmses_b1)
        tot_cnt = np.sum(boot_cnt)
        if tot_cnt > 0:
            d_w = np.sqrt(np.sum(boot_sse_m) / tot_cnt) - np.sqrt(np.sum(boot_sse_b1) / tot_cnt)
        else:
            d_w = np.nan

        boot_delta_unw.append(d_unw)
        boot_delta_w.append(d_w)

    def ci_stats(arr):
        v = np.array(arr)
        v = v[~np.isnan(v)]
        return {
            "mean": round(float(np.mean(v)), 4),
            "ci_95_low": round(float(np.percentile(v, 2.5)), 4),
            "ci_95_high": round(float(np.percentile(v, 97.5)), 4),
            "std": round(float(np.std(v)), 4)
        }

    overall_ci = {
        "unweighted_delta_rmse": ci_stats(boot_delta_unw),
        "sample_weighted_delta_rmse": ci_stats(boot_delta_w)
    }

    depth_cis = []
    for d in range(n_depths):
        depth_cis.append({
            "depth_m": float(depth_levels[d]),
            **ci_stats(boot_delta_per_depth[d])
        })

    return overall_ci, depth_cis


def evaluate_and_record_model(model, model_id, model_name, dataset, y_clim_val, y_clim_test, b1_test_rmse):
    print(f"\n{'='*70}\nEVALUATING {model_id}: {model_name}\n{'='*70}")
    t0 = time.time()

    val_data = dataset["val"]
    test_data = dataset["test"]

    print(f"Generating predictions for {model_id} on Validation and Test splits...")
    preds_val = model.predict(val_data)
    preds_test = model.predict(test_data)

    print(f"Computing comprehensive metrics for {model_id}...")
    val_metrics, _ = compute_comprehensive_metrics(
        y_true=val_data["Y"],
        y_pred=preds_val,
        mask=val_data["mask"],
        lats=val_data["lat"],
        lons=val_data["lon"],
        time_indices=val_data["time_idx"],
        y_clim=y_clim_val
    )
    test_metrics, _ = compute_comprehensive_metrics(
        y_true=test_data["Y"],
        y_pred=preds_test,
        mask=test_data["mask"],
        lats=test_data["lat"],
        lons=test_data["lon"],
        time_indices=test_data["time_idx"],
        y_clim=y_clim_test
    )

    print(f"Computing 7-day Block-Bootstrap CIs for {model_id} (N=1000)...")
    val_ci = compute_block_bootstrap_ci(
        y_true=val_data["Y"],
        y_pred=preds_val,
        mask=val_data["mask"],
        time_indices=val_data["time_idx"],
        n_bootstraps=1000,
        return_overall=True
    )
    test_ci = compute_block_bootstrap_ci(
        y_true=test_data["Y"],
        y_pred=preds_test,
        mask=test_data["mask"],
        time_indices=test_data["time_idx"],
        n_bootstraps=1000,
        return_overall=True
    )

    print(f"Computing paired block-bootstrap Delta RMSE vs B1 for {model_id}...")
    paired_overall, paired_depths = compute_paired_block_bootstrap(
        y_true=test_data["Y"],
        y_pred_m=preds_test,
        y_pred_b1=y_clim_test,
        mask=test_data["mask"],
        time_indices=test_data["time_idx"],
        n_bootstraps=1000
    )

    param_count = model.parameter_count() if hasattr(model, "parameter_count") else 0
    delta_unw = round(test_metrics["unweighted_depth_mean"]["rmse"] - b1_test_rmse, 4)
    rel_imp = round((b1_test_rmse - test_metrics["unweighted_depth_mean"]["rmse"]) / b1_test_rmse * 100.0, 2)

    result_record = {
        "model_id": model_id,
        "model_name": model_name,
        "parameter_count": param_count,
        "metadata": model.metadata() if hasattr(model, "metadata") else {},
        "validation": {
            "overall": val_metrics["unweighted_depth_mean"],
            "weighted_overall": val_metrics["sample_weighted_depth_mean"],
            "depth_breakdown": val_metrics["depth_breakdown"],
            "regions": val_metrics.get("regions", {}),
            "bootstrap_ci_95": val_ci[1]
        },
        "test": {
            "overall": test_metrics["unweighted_depth_mean"],
            "weighted_overall": test_metrics["sample_weighted_depth_mean"],
            "depth_breakdown": test_metrics["depth_breakdown"],
            "regions": test_metrics.get("regions", {}),
            "seasons": test_metrics.get("seasons", {}),
            "bootstrap_ci_95": test_ci[1],
            "depth_cis": test_ci[0]
        },
        "paired_bootstrap_delta_vs_b1": {
            "overall": paired_overall,
            "depth_breakdown": paired_depths
        },
        "comparisons_vs_b1": {
            "delta_unweighted_rmse": delta_unw,
            "relative_improvement_pct": rel_imp,
            "statistically_significant_improvement": bool(paired_overall["unweighted_delta_rmse"]["ci_95_high"] < 0)
        }
    }

    out_json = os.path.join(repo_root, "results", f"{model_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)

    elapsed = time.time() - t0
    print(f"Saved {out_json} ({elapsed:.1f}s)")
    print(f"  Test RMSE: {test_metrics['unweighted_depth_mean']['rmse']:.4f}°C | MAE: {test_metrics['unweighted_depth_mean']['mae']:.4f}°C | R²: {test_metrics['unweighted_depth_mean']['r2']:.4f}")
    print(f"  Delta vs B1: {delta_unw:+.4f}°C (Relative: {rel_imp:+.2f}%) | 95% CI: [{paired_overall['unweighted_delta_rmse']['ci_95_low']:.4f}, {paired_overall['unweighted_delta_rmse']['ci_95_high']:.4f}]")
    return result_record


def run_full_suite():
    print("=" * 80)
    print("EXECUTING OFFICIAL ML BENCHMARK SUITE: B3 THROUGH B8")
    print("=" * 80)
    total_start = time.time()

    # Load dataset with genuine context tensors
    dataset = load_tabular_dataset(build_context=True)
    train_data = dataset["train"]
    val_data = dataset["val"]
    test_data = dataset["test"]

    # Load B1 Climatology reference predictions for R^2 and paired bootstrap
    print("\nFitting reference B1 Climatology strictly on Train split...")
    b1_model = B1_Climatology()
    b1_model.fit(train_data)
    y_clim_val = b1_model.predict(val_data)
    y_clim_test = b1_model.predict(test_data)

    # Reference B1 test RMSE
    with open(os.path.join(repo_root, "results", "B1.json"), "r") as f:
        b1_json = json.load(f)
    b1_test_rmse = b1_json["test"]["overall"]["rmse"]
    print(f"Reference B1 Test Unweighted RMSE: {b1_test_rmse:.4f}°C")

    # Representative training subsets for high efficiency
    N_train = len(train_data["X_norm"])
    np.random.seed(42)
    sub_100k = np.random.choice(N_train, size=min(100000, N_train), replace=False)
    sub_200k = np.random.choice(N_train, size=min(200000, N_train), replace=False)

    train_100k = {k: v[sub_100k] if isinstance(v, np.ndarray) and len(v) == N_train else v for k, v in train_data.items()}
    train_200k = {k: v[sub_200k] if isinstance(v, np.ndarray) and len(v) == N_train else v for k, v in train_data.items()}

    all_results = {}

    # --- MODEL B3: Random Forest ---
    print("\nFitting B3: Random Forest (15 depth-wise regressors, n_est=50, max_depth=15)...")
    b3 = B3_RandomForest(n_estimators=50, max_depth=15, sample_train_size=100000)
    b3.fit(train_100k)
    all_results["B3"] = evaluate_and_record_model(b3, "B3", "Random Forest", dataset, y_clim_val, y_clim_test, b1_test_rmse)

    # --- MODEL B4: Gradient Boosting (LightGBM) ---
    print("\nFitting B4: Gradient Boosting (LightGBM 4.7.0, num_leaves=63, lr=0.1, n_est=50)...")
    b4 = B4_GradientBoosting(n_estimators=50, learning_rate=0.1, num_leaves=63)
    b4.fit(train_100k)
    all_results["B4"] = evaluate_and_record_model(b4, "B4", "Gradient Boosting (LightGBM)", dataset, y_clim_val, y_clim_test, b1_test_rmse)

    # --- MODEL B5: Pointwise MLP ---
    print("\nFitting B5: Pointwise MLP (PyTorch, hidden=[128, 128, 64], epochs=8, batch_size=4096)...")
    b5 = B5_PointwiseMLP(hidden_dims=[128, 128, 64], lr=1e-3)
    b5.fit(train_200k, val_data=val_data, epochs=8, batch_size=4096)
    all_results["B5"] = evaluate_and_record_model(b5, "B5", "Pointwise MLP", dataset, y_clim_val, y_clim_test, b1_test_rmse)

    # --- MODEL B6: Spatial CNN ---
    print("\nFitting B6: Spatial CNN (PyTorch Conv2D, patch=3x3, hidden=64, epochs=4, batch_size=2048)...")
    b6 = B6_SpatialCNN(patch_size=3, hidden_dim=64, lr=1e-3)
    b6.fit(train_200k, val_data=val_data, epochs=4, batch_size=2048)
    all_results["B6"] = evaluate_and_record_model(b6, "B6", "Spatial CNN (3x3)", dataset, y_clim_val, y_clim_test, b1_test_rmse)

    # --- MODEL B7: Temporal GRU ---
    print("\nFitting B7: Temporal GRU (PyTorch GRU, window=5, hidden=64, epochs=4, batch_size=2048)...")
    b7 = B7_TemporalModel(window_size=5, hidden_size=64, num_layers=2, lr=1e-3)
    b7.fit(train_200k, val_data=val_data, epochs=4, batch_size=2048)
    all_results["B7"] = evaluate_and_record_model(b7, "B7", "Temporal GRU (window=5)", dataset, y_clim_val, y_clim_test, b1_test_rmse)

    # --- MODEL B8: Spatiotemporal Embedding Model ---
    print("\nFitting B8: Spatiotemporal Embedding Model (Conv2D + GRU bottleneck, embed_dim=128, epochs=4, batch_size=1024)...")
    b8 = B8_EmbeddingModel(patch_size=3, window_size=5, embed_dim=128, lr=1e-3)
    b8.fit(train_200k, val_data=val_data, epochs=4, batch_size=1024)
    all_results["B8"] = evaluate_and_record_model(b8, "B8", "Spatiotemporal Embedding Model", dataset, y_clim_val, y_clim_test, b1_test_rmse)

    total_time = time.time() - total_start
    print(f"\n{'='*80}\nALL MODELS B3–B8 TRAINED AND EVALUATED IN {total_time:.1f}s\n{'='*80}")


if __name__ == "__main__":
    run_full_suite()
