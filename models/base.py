"""
models/base.py
Shared evaluation metrics and reporting utilities for Phase 1 pointwise models:
- Computes masked RMSE, MAE, Bias, and R^2 over valid depth points [N, 15]
- Depth-by-depth performance breakdown across all 15 target depths
- Strictly respects target masks (shallow vs deep bathymetry)
"""

import os
import json
import numpy as np
import pandas as pd

from preprocessing.canonical_grid import CANONICAL_DEPTHS

def compute_masked_metrics(y_true, y_pred, mask, depth_levels=CANONICAL_DEPTHS, model_name="Model", split_name="val"):
    """
    Computes overall and depth-wise evaluation metrics respecting target masks.
    y_true: (N, 15) float array (ground truth)
    y_pred: (N, 15) float array (predictions)
    mask:   (N, 15) boolean array (True where valid ocean depth)
    """
    # 1. Overall Masked Metrics across all (sample, depth) pairs
    valid_true = y_true[mask]
    valid_pred = y_pred[mask]
    
    if len(valid_true) == 0:
        raise ValueError("No valid masked points found for evaluation!")
        
    diff = valid_pred - valid_true
    overall_rmse = float(np.sqrt(np.mean(diff ** 2)))
    overall_mae = float(np.mean(np.abs(diff)))
    overall_bias = float(np.mean(diff))
    
    # R^2 score
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((valid_true - np.mean(valid_true)) ** 2)
    overall_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Pearson correlation
    corr = float(np.corrcoef(valid_pred, valid_true)[0, 1]) if len(valid_true) > 1 else 0.0

    # 2. Depth-Wise Breakdown
    depth_records = []
    for d_idx, d_val in enumerate(depth_levels):
        d_mask = mask[:, d_idx]
        n_valid = int(d_mask.sum())
        
        if n_valid == 0:
            depth_records.append({
                "depth_m": float(d_val),
                "n_valid": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "bias": np.nan,
                "corr": np.nan
            })
            continue
            
        t_d = y_true[d_mask, d_idx]
        p_d = y_pred[d_mask, d_idx]
        diff_d = p_d - t_d
        
        rmse_d = float(np.sqrt(np.mean(diff_d ** 2)))
        mae_d = float(np.mean(np.abs(diff_d)))
        bias_d = float(np.mean(diff_d))
        corr_d = float(np.corrcoef(p_d, t_d)[0, 1]) if (len(t_d) > 1 and np.std(t_d) > 1e-6 and np.std(p_d) > 1e-6) else 0.0
        
        depth_records.append({
            "depth_m": float(d_val),
            "n_valid": n_valid,
            "rmse": round(rmse_d, 3),
            "mae": round(mae_d, 3),
            "bias": round(bias_d, 3),
            "corr": round(corr_d, 3)
        })

    df_depths = pd.DataFrame(depth_records)
    
    summary = {
        "model": model_name,
        "split": split_name,
        "total_evaluated_points": int(mask.sum()),
        "overall_rmse": round(overall_rmse, 3),
        "overall_mae": round(overall_mae, 3),
        "overall_bias": round(overall_bias, 3),
        "overall_r2": round(overall_r2, 4),
        "overall_corr": round(corr, 4),
        "depth_breakdown": depth_records
    }
    
    return summary, df_depths

def print_evaluation_summary(summary, df_depths):
    print("\n" + "=" * 65)
    print(f"  EVALUATION SUMMARY: {summary['model']} [{summary['split'].upper()}]")
    print("=" * 65)
    print(f"  Overall RMSE:   {summary['overall_rmse']:.3f} °C")
    print(f"  Overall MAE:    {summary['overall_mae']:.3f} °C")
    print(f"  Overall Bias:   {summary['overall_bias']:.3f} °C")
    print(f"  Overall R^2:    {summary['overall_r2']:.4f}")
    print(f"  Overall Corr r: {summary['overall_corr']:.4f}")
    print(f"  Total Points:   {summary['total_evaluated_points']:,}")
    print("\nDepth-Wise Breakdown:")
    print(df_depths.to_string(index=False))

def save_model_evaluation(summary, df_depths, out_dir="reports/phase1"):
    os.makedirs(out_dir, exist_ok=True)
    m_name = summary["model"].lower().replace(" ", "_")
    s_name = summary["split"].lower()
    
    # Save JSON summary
    json_path = os.path.join(out_dir, f"{m_name}_{s_name}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    # Save CSV depth breakdown
    csv_path = os.path.join(out_dir, f"{m_name}_{s_name}_depth_metrics.csv")
    df_depths.to_csv(csv_path, index=False)
    print(f"Saved evaluation metrics to {json_path} and {csv_path}")
