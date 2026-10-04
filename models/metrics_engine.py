"""
models/metrics_engine.py
Authoritative Metric Engine implementing Section 7 of the Scientific ML Experiment Protocol:

Metrics Evaluated:
- RMSE (Root Mean Squared Error, °C)
- MAE (Mean Absolute Error, °C)
- Bias (Mean Prediction Bias, °C)
- Pearson r (Pattern Correlation)
- R² relative to B1 Climatology: R²_d = 1 - SS_res / SS_clim (Strictly relative to local climatology, never global mean)

Breakdowns & Aggregations:
- 15 Canonical Depths (0 to 1000 m)
- Unweighted Depth Mean: (1/15) * sum(Metric_d)
- Sample-Count Weighted Depth Mean: sum(N_d * Metric_d) / sum(N_d)
- Regional Analysis with Cosine(Latitude) Area Weighting:
    * Full Domain: 5–30°N, 45–105°E
    * Arabian Sea: 5–25°N, 45–77°E
    * Bay of Bengal: 5–25°N, 77–100°E
- Seasonal Periods:
    * Late Fall: November (days 313–342)
    * Early Winter: December (days 343–365)
- Block-Bootstrap 95% Confidence Intervals (with block length = 7 days)
"""

import os
import sys
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_DEPTHS

# ---------------------------------------------------------------------------
# Core Metric Calculations
# ---------------------------------------------------------------------------
def compute_single_depth_metrics(y_true, y_pred, mask, y_clim=None, weights=None):
    """
    Computes RMSE, MAE, Bias, Pearson r, and R² (relative to climatology) for a single depth level.
    """
    if not np.any(mask):
        return {
            "n_valid": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "r2": np.nan,
            "corr": np.nan
        }
        
    t = y_true[mask]
    p = y_pred[mask]
    diff = p - t
    
    if weights is not None:
        w = weights[mask]
        w_norm = w / np.sum(w)
        rmse = float(np.sqrt(np.sum(w_norm * (diff ** 2))))
        mae = float(np.sum(w_norm * np.abs(diff)))
        bias = float(np.sum(w_norm * diff))
    else:
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        bias = float(np.mean(diff))

    # Pearson correlation
    if len(t) > 1 and np.std(t) > 1e-6 and np.std(p) > 1e-6:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = 0.0

    # R^2 relative to climatology
    ss_res = np.sum(diff ** 2) if weights is None else np.sum(weights[mask] * (diff ** 2))
    if y_clim is not None:
        c = y_clim[mask]
        ss_clim = np.sum((t - c) ** 2) if weights is None else np.sum(weights[mask] * ((t - c) ** 2))
    else:
        ss_clim = np.sum((t - np.mean(t)) ** 2)
        
    r2 = float(1.0 - (ss_res / ss_clim)) if ss_clim > 1e-8 else 0.0

    return {
        "n_valid": int(mask.sum()),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "bias": round(bias, 4),
        "r2": round(r2, 4),
        "corr": round(corr, 4)
    }


def compute_comprehensive_metrics(y_true, y_pred, mask, y_clim,
                                  lats=None, lons=None, time_indices=None,
                                  depth_levels=CANONICAL_DEPTHS,
                                  compute_regions=True,
                                  compute_seasons=True):
    """
    Computes the complete hierarchy of scientific metrics:
    - 15 depth-wise records
    - Unweighted & weighted depth averages
    - Regional cosine-weighted breakdowns (Arabian Sea, Bay of Bengal)
    - Seasonal breakdowns (Late Fall, Early Winter)
    """
    n_depths = len(depth_levels)
    depth_records = []
    
    # 1. Depth-by-depth evaluation across full domain
    for d_idx, d_val in enumerate(depth_levels):
        d_mask = mask[:, d_idx]
        t_d = y_true[:, d_idx]
        p_d = y_pred[:, d_idx]
        c_d = y_clim[:, d_idx] if y_clim is not None else None
        
        m_dict = compute_single_depth_metrics(t_d, p_d, d_mask, c_d)
        m_dict["depth_m"] = float(d_val)
        depth_records.append(m_dict)
        
    df_depths = pd.DataFrame(depth_records)
    
    # 2. Aggregations across depths
    valid_depth_rows = df_depths[df_depths["n_valid"] > 0]
    if len(valid_depth_rows) > 0:
        unweighted_rmse = float(valid_depth_rows["rmse"].mean())
        unweighted_mae = float(valid_depth_rows["mae"].mean())
        unweighted_bias = float(valid_depth_rows["bias"].mean())
        unweighted_r2 = float(valid_depth_rows["r2"].mean())
        
        total_valid = valid_depth_rows["n_valid"].sum()
        weighted_rmse = float((valid_depth_rows["n_valid"] * valid_depth_rows["rmse"]).sum() / total_valid)
        weighted_mae = float((valid_depth_rows["n_valid"] * valid_depth_rows["mae"]).sum() / total_valid)
        weighted_bias = float((valid_depth_rows["n_valid"] * valid_depth_rows["bias"]).sum() / total_valid)
        weighted_r2 = float((valid_depth_rows["n_valid"] * valid_depth_rows["r2"]).sum() / total_valid)
    else:
        unweighted_rmse = unweighted_mae = unweighted_bias = unweighted_r2 = np.nan
        weighted_rmse = weighted_mae = weighted_bias = weighted_r2 = np.nan

    result = {
        "total_evaluated_points": int(mask.sum()),
        "unweighted_depth_mean": {
            "rmse": round(unweighted_rmse, 4),
            "mae": round(unweighted_mae, 4),
            "bias": round(unweighted_bias, 4),
            "r2": round(unweighted_r2, 4)
        },
        "sample_weighted_depth_mean": {
            "rmse": round(weighted_rmse, 4),
            "mae": round(weighted_mae, 4),
            "bias": round(weighted_bias, 4),
            "r2": round(weighted_r2, 4)
        },
        "depth_breakdown": depth_records
    }

    # 3. Regional Cosine-Weighted Breakdowns
    if compute_regions and lats is not None and lons is not None:
        result["regions"] = {}
        regions = {
            "full_domain": (lats >= 5.0) & (lats <= 30.0) & (lons >= 45.0) & (lons <= 105.0),
            "arabian_sea": (lats >= 5.0) & (lats <= 25.0) & (lons >= 45.0) & (lons <= 77.0),
            "bay_of_bengal": (lats >= 5.0) & (lats <= 25.0) & (lons >= 77.0) & (lons <= 100.0)
        }
        
        cos_lat_weights = np.cos(np.radians(lats))
        for reg_name, reg_mask in regions.items():
            reg_records = []
            for d_idx, d_val in enumerate(depth_levels):
                combined_mask = mask[:, d_idx] & reg_mask
                t_d = y_true[:, d_idx]
                p_d = y_pred[:, d_idx]
                c_d = y_clim[:, d_idx] if y_clim is not None else None
                
                m_d = compute_single_depth_metrics(t_d, p_d, combined_mask, c_d, weights=cos_lat_weights)
                m_d["depth_m"] = float(d_val)
                reg_records.append(m_d)
                
            df_reg = pd.DataFrame(reg_records)
            v_reg = df_reg[df_reg["n_valid"] > 0]
            result["regions"][reg_name] = {
                "unweighted_rmse": round(float(v_reg["rmse"].mean()), 4) if len(v_reg) > 0 else np.nan,
                "weighted_rmse": round(float((v_reg["n_valid"] * v_reg["rmse"]).sum() / v_reg["n_valid"].sum()), 4) if len(v_reg) > 0 else np.nan,
                "depth_breakdown": reg_records
            }

    # 4. Seasonal Breakdowns (Nov vs Dec on Test Partition)
    if compute_seasons and time_indices is not None:
        result["seasons"] = {}
        # Test split: Days 313–365
        # Late Fall: Days 313–342 (November)
        # Early Winter: Days 343–365 (December)
        seasons = {
            "late_fall_nov": (time_indices >= 313) & (time_indices <= 342),
            "early_winter_dec": (time_indices >= 343) & (time_indices <= 365)
        }
        for s_name, s_mask in seasons.items():
            if not np.any(s_mask):
                continue
            s_records = []
            for d_idx, d_val in enumerate(depth_levels):
                comb_mask = mask[:, d_idx] & s_mask
                t_d = y_true[:, d_idx]
                p_d = y_pred[:, d_idx]
                c_d = y_clim[:, d_idx] if y_clim is not None else None
                m_d = compute_single_depth_metrics(t_d, p_d, comb_mask, c_d)
                m_d["depth_m"] = float(d_val)
                s_records.append(m_d)
            df_s = pd.DataFrame(s_records)
            v_s = df_s[df_s["n_valid"] > 0]
            result["seasons"][s_name] = {
                "unweighted_rmse": round(float(v_s["rmse"].mean()), 4) if len(v_s) > 0 else np.nan,
                "weighted_rmse": round(float((v_s["n_valid"] * v_s["rmse"]).sum() / v_s["n_valid"].sum()), 4) if len(v_s) > 0 else np.nan,
                "depth_breakdown": s_records
            }

    return result, df_depths


# ---------------------------------------------------------------------------
# Block-Bootstrap Uncertainty Estimation (95% CI)
# ---------------------------------------------------------------------------
def compute_block_bootstrap_ci(y_true, y_pred, mask, time_indices,
                               block_length_days=7, n_bootstraps=200,
                               depth_levels=CANONICAL_DEPTHS, random_seed=42,
                               return_overall=False):
    """
    Computes 95% block-bootstrap confidence intervals over temporal blocks:
    Accounting for temporal ocean memory decorrelation timescales.
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

    # Pre-aggregate statistics per day and depth for vectorization
    day_sse = np.zeros((n_days, n_depths), dtype=np.float64)
    day_cnt = np.zeros((n_days, n_depths), dtype=np.int64)
    day_abs = np.zeros((n_days, n_depths), dtype=np.float64)
    day_sum = np.zeros((n_days, n_depths), dtype=np.float64)

    for t_idx, t_val in enumerate(unique_times):
        day_m = (time_indices == t_val)
        for d in range(n_depths):
            comb = mask[:, d] & day_m
            if np.any(comb):
                diff = y_pred[comb, d] - y_true[comb, d]
                day_sse[t_idx, d] = np.sum(diff ** 2)
                day_abs[t_idx, d] = np.sum(np.abs(diff))
                day_sum[t_idx, d] = np.sum(diff)
                day_cnt[t_idx, d] = int(np.sum(comb))

    boot_rmse_per_depth = [[] for _ in range(n_depths)]
    boot_overall_unweighted = []
    boot_overall_weighted = []
    boot_overall_mae = []
    boot_overall_bias = []
    
    for _ in range(n_bootstraps):
        sampled_block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        sampled_day_vals = []
        for b_idx in sampled_block_indices:
            sampled_day_vals.extend(blocks[b_idx])
        
        # Map day vals to t_idx
        sampled_t_indices = np.searchsorted(unique_times, sampled_day_vals)
        
        tot_sse = np.sum(day_sse[sampled_t_indices], axis=0) # (15,)
        tot_cnt = np.sum(day_cnt[sampled_t_indices], axis=0) # (15,)
        tot_abs = np.sum(day_abs[sampled_t_indices], axis=0) # (15,)
        tot_sum = np.sum(day_sum[sampled_t_indices], axis=0) # (15,)

        d_rmse = np.zeros(n_depths, dtype=np.float64)
        for d in range(n_depths):
            if tot_cnt[d] > 0:
                rmse_d = float(np.sqrt(tot_sse[d] / tot_cnt[d]))
                boot_rmse_per_depth[d].append(rmse_d)
                d_rmse[d] = rmse_d
            else:
                boot_rmse_per_depth[d].append(np.nan)
                d_rmse[d] = np.nan

        # Overall bootstrap metrics
        valid_d = ~np.isnan(d_rmse)
        if np.any(valid_d):
            boot_overall_unweighted.append(float(np.mean(d_rmse[valid_d])))
            total_valid_cnt = np.sum(tot_cnt[valid_d])
            if total_valid_cnt > 0:
                boot_overall_weighted.append(float(np.sum(tot_cnt[valid_d] * d_rmse[valid_d]) / total_valid_cnt))
            d_mae = np.where(tot_cnt > 0, tot_abs / np.maximum(tot_cnt, 1), np.nan)
            d_bias = np.where(tot_cnt > 0, tot_sum / np.maximum(tot_cnt, 1), np.nan)
            boot_overall_mae.append(float(np.mean(d_mae[valid_d])))
            boot_overall_bias.append(float(np.mean(d_bias[valid_d])))
                
    ci_records = []
    for d, d_val in enumerate(depth_levels):
        vals = [v for v in boot_rmse_per_depth[d] if not np.isnan(v)]
        if len(vals) > 0:
            ci_low = float(np.percentile(vals, 2.5))
            ci_high = float(np.percentile(vals, 97.5))
        else:
            ci_low = ci_high = np.nan
        ci_records.append({
            "depth_m": float(d_val),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4)
        })

    if not return_overall:
        return ci_records

    overall_ci = {
        "unweighted_rmse": {
            "ci_95_low": round(float(np.percentile(boot_overall_unweighted, 2.5)), 4) if boot_overall_unweighted else np.nan,
            "ci_95_high": round(float(np.percentile(boot_overall_unweighted, 97.5)), 4) if boot_overall_unweighted else np.nan
        },
        "weighted_rmse": {
            "ci_95_low": round(float(np.percentile(boot_overall_weighted, 2.5)), 4) if boot_overall_weighted else np.nan,
            "ci_95_high": round(float(np.percentile(boot_overall_weighted, 97.5)), 4) if boot_overall_weighted else np.nan
        },
        "mae": {
            "ci_95_low": round(float(np.percentile(boot_overall_mae, 2.5)), 4) if boot_overall_mae else np.nan,
            "ci_95_high": round(float(np.percentile(boot_overall_mae, 97.5)), 4) if boot_overall_mae else np.nan
        },
        "bias": {
            "ci_95_low": round(float(np.percentile(boot_overall_bias, 2.5)), 4) if boot_overall_bias else np.nan,
            "ci_95_high": round(float(np.percentile(boot_overall_bias, 97.5)), 4) if boot_overall_bias else np.nan
        }
    }
    return ci_records, overall_ci

