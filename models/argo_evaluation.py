"""
models/argo_evaluation.py
Comprehensive in-situ ARGO evaluation infrastructure adhering strictly to Section 4
of the Scientific ML Experiment Protocol.

Explicit Dual Evaluation Tracks:
  1. "ARGO–GLORYS reference consistency assessment":
     Quantifies the observational discrepancy between GLORYS reanalysis target and in-situ ARGO floats.
  2. "ML–ARGO observational evaluation":
     Evaluates whether ML surface-to-depth predictions degrade or preserve reanalysis fidelity against in-situ ARGO.

Eligibility & QC Rules:
- Test-period only: Profile date strictly within test partition (days 313–365: Nov 9 – Dec 31, 2020)
- Temporal tolerance: +/- 1 day (<= 24.0 hours)
- Spatial matching: Nearest-neighbor to 0.25 deg grid (<= 0.25 deg / 25 km), bathymetric depth >= 1000 m
- Vertical interpolation: Linear interpolation to canonical depths, strictly no extrapolation
- Quality Control:
    * QC flag == 1 (good data)
    * Spike test: temperature gradient < 5°C / 10 dbar
    * Diurnal test: T < 35°C in top 10 m
    * Minimum valid levels >= 5
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_DEPTHS

def filter_eligible_argo_profiles(df_argo, test_start_date="2020-11-09", test_end_date="2020-12-31"):
    """
    Filters ARGO profiles strictly for test-period eligibility and scientific QC standards.
    """
    df = df_argo.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    
    # 1. Temporal withholding: Test-period eligibility only
    t_start = pd.to_datetime(test_start_date)
    t_end = pd.to_datetime(test_end_date)
    df = df[(df["time"] >= t_start) & (df["time"] <= t_end)]
    
    # 2. QC flag check (flag == 1)
    if "quality_flag" in df.columns:
        df = df[df["quality_flag"].astype(int) == 1]
        
    # Group by profile (argo_id, time) to perform profile-level checks
    profile_groups = df.groupby(["argo_id", "time"])
    eligible_dfs = []
    
    for (argo_id, p_time), prof in profile_groups:
        prof = prof.sort_values("depth")
        # Check minimum levels
        if len(prof) < 5:
            continue
            
        depths = prof["depth"].values
        temps = prof["observed_temperature"].values
        
        # Diurnal check: T < 35°C in top 10m
        shallow_mask = (depths <= 10.0)
        if np.any(shallow_mask) and np.any(temps[shallow_mask] >= 35.0):
            continue
            
        # Spike test: gradient < 5°C per 10 dbar (approx 10 m)
        d_depth = np.diff(depths)
        d_temp = np.abs(np.diff(temps))
        valid_steps = d_depth > 0.5
        if np.any(valid_steps):
            gradient_per_10m = (d_temp[valid_steps] / d_depth[valid_steps]) * 10.0
            if np.any(gradient_per_10m > 5.0):
                continue
                
        eligible_dfs.append(prof)
        
    if len(eligible_dfs) == 0:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(eligible_dfs, ignore_index=True)


def interpolate_profile_to_canonical_depths(prof_depths, prof_temps, canonical_depths=CANONICAL_DEPTHS):
    """
    Interpolates observed profile temperatures to canonical depths strictly without extrapolation.
    """
    prof_depths = np.asarray(prof_depths)
    prof_temps = np.asarray(prof_temps)
    
    # Sort depths
    s_idx = np.argsort(prof_depths)
    z_sorted = prof_depths[s_idx]
    t_sorted = prof_temps[s_idx]
    
    min_z, max_z = z_sorted[0], z_sorted[-1]
    f_interp = interp1d(z_sorted, t_sorted, kind="linear", bounds_error=False, fill_value=np.nan)
    
    out_temps = f_interp(canonical_depths)
    # Ensure no extrapolation beyond observed profile span
    out_temps[canonical_depths < min_z] = np.nan
    out_temps[canonical_depths > max_z] = np.nan
    return out_temps


def compute_dual_argo_evaluation(matched_records_df, n_bootstrap=1000, random_seed=42):
    """
    Computes the authoritative paired evaluation metrics:
    Track 1: "ARGO–GLORYS reference consistency assessment"
    Track 2: "ML–ARGO observational evaluation"
    
    matched_records_df must contain:
      ['profile_id', 'depth', 'observed_temp', 'glorys_temp', 'ml_temp', 'latitude', 'longitude']
    """
    df = matched_records_df.dropna(subset=['observed_temp', 'glorys_temp', 'ml_temp']).copy()
    
    # Paired errors
    df["error_glorys"] = df["glorys_temp"] - df["observed_temp"]
    df["error_ml"] = df["ml_temp"] - df["observed_temp"]
    
    # Overall Paired Metrics
    rmse_glorys = float(np.sqrt(np.mean(df["error_glorys"] ** 2)))
    rmse_ml = float(np.sqrt(np.mean(df["error_ml"] ** 2)))
    bias_glorys = float(np.mean(df["error_glorys"]))
    bias_ml = float(np.mean(df["error_ml"]))
    delta_rmse = float(rmse_ml - rmse_glorys)
    
    # Depth Bins: 0-50m, 50-200m, 200-500m, 500-1000m
    bins = [
        ("0-50m", (df["depth"] >= 0) & (df["depth"] <= 50)),
        ("50-200m", (df["depth"] > 50) & (df["depth"] <= 200)),
        ("200-500m", (df["depth"] > 200) & (df["depth"] <= 500)),
        ("500-1000m", (df["depth"] > 500) & (df["depth"] <= 1000))
    ]
    depth_bin_metrics = []
    for b_name, b_mask in bins:
        sub = df[b_mask]
        if len(sub) == 0:
            continue
        depth_bin_metrics.append({
            "depth_bin": b_name,
            "n_points": len(sub),
            "glorys_rmse": round(float(np.sqrt(np.mean(sub["error_glorys"] ** 2))), 3),
            "ml_rmse": round(float(np.sqrt(np.mean(sub["error_ml"] ** 2))), 3),
            "glorys_bias": round(float(np.mean(sub["error_glorys"])), 3),
            "ml_bias": round(float(np.mean(sub["error_ml"])), 3),
            "delta_rmse": round(float(np.sqrt(np.mean(sub["error_ml"] ** 2)) - np.sqrt(np.mean(sub["error_glorys"] ** 2))), 3)
        })

    # Bootstrap 95% CI on profiles
    np.random.seed(random_seed)
    unique_profiles = df["profile_id"].unique() if "profile_id" in df.columns else np.arange(len(df))
    n_p = len(unique_profiles)
    
    boot_rmse_g, boot_rmse_ml = [], []
    for _ in range(n_bootstrap):
        sampled_p = np.random.choice(unique_profiles, size=n_p, replace=True)
        if "profile_id" in df.columns:
            boot_df = df[df["profile_id"].isin(sampled_p)]
        else:
            boot_df = df.iloc[sampled_p]
            
        if len(boot_df) > 0:
            boot_rmse_g.append(float(np.sqrt(np.mean(boot_df["error_glorys"] ** 2))))
            boot_rmse_ml.append(float(np.sqrt(np.mean(boot_df["error_ml"] ** 2))))
            
    ci_glorys = [float(np.percentile(boot_rmse_g, 2.5)), float(np.percentile(boot_rmse_g, 97.5))] if len(boot_rmse_g) > 0 else [np.nan, np.nan]
    ci_ml = [float(np.percentile(boot_rmse_ml, 2.5)), float(np.percentile(boot_rmse_ml, 97.5))] if len(boot_rmse_ml) > 0 else [np.nan, np.nan]

    return {
        "assessment_label_1": "ARGO–GLORYS reference consistency assessment",
        "assessment_label_2": "ML–ARGO observational evaluation",
        "total_matched_points": len(df),
        "overall": {
            "glorys_rmse": round(rmse_glorys, 3),
            "glorys_rmse_95_ci": [round(ci_glorys[0], 3), round(ci_glorys[1], 3)],
            "ml_rmse": round(rmse_ml, 3),
            "ml_rmse_95_ci": [round(ci_ml[0], 3), round(ci_ml[1], 3)],
            "glorys_bias": round(bias_glorys, 3),
            "ml_bias": round(bias_ml, 3),
            "delta_rmse": round(delta_rmse, 3)
        },
        "depth_bins": depth_bin_metrics
    }
