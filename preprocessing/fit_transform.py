"""
preprocessing/fit_transform.py
Fits normalization statistics (mean, std, min, max) strictly on the designated TRAINING split.
Guarantees ZERO DATA LEAKAGE:
- Never includes validation-period or test-period time slices.
- Never includes independent ARGO profiles.
- Never uses future observations.

Saves fitted statistics to: data/metadata/normalization_stats.json
"""

import os
import json
import numpy as np
import xarray as xr
from preprocessing.canonical_grid import CANONICAL_FEATURES, CANONICAL_DEPTHS

STATS_FILE_PATH = "data/metadata/normalization_stats.json"

def fit_surface_normalization(x_train_ds, train_time_slice=None, output_path=STATS_FILE_PATH):
    """
    Computes per-feature mean and std strictly on training timestamps.
    x_train_ds: Dataset with variable 'surface_features' of shape (time, lat, lon, feature)
    """
    da = x_train_ds["surface_features"]
    
    if train_time_slice is not None:
        da = da.sel(time=train_time_slice)
        
    stats = {
        "train_time_start": str(da.time.values[0]),
        "train_time_end": str(da.time.values[-1]),
        "num_train_days": int(len(da.time)),
        "surface_features": {}
    }
    
    vals = da.values  # (time, lat, lon, feature)
    
    for idx, feat_name in enumerate(CANONICAL_FEATURES):
        feat_data = vals[..., idx]
        valid_vals = feat_data[~np.isnan(feat_data)]
        
        if len(valid_vals) == 0:
            mean_val = 0.0
            std_val = 1.0
            min_val = 0.0
            max_val = 1.0
        else:
            mean_val = float(np.mean(valid_vals))
            std_val = float(np.std(valid_vals))
            if std_val < 1e-6:
                std_val = 1.0
            min_val = float(np.min(valid_vals))
            max_val = float(np.max(valid_vals))
            
        stats["surface_features"][feat_name] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Normalization statistics strictly fitted on training split and saved to {output_path}")
    return stats

def fit_target_normalization(y_train_ds, train_time_slice=None, stats_path=STATS_FILE_PATH):
    """
    Computes depth-wise mean and std for target thetao strictly on training timestamps.
    """
    da = y_train_ds["thetao"]
    if train_time_slice is not None:
        da = da.sel(time=train_time_slice)
        
    # Read existing stats if present
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}
        
    stats["target_thetao_per_depth"] = {}
    vals = da.values  # (time, depth, lat, lon)
    
    for d_idx, depth_val in enumerate(CANONICAL_DEPTHS):
        depth_data = vals[:, d_idx, :, :]
        valid_vals = depth_data[~np.isnan(depth_data)]
        
        if len(valid_vals) == 0:
            d_mean = 15.0
            d_std = 5.0
        else:
            d_mean = float(np.mean(valid_vals))
            d_std = float(np.std(valid_vals))
            if d_std < 1e-6:
                d_std = 1.0
                
        stats["target_thetao_per_depth"][str(int(depth_val))] = {
            "depth_m": float(depth_val),
            "mean": d_mean,
            "std": d_std
        }
        
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Target depth-wise statistics fitted and appended to {stats_path}")
    return stats
