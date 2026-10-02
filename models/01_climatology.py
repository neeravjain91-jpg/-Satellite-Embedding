"""
models/01_climatology.py
Model 1: Training-Only Climatology Baseline

Methodology:
- Computes spatial-depth mean climatological profile:
    T_clim(lat, lon, depth) = mean_{t in Train} [ thetao(t, lat, lon, depth) ]
  computed strictly on the 256-day Training split (Zero Data Leakage).
- Evaluates on Validation split (54 days).
- Evaluates on Test split (56 days) without retraining or threshold adjustment.
- Preserves [N, 15] target masks.
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.tabular_dataset import load_tabular_dataset
from models.base import compute_masked_metrics, print_evaluation_summary, save_model_evaluation
from preprocessing.canonical_grid import CANONICAL_DEPTHS

class SpatialDepthClimatology:
    """
    Computes spatial-depth climatology strictly on the training period:
    T_clim(lat, lon, depth)
    """
    def __init__(self):
        self.spatial_clim_map = {}   # (round_lat, round_lon) -> profile (15,)
        self.global_depth_clim = None # array of shape (15,)
        self.depth_levels = CANONICAL_DEPTHS

    def fit(self, train_data):
        print("Fitting Spatial-Depth Climatology on Training Split (Days 0-255)...")
        Y_train = train_data["Y"]       # (N_train, 15)
        M_train = train_data["mask"]    # (N_train, 15)
        lats = train_data["lat"]
        lons = train_data["lon"]
        
        # 1. Global depth-wise mean
        self.global_depth_clim = np.zeros(15, dtype=np.float32)
        for d in range(15):
            d_mask = M_train[:, d]
            if np.any(d_mask):
                self.global_depth_clim[d] = float(np.mean(Y_train[d_mask, d]))
            else:
                self.global_depth_clim[d] = 15.0

        # 2. Fast vectorized grouping by unique coordinate pair
        # Combine lat and lon into a single unique integer or float key
        # Lat: 5.0 to 30.0 step 0.25 -> 101 values
        # Lon: 45.0 to 105.0 step 0.25 -> 241 values
        coord_keys = np.round(lats, 2) * 1000.0 + np.round(lons, 2)
        unique_keys, inverse_indices = np.unique(coord_keys, return_inverse=True)
        n_unique = len(unique_keys)
        print(f"Aggregating profiles across {n_unique:,} unique ocean grid points...")
        
        # Vectorized sum and count using np.bincount
        clim_profiles = np.zeros((n_unique, 15), dtype=np.float32)
        for d in range(15):
            valid_d = M_train[:, d]
            y_d = np.where(valid_d, Y_train[:, d], 0.0)
            
            sum_d = np.bincount(inverse_indices, weights=y_d, minlength=n_unique)
            count_d = np.bincount(inverse_indices, weights=valid_d.astype(float), minlength=n_unique)
            
            valid_counts = count_d > 0
            clim_profiles[valid_counts, d] = sum_d[valid_counts] / count_d[valid_counts]
            clim_profiles[~valid_counts, d] = self.global_depth_clim[d]

        # Map back to coordinate dictionary
        for u_idx, u_key in enumerate(unique_keys):
            lat_val = round(float(u_key // 1000.0 * 1.0), 2)
            lon_val = round(float(u_key % 1000.0), 2)
            self.spatial_clim_map[(lat_val, lon_val)] = clim_profiles[u_idx]
            
        print("Climatology fitting complete.")

    def predict(self, eval_data):
        lats = eval_data["lat"]
        lons = eval_data["lon"]
        N = len(lats)
        
        coord_keys = np.round(lats, 2) * 1000.0 + np.round(lons, 2)
        unique_keys, inverse_indices = np.unique(coord_keys, return_inverse=True)
        
        # Build lookup table for unique keys
        unique_pred = np.zeros((len(unique_keys), 15), dtype=np.float32)
        for u_idx, u_key in enumerate(unique_keys):
            lat_val = round(float(u_key // 1000.0 * 1.0), 2)
            lon_val = round(float(u_key % 1000.0), 2)
            if (lat_val, lon_val) in self.spatial_clim_map:
                unique_pred[u_idx] = self.spatial_clim_map[(lat_val, lon_val)]
            else:
                unique_pred[u_idx] = self.global_depth_clim
                
        # Vectorized broadcast back to all N samples
        Y_pred = unique_pred[inverse_indices]
        return Y_pred

def run_climatology_benchmark():
    dataset = load_tabular_dataset()
    
    # 1. Fit Climatology strictly on Train
    model = SpatialDepthClimatology()
    model.fit(dataset["train"])
    
    # 2. Evaluate on Validation Split
    print("\n--- Evaluating Climatology on Validation Split (Days 256-309) ---")
    val_pred = model.predict(dataset["val"])
    val_summary, df_val_depths = compute_masked_metrics(
        y_true=dataset["val"]["Y"],
        y_pred=val_pred,
        mask=dataset["val"]["mask"],
        model_name="Climatology",
        split_name="val"
    )
    print_evaluation_summary(val_summary, df_val_depths)
    save_model_evaluation(val_summary, df_val_depths)
    
    # 3. Evaluate on Final Test Split
    print("\n--- Evaluating Climatology on Final Test Split (Days 310-365) ---")
    test_pred = model.predict(dataset["test"])
    test_summary, df_test_depths = compute_masked_metrics(
        y_true=dataset["test"]["Y"],
        y_pred=test_pred,
        mask=dataset["test"]["mask"],
        model_name="Climatology",
        split_name="test"
    )
    print_evaluation_summary(test_summary, df_test_depths)
    save_model_evaluation(test_summary, df_test_depths)
    
    return val_summary, test_summary

if __name__ == "__main__":
    run_climatology_benchmark()
