"""
preprocessing/apply_transform.py
Applies previously fitted normalization statistics to any dataset (train, val, test)
and provides inverse transforms for reconstructed temperature profiles.

Prevents data leakage by ensuring no statistics are computed or updated on the evaluation splits.
"""

import json
import os
import numpy as np
import xarray as xr
from preprocessing.canonical_grid import CANONICAL_FEATURES, CANONICAL_DEPTHS

STATS_FILE_PATH = "data/metadata/normalization_stats.json"

class OceanDataTransformer:
    def __init__(self, stats_path=STATS_FILE_PATH):
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Statistics file {stats_path} not found. Run fit_transform first.")
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

    def transform_surface(self, surface_array_or_ds):
        """
        Standardizes surface features (X) using fitted training statistics: (X - mean) / std.
        Maintains NaNs over land.
        """
        if isinstance(surface_array_or_ds, xr.Dataset):
            da = surface_array_or_ds["surface_features"].copy()
            vals = da.values.copy()
        elif isinstance(surface_array_or_ds, xr.DataArray):
            vals = surface_array_or_ds.values.copy()
        else:
            vals = np.array(surface_array_or_ds, copy=True)

        for idx, feat_name in enumerate(CANONICAL_FEATURES):
            f_stats = self.stats["surface_features"][feat_name]
            m, s = f_stats["mean"], f_stats["std"]
            vals[..., idx] = (vals[..., idx] - m) / s

        if isinstance(surface_array_or_ds, xr.Dataset):
            da.values = vals
            return surface_array_or_ds.assign(surface_features=da)
        return vals

    def transform_target(self, target_array_or_ds):
        """Standardizes target thetao (Y) depth by depth: (Y - mean_d) / std_d."""
        if isinstance(target_array_or_ds, xr.Dataset):
            da = target_array_or_ds["thetao"].copy()
            vals = da.values.copy()
        elif isinstance(target_array_or_ds, xr.DataArray):
            vals = target_array_or_ds.values.copy()
        else:
            vals = np.array(target_array_or_ds, copy=True)

        # vals shape: (time, depth, lat, lon) or (depth, lat, lon)
        d_axis = 1 if vals.ndim == 4 else 0
        
        for d_idx, depth_val in enumerate(CANONICAL_DEPTHS):
            d_key = str(int(depth_val))
            d_stats = self.stats.get("target_thetao_per_depth", {}).get(d_key, {"mean": 0.0, "std": 1.0})
            m, s = d_stats["mean"], d_stats["std"]
            if d_axis == 1:
                vals[:, d_idx, :, :] = (vals[:, d_idx, :, :] - m) / s
            else:
                vals[d_idx, :, :] = (vals[d_idx, :, :] - m) / s

        if isinstance(target_array_or_ds, xr.Dataset):
            da.values = vals
            return target_array_or_ds.assign(thetao=da)
        return vals

    def inverse_transform_target(self, norm_target_vals):
        """
        Denormalizes standardized predictions back to physical temperature (°C):
        T = (T_norm * std_d) + mean_d.
        """
        vals = np.array(norm_target_vals, copy=True)
        d_axis = 1 if vals.ndim == 4 else 0

        for d_idx, depth_val in enumerate(CANONICAL_DEPTHS):
            d_key = str(int(depth_val))
            d_stats = self.stats.get("target_thetao_per_depth", {}).get(d_key, {"mean": 0.0, "std": 1.0})
            m, s = d_stats["mean"], d_stats["std"]
            if d_axis == 1:
                vals[:, d_idx, :, :] = (vals[:, d_idx, :, :] * s) + m
            else:
                vals[d_idx, :, :] = (vals[d_idx, :, :] * s) + m

        return vals
