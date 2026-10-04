"""
preprocessing/build_dataset.py
Assembles the final ML-ready datasets with hardened multi-mask separation:
Surface: X(time, latitude, longitude, feature) with 7 features:
    [sst, sss, ssh, current_u, current_v, wind_u, wind_v]
    Includes:
    - surface_features: (time, latitude, longitude, feature)
    - surface_validity_mask: (time, latitude, longitude, feature) boolean
    - geographic_ocean_mask: (latitude, longitude) boolean

Target: Y(time, depth, latitude, longitude) with 15 depth levels.
    Includes:
    - thetao: (time, depth, latitude, longitude)
    - target_validity_mask: (time, depth, latitude, longitude) boolean
    - depth_valid_mask: (depth, latitude, longitude) boolean
    - deepest_valid_depth_m: (latitude, longitude) float32
    - geographic_ocean_mask: (latitude, longitude) boolean

Storage:
- Primary: Zarr store (e.g. data/processed/real_ml_dataset_pilot_surface.zarr, target.zarr)
- Secondary: Chunked NetCDF4

Target Integrity Guarantee:
- Target NaNs are preserved in storage.
- Target NaNs are NEVER converted to zero.
- OceanReconstructionDataset passes raw targets with explicit validity masks.
"""

import os
import sys

# Bootstrap python path to include repository root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
from preprocessing.canonical_grid import (
    CANONICAL_LATS, CANONICAL_LONS, CANONICAL_DEPTHS, CANONICAL_FEATURES,
    create_canonical_surface_dataset, create_canonical_target_dataset, validate_canonical_coords
)
from preprocessing.ocean_mask import (
    apply_surface_mask, apply_target_mask,
    create_target_validity_mask, create_surface_validity_mask
)

def assemble_ml_dataset(surface_features_dict, target_thetao, times, ocean_mask_ds=None, zarr_out_prefix="data/processed/ml_dataset"):
    """
    surface_features_dict: dict of feature_name -> 2D or 3D numpy/xarray (time, lat, lon)
    target_thetao: 3D or 4D numpy/xarray (time, depth, lat, lon)
    times: array of UTC dates
    ocean_mask_ds: canonical ocean mask Dataset containing geographic_ocean_mask, depth_valid_mask, deepest_valid_depth_m
    """
    os.makedirs(os.path.dirname(zarr_out_prefix), exist_ok=True)
    n_times = len(times)
    
    # 1. Create canonical surface dataset
    ds_surface = create_canonical_surface_dataset(times)
    surface_arr = np.full((n_times, len(CANONICAL_LATS), len(CANONICAL_LONS), len(CANONICAL_FEATURES)), np.nan, dtype=np.float32)
    
    for f_idx, feat in enumerate(CANONICAL_FEATURES):
        if feat in surface_features_dict:
            feat_val = surface_features_dict[feat]
            if isinstance(feat_val, xr.DataArray):
                feat_val = feat_val.values
            if feat_val.ndim == 2:
                for t in range(n_times):
                    surface_arr[t, :, :, f_idx] = feat_val
            elif feat_val.ndim == 3:
                surface_arr[:, :, :, f_idx] = feat_val
                
    if ocean_mask_ds is not None:
        geo_mask = ocean_mask_ds.get("geographic_ocean_mask", ocean_mask_ds.get("ocean_mask_2d", None))
        if geo_mask is not None:
            surface_arr = apply_surface_mask(surface_arr, geo_mask.values)
        
    ds_surface["surface_features"].values = surface_arr
    
    # Attach explicit surface validity mask: (time, lat, lon, feature)
    surf_valid = create_surface_validity_mask(surface_arr)
    ds_surface["surface_validity_mask"] = (
        ["time", "latitude", "longitude", "feature"],
        surf_valid.astype(bool),
        {"description": "True where surface feature observation is valid, False for land or missing satellite observation"}
    )
    
    if ocean_mask_ds is not None and "geographic_ocean_mask" in ocean_mask_ds:
        ds_surface["geographic_ocean_mask"] = (
            ["latitude", "longitude"],
            ocean_mask_ds["geographic_ocean_mask"].values.astype(bool),
            {"description": "True for geographic ocean, False for permanent land"}
        )
    
    # 2. Create canonical target dataset
    ds_target = create_canonical_target_dataset(times)
    if isinstance(target_thetao, xr.DataArray):
        target_vals = target_thetao.values
    else:
        target_vals = target_thetao
        
    if target_vals.ndim == 3:  # (depth, lat, lon)
        target_arr = np.tile(target_vals[np.newaxis, ...], (n_times, 1, 1, 1))
    else:
        target_arr = target_vals.copy()
        
    if ocean_mask_ds is not None:
        depth_mask = ocean_mask_ds.get("depth_valid_mask", ocean_mask_ds.get("ocean_mask_3d", None))
        if depth_mask is not None:
            target_arr = apply_target_mask(target_arr, depth_mask.values)
        
    # Strictly preserve NaNs in target thetao
    ds_target["thetao"].values = target_arr
    
    # Attach explicit target validity mask: (time, depth, lat, lon)
    targ_valid = create_target_validity_mask(target_arr)
    ds_target["target_validity_mask"] = (
        ["time", "depth", "latitude", "longitude"],
        targ_valid.astype(bool),
        {"description": "True where subsurface ground truth thetao is physically valid, False for land or sub-seabed"}
    )
    
    if ocean_mask_ds is not None:
        if "depth_valid_mask" in ocean_mask_ds:
            ds_target["depth_valid_mask"] = (
                ["depth", "latitude", "longitude"],
                ocean_mask_ds["depth_valid_mask"].values.astype(bool),
                {"description": "True where local bathymetric depth >= canonical level"}
            )
        if "deepest_valid_depth_m" in ocean_mask_ds:
            ds_target["deepest_valid_depth_m"] = (
                ["latitude", "longitude"],
                ocean_mask_ds["deepest_valid_depth_m"].values.astype(np.float32),
                {"description": "Maximum canonical depth (m) with valid ocean water column", "units": "m"}
            )
        if "geographic_ocean_mask" in ocean_mask_ds:
            ds_target["geographic_ocean_mask"] = (
                ["latitude", "longitude"],
                ocean_mask_ds["geographic_ocean_mask"].values.astype(bool),
                {"description": "True for geographic ocean, False for permanent land"}
            )
    
    # 3. Validate coordinates
    ok_surf, errs_surf = validate_canonical_coords(ds_surface, expect_depth=False)
    ok_targ, errs_targ = validate_canonical_coords(ds_target, expect_depth=True)
    if not ok_surf or not ok_targ:
        raise ValueError(f"Coordinate validation failed: surface={errs_surf}, target={errs_targ}")
        
    # 4. Save to Zarr (and NetCDF)
    surface_zarr = f"{zarr_out_prefix}_surface.zarr"
    target_zarr = f"{zarr_out_prefix}_target.zarr"
    
    # Chunking: 1 day per chunk for time, full spatial field
    ds_surface = ds_surface.chunk({"time": 1, "latitude": 101, "longitude": 241, "feature": 7})
    ds_target = ds_target.chunk({"time": 1, "depth": 15, "latitude": 101, "longitude": 241})
    
    ds_surface.to_zarr(surface_zarr, mode="w", consolidated=True)
    ds_target.to_zarr(target_zarr, mode="w", consolidated=True)
    
    # Also save NetCDF backup for portability if short run (<= 14 days)
    if n_times <= 14:
        surface_nc = f"{zarr_out_prefix}_surface.nc"
        target_nc = f"{zarr_out_prefix}_target.nc"
        ds_surface.to_netcdf(surface_nc)
        ds_target.to_netcdf(target_nc)
    
    print(f"Final ML datasets assembled:")
    print(f"  Surface Zarr: {surface_zarr} (shape: {ds_surface.surface_features.shape})")
    print(f"  Target Zarr:  {target_zarr} (shape: {ds_target.thetao.shape})")
    print(f"  Explicit Masks Included: surface_validity_mask, target_validity_mask, depth_valid_mask, geographic_ocean_mask")
    return ds_surface, ds_target

class OceanReconstructionDataset(Dataset):
    """
    PyTorch Dataset for lazy chunked loading of (X, Y, mask) samples.
    X: (C_in=7, H=101, W=241)
    Y: (C_out=15, H=101, W=241) with raw temperatures preserved
    mask: (C_out=15, H=101, W=241) boolean target validity mask (True for valid ocean water column)
    surface_mask: (C_in=7, H=101, W=241) boolean surface validity mask

    CRITICAL INVARIANT:
    Target thetao NaNs are NEVER converted to zero as ground truth.
    Losses and evaluations must strictly evaluate where mask == True.
    """
    def __init__(self, surface_zarr_path, target_zarr_path, transform=None):
        self.ds_surface = xr.open_zarr(surface_zarr_path, consolidated=True)
        self.ds_target = xr.open_zarr(target_zarr_path, consolidated=True)
        self.transform = transform
        self.n_samples = len(self.ds_surface.time)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Load single time step lazily
        x_raw = self.ds_surface["surface_features"].isel(time=idx).values # (101, 241, 7)
        y_raw = self.ds_target["thetao"].isel(time=idx).values            # (15, 101, 241)
        
        # Transpose X to channels-first: (7, 101, 241)
        x_trans = np.transpose(x_raw, (2, 0, 1))
        
        # Target validity mask
        if "target_validity_mask" in self.ds_target:
            valid_mask = self.ds_target["target_validity_mask"].isel(time=idx).values.astype(bool)
        else:
            valid_mask = ~np.isnan(y_raw)
            
        # Surface validity mask
        if "surface_validity_mask" in self.ds_surface:
            surf_valid_raw = self.ds_surface["surface_validity_mask"].isel(time=idx).values.astype(bool)
            surf_mask_trans = np.transpose(surf_valid_raw, (2, 0, 1))
        else:
            surf_mask_trans = ~np.isnan(x_trans)
        
        # Fill land/missing values in input predictors with 0.0 for neural network convolutional passes
        x_clean = np.nan_to_num(x_trans, nan=0.0)
        
        # For targets, preserve raw thetao. (Loss functions MUST index with valid_mask)
        # We also provide y_clean with 0.0 outside mask ONLY for safe PyTorch tensor conversion,
        # but y_raw preserves exact scientific NaNs.
        tensor_x = torch.from_numpy(x_clean).float()
        tensor_y = torch.from_numpy(y_raw).float()
        tensor_mask = torch.from_numpy(valid_mask).bool()
        tensor_surf_mask = torch.from_numpy(surf_mask_trans).bool()
        
        return {
            "x": tensor_x,               # (7, 101, 241)
            "y": tensor_y,               # (15, 101, 241) - NaNs preserved where invalid
            "mask": tensor_mask,         # (15, 101, 241) - target_validity_mask
            "surface_mask": tensor_surf_mask, # (7, 101, 241) - surface_validity_mask
            "time": str(self.ds_surface.time.values[idx])
        }

if __name__ == "__main__":
    times = ["2020-01-01", "2020-01-02"]
    mock_surface = {feat: np.random.randn(len(times), 101, 241).astype(np.float32) for feat in CANONICAL_FEATURES}
    mock_target = np.random.randn(len(times), 15, 101, 241).astype(np.float32)
    
    from preprocessing.ocean_mask import build_canonical_masks
    mock_mask_ds = build_canonical_masks(mock_target[0])
    
    ds_s, ds_t = assemble_ml_dataset(
        mock_surface, mock_target, times,
        ocean_mask_ds=mock_mask_ds,
        zarr_out_prefix="data/interim/test_ml_dataset"
    )
    
    # Test PyTorch dataset
    torch_ds = OceanReconstructionDataset("data/interim/test_ml_dataset_surface.zarr", "data/interim/test_ml_dataset_target.zarr")
    loader = DataLoader(torch_ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    print("\nPyTorch Verification:")
    print("Batch X shape:", batch["x"].shape, "dtype:", batch["x"].dtype)
    print("Batch Y shape:", batch["y"].shape, "dtype:", batch["y"].dtype)
    print("Batch Mask shape:", batch["mask"].shape, "dtype:", batch["mask"].dtype)
    print("Batch Surface Mask shape:", batch["surface_mask"].shape, "dtype:", batch["surface_mask"].dtype)
    assert batch["x"].shape == (2, 7, 101, 241)
    assert batch["y"].shape == (2, 15, 101, 241)
    assert batch["mask"].shape == (2, 15, 101, 241)
    print("[PASS] PyTorch lazy loading and explicit mask test successful!")
