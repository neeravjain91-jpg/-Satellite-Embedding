"""
preprocessing/build_dataset.py
Assembles the final ML-ready datasets:
Surface: X(time, latitude, longitude, feature) with 7 features:
    [sst, sss, ssh, current_u, current_v, wind_u, wind_v]
Target: Y(time, depth, latitude, longitude) with 15 depth levels.

Storage:
- Primary: Zarr store (e.g. data/processed/ml_dataset_surface.zarr, data/processed/ml_dataset_target.zarr)
- Secondary: Chunked NetCDF4

Includes PyTorch Dataset class for lazy chunked training: OceanReconstructionDataset.
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
from preprocessing.ocean_mask import apply_surface_mask, apply_target_mask

def assemble_ml_dataset(surface_features_dict, target_thetao, times, ocean_mask_ds=None, zarr_out_prefix="data/processed/ml_dataset"):
    """
    surface_features_dict: dict of feature_name -> 2D or 3D numpy/xarray (time, lat, lon)
    target_thetao: 3D or 4D numpy/xarray (time, depth, lat, lon)
    times: array of UTC dates
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
                
    if ocean_mask_ds is not None and "ocean_mask_2d" in ocean_mask_ds:
        m2d = ocean_mask_ds.ocean_mask_2d.values
        surface_arr = apply_surface_mask(surface_arr, m2d)
        
    ds_surface["surface_features"].values = surface_arr
    
    # 2. Create canonical target dataset
    ds_target = create_canonical_target_dataset(times)
    if isinstance(target_thetao, xr.DataArray):
        target_vals = target_thetao.values
    else:
        target_vals = target_thetao
        
    if target_vals.ndim == 3:  # (depth, lat, lon)
        target_arr = np.tile(target_vals[np.newaxis, ...], (n_times, 1, 1, 1))
    else:
        target_arr = target_vals
        
    if ocean_mask_ds is not None and "ocean_mask_3d" in ocean_mask_ds:
        m3d = ocean_mask_ds.ocean_mask_3d.values
        target_arr = apply_target_mask(target_arr, m3d)
        
    ds_target["thetao"].values = target_arr
    
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
    
    # Also save netcdf backup for portability
    surface_nc = f"{zarr_out_prefix}_surface.nc"
    target_nc = f"{zarr_out_prefix}_target.nc"
    ds_surface.to_netcdf(surface_nc)
    ds_target.to_netcdf(target_nc)
    
    print(f"Final ML datasets assembled:")
    print(f"  Surface Zarr: {surface_zarr} (shape: {ds_surface.surface_features.shape})")
    print(f"  Target Zarr: {target_zarr} (shape: {ds_target.thetao.shape})")
    return ds_surface, ds_target

class OceanReconstructionDataset(Dataset):
    """
    PyTorch Dataset for lazy chunked loading of (X, Y, mask) samples.
    X: (C_in=7, H=101, W=241)
    Y: (C_out=15, H=101, W=241)
    mask: (C_out=15, H=101, W=241) boolean validity mask (True for valid ocean water column)
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
        
        # Create validity mask: True where Y is not NaN
        valid_mask = ~np.isnan(y_raw)
        
        # Replace NaNs in inputs and targets with 0 for neural network computation
        x_clean = np.nan_to_num(x_trans, nan=0.0)
        y_clean = np.nan_to_num(y_raw, nan=0.0)
        
        # Convert to PyTorch tensors
        tensor_x = torch.from_numpy(x_clean).float()
        tensor_y = torch.from_numpy(y_clean).float()
        tensor_mask = torch.from_numpy(valid_mask).bool()
        
        return {
            "x": tensor_x,       # (7, 101, 241)
            "y": tensor_y,       # (15, 101, 241)
            "mask": tensor_mask, # (15, 101, 241)
            "time": str(self.ds_surface.time.values[idx])
        }

if __name__ == "__main__":
    # Test dataset creation and PyTorch DataLoader
    times = ["2020-01-01", "2020-01-02"]
    mock_surface = {feat: np.random.randn(len(times), 101, 241).astype(np.float32) for feat in CANONICAL_FEATURES}
    mock_target = np.random.randn(len(times), 15, 101, 241).astype(np.float32)
    
    ds_s, ds_t = assemble_ml_dataset(mock_surface, mock_target, times, zarr_out_prefix="data/interim/test_ml_dataset")
    
    # Test PyTorch dataset
    torch_ds = OceanReconstructionDataset("data/interim/test_ml_dataset_surface.zarr", "data/interim/test_ml_dataset_target.zarr")
    loader = DataLoader(torch_ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    print("\nPyTorch Verification:")
    print("Batch X shape:", batch["x"].shape, "dtype:", batch["x"].dtype)
    print("Batch Y shape:", batch["y"].shape, "dtype:", batch["y"].dtype)
    print("Batch Mask shape:", batch["mask"].shape, "dtype:", batch["mask"].dtype)
    assert batch["x"].shape == (2, 7, 101, 241)
    assert batch["y"].shape == (2, 15, 101, 241)
    print("[PASS] PyTorch lazy loading test successful!")
