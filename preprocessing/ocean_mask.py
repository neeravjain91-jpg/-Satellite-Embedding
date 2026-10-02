"""
preprocessing/ocean_mask.py
Defines and maintains ONE canonical ocean mask for the North Indian Ocean domain:
- 2D Surface Ocean Mask: (latitude, longitude) -> True for ocean, False for land.
- 3D Subsurface Bathymetry Mask: (depth, latitude, longitude) -> True where ocean depth >= level, False for land or below seabed.

Prevents:
1. Interpolation of oceanic variables over landmasses (India, Arabia, Indochina, etc.).
2. Fabrication of deep-water temperatures where the seafloor is shallower than target depths.
"""

import os
import numpy as np
import xarray as xr
from preprocessing.canonical_grid import CANONICAL_LATS, CANONICAL_LONS, CANONICAL_DEPTHS

MASK_FILE_PATH = "data/processed/canonical_ocean_mask.nc"

def generate_canonical_ocean_mask_from_glorys(glorys_target_ds):
    """
    Derives the canonical 2D and 3D ocean masks from a canonical-grid GLORYS target dataset.
    """
    # glorys_target_ds has coords: (time, depth, latitude, longitude) or (depth, latitude, longitude)
    if "time" in glorys_target_ds.dims:
        # Take the first time step or consensus across time steps
        sample = glorys_target_ds["thetao"].isel(time=0)
    else:
        sample = glorys_target_ds["thetao"]
        
    # 3D Mask: True where temperature is valid (i.e. not land and above seabed)
    mask_3d = ~np.isnan(sample.values)
    
    # 2D Mask: Surface level (depth index 0)
    mask_2d = mask_3d[0, :, :]
    
    ds_mask = xr.Dataset(
        data_vars={
            "ocean_mask_2d": (["latitude", "longitude"], mask_2d.astype(bool), {"description": "True for ocean, False for land"}),
            "ocean_mask_3d": (["depth", "latitude", "longitude"], mask_3d.astype(bool), {"description": "True for ocean water column, False for land or below seabed"})
        },
        coords={
            "depth": CANONICAL_DEPTHS,
            "latitude": CANONICAL_LATS,
            "longitude": CANONICAL_LONS
        },
        attrs={
            "title": "Canonical Land/Ocean and Bathymetry Mask for North Indian Ocean",
            "resolution": "0.25 degree",
            "domain": "5-30N, 45-105E"
        }
    )
    return ds_mask

def save_canonical_ocean_mask(ds_mask, path=MASK_FILE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds_mask.to_netcdf(path)
    print(f"Canonical ocean mask saved to {path}")

def load_canonical_ocean_mask(path=MASK_FILE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ocean mask not found at {path}. Generate it first using GLORYS or bathymetry.")
    return xr.open_dataset(path)

def apply_surface_mask(data_array_or_numpy, mask_2d):
    """Masks land points with NaN."""
    out = data_array_or_numpy.copy()
    if isinstance(out, xr.DataArray):
        return out.where(mask_2d, np.nan)
    else:
        if out.ndim == 4:  # (time, lat, lon, feature)
            for f in range(out.shape[-1]):
                for t in range(out.shape[0]):
                    out[t, ~mask_2d, f] = np.nan
        elif out.ndim == 3:  # (time, lat, lon) or (lat, lon, feature)
            for i in range(out.shape[0]):
                out[i, ~mask_2d] = np.nan
        elif out.ndim == 2:
            out[~mask_2d] = np.nan
        return out

def apply_target_mask(target_data, mask_3d):
    """Masks land and sub-seabed points with NaN."""
    out = target_data.copy()
    if isinstance(out, xr.DataArray):
        return out.where(mask_3d, np.nan)
    else:
        if out.ndim == 4:  # (time, depth, lat, lon)
            for t in range(out.shape[0]):
                out[t, ~mask_3d] = np.nan
        elif out.ndim == 3:  # (depth, lat, lon)
            out[~mask_3d] = np.nan
        return out
