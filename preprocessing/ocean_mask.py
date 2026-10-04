"""
preprocessing/ocean_mask.py
Defines and enforces explicit mask separation for the North Indian Ocean research domain.

SCIENTIFIC MASK SEMANTICS (GLORYS/ORCA12 MODEL-BATHYMETRY):
This codebase uses GLORYS/ORCA12 numerical model bathymetry as the authoritative validity boundary
because GLORYS thetao is the training/reference target. This represents MODEL-BATHYMETRY validity,
explicitly distinguished from an independent bathymetric survey (e.g. GEBCO or ETOPO).

1. geographic_ocean_mask: (latitude: 101, longitude: 241)
   Meaning: GLORYS/ORCA12 model-domain ocean mask (True for model ocean, False for permanent land).
   Independent of single-depth missing values or temporal dropouts.

2. depth_valid_mask: (depth: 15, latitude: 101, longitude: 241)
   Meaning: Whether canonical depth lies within valid GLORYS/ORCA12 model bathymetry (water column).
   Strictly enforces shallow-water bathymetric cutoffs (no extrapolation into seafloor/sub-seabed).

3. deepest_valid_depth_m: (latitude: 101, longitude: 241)
   Meaning: Maximum canonical depth (in meters) supported by GLORYS/ORCA12 model bathymetry.

4. target_validity_mask: (time: T, depth: 15, latitude: 101, longitude: 241)
   Meaning: Whether thetao is actually valid/observed at that specific time, depth, and location.

5. surface_validity_mask: (time: T, latitude: 101, longitude: 241, feature: 7)
   Meaning: Whether satellite surface predictors are available and valid.
"""

import os
import numpy as np
import xarray as xr
from preprocessing.canonical_grid import CANONICAL_LATS, CANONICAL_LONS, CANONICAL_DEPTHS

MASK_FILE_PATH = "data/processed/canonical_ocean_mask.nc"

def build_canonical_masks(target_array_or_ds):
    """
    Derives the canonical geographic, bathymetric, and deepest-depth masks
    from target potential temperature field (GLORYS).

    target_array_or_ds: xarray Dataset/DataArray or numpy array with
        dims (time, depth, lat, lon) or (depth, lat, lon).
    """
    if isinstance(target_array_or_ds, xr.Dataset):
        vals = target_array_or_ds["thetao"].values
    elif isinstance(target_array_or_ds, xr.DataArray):
        vals = target_array_or_ds.values
    else:
        vals = np.asarray(target_array_or_ds)

    # If 4D (time, depth, lat, lon), aggregate across time to derive static bathymetry
    if vals.ndim == 4:
        # A level is valid if observed in ANY time step (or consensus)
        # GLORYS bathymetry mask is temporally static, so any valid time step reflects water column
        depth_valid_3d = np.any(~np.isnan(vals), axis=0)  # (15, 101, 241)
    elif vals.ndim == 3:
        depth_valid_3d = ~np.isnan(vals)                  # (15, 101, 241)
    else:
        raise ValueError(f"Expected 3D or 4D array for mask derivation, got shape {vals.shape}")

    # 1. Geographic Ocean Mask: Cell is ocean if ANY depth level is valid water column
    # This guarantees that if depth=0m has a numerical gap, it is NOT misclassified as land.
    geo_ocean_2d = np.any(depth_valid_3d, axis=0)  # (101, 241)

    # Ensure depth_valid_mask is False wherever geographic_ocean_mask is False
    depth_valid_3d = depth_valid_3d & geo_ocean_2d[np.newaxis, :, :]

    # 2. Deepest Valid Depth (in meters)
    H, W = geo_ocean_2d.shape
    deepest_depth_2d = np.full((H, W), np.nan, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            if geo_ocean_2d[i, j]:
                valid_levels = CANONICAL_DEPTHS[depth_valid_3d[:, i, j]]
                if len(valid_levels) > 0:
                    deepest_depth_2d[i, j] = float(np.max(valid_levels))

    coords = {
        "depth": CANONICAL_DEPTHS if len(CANONICAL_DEPTHS) == depth_valid_3d.shape[0] else np.arange(depth_valid_3d.shape[0]),
        "latitude": CANONICAL_LATS if len(CANONICAL_LATS) == H else np.arange(H),
        "longitude": CANONICAL_LONS if len(CANONICAL_LONS) == W else np.arange(W)
    }

    ds_mask = xr.Dataset(
        data_vars={
            "geographic_ocean_mask": (
                ["latitude", "longitude"],
                geo_ocean_2d.astype(bool),
                {"description": "GLORYS/ORCA12 model-domain ocean mask (True for ocean, False for permanent land)"}
            ),
            "depth_valid_mask": (
                ["depth", "latitude", "longitude"],
                depth_valid_3d.astype(bool),
                {"description": "True where canonical depth lies within valid GLORYS/ORCA12 model bathymetry, False for land or below model seafloor"}
            ),
            "deepest_valid_depth_m": (
                ["latitude", "longitude"],
                deepest_depth_2d.astype(np.float32),
                {"description": "Maximum canonical depth (m) supported by GLORYS/ORCA12 model bathymetry", "units": "m"}
            ),
            # Backward-compatibility aliases
            "ocean_mask_2d": (
                ["latitude", "longitude"],
                geo_ocean_2d.astype(bool),
                {"description": "Alias for geographic_ocean_mask"}
            ),
            "ocean_mask_3d": (
                ["depth", "latitude", "longitude"],
                depth_valid_3d.astype(bool),
                {"description": "Alias for depth_valid_mask"}
            )
        },
        coords=coords,
        attrs={
            "title": "Canonical Land/Ocean and GLORYS/ORCA12 Model Bathymetry Validity Masks for North Indian Ocean",
            "bathymetry_authority": "GLORYS12V1 (NEMO ORCA12 dynamical model seafloor)",
            "resolution": "0.25 degree x 0.25 degree",
            "domain": "5-30N, 45-105E",
            "canonical_depths_count": len(CANONICAL_DEPTHS)
        }
    )
    return ds_mask

def generate_canonical_ocean_mask_from_glorys(glorys_target_ds):
    """Alias for build_canonical_masks for compatibility."""
    return build_canonical_masks(glorys_target_ds)

def create_target_validity_mask(target_thetao):
    """
    Constructs a 4D boolean target validity mask:
    (time, depth, latitude, longitude) -> True where thetao is a valid real number.
    """
    if isinstance(target_thetao, (xr.DataArray, xr.Dataset)):
        vals = target_thetao["thetao"].values if isinstance(target_thetao, xr.Dataset) else target_thetao.values
    else:
        vals = np.asarray(target_thetao)
    return (~np.isnan(vals)).astype(bool)

def create_surface_validity_mask(surface_features):
    """
    Constructs a 4D boolean surface validity mask:
    (time, latitude, longitude, feature) -> True where the surface predictor is valid.
    """
    if isinstance(surface_features, (xr.DataArray, xr.Dataset)):
        vals = surface_features["surface_features"].values if isinstance(surface_features, xr.Dataset) else surface_features.values
    else:
        vals = np.asarray(surface_features)
    return (~np.isnan(vals)).astype(bool)

def save_canonical_ocean_mask(ds_mask, path=MASK_FILE_PATH):
    """Saves the canonical mask dataset to NetCDF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds_mask.to_netcdf(path)
    print(f"Canonical ocean masks saved to {path}")

def load_canonical_ocean_mask(path=MASK_FILE_PATH):
    """Loads the canonical mask dataset from NetCDF."""
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
