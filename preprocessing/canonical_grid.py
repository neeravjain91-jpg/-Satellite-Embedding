"""
preprocessing/canonical_grid.py
Defines the single canonical grid for the North Indian Ocean research domain:
Latitude: 5.00°N to 30.00°N (0.25° spacing, 101 points)
Longitude: 45.00°E to 105.00°E (0.25° spacing, 241 points)
Depth: 15 standard levels: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] meters.
"""

import numpy as np
import xarray as xr

# Canonical coordinate specifications
CANONICAL_LATS = np.round(np.arange(5.00, 30.25, 0.25), 2)  # 101 points
CANONICAL_LONS = np.round(np.arange(45.00, 105.25, 0.25), 2)  # 241 points
CANONICAL_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)

CANONICAL_FEATURES = [
    "sst",
    "sss",
    "ssh",
    "current_u",
    "current_v",
    "wind_u",
    "wind_v"
]

REGIONAL_BOUNDS = {
    "entire_domain": {"lat_min": 5.0, "lat_max": 30.0, "lon_min": 45.0, "lon_max": 105.0},
    "arabian_sea":   {"lat_min": 5.0, "lat_max": 30.0, "lon_min": 45.0, "lon_max": 77.5},
    "bay_of_bengal": {"lat_min": 5.0, "lat_max": 30.0, "lon_min": 77.5, "lon_max": 105.0}
}

def get_region_mask(region_name, lats=CANONICAL_LATS, lons=CANONICAL_LONS):
    """
    Returns a 2D boolean mask (latitude, longitude) for the given regional sub-domain.
    """
    if region_name not in REGIONAL_BOUNDS:
        raise KeyError(f"Unknown region '{region_name}'. Available: {list(REGIONAL_BOUNDS.keys())}")
    bounds = REGIONAL_BOUNDS[region_name]
    lat_mask = (lats >= bounds["lat_min"]) & (lats <= bounds["lat_max"])
    lon_mask = (lons >= bounds["lon_min"]) & (lons <= bounds["lon_max"])
    return np.outer(lat_mask, lon_mask)


def get_canonical_coords():
    """Returns a dictionary of canonical coordinates."""
    return {
        "latitude": CANONICAL_LATS,
        "longitude": CANONICAL_LONS,
        "depth": CANONICAL_DEPTHS,
        "feature": CANONICAL_FEATURES
    }

def create_canonical_surface_dataset(times):
    """
    Creates an empty canonical surface dataset for given UTC timestamps:
    Dimensions: (time, latitude, longitude, feature)
    """
    time_arr = np.array(times, dtype="datetime64[ns]")
    coords = {
        "time": time_arr,
        "latitude": CANONICAL_LATS,
        "longitude": CANONICAL_LONS,
        "feature": CANONICAL_FEATURES
    }
    
    empty_data = np.full(
        (len(time_arr), len(CANONICAL_LATS), len(CANONICAL_LONS), len(CANONICAL_FEATURES)),
        np.nan,
        dtype=np.float32
    )
    
    da = xr.DataArray(
        empty_data,
        coords=coords,
        dims=["time", "latitude", "longitude", "feature"],
        name="surface_features"
    )
    
    ds = da.to_dataset()
    ds.latitude.attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds.longitude.attrs = {"units": "degrees_east", "standard_name": "longitude"}
    ds.attrs = {
        "title": "Standardized Surface Inputs for North Indian Ocean",
        "spatial_resolution": "0.25 deg x 0.25 deg",
        "temporal_resolution": "daily",
        "domain": "5-30N, 45-105E"
    }
    return ds

def create_canonical_target_dataset(times):
    """
    Creates an empty canonical target dataset:
    Dimensions: (time, depth, latitude, longitude)
    with exactly 15 depth levels.
    """
    time_arr = np.array(times, dtype="datetime64[ns]")
    coords = {
        "time": time_arr,
        "depth": CANONICAL_DEPTHS,
        "latitude": CANONICAL_LATS,
        "longitude": CANONICAL_LONS
    }
    
    empty_data = np.full(
        (len(time_arr), len(CANONICAL_DEPTHS), len(CANONICAL_LATS), len(CANONICAL_LONS)),
        np.nan,
        dtype=np.float32
    )
    
    da = xr.DataArray(
        empty_data,
        coords=coords,
        dims=["time", "depth", "latitude", "longitude"],
        name="thetao",
        attrs={"units": "degC", "standard_name": "sea_water_potential_temperature"}
    )
    
    ds = da.to_dataset()
    ds.latitude.attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds.longitude.attrs = {"units": "degrees_east", "standard_name": "longitude"}
    ds.depth.attrs = {"units": "m", "standard_name": "depth", "positive": "down"}
    ds.attrs = {
        "title": "Standardized Subsurface Target Potential Temperature (GLORYS)",
        "spatial_resolution": "0.25 deg x 0.25 deg",
        "depth_levels_count": 15,
        "domain": "5-30N, 45-105E"
    }
    return ds

def validate_canonical_coords(ds, expect_depth=False):
    """Validates that a dataset conforms exactly to canonical coordinates."""
    errors = []
    if "latitude" not in ds.coords:
        errors.append("Missing 'latitude' coordinate")
    elif len(ds.latitude) != len(CANONICAL_LATS) or not np.allclose(ds.latitude.values, CANONICAL_LATS):
        errors.append(f"Latitude mismatch: got {len(ds.latitude)} points, expected {len(CANONICAL_LATS)}")
        
    if "longitude" not in ds.coords:
        errors.append("Missing 'longitude' coordinate")
    elif len(ds.longitude) != len(CANONICAL_LONS) or not np.allclose(ds.longitude.values, CANONICAL_LONS):
        errors.append(f"Longitude mismatch: got {len(ds.longitude)} points, expected {len(CANONICAL_LONS)}")
        
    if expect_depth:
        if "depth" not in ds.coords:
            errors.append("Missing 'depth' coordinate")
        elif len(ds.depth) != len(CANONICAL_DEPTHS) or not np.allclose(ds.depth.values, CANONICAL_DEPTHS):
            errors.append(f"Depth mismatch: got {len(ds.depth)} levels, expected {len(CANONICAL_DEPTHS)}")
            
    return len(errors) == 0, errors

if __name__ == "__main__":
    print(f"Canonical Latitudes: {len(CANONICAL_LATS)} points ({CANONICAL_LATS[0]} to {CANONICAL_LATS[-1]})")
    print(f"Canonical Longitudes: {len(CANONICAL_LONS)} points ({CANONICAL_LONS[0]} to {CANONICAL_LONS[-1]})")
    print(f"Canonical Depths: {len(CANONICAL_DEPTHS)} levels: {CANONICAL_DEPTHS}")
    
    test_surf = create_canonical_surface_dataset(["2020-01-01"])
    ok, errs = validate_canonical_coords(test_surf, expect_depth=False)
    print("Surface dataset validation:", ok, errs)
    
    test_target = create_canonical_target_dataset(["2020-01-01"])
    ok, errs = validate_canonical_coords(test_target, expect_depth=True)
    print("Target dataset validation:", ok, errs)
