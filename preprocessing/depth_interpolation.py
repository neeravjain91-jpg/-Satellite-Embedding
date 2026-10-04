"""
preprocessing/depth_interpolation.py
Performs vertical interpolation from native GLORYS depth levels to the 15 canonical target depths:
[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] meters.

Physical constraints:
1. Interpolates once only.
2. At depth = 0m, extends the topmost native level (0.494m) assuming well-mixed surface layer.
3. Where the ocean is shallower than a target depth, the value remains NaN (no fabrication).
4. Maintains land/bathymetry mask integrity.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

CANONICAL_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)

def interpolate_depths_1d(profile, native_depths, target_depths=CANONICAL_DEPTHS):
    """
    Interpolates a single 1D temperature profile from native depths to target depths.
    Preserves NaNs where ocean is shallower than the target depth.
    """
    valid_mask = ~np.isnan(profile)
    if not np.any(valid_mask):
        return np.full(len(target_depths), np.nan, dtype=np.float32)
    
    valid_depths = native_depths[valid_mask]
    valid_temps = profile[valid_mask]
    
    if len(valid_temps) == 0:
        return np.full(len(target_depths), np.nan, dtype=np.float32)
        
    max_valid_depth = valid_depths[-1]
    min_valid_depth = valid_depths[0]
    
    # Linear interpolator without extrapolation below max depth
    f = interp1d(valid_depths, valid_temps, kind='linear', bounds_error=False, fill_value=np.nan)
    interp_vals = f(target_depths)
    
    # For target depth 0m: if min_valid_depth <= 2.0m, set surface value = topmost valid value
    if target_depths[0] == 0.0 and np.isnan(interp_vals[0]) and min_valid_depth <= 2.0:
        interp_vals[0] = valid_temps[0]
        
    # Strictly enforce: any target depth deeper than max_valid_depth must be NaN
    interp_vals[target_depths > max_valid_depth] = np.nan
    
    return interp_vals.astype(np.float32)

def interpolate_glorys_to_canonical_depths(da_thetao, target_depths=CANONICAL_DEPTHS):
    """
    Takes an xarray DataArray of GLORYS thetao (dims: [time, depth, lat, lon] or [depth, lat, lon])
    and interpolates along the depth coordinate to target_depths in a fast, vectorized manner.
    Preserves NaNs where ocean is shallower than target depth (no extrapolation below max valid depth).
    Returns DataArray with dimension 'depth' matching target_depths.
    """
    native_depths = np.asarray(da_thetao.depth.values, dtype=np.float32)
    target_depths = np.asarray(target_depths, dtype=np.float32)
    
    # If already exact canonical depths
    if np.array_equal(native_depths, target_depths):
        return da_thetao
        
    depth_axis = da_thetao.dims.index("depth")
    vals = da_thetao.values
    
    # Move depth to axis 0 for vectorization
    vals_trans = np.moveaxis(vals, depth_axis, 0)
    
    n_targ = len(target_depths)
    out_shape = (n_targ,) + vals_trans.shape[1:]
    out = np.full(out_shape, np.nan, dtype=np.float32)
    
    indices = np.searchsorted(native_depths, target_depths)
    
    for k, z in enumerate(target_depths):
        if z == 0.0:
            if native_depths[0] <= 2.0:
                out[k] = vals_trans[0]
        else:
            idx = indices[k]
            if idx == 0:
                out[k] = vals_trans[0]
            elif idx >= len(native_depths):
                pass  # deeper than maximum native depth -> strictly NaN
            else:
                z0 = native_depths[idx - 1]
                z1 = native_depths[idx]
                w1 = (z - z0) / (z1 - z0)
                w0 = 1.0 - w1
                out[k] = w0 * vals_trans[idx - 1] + w1 * vals_trans[idx]
                
    # Move depth back to original axis
    out = np.moveaxis(out, 0, depth_axis)
    
    new_dims = list(da_thetao.dims)
    new_coords = dict(da_thetao.coords)
    new_coords["depth"] = target_depths
    
    result_data = xr.DataArray(
        out,
        dims=new_dims,
        coords=new_coords,
        attrs=da_thetao.attrs,
        name=da_thetao.name or "thetao"
    )
    return result_data

if __name__ == "__main__":
    # Test with simulated native GLORYS depths
    native_d = np.array([0.49, 1.54, 2.65, 5.08, 9.57, 15.81, 25.21, 40.34, 65.81, 109.73, 186.13, 318.13, 541.09, 902.34, 1245.29])
    temp_profile = 28.0 - 0.02 * native_d  # Mock cooling with depth
    temp_profile[native_d > 600] = np.nan  # Simulating seabed at 600m
    
    interp = interpolate_depths_1d(temp_profile, native_d, CANONICAL_DEPTHS)
    print("Native depths:", native_d)
    print("Native temps:", temp_profile)
    print("Canonical depths:", CANONICAL_DEPTHS)
    print("Interpolated temps:", interp)
    assert not np.isnan(interp[0]), "Surface 0m should not be NaN"
    assert not np.isnan(interp[CANONICAL_DEPTHS == 500][0]), "500m should be valid"
    assert np.isnan(interp[CANONICAL_DEPTHS == 700][0]), "700m should be NaN (seabed at 600m)"
    assert np.isnan(interp[CANONICAL_DEPTHS == 1000][0]), "1000m should be NaN (seabed at 600m)"
    print("[PASS] Depth interpolation validation successful.")
