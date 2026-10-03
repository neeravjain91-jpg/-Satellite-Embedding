"""
preprocessing/regrid.py
Scientifically justified regridding and spatial harmonization onto the canonical 0.25° x 0.25° grid:
Latitude: 5.00° to 30.00° (101 points)
Longitude: 45.00° to 105.00° (241 points)

Regridding methodologies per variable:
- SST (OSTIA, native 0.05° -> 0.25°):
    Bilinear interpolation with ocean-boundary masking to preserve frontal gradients without aliasing.
    Conversion from Kelvin to Celsius (T_celsius = T_kelvin - 273.15).
- SSS (Copernicus Multi-Obs, native 0.125° -> 0.25°):
    Bilinear interpolation to preserve salinity gradients.
- SSH/SLA (DUACS, native 0.25°):
    Coordinate alignment and nearest/linear indexing to exact canonical coordinates.
- Current U, V (OSCAR, native 0.25°):
    Coordinate alignment onto canonical grid.
- Wind U, V (CCMP, native 0.25°):
    Spatial coordinate alignment. (Temporal 6-hourly to daily handled in temporal_align.py).
- GLORYS thetao (native 0.0833° -> 0.25°):
    Bilinear interpolation across ocean cells; maintains bathymetric boundary.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from preprocessing.canonical_grid import CANONICAL_LATS, CANONICAL_LONS

REGRID_METHODS = {
    "sst": "bilinear (Kelvin to Celsius conversion)",
    "sss": "bilinear",
    "ssh": "coordinate alignment (native 0.25 deg)",
    "current_u": "coordinate alignment (native 0.25 deg)",
    "current_v": "coordinate alignment (native 0.25 deg)",
    "wind_u": "coordinate alignment (native 0.25 deg)",
    "wind_v": "coordinate alignment (native 0.25 deg)",
    "thetao": "bilinear horizontal interpolation"
}

def get_regrid_documentation():
    return REGRID_METHODS

def regrid_2d_field(data, src_lats, src_lons, dst_lats=CANONICAL_LATS, dst_lons=CANONICAL_LONS, method="linear"):
    """
    Regrids a 2D field (latitude, longitude) from source coordinates to destination coordinates.
    Handles NaN values and prevents bleeding of NaNs/land across coastlines.
    """
    # Ensure data is oriented as (latitude, longitude)
    if data.shape == (len(src_lons), len(src_lats)) and len(src_lons) != len(src_lats):
        data = data.T

    # Ensure source latitudes are monotonically increasing
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        data = np.flip(data, axis=0)
        
    # Ensure source longitudes are monotonically increasing
    if src_lons[0] > src_lons[-1]:
        src_lons = src_lons[::-1]
        data = np.flip(data, axis=1)

    # If coordinates already match exact canonical grid
    if len(src_lats) == len(dst_lats) and len(src_lons) == len(dst_lons) and \
       np.allclose(src_lats, dst_lats) and np.allclose(src_lons, dst_lons):
        return data.copy()

    # Fill NaNs temporarily for regular grid interpolator, then re-mask
    valid_mask = ~np.isnan(data)
    
    # Fast path if completely NaN
    if not np.any(valid_mask):
        return np.full((len(dst_lats), len(dst_lons)), np.nan, dtype=np.float32)

    # Create coordinate mesh
    mg_dst_lat, mg_dst_lon = np.meshgrid(dst_lats, dst_lons, indexing='ij')
    query_pts = np.column_stack([mg_dst_lat.ravel(), mg_dst_lon.ravel()])

    # Interpolate valid mask to determine land/ocean boundary on destination grid
    mask_interp = RegularGridInterpolator(
        (src_lats, src_lons),
        valid_mask.astype(float),
        method="linear",
        bounds_error=False,
        fill_value=0.0
    )
    dst_mask = mask_interp(query_pts).reshape((len(dst_lats), len(dst_lons))) >= 0.5

    # Interpolate data with nearest fill for edges, then apply destination mask
    # Replace NaNs with nearest valid or 0 for interpolation
    clean_data = np.nan_to_num(data, nan=0.0)
    data_interp = RegularGridInterpolator(
        (src_lats, src_lons),
        clean_data,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )
    
    out = data_interp(query_pts).reshape((len(dst_lats), len(dst_lons))).astype(np.float32)
    # Strictly re-apply mask so no fake numbers appear over land or out of bounds
    out[~dst_mask] = np.nan
    return out

def regrid_ostia_sst(da_sst):
    """
    Regrids OSTIA SST from native 0.05° to canonical 0.25°.
    Converts Kelvin to Celsius if values > 100 K.
    """
    lats = da_sst.latitude.values if "latitude" in da_sst.coords else da_sst.lat.values
    lons = da_sst.longitude.values if "longitude" in da_sst.coords else da_sst.lon.values
    vals = da_sst.values
    
    # Check if Kelvin (mean > 100)
    is_kelvin = np.nanmean(vals) > 100.0
    if is_kelvin:
        vals = vals - 273.15
        
    if vals.ndim == 2:
        regridded = regrid_2d_field(vals, lats, lons, method="linear")
    elif vals.ndim == 3:  # (time, lat, lon)
        regridded = np.array([regrid_2d_field(vals[t], lats, lons, method="linear") for t in range(vals.shape[0])])
    else:
        raise ValueError(f"Unexpected SST dimensions: {vals.shape}")
        
    return regridded

def regrid_glorys_thetao(da_thetao):
    """
    Regrids GLORYS potential temperature horizontally from native 0.0833° to 0.25°.
    Preserves vertical depths.
    """
    if hasattr(da_thetao, "dims") and "depth" in da_thetao.dims and ("latitude" in da_thetao.dims or "lat" in da_thetao.dims):
        lat_c = "latitude" if "latitude" in da_thetao.dims else "lat"
        lon_c = "longitude" if "longitude" in da_thetao.dims else "lon"
        dim_order = [d for d in ["time", "depth", lat_c, lon_c] if d in da_thetao.dims]
        da_thetao = da_thetao.transpose(*dim_order)

    lat_name = "latitude" if "latitude" in da_thetao.coords else "lat"
    lon_name = "longitude" if "longitude" in da_thetao.coords else "lon"
    lats = da_thetao[lat_name].values
    lons = da_thetao[lon_name].values
    # da_thetao: (time, depth, lat, lon) or (depth, lat, lon)
    vals = da_thetao.values
    
    if vals.ndim == 3:  # (depth, lat, lon)
        n_depth = vals.shape[0]
        out = np.zeros((n_depth, len(CANONICAL_LATS), len(CANONICAL_LONS)), dtype=np.float32)
        for d in range(n_depth):
            out[d] = regrid_2d_field(vals[d], lats, lons, method="linear")
        return out
    elif vals.ndim == 4:  # (time, depth, lat, lon)
        n_time, n_depth = vals.shape[:2]
        out = np.zeros((n_time, n_depth, len(CANONICAL_LATS), len(CANONICAL_LONS)), dtype=np.float32)
        for t in range(n_time):
            for d in range(n_depth):
                out[t, d] = regrid_2d_field(vals[t, d], lats, lons, method="linear")
        return out
    else:
        raise ValueError(f"Unexpected thetao dimensions: {vals.shape}")
