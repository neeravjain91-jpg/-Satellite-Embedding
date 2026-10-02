"""
preprocessing/argo_matchup.py
Independent validation pipeline matching in-situ ARGO / INCOIS float profiles
against the reconstructed / model temperature field.

For every matched observation record:
    - argo_id
    - time
    - latitude
    - longitude
    - depth
    - observed_temperature
    - model_temperature
    - spatial_distance (km)
    - temporal_distance (hours)
    - quality_flag

STRICT RULE: Do not train on ARGO. ARGO is solely for independent evaluation.
"""

import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from preprocessing.canonical_grid import CANONICAL_LATS, CANONICAL_LONS, CANONICAL_DEPTHS

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculates great circle distance in km between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def match_argo_profiles_with_model(df_argo, ds_model_target, max_spatial_dist_km=25.0, max_temporal_dist_hours=24.0):
    """
    df_argo: DataFrame containing columns:
        ['argo_id', 'time', 'latitude', 'longitude', 'depth', 'observed_temperature', 'quality_flag']
    ds_model_target: xarray Dataset containing:
        'thetao' with dims (time, depth, latitude, longitude)
    """
    matched_records = []
    
    # Pre-build spatial-depth interpolators per time step
    model_times = pd.to_datetime(ds_model_target.time.values).normalize()
    
    for idx, row in df_argo.iterrows():
        argo_t = pd.to_datetime(row['time']).tz_localize(None)
        argo_lat = float(row['latitude'])
        argo_lon = float(row['longitude'])
        argo_depth = float(row['depth'])
        obs_temp = float(row['observed_temperature'])
        q_flag = int(row.get('quality_flag', 1))
        argo_id = str(row.get('argo_id', row.get('platform_number', 'UNKNOWN')))
        
        # Spatial bounding check
        if not (CANONICAL_LATS[0] <= argo_lat <= CANONICAL_LATS[-1] and
                CANONICAL_LONS[0] <= argo_lon <= CANONICAL_LONS[-1] and
                CANONICAL_DEPTHS[0] <= argo_depth <= CANONICAL_DEPTHS[-1]):
            continue

        # Find closest model time
        time_diffs_h = np.abs((model_times - argo_t.normalize()).total_seconds()) / 3600.0
        min_t_idx = np.argmin(time_diffs_h)
        t_dist_h = float(time_diffs_h[min_t_idx])
        
        if t_dist_h > max_temporal_dist_hours:
            continue
            
        # Nearest canonical grid point distance
        nearest_lat_idx = np.argmin(np.abs(CANONICAL_LATS - argo_lat))
        nearest_lon_idx = np.argmin(np.abs(CANONICAL_LONS - argo_lon))
        nearest_lat = CANONICAL_LATS[nearest_lat_idx]
        nearest_lon = CANONICAL_LONS[nearest_lon_idx]
        
        spat_dist_km = haversine_distance_km(argo_lat, argo_lon, nearest_lat, nearest_lon)
        if spat_dist_km > max_spatial_dist_km:
            continue
            
        # Interpolate model temperature at (depth, lat, lon)
        try:
            slice_t = ds_model_target["thetao"].isel(time=min_t_idx)
            # 1D depth interpolation at nearest (lat, lon)
            prof_temps = slice_t.isel(latitude=nearest_lat_idx, longitude=nearest_lon_idx).values
            valid_d_mask = ~np.isnan(prof_temps)
            
            if not np.any(valid_d_mask):
                continue
                
            model_t_interp = np.interp(
                argo_depth,
                CANONICAL_DEPTHS[valid_d_mask],
                prof_temps[valid_d_mask],
                left=np.nan,
                right=np.nan
            )
            
            if np.isnan(model_t_interp):
                continue
                
            matched_records.append({
                "argo_id": argo_id,
                "time": str(argo_t),
                "latitude": round(argo_lat, 4),
                "longitude": round(argo_lon, 4),
                "depth": round(argo_depth, 2),
                "observed_temperature": round(obs_temp, 3),
                "model_temperature": round(float(model_t_interp), 3),
                "spatial_distance": round(spat_dist_km, 2),
                "temporal_distance": round(t_dist_h, 2),
                "quality_flag": q_flag
            })
        except Exception as e:
            continue
            
    df_matched = pd.DataFrame(matched_records)
    print(f"Matched {len(df_matched)} ARGO observation points with model grid.")
    return df_matched

def evaluate_argo_matchups(df_matched, out_csv="data/processed/argo_matchup_evaluation.csv"):
    """Computes statistical metrics (RMSE, Bias, MAE, R) and saves evaluation table."""
    if len(df_matched) == 0:
        print("[WARN] No ARGO matchups to evaluate.")
        return {}
        
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_matched.to_csv(out_csv, index=False)
    
    diff = df_matched["model_temperature"] - df_matched["observed_temperature"]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    corr = float(np.corrcoef(df_matched["model_temperature"], df_matched["observed_temperature"])[0, 1])
    
    metrics = {
        "num_matched_points": len(df_matched),
        "rmse_degC": round(rmse, 3),
        "mae_degC": round(mae, 3),
        "bias_degC": round(bias, 3),
        "correlation_r": round(corr, 3)
    }
    
    print("\n--- Independent ARGO Validation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Matchup records saved to: {out_csv}")
    return metrics
