"""
preprocessing/argo_matchup.py
ARGO–GLORYS Reference Consistency Assessment pipeline matching in-situ ARGO / INCOIS float profiles
against the GLORYS reference / model grid.

IMPORTANT SCIENTIFIC CONSTRAINT:
This assessment evaluates reference consistency between in-situ ARGO observations
and the GLORYS ocean reanalysis on the canonical grid. It does NOT constitute
validation of an ML model until an ML model is actively producing predictions.

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

STRICT RULE: Do not train on ARGO. ARGO is reserved for reference consistency assessment.
Note: Because operational ARGO is assimilated into GLORYS, true observational independence
requires an evaluation set whose relationship to GLORYS assimilation is explicitly established.
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
    if len(df_argo) == 0:
        return pd.DataFrame()

    model_times = pd.to_datetime(ds_model_target.time.values).normalize()
    thetao_vals = ds_model_target["thetao"].values  # (time, depth, lat, lon)
    
    argo_lats = df_argo['latitude'].to_numpy(dtype=np.float64)
    argo_lons = df_argo['longitude'].to_numpy(dtype=np.float64)
    argo_depths = df_argo['depth'].to_numpy(dtype=np.float64)
    obs_temps = df_argo['observed_temperature'].to_numpy(dtype=np.float64)
    
    id_col = 'argo_id' if 'argo_id' in df_argo.columns else ('platform_number' if 'platform_number' in df_argo.columns else None)
    if id_col:
        argo_ids = df_argo[id_col].astype(str).to_numpy()
    else:
        argo_ids = np.array(['UNKNOWN'] * len(df_argo))
        
    q_flags = df_argo['quality_flag'].to_numpy(dtype=np.int32) if 'quality_flag' in df_argo.columns else np.ones(len(df_argo), dtype=np.int32)
    raw_times_series = pd.to_datetime(df_argo['time'], utc=True).dt.tz_localize(None)

    # 1. Spatial & depth bounding check
    valid_box = (
        (argo_lats >= CANONICAL_LATS[0]) & (argo_lats <= CANONICAL_LATS[-1]) &
        (argo_lons >= CANONICAL_LONS[0]) & (argo_lons <= CANONICAL_LONS[-1]) &
        (argo_depths >= CANONICAL_DEPTHS[0]) & (argo_depths <= CANONICAL_DEPTHS[-1])
    )
    if not np.any(valid_box):
        return pd.DataFrame()

    # 2. Nearest grid indices and spatial distance
    nearest_lat_idx = np.clip(np.round((argo_lats - CANONICAL_LATS[0]) / 0.25).astype(np.int32), 0, len(CANONICAL_LATS) - 1)
    nearest_lon_idx = np.clip(np.round((argo_lons - CANONICAL_LONS[0]) / 0.25).astype(np.int32), 0, len(CANONICAL_LONS) - 1)
    nearest_lats = CANONICAL_LATS[nearest_lat_idx]
    nearest_lons = CANONICAL_LONS[nearest_lon_idx]
    spat_dist_km = haversine_distance_km(argo_lats, argo_lons, nearest_lats, nearest_lons)
    valid_spat = spat_dist_km <= max_spatial_dist_km

    # 3. Time matching
    norm_times = raw_times_series.dt.floor('D')
    start_model_t = model_times[0]
    time_diff_days = (norm_times - start_model_t).dt.days.to_numpy()
    valid_t_range = (time_diff_days >= 0) & (time_diff_days < len(model_times))
    
    # Combined pre-filter
    valid_candidates = valid_box & valid_spat & valid_t_range
    if not np.any(valid_candidates):
        return pd.DataFrame()

    # Compute temporal distance in hours for valid candidates
    cand_indices = np.where(valid_candidates)[0]
    cand_t_idx = time_diff_days[cand_indices]
    cand_raw_dt = raw_times_series.iloc[cand_indices].to_numpy()
    cand_model_dt = model_times[cand_t_idx].to_numpy()
    cand_t_dist_h = np.abs((cand_raw_dt - cand_model_dt) / np.timedelta64(1, 'h'))
    
    valid_temporal = cand_t_dist_h <= max_temporal_dist_hours
    final_indices = cand_indices[valid_temporal]
    if len(final_indices) == 0:
        return pd.DataFrame()

    # Extract filtered arrays
    f_argo_id = argo_ids[final_indices]
    f_raw_time = raw_times_series.iloc[final_indices].dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy()
    f_lats = argo_lats[final_indices]
    f_lons = argo_lons[final_indices]
    f_depths = argo_depths[final_indices]
    f_obs_temp = obs_temps[final_indices]
    f_q_flags = q_flags[final_indices]
    f_spat_dist = spat_dist_km[final_indices]
    f_t_dist = cand_t_dist_h[valid_temporal]
    
    f_t_idx = time_diff_days[final_indices]
    f_lat_idx = nearest_lat_idx[final_indices]
    f_lon_idx = nearest_lon_idx[final_indices]

    # Vectorized 1D Depth Interpolation across CANONICAL_DEPTHS:
    z_idx = np.searchsorted(CANONICAL_DEPTHS, f_depths)
    z_idx = np.clip(z_idx, 1, len(CANONICAL_DEPTHS) - 1)
    z0 = CANONICAL_DEPTHS[z_idx - 1]
    z1 = CANONICAL_DEPTHS[z_idx]
    denom = np.where((z1 - z0) == 0, 1.0, (z1 - z0))
    w = (f_depths - z0) / denom

    # Gather model thetao values
    v0 = thetao_vals[f_t_idx, z_idx - 1, f_lat_idx, f_lon_idx]
    v1 = thetao_vals[f_t_idx, z_idx, f_lat_idx, f_lon_idx]
    model_t = v0 + w * (v1 - v0)

    # Handle exact match at z=0 (depth 0)
    exact_zero = (f_depths == CANONICAL_DEPTHS[0])
    model_t[exact_zero] = thetao_vals[f_t_idx[exact_zero], 0, f_lat_idx[exact_zero], f_lon_idx[exact_zero]]

    # Filter out NaNs (e.g. seafloor bathymetry cutoffs)
    finite_mask = ~np.isnan(model_t) & ~np.isnan(f_obs_temp)
    if not np.any(finite_mask):
        return pd.DataFrame()

    matched_records = {
        "argo_id": f_argo_id[finite_mask],
        "time": f_raw_time[finite_mask],
        "latitude": np.round(f_lats[finite_mask], 4),
        "longitude": np.round(f_lons[finite_mask], 4),
        "depth": np.round(f_depths[finite_mask], 2),
        "observed_temperature": np.round(f_obs_temp[finite_mask], 3),
        "model_temperature": np.round(model_t[finite_mask], 3),
        "spatial_distance": np.round(f_spat_dist[finite_mask], 2),
        "temporal_distance": np.round(f_t_dist[finite_mask], 2),
        "quality_flag": f_q_flags[finite_mask]
    }
    df_matched = pd.DataFrame(matched_records)
    print(f"Matched {len(df_matched):,} ARGO observation points with model grid.")
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
    
    print("\n--- ARGO–GLORYS Reference Consistency Assessment ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Assessment records saved to: {out_csv}")
    return metrics
