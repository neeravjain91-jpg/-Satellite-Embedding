"""
scripts/estimate_sizes.py
Calculates download sizes, grid points, disk safety thresholds, and split planning:
- estimate_request_size()
- check_free_disk()
- plan_chunk_splits()

Strict safety limits:
- MAX_REQUEST_GB = 5.0 GB
- MAX_FILE_GB = 2.0 GB
- MIN_FREE_DISK_GB = 20.0 GB
"""

import os
import shutil
import math
import pandas as pd
import numpy as np

# Safety thresholds (in GB)
MAX_REQUEST_GB = 5.0
MAX_FILE_GB = 2.0
MIN_FREE_DISK_GB = 20.0

# Resolution and variable specifications for volume estimation
DATASET_SPECS = {
    "glorys": {
        "res_deg": 0.083333,
        "depth_levels_in_1000m": 35,
        "bytes_per_val": 4,  # float32
        "compression_ratio": 2.5,
        "default_vars": ["thetao"]
    },
    "ostia": {
        "res_deg": 0.05,
        "depth_levels_in_1000m": 1,
        "bytes_per_val": 4,
        "compression_ratio": 3.0,
        "default_vars": ["analysed_sst"]
    },
    "sss": {
        "res_deg": 0.125,
        "depth_levels_in_1000m": 1,
        "bytes_per_val": 4,
        "compression_ratio": 2.5,
        "default_vars": ["sos"]
    },
    "duacs": {
        "res_deg": 0.25,
        "depth_levels_in_1000m": 1,
        "bytes_per_val": 4,
        "compression_ratio": 2.0,
        "default_vars": ["sla"]
    },
    "oscar": {
        "res_deg": 0.25,
        "depth_levels_in_1000m": 1,
        "bytes_per_val": 4,
        "compression_ratio": 2.0,
        "default_vars": ["u", "v"]
    },
    "ccmp": {
        "res_deg": 0.25,
        "depth_levels_in_1000m": 1,
        "bytes_per_val": 4,
        "compression_ratio": 2.0,
        "steps_per_day": 4,  # 6-hourly
        "default_vars": ["uwnd", "vwnd"]
    }
}

def get_free_disk_gb(path="."):
    """Returns available free disk space on the drive in GB."""
    total, used, free = shutil.disk_usage(os.path.abspath(path))
    return free / (1024 ** 3)

def check_free_disk(min_free_gb=MIN_FREE_DISK_GB, path="."):
    """
    Checks that available disk space strictly exceeds min_free_gb.
    Raises RuntimeError if safety margin is violated.
    """
    free_gb = get_free_disk_gb(path)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"[DISK SAFETY ALERT] Insufficient disk space! "
            f"Free disk is {free_gb:.2f} GB, which is below the mandatory safety threshold of {min_free_gb:.2f} GB. "
            f"Halting operation safely to protect user disk."
        )
    return free_gb

def estimate_request_size(dataset_key, start_date, end_date, bbox=(5.0, 30.0, 45.0, 105.0), depth_range=(0.0, 1000.0), vars_list=None):
    """
    Computes:
    - estimated number of grid points
    - number of depth levels
    - number of time steps
    - variable count
    - estimated raw bytes
    - estimated compressed output
    - available disk
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    spec = DATASET_SPECS.get(dataset_key.lower(), DATASET_SPECS["glorys"])
    res = spec["res_deg"]
    
    n_lats = int(math.ceil((lat_max - lat_min) / res)) + 1
    n_lons = int(math.ceil((lon_max - lon_min) / res)) + 1
    grid_points = n_lats * n_lons
    
    n_depths = spec["depth_levels_in_1000m"] if depth_range[1] > 0 else 1
    
    # Calculate days
    dt_start = pd.to_datetime(start_date)
    dt_end = pd.to_datetime(end_date)
    num_days = (dt_end - dt_start).days + 1
    steps_per_day = spec.get("steps_per_day", 1)
    time_steps = num_days * steps_per_day
    
    variables = vars_list if vars_list is not None else spec["default_vars"]
    var_count = len(variables)
    
    raw_bytes = grid_points * n_depths * time_steps * var_count * spec["bytes_per_val"]
    raw_gb = raw_bytes / (1024 ** 3)
    compressed_gb = raw_gb / spec["compression_ratio"]
    
    free_disk_gb = get_free_disk_gb(".")
    exceeds_max_request = compressed_gb > MAX_REQUEST_GB
    
    return {
        "dataset": dataset_key,
        "grid_points_per_slice": grid_points,
        "n_lats": n_lats,
        "n_lons": n_lons,
        "n_depths": n_depths,
        "time_steps": time_steps,
        "num_days": num_days,
        "var_count": var_count,
        "raw_bytes": raw_bytes,
        "raw_gb": round(raw_gb, 4),
        "compressed_gb": round(compressed_gb, 4),
        "free_disk_gb": round(free_disk_gb, 2),
        "exceeds_limit": exceeds_max_request
    }

def plan_chunk_splits(dataset_key, start_date, end_date, max_chunk_gb=MAX_REQUEST_GB):
    """
    Splits long date ranges into manageable monthly chunks (or shorter if required).
    Preferred split order: month, then shorter temporal chunks.
    Never reduces spatial domain or depth levels.
    """
    dt_start = pd.to_datetime(start_date)
    dt_end = pd.to_datetime(end_date)
    
    # Monthly periods
    periods = []
    current_dt = dt_start
    while current_dt <= dt_end:
        # End of current month or end_date
        month_end = current_dt + pd.offsets.MonthEnd(1)
        chunk_end = min(month_end, dt_end)
        
        # Estimate size for this month
        est = estimate_request_size(dataset_key, current_dt, chunk_end)
        
        # If a single month still exceeds max_chunk_gb, split by 10-day intervals
        if est["compressed_gb"] > max_chunk_gb:
            sub_start = current_dt
            while sub_start <= chunk_end:
                sub_end = min(sub_start + pd.Timedelta(days=9), chunk_end)
                periods.append((sub_start.strftime("%Y-%m-%d"), sub_end.strftime("%Y-%m-%d")))
                sub_start = sub_end + pd.Timedelta(days=1)
        else:
            periods.append((current_dt.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
            
        current_dt = chunk_end + pd.Timedelta(days=1)
        
    return periods

if __name__ == "__main__":
    print(f"Current Available Free Disk: {get_free_disk_gb():.2f} GB")
    check_free_disk()
    
    print("\n=== Size Estimations for 7-Day Pilot (2020-01-01 to 2020-01-07) ===")
    for ds_name in DATASET_SPECS:
        est = estimate_request_size(ds_name, "2020-01-01", "2020-01-07")
        print(f"{ds_name.upper():8s} -> Grid: {est['grid_points_per_slice']} pts | Depths: {est['n_depths']:2d} | Raw: {est['raw_gb']*1024:6.2f} MB | Compressed: {est['compressed_gb']*1024:6.2f} MB")

    print("\n=== Size Estimations for Full 1-Year (2020-01-01 to 2020-12-31) ===")
    total_raw_gb = 0.0
    total_comp_gb = 0.0
    for ds_name in DATASET_SPECS:
        est = estimate_request_size(ds_name, "2020-01-01", "2020-12-31")
        total_raw_gb += est['raw_gb']
        total_comp_gb += est['compressed_gb']
        print(f"{ds_name.upper():8s} -> Raw: {est['raw_gb']:6.2f} GB | Compressed: {est['compressed_gb']:6.2f} GB | Exceeds 5GB limit: {est['exceeds_limit']}")
    print(f"TOTAL 1-Year Raw Data Volume: {total_raw_gb:.2f} GB (Compressed: {total_comp_gb:.2f} GB)")
    
    chunks_glorys = plan_chunk_splits("glorys", "2020-01-01", "2020-12-31")
    print(f"\nGLORYS 1-Year Chunk Plan: {len(chunks_glorys)} monthly chunks.")
