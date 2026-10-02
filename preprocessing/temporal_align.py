"""
preprocessing/temporal_align.py
Defines canonical time coordinates, UTC consistency, and aggregation rules:
- All datasets aligned to daily UTC dates: YYYY-MM-DDT00:00:00Z.
- OSTIA SST: Daily L4 foundation SST.
- Copernicus SSS: Daily L4 analysis.
- DUACS SSH/SLA: Daily L4 analysis.
- OSCAR Current U, V: Daily L4 analysis.
- CCMP Wind U, V: 4 daily synoptic timestamps (00:00, 06:00, 12:00, 18:00 UTC) -> Daily mean.
- GLORYS thetao: Daily mean potential temperature.

Strict constraint: No operation may use future observations relative to the target prediction date.
"""

import pandas as pd
import numpy as np
import xarray as xr

def to_canonical_daily_date(date_str_or_dt):
    """Normalizes any date or timestamp to UTC daily date at 00:00:00 without tz info."""
    ts = pd.to_datetime(date_str_or_dt, utc=True)
    return ts.tz_localize(None).normalize()

def generate_daily_time_range(start_date, end_date):
    """Generates an array of daily pandas timestamps at 00:00:00 UTC (tz-naive for NetCDF/xarray)."""
    return pd.date_range(start=start_date, end=end_date, freq="D")

def aggregate_ccmp_6hourly_to_daily(ds_ccmp):
    """
    Aggregates CCMP 6-hourly winds (00, 06, 12, 18 UTC) to daily mean.
    Strictly uses only same-day observations (00:00, 06:00, 12:00, 18:00 UTC).
    """
    # Ensure time is UTC
    if "time" not in ds_ccmp.coords:
        raise ValueError("CCMP dataset missing 'time' coordinate")
        
    # Group by calendar date and compute mean across time
    daily_ds = ds_ccmp.resample(time="1D").mean(dim="time")
    return daily_ds

def align_to_target_date(source_da, target_date_utc):
    """
    Extracts or matches the single day corresponding to target_date_utc.
    """
    target_dt = pd.to_datetime(target_date_utc, utc=True).normalize()
    # Find matching time index
    if "time" in source_da.coords:
        source_dates = pd.to_datetime(source_da.time.values, utc=True).normalize()
        matches = np.where(source_dates == target_dt)[0]
        if len(matches) > 0:
            return source_da.isel(time=matches[0])
            
    return source_da
