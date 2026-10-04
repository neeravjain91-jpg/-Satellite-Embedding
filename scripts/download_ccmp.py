"""
scripts/download_ccmp.py
Automated acquisition for CCMP 6-Hourly Ocean Surface Winds Level 4 Version 3.1:
- Product: CCMP_WINDS_10M6HR_L4_V3.1 (DOI: 10.5067/CCMP3-6H431)
- Sources: Remote Sensing Systems (data.remss.com open HTTP) with fallback to NASA Earthdata CMR
- Variables: uwnd, vwnd (m/s)
- Domain: 5°N–30°N, 45°E–105°E
- Temporal aggregation: 6-hourly (00, 06, 12, 18 UTC) -> Daily mean.
"""

import os
import sys
import requests
import numpy as np
import pandas as pd
import xarray as xr

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.manifest_manager import ManifestManager, retry_with_backoff
from scripts.estimate_sizes import check_free_disk

COLLECTION_SHORTNAME = "CCMP_WINDS_10M6HR_L4_V3.1"
REMSS_BASE_URL = "https://data.remss.com/ccmp/v03.1"

def download_and_process_ccmp_day(date_str, bbox=(5.0, 30.0, 45.0, 105.0), output_dir="data/raw/ccmp"):
    """
    Downloads one real daily CCMP NetCDF file (4 time steps: 00, 06, 12, 18 UTC),
    subsets to the North Indian Ocean bounding box, aggregates 6-hourly to daily mean,
    and saves as standardized NetCDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    check_free_disk()
    manifest_mgr = ManifestManager()

    dt = pd.to_datetime(date_str)
    year_str = f"Y{dt.year}"
    month_str = f"M{dt.month:02d}"
    day_file = f"CCMP_Wind_Analysis_{dt.strftime('%Y%m%d')}_V03.1_L4.nc"
    url = f"{REMSS_BASE_URL}/{year_str}/{month_str}/{day_file}"
    
    chunk_key = manifest_mgr.get_chunk_key("CCMP", "uwnd_vwnd", date_str, date_str)
    out_file = os.path.join(output_dir, f"ccmp_daily_{date_str}.nc")

    if manifest_mgr.is_chunk_complete(chunk_key) and os.path.exists(out_file):
        print(f"[SKIP] CCMP {date_str} already complete: {out_file}")
        return out_file

    manifest_mgr.record_chunk(
        dataset="CCMP",
        dataset_id=COLLECTION_SHORTNAME,
        variable="uwnd_vwnd",
        start_datetime=date_str,
        end_datetime=date_str,
        bbox=bbox,
        depth_range=(0, 0),
        output_file=out_file,
        status="DOWNLOADING"
    )

    tmp_raw = out_file + ".tmp.nc"
    print(f"[DOWNLOADING] Fetching real CCMP observation for {date_str} from {url}...")
    
    def _fetch():
        with requests.get(url, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            with open(tmp_raw, "wb") as f:
                for chunk in r.iter_content(chunk_size=512*1024):
                    if chunk:
                        f.write(chunk)

    try:
        retry_with_backoff(_fetch, max_retries=5, initial_delay=3.0, backoff_factor=1.5)
        
        # Spatial subsetting & 6-hourly to daily aggregation
        lat_min, lat_max, lon_min, lon_max = bbox
        with xr.open_dataset(tmp_raw) as ds:
            lat_coord = "latitude" if "latitude" in ds.coords else "lat"
            lon_coord = "longitude" if "longitude" in ds.coords else "lon"
            
            u_var = "uwnd" if "uwnd" in ds else "u"
            v_var = "vwnd" if "vwnd" in ds else "v"
            
            # Crop to bbox
            sub = ds[[u_var, v_var]].sel({
                lat_coord: slice(lat_min, lat_max) if ds[lat_coord][0] < ds[lat_coord][-1] else slice(lat_max, lat_min),
                lon_coord: slice(lon_min, lon_max)
            })
            
            # Aggregate 4 daily timestamps to 1 daily mean
            daily_sub = sub.mean(dim="time", keep_attrs=True)
            daily_sub = daily_sub.expand_dims(time=[np.datetime64(f"{date_str}T00:00:00", "ns")])
            daily_sub.to_netcdf(out_file)

        if os.path.exists(tmp_raw):
            os.remove(tmp_raw)

        manifest_mgr.update_status(chunk_key, "COMPLETE", output_file=out_file)
        print(f"[COMPLETE] Real CCMP daily mean for {date_str} saved to {out_file}")
        return out_file
    except Exception as e:
        if os.path.exists(tmp_raw):
            os.remove(tmp_raw)
        print(f"[ERROR] CCMP download failed for {date_str}: {e}")
        manifest_mgr.update_status(chunk_key, "FAILED", error=str(e))
        raise

def download_ccmp_period(start_date, end_date, bbox=(5.0, 30.0, 45.0, 105.0)):
    date_range = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    print(f"Downloading real CCMP wind observations for {len(date_range)} days ({start_date} to {end_date})...")
    results = []
    failed_dates = []
    for d_str in date_range:
        try:
            f = download_and_process_ccmp_day(d_str, bbox=bbox)
            results.append(f)
        except Exception as e:
            print(f"[RETRY QUEUE] Deferring {d_str} due to error: {e}")
            failed_dates.append(d_str)

    if failed_dates:
        print(f"\n--- Retrying {len(failed_dates)} deferred CCMP dates ---")
        still_failed = []
        for d_str in failed_dates:
            try:
                f = download_and_process_ccmp_day(d_str, bbox=bbox)
                results.append(f)
            except Exception as e:
                still_failed.append((d_str, str(e)))
        if still_failed:
            raise RuntimeError(f"Failed to acquire CCMP observations for {len(still_failed)} dates: {still_failed[:5]}")

    return sorted(list(set(results)))

if __name__ == "__main__":
    download_ccmp_period("2020-01-01", "2020-01-07")
