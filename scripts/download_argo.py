"""
scripts/download_argo.py
Acquires in-situ ARGO float temperature profiles for independent validation:
- Domain: 5°N–30°N, 45°E–105°E
- Depths: 0–1000m
- Source: IFREMER ERDDAP ArgoFloats
- Quality control: Real-time & delayed-mode profiles with QC flag filtering
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.manifest_manager import ManifestManager, retry_with_backoff

ARGO_ERDDAP_URL = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json"

def download_argo_profiles(start_date, end_date, bbox=(5.0, 30.0, 45.0, 105.0),
                           depth_range=(0.0, 1000.0), output_dir="data/raw/argo"):
    """
    Retrieves ARGO float profile data points for the given bounding box and time period.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_mgr = ManifestManager()
    
    lat_min, lat_max, lon_min, lon_max = bbox
    pres_min, pres_max = depth_range
    
    chunk_key = manifest_mgr.get_chunk_key("argo", "TEMP", start_date, end_date)
    out_file = os.path.join(output_dir, f"argo_profiles_{start_date}_{end_date}.csv")
    
    if manifest_mgr.is_chunk_complete(chunk_key):
        print(f"[SKIP] ARGO data for {start_date} to {end_date} already downloaded and validated.")
        return out_file
        
    manifest_mgr.record_chunk(
        dataset="ARGO",
        dataset_id="ArgoFloats",
        variable="TEMP",
        start_datetime=start_date,
        end_datetime=end_date,
        bbox=bbox,
        depth_range=depth_range,
        output_file=out_file,
        status="DOWNLOADING"
    )
    
    # Query parameters
    # Request: platform_number, time, latitude, longitude, pres, temp, temp_qc
    query = (
        f"?platform_number,time,latitude,longitude,pres,temp,temp_qc"
        f"&time>={start_date}T00:00:00Z&time<={end_date}T23:59:59Z"
        f"&latitude>={lat_min}&latitude<={lat_max}"
        f"&longitude>={lon_min}&longitude<={lon_max}"
        f"&pres>={pres_min}&pres<={pres_max}"
    )
    req_url = ARGO_ERDDAP_URL + query
    print(f"Querying ARGO ERDDAP for {start_date} to {end_date}...")
    
    def _fetch():
        resp = requests.get(req_url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    try:
        data = retry_with_backoff(_fetch, max_retries=3)
        col_names = data["table"]["columnNames"]
        rows = data["table"]["rows"]
        
        df = pd.DataFrame(rows, columns=col_names)
        # Rename columns to standard schema
        df = df.rename(columns={
            "platform_number": "argo_id",
            "pres": "depth",
            "temp": "observed_temperature",
            "temp_qc": "quality_flag"
        })
        
        # Filter for good quality flags (1 = good, 2 = probably good)
        df["quality_flag"] = pd.to_numeric(df["quality_flag"], errors="coerce").fillna(1).astype(int)
        df = df[df["quality_flag"].isin([1, 2])]
        
        # Save to CSV
        df.to_csv(out_file, index=False)
        print(f"Downloaded {len(df)} quality-controlled ARGO observation points to {out_file}")
        
        manifest_mgr.update_status(chunk_key, "COMPLETE", output_file=out_file)
        return out_file
    except Exception as e:
        print(f"[ERROR] Failed to fetch ARGO profiles: {e}")
        manifest_mgr.update_status(chunk_key, "FAILED", error=str(e))
        raise

if __name__ == "__main__":
    download_argo_profiles("2020-01-01", "2020-01-07")
