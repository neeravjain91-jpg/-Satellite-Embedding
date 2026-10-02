"""
scripts/download_ccmp.py
Automated acquisition for CCMP 6-Hourly Ocean Surface Winds:
- Product: CCMP_WINDS_10M6HR_L4_V3.1
- Collection: C2916514952-POCLOUD
- Variables: uwnd, vwnd (or u10, v10, m/s)
- Domain: 5°N–30°N, 45°E–105°E
- Temporal aggregation: 6-hourly (00, 06, 12, 18 UTC) -> Daily mean.
"""

import os
import sys
import requests
import pandas as pd
import xarray as xr

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.manifest_manager import ManifestManager, retry_with_backoff
from scripts.estimate_sizes import check_free_disk
from preprocessing.temporal_align import aggregate_ccmp_6hourly_to_daily

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION_SHORTNAME = "CCMP_WINDS_10M6HR_L4_V3.1"

def query_ccmp_granules(start_date, end_date):
    """Queries NASA CMR for CCMP granule URLs in date range."""
    params = {
        "short_name": COLLECTION_SHORTNAME,
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "page_size": 2000,
        "sort_key": "start_date"
    }
    resp = requests.get(CMR_GRANULES_URL, params=params, timeout=15)
    resp.raise_for_status()
    entries = resp.json().get("feed", {}).get("entry", [])
    
    granules = []
    for e in entries:
        title = e.get("title")
        download_url = None
        for l in e.get("links", []):
            href = l.get("href", "")
            if "podaac-ops-cumulus-protected" in href and href.endswith(".nc"):
                download_url = href
                break
        if download_url:
            granules.append({"title": title, "url": download_url, "time_start": e.get("time_start")})
    return granules

def download_and_process_ccmp(granule_info, bbox=(5.0, 30.0, 45.0, 105.0), output_dir="data/raw/ccmp"):
    """Downloads one CCMP daily file (4 time steps), subsets to bbox, aggregates to daily mean."""
    os.makedirs(output_dir, exist_ok=True)
    check_free_disk()
    manifest_mgr = ManifestManager()

    g_title = granule_info["title"]
    url = granule_info["url"]
    t_start = granule_info["time_start"][:10]
    
    chunk_key = manifest_mgr.get_chunk_key("CCMP", "UWND_VWND", t_start, t_start)
    out_file = os.path.join(output_dir, f"ccmp_daily_{t_start}.nc")

    if manifest_mgr.is_chunk_complete(chunk_key):
        print(f"[SKIP] CCMP {t_start} already downloaded and processed.")
        return out_file

    manifest_mgr.record_chunk(
        dataset="CCMP",
        dataset_id=COLLECTION_SHORTNAME,
        variable="uwnd,vwnd",
        start_datetime=t_start,
        end_datetime=t_start,
        bbox=bbox,
        depth_range=(0, 0),
        output_file=out_file,
        status="DOWNLOADING"
    )

    session = requests.Session()
    edl_token = os.environ.get("EARTHDATA_TOKEN")
    if edl_token:
        session.headers.update({"Authorization": f"Bearer {edl_token}"})

    tmp_raw = out_file + ".tmp.nc"
    def _fetch():
        with session.get(url, stream=True, timeout=30) as r:
            if r.status_code == 401 or "login.earthdata.nasa.gov" in r.url:
                raise PermissionError("NASA Earthdata authentication required for CCMP direct download.")
            r.raise_for_status()
            with open(tmp_raw, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)

    try:
        retry_with_backoff(_fetch, max_retries=3)
        # Subset to bbox and aggregate 6-hourly to daily
        lat_min, lat_max, lon_min, lon_max = bbox
        with xr.open_dataset(tmp_raw) as ds:
            lat_coord = "latitude" if "latitude" in ds.coords else "lat"
            lon_coord = "longitude" if "longitude" in ds.coords else "lon"
            
            # Identify wind variables (uwnd, vwnd)
            u_var = "uwnd" if "uwnd" in ds else "u"
            v_var = "vwnd" if "vwnd" in ds else "v"
            
            sub = ds[[u_var, v_var]].sel({
                lat_coord: slice(lat_min, lat_max) if ds[lat_coord][0] < ds[lat_coord][-1] else slice(lat_max, lat_min),
                lon_coord: slice(lon_min, lon_max)
            })
            
            # Aggregate 4 time steps to single daily mean
            daily_sub = sub.mean(dim="time", keep_attrs=True)
            daily_sub = daily_sub.expand_dims(time=[pd.to_datetime(t_start, utc=True)])
            daily_sub.to_netcdf(out_file)

        if os.path.exists(tmp_raw):
            os.remove(tmp_raw)

        manifest_mgr.update_status(chunk_key, "COMPLETE", output_file=out_file)
        print(f"[COMPLETE] CCMP daily mean for {t_start} saved to {out_file}")
        return out_file
    except Exception as e:
        if os.path.exists(tmp_raw):
            os.remove(tmp_raw)
        print(f"[ERROR] CCMP download failed for {t_start}: {e}")
        manifest_mgr.update_status(chunk_key, "FAILED", error=str(e))
        raise

def download_ccmp_period(start_date, end_date, bbox=(5.0, 30.0, 45.0, 105.0)):
    print(f"Querying CCMP granules for {start_date} to {end_date}...")
    granules = query_ccmp_granules(start_date, end_date)
    print(f"Found {len(granules)} CCMP granules.")
    results = []
    for g in granules:
        f = download_and_process_ccmp(g, bbox=bbox)
        results.append(f)
    return results

if __name__ == "__main__":
    download_ccmp_period("2020-01-01", "2020-01-07")
