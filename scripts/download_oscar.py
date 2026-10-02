"""
scripts/download_oscar.py
Automated acquisition for OSCAR Surface Currents:
- Product: OSCAR_L4_OC_FINAL_V2.0
- Collection: C2098858642-POCLOUD
- Variables: u, v (surface current velocities, m/s)
- Domain: 5°N–30°N, 45°E–105°E
- Automatically subsets to study region and updates manifest and checksums.
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

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION_SHORTNAME = "OSCAR_L4_OC_FINAL_V2.0"

def query_oscar_granules(start_date, end_date):
    """Queries NASA CMR for OSCAR granule URLs in date range."""
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

def download_and_subset_oscar(granule_info, bbox=(5.0, 30.0, 45.0, 105.0), output_dir="data/raw/oscar"):
    """Downloads one OSCAR granule, subsets to North Indian Ocean bbox, and saves."""
    os.makedirs(output_dir, exist_ok=True)
    check_free_disk()
    manifest_mgr = ManifestManager()

    g_title = granule_info["title"]
    url = granule_info["url"]
    t_start = granule_info["time_start"][:10]
    
    chunk_key = manifest_mgr.get_chunk_key("OSCAR", "UV", t_start, t_start)
    out_file = os.path.join(output_dir, f"oscar_{t_start}.nc")

    if manifest_mgr.is_chunk_complete(chunk_key):
        print(f"[SKIP] OSCAR {t_start} already downloaded and validated.")
        return out_file

    manifest_mgr.record_chunk(
        dataset="OSCAR",
        dataset_id=COLLECTION_SHORTNAME,
        variable="u,v",
        start_datetime=t_start,
        end_datetime=t_start,
        bbox=bbox,
        depth_range=(0, 0),
        output_file=out_file,
        status="DOWNLOADING"
    )

    # Use NASA Earthdata session
    session = requests.Session()
    edl_token = os.environ.get("EARTHDATA_TOKEN")
    edl_user = os.environ.get("EARTHDATA_USERNAME")
    edl_pass = os.environ.get("EARTHDATA_PASSWORD")
    if edl_token:
        session.headers.update({"Authorization": f"Bearer {edl_token}"})
    elif edl_user and edl_pass:
        session.auth = (edl_user, edl_pass)

    tmp_raw = out_file + ".tmp.nc"
    def _fetch():
        with session.get(url, stream=True, timeout=30) as r:
            if r.status_code == 401 or "login.earthdata.nasa.gov" in r.url:
                raise PermissionError(
                    "NASA Earthdata authentication required for OSCAR direct download.\n"
                    "Please set EARTHDATA_TOKEN or EARTHDATA_USERNAME/EARTHDATA_PASSWORD environment variables."
                )
            r.raise_for_status()
            with open(tmp_raw, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)

    try:
        retry_with_backoff(_fetch, max_retries=3)
        # Subset to bounding box
        lat_min, lat_max, lon_min, lon_max = bbox
        with xr.open_dataset(tmp_raw) as ds:
            # Harmonize coordinate names (lat/latitude, lon/longitude)
            lat_coord = "latitude" if "latitude" in ds.coords else "lat"
            lon_coord = "longitude" if "longitude" in ds.coords else "lon"
            
            # Select bounding box
            sub = ds[["u", "v"]].sel({
                lat_coord: slice(lat_min, lat_max) if ds[lat_coord][0] < ds[lat_coord][-1] else slice(lat_max, lat_min),
                lon_coord: slice(lon_min, lon_max)
            })
            sub.to_netcdf(out_file)
            
        if os.path.exists(tmp_raw):
            os.remove(tmp_raw)
            
        manifest_mgr.update_status(chunk_key, "COMPLETE", output_file=out_file)
        print(f"[COMPLETE] OSCAR granule {t_start} saved to {out_file}")
        return out_file
    except Exception as e:
        if os.path.exists(tmp_raw):
            os.remove(tmp_raw)
        print(f"[ERROR] OSCAR download failed for {t_start}: {e}")
        manifest_mgr.update_status(chunk_key, "FAILED", error=str(e))
        raise

def download_oscar_period(start_date, end_date, bbox=(5.0, 30.0, 45.0, 105.0)):
    print(f"Querying OSCAR granules for {start_date} to {end_date}...")
    granules = query_oscar_granules(start_date, end_date)
    print(f"Found {len(granules)} OSCAR granules.")
    results = []
    for g in granules:
        f = download_and_subset_oscar(g, bbox=bbox)
        results.append(f)
    return results

if __name__ == "__main__":
    download_oscar_period("2020-01-01", "2020-01-07")
