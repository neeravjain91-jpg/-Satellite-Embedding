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
import numpy as np
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

def subset_oscar_dataset(ds, bbox=(5.0, 30.0, 45.0, 105.0)):
    """
    Schema-aware spatial subsetting for OSCAR v2.0 NetCDF datasets.

    Official OSCAR v2.0 schema:
        dimensions:
            latitude = 719
            longitude = 1440
            time = 1
        coordinate variables:
            lat(latitude)
            lon(longitude)
        velocity variables:
            u(time, longitude, latitude)
            v(time, longitude, latitude)

    Subsets using integer indices along actual dimensions 'latitude' and 'longitude'
    based on boolean masks of coordinate variables 'lat' and 'lon'.
    """
    lat_min, lat_max, lon_min, lon_max = bbox

    # 1. Use actual coordinate arrays
    if "lat" in ds.coords or "lat" in ds.variables:
        lat_values = ds["lat"].values
        lat_coord_name = "lat"
    elif "latitude" in ds.coords or "latitude" in ds.variables:
        lat_values = ds["latitude"].values
        lat_coord_name = "latitude"
    else:
        raise KeyError("Neither 'lat' nor 'latitude' coordinate found in OSCAR dataset.")

    if "lon" in ds.coords or "lon" in ds.variables:
        lon_values = ds["lon"].values
        lon_coord_name = "lon"
    elif "longitude" in ds.coords or "longitude" in ds.variables:
        lon_values = ds["longitude"].values
        lon_coord_name = "longitude"
    else:
        raise KeyError("Neither 'lon' nor 'longitude' coordinate found in OSCAR dataset.")

    # 2. Build boolean masks for bounding box
    lat_mask = (lat_values >= lat_min) & (lat_values <= lat_max)
    lon_mask = (lon_values >= lon_min) & (lon_values <= lon_max)

    # 3. Convert masks to integer indices using np.where()
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    if len(lat_idx) == 0:
        raise ValueError(f"No latitude coordinates found in range [{lat_min}, {lat_max}]")
    if len(lon_idx) == 0:
        raise ValueError(f"No longitude coordinates found in range [{lon_min}, {lon_max}]")

    # 4. Identify dimension names
    lat_dim = ds[lat_coord_name].dims[0] if ds[lat_coord_name].dims else ("latitude" if "latitude" in ds.dims else "lat")
    lon_dim = ds[lon_coord_name].dims[0] if ds[lon_coord_name].dims else ("longitude" if "longitude" in ds.dims else "lon")

    # 5. Subset using actual dimensions with isel()
    sub = ds[["u", "v"]].isel({lat_dim: lat_idx, lon_dim: lon_idx})

    # 6. Preserve original lat/lon coordinate variables and time coordinate
    if lat_coord_name not in sub.coords and lat_coord_name in ds:
        sub = sub.assign_coords({lat_coord_name: ds[lat_coord_name].isel({lat_dim: lat_idx})})
    if lon_coord_name not in sub.coords and lon_coord_name in ds:
        sub = sub.assign_coords({lon_coord_name: ds[lon_coord_name].isel({lon_dim: lon_idx})})
    if "time" in ds.coords and "time" not in sub.coords:
        sub = sub.assign_coords({"time": ds["time"]})

    # 7. Explicit assertions after subsetting
    sub_lat_min = float(sub[lat_coord_name].min())
    sub_lat_max = float(sub[lat_coord_name].max())
    sub_lon_min = float(sub[lon_coord_name].min())
    sub_lon_max = float(sub[lon_coord_name].max())

    assert sub_lat_min >= lat_min - 1e-5, f"Assertion failed: min(lat) {sub_lat_min} < {lat_min}"
    assert sub_lat_max <= lat_max + 1e-5, f"Assertion failed: max(lat) {sub_lat_max} > {lat_max}"
    assert sub_lon_min >= lon_min - 1e-5, f"Assertion failed: min(lon) {sub_lon_min} < {lon_min}"
    assert sub_lon_max <= lon_max + 1e-5, f"Assertion failed: max(lon) {sub_lon_max} > {lon_max}"
    assert "u" in sub and "v" in sub, "Assertion failed: 'u' and 'v' must exist in subset"
    assert len(sub[lat_dim]) == 101, f"Assertion failed: Expected 101 latitude points, got {len(sub[lat_dim])}"
    assert len(sub[lon_dim]) == 241, f"Assertion failed: Expected 241 longitude points, got {len(sub[lon_dim])}"
    assert "time" in sub.dims or "time" in sub.coords, "Assertion failed: 'time' dimension/coordinate must exist"
    assert sub["u"].size > 0 and sub["v"].size > 0, "Assertion failed: Output velocity arrays must be non-empty"

    expected_dims = {lat_dim, lon_dim, "time"}
    actual_dims = set(sub.dims)
    assert actual_dims.issubset(expected_dims), f"Assertion failed: unexpected dimensions {actual_dims - expected_dims}"

    return sub

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
        # Subset to bounding box using schema-aware function
        with xr.open_dataset(tmp_raw) as ds:
            sub = subset_oscar_dataset(ds, bbox=bbox)
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
