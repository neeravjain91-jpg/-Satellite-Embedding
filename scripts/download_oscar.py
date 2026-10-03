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
import glob
import shutil
import subprocess
import netrc
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

def ensure_earthdata_netrc():
    """
    Ensures that a valid .netrc credential configuration exists in the user's home directory.
    Supports Windows robustly:
    - Checks for both ~/.netrc and ~/_netrc.
    - If ~/_netrc exists but ~/.netrc does not, copies ~/_netrc to ~/.netrc so Python's
      netrc module can parse it without issue.
    - If neither exists, checks for EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment
      variables. If present, creates a compliant ~/.netrc entry for urs.earthdata.nasa.gov.
    - Validates that ~/.netrc exists and contains a non-empty entry for urs.earthdata.nasa.gov.
    Returns:
        (is_configured: bool, message: str)
    """
    home = os.path.expanduser("~")
    dot_netrc = os.path.join(home, ".netrc")
    underscore_netrc = os.path.join(home, "_netrc")

    # If _netrc exists on Windows but .netrc doesn't, sync _netrc to .netrc
    if os.path.exists(underscore_netrc) and not os.path.exists(dot_netrc):
        try:
            with open(underscore_netrc, "r", encoding="utf-8") as src, open(dot_netrc, "w", encoding="utf-8") as dst:
                dst.write(src.read())
        except Exception as e:
            return False, f"Failed to synchronize ~/_netrc to ~/.netrc: {e}"

    # If .netrc does not exist, check environment variables
    if not os.path.exists(dot_netrc):
        user = os.environ.get("EARTHDATA_USERNAME")
        pwd = os.environ.get("EARTHDATA_PASSWORD")
        if user and pwd:
            try:
                with open(dot_netrc, "w", encoding="utf-8") as f:
                    f.write(f"machine urs.earthdata.nasa.gov login {user} password {pwd}\n")
            except Exception as e:
                return False, f"Failed to write ~/.netrc from environment variables: {e}"

    if not os.path.exists(dot_netrc):
        return False, (
            "Earthdata credentials file (~/.netrc or ~/_netrc) not found in user home directory, "
            "and EARTHDATA_USERNAME/EARTHDATA_PASSWORD environment variables are not set. "
            "Please create ~/.netrc containing: 'machine urs.earthdata.nasa.gov login <user> password <pwd>'."
        )

    try:
        n = netrc.netrc(dot_netrc)
        auth = n.authenticators("urs.earthdata.nasa.gov")
        if not auth or not auth[0] or not auth[2]:
            return False, f"~/.netrc exists at {dot_netrc} but lacks credentials for 'urs.earthdata.nasa.gov'."
        return True, f"Earthdata credentials found in {dot_netrc}"
    except Exception as e:
        return False, f"Failed to parse credentials from {dot_netrc}: {e}"

def build_podaac_downloader_cmd(collection, output_dir, start_date, end_date, granule_name=None, dry_run=False):
    """
    Builds the official podaac-data-downloader command arguments.
    Uses 'podaac-data-downloader' CLI or falls back to 'python -m subscriber.podaac_data_downloader'.
    """
    exe = shutil.which("podaac-data-downloader")
    base_cmd = [exe] if exe else [sys.executable, "-m", "subscriber.podaac_data_downloader"]
    cmd = base_cmd + [
        "-c", collection,
        "-d", output_dir,
        "-sd", f"{start_date}T00:00:00Z",
        "-ed", f"{end_date}T23:59:59Z",
        "-e", ".nc",
        "--verbose"
    ]
    if granule_name:
        cmd.extend(["-gr", granule_name])
    if dry_run:
        cmd.append("--dry-run")
    return cmd

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

def download_and_subset_oscar(granule_info, bbox=(5.0, 30.0, 45.0, 105.0), output_dir="data/raw/oscar", manifest_mgr=None):
    """
    Acquires one OSCAR granule using the official PO.DAAC downloader (podaac-data-downloader),
    subsets to the North Indian Ocean bounding box using schema-aware logic, and records provenance.
    """
    os.makedirs(output_dir, exist_ok=True)
    check_free_disk()
    if manifest_mgr is None:
        manifest_mgr = ManifestManager()

    if isinstance(granule_info, dict):
        t_start = granule_info["time_start"][:10]
        g_title = granule_info.get("title", f"oscar_currents_final_{t_start.replace('-', '')}")
    else:
        t_start = str(granule_info)[:10]
        g_title = f"oscar_currents_final_{t_start.replace('-', '')}"

    granule_date = t_start.replace("-", "")
    granule_pattern = f"oscar_currents_final_{granule_date}*"

    chunk_key = manifest_mgr.get_chunk_key("OSCAR", "u_v", t_start, t_start)
    out_file = os.path.join(output_dir, f"oscar_{t_start}.nc")

    # 1. Resumable check: skip if already verified COMPLETE
    if manifest_mgr.is_chunk_complete(chunk_key) and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        print(f"[SKIP] OSCAR {t_start} already downloaded and validated.")
        return out_file

    manifest_mgr.record_chunk(
        dataset="OSCAR",
        dataset_id=COLLECTION_SHORTNAME,
        variable="u_v",
        start_datetime=t_start,
        end_datetime=t_start,
        bbox=bbox,
        depth_range=(0, 0),
        output_file=out_file,
        status="DOWNLOADING"
    )

    # 2. Authentication check
    auth_ok, auth_msg = ensure_earthdata_netrc()
    if not auth_ok:
        manifest_mgr.update_status(chunk_key, "FAILED", error=auth_msg)
        raise PermissionError(f"NASA Earthdata authentication required for OSCAR download:\n{auth_msg}")

    # 3. Download full granule using official PO.DAAC downloader into isolated staging dir
    staging_dir = os.path.join(output_dir, f"_staging_{t_start}")
    os.makedirs(staging_dir, exist_ok=True)

    cmd = build_podaac_downloader_cmd(
        collection=COLLECTION_SHORTNAME,
        output_dir=staging_dir,
        start_date=t_start,
        end_date=t_start,
        granule_name=granule_pattern
    )

    try:
        print(f"[PO.DAAC DOWNLOADER] Executing: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        combined_output = (proc.stdout or "") + " " + (proc.stderr or "")

        if proc.returncode != 0:
            if "netrc" in combined_output.lower() or "401" in combined_output or "unauthorized" in combined_output.lower():
                raise PermissionError(f"PO.DAAC downloader authentication error:\n{combined_output.strip()}")
            raise RuntimeError(f"PO.DAAC downloader failed (exit {proc.returncode}):\n{combined_output.strip()}")

        # Locate downloaded granule
        downloaded_nc = [
            f for f in glob.glob(os.path.join(staging_dir, "**", "*.nc"), recursive=True)
            if not f.endswith(".tmp.nc")
        ]

        if not downloaded_nc:
            raise FileNotFoundError(
                f"PO.DAAC downloader completed successfully but no .nc granule was found in {staging_dir}.\n"
                f"Output: {combined_output.strip()}"
            )

        raw_granule = downloaded_nc[0]

        # 4. Schema-aware spatial subsetting
        with xr.open_dataset(raw_granule) as ds:
            sub = subset_oscar_dataset(ds, bbox=bbox)
            sub.to_netcdf(out_file)

        # 5. Cleanup staging directory
        shutil.rmtree(staging_dir, ignore_errors=True)

        # 6. Verify non-empty and update manifest
        if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
            raise RuntimeError(f"Subsetted OSCAR file was not written properly: {out_file}")

        manifest_mgr.update_status(chunk_key, "COMPLETE", output_file=out_file)
        print(f"[COMPLETE] OSCAR granule {t_start} saved to {out_file}")
        return out_file

    except Exception as e:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        if os.path.exists(out_file) and os.path.getsize(out_file) == 0:
            try:
                os.remove(out_file)
            except Exception:
                pass
        print(f"[ERROR] OSCAR acquisition failed for {t_start}: {e}")
        manifest_mgr.update_status(chunk_key, "FAILED", error=str(e))
        raise

def download_oscar_period(start_date, end_date, bbox=(5.0, 30.0, 45.0, 105.0), output_dir="data/raw/oscar", manifest_mgr=None):
    """
    Downloads and subsets OSCAR granules for a given date range.
    Uses monthly batching with podaac-data-downloader for multi-day periods,
    and falls back to per-granule acquisition if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    check_free_disk()
    if manifest_mgr is None:
        manifest_mgr = ManifestManager()

    dt_start = pd.to_datetime(start_date)
    dt_end = pd.to_datetime(end_date)
    all_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, end_date)]

    # If single day, call download_and_subset_oscar directly
    if dt_start == dt_end:
        return [download_and_subset_oscar(start_date, bbox=bbox, output_dir=output_dir, manifest_mgr=manifest_mgr)]

    print(f"Executing OSCAR acquisition for {start_date} to {end_date} ({len(all_dates)} days)...")
    results = []

    # Process in monthly chunks
    curr = dt_start
    while curr <= dt_end:
        m_end = min(curr + pd.offsets.MonthEnd(1), dt_end)
        m_start_str = curr.strftime("%Y-%m-%d")
        m_end_str = m_end.strftime("%Y-%m-%d")
        m_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(m_start_str, m_end_str)]

        # Check which dates in this month are already complete
        missing_dates = []
        for d in m_dates:
            chk_key = manifest_mgr.get_chunk_key("OSCAR", "u_v", d, d)
            out_f = os.path.join(output_dir, f"oscar_{d}.nc")
            if manifest_mgr.is_chunk_complete(chk_key) and os.path.exists(out_f) and os.path.getsize(out_f) > 0:
                results.append(out_f)
            else:
                missing_dates.append(d)

        if not missing_dates:
            print(f"[SKIP] OSCAR month {m_start_str} to {m_end_str} already complete ({len(m_dates)} files).")
            curr = m_end + pd.Timedelta(days=1)
            continue

        print(f"[OSCAR BATCH] Downloading {len(missing_dates)} missing granules for {m_start_str} to {m_end_str}...")
        staging_dir = os.path.join(output_dir, f"_staging_{m_start_str}_{m_end_str}")
        os.makedirs(staging_dir, exist_ok=True)

        cmd = build_podaac_downloader_cmd(
            collection=COLLECTION_SHORTNAME,
            output_dir=staging_dir,
            start_date=m_start_str,
            end_date=m_end_str
        )

        try:
            print(f"[PO.DAAC BATCH] Executing: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            combined_output = (proc.stdout or "") + " " + (proc.stderr or "")

            if proc.returncode != 0:
                if "netrc" in combined_output.lower() or "401" in combined_output:
                    raise PermissionError(f"PO.DAAC downloader authentication error:\n{combined_output.strip()}")
                print(f"[WARN] Batch download returned non-zero code {proc.returncode}. Output: {combined_output.strip()[:300]}")

            downloaded_nc = [
                f for f in glob.glob(os.path.join(staging_dir, "**", "*.nc"), recursive=True)
                if not f.endswith(".tmp.nc")
            ]

            for raw_file in downloaded_nc:
                bname = os.path.basename(raw_file)
                # Format: oscar_currents_final_YYYYMMDD.nc
                date_part = bname.replace("oscar_currents_final_", "").replace(".nc", "").strip()
                if len(date_part) == 8 and date_part.isdigit():
                    f_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                else:
                    continue

                out_f = os.path.join(output_dir, f"oscar_{f_date}.nc")
                chk_key = manifest_mgr.get_chunk_key("OSCAR", "u_v", f_date, f_date)

                if f_date in missing_dates:
                    try:
                        with xr.open_dataset(raw_file) as ds:
                            sub = subset_oscar_dataset(ds, bbox=bbox)
                            sub.to_netcdf(out_f)

                        if os.path.exists(out_f) and os.path.getsize(out_f) > 0:
                            manifest_mgr.update_status(chk_key, "COMPLETE", output_file=out_f)
                            print(f"[COMPLETE] OSCAR granule {f_date} saved to {out_f}")
                            results.append(out_f)
                            missing_dates.remove(f_date)
                    except Exception as e:
                        print(f"[ERROR] Failed to subset {raw_file}: {e}")

            shutil.rmtree(staging_dir, ignore_errors=True)

            # Fallback for any dates still missing
            for d in list(missing_dates):
                f = download_and_subset_oscar(d, bbox=bbox, output_dir=output_dir, manifest_mgr=manifest_mgr)
                results.append(f)

        except Exception as e:
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)
            print(f"[ERROR] Batch download failed for {m_start_str} to {m_end_str}: {e}. Retrying daily fallback...")
            for d in missing_dates:
                f = download_and_subset_oscar(d, bbox=bbox, output_dir=output_dir, manifest_mgr=manifest_mgr)
                results.append(f)

        curr = m_end + pd.Timedelta(days=1)

    return sorted(list(set(results)))

if __name__ == "__main__":
    download_oscar_period("2020-01-01", "2020-01-07")

