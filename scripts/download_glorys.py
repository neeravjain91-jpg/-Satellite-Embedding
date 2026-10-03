"""
scripts/download_glorys.py
Automated acquisition for GLORYS Global Ocean Physics Reanalysis:
- Product: GLOBAL_MULTIYEAR_PHY_001_030
- Dataset ID: cmems_mod_glo_phy_my_0.083deg_P1D-m
- Variable: thetao (potential temperature)
- Domain: 5°N–30°N, 45°E–105°E
- Depths: 0m to 1100m (captures level 36 at 1062.4m to bracket canonical 1000m)
- Automatic size check, monthly chunking, SHA-256 verification, and manifest tracking.
"""

import os
import sys
import copernicusmarine

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.manifest_manager import ManifestManager, retry_with_backoff
from scripts.estimate_sizes import estimate_request_size, check_free_disk, plan_chunk_splits

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLE = "thetao"

def check_copernicus_credentials():
    cred_file = os.path.expanduser("~/.copernicusmarine/.copernicusmarine-credentials")
    has_env = "COPERNICUSMARINE_SERVICE_USERNAME" in os.environ and "COPERNICUSMARINE_SERVICE_PASSWORD" in os.environ
    if not (os.path.exists(cred_file) or has_env):
        raise PermissionError(
            "Copernicus Marine credentials required for GLORYS download.\n"
            "Please configure COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD environment variables,\n"
            "or run 'copernicusmarine login' in terminal."
        )

def download_glorys_chunk(start_date, end_date, bbox=(5.0, 30.0, 45.0, 105.0),
                          depth_range=(0.0, 1100.0), output_dir="data/raw/glorys",
                          overwrite=False):
    """
    Downloads a single temporal chunk of GLORYS thetao via copernicusmarine.subset.
    """
    os.makedirs(output_dir, exist_ok=True)
    check_free_disk()
    check_copernicus_credentials()
    manifest_mgr = ManifestManager()

    lat_min, lat_max, lon_min, lon_max = bbox
    depth_min, depth_max = depth_range
    
    chunk_key = manifest_mgr.get_chunk_key("GLORYS", VARIABLE, start_date, end_date)
    out_filename = f"glorys_thetao_{start_date}_{end_date}.nc"
    out_filepath = os.path.join(output_dir, out_filename)

    if not overwrite and manifest_mgr.is_chunk_complete(chunk_key):
        if os.path.exists(out_filepath):
            try:
                import xarray as _xr
                with _xr.open_dataset(out_filepath) as _ds:
                    if "depth" in _ds and float(_ds.depth.values.max()) >= min(depth_max, 1000.0):
                        print(f"[SKIP] GLORYS chunk {start_date} to {end_date} already complete and depth-verified: {out_filepath}")
                        return out_filepath
                    else:
                        print(f"[REACQUIRE] Existing GLORYS chunk {out_filepath} max depth {float(_ds.depth.values.max()):.1f}m does not reach requested depth {depth_max}m. Redownloading...")
            except Exception:
                pass
        else:
            print(f"[REACQUIRE] Manifest marks chunk complete but file missing: {out_filepath}. Redownloading...")

    # Size estimation
    est = estimate_request_size("glorys", start_date, end_date, bbox=bbox, depth_range=depth_range)
    print(f"[PLAN] GLORYS {start_date} to {end_date}: Est Raw: {est['raw_gb']*1024:.1f} MB, Compressed: {est['compressed_gb']*1024:.1f} MB")

    manifest_mgr.record_chunk(
        dataset="GLORYS",
        dataset_id=DATASET_ID,
        variable=VARIABLE,
        start_datetime=start_date,
        end_datetime=end_date,
        bbox=bbox,
        depth_range=depth_range,
        output_file=out_filepath,
        status="DOWNLOADING"
    )

    def _execute_subset():
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            variables=[VARIABLE],
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            minimum_depth=depth_min,
            maximum_depth=depth_max,
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
            output_directory=output_dir,
            output_filename=out_filename,
            overwrite=True,
            username=os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"),
            password=os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
        )
        return out_filepath

    try:
        saved_path = retry_with_backoff(_execute_subset, max_retries=3, initial_delay=3.0)
        manifest_mgr.update_status(chunk_key, "COMPLETE", output_file=saved_path)
        print(f"[COMPLETE] GLORYS chunk saved to {saved_path}")
        return saved_path
    except Exception as e:
        print(f"[ERROR] GLORYS download failed for {start_date} to {end_date}: {e}")
        manifest_mgr.update_status(chunk_key, "FAILED", error=str(e))
        raise

def download_glorys_period(start_date, end_date, bbox=(5.0, 30.0, 45.0, 105.0),
                           depth_range=(0.0, 1100.0)):
    """
    Downloads period by splitting into safe monthly chunks.
    """
    chunks = plan_chunk_splits("glorys", start_date, end_date)
    print(f"Executing GLORYS download in {len(chunks)} chunks...")
    downloaded_files = []
    for c_start, c_end in chunks:
        f = download_glorys_chunk(c_start, c_end, bbox=bbox, depth_range=depth_range)
        downloaded_files.append(f)
    return downloaded_files

if __name__ == "__main__":
    download_glorys_chunk("2020-01-01", "2020-01-07")
