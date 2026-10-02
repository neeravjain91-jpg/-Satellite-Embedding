"""
scripts/download_duacs.py
Automated acquisition for DUACS Sea Surface Height / Sea Level Anomaly:
- Product: SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057
- Dataset ID: c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D
- Variable: sla (and adt)
- Domain: 5°N–30°N, 45°E–105°E
"""

import os
import sys
import copernicusmarine

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.manifest_manager import ManifestManager, retry_with_backoff
from scripts.estimate_sizes import estimate_request_size, check_free_disk, plan_chunk_splits

DATASET_ID = "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D"
VARIABLE = "sla"

def download_duacs_chunk(start_date, end_date, bbox=(5.0, 30.0, 45.0, 105.0), output_dir="data/raw/duacs"):
    os.makedirs(output_dir, exist_ok=True)
    check_free_disk()
    manifest_mgr = ManifestManager()

    lat_min, lat_max, lon_min, lon_max = bbox
    chunk_key = manifest_mgr.get_chunk_key("DUACS", VARIABLE, start_date, end_date)
    out_filename = f"duacs_sla_{start_date}_{end_date}.nc"
    out_filepath = os.path.join(output_dir, out_filename)

    if manifest_mgr.is_chunk_complete(chunk_key):
        print(f"[SKIP] DUACS chunk {start_date} to {end_date} already complete: {out_filepath}")
        return out_filepath

    est = estimate_request_size("duacs", start_date, end_date, bbox=bbox, depth_range=(0, 0))
    print(f"[PLAN] DUACS {start_date} to {end_date}: Est Raw: {est['raw_gb']*1024:.1f} MB, Compressed: {est['compressed_gb']*1024:.1f} MB")

    manifest_mgr.record_chunk(
        dataset="DUACS",
        dataset_id=DATASET_ID,
        variable=VARIABLE,
        start_datetime=start_date,
        end_datetime=end_date,
        bbox=bbox,
        depth_range=(0, 0),
        output_file=out_filepath,
        status="DOWNLOADING"
    )

    def _execute_subset():
        copernicusmarine.subset(
            dataset_id=DATASET_ID,
            variables=[VARIABLE, "adt"],
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
            output_directory=output_dir,
            output_filename=out_filename,
            overwrite=True
        )
        return out_filepath

    try:
        saved_path = retry_with_backoff(_execute_subset, max_retries=3)
        manifest_mgr.update_status(chunk_key, "COMPLETE", output_file=saved_path)
        print(f"[COMPLETE] DUACS chunk saved to {saved_path}")
        return saved_path
    except Exception as e:
        print(f"[ERROR] DUACS download failed: {e}")
        manifest_mgr.update_status(chunk_key, "FAILED", error=str(e))
        raise

def download_duacs_period(start_date, end_date, bbox=(5.0, 30.0, 45.0, 105.0)):
    chunks = plan_chunk_splits("duacs", start_date, end_date)
    files = []
    for c_start, c_end in chunks:
        files.append(download_duacs_chunk(c_start, c_end, bbox=bbox))
    return files

if __name__ == "__main__":
    download_duacs_chunk("2020-01-01", "2020-01-07")
