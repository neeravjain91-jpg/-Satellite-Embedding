"""
scripts/download_all.py
Master automated acquisition orchestrator:
- Supports 7-day pilot execution: python scripts/download_all.py --pilot
- Supports full 1-year execution: python scripts/download_all.py --full-year
- Enforces pre-flight disk safety checks (MIN_FREE_DISK_GB = 20 GB)
- Pre-computes request sizes and splits into monthly chunks
- Tracks every chunk in data/manifests/download_manifest.csv and data/checksums.csv
"""

import os
import sys
import argparse
import yaml

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.estimate_sizes import check_free_disk, estimate_request_size, get_free_disk_gb
from scripts.manifest_manager import ManifestManager
from scripts.download_glorys import download_glorys_period
from scripts.download_ostia import download_ostia_period
from scripts.download_sss import download_sss_period
from scripts.download_duacs import download_duacs_period
from scripts.download_oscar import download_oscar_period
from scripts.download_ccmp import download_ccmp_period
from scripts.download_argo import download_argo_profiles

def load_config():
    with open("config/data_config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_download_pipeline(mode="pilot"):
    config = load_config()
    print("=" * 70)
    print(f"STARTING DATA ACQUISITION PIPELINE [MODE: {mode.upper()}]")
    print("=" * 70)

    # 1. Pre-flight disk space check
    free_gb = check_free_disk(config["size_control"]["min_free_disk_gb"])
    print(f"[PASS] Pre-flight disk check: {free_gb:.2f} GB available (minimum required: {config['size_control']['min_free_disk_gb']} GB)")

    # 2. Determine date bounds
    if mode == "pilot":
        start_date = config["scientific_problem"]["temporal"]["pilot_start"]
        end_date = config["scientific_problem"]["temporal"]["pilot_end"]
    elif mode == "full-year":
        start_date = config["scientific_problem"]["temporal"]["year_start"]
        end_date = config["scientific_problem"]["temporal"]["year_end"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Target Period: {start_date} to {end_date}")
    bbox = (
        config["scientific_problem"]["spatial_bounds"]["lat_min"],
        config["scientific_problem"]["spatial_bounds"]["lat_max"],
        config["scientific_problem"]["spatial_bounds"]["lon_min"],
        config["scientific_problem"]["spatial_bounds"]["lon_max"]
    )
    print(f"Spatial Bounding Box: {bbox}")

    results = {}
    
    # 3. Execute dataset acquisitions
    print("\n--- 1/7: ARGO In-Situ Validation Profiles ---")
    try:
        results["argo"] = download_argo_profiles(start_date, end_date, bbox=bbox)
    except Exception as e:
        results["argo"] = f"FAILED: {e}"

    print("\n--- 2/7: GLORYS Subsurface Temperature (thetao) ---")
    try:
        results["glorys"] = download_glorys_period(start_date, end_date, bbox=bbox)
    except Exception as e:
        results["glorys"] = f"FAILED: {e}"

    print("\n--- 3/7: OSTIA Sea Surface Temperature (SST) ---")
    try:
        results["ostia"] = download_ostia_period(start_date, end_date, bbox=bbox)
    except Exception as e:
        results["ostia"] = f"FAILED: {e}"

    print("\n--- 4/7: Copernicus Multi-Obs Sea Surface Salinity (SSS) ---")
    try:
        results["sss"] = download_sss_period(start_date, end_date, bbox=bbox)
    except Exception as e:
        results["sss"] = f"FAILED: {e}"

    print("\n--- 5/7: DUACS Sea Level Anomaly (SSH/SLA) ---")
    try:
        results["duacs"] = download_duacs_period(start_date, end_date, bbox=bbox)
    except Exception as e:
        results["duacs"] = f"FAILED: {e}"

    print("\n--- 6/7: OSCAR Surface Currents (U, V) ---")
    try:
        results["oscar"] = download_oscar_period(start_date, end_date, bbox=bbox)
    except Exception as e:
        results["oscar"] = f"FAILED: {e}"

    print("\n--- 7/7: CCMP Surface Winds (U, V) ---")
    try:
        results["ccmp"] = download_ccmp_period(start_date, end_date, bbox=bbox)
    except Exception as e:
        results["ccmp"] = f"FAILED: {e}"

    print("\n" + "=" * 70)
    print("ACQUISITION RUN SUMMARY")
    print("=" * 70)
    for ds_key, res in results.items():
        status_str = "SUCCESS" if not isinstance(res, str) or not res.startswith("FAILED") else "FAILED"
        print(f"  {ds_key.upper():10s}: {status_str}")
        if status_str == "FAILED":
            print(f"      Reason: {res}")
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquisition pipeline runner")
    parser.add_argument("--pilot", action="store_true", help="Run 7-day pilot")
    parser.add_argument("--full-year", action="store_true", help="Run full 1-year acquisition")
    args = parser.parse_args()
    
    mode = "full-year" if args.full_year else "pilot"
    run_download_pipeline(mode)
