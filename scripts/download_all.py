"""
scripts/download_all.py
Master automated acquisition orchestrator:
- Supports 7-day pilot execution: python scripts/download_all.py --pilot
- Supports full 1-year execution: python scripts/download_all.py --full-year
- Enforces pre-flight disk safety checks (MIN_FREE_DISK_GB = 20 GB)
- Pre-computes request sizes and splits into monthly chunks
- Tracks every chunk in data/manifests/download_manifest.csv and data/checksums.csv
- Enforces strict Pilot Acquisition Gate: any missing/failed required source blocks certification and exits with code 1.
"""

import os
import sys
import glob
import argparse
import yaml
import pandas as pd

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
from scripts.preflight_and_pilot_gate import run_preflight_checks, is_pilot_acceptance_certified, PILOT_VALIDATION_MARKER

REQUIRED_SOURCES = {
    "argo",
    "glorys",
    "ostia",
    "sss",
    "duacs",
    "oscar",
    "ccmp"
}

def load_config():
    with open("config/data_config.yaml", "r") as f:
        return yaml.safe_load(f)

def check_pilot_completeness(start_date="2020-01-01", end_date="2020-01-07",
                             acquisition_results=None, data_dir=None):
    """
    Explicit final gate: verifies whether every required source is present,
    non-empty, provenance-verified, and covers the complete pilot date range.
    Returns:
        (pilot_certifiable: bool, source_status: dict, missing_or_failed: list)
    """
    if data_dir is None:
        data_dir = os.path.join(repo_root, "data", "raw")

    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, end_date)]
    source_status = {}
    missing_or_failed = []

    # Load manifest & checksum records for provenance verification
    checksums_file = os.path.join(repo_root, "data", "checksums.csv")
    verified_paths = set()
    if os.path.exists(checksums_file):
        try:
            df_c = pd.read_csv(checksums_file)
            for p in df_c["filepath"].dropna():
                norm_p = os.path.normpath(os.path.join(repo_root, p) if not os.path.isabs(p) else p)
                verified_paths.add(norm_p)
        except Exception:
            pass

    def _is_provenance_verified(filepath):
        """Checks if file exists, is non-empty, and is recorded in checksums or manifest."""
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False
        norm_fp = os.path.normpath(os.path.join(repo_root, filepath) if not os.path.isabs(filepath) else filepath)
        if norm_fp in verified_paths:
            return True
        try:
            mgr = ManifestManager()
            for row in mgr.manifest.get("chunks", {}).values():
                out_f = row.get("output_file", "")
                if out_f and os.path.normpath(os.path.join(repo_root, out_f) if not os.path.isabs(out_f) else out_f) == norm_fp and row.get("status") == "COMPLETE":
                    return True
        except Exception:
            pass
        return False

    # 1. ARGO Validation Profiles
    argo_path = os.path.join(data_dir, "argo", f"argo_profiles_{start_date}_{end_date}.csv")
    argo_ok = _is_provenance_verified(argo_path)
    if acquisition_results and str(acquisition_results.get("argo", "")).startswith("FAILED"):
        argo_ok = False
    source_status["argo"] = argo_ok
    if not argo_ok:
        missing_or_failed.append("ARGO")

    # 2. GLORYS Subsurface Temperature (thetao)
    glorys_chunk = os.path.join(data_dir, "glorys", f"glorys_thetao_{start_date}_{end_date}.nc")
    glorys_chunk_ok = _is_provenance_verified(glorys_chunk)
    glorys_daily_ok = len(dates) > 0 and all(
        any(_is_provenance_verified(f) for f in glob.glob(os.path.join(data_dir, "glorys", f"*{d}*.nc")))
        for d in dates
    )
    glorys_ok = glorys_chunk_ok or glorys_daily_ok
    if acquisition_results and str(acquisition_results.get("glorys", "")).startswith("FAILED"):
        glorys_ok = False
    source_status["glorys"] = glorys_ok
    if not glorys_ok:
        missing_or_failed.append("GLORYS")

    # 3. OSTIA Sea Surface Temperature (SST)
    ostia_chunk = os.path.join(data_dir, "ostia", f"ostia_sst_{start_date}_{end_date}.nc")
    ostia_chunk_ok = _is_provenance_verified(ostia_chunk)
    ostia_daily_ok = len(dates) > 0 and all(
        any(_is_provenance_verified(f) for f in glob.glob(os.path.join(data_dir, "ostia", f"*{d}*.nc")))
        for d in dates
    )
    ostia_ok = ostia_chunk_ok or ostia_daily_ok
    if acquisition_results and str(acquisition_results.get("ostia", "")).startswith("FAILED"):
        ostia_ok = False
    source_status["ostia"] = ostia_ok
    if not ostia_ok:
        missing_or_failed.append("OSTIA")

    # 4. Copernicus Multi-Obs Sea Surface Salinity (SSS)
    sss_chunk = os.path.join(data_dir, "sss", f"sss_multi_{start_date}_{end_date}.nc")
    sss_chunk_ok = _is_provenance_verified(sss_chunk)
    sss_daily_ok = len(dates) > 0 and all(
        any(_is_provenance_verified(f) for f in glob.glob(os.path.join(data_dir, "sss", f"*{d}*.nc")))
        for d in dates
    )
    sss_ok = sss_chunk_ok or sss_daily_ok
    if acquisition_results and str(acquisition_results.get("sss", "")).startswith("FAILED"):
        sss_ok = False
    source_status["sss"] = sss_ok
    if not sss_ok:
        missing_or_failed.append("SSS")

    # 5. DUACS Sea Level Anomaly (SSH/SLA)
    duacs_chunk = os.path.join(data_dir, "duacs", f"duacs_sla_{start_date}_{end_date}.nc")
    duacs_chunk_ok = _is_provenance_verified(duacs_chunk)
    duacs_daily_ok = len(dates) > 0 and all(
        any(_is_provenance_verified(f) for f in glob.glob(os.path.join(data_dir, "duacs", f"*{d}*.nc")))
        for d in dates
    )
    duacs_ok = duacs_chunk_ok or duacs_daily_ok
    if acquisition_results and str(acquisition_results.get("duacs", "")).startswith("FAILED"):
        duacs_ok = False
    source_status["duacs"] = duacs_ok
    if not duacs_ok:
        missing_or_failed.append("DUACS")

    # 6. OSCAR Surface Currents (U, V)
    oscar_chunk = os.path.join(data_dir, "oscar", f"oscar_{start_date}_{end_date}.nc")
    oscar_chunk_ok = _is_provenance_verified(oscar_chunk)
    oscar_daily_ok = len(dates) > 0 and all(
        any(_is_provenance_verified(f) for f in glob.glob(os.path.join(data_dir, "oscar", f"*{d}*.nc")))
        for d in dates
    )
    oscar_ok = oscar_chunk_ok or oscar_daily_ok
    if acquisition_results and str(acquisition_results.get("oscar", "")).startswith("FAILED"):
        oscar_ok = False
    source_status["oscar"] = oscar_ok
    if not oscar_ok:
        missing_or_failed.append("OSCAR")

    # 7. CCMP Surface Winds (U, V)
    ccmp_chunk = os.path.join(data_dir, "ccmp", f"ccmp_{start_date}_{end_date}.nc")
    ccmp_chunk_ok = _is_provenance_verified(ccmp_chunk)
    ccmp_daily_ok = len(dates) > 0 and all(
        any(_is_provenance_verified(f) for f in glob.glob(os.path.join(data_dir, "ccmp", f"*{d}*.nc")))
        for d in dates
    )
    ccmp_ok = ccmp_chunk_ok or ccmp_daily_ok
    if acquisition_results and str(acquisition_results.get("ccmp", "")).startswith("FAILED"):
        ccmp_ok = False
    source_status["ccmp"] = ccmp_ok
    if not ccmp_ok:
        missing_or_failed.append("CCMP")

    pilot_certifiable = len(missing_or_failed) == 0
    return pilot_certifiable, source_status, missing_or_failed

def run_download_pipeline(mode="pilot", exit_on_failure=True):
    config = load_config()
    print("=" * 70)
    print(f"STARTING DATA ACQUISITION PIPELINE [MODE: {mode.upper()}]")
    print("=" * 70)

    # 1. Pilot-First Gate Enforcement
    if mode == "full-year" and not is_pilot_acceptance_certified():
        raise PermissionError(
            "[PILOT ACCEPTANCE GATE] Full-year 2020 acquisition is strictly locked until the 7-day authentic pilot has been downloaded, harmonized, and certified through all 3 QA gates.\n"
            "Run 'python scripts/download_all.py --pilot' and 'python scripts/harmonize_and_validate.py' first."
        )

    # 2. Pre-flight verification (Auth, Dry-runs & Endpoints)
    run_preflight_checks()

    # 3. Pre-flight disk space check
    free_gb = check_free_disk(config["size_control"]["min_free_disk_gb"])
    print(f"[PASS] Pre-flight disk check: {free_gb:.2f} GB available (minimum required: {config['size_control']['min_free_disk_gb']} GB)")

    # 4. Determine date bounds
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
    
    # 5. Execute dataset acquisitions (resumable)
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

    # 6. Explicit Final Gate for Pilot Acquisition
    if mode == "pilot":
        pilot_certifiable, source_status, missing_or_failed = check_pilot_completeness(
            start_date, end_date, acquisition_results=results
        )
        print("\n" + "=" * 70)
        print("PILOT ACQUISITION GATE")
        print("=" * 70)
        if not pilot_certifiable:
            # Invalidate any stale certification marker
            if os.path.exists(PILOT_VALIDATION_MARKER):
                try:
                    os.remove(PILOT_VALIDATION_MARKER)
                except Exception:
                    pass
            print("PILOT STATUS: INCOMPLETE")
            print(f"Missing/failed: {', '.join(missing_or_failed)}")
            print("Certification: BLOCKED")
            print("=" * 70)
            if exit_on_failure:
                sys.exit(1)
            return False
        else:
            print("PILOT STATUS: COMPLETE")
            print("All 7 required sources present, non-empty, and verified.")
            print("Certification: READY FOR HARMONIZATION")
            print("=" * 70)
            return True
    else:
        # Full-year check
        failed_sources = [k.upper() for k, v in results.items() if isinstance(v, str) and v.startswith("FAILED")]
        if failed_sources:
            print("\n" + "=" * 70)
            print("FULL-YEAR ACQUISITION GATE")
            print("=" * 70)
            print("ACQUISITION STATUS: INCOMPLETE")
            print(f"Missing/failed: {', '.join(failed_sources)}")
            print("=" * 70)
            if exit_on_failure:
                sys.exit(1)
            return False
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquisition pipeline runner")
    parser.add_argument("--pilot", action="store_true", help="Run 7-day pilot")
    parser.add_argument("--full-year", action="store_true", help="Run full 1-year acquisition")
    args = parser.parse_args()
    
    mode = "full-year" if args.full_year else "pilot"
    run_download_pipeline(mode, exit_on_failure=True)
