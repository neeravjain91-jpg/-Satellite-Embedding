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
from scripts.manifest_manager import ManifestManager, compute_sha256
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

def is_provenance_verified(filepath, repo_root_dir=None, checksums_file=None, manifest_mgr=None,
                           checksums_map=None, manifest_map=None, hash_cache=None):
    """
    Validates provenance integrity for a given dataset file:
    - Verifies file exists on disk and has non-zero size.
    - Recomputes the SHA-256 hash directly from file bytes.
    - Locates the recorded SHA-256 in data/checksums.csv and/or download_manifest.json.
    - Requires manifest status == 'COMPLETE' when a manifest record exists.
    - Returns True only when the recomputed SHA-256 matches the recorded SHA-256 exactly.
    - Returns False on checksum mismatch, non-COMPLETE manifest status, missing records, or missing files.
    """
    if not filepath:
        return False

    root = repo_root_dir if repo_root_dir is not None else repo_root
    abs_fp = os.path.abspath(os.path.join(root, filepath) if not os.path.isabs(filepath) else filepath)

    if not os.path.exists(abs_fp) or os.path.getsize(abs_fp) == 0:
        return False

    norm_fp = os.path.normpath(abs_fp).replace("\\", "/").lower()
    rel_fp = os.path.relpath(abs_fp, root).replace("\\", "/").lower()

    if hash_cache is not None and norm_fp in hash_cache:
        actual_hash = hash_cache[norm_fp]
    else:
        actual_hash = compute_sha256(abs_fp)
        if not actual_hash:
            return False
        actual_hash = actual_hash.strip().lower()
        if hash_cache is not None:
            hash_cache[norm_fp] = actual_hash

    if manifest_map is None:
        manifest_map = {}
        try:
            mgr = manifest_mgr if manifest_mgr is not None else ManifestManager()
            chunks = mgr.manifest.get("chunks", mgr.manifest) if isinstance(mgr.manifest.get("chunks"), dict) else mgr.manifest
            for row in chunks.values():
                if isinstance(row, dict):
                    out_f = row.get("output_file", "")
                    if out_f:
                        row_abs = os.path.abspath(os.path.join(root, out_f) if not os.path.isabs(out_f) else out_f)
                        manifest_map[os.path.normpath(row_abs).replace("\\", "/").lower()] = row
                        manifest_map[str(out_f).replace("\\", "/").lower()] = row
        except Exception:
            pass

    if checksums_map is None:
        checksums_map = {}
        ck_path = checksums_file if checksums_file is not None else os.path.join(root, "data", "checksums.csv")
        if os.path.exists(ck_path):
            try:
                df_c = pd.read_csv(ck_path)
                for _, r in df_c.iterrows():
                    p = r.get("filepath")
                    sha = r.get("sha256")
                    if pd.notna(p) and pd.notna(sha):
                        p_abs = os.path.abspath(os.path.join(root, str(p)) if not os.path.isabs(str(p)) else str(p))
                        sha_val = str(sha).strip().lower()
                        checksums_map[os.path.normpath(p_abs).replace("\\", "/").lower()] = sha_val
                        checksums_map[str(p).replace("\\", "/").lower()] = sha_val
            except Exception:
                pass

    man_record = manifest_map.get(norm_fp) or manifest_map.get(rel_fp)
    if man_record is not None:
        # Require manifest status == 'COMPLETE'
        if man_record.get("status") != "COMPLETE":
            return False
        m_sha = man_record.get("checksum")
        if m_sha and str(m_sha).strip().lower() != actual_hash:
            return False

    # Locate recorded checksum in checksums_map or manifest
    expected_sha = checksums_map.get(norm_fp) or checksums_map.get(rel_fp)
    if not expected_sha and man_record:
        expected_sha = str(man_record.get("checksum", "")).strip().lower()

    if not expected_sha or expected_sha != actual_hash:
        return False

    return True

_is_provenance_verified = is_provenance_verified

def check_pilot_completeness(start_date="2020-01-01", end_date="2020-01-07",
                             acquisition_results=None, data_dir=None,
                             manifest_mgr=None, checksums_file=None):
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

    # Pre-build lookup maps once for performance
    manifest_map = {}
    try:
        mgr = manifest_mgr if manifest_mgr is not None else ManifestManager()
        chunks = mgr.manifest.get("chunks", mgr.manifest) if isinstance(mgr.manifest.get("chunks"), dict) else mgr.manifest
        for row in chunks.values():
            if isinstance(row, dict):
                out_f = row.get("output_file", "")
                if out_f:
                    row_abs = os.path.abspath(os.path.join(repo_root, out_f) if not os.path.isabs(out_f) else out_f)
                    manifest_map[os.path.normpath(row_abs).replace("\\", "/").lower()] = row
                    manifest_map[str(out_f).replace("\\", "/").lower()] = row
    except Exception:
        pass

    checksums_map = {}
    ck_file = checksums_file if checksums_file is not None else os.path.join(repo_root, "data", "checksums.csv")
    if os.path.exists(ck_file):
        try:
            df_c = pd.read_csv(ck_file)
            for _, r in df_c.iterrows():
                p = r.get("filepath")
                sha = r.get("sha256")
                if pd.notna(p) and pd.notna(sha):
                    p_abs = os.path.abspath(os.path.join(repo_root, str(p)) if not os.path.isabs(str(p)) else str(p))
                    sha_val = str(sha).strip().lower()
                    checksums_map[os.path.normpath(p_abs).replace("\\", "/").lower()] = sha_val
                    checksums_map[str(p).replace("\\", "/").lower()] = sha_val
        except Exception:
            pass

    hash_cache = {}

    def _is_provenance_verified(filepath):
        return is_provenance_verified(
            filepath,
            repo_root_dir=repo_root,
            checksums_map=checksums_map,
            manifest_map=manifest_map,
            hash_cache=hash_cache
        )

    def get_source_covered_dates(ds_name):
        covered = set()
        files = glob.glob(os.path.join(data_dir, ds_name, "*.nc"))
        for f in files:
            if not _is_provenance_verified(f):
                continue
            bname = os.path.basename(f)
            parts = bname.replace(".nc", "").split("_")
            date_parts = [p for p in parts if len(p) == 10 and p[4] == "-" and p[7] == "-"]
            if len(date_parts) == 1:
                covered.add(date_parts[0])
            elif len(date_parts) >= 2:
                c_start = date_parts[-2]
                c_end = date_parts[-1]
                try:
                    for dt in pd.date_range(c_start, c_end):
                        covered.add(dt.strftime("%Y-%m-%d"))
                except Exception:
                    pass
        return covered

    req_dates_set = set(dates)

    # 1. ARGO Validation Profiles
    argo_path = os.path.join(data_dir, "argo", f"argo_profiles_{start_date}_{end_date}.csv")
    argo_ok = _is_provenance_verified(argo_path)
    if not argo_ok:
        argo_candidates = glob.glob(os.path.join(data_dir, "argo", "*.csv"))
        argo_ok = any(_is_provenance_verified(f) for f in argo_candidates)
    if acquisition_results and str(acquisition_results.get("argo", "")).startswith("FAILED"):
        argo_ok = False
    source_status["argo"] = argo_ok
    if not argo_ok:
        missing_or_failed.append("ARGO")

    # 2. GLORYS Subsurface Temperature (thetao)
    glorys_covered = get_source_covered_dates("glorys")
    glorys_ok = len(req_dates_set) > 0 and req_dates_set.issubset(glorys_covered)
    if acquisition_results and str(acquisition_results.get("glorys", "")).startswith("FAILED"):
        glorys_ok = False
    source_status["glorys"] = glorys_ok
    if not glorys_ok:
        missing_or_failed.append("GLORYS")

    # 3. OSTIA Sea Surface Temperature (SST)
    ostia_covered = get_source_covered_dates("ostia")
    ostia_ok = len(req_dates_set) > 0 and req_dates_set.issubset(ostia_covered)
    if acquisition_results and str(acquisition_results.get("ostia", "")).startswith("FAILED"):
        ostia_ok = False
    source_status["ostia"] = ostia_ok
    if not ostia_ok:
        missing_or_failed.append("OSTIA")

    # 4. Copernicus Multi-Obs Sea Surface Salinity (SSS)
    sss_covered = get_source_covered_dates("sss")
    sss_ok = len(req_dates_set) > 0 and req_dates_set.issubset(sss_covered)
    if acquisition_results and str(acquisition_results.get("sss", "")).startswith("FAILED"):
        sss_ok = False
    source_status["sss"] = sss_ok
    if not sss_ok:
        missing_or_failed.append("SSS")

    # 5. DUACS Sea Level Anomaly (SSH/SLA)
    duacs_covered = get_source_covered_dates("duacs")
    duacs_ok = len(req_dates_set) > 0 and req_dates_set.issubset(duacs_covered)
    if acquisition_results and str(acquisition_results.get("duacs", "")).startswith("FAILED"):
        duacs_ok = False
    source_status["duacs"] = duacs_ok
    if not duacs_ok:
        missing_or_failed.append("DUACS")

    # 6. OSCAR Surface Currents (U, V)
    oscar_covered = get_source_covered_dates("oscar")
    oscar_ok = len(req_dates_set) > 0 and req_dates_set.issubset(oscar_covered)
    if acquisition_results and str(acquisition_results.get("oscar", "")).startswith("FAILED"):
        oscar_ok = False
    source_status["oscar"] = oscar_ok
    if not oscar_ok:
        missing_or_failed.append("OSCAR")

    # 7. CCMP Surface Winds (U, V)
    ccmp_covered = get_source_covered_dates("ccmp")
    ccmp_ok = len(req_dates_set) > 0 and req_dates_set.issubset(ccmp_covered)
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
        fy_certifiable, fy_source_status, fy_missing_or_failed = check_pilot_completeness(
            start_date, end_date, acquisition_results=results
        )
        print("\n" + "=" * 70)
        print("FULL-YEAR ACQUISITION GATE")
        print("=" * 70)
        if not fy_certifiable:
            print("FULL-YEAR STATUS: INCOMPLETE")
            print(f"Missing/failed: {', '.join(fy_missing_or_failed)}")
            print("=" * 70)
            if exit_on_failure:
                sys.exit(1)
            return False
        else:
            print("FULL-YEAR STATUS: COMPLETE")
            print("All 7 required sources present, non-empty, and verified for the full year 2020.")
            print("Certification: READY FOR FULL-YEAR HARMONIZATION")
            print("=" * 70)
            return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquisition pipeline runner")
    parser.add_argument("--pilot", action="store_true", help="Run 7-day pilot")
    parser.add_argument("--full-year", action="store_true", help="Run full 1-year acquisition")
    args = parser.parse_args()
    
    mode = "full-year" if args.full_year else "pilot"
    run_download_pipeline(mode, exit_on_failure=True)
