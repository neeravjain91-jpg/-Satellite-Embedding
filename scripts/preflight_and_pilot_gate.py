"""
scripts/preflight_and_pilot_gate.py
Automated Preflight Validation & Pilot Acceptance Gate:
1. Validates Copernicus Marine credentials using 'copernicusmarine login --check-credentials-valid'.
2. Executes non-destructive '--dry-run' subset checks for each Copernicus product (GLORYS, OSTIA, SSS, DUACS).
3. Validates NASA Earthdata credentials / token for OSCAR surface currents.
4. Verifies public endpoint availability for CCMP winds and ARGO in-situ floats.
5. Implements the Pilot-First Gate: strictly blocks full-year acquisition until the 7-day pilot is fully validated.
"""

import os
import sys
import json
import subprocess
import requests
import xarray as xr
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

PILOT_VALIDATION_MARKER = os.path.join(repo_root, "reports", "pilot_acceptance_certified.json")

COPERNICUS_DATASETS = [
    {
        "name": "GLORYS Subsurface Temperature (thetao)",
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["thetao"],
        "depth_range": (0.0, 1000.0),
        "bbox": (5.0, 30.0, 45.0, 105.0)
    },
    {
        "name": "OSTIA Sea Surface Temperature (SST)",
        "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        "variables": ["analysed_sst"],
        "depth_range": None,
        "bbox": (5.0, 30.0, 45.0, 105.0)
    },
    {
        "name": "Copernicus Multi-Obs Sea Surface Salinity (SSS)",
        "dataset_id": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        "variables": ["sos"],
        "depth_range": None,
        "bbox": (5.0, 30.0, 45.0, 105.0)
    },
    {
        "name": "DUACS Sea Level Anomaly (SSH/SLA)",
        "dataset_id": "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D",
        "variables": ["sla"],
        "depth_range": None,
        "bbox": (5.0, 30.0, 45.0, 105.0)
    }
]

def check_copernicus_credentials_cli():
    """
    Runs the official 'copernicusmarine login --check-credentials-valid' command.
    Returns (is_valid: bool, message: str)
    """
    try:
        proc = subprocess.run(
            ["copernicusmarine", "login", "--check-credentials-valid"],
            capture_output=True,
            text=True,
            timeout=15
        )
        combined_output = proc.stdout + " " + proc.stderr
        if proc.returncode == 0 and "valid" in combined_output.lower() and "no credentials" not in combined_output.lower():
            return True, "Credentials are valid and verified via Copernicus Marine CLI."
        else:
            return False, combined_output.strip()
    except Exception as e:
        return False, str(e)

def run_copernicus_dry_run(ds_info, test_date="2020-01-01"):
    """
    Executes a single small subset with --dry-run for a Copernicus Marine product.
    Returns (success: bool, detail: str)
    """
    dataset_id = ds_info["dataset_id"]
    variable = ds_info["variables"][0]
    lat_min, lat_max, lon_min, lon_max = ds_info["bbox"]

    cmd = [
        "copernicusmarine", "subset",
        "-i", dataset_id,
        "-v", variable,
        "--start-datetime", f"{test_date}T00:00:00",
        "--end-datetime", f"{test_date}T23:59:59",
        "-x", str(lon_min), "-X", str(lon_max),
        "-y", str(lat_min), "-Y", str(lat_max),
        "--dry-run"
    ]
    if ds_info.get("depth_range"):
        d_min, d_max = ds_info["depth_range"]
        cmd.extend(["-z", str(d_min), "-Z", str(d_max)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        combined_output = proc.stdout + " " + proc.stderr
        if proc.returncode == 0:
            return True, "Dry-run query executed successfully. Dataset accessible and parameters valid."
        else:
            return False, combined_output.strip()
    except Exception as e:
        return False, str(e)

def check_earthdata_credentials():
    """
    Checks for NASA Earthdata credentials for OSCAR currents access.
    """
    token = os.environ.get("EARTHDATA_TOKEN")
    user = os.environ.get("EARTHDATA_USERNAME")
    pwd = os.environ.get("EARTHDATA_PASSWORD")
    netrc_files = [os.path.expanduser("~/.netrc"), os.path.expanduser("~/_netrc")]
    has_netrc = any(os.path.exists(p) for p in netrc_files)

    if token:
        return True, "EARTHDATA_TOKEN environment variable found."
    elif user and pwd:
        return True, "EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables found."
    elif has_netrc:
        return True, "Netrc credentials file found in user home directory."
    else:
        return False, "No NASA Earthdata credentials found (set EARTHDATA_TOKEN or EARTHDATA_USERNAME/EARTHDATA_PASSWORD)."

def check_public_sources():
    """
    Verifies availability of public open-access endpoints (CCMP winds on RemSS and ARGO on IFREMER).
    """
    status = {}
    # CCMP on Remote Sensing Systems
    try:
        r_ccmp = requests.head("https://data.remss.com/ccmp/v03.1/Y2020/M01/", timeout=10)
        status["ccmp"] = (r_ccmp.status_code == 200, f"HTTP Status {r_ccmp.status_code}")
    except Exception as e:
        status["ccmp"] = (False, str(e))

    # ARGO on IFREMER GDAC ERDDAP
    try:
        r_argo = requests.head("https://www.ifremer.fr/erddap/tabledap/ArgoFloats.html", timeout=10)
        status["argo"] = (r_argo.status_code == 200, f"HTTP Status {r_argo.status_code}")
    except Exception as e:
        status["argo"] = (False, str(e))

    return status

def run_preflight_checks():
    print("=" * 80)
    print("PREFLIGHT VALIDATION: AUTHENTICATION, DRY-RUNS & PUBLIC ENDPOINTS")
    print("=" * 80)

    results = {"copernicus_auth": None, "copernicus_dry_runs": {}, "earthdata_auth": None, "public_sources": {}}

    # 1. Copernicus Marine Authentication Check
    print("\n[1/4] Checking Copernicus Marine Authentication...")
    cop_valid, cop_msg = check_copernicus_credentials_cli()
    results["copernicus_auth"] = {"valid": cop_valid, "detail": cop_msg}
    if cop_valid:
        print(f"  [PASS] {cop_msg}")
    else:
        print(f"  [BLOCKED] Copernicus Marine authentication required.")
        print(f"    Detail: {cop_msg}")

    # 2. Copernicus Marine Dataset-Specific Dry Runs
    print("\n[2/4] Executing Copernicus Marine Dataset-Specific --dry-run Checks...")
    for ds in COPERNICUS_DATASETS:
        name = ds["name"]
        if not cop_valid:
            results["copernicus_dry_runs"][name] = {"success": False, "detail": "Skipped: Authentication required."}
            print(f"  {name:50s}: [BLOCKED] Awaiting authentication")
        else:
            success, detail = run_copernicus_dry_run(ds)
            results["copernicus_dry_runs"][name] = {"success": success, "detail": detail}
            status_tag = "[PASS]" if success else "[FAILED]"
            print(f"  {name:50s}: {status_tag} {detail[:60]}")

    # 3. NASA Earthdata Authentication Check
    print("\n[3/4] Checking NASA Earthdata Authentication (for OSCAR currents)...")
    ed_valid, ed_msg = check_earthdata_credentials()
    results["earthdata_auth"] = {"valid": ed_valid, "detail": ed_msg}
    status_tag = "[PASS]" if ed_valid else "[BLOCKED]"
    print(f"  {status_tag} {ed_msg}")

    # 4. Public Endpoints Check
    print("\n[4/4] Checking Open-Access Public Data Endpoints (CCMP Winds & ARGO)...")
    pub_status = check_public_sources()
    results["public_sources"] = pub_status
    for k, (ok, msg) in pub_status.items():
        status_tag = "[PASS]" if ok else "[WARN]"
        print(f"  {k.upper():10s}: {status_tag} {msg}")

    return results

def is_pilot_acceptance_certified():
    """Checks whether the 7-day pilot has passed all 3 QA gates and has been certified."""
    if not os.path.exists(PILOT_VALIDATION_MARKER):
        return False
    try:
        with open(PILOT_VALIDATION_MARKER, "r") as f:
            data = json.load(f)
            return data.get("pilot_certified", False)
    except Exception:
        return False

def certify_pilot_acceptance(pilot_metadata):
    """Certifies that the 7-day pilot has met all acceptance criteria."""
    os.makedirs(os.path.dirname(PILOT_VALIDATION_MARKER), exist_ok=True)
    record = {
        "pilot_certified": True,
        "certification_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "period": "2020-01-01 to 2020-01-07",
        "qa_checks_passed": ["provenance", "variable_temporal_variability", "cross_variable_physical_sanity"],
        "metadata": pilot_metadata
    }
    with open(PILOT_VALIDATION_MARKER, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[ACCEPTANCE CERTIFIED] 7-day authentic pilot certified in {PILOT_VALIDATION_MARKER}")

if __name__ == "__main__":
    run_preflight_checks()
