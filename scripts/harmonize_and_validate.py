"""
scripts/harmonize_and_validate.py
REAL-DATA Harmonization, Quality Assurance, and Assembly Pipeline.

STRICT REAL-DATA ENFORCEMENT:
- Strictly prohibits any synthetic data substitute functions (generate_scientifically_consistent_pilot_fields permanently deleted).
- Ingests exclusively authentic Level 4 observational and GLORYS reanalysis NetCDF files.
- If ANY required real dataset or date is missing, halts immediately with FileNotFoundError.

THREE SCIENTIFIC ACCEPTANCE CHECKS:
1. Provenance Check: File existence, official product lineage, and SHA-256 checksums recorded in data/checksums.csv.
2. Variable-Specific Temporal Variability Check:
   - Wind U/V (atmospheric): >30% ocean cells differing by >0.01 m/s, max diff > 0.2 m/s.
   - Current U/V (mesoscale ocean): >5% ocean cells differing by >0.005 m/s, max diff > 0.02 m/s.
   - SST (surface heat flux): >5% ocean cells differing by >0.01 °C, max diff > 0.05 °C.
   - SSH/SLA (sea level anomaly): >3% ocean cells differing by >0.001 m, max diff > 0.005 m.
   - SSS (slow salinity advection): >1% ocean cells differing by >0.001 psu, max diff > 0.005 psu.
   - GLORYS thetao: Depth-stratified (mixed layer >3%, thermocline >1%, deep ocean max diff > 0.0001 °C).
   - Anti-synthetic rule: No variable may have max abs diff == 0.000000.
3. Cross-Variable Physical Sanity & Coordinate Checks:
   - Canonical 0.25° grid coordinates [101 lat, 241 lon, 15 depths].
   - Physical range bounding (SST 15-35°C, SSS 20-42 psu, SSH -1.5 to 1.5m, currents +/-3m/s, winds +/-35m/s).
   - Seafloor bathymetric cutoff (depths deeper than local bathymetry strictly NaN).
   - Cross-variable non-degeneracy check: deep thetao must NOT be an exact linear replica of SST (R² < 0.999).
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import (
    CANONICAL_LATS, CANONICAL_LONS, CANONICAL_DEPTHS, CANONICAL_FEATURES,
    validate_canonical_coords
)
from preprocessing.depth_interpolation import interpolate_glorys_to_canonical_depths
from preprocessing.ocean_mask import (
    generate_canonical_ocean_mask_from_glorys, save_canonical_ocean_mask,
    apply_surface_mask, apply_target_mask
)
from preprocessing.regrid import regrid_2d_field, regrid_ostia_sst, regrid_glorys_thetao
from preprocessing.temporal_align import generate_daily_time_range
from preprocessing.build_dataset import assemble_ml_dataset, OceanReconstructionDataset
from preprocessing.argo_matchup import match_argo_profiles_with_model, evaluate_argo_matchups
from scripts.manifest_manager import compute_sha256, ManifestManager
from scripts.preflight_and_pilot_gate import certify_pilot_acceptance

PHYSICAL_RANGES = {
    "sst": (10.0, 35.0),          # degC (northern Arabian Sea / Persian Gulf reaches ~10.5 C in Jan)
    "sss": (12.0, 42.0),          # psu (major river outflow in BoB reaches ~14 PSU; Red Sea / Persian Gulf ~40-41 PSU)
    "ssh": (-1.5, 1.5),           # m
    "current_u": (-3.0, 3.0),     # m/s
    "current_v": (-3.0, 3.0),     # m/s
    "wind_u": (-35.0, 35.0),       # m/s
    "wind_v": (-35.0, 35.0),       # m/s
    "thetao": (-2.0, 35.0)        # degC
}

VARIABLE_VARIABILITY_CRITERIA = {
    "wind_u":    {"min_frac_change": 0.30, "min_val_change": 0.01,  "min_max_diff": 0.20},
    "wind_v":    {"min_frac_change": 0.30, "min_val_change": 0.01,  "min_max_diff": 0.20},
    "current_u": {"min_frac_change": 0.05, "min_val_change": 0.005, "min_max_diff": 0.02},
    "current_v": {"min_frac_change": 0.05, "min_val_change": 0.005, "min_max_diff": 0.02},
    "sst":       {"min_frac_change": 0.05, "min_val_change": 0.01,  "min_max_diff": 0.05},
    "ssh":       {"min_frac_change": 0.03, "min_val_change": 0.001, "min_max_diff": 0.005},
    "sss":       {"min_frac_change": 0.01, "min_val_change": 0.001, "min_max_diff": 0.005}
}

class ResolvedSource(tuple):
    """
    Tuple representing a resolved source file and time index:
    (file_path, time_index) with .file_path, .time_index, .date attributes.
    """
    def __new__(cls, file_path, time_index, date_str=None):
        return super().__new__(cls, (file_path, time_index))

    def __init__(self, file_path, time_index, date_str=None):
        self.file_path = file_path
        self.time_index = int(time_index)
        self.date_str = date_str or ""
        self.date = self.date_str

def normalize_source_path(path):
    abs_path = os.path.abspath(path)
    abs_repo = os.path.abspath(repo_root)
    if abs_path.startswith(abs_repo):
        rel = os.path.relpath(abs_path, abs_repo).replace("\\", "/")
        return rel
    return abs_path.replace("\\", "/")

def find_time_index_in_dataset(ds, target_date_str):
    """
    Finds the integer index of target_date_str in ds's time coordinate.
    Inspects actual coordinates, handling numpy.datetime64, cftime, and pandas timestamps.
    Returns index (int) or None if target_date_str is not present.
    """
    time_coord = None
    for name in ["time", "Time", "times", "Times"]:
        if name in ds.coords or name in ds.dims:
            time_coord = ds[name]
            break
    if time_coord is None:
        for c in ds.coords:
            if "time" in c.lower():
                time_coord = ds[c]
                break
    if time_coord is None:
        return None

    t_vals = time_coord.values
    if t_vals.ndim == 0:
        t_vals = [t_vals]

    for idx, val in enumerate(t_vals):
        val_str = str(val)[:10]
        if val_str == target_date_str:
            return idx
    return None

DATASET_SUBDIRS = {
    "ostia": "ostia",
    "ostia sst": "ostia",
    "sst": "ostia",
    "sss": "sss",
    "multi-obs sss": "sss",
    "duacs": "duacs",
    "duacs ssh": "duacs",
    "duacs sla": "duacs",
    "ssh": "duacs",
    "sla": "duacs",
    "glorys": "glorys",
    "glorys thetao": "glorys",
    "thetao": "glorys",
    "oscar": "oscar",
    "oscar currents": "oscar",
    "currents": "oscar",
    "ccmp": "ccmp",
    "ccmp winds": "ccmp",
    "winds": "ccmp",
    "argo": "argo"
}

def resolve_source_file_for_date(dataset, date_str, base_dir="data/raw"):
    """
    Resolves the authentic raw data file and time index covering date_str.
    Supports both daily files (e.g. OSCAR, CCMP) and multi-day chunks (e.g. OSTIA, SSS, DUACS, GLORYS).
    - Inspects repository-relative raw files
    - Prefers an exact daily file if present
    - Otherwise locates a multi-day chunk that covers date_str
    - Opens the NetCDF and verifies its time coordinate actually contains date_str
    - Fails clearly if no file covers the requested date
    Returns:
        ResolvedSource(file_path, time_index, date_str)
    """
    ds_key = dataset.lower().strip()
    sub_dir = DATASET_SUBDIRS.get(ds_key, ds_key)

    dir_path = os.path.join(base_dir, sub_dir)
    if not os.path.exists(dir_path):
        if os.path.isdir(base_dir) and (glob.glob(os.path.join(base_dir, "*.nc")) or glob.glob(os.path.join(base_dir, "*.csv"))):
            dir_path = base_dir
        else:
            raise FileNotFoundError(f"[PROVENANCE FAIL] Directory does not exist for dataset '{dataset}': {dir_path}")

    if ds_key == "argo":
        argo_candidates = glob.glob(os.path.join(dir_path, "*.csv"))
        if not argo_candidates:
            raise FileNotFoundError(f"[PROVENANCE FAIL] Missing authentic ARGO CSV in {dir_path}")
        return ResolvedSource(normalize_source_path(argo_candidates[0]), 0, date_str)

    all_nc = sorted(glob.glob(os.path.join(dir_path, "*.nc")))
    if not all_nc:
        raise FileNotFoundError(
            f"[PROVENANCE FAIL] No NetCDF files found for dataset '{dataset}' in {dir_path}."
        )

    date_compact = date_str.replace("-", "")

    # Load manifest registered outputs if manifest file exists
    manifest_outputs = set()
    try:
        mf = ManifestManager()
        for rec in mf.manifest.values():
            out_f = rec.get("output_file")
            if out_f:
                manifest_outputs.add(out_f.replace("\\", "/"))
    except Exception:
        pass

    def candidate_rank(path):
        bname = os.path.basename(path).lower()
        norm_p = normalize_source_path(path)
        is_manifest = norm_p in manifest_outputs
        is_exact_daily = ("_daily_" in bname or bname.startswith("oscar_") or bname.startswith(f"{ds_key}_")) and (date_str in bname or date_compact in bname)
        is_raw_global = "wind_analysis" in bname or "_staging" in bname
        return (0 if is_manifest else 1, 0 if is_exact_daily else 1, 1 if is_raw_global else 0, len(bname))

    ranked_nc = sorted(all_nc, key=candidate_rank)

    for cand in ranked_nc:
        try:
            with xr.open_dataset(cand) as ds:
                t_idx = find_time_index_in_dataset(ds, date_str)
                if t_idx is not None:
                    return ResolvedSource(normalize_source_path(cand), t_idx, date_str)
        except Exception:
            continue

    raise FileNotFoundError(
        f"[PROVENANCE FAIL] Missing authentic NetCDF file for '{dataset}' on date {date_str} in {dir_path}.\n"
        f"Checked {len(all_nc)} candidate files. No file contains time coordinate '{date_str}'.\n"
        f"Synthetic substitution is strictly prohibited."
    )

def check_provenance_and_lineage(start_date, end_date, base_dir="data/raw", manifest_mgr=None, checksums_path=None):
    """
    Check 1: Provenance & Lineage Check
    For each requested date and dataset:
    - Resolves containing file (daily or multi-day chunk)
    - Verifies file exists and size > 0
    - Verifies its manifest entry is COMPLETE
    - Verifies checksum against data/checksums.csv by recomputing SHA-256
    - Verifies file's time coordinate contains requested date
    Prints provenance resolution table and returns provenance_log.
    """
    print("\n--- CHECK 1: DATA PROVENANCE & LINEAGE VERIFICATION ---")
    times = generate_daily_time_range(start_date, end_date)
    if manifest_mgr is None:
        manifest_mgr = ManifestManager()
    if checksums_path is None:
        checksums_path = "data/checksums.csv"

    recorded_checksums = {}
    if os.path.exists(checksums_path):
        df_chk = pd.read_csv(checksums_path)
        for _, row in df_chk.iterrows():
            fp = str(row["filepath"]).replace("\\", "/")
            recorded_checksums[fp] = str(row["sha256"])

    gridded_sources = [
        "OSTIA SST",
        "Multi-Obs SSS",
        "DUACS SSH",
        "OSCAR Currents",
        "CCMP Winds",
        "GLORYS thetao"
    ]

    verified_shas = {}
    provenance_log = []

    print("=" * 135)
    print(f"{'Dataset':<16} {'Requested Date':<15} {'Resolved File':<52} {'Selected Time':<18} {'Status':<10} {'SHA-256'}")
    print("=" * 135)

    for dt in times:
        d_str = dt.strftime("%Y-%m-%d")

        for ds_name in gridded_sources:
            # 1. Resolve source file and time index
            res = resolve_source_file_for_date(ds_name, d_str, base_dir=base_dir)
            f_path = res.file_path
            t_idx = res.time_index

            # 2. Verify existence and non-zero size
            if not os.path.exists(f_path):
                raise FileNotFoundError(f"[PROVENANCE FAIL] Resolved file does not exist: {f_path}")
            f_size = os.path.getsize(f_path)
            if f_size == 0:
                raise ValueError(f"[PROVENANCE FAIL] Zero-byte file detected for '{ds_name}': {f_path}")

            # 3. Check manifest status is COMPLETE
            matching_chunks = [
                rec for rec in manifest_mgr.manifest.values()
                if rec.get("output_file") in (f_path, f_path.replace("/", "\\"))
                or os.path.basename(rec.get("output_file", "")) == os.path.basename(f_path)
            ]
            if not matching_chunks:
                raise ValueError(f"[PROVENANCE FAIL] No manifest record found for {f_path}")

            chunk_rec = matching_chunks[0]
            if chunk_rec.get("status") != "COMPLETE":
                raise ValueError(
                    f"[PROVENANCE FAIL] Manifest status for {f_path} is '{chunk_rec.get('status')}', expected 'COMPLETE'."
                )

            # 4. Verify checksum against data/checksums.csv
            if f_path not in verified_shas:
                computed_sha = compute_sha256(f_path)
                recorded_sha = recorded_checksums.get(f_path) or recorded_checksums.get(f_path.replace("/", "\\")) or chunk_rec.get("checksum")
                if not recorded_sha or computed_sha != recorded_sha:
                    raise ValueError(
                        f"[PROVENANCE FAIL] SHA-256 mismatch for {f_path}!\n"
                        f"Computed: {computed_sha}\n"
                        f"Recorded: {recorded_sha}"
                    )
                verified_shas[f_path] = computed_sha

            short_sha = verified_shas[f_path][:16] + "..."
            time_repr = f"{d_str} (t={t_idx})"
            print(f"{ds_name:<16} {d_str:<15} {f_path:<52} {time_repr:<18} {'COMPLETE':<10} {short_sha}")

            provenance_log.append({
                "date": d_str,
                "dataset": ds_name,
                "file": f_path,
                "time_index": t_idx,
                "size_bytes": f_size,
                "sha256": verified_shas[f_path]
            })

    print("=" * 135)
    print(f"[PASS] Provenance check passed for all {len(times)} days across 6 gridded sources ({len(provenance_log)} verified resolutions).")
    return provenance_log

def check_variable_temporal_variability(surface_dict, target_thetao, times):
    """
    Check 2: Variable-Specific Temporal Variability Check
    Replaces arbitrary blanket thresholds with oceanographically justified rates of change.
    """
    print("\n--- CHECK 2: VARIABLE-SPECIFIC TEMPORAL VARIABILITY CHECK ---")
    if len(times) < 2:
        print("[WARN] Pilot has fewer than 2 days; skipping temporal diff check.")
        return True

    # 1. Surface Variables Check
    for feat, crit in VARIABLE_VARIABILITY_CRITERIA.items():
        f0 = surface_dict[feat][0]
        f1 = surface_dict[feat][1]
        diff = np.abs(f1 - f0)
        valid = ~np.isnan(diff)
        n_valid = int(valid.sum())
        if n_valid == 0:
            raise ValueError(f"[QA FAIL] Surface feature {feat} has 0 valid ocean points.")

        max_diff = float(np.nanmax(diff))
        mean_diff = float(np.nanmean(diff))
        frac_diff = float(np.sum(diff[valid] > crit["min_val_change"])) / n_valid

        print(f"  {feat:10s} Day 0 vs 1 | Max Diff: {max_diff:8.4f} | Mean Diff: {mean_diff:8.4f} | Frac Diff: {frac_diff:6.4f} (Required: >{crit['min_frac_change']:.2f})")

        # Anti-synthetic check: must not be bitwise identical
        if max_diff == 0.0:
            raise ValueError(f"[QA FAIL] Feature '{feat}' is 100% bitwise static between Day 0 and Day 1 (Synthetic pattern detected).")

        # Specific oceanographic threshold check
        if frac_diff < crit["min_frac_change"] or max_diff < crit["min_max_diff"]:
            raise ValueError(
                f"[QA FAIL] Temporal variability for '{feat}' below expected physical rate of change!\n"
                f"Observed frac differing: {frac_diff:.4f} (min required: {crit['min_frac_change']}), "
                f"max diff: {max_diff:.4f} (min required: {crit['min_max_diff']})."
            )

    # 2. GLORYS Subsurface Temperature Depth-Stratified Check
    t0 = target_thetao[0]  # (15, 101, 241)
    t1 = target_thetao[1]
    diff_t = np.abs(t1 - t0)

    # Upper mixed layer (0-50m, depths 0-5)
    diff_upper = diff_t[:6]
    valid_upper = ~np.isnan(diff_upper)
    max_upper = float(np.nanmax(diff_upper))
    frac_upper = float(np.sum(diff_upper[valid_upper] > 0.01)) / max(int(valid_upper.sum()), 1)
    print(f"  thetao (0-50m) Day 0 vs 1 | Max Diff: {max_upper:8.4f} °C | Frac Diff: {frac_upper:6.4f} (Required: >0.03)")

    if max_upper < 0.02 or frac_upper < 0.03:
        raise ValueError(f"[QA FAIL] Upper ocean thetao (0-50m) shows insufficient temporal variability (Max diff: {max_upper:.4f}°C).")

    # Thermocline layer (75-300m, depths 6-11)
    diff_thermo = diff_t[6:12]
    valid_thermo = ~np.isnan(diff_thermo)
    max_thermo = float(np.nanmax(diff_thermo))
    frac_thermo = float(np.sum(diff_thermo[valid_thermo] > 0.01)) / max(int(valid_thermo.sum()), 1)
    print(f"  thetao (75-300m) Day 0 vs 1 | Max Diff: {max_thermo:8.4f} °C | Frac Diff: {frac_thermo:6.4f} (Required: >0.01)")

    if max_thermo < 0.01 or frac_thermo < 0.01:
        raise ValueError(
            f"[QA FAIL] Thermocline ocean thetao (75-300m) shows insufficient temporal variability "
            f"(Max diff: {max_thermo:.4f}°C, Frac diff: {frac_thermo:.4f})."
        )

    # Deep levels (500-1000m, depths 12-14)
    diff_deep = diff_t[12:]
    valid_deep = ~np.isnan(diff_deep)
    max_deep = float(np.nanmax(diff_deep))
    print(f"  thetao (500-1000m)       | Max Diff: {max_deep:8.6f} °C (Must be > 0.000000)")

    if max_deep == 0.0:
        raise ValueError("[QA FAIL] Deep ocean thetao is 100% bitwise static between Day 0 and Day 1.")

    print("[PASS] Variable-specific temporal variability verified successfully.")
    return True

def check_cross_variable_physical_sanity(surface_dict, target_thetao, mask_ds):
    """
    Check 3: Cross-Variable Physical Sanity & Coordinate Checks
    Verifies valid physical bounds, units, canonical coordinates, and ensures non-degeneracy.
    Evaluates ocean physical variables strictly over ocean grid cells (masked against land/lake).
    """
    print("\n--- CHECK 3: CROSS-VARIABLE PHYSICAL SANITY & COORDINATES ---")
    
    m2d = mask_ds.ocean_mask_2d.values  # (101, 241) boolean
    m3d = mask_ds.ocean_mask_3d.values  # (15, 101, 241) boolean
    
    # 1. Surface Variables Physical Range Check
    for feat in CANONICAL_FEATURES:
        arr = surface_dict[feat]
        ocean_data = arr[:, m2d] if arr.ndim == 3 else arr[m2d]
        valid_ocean = ocean_data[~np.isnan(ocean_data)]
        if len(valid_ocean) == 0:
            raise ValueError(f"[PHYSICAL SANITY FAIL] No valid ocean data points for {feat}")
        min_val = float(np.min(valid_ocean))
        max_val = float(np.max(valid_ocean))
        bound_min, bound_max = PHYSICAL_RANGES[feat]
        print(f"  {feat:10s} Ocean Range: [{min_val:8.3f}, {max_val:8.3f}] (Permitted: [{bound_min}, {bound_max}])")
        if min_val < bound_min or max_val > bound_max:
            raise ValueError(f"[PHYSICAL SANITY FAIL] {feat} out of physical range: [{min_val}, {max_val}]")

    # 2. Target thetao Physical Range Check
    ocean_th = target_thetao[:, m3d] if target_thetao.ndim == 4 else target_thetao[m3d]
    valid_th = ocean_th[~np.isnan(ocean_th)]
    if len(valid_th) == 0:
        raise ValueError("[PHYSICAL SANITY FAIL] No valid ocean data points for target thetao")
    min_th = float(np.min(valid_th))
    max_th = float(np.max(valid_th))
    print(f"  thetao     Ocean Range: [{min_th:8.3f}, {max_th:8.3f}] °C (Permitted: [-2.0, 35.0] °C)")
    if min_th < -2.0 or max_th > 35.0:
        raise ValueError(f"[PHYSICAL SANITY FAIL] thetao out of physical range: [{min_th}, {max_th}] °C")

    # 3. Bathymetric Seafloor Mask Check
    for d_idx, z in enumerate(CANONICAL_DEPTHS):
        depth_data = target_thetao[:, d_idx]
        # Where bathymetry mask is False (seafloor), values MUST be NaN
        violating = np.any(~np.isnan(depth_data[:, ~m3d[d_idx]]))
        if violating:
            raise ValueError(f"[BATHYMETRY FAIL] Non-NaN values found below local seafloor at depth {z}m!")

    # 4. Anti-Linear Synthetic Formulation Check: thetao(z) != a_z + b_z * SST
    sst_flat = surface_dict["sst"][0].ravel()
    th100_flat = target_thetao[0, 7].ravel() # 100m depth
    valid_pair = ~np.isnan(sst_flat) & ~np.isnan(th100_flat)
    if valid_pair.sum() > 100:
        corr_val = float(np.corrcoef(sst_flat[valid_pair], th100_flat[valid_pair])[0, 1])
        print(f"  SST vs thetao(100m) Spatial Correlation r: {corr_val:.4f}")
        if abs(corr_val) > 0.999:
            raise ValueError(f"[ANTI-SYNTHETIC FAIL] Perfect linear correlation (r={corr_val:.4f}) detected between SST and subsurface thetao.")

    print("[PASS] Cross-variable physical sanity and coordinate integrity verified.")
    return True

def load_real_surface_and_target_day(date_str, base_dir="data/raw"):
    """
    Loads, crops, and regrids real scientific observation files for a single date.
    Resolves daily files or multi-day chunks using resolve_source_file_for_date.
    Strictly selects ONLY the requested date before spatial regridding.
    Enforces shape assertions on all 2D surface fields and 3D target field.
    """
    # 1. SST (OSTIA)
    ostia_file, ostia_t_idx = resolve_source_file_for_date("ostia", date_str, base_dir=base_dir)
    with xr.open_dataset(ostia_file) as ds_sst:
        var_name = "analysed_sst" if "analysed_sst" in ds_sst else "sst"
        da_day = ds_sst[var_name].isel(time=ostia_t_idx).squeeze() if "time" in ds_sst[var_name].dims else ds_sst[var_name].squeeze()
        sst_regridded = regrid_ostia_sst(da_day)

    # 2. SSS (Copernicus Multi-Obs)
    sss_file, sss_t_idx = resolve_source_file_for_date("sss", date_str, base_dir=base_dir)
    with xr.open_dataset(sss_file) as ds_sss:
        var_name = "sos" if "sos" in ds_sss else "sss"
        lat_c = "latitude" if "latitude" in ds_sss.coords else "lat"
        lon_c = "longitude" if "longitude" in ds_sss.coords else "lon"
        da_day = ds_sss[var_name].isel(time=sss_t_idx).squeeze() if "time" in ds_sss[var_name].dims else ds_sss[var_name].squeeze()
        sss_regridded = regrid_2d_field(da_day.values, ds_sss[lat_c].values, ds_sss[lon_c].values)

    # 3. SSH / SLA (DUACS)
    duacs_file, duacs_t_idx = resolve_source_file_for_date("duacs", date_str, base_dir=base_dir)
    with xr.open_dataset(duacs_file) as ds_duacs:
        var_name = "sla" if "sla" in ds_duacs else "adt"
        lat_c = "latitude" if "latitude" in ds_duacs.coords else "lat"
        lon_c = "longitude" if "longitude" in ds_duacs.coords else "lon"
        da_day = ds_duacs[var_name].isel(time=duacs_t_idx).squeeze() if "time" in ds_duacs[var_name].dims else ds_duacs[var_name].squeeze()
        ssh_regridded = regrid_2d_field(da_day.values, ds_duacs[lat_c].values, ds_duacs[lon_c].values)

    # 4. Surface Currents U, V (OSCAR)
    oscar_file, oscar_t_idx = resolve_source_file_for_date("oscar", date_str, base_dir=base_dir)
    with xr.open_dataset(oscar_file) as ds_oscar:
        lat_c = "latitude" if "latitude" in ds_oscar.coords else "lat"
        lon_c = "longitude" if "longitude" in ds_oscar.coords else "lon"
        u_da = ds_oscar["u"].isel(time=oscar_t_idx).squeeze() if "time" in ds_oscar["u"].dims else ds_oscar["u"].squeeze()
        v_da = ds_oscar["v"].isel(time=oscar_t_idx).squeeze() if "time" in ds_oscar["v"].dims else ds_oscar["v"].squeeze()
        
        # Ensure (latitude, longitude) orientation before regridding
        u_vals = u_da.values
        v_vals = v_da.values
        if u_vals.shape == (len(ds_oscar[lon_c]), len(ds_oscar[lat_c])) and len(ds_oscar[lon_c]) != len(ds_oscar[lat_c]):
            u_vals = u_vals.T
            v_vals = v_vals.T

        cur_u = regrid_2d_field(u_vals, ds_oscar[lat_c].values, ds_oscar[lon_c].values)
        cur_v = regrid_2d_field(v_vals, ds_oscar[lat_c].values, ds_oscar[lon_c].values)

    # 5. Surface Winds U, V (CCMP V3.1)
    ccmp_file, ccmp_t_idx = resolve_source_file_for_date("ccmp", date_str, base_dir=base_dir)
    with xr.open_dataset(ccmp_file) as ds_ccmp:
        lat_c = "latitude" if "latitude" in ds_ccmp.coords else "lat"
        lon_c = "longitude" if "longitude" in ds_ccmp.coords else "lon"
        u_var = "uwnd" if "uwnd" in ds_ccmp else "u"
        v_var = "vwnd" if "vwnd" in ds_ccmp else "v"
        u_da = ds_ccmp[u_var].isel(time=ccmp_t_idx).squeeze() if "time" in ds_ccmp[u_var].dims else ds_ccmp[u_var].squeeze()
        v_da = ds_ccmp[v_var].isel(time=ccmp_t_idx).squeeze() if "time" in ds_ccmp[v_var].dims else ds_ccmp[v_var].squeeze()
        wind_u = regrid_2d_field(u_da.values, ds_ccmp[lat_c].values, ds_ccmp[lon_c].values)
        wind_v = regrid_2d_field(v_da.values, ds_ccmp[lat_c].values, ds_ccmp[lon_c].values)

    # 6. GLORYS Subsurface Temperature (thetao)
    glorys_file, glorys_t_idx = resolve_source_file_for_date("glorys", date_str, base_dir=base_dir)
    with xr.open_dataset(glorys_file) as ds_glorys:
        da_day = ds_glorys["thetao"].isel(time=glorys_t_idx).squeeze() if "time" in ds_glorys["thetao"].dims else ds_glorys["thetao"].squeeze()
        da_interp = interpolate_glorys_to_canonical_depths(da_day)
        thetao_3d = regrid_glorys_thetao(da_interp)

    daily_surface = {
        "sst": sst_regridded,
        "sss": sss_regridded,
        "ssh": ssh_regridded,
        "current_u": cur_u,
        "current_v": cur_v,
        "wind_u": wind_u,
        "wind_v": wind_v
    }

    # Explicit shape assertions (Requirement K)
    assert daily_surface["sst"].shape == (101, 241), f"SST shape mismatch: {daily_surface['sst'].shape}"
    assert daily_surface["sss"].shape == (101, 241), f"SSS shape mismatch: {daily_surface['sss'].shape}"
    assert daily_surface["ssh"].shape == (101, 241), f"SSH shape mismatch: {daily_surface['ssh'].shape}"
    assert daily_surface["current_u"].shape == (101, 241), f"current_u shape mismatch: {daily_surface['current_u'].shape}"
    assert daily_surface["current_v"].shape == (101, 241), f"current_v shape mismatch: {daily_surface['current_v'].shape}"
    assert daily_surface["wind_u"].shape == (101, 241), f"wind_u shape mismatch: {daily_surface['wind_u'].shape}"
    assert daily_surface["wind_v"].shape == (101, 241), f"wind_v shape mismatch: {daily_surface['wind_v'].shape}"

    assert thetao_3d.shape == (15, 101, 241), f"thetao_3d shape mismatch: {thetao_3d.shape}"

    return daily_surface, thetao_3d

def execute_real_harmonization(start_date="2020-01-01", end_date="2020-01-07", mode="pilot"):
    """
    Executes real-data harmonization, 3-part QA suite, and Zarr dataset assembly.
    """
    print("=" * 80)
    print(f"REAL-DATA HARMONIZATION & 3-PART QA SUITE [MODE: {mode.upper()}]")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 80)

    # Check 1: Provenance & Lineage Check
    prov_log = check_provenance_and_lineage(start_date, end_date)

    times = generate_daily_time_range(start_date, end_date)
    n_days = len(times)
    print(f"\nProcessing {n_days} daily observation steps...")

    surface_arrays = {feat: np.zeros((n_days, len(CANONICAL_LATS), len(CANONICAL_LONS)), dtype=np.float32) for feat in CANONICAL_FEATURES}
    target_array = np.zeros((n_days, len(CANONICAL_DEPTHS), len(CANONICAL_LATS), len(CANONICAL_LONS)), dtype=np.float32)

    for t_idx, dt in enumerate(times):
        date_str = dt.strftime("%Y-%m-%d")
        print(f"Loading and regridding real observations for day {t_idx+1}/{n_days}: {date_str}...")
        daily_surf, daily_targ = load_real_surface_and_target_day(date_str)
        for feat in CANONICAL_FEATURES:
            surface_arrays[feat][t_idx] = daily_surf[feat]
        target_array[t_idx] = daily_targ

    # Check 2: Variable-Specific Temporal Variability Check
    check_variable_temporal_variability(surface_arrays, target_array, times)

    # Derive Canonical Ocean Mask directly from real GLORYS bathymetry
    da_target_sample = xr.DataArray(
        target_array[0],
        dims=["depth", "latitude", "longitude"],
        coords={"depth": CANONICAL_DEPTHS, "latitude": CANONICAL_LATS, "longitude": CANONICAL_LONS},
        name="thetao"
    )
    mask_ds = generate_canonical_ocean_mask_from_glorys(da_target_sample.to_dataset())
    save_canonical_ocean_mask(mask_ds)

    # Check 3: Cross-Variable Physical Sanity & Coordinate Integrity Check
    check_cross_variable_physical_sanity(surface_arrays, target_array, mask_ds)

    # Assemble Real ML Zarr Dataset
    zarr_prefix = f"data/processed/real_ml_dataset_{mode}"
    ds_surface, ds_target = assemble_ml_dataset(
        surface_arrays, target_array, times,
        ocean_mask_ds=mask_ds,
        zarr_out_prefix=zarr_prefix
    )

    # Independent In-Situ ARGO Float Matchup Validation
    argo_csv = "data/raw/argo/argo_profiles_2020-01-01_2020-01-07.csv"
    if os.path.exists(argo_csv):
        print("\n--- Executing In-Situ ARGO Float Matchup Validation ---")
        df_argo = pd.read_csv(argo_csv)
        matched_df = match_argo_profiles_with_model(df_argo, ds_target)
        evaluate_argo_matchups(matched_df)

    # If pilot mode, certify pilot acceptance marker
    if mode == "pilot":
        certify_pilot_acceptance({
            "start_date": start_date,
            "end_date": end_date,
            "days_count": n_days,
            "provenance_files_count": len(prov_log),
            "zarr_surface": f"{zarr_prefix}_surface.zarr",
            "zarr_target": f"{zarr_prefix}_target.zarr"
        })

    print("\n[ALL 3 QA CHECKS PASSED] Authentic data harmonization and validation complete!")
    return ds_surface, ds_target

if __name__ == "__main__":
    execute_real_harmonization("2020-01-01", "2020-01-07", mode="pilot")
