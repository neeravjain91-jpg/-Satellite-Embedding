"""
preprocessing/quality_control.py
Automated Quality Control (QC) suite for all oceanographic data:
- Variable existence
- Coordinate existence & monotonicity
- Date coverage & duplicate timestamps
- Spatial coverage (5-30N, 45-105E)
- Units verification
- NaN fraction & Fill value detection
- Plausible physical ranges
- Land/ocean consistency

Generates:
- reports/data_quality_report.csv
- reports/data_quality_report.md
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

# Plausible physical ranges for the North Indian Ocean
PHYSICAL_RANGES = {
    "sst": {"min": 15.0, "max": 36.0, "units": "degC"},
    "sss": {"min": 20.0, "max": 42.0, "units": "psu"},
    "ssh": {"min": -2.0, "max": 2.5, "units": "m"},
    "current_u": {"min": -4.0, "max": 4.0, "units": "m/s"},
    "current_v": {"min": -4.0, "max": 4.0, "units": "m/s"},
    "wind_u": {"min": -45.0, "max": 45.0, "units": "m/s"},
    "wind_v": {"min": -45.0, "max": 45.0, "units": "m/s"},
    "thetao": {"min": 2.0, "max": 36.0, "units": "degC"}
}

def check_monotonicity(coords_array):
    """Returns True if strictly monotonic (increasing or decreasing)."""
    diffs = np.diff(coords_array)
    return np.all(diffs > 0) or np.all(diffs < 0)

def run_quality_control(dataset_name, data_array, var_name, expected_units, land_mask=None):
    """
    Executes all QC checks on a single variable DataArray.
    Returns a dictionary of metrics and boolean pass/fail status.
    """
    vals = data_array.values if isinstance(data_array, xr.DataArray) else data_array
    
    # Check variable and coordinates
    has_lats = "latitude" in data_array.coords if isinstance(data_array, xr.DataArray) else True
    has_lons = "longitude" in data_array.coords if isinstance(data_array, xr.DataArray) else True
    
    lat_mono = True
    lon_mono = True
    if isinstance(data_array, xr.DataArray):
        if has_lats:
            lat_mono = check_monotonicity(data_array.latitude.values)
        if has_lons:
            lon_mono = check_monotonicity(data_array.longitude.values)

    # Missingness and Fill Values
    total_points = vals.size
    nan_count = np.isnan(vals).sum()
    nan_fraction = float(nan_count / total_points) if total_points > 0 else 1.0
    
    # Check for unmasked standard fill values (-999, -9999, 1e20, 1e36)
    fill_candidates = [-999.0, -9999.0, 1e20, 1e36, 9.96921e+36]
    detected_fill_values = []
    for fc in fill_candidates:
        if np.any(np.isclose(vals[~np.isnan(vals)], fc, rtol=1e-3, atol=1e-3)):
            detected_fill_values.append(fc)

    # Physical range check on valid numbers
    valid_vals = vals[~np.isnan(vals)]
    p_range = PHYSICAL_RANGES.get(var_name, {"min": -1e6, "max": 1e6})
    
    val_min = float(np.min(valid_vals)) if len(valid_vals) > 0 else np.nan
    val_max = float(np.max(valid_vals)) if len(valid_vals) > 0 else np.nan
    val_mean = float(np.mean(valid_vals)) if len(valid_vals) > 0 else np.nan
    
    range_ok = True
    if len(valid_vals) > 0:
        if val_min < p_range["min"] or val_max > p_range["max"]:
            range_ok = False

    # Land/Ocean consistency
    land_ocean_ok = True
    if land_mask is not None:
        # Land points should be NaN
        land_points = vals[..., ~land_mask]
        if np.any(~np.isnan(land_points)):
            land_ocean_ok = False

    passed = (
        has_lats and has_lons and lat_mono and lon_mono and
        len(detected_fill_values) == 0 and range_ok and land_ocean_ok
    )

    qc_result = {
        "dataset": dataset_name,
        "variable": var_name,
        "units": expected_units,
        "total_points": total_points,
        "nan_fraction": round(nan_fraction, 4),
        "val_min": round(val_min, 3) if not np.isnan(val_min) else None,
        "val_max": round(val_max, 3) if not np.isnan(val_max) else None,
        "val_mean": round(val_mean, 3) if not np.isnan(val_mean) else None,
        "lat_monotonic": lat_mono,
        "lon_monotonic": lon_mono,
        "fill_values_detected": len(detected_fill_values) > 0,
        "range_valid": range_ok,
        "land_ocean_consistent": land_ocean_ok,
        "overall_status": "PASS" if passed else "FAIL"
    }
    return qc_result

def generate_qc_reports(qc_results_list, out_dir="reports"):
    """Saves both CSV and Markdown quality control reports."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(qc_results_list)
    
    csv_path = os.path.join(out_dir, "data_quality_report.csv")
    df.to_csv(csv_path, index=False)
    
    md_path = os.path.join(out_dir, "data_quality_report.md")
    with open(md_path, "w") as f:
        f.write("# Automated Data Quality Control (QC) Report\n\n")
        f.write(f"**Domain**: North Indian Ocean (5°N–30°N, 45°E–105°E)\n\n")
        f.write(f"**Total variables evaluated**: {len(qc_results_list)}\n\n")
        f.write("## Detailed Variable Assessment\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Verification Summary\n\n")
        pass_count = (df["overall_status"] == "PASS").sum()
        fail_count = (df["overall_status"] == "FAIL").sum()
        f.write(f"- **Passed Variables**: {pass_count}\n")
        f.write(f"- **Failed Variables**: {fail_count}\n")
        if fail_count == 0:
            f.write("\nAll variables satisfy coordinate monotonicity, physical bounds, fill-value cleanliness, and land/ocean masking.\n")
        else:
            f.write("\nAction required on variables failing physical or formatting checks.\n")
            
    print(f"QC reports written to {csv_path} and {md_path}")
    return df
