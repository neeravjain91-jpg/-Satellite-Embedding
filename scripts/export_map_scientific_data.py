"""
scripts/export_map_scientific_data.py
Exports production-ready real scientific geospatial data slices from the
full-year 2020 Zarr archives into public/data/daily/*.bin and public/data/ocean_manifest.json.

Zero synthetic data. All values are direct extractions from:
- real_ml_dataset_full_year_surface.zarr
- real_ml_dataset_full_year_target.zarr
- results/B2.json (Multi-Output Ridge coefficients)
- data/metadata/tabular_scaler_stats.json
"""

import os
import sys
import json
import zarr
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def export_data():
    print("=" * 70)
    print("EXPORTING GEOSPATIAL MAP SCIENTIFIC DATA ARTIFACTS")
    print("=" * 70)

    surf_path = os.path.join(repo_root, "data", "processed", "real_ml_dataset_full_year_surface.zarr")
    targ_path = os.path.join(repo_root, "data", "processed", "real_ml_dataset_full_year_target.zarr")
    scaler_path = os.path.join(repo_root, "data", "metadata", "tabular_scaler_stats.json")
    b2_path = os.path.join(repo_root, "results", "B2.json")

    surf = zarr.open(surf_path)
    targ = zarr.open(targ_path)
    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    with open(b2_path, "r") as f:
        b2 = json.load(f)

    geo_mask = surf["geographic_ocean_mask"][:] # (101, 241)
    ocean_y, ocean_x = np.where(geo_mask)
    n_ocean = len(ocean_y)
    print(f"Ocean points: {n_ocean} / {geo_mask.size} ({geo_mask.shape})")

    # Grid to ocean index mapping (-1 for land, 0..12037 for ocean)
    grid_to_ocean = np.full((101, 241), -1, dtype=np.int32)
    for idx, (y, x) in enumerate(zip(ocean_y, ocean_x)):
        grid_to_ocean[y, x] = idx

    # Deepest valid depth
    deepest_raw = targ["deepest_valid_depth_m"][:] # (101, 241)
    deepest_ocean = [
        int(deepest_raw[y, x]) if not np.isnan(deepest_raw[y, x]) else 0
        for y, x in zip(ocean_y, ocean_x)
    ]

    canonical_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    # Select representative dates
    # 1. Daily sequence in Test Split (Days 313 to 326: 2020-11-09 to 2020-11-22)
    # 2. Test milestones: 2020-12-01 (335), 2020-12-15 (349), 2020-12-31 (365)
    # 3. Validation milestones: 2020-09-16 (259), 2020-10-01 (274), 2020-10-15 (288), 2020-11-02 (306)
    # 4. Seasonal train milestones: 2020-01-15 (14), 2020-04-15 (105), 2020-07-15 (196), 2020-09-01 (244)
    target_day_indices = [
        14,   # 2020-01-15: Winter NE Monsoon
        105,  # 2020-04-15: Spring Transition
        196,  # 2020-07-15: Summer SW Monsoon Peak
        244,  # 2020-09-01: Late Summer
        259,  # 2020-09-16: Validation Start
        274,  # 2020-10-01: Post-Monsoon Transition
        288,  # 2020-10-15: Validation Mid
        306,  # 2020-11-02: Validation End
    ]
    # Consecutive test sequence: 313 to 326 (14 days)
    for d in range(313, 327):
        target_day_indices.append(d)
    # Additional test milestones
    target_day_indices.extend([335, 349, 365])
    target_day_indices = sorted(list(set(target_day_indices)))

    out_dir = os.path.join(repo_root, "public", "data")
    daily_dir = os.path.join(out_dir, "daily")
    os.makedirs(daily_dir, exist_ok=True)

    date_entries = []

    print(f"Exporting {len(target_day_indices)} daily binary slices...")
    for day_idx in target_day_indices:
        # Date string from Day 0 = 2020-01-01
        dt = np.datetime64("2020-01-01") + np.timedelta64(day_idx, "D")
        date_str = str(dt)

        if day_idx < 253:
            split_name = "train"
        elif day_idx <= 258:
            split_name = "purge1"
        elif day_idx <= 306:
            split_name = "validation"
        elif day_idx <= 312:
            split_name = "purge2"
        else:
            split_name = "test"

        date_entries.append({
            "date": date_str,
            "dayIdx": int(day_idx),
            "split": split_name,
            "label": f"{date_str} (Day {day_idx} · {split_name.capitalize()})"
        })

        s_feat = surf["surface_features"][day_idx] # (101, 241, 7)
        ref_t = targ["thetao"][day_idx]           # (15, 101, 241)

        # 22 channels: 7 surface + 15 reference thetao depths
        buf = np.full((22, n_ocean), -32768, dtype=np.int16)

        # Helper to scale safely and encode NaNs as -32768
        def pack_field(raw_vals, scale):
            valid = ~np.isnan(raw_vals)
            res = np.full(len(raw_vals), -32768, dtype=np.int16)
            res[valid] = np.clip(np.round(raw_vals[valid] * scale), -32767, 32767).astype(np.int16)
            return res

        # Surface variables
        buf[0] = pack_field(s_feat[ocean_y, ocean_x, 0], 100)   # SST * 100
        buf[1] = pack_field(s_feat[ocean_y, ocean_x, 1], 100)   # SSS * 100
        buf[2] = pack_field(s_feat[ocean_y, ocean_x, 2], 1000)  # SSH * 1000
        buf[3] = pack_field(s_feat[ocean_y, ocean_x, 3], 1000)  # CurU * 1000
        buf[4] = pack_field(s_feat[ocean_y, ocean_x, 4], 1000)  # CurV * 1000
        buf[5] = pack_field(s_feat[ocean_y, ocean_x, 5], 100)   # WindU * 100
        buf[6] = pack_field(s_feat[ocean_y, ocean_x, 6], 100)   # WindV * 100

        # Reference thetao at 15 depths
        for d_idx in range(15):
            buf[7 + d_idx] = pack_field(ref_t[d_idx, ocean_y, ocean_x], 100) # thetao * 100

        bin_filename = os.path.join(daily_dir, f"{date_str}.bin")
        buf.tofile(bin_filename)

    print(f"Exported {len(date_entries)} binary daily files to {daily_dir}.")

    # Manifest metadata
    manifest = {
        "domain": {
            "latMin": 5.0,
            "latMax": 30.0,
            "lonMin": 45.0,
            "lonMax": 105.0,
            "latStep": 0.25,
            "lonStep": 0.25,
            "nLat": 101,
            "nLon": 241,
            "nOcean": n_ocean
        },
        "canonicalDepths": canonical_depths,
        "availableDates": date_entries,
        "defaultDate": "2020-11-09",
        "defaultDepth": 0,
        "scaler": {
            "features": scaler["features"],
            "mean": scaler["mean"],
            "std": scaler["std"]
        },
        "b2Coefficients": b2["coefficients"],
        "oceanCoords": {
            "y": ocean_y.tolist(),
            "x": ocean_x.tolist()
        },
        "gridToOcean": grid_to_ocean.tolist(),
        "deepestValidDepth": deepest_ocean,
        "coverage": {
            "domain": "North Indian Ocean (5.0°N–30.0°N, 45.0°E–105.0°E)",
            "gridResolution": "0.25° x 0.25° (101 latitude x 241 longitude points)",
            "oceanPoints": n_ocean,
            "fullResearchDatasetTotalDays": 366,
            "fullDatasetPeriod": "2020-01-01 to 2020-12-31 (Leap Year 2020)",
            "publishedBenchmarkSlicesCount": len(date_entries),
            "publishedSlicesDescription": "25 curated benchmark dates spanning the official test split (Days 313–326), test milestones (Days 335, 349, 365), validation benchmarks (Days 259, 274, 288, 306), and seasonal monsoon milestones.",
            "canonicalStoreLocation": "data/processed/real_ml_dataset_full_year_surface.zarr"
        },
        "layers": {
            "sst": {
                "id": "sst",
                "name": "Sea Surface Temperature",
                "unit": "°C",
                "source": "UK Met Office OSTIA (Global L4 Reprocessed SST, 0.05° daily, CMEMS)",
                "min": 22.0,
                "max": 31.5,
                "is3D": False,
                "description": "Daily 0.05° reprocessed sea surface temperature from UK Met Office OSTIA regridded to canonical 0.25° grid.",
                "channelIdx": 0,
                "scale": 100.0
            },
            "sss": {
                "id": "sss",
                "name": "Sea Surface Salinity",
                "unit": "PSU",
                "source": "CMEMS Multi-Mission Global Analysed SSS & Density (Multi-Obs L4)",
                "min": 31.0,
                "max": 37.0,
                "is3D": False,
                "description": "Multi-satellite combined sea surface salinity resolving Bay of Bengal freshwater plumes and Arabian Sea high-salinity water.",
                "channelIdx": 1,
                "scale": 100.0
            },
            "ssh": {
                "id": "ssh",
                "name": "Sea Surface Height Anomaly",
                "unit": "m",
                "source": "SSALTO/DUACS Gridded Sea Surface Height & SLA (L4 Multi-Satellite Merged, CMEMS/CNES)",
                "min": -0.30,
                "max": 0.30,
                "is3D": False,
                "description": "Altimeter dynamic sea surface height anomaly capturing mesoscale eddies, coastal Kelvin waves, and Rossby waves.",
                "channelIdx": 2,
                "scale": 1000.0
            },
            "currents": {
                "id": "currents",
                "name": "Surface Ocean Currents",
                "unit": "m/s",
                "source": "OSCAR (Ocean Surface Current Analyses Real-time) Final 0.25° v2.0 (ESR/PO.DAAC)",
                "min": 0.0,
                "max": 1.2,
                "is3D": False,
                "isVector": True,
                "description": "Near-surface geostrophic and wind-driven ocean current velocity vectors (Eastward U and Northward V).",
                "uChannelIdx": 3,
                "vChannelIdx": 4,
                "scale": 1000.0
            },
            "winds": {
                "id": "winds",
                "name": "Surface Wind Velocity",
                "unit": "m/s",
                "source": "RSS CCMP (Cross-Calibrated Multi-Platform) v3.1 6-hourly 0.25° Level 4 (Remote Sensing Systems)",
                "min": 0.0,
                "max": 16.0,
                "is3D": False,
                "isVector": True,
                "description": "Surface 10m ocean wind velocity vectors driving monsoon circulation and upwelling.",
                "uChannelIdx": 5,
                "vChannelIdx": 6,
                "scale": 100.0
            },
            "thetao_ref": {
                "id": "thetao_ref",
                "name": "GLORYS12V1 Reference Subsurface Temp",
                "unit": "°C",
                "source": "Mercator Ocean GLORYS12V1 Global Numerical Ocean Reanalysis (CMEMS)",
                "min": 5.0,
                "max": 31.5,
                "is3D": True,
                "description": "Numerical ocean physics reanalysis reference field across 15 standard oceanographic depth levels.",
                "channelOffset": 7,
                "scale": 100.0
            },
            "b2_recon": {
                "id": "b2_recon",
                "name": "B2 Multi-Output Ridge Reconstruction",
                "unit": "°C",
                "source": "B2 Baseline Multi-Output Ridge Regression Model (alpha=100000)",
                "min": 5.0,
                "max": 31.5,
                "is3D": True,
                "description": "Subsurface potential temperature reconstructed purely from the 7 standardized surface satellite predictors.",
                "computed": True
            },
            "b2_residual": {
                "id": "b2_residual",
                "name": "B2 Absolute Residual Error",
                "unit": "°C",
                "source": "Direct residual |Reconstruction - GLORYS12V1 Reference|",
                "min": 0.0,
                "max": 3.5,
                "is3D": True,
                "description": "Pointwise absolute temperature discrepancy showing thermocline depth error distribution.",
                "computed": True
            }
        }
    }

    manifest_path = os.path.join(out_dir, "ocean_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest: {manifest_path} ({os.path.getsize(manifest_path) / 1024:.1f} KB)")
    print("=" * 70)

if __name__ == "__main__":
    export_data()
