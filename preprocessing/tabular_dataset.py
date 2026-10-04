"""
preprocessing/tabular_dataset.py
Vectorized extraction and tabular flattening for Phase 1 pointwise models:
- Preserves [N, 15] target validity masks
- Keeps any point where ANY depth is valid (ocean points)
- Does not discard points because deep levels are invalid
- Strict chronological split: 253 train (days 0–252) / 6 purge1 (days 253–258) / 48 validation (days 259–306) / 6 purge2 (days 307–312) / 53 test (days 313–365)
- Normalizes features strictly using training-split statistics (Zero Data Leakage)
- Preserves timestamp, time_idx, latitude, longitude, and target-validity mask
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_FEATURES, CANONICAL_DEPTHS

SURF_ZARR = "data/processed/ml_dataset_full-year_surface.zarr"
TARG_ZARR = "data/processed/ml_dataset_full-year_target.zarr"
SCALER_JSON = "data/metadata/tabular_scaler_stats.json"

import hashlib

def extract_tabular_split(surf_slice, targ_slice, time_indices, timestamps, lats, lons,
                          geo_ocean_2d=None, depth_valid_3d=None):
    """
    Vectorized extraction of a chronological slice enforcing unified 4-way mask:
    geographic_ocean_mask AND surface_validity_mask AND target_validity_mask AND depth_valid_mask
    surf_slice: (T_split, 101, 241, 7)
    targ_slice: (T_split, 15, 101, 241)
    """
    T_split, H, W, n_feat = surf_slice.shape
    _, n_depth, _, _ = targ_slice.shape
    
    # Transpose target to (T_split, H, W, 15) to match spatial coordinates
    targ_trans = np.transpose(targ_slice, (0, 2, 3, 1))
    
    # 1. Target validity mask: True where thetao is not NaN
    target_valid = ~np.isnan(targ_trans)  # (T_split, H, W, 15)
    
    # 2. Surface validity mask: True where all 7 features are valid
    surface_valid = ~np.isnan(surf_slice).any(axis=-1)  # (T_split, H, W)

    # 3. Geographic ocean mask (if not provided, default all true)
    if geo_ocean_2d is None:
        geo_ocean_2d = target_valid.any(axis=(0, -1)) # (H, W)

    # 4. Depth validity mask (if not provided, default all true)
    if depth_valid_3d is None:
        depth_valid_trans = target_valid.any(axis=0) # (H, W, 15)
    else:
        depth_valid_trans = np.transpose(depth_valid_3d, (1, 2, 0)) # (H, W, 15)

    # 4-way unified mask: (T_split, H, W, 15)
    unified_mask = (target_valid &
                    surface_valid[:, :, :, np.newaxis] &
                    depth_valid_trans[np.newaxis, :, :, :] &
                    geo_ocean_2d[np.newaxis, :, :, np.newaxis])
    
    # Condition: Keep a point if surface is valid, ocean is true, and ANY depth is valid
    spatial_valid = surface_valid & geo_ocean_2d[np.newaxis, :, :] & unified_mask.any(axis=-1)  # (T_split, H, W)
    
    # Extract flattened valid samples using boolean indexing
    X_flat = surf_slice[spatial_valid]          # (N, 7)
    Y_flat = targ_trans[spatial_valid]          # (N, 15)
    M_flat = unified_mask[spatial_valid]        # (N, 15)
    
    # Guarantee invalid targets outside M_flat are strictly NaN
    Y_flat = np.where(M_flat, Y_flat, np.nan)
    
    # Construct corresponding coordinate and temporal metadata
    mg_lat, mg_lon = np.meshgrid(lats, lons, indexing="ij")  # (H, W)
    time_idx_3d = np.repeat(time_indices[:, np.newaxis, np.newaxis], H, axis=1)
    time_idx_3d = np.repeat(time_idx_3d, W, axis=2)          # (T_split, H, W)
    
    lat_3d = np.tile(mg_lat[np.newaxis, :, :], (T_split, 1, 1)) # (T_split, H, W)
    lon_3d = np.tile(mg_lon[np.newaxis, :, :], (T_split, 1, 1)) # (T_split, H, W)
    
    timestamps_arr = np.array(timestamps)
    ts_3d = np.repeat(timestamps_arr[:, np.newaxis, np.newaxis], H, axis=1)
    ts_3d = np.repeat(ts_3d, W, axis=2)                      # (T_split, H, W)
    
    flat_time_idx = time_idx_3d[spatial_valid]
    flat_lats = lat_3d[spatial_valid]
    flat_lons = lon_3d[spatial_valid]
    flat_timestamps = ts_3d[spatial_valid]
    
    return {
        "X": X_flat.astype(np.float32),
        "Y": Y_flat.astype(np.float32),
        "mask": M_flat.astype(bool),
        "time_idx": flat_time_idx.astype(np.int32),
        "timestamp": flat_timestamps,
        "lat": flat_lats.astype(np.float32),
        "lon": flat_lons.astype(np.float32)
    }

def load_tabular_dataset(surf_zarr=SURF_ZARR, targ_zarr=TARG_ZARR,
                         n_train_days=None, n_val_days=None, n_test_days=None,
                         purge_buffer_days=6,
                         scaler_path=SCALER_JSON,
                         build_context=True):
    """
    Loads, splits, vector-flattens, and standardizes dataset strictly using training statistics.
    Enforces exact temporal partitions and purge buffers according to the locked scientific protocol:
    TRAIN  = days 0–252 (253 days: Jan 1 – Sep 9, 2020)
    PURGE1 = days 253–258 (6 days: Sep 10 – Sep 15, 2020) [DISCARDED]
    VAL    = days 259–306 (48 days: Sep 16 – Nov 2, 2020)
    PURGE2 = days 307–312 (6 days: Nov 3 – Nov 8, 2020) [DISCARDED]
    TEST   = days 313–365 (53 days: Nov 9 – Dec 31, 2020)
    
    Returns:
        split_dict: {'train': dict, 'val': dict, 'test': dict, 'scaler': dict, 'split_metadata': dict, ...}
    """
    print("=" * 70)
    print("LOADING AND FLATTENING DATASET FOR POINTWISE TABULAR MODELS")
    print("=" * 70)
    
    if not os.path.exists(surf_zarr):
        for candidate in ["data/processed/real_ml_dataset_full_year_surface.zarr", "data/processed/ml_dataset_full-year_surface.zarr"]:
            if os.path.exists(candidate):
                surf_zarr = candidate
                break
    if not os.path.exists(targ_zarr):
        for candidate in ["data/processed/real_ml_dataset_full_year_target.zarr", "data/processed/ml_dataset_full-year_target.zarr"]:
            if os.path.exists(candidate):
                targ_zarr = candidate
                break

    ds_s = xr.open_zarr(surf_zarr, consolidated=True)
    ds_t = xr.open_zarr(targ_zarr, consolidated=True)
    
    lats = ds_s.latitude.values
    lons = ds_s.longitude.values
    times = pd.to_datetime(ds_s.time.values)
    total_days = len(times)
    
    # Try loading canonical ocean masks if present
    geo_mask_2d = None
    depth_mask_3d = None
    mask_path = "data/processed/canonical_ocean_mask.nc"
    if os.path.exists(mask_path):
        try:
            ds_m = xr.open_dataset(mask_path)
            geo_mask_2d = ds_m["geographic_ocean_mask"].values
            depth_mask_3d = ds_m["depth_valid_mask"].values
        except Exception:
            pass

    # Resolve partition sizes and purge buffers according to protocol
    purge = int(purge_buffer_days)
    if total_days == 366 and n_train_days is None and purge == 6:
        # EXACT LOCKED SCIENTIFIC EXPERIMENT PROTOCOL PARTITION
        n_train_days = 253
        purge1 = 6
        n_val_days = 48
        purge2 = 6
        n_test_days = 53
    elif total_days == 366 and n_train_days is None and purge == 7:
        n_train_days = 246
        purge1 = 7
        n_val_days = 50
        purge2 = 7
        n_test_days = 56
    elif n_train_days is None:
        available_days = total_days - 2 * purge
        if available_days < 3:
            # Fallback for small pilot/unit test slices
            purge1 = 0
            purge2 = 0
            n_train_days = max(1, int(round(total_days * 0.70)))
            n_val_days = max(1, int(round(total_days * 0.15)))
            n_test_days = max(1, total_days - n_train_days - n_val_days)
            if n_train_days + n_val_days + n_test_days != total_days:
                n_train_days = total_days - n_val_days - n_test_days
        else:
            purge1 = purge
            purge2 = purge
            n_train_days = int(round(available_days * 0.70))
            n_val_days = int(round(available_days * 0.15))
            n_test_days = available_days - n_train_days - n_val_days
    else:
        if n_train_days + 2 * purge + n_val_days + n_test_days == total_days:
            purge1 = purge
            purge2 = purge
        elif n_train_days + n_val_days + n_test_days == total_days:
            # Caller specified exact 0-buffer partition (e.g. test isolation mocks)
            purge1 = 0
            purge2 = 0
        elif n_train_days + 6 + n_val_days + 6 + n_test_days == total_days:
            purge1 = 6
            purge2 = 6
        else:
            expected_days = n_train_days + 2 * purge + n_val_days + n_test_days
            raise ValueError(
                f"Total days in dataset ({total_days}) does not match expected splits ({expected_days} "
                f"= {n_train_days} Train + {purge} Purge + {n_val_days} Val + {purge} Purge + {n_test_days} Test)"
            )
        
    train_start = 0
    train_end = n_train_days

    purge1_start = train_end
    purge1_end = train_end + purge1

    val_start = purge1_end
    val_end = val_start + n_val_days

    purge2_start = val_end
    purge2_end = val_end + purge2

    test_start = purge2_end
    test_end = test_start + n_test_days

    if test_end != total_days:
        raise ValueError(f"Computed test_end ({test_end}) does not equal total_days ({total_days})")

    print(f"Total Temporal Span: {total_days} days ({times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')})")
    print(f"Partition Structure (Purge Buffers = {purge1}, {purge2} days):")
    print(f"  TRAIN:        Days {train_start:3d}..{train_end-1:3d} ({n_train_days} days: {times[train_start].strftime('%Y-%m-%d')} to {times[train_end-1].strftime('%Y-%m-%d')})")
    if purge1 > 0:
        print(f"  [PURGE 1]:    Days {purge1_start:3d}..{purge1_end-1:3d} ({purge1} days: {times[purge1_start].strftime('%Y-%m-%d')} to {times[purge1_end-1].strftime('%Y-%m-%d')}) [DISCARDED]")
    print(f"  VALIDATION:   Days {val_start:3d}..{val_end-1:3d} ({n_val_days} days: {times[val_start].strftime('%Y-%m-%d')} to {times[val_end-1].strftime('%Y-%m-%d')})")
    if purge2 > 0:
        print(f"  [PURGE 2]:    Days {purge2_start:3d}..{purge2_end-1:3d} ({purge2} days: {times[purge2_start].strftime('%Y-%m-%d')} to {times[purge2_end-1].strftime('%Y-%m-%d')}) [DISCARDED]")
    print(f"  TEST:         Days {test_start:3d}..{test_end-1:3d} ({n_test_days} days: {times[test_start].strftime('%Y-%m-%d')} to {times[test_end-1].strftime('%Y-%m-%d')})")
    
    # Read entire arrays into memory
    print("Extracting surface and target arrays...")
    surf_all = ds_s["surface_features"].values
    targ_all = ds_t["thetao"].values
    
    # 1. Train Split
    print("\nExtracting Training Split...")
    train_data = extract_tabular_split(
        surf_slice=surf_all[train_start:train_end],
        targ_slice=targ_all[train_start:train_end],
        time_indices=np.arange(train_start, train_end),
        timestamps=times[train_start:train_end],
        lats=lats,
        lons=lons,
        geo_ocean_2d=geo_mask_2d,
        depth_valid_3d=depth_mask_3d
    )
    print(f"  Train: N = {len(train_data['X']):,} valid ocean samples across {n_train_days} days.")
    
    # 2. Validation Split
    print("Extracting Validation Split...")
    val_data = extract_tabular_split(
        surf_slice=surf_all[val_start:val_end],
        targ_slice=targ_all[val_start:val_end],
        time_indices=np.arange(val_start, val_end),
        timestamps=times[val_start:val_end],
        lats=lats,
        lons=lons,
        geo_ocean_2d=geo_mask_2d,
        depth_valid_3d=depth_mask_3d
    )
    print(f"  Val:   N = {len(val_data['X']):,} valid ocean samples across {n_val_days} days.")
    
    # 3. Test Split
    print("Extracting Test Split...")
    test_data = extract_tabular_split(
        surf_slice=surf_all[test_start:test_end],
        targ_slice=targ_all[test_start:test_end],
        time_indices=np.arange(test_start, test_end),
        timestamps=times[test_start:test_end],
        lats=lats,
        lons=lons,
        geo_ocean_2d=geo_mask_2d,
        depth_valid_3d=depth_mask_3d
    )
    print(f"  Test:  N = {len(test_data['X']):,} valid ocean samples across {n_test_days} days.")

    # Programmatic Purge Leakage Verification
    train_ids = set(train_data["time_idx"])
    val_ids = set(val_data["time_idx"])
    test_ids = set(test_data["time_idx"])
    purge1_ids = set(range(purge1_start, purge1_end))
    purge2_ids = set(range(purge2_start, purge2_end))

    assert len(train_ids.intersection(purge1_ids)) == 0, "Purge 1 leaked into Train!"
    assert len(train_ids.intersection(purge2_ids)) == 0, "Purge 2 leaked into Train!"
    assert len(val_ids.intersection(purge1_ids)) == 0, "Purge 1 leaked into Validation!"
    assert len(val_ids.intersection(purge2_ids)) == 0, "Purge 2 leaked into Validation!"
    assert len(test_ids.intersection(purge1_ids)) == 0, "Purge 1 leaked into Test!"
    assert len(test_ids.intersection(purge2_ids)) == 0, "Purge 2 leaked into Test!"
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Validation overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Validation and Test overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap!"

    split_metadata = {
        "total_days": total_days,
        "purge_buffer_days": purge1,
        "purge1_days": purge1,
        "purge2_days": purge2,
        "train": {
            "start_idx": train_start,
            "end_idx": train_end - 1,
            "start_date": times[train_start].strftime("%Y-%m-%d"),
            "end_date": times[train_end - 1].strftime("%Y-%m-%d"),
            "n_days": n_train_days,
            "n_samples": len(train_data["X"])
        },
        "purge_buffer_1": {
            "start_idx": purge1_start,
            "end_idx": purge1_end - 1,
            "start_date": times[purge1_start].strftime("%Y-%m-%d"),
            "end_date": times[purge1_end - 1].strftime("%Y-%m-%d"),
            "n_days": purge1
        } if purge1 > 0 else None,
        "val": {
            "start_idx": val_start,
            "end_idx": val_end - 1,
            "start_date": times[val_start].strftime("%Y-%m-%d"),
            "end_date": times[val_end - 1].strftime("%Y-%m-%d"),
            "n_days": n_val_days,
            "n_samples": len(val_data["X"])
        },
        "purge_buffer_2": {
            "start_idx": purge2_start,
            "end_idx": purge2_end - 1,
            "start_date": times[purge2_start].strftime("%Y-%m-%d"),
            "end_date": times[purge2_end - 1].strftime("%Y-%m-%d"),
            "n_days": purge2
        } if purge2 > 0 else None,
        "test": {
            "start_idx": test_start,
            "end_idx": test_end - 1,
            "start_date": times[test_start].strftime("%Y-%m-%d"),
            "end_date": times[test_end - 1].strftime("%Y-%m-%d"),
            "n_days": n_test_days,
            "n_samples": len(test_data["X"])
        }
    }

    # 4. Strict Training-Only Normalization
    print("\nComputing Standard Scaler strictly on Training Split (Zero Leakage)...")
    X_train_raw = train_data["X"]
    
    mean_vec = np.nanmean(X_train_raw, axis=0)
    std_vec = np.nanstd(X_train_raw, axis=0)
    # Guard against zero variance
    std_vec[std_vec < 1e-6] = 1.0

    # Target statistics strictly on valid training points
    target_mean = np.zeros(len(CANONICAL_DEPTHS), dtype=np.float32)
    target_std = np.zeros(len(CANONICAL_DEPTHS), dtype=np.float32)
    for d_idx in range(len(CANONICAL_DEPTHS)):
        d_mask = train_data["mask"][:, d_idx]
        if np.any(d_mask):
            target_mean[d_idx] = float(np.nanmean(train_data["Y"][d_mask, d_idx]))
            target_std[d_idx] = float(np.nanstd(train_data["Y"][d_mask, d_idx]))
            if target_std[d_idx] < 1e-6:
                target_std[d_idx] = 1.0
        else:
            target_mean[d_idx] = 15.0
            target_std[d_idx] = 5.0
    
    scaler_stats = {
        "features": list(CANONICAL_FEATURES),
        "mean": mean_vec.tolist(),
        "std": std_vec.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "train_samples_count": len(X_train_raw),
        "split_metadata": split_metadata
    }

    stats_json_str = json.dumps(scaler_stats, sort_keys=True, indent=2)
    scaler_sha256 = hashlib.sha256(stats_json_str.encode("utf-8")).hexdigest()
    scaler_stats["scaler_sha256"] = scaler_sha256
    
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "w") as f:
        json.dump(scaler_stats, f, indent=2)
    print(f"Scaler parameters saved to {scaler_path} (SHA-256: {scaler_sha256[:16]}...)")
    
    for f_idx, feat in enumerate(CANONICAL_FEATURES):
        print(f"  {feat:12s} -> Mean: {mean_vec[f_idx]:8.3f} | Std: {std_vec[f_idx]:8.3f}")

    # Standardize X arrays
    train_data["X_norm"] = (train_data["X"] - mean_vec) / std_vec
    val_data["X_norm"] = (val_data["X"] - mean_vec) / std_vec
    test_data["X_norm"] = (test_data["X"] - mean_vec) / std_vec

    # 5. Automatically Construct Required Context Tensors (B6, B7, B8)
    if build_context:
        print("\n" + "=" * 70)
        print("CONSTRUCTING SPATIAL-TEMPORAL CONTEXT TENSORS (B6, B7, B8)")
        print("=" * 70)
        from preprocessing.spatial_temporal_context import (
            augment_split_with_spatial_patches,
            augment_split_with_temporal_sequences,
            augment_split_with_spatiotemporal_cubes,
        )
        p_intervals = []
        if purge1 > 0:
            p_intervals.append((purge1_start, purge1_end - 1))
        if purge2 > 0:
            p_intervals.append((purge2_start, purge2_end - 1))
        p_intervals = tuple(p_intervals)

        for sname, sdata in [("train", train_data), ("val", val_data), ("test", test_data)]:
            print(f"  Building spatial patches (3x3) for {sname} split...")
            augment_split_with_spatial_patches(sdata, surf_all, scaler_stats, patch_size=3)
            print(f"  Building temporal sequences (window=5) for {sname} split...")
            augment_split_with_temporal_sequences(sdata, surf_all, scaler_stats, window_size=5, purge_intervals=p_intervals)
            print(f"  Building spatiotemporal cubes (5x7x3x3) for {sname} split...")
            augment_split_with_spatiotemporal_cubes(sdata, surf_all, scaler_stats, patch_size=3, window_size=5, purge_intervals=p_intervals)

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "scaler": scaler_stats,
        "split_metadata": split_metadata,
        "feature_names": list(CANONICAL_FEATURES),
        "depth_levels": list(CANONICAL_DEPTHS)
    }

if __name__ == "__main__":
    dataset = load_tabular_dataset()
    print("\nSanity Check:")
    print("Train X_norm shape:", dataset["train"]["X_norm"].shape, "mean:", np.mean(dataset["train"]["X_norm"], axis=0).round(3))
    print("Train Y shape:", dataset["train"]["Y"].shape)
    print("Train mask shape:", dataset["train"]["mask"].shape)
    print("Validation samples:", dataset["val"]["X_norm"].shape[0])
    print("Test samples:", dataset["test"]["X_norm"].shape[0])
    print("[PASS] Tabular dataset extraction verified successfully.")

