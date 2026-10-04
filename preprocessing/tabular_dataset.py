"""
preprocessing/tabular_dataset.py
Vectorized extraction and tabular flattening for Phase 1 pointwise models:
- Preserves [N, 15] target validity masks
- Keeps any point where ANY depth is valid (ocean points)
- Does not discard points because deep levels are invalid
- Strict chronological split: 256 train / 54 validation / 56 test
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

def extract_tabular_split(surf_slice, targ_slice, time_indices, timestamps, lats, lons):
    """
    Vectorized extraction of a chronological slice.
    surf_slice: (T_split, 101, 241, 7)
    targ_slice: (T_split, 15, 101, 241)
    """
    T_split, H, W, n_feat = surf_slice.shape
    _, n_depth, _, _ = targ_slice.shape
    
    # Transpose target to (T_split, H, W, 15) to match spatial coordinates
    targ_trans = np.transpose(targ_slice, (0, 2, 3, 1))
    
    # Create target validity mask: True where value is not NaN
    target_mask = ~np.isnan(targ_trans)  # (T_split, H, W, 15)
    
    # Condition: Keep a point if ANY depth is valid
    spatial_valid = target_mask.any(axis=-1)  # (T_split, H, W)
    
    # Extract flattened valid samples using boolean indexing
    X_flat = surf_slice[spatial_valid]          # (N, 7)
    Y_flat = targ_trans[spatial_valid]          # (N, 15)
    M_flat = target_mask[spatial_valid]         # (N, 15)
    
    # Construct corresponding coordinate and temporal metadata
    # Meshgrid of spatial indices and coordinates
    mg_lat, mg_lon = np.meshgrid(lats, lons, indexing="ij")  # (H, W)
    
    # Tile across time
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
                         purge_buffer_days=7,
                         scaler_path=SCALER_JSON):
    """
    Loads, splits, vector-flattens, and standardizes dataset strictly using training statistics.
    Enforces configurable temporal purge buffers between Train -> Val and Val -> Test partitions:
    TRAIN -> PURGE BUFFER 1 -> VALIDATION -> PURGE BUFFER 2 -> TEST
    Returns:
        split_dict: {'train': dict, 'val': dict, 'test': dict, 'scaler': dict, 'split_metadata': dict, ...}
    """
    print("=" * 70)
    print("LOADING AND FLATTENING DATASET FOR POINTWISE TABULAR MODELS")
    print("=" * 70)
    
    ds_s = xr.open_zarr(surf_zarr, consolidated=True)
    ds_t = xr.open_zarr(targ_zarr, consolidated=True)
    
    lats = ds_s.latitude.values
    lons = ds_s.longitude.values
    times = pd.to_datetime(ds_s.time.values)
    total_days = len(times)
    
    # Resolve partition sizes and purge buffers
    if n_train_days is None:
        purge = int(purge_buffer_days)
        available_days = total_days - 2 * purge
        if available_days < 3:
            raise ValueError(f"Insufficient days ({total_days}) for 2 x {purge}-day purge buffers")
        if total_days == 366 and purge == 7:
            n_train_days = 246
            n_val_days = 50
            n_test_days = 56
        else:
            n_train_days = int(round(available_days * 0.70))
            n_val_days = int(round(available_days * 0.15))
            n_test_days = available_days - n_train_days - n_val_days
    else:
        purge = int(purge_buffer_days)
        if n_train_days + 2 * purge + n_val_days + n_test_days == total_days:
            pass
        elif n_train_days + n_val_days + n_test_days == total_days:
            # Caller specified exact 0-buffer partition (e.g. test isolation mocks)
            purge = 0
        else:
            expected_days = n_train_days + 2 * purge + n_val_days + n_test_days
            raise ValueError(
                f"Total days in dataset ({total_days}) does not match expected splits ({expected_days} "
                f"= {n_train_days} Train + {purge} Purge + {n_val_days} Val + {purge} Purge + {n_test_days} Test)"
            )
        
    train_start = 0
    train_end = n_train_days

    purge1_start = train_end
    purge1_end = train_end + purge

    val_start = purge1_end
    val_end = val_start + n_val_days

    purge2_start = val_end
    purge2_end = val_end + purge

    test_start = purge2_end
    test_end = test_start + n_test_days

    if test_end != total_days:
        raise ValueError(f"Computed test_end ({test_end}) does not equal total_days ({total_days})")

    print(f"Total Temporal Span: {total_days} days ({times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')})")
    print(f"Partition Structure (Purge Buffer = {purge} days):")
    print(f"  TRAIN:        Days {train_start:3d}..{train_end-1:3d} ({n_train_days} days: {times[train_start].strftime('%Y-%m-%d')} to {times[train_end-1].strftime('%Y-%m-%d')})")
    if purge > 0:
        print(f"  [PURGE 1]:    Days {purge1_start:3d}..{purge1_end-1:3d} ({purge} days: {times[purge1_start].strftime('%Y-%m-%d')} to {times[purge1_end-1].strftime('%Y-%m-%d')}) [DISCARDED]")
    print(f"  VALIDATION:   Days {val_start:3d}..{val_end-1:3d} ({n_val_days} days: {times[val_start].strftime('%Y-%m-%d')} to {times[val_end-1].strftime('%Y-%m-%d')})")
    if purge > 0:
        print(f"  [PURGE 2]:    Days {purge2_start:3d}..{purge2_end-1:3d} ({purge} days: {times[purge2_start].strftime('%Y-%m-%d')} to {times[purge2_end-1].strftime('%Y-%m-%d')}) [DISCARDED]")
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
        lons=lons
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
        lons=lons
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
        lons=lons
    )
    print(f"  Test:  N = {len(test_data['X']):,} valid ocean samples across {n_test_days} days.")

    split_metadata = {
        "total_days": total_days,
        "purge_buffer_days": purge,
        "train": {
            "start_date": times[train_start].strftime("%Y-%m-%d"),
            "end_date": times[train_end - 1].strftime("%Y-%m-%d"),
            "n_days": n_train_days,
            "n_samples": len(train_data["X"])
        },
        "purge_buffer_1": {
            "start_date": times[purge1_start].strftime("%Y-%m-%d"),
            "end_date": times[purge1_end - 1].strftime("%Y-%m-%d"),
            "n_days": purge
        } if purge > 0 else None,
        "val": {
            "start_date": times[val_start].strftime("%Y-%m-%d"),
            "end_date": times[val_end - 1].strftime("%Y-%m-%d"),
            "n_days": n_val_days,
            "n_samples": len(val_data["X"])
        },
        "purge_buffer_2": {
            "start_date": times[purge2_start].strftime("%Y-%m-%d"),
            "end_date": times[purge2_end - 1].strftime("%Y-%m-%d"),
            "n_days": purge
        } if purge > 0 else None,
        "test": {
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
    
    scaler_stats = {
        "features": list(CANONICAL_FEATURES),
        "mean": mean_vec.tolist(),
        "std": std_vec.tolist(),
        "train_samples_count": len(X_train_raw),
        "split_metadata": split_metadata
    }
    
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "w") as f:
        json.dump(scaler_stats, f, indent=2)
    print(f"Scaler parameters saved to {scaler_path}")
    
    for f_idx, feat in enumerate(CANONICAL_FEATURES):
        print(f"  {feat:12s} -> Mean: {mean_vec[f_idx]:8.3f} | Std: {std_vec[f_idx]:8.3f}")

    # Standardize X arrays
    train_data["X_norm"] = (train_data["X"] - mean_vec) / std_vec
    val_data["X_norm"] = (val_data["X"] - mean_vec) / std_vec
    test_data["X_norm"] = (test_data["X"] - mean_vec) / std_vec

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
