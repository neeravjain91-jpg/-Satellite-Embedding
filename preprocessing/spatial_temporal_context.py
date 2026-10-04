"""
preprocessing/spatial_temporal_context.py
Builds genuine spatial, temporal, and spatiotemporal context tensors
from the canonical gridded surface arrays for B6, B7, and B8 models.

This module resolves the data pipeline BLOCKER where _prepare_patches,
_prepare_sequences, and _prepare_cubes replicated pointwise features
instead of extracting genuine multi-pixel / multi-timestep context.

Usage:
    After load_tabular_dataset() returns split dicts, call:
        augment_split_with_spatial_patches(split_dict, surf_all, scaler, ...)
        augment_split_with_temporal_sequences(split_dict, surf_all, scaler, ...)
        augment_split_with_spatiotemporal_cubes(split_dict, surf_all, scaler, ...)
    This adds 'X_patches', 'X_seq', or 'X_cubes' keys that the
    B6/B7/B8 _prepare_* methods already check for.
"""

import os
import sys
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_LATS, CANONICAL_LONS


def _lat_lon_to_grid_indices(lats, lons):
    """
    Converts latitude/longitude float arrays to canonical grid integer indices.
    lat_idx: index into CANONICAL_LATS (5.00 to 30.00, step 0.25, 101 points)
    lon_idx: index into CANONICAL_LONS (45.00 to 105.00, step 0.25, 241 points)
    """
    lat_idx = np.round((np.asarray(lats) - CANONICAL_LATS[0]) / 0.25).astype(np.int32)
    lon_idx = np.round((np.asarray(lons) - CANONICAL_LONS[0]) / 0.25).astype(np.int32)
    # Clip to valid range
    lat_idx = np.clip(lat_idx, 0, len(CANONICAL_LATS) - 1)
    lon_idx = np.clip(lon_idx, 0, len(CANONICAL_LONS) - 1)
    return lat_idx, lon_idx


def _normalize_grid(surf_all, mean_vec, std_vec):
    """
    Normalizes the full gridded surface array using training scaler statistics.
    surf_all: (T, H, W, C) raw surface features
    Returns: (T, H, W, C) normalized surface features
    """
    mean_vec = np.asarray(mean_vec, dtype=np.float32)
    std_vec = np.asarray(std_vec, dtype=np.float32)
    # Broadcast: (T, H, W, C) - (C,) / (C,)
    return (surf_all - mean_vec[np.newaxis, np.newaxis, np.newaxis, :]) / \
           std_vec[np.newaxis, np.newaxis, np.newaxis, :]


def build_spatial_patches(surf_norm, time_indices, lat_indices, lon_indices,
                          patch_size=3):
    """
    Extracts genuine P x P spatial patches from the normalized gridded array.

    For each sample (t, lat_idx, lon_idx), extracts the local spatial
    neighborhood of size patch_size x patch_size centered at (lat_idx, lon_idx)
    from the surface field at time step t.

    Boundary handling: numpy 'edge' padding (replicate border values).

    Parameters:
        surf_norm: (T, H, W, C) normalized surface array
        time_indices: (N,) integer day indices
        lat_indices: (N,) integer latitude grid indices
        lon_indices: (N,) integer longitude grid indices
        patch_size: spatial patch size (default 3)

    Returns:
        patches: (N, C, P, P) float32 array with genuine spatial variation
    """
    T_total, H, W, C = surf_norm.shape
    pad = patch_size // 2
    N = len(time_indices)

    # Pad the spatial dimensions with edge values: (T, H+2*pad, W+2*pad, C)
    surf_padded = np.pad(
        surf_norm,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="edge"
    )

    t = np.asarray(time_indices, dtype=np.int32)
    li = np.asarray(lat_indices, dtype=np.int32) + pad
    lo = np.asarray(lon_indices, dtype=np.int32) + pad

    patches = np.empty((N, C, patch_size, patch_size), dtype=np.float32)

    for dy in range(patch_size):
        for dx in range(patch_size):
            # Vectorized extraction of (N, C) slice
            slice_nc = surf_padded[t, li + dy - pad, lo + dx - pad, :]
            patches[:, :, dy, dx] = slice_nc

    return np.nan_to_num(patches, nan=0.0)


def build_temporal_sequences(surf_norm, time_indices, lat_indices, lon_indices,
                             window_size=5,
                             purge_intervals=((253, 258), (307, 312))):
    """
    Builds genuine causal temporal sequences from the normalized gridded array.

    For each sample (t, lat_idx, lon_idx), constructs a window of
    [t - window_size + 1, ..., t] at the SAME spatial location,
    using the actual surface features from each past day.

    Purge-boundary aware: sequences never cross purge intervals.
    If the window would reach into a purge zone or a different partition,
    the earliest valid day is replicated to fill the remaining positions.

    Parameters:
        surf_norm: (T, H, W, C) normalized surface array
        time_indices: (N,) integer day indices
        lat_indices: (N,) integer latitude grid indices
        lon_indices: (N,) integer longitude grid indices
        window_size: temporal window size (default 5)
        purge_intervals: tuple of (start, end) purge day ranges

    Returns:
        sequences: (N, window_size, C) float32 array with genuine temporal variation
    """
    T_total, H, W, C = surf_norm.shape
    N = len(time_indices)
    sequences = np.empty((N, window_size, C), dtype=np.float32)

    t = np.asarray(time_indices, dtype=np.int32)
    li = np.asarray(lat_indices, dtype=np.int32)
    lo = np.asarray(lon_indices, dtype=np.int32)

    part_start = np.zeros(N, dtype=np.int32)
    val_mask = (t >= 259) & (t <= 306)
    test_mask = (t >= 313) & (t <= 365)
    other_mask = ~((t <= 252) | val_mask | test_mask)
    part_start[val_mask] = 259
    part_start[test_mask] = 313
    part_start[other_mask] = t[other_mask]

    for w in range(window_size):
        day_offset = t - (window_size - 1 - w)
        day_offset = np.maximum(day_offset, part_start)
        day_offset = np.clip(day_offset, 0, T_total - 1)

        for p_start, p_end in purge_intervals:
            in_purge = (day_offset >= p_start) & (day_offset <= p_end)
            if np.any(in_purge):
                day_offset[in_purge] = t[in_purge]

        sequences[:, w, :] = surf_norm[day_offset, li, lo, :]

    return np.nan_to_num(sequences, nan=0.0)


def build_spatiotemporal_cubes(surf_norm, time_indices, lat_indices, lon_indices,
                              patch_size=3, window_size=5,
                              purge_intervals=((253, 258), (307, 312))):
    """
    Builds genuine spatiotemporal cubes [N, T, C, P, P] from the gridded array.

    Combines spatial patch extraction and temporal sequence construction:
    for each sample (t, lat_idx, lon_idx), extracts a P x P spatial patch
    at each of the T past time steps [t - window_size + 1, ..., t].

    Parameters:
        surf_norm: (T, H, W, C) normalized surface array
        time_indices, lat_indices, lon_indices: (N,) integer arrays
        patch_size: spatial patch size (default 3)
        window_size: temporal window size (default 5)
        purge_intervals: purge day ranges

    Returns:
        cubes: (N, window_size, C, patch_size, patch_size) float32 array
    """
    T_total, H, W, C = surf_norm.shape
    pad = patch_size // 2
    N = len(time_indices)

    surf_padded = np.pad(
        surf_norm,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="edge"
    )

    t = np.asarray(time_indices, dtype=np.int32)
    li = np.asarray(lat_indices, dtype=np.int32) + pad
    lo = np.asarray(lon_indices, dtype=np.int32) + pad

    part_start = np.zeros(N, dtype=np.int32)
    val_mask = (t >= 259) & (t <= 306)
    test_mask = (t >= 313) & (t <= 365)
    other_mask = ~((t <= 252) | val_mask | test_mask)
    part_start[val_mask] = 259
    part_start[test_mask] = 313
    part_start[other_mask] = t[other_mask]

    cubes = np.empty((N, window_size, C, patch_size, patch_size), dtype=np.float32)

    for w in range(window_size):
        day_offset = t - (window_size - 1 - w)
        day_offset = np.maximum(day_offset, part_start)
        day_offset = np.clip(day_offset, 0, T_total - 1)

        for p_start, p_end in purge_intervals:
            in_purge = (day_offset >= p_start) & (day_offset <= p_end)
            if np.any(in_purge):
                day_offset[in_purge] = t[in_purge]

        for dy in range(patch_size):
            for dx in range(patch_size):
                cubes[:, w, :, dy, dx] = surf_padded[day_offset, li + dy - pad, lo + dx - pad, :]

    return np.nan_to_num(cubes, nan=0.0)


def augment_split_with_spatial_patches(split_data, surf_all, scaler,
                                       patch_size=3):
    """
    Augments a tabular split dictionary with genuine spatial patches.
    Adds 'X_patches' key of shape (N, C, P, P).

    Parameters:
        split_data: dict with 'time_idx', 'lat', 'lon' keys
        surf_all: (T_total, H, W, C) raw gridded surface array
        scaler: dict with 'mean' and 'std' lists
        patch_size: spatial patch size
    """
    surf_norm = _normalize_grid(surf_all,
                                np.array(scaler["mean"], dtype=np.float32),
                                np.array(scaler["std"], dtype=np.float32))

    lat_idx, lon_idx = _lat_lon_to_grid_indices(split_data["lat"],
                                                 split_data["lon"])

    split_data["X_patches"] = build_spatial_patches(
        surf_norm, split_data["time_idx"], lat_idx, lon_idx,
        patch_size=patch_size
    )
    return split_data


def augment_split_with_temporal_sequences(split_data, surf_all, scaler,
                                           window_size=5,
                                           purge_intervals=((253, 258), (307, 312))):
    """
    Augments a tabular split dictionary with genuine temporal sequences.
    Adds 'X_seq' key of shape (N, T, C).
    """
    surf_norm = _normalize_grid(surf_all,
                                np.array(scaler["mean"], dtype=np.float32),
                                np.array(scaler["std"], dtype=np.float32))

    lat_idx, lon_idx = _lat_lon_to_grid_indices(split_data["lat"],
                                                 split_data["lon"])

    split_data["X_seq"] = build_temporal_sequences(
        surf_norm, split_data["time_idx"], lat_idx, lon_idx,
        window_size=window_size,
        purge_intervals=purge_intervals
    )
    return split_data


def augment_split_with_spatiotemporal_cubes(split_data, surf_all, scaler,
                                             patch_size=3, window_size=5,
                                             purge_intervals=((253, 258), (307, 312))):
    """
    Augments a tabular split dictionary with genuine spatiotemporal cubes.
    Adds 'X_cubes' key of shape (N, T, C, P, P).
    """
    surf_norm = _normalize_grid(surf_all,
                                np.array(scaler["mean"], dtype=np.float32),
                                np.array(scaler["std"], dtype=np.float32))

    lat_idx, lon_idx = _lat_lon_to_grid_indices(split_data["lat"],
                                                 split_data["lon"])

    split_data["X_cubes"] = build_spatiotemporal_cubes(
        surf_norm, split_data["time_idx"], lat_idx, lon_idx,
        patch_size=patch_size, window_size=window_size,
        purge_intervals=purge_intervals
    )
    return split_data
