"""
tests/test_scientific_masks_and_integrity.py
Scientific regression test suite for explicit mask separation and data integrity:

1. geographic_ocean_mask does not change because thetao is NaN at one depth
2. target_validity_mask correctly reflects thetao availability
3. shallow-water locations can be valid at 0–50 m but invalid at deeper depths
4. target NaNs never become zero during preprocessing
5. target masks preserve all valid depths independently
6. regridding does not incorrectly convert land to ocean
7. temporal alignment remains exact after multi-day chunk resolution
8. surface predictor missingness remains distinguishable from geographic land
9. dataset shapes remain correct across all surface and target arrays
10. all canonical depth levels are preserved exactly
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import (
    CANONICAL_LATS, CANONICAL_LONS, CANONICAL_DEPTHS, CANONICAL_FEATURES
)
from preprocessing.ocean_mask import (
    build_canonical_masks, create_target_validity_mask, create_surface_validity_mask
)
from preprocessing.depth_interpolation import interpolate_depths_1d, interpolate_glorys_to_canonical_depths
from preprocessing.regrid import regrid_2d_field
from preprocessing.build_dataset import assemble_ml_dataset, OceanReconstructionDataset
import importlib
pointwise_mlp_mod = importlib.import_module("models.05_pointwise_mlp")
masked_mse_loss = pointwise_mlp_mod.masked_mse_loss

class TestScientificMasksAndIntegrity(unittest.TestCase):
    """
    Validates explicit mask separation, bathymetric preservation,
    and anti-corruption rules across the scientific data pipeline.
    """

    def test_01_geographic_ocean_mask_invariant_to_single_depth_nan(self):
        """
        1. geographic_ocean_mask does NOT change because thetao is NaN at one depth.
        Even if depth 0m is NaN, if subsurface depths (e.g. 5m, 10m) are valid,
        the cell remains identified as geographic ocean.
        """
        # Miniature grid: 15 depths, 3 lats, 3 lons
        mock_thetao = np.full((15, 3, 3), np.nan, dtype=np.float32)
        # Coordinate (1, 1): Surface 0m is NaN, but 5m and 10m are valid ocean
        mock_thetao[1, 1, 1] = 28.5  # 5m
        mock_thetao[2, 1, 1] = 28.0  # 10m

        # Coordinate (0, 0): Permanent land (all depths NaN)
        # Coordinate (2, 2): All depths valid down to 1000m
        mock_thetao[:, 2, 2] = 20.0

        ds_masks = build_canonical_masks(mock_thetao)
        geo_mask = ds_masks["geographic_ocean_mask"].values

        # Coordinate (1, 1) MUST be geographic ocean despite 0m being NaN
        self.assertTrue(geo_mask[1, 1], "Cell with valid subsurface depths must remain geographic ocean!")
        # Coordinate (2, 2) is ocean
        self.assertTrue(geo_mask[2, 2])
        # Coordinate (0, 0) is land
        self.assertFalse(geo_mask[0, 0])

    def test_02_target_validity_mask_reflects_thetao_availability(self):
        """
        2. target_validity_mask correctly reflects thetao availability across time and depths.
        """
        t_steps = 3
        mock_thetao = np.random.randn(t_steps, 15, 5, 5).astype(np.float32)
        # Introduce targeted NaNs
        mock_thetao[0, 0, 1, 1] = np.nan
        mock_thetao[1, 5, 2, 2] = np.nan
        mock_thetao[2, 14, 4, 4] = np.nan

        targ_mask = create_target_validity_mask(mock_thetao)
        self.assertEqual(targ_mask.shape, mock_thetao.shape)
        self.assertFalse(targ_mask[0, 0, 1, 1])
        self.assertFalse(targ_mask[1, 5, 2, 2])
        self.assertFalse(targ_mask[2, 14, 4, 4])
        self.assertTrue(targ_mask[0, 1, 1, 1])
        self.assertTrue(np.array_equal(targ_mask, ~np.isnan(mock_thetao)))

    def test_03_shallow_water_depth_validity(self):
        """
        3. Shallow-water locations can be valid at 0–50 m but invalid at deeper depths.
        Deepest valid depth reflects bathymetry limit (no fake deep-water extrapolation).
        """
        # Miniature column: valid from 0 to 50m (depth indices 0..5), NaN below 50m
        profile = np.full(len(CANONICAL_DEPTHS), np.nan, dtype=np.float32)
        for d_idx, z in enumerate(CANONICAL_DEPTHS):
            if z <= 50.0:
                profile[d_idx] = 28.0 - 0.05 * z

        mock_3d = np.tile(profile[:, np.newaxis, np.newaxis], (1, 3, 3))
        ds_masks = build_canonical_masks(mock_3d)

        depth_mask = ds_masks["depth_valid_mask"].values  # (15, 3, 3)
        deepest = ds_masks["deepest_valid_depth_m"].values  # (3, 3)

        # Depths <= 50m are valid
        for d_idx, z in enumerate(CANONICAL_DEPTHS):
            if z <= 50.0:
                self.assertTrue(depth_mask[d_idx, 1, 1], f"Depth {z}m should be valid")
            else:
                self.assertFalse(depth_mask[d_idx, 1, 1], f"Depth {z}m should be invalid in shallow water")

        # Deepest valid depth is exactly 50.0m
        self.assertEqual(deepest[1, 1], 50.0)

    def test_04_target_nans_never_become_zero_during_preprocessing(self):
        """
        4. Target NaNs never become zero during preprocessing or dataset loading.
        thetao NaN -> 0 -> model target must NEVER happen.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            times = ["2020-01-01"]
            mock_surf = {feat: np.ones((1, 101, 241), dtype=np.float32) for feat in CANONICAL_FEATURES}
            # Target has NaNs at deep levels (e.g. 700m and 1000m)
            mock_targ = np.full((1, 15, 101, 241), 22.0, dtype=np.float32)
            mock_targ[0, 13:] = np.nan  # Depths 700m and 1000m are NaN

            mock_mask_ds = build_canonical_masks(mock_targ[0])
            zarr_prefix = os.path.join(tmp_dir, "test_dataset")

            ds_s, ds_t = assemble_ml_dataset(
                mock_surf, mock_targ, times,
                ocean_mask_ds=mock_mask_ds,
                zarr_out_prefix=zarr_prefix
            )

            # Check raw Zarr target values
            zarr_target_ds = xr.open_zarr(f"{zarr_prefix}_target.zarr", consolidated=True)
            targ_vals = zarr_target_ds["thetao"].values

            # Verify NaNs are preserved and NOT converted to 0.0 in storage
            self.assertTrue(np.isnan(targ_vals[0, 13, 50, 50]), "Stored target must preserve NaN!")
            self.assertTrue(np.isnan(targ_vals[0, 14, 50, 50]), "Stored target must preserve NaN!")
            self.assertFalse(np.any(targ_vals[0, 13:] == 0.0), "Target NaNs must NEVER become zero!")

            # Verify PyTorch OceanReconstructionDataset preserves target validity
            torch_ds = OceanReconstructionDataset(f"{zarr_prefix}_surface.zarr", f"{zarr_prefix}_target.zarr")
            sample = torch_ds[0]

            # In sample["y"], invalid depths retain NaN
            y_tensor = sample["y"]
            mask_tensor = sample["mask"]

            self.assertTrue(torch.isnan(y_tensor[13, 50, 50]), "Target tensor must preserve NaNs at invalid depths")
            self.assertFalse(mask_tensor[13, 50, 50], "Target mask must be False at invalid depths")

            # Verify masked_mse_loss evaluates strictly on valid mask and produces zero NaN contamination
            dummy_pred = torch.full((1, 15, 101, 241), 20.0, requires_grad=True)
            loss = masked_mse_loss(dummy_pred, y_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))
            self.assertFalse(torch.isnan(loss), "Masked loss must not be NaN despite y having NaNs outside mask")
            self.assertGreater(float(loss), 0.0)

    def test_05_target_masks_preserve_all_valid_depths_independently(self):
        """
        5. Target masks preserve all valid depths independently without cross-depth contamination.
        """
        # Column with stepped bathymetry
        # Depth 100m valid, 125m invalid
        profile = np.full(15, np.nan, dtype=np.float32)
        profile[:8] = 25.0  # depths 0 through 100m are valid (indices 0..7)
        mock_3d = np.tile(profile[:, np.newaxis, np.newaxis], (1, 2, 2))

        ds_masks = build_canonical_masks(mock_3d)
        depth_mask = ds_masks["depth_valid_mask"].values

        self.assertTrue(depth_mask[7, 0, 0], "Depth 100m (idx 7) must be valid")
        self.assertFalse(depth_mask[8, 0, 0], "Depth 125m (idx 8) must be invalid")
        self.assertFalse(depth_mask[14, 0, 0], "Depth 1000m (idx 14) must be invalid")

    def test_06_regridding_does_not_convert_land_to_ocean(self):
        """
        6. Regridding does not incorrectly convert land to ocean.
        """
        src_lats = np.array([10.0, 11.0, 12.0])
        src_lons = np.array([70.0, 71.0, 72.0])
        # Grid with all land (all NaNs)
        src_data = np.full((3, 3), np.nan, dtype=np.float32)

        dst_lats = np.array([10.5, 11.5])
        dst_lons = np.array([70.5, 71.5])

        out = regrid_2d_field(src_data, src_lats, src_lons, dst_lats, dst_lons)
        self.assertTrue(np.all(np.isnan(out)), "Regridded land cells must remain NaN")

    def test_07_temporal_alignment_exact_after_chunk_resolution(self):
        """
        7. Temporal alignment remains exact after multi-day chunk resolution.
        """
        from scripts.harmonize_and_validate import find_time_index_in_dataset
        times = pd.date_range("2020-01-01", "2020-01-07")
        ds_chunk = xr.Dataset(coords={"time": times})

        for exp_idx, dt in enumerate(times):
            d_str = dt.strftime("%Y-%m-%d")
            found_idx = find_time_index_in_dataset(ds_chunk, d_str)
            self.assertEqual(found_idx, exp_idx, f"Time index for {d_str} must match exactly")

    def test_08_surface_predictor_missingness_distinguishable_from_land(self):
        """
        8. Surface predictor missingness remains distinguishable from geographic land.
        """
        geo_ocean_mask = np.ones((5, 5), dtype=bool)
        geo_ocean_mask[0, :] = False  # Row 0 is land

        # Surface feature: has a satellite missing dropout at ocean point (2, 2)
        surface_feature = np.full((1, 5, 5, 1), 25.0, dtype=np.float32)
        surface_feature[0, 2, 2, 0] = np.nan  # Satellite dropout over ocean
        surface_feature[0, ~geo_ocean_mask, 0] = np.nan  # Land is also NaN

        surf_valid_mask = create_surface_validity_mask(surface_feature)

        # At (2, 2): Cell is ocean, but predictor is missing
        is_ocean = geo_ocean_mask[2, 2]
        is_observed = surf_valid_mask[0, 2, 2, 0]
        self.assertTrue(is_ocean, "Point (2, 2) is geographic ocean")
        self.assertFalse(is_observed, "Point (2, 2) has missing surface observation")

        # At (0, 0): Cell is land
        is_land = not geo_ocean_mask[0, 0]
        self.assertTrue(is_land, "Point (0, 0) is geographic land")

    def test_09_dataset_shapes_and_masks_dimensions(self):
        """
        9. Dataset shapes remain correct across surface, target, and all 4 explicit masks.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            times = ["2020-01-01", "2020-01-02"]
            n_t = len(times)
            surf_dict = {feat: np.ones((n_t, 101, 241), dtype=np.float32) for feat in CANONICAL_FEATURES}
            targ_arr = np.ones((n_t, 15, 101, 241), dtype=np.float32)

            mask_ds = build_canonical_masks(targ_arr)
            zarr_prefix = os.path.join(tmp_dir, "shape_test")

            ds_s, ds_t = assemble_ml_dataset(
                surf_dict, targ_arr, times,
                ocean_mask_ds=mask_ds,
                zarr_out_prefix=zarr_prefix
            )

            # Surface shapes
            self.assertEqual(ds_s.surface_features.shape, (n_t, 101, 241, 7))
            self.assertEqual(ds_s.surface_validity_mask.shape, (n_t, 101, 241, 7))
            self.assertEqual(ds_s.geographic_ocean_mask.shape, (101, 241))

            # Target shapes
            self.assertEqual(ds_t.thetao.shape, (n_t, 15, 101, 241))
            self.assertEqual(ds_t.target_validity_mask.shape, (n_t, 15, 101, 241))
            self.assertEqual(ds_t.depth_valid_mask.shape, (15, 101, 241))
            self.assertEqual(ds_t.deepest_valid_depth_m.shape, (101, 241))
            self.assertEqual(ds_t.geographic_ocean_mask.shape, (101, 241))

    def test_10_all_canonical_depth_levels_preserved(self):
        """
        10. All 15 canonical depth levels are preserved exactly.
        """
        expected_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        self.assertEqual(len(CANONICAL_DEPTHS), 15)
        self.assertTrue(np.allclose(CANONICAL_DEPTHS, expected_depths))

        # Test vertical interpolation maps to exact canonical depths
        native_d = np.array([0.49, 1.54, 2.65, 5.08, 9.57, 15.81, 25.21, 40.34, 65.81, 109.73, 186.13, 318.13, 541.09, 902.34, 1245.29])
        native_t = 28.0 - 0.015 * native_d
        interp = interpolate_depths_1d(native_t, native_d, CANONICAL_DEPTHS)

    def test_11_open_ocean_1000m_interpolation_finite_and_non_extrapolated(self):
        """
        11. Open-ocean 1000m depth is finite and non-extrapolated when bracketed by native levels.
        When native levels end at 902.34m, 1000m must be strictly NaN (no extrapolation).
        When native levels reach 1062.44m, 1000m must be linearly interpolated and finite.
        """
        # Case A: Native depths stop at 902.34m -> 1000m MUST be NaN (extrapolation refused)
        d_short = np.array([0.49, 10.0, 50.0, 100.0, 300.0, 500.0, 700.0, 902.34], dtype=np.float32)
        t_short = np.array([28.0, 27.5, 25.0, 20.0, 12.0, 8.0, 6.0, 5.0], dtype=np.float32)
        interp_short = interpolate_depths_1d(t_short, d_short, CANONICAL_DEPTHS)
        idx_1000 = list(CANONICAL_DEPTHS).index(1000.0)
        self.assertTrue(np.isnan(interp_short[idx_1000]), "1000m must be NaN when deepest native level is 902.34m")

        # Case B: Native depths reach 1062.44m -> 1000m MUST be finite and bracketed
        d_full = np.array([0.49, 10.0, 50.0, 100.0, 300.0, 500.0, 700.0, 902.34, 1062.44], dtype=np.float32)
        t_full = np.array([28.0, 27.5, 25.0, 20.0, 12.0, 8.0, 6.0, 5.0, 4.2], dtype=np.float32)
        interp_full = interpolate_depths_1d(t_full, d_full, CANONICAL_DEPTHS)
        self.assertFalse(np.isnan(interp_full[idx_1000]), "1000m must be finite when bracketed by 902.34m and 1062.44m")
        # Linearly interpolated value must be strictly between 5.0 and 4.2
        self.assertTrue(4.2 <= interp_full[idx_1000] <= 5.0, f"Interpolated 1000m value {interp_full[idx_1000]} must be between 4.2 and 5.0")

        # Case C: If actual GLORYS pilot file is present, verify native depth >= 1062m
        glorys_path = os.path.join(repo_root, "data", "raw", "glorys", "glorys_thetao_2020-01-01_2020-01-07.nc")
        if os.path.exists(glorys_path):
            with xr.open_dataset(glorys_path) as ds_g:
                max_d = float(ds_g.depth.values.max())
                self.assertGreater(max_d, 1000.0, f"GLORYS pilot chunk max depth {max_d}m must exceed 1000m")
                self.assertAlmostEqual(max_d, 1062.44, delta=1.0)

if __name__ == "__main__":
    unittest.main()
