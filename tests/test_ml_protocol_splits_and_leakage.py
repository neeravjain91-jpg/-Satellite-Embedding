"""
tests/test_ml_protocol_splits_and_leakage.py
Automated Leakage & Protocol Verification Test Suite:
1. Exact temporal partition:
   Train (days 0–252, 253 days) -> Purge 1 (days 253–258, 6 days) -> Val (days 259–306, 48 days)
   -> Purge 2 (days 307–312, 6 days) -> Test (days 313–365, 53 days)
2. Zero Purge Leakage: No purge-day index ever enters train, val, or test.
3. Train-Only Normalization Isolation: Changing validation/test data cannot alter scaler stats or checksum.
4. Final 4-way unified mask: geo AND surface AND target AND depth validity.
5. Masked loss integrity: No invalid or NaN target can ever enter loss or gradient.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
import json
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_DEPTHS, CANONICAL_FEATURES, CANONICAL_LATS, CANONICAL_LONS
from preprocessing.tabular_dataset import extract_tabular_split, load_tabular_dataset
from preprocessing.ocean_mask import build_final_training_mask
from models.base import masked_mse_loss, compute_masked_metrics


class TestMLProtocolSplitsAndLeakage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.H = len(CANONICAL_LATS)
        self.W = len(CANONICAL_LONS)
        self.n_depths = len(CANONICAL_DEPTHS)
        self.n_features = len(CANONICAL_FEATURES)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_366_day_zarr(self, surf_path, targ_path, fill_val=20.0):
        times = pd.date_range("2020-01-01", periods=366, freq="D")
        
        # Surface features: (366, 101, 241, 7)
        surf_arr = np.full((366, self.H, self.W, self.n_features), fill_val, dtype=np.float32)
        # Add slight variation per day
        for t in range(366):
            surf_arr[t, :, :, 0] += float(t) * 0.01
            
        ds_surf = xr.Dataset(
            data_vars={"surface_features": (["time", "latitude", "longitude", "feature"], surf_arr)},
            coords={
                "time": times,
                "latitude": CANONICAL_LATS,
                "longitude": CANONICAL_LONS,
                "feature": list(CANONICAL_FEATURES)
            }
        )
        ds_surf.to_zarr(surf_path, mode="w", consolidated=True)

        # Target features: (366, 15, 101, 241)
        targ_arr = np.full((366, self.n_depths, self.H, self.W), fill_val, dtype=np.float32)
        # Deep levels in corner are NaN (simulating shallow bathymetry)
        targ_arr[:, 10:, :10, :10] = np.nan
        
        ds_targ = xr.Dataset(
            data_vars={"thetao": (["time", "depth", "latitude", "longitude"], targ_arr)},
            coords={
                "time": times,
                "depth": CANONICAL_DEPTHS,
                "latitude": CANONICAL_LATS,
                "longitude": CANONICAL_LONS
            }
        )
        ds_targ.to_zarr(targ_path, mode="w", consolidated=True)

    def test_01_exact_366_day_temporal_split_and_purge_buffers(self):
        """Verifies exact protocol partition: 253 train (0-252), 6 purge1 (253-258), 48 val (259-306), 6 purge2 (307-312), 53 test (313-365)."""
        surf_path = os.path.join(self.temp_dir.name, "surf_366.zarr")
        targ_path = os.path.join(self.temp_dir.name, "targ_366.zarr")
        scaler_path = os.path.join(self.temp_dir.name, "scaler.json")
        
        self._create_mock_366_day_zarr(surf_path, targ_path)
        
        dataset = load_tabular_dataset(
            surf_zarr=surf_path,
            targ_zarr=targ_path,
            scaler_path=scaler_path
        )
        
        meta = dataset["split_metadata"]
        self.assertEqual(meta["total_days"], 366)
        self.assertEqual(meta["train"]["n_days"], 253)
        self.assertEqual(meta["train"]["start_idx"], 0)
        self.assertEqual(meta["train"]["end_idx"], 252)
        self.assertEqual(meta["train"]["start_date"], "2020-01-01")
        self.assertEqual(meta["train"]["end_date"], "2020-09-09")
        
        self.assertEqual(meta["purge_buffer_1"]["n_days"], 6)
        self.assertEqual(meta["purge_buffer_1"]["start_idx"], 253)
        self.assertEqual(meta["purge_buffer_1"]["end_idx"], 258)
        self.assertEqual(meta["purge_buffer_1"]["start_date"], "2020-09-10")
        self.assertEqual(meta["purge_buffer_1"]["end_date"], "2020-09-15")
        
        self.assertEqual(meta["val"]["n_days"], 48)
        self.assertEqual(meta["val"]["start_idx"], 259)
        self.assertEqual(meta["val"]["end_idx"], 306)
        self.assertEqual(meta["val"]["start_date"], "2020-09-16")
        self.assertEqual(meta["val"]["end_date"], "2020-11-02")
        
        self.assertEqual(meta["purge_buffer_2"]["n_days"], 6)
        self.assertEqual(meta["purge_buffer_2"]["start_idx"], 307)
        self.assertEqual(meta["purge_buffer_2"]["end_idx"], 312)
        self.assertEqual(meta["purge_buffer_2"]["start_date"], "2020-11-03")
        self.assertEqual(meta["purge_buffer_2"]["end_date"], "2020-11-08")
        
        self.assertEqual(meta["test"]["n_days"], 53)
        self.assertEqual(meta["test"]["start_idx"], 313)
        self.assertEqual(meta["test"]["end_idx"], 365)
        self.assertEqual(meta["test"]["start_date"], "2020-11-09")
        self.assertEqual(meta["test"]["end_date"], "2020-12-31")

    def test_02_purge_leakage_strictly_zero(self):
        """Verifies that no sample from purge 1 (days 253-258) or purge 2 (days 307-312) exists in any partition."""
        surf_path = os.path.join(self.temp_dir.name, "surf_leakage.zarr")
        targ_path = os.path.join(self.temp_dir.name, "targ_leakage.zarr")
        scaler_path = os.path.join(self.temp_dir.name, "scaler_leakage.json")
        
        self._create_mock_366_day_zarr(surf_path, targ_path)
        dataset = load_tabular_dataset(surf_zarr=surf_path, targ_zarr=targ_path, scaler_path=scaler_path)
        
        purge1_days = set(range(253, 259))
        purge2_days = set(range(307, 313))
        all_purge_days = purge1_days | purge2_days
        
        train_days = set(dataset["train"]["time_idx"])
        val_days = set(dataset["val"]["time_idx"])
        test_days = set(dataset["test"]["time_idx"])
        
        self.assertEqual(len(train_days.intersection(all_purge_days)), 0)
        self.assertEqual(len(val_days.intersection(all_purge_days)), 0)
        self.assertEqual(len(test_days.intersection(all_purge_days)), 0)
        
        # Test exact partition limits
        self.assertEqual(max(train_days), 252)
        self.assertEqual(min(val_days), 259)
        self.assertEqual(max(val_days), 306)
        self.assertEqual(min(test_days), 313)
        self.assertEqual(max(test_days), 365)

    def test_03_train_only_normalization_isolation_and_checksum(self):
        """Proves that modifying validation or test raw data produces zero change in scaler mean, std, and sha256."""
        surf_a = os.path.join(self.temp_dir.name, "surf_a.zarr")
        targ_a = os.path.join(self.temp_dir.name, "targ_a.zarr")
        scaler_a = os.path.join(self.temp_dir.name, "scaler_a.json")
        self._create_mock_366_day_zarr(surf_a, targ_a, fill_val=20.0)
        
        dataset_a = load_tabular_dataset(surf_zarr=surf_a, targ_zarr=targ_a, scaler_path=scaler_a)
        
        # Create modified version B where validation and test have extreme outliers
        surf_b = os.path.join(self.temp_dir.name, "surf_b.zarr")
        targ_b = os.path.join(self.temp_dir.name, "targ_b.zarr")
        scaler_b = os.path.join(self.temp_dir.name, "scaler_b.json")
        self._create_mock_366_day_zarr(surf_b, targ_b, fill_val=20.0)
        
        # Corrupt val and test days with extreme values in dataset B
        ds_s_b = xr.open_zarr(surf_b)
        val_surf = ds_s_b["surface_features"].values
        val_surf[259:, :, :, :] += 999.0
        ds_s_b["surface_features"].values = val_surf
        ds_s_b.to_zarr(surf_b, mode="a")
        
        dataset_b = load_tabular_dataset(surf_zarr=surf_b, targ_zarr=targ_b, scaler_path=scaler_b)
        
        # Scaler stats from A and B must be IDENTICAL
        np.testing.assert_allclose(dataset_a["scaler"]["mean"], dataset_b["scaler"]["mean"], rtol=1e-6)
        np.testing.assert_allclose(dataset_a["scaler"]["std"], dataset_b["scaler"]["std"], rtol=1e-6)
        np.testing.assert_allclose(dataset_a["scaler"]["target_mean"], dataset_b["scaler"]["target_mean"], rtol=1e-6)
        np.testing.assert_allclose(dataset_a["scaler"]["target_std"], dataset_b["scaler"]["target_std"], rtol=1e-6)
        self.assertEqual(dataset_a["scaler"]["scaler_sha256"], dataset_b["scaler"]["scaler_sha256"])

    def test_04_four_way_unified_mask_construction(self):
        """Verifies build_final_training_mask correctly computes geo & surface & target & depth."""
        geo = np.ones((self.H, self.W), dtype=bool)
        geo[0, 0] = False # Point (0,0) is land
        
        depth = np.ones((self.n_depths, self.H, self.W), dtype=bool)
        depth[-1, 1, 1] = False # Point (1,1) at 1000m is below seafloor
        
        surf = np.ones((10, self.H, self.W), dtype=bool)
        surf[0, 2, 2] = False # Point (2,2) at time 0 missing surface
        
        targ = np.ones((10, self.n_depths, self.H, self.W), dtype=bool)
        targ[1, 3, 3, 3] = False # Point (3,3) at time 1 depth 3 missing thetao
        
        unified = build_final_training_mask(geo, surf, targ, depth)
        self.assertEqual(unified.shape, (10, self.n_depths, self.H, self.W))
        
        # Check invalid conditions are strictly False
        self.assertFalse(unified[:, :, 0, 0].any()) # Land
        self.assertFalse(unified[:, -1, 1, 1].any()) # Below seafloor
        self.assertFalse(unified[0, :, 2, 2].any()) # Missing surface
        self.assertFalse(unified[1, 3, 3, 3]) # Missing target
        
        # Valid point is True
        self.assertTrue(unified[2, 0, 10, 10])

    def test_05_masked_loss_guarantees_invalid_targets_never_propagate(self):
        """Verifies that masked_mse_loss never propagates invalid values and rejects NaN targets."""
        y_pred = torch.tensor([[10.0, 20.0], [15.0, 25.0]], requires_grad=True)
        y_true = torch.tensor([[10.5, np.nan], [14.8, 25.2]])
        mask = torch.tensor([[True, False], [True, True]])
        
        # Loss must succeed because the NaN is outside the mask
        loss = masked_mse_loss(y_pred, y_true, mask)
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        self.assertFalse(torch.isnan(y_pred.grad).any())
        
        # If an unmasked NaN target is passed inside mask, must raise AssertionError
        bad_mask = torch.tensor([[True, True], [True, True]])
        with self.assertRaises(AssertionError):
            masked_mse_loss(y_pred, y_true, bad_mask)


if __name__ == "__main__":
    unittest.main()
