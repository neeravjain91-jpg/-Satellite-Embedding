import os
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
import xarray as xr

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from preprocessing.canonical_grid import CANONICAL_FEATURES
from preprocessing.build_dataset import assemble_ml_dataset
from preprocessing.tabular_dataset import load_tabular_dataset

class TestTemporalPurgeBuffer(unittest.TestCase):
    """
    Regression test suite for temporal purge buffer enforcement:
    Ensures that chronological train, validation, and test splits are separated
    by explicit purge buffers (default >= 7 days) to prevent physical autocorrelation
    and assimilation smoothing leakage.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.total_days = 366
        self.dates = pd.date_range("2020-01-01", periods=self.total_days, freq="D").strftime("%Y-%m-%d").tolist()

        # Mock compact spatial grid (5x5) for fast zarr dataset generation
        surf_dict = {
            feat: np.ones((self.total_days, 101, 241), dtype=np.float32) * 25.0
            for feat in CANONICAL_FEATURES
        }
        targ_arr = np.ones((self.total_days, 15, 101, 241), dtype=np.float32) * 15.0

        self.zarr_prefix = os.path.join(self.tmp_dir.name, "purge_test")
        assemble_ml_dataset(surf_dict, targ_arr, self.dates, zarr_out_prefix=self.zarr_prefix)
        self.surf_zarr = f"{self.zarr_prefix}_surface.zarr"
        self.targ_zarr = f"{self.zarr_prefix}_target.zarr"
        self.scaler_json = os.path.join(self.tmp_dir.name, "scaler_test.json")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_default_purge_buffer_7_days_separation(self):
        """
        1. Default partition must enforce minimum separation >= 7 days between:
           - train and validation
           - validation and test
        """
        dataset = load_tabular_dataset(
            surf_zarr=self.surf_zarr,
            targ_zarr=self.targ_zarr,
            purge_buffer_days=7,
            scaler_path=self.scaler_json
        )

        meta = dataset["split_metadata"]
        self.assertEqual(meta["purge_buffer_days"], 7)

        t_train_end = pd.to_datetime(meta["train"]["end_date"])
        t_val_start = pd.to_datetime(meta["val"]["start_date"])
        t_val_end = pd.to_datetime(meta["val"]["end_date"])
        t_test_start = pd.to_datetime(meta["test"]["start_date"])

        # Separation check
        train_val_gap_days = (t_val_start - t_train_end).days - 1
        val_test_gap_days = (t_test_start - t_val_end).days - 1

        self.assertGreaterEqual(train_val_gap_days, 7, f"Train/Val gap {train_val_gap_days} must be >= 7 days")
        self.assertGreaterEqual(val_test_gap_days, 7, f"Val/Test gap {val_test_gap_days} must be >= 7 days")

    def test_partition_disjointness_no_buffer_leakage(self):
        """
        2. No purge buffer date may appear in any active data split (Train, Val, Test),
        and splits must be strictly mutually exclusive and non-overlapping.
        """
        dataset = load_tabular_dataset(
            surf_zarr=self.surf_zarr,
            targ_zarr=self.targ_zarr,
            purge_buffer_days=7,
            scaler_path=self.scaler_json
        )

        train_ts = set(pd.to_datetime(dataset["train"]["timestamp"]).strftime("%Y-%m-%d"))
        val_ts = set(pd.to_datetime(dataset["val"]["timestamp"]).strftime("%Y-%m-%d"))
        test_ts = set(pd.to_datetime(dataset["test"]["timestamp"]).strftime("%Y-%m-%d"))

        # Zero intersection between splits
        self.assertTrue(train_ts.isdisjoint(val_ts), "Train and Val must be disjoint")
        self.assertTrue(val_ts.isdisjoint(test_ts), "Val and Test must be disjoint")
        self.assertTrue(train_ts.isdisjoint(test_ts), "Train and Test must be disjoint")

        # Purge buffer dates must not be in any split
        meta = dataset["split_metadata"]
        p1_dates = set(pd.date_range(meta["purge_buffer_1"]["start_date"], meta["purge_buffer_1"]["end_date"]).strftime("%Y-%m-%d"))
        p2_dates = set(pd.date_range(meta["purge_buffer_2"]["start_date"], meta["purge_buffer_2"]["end_date"]).strftime("%Y-%m-%d"))

        self.assertEqual(len(p1_dates), 7)
        self.assertEqual(len(p2_dates), 7)

        self.assertTrue(train_ts.isdisjoint(p1_dates), "Purge 1 dates must not be in Train")
        self.assertTrue(val_ts.isdisjoint(p1_dates), "Purge 1 dates must not be in Val")
        self.assertTrue(val_ts.isdisjoint(p2_dates), "Purge 2 dates must not be in Val")
        self.assertTrue(test_ts.isdisjoint(p2_dates), "Purge 2 dates must not be in Test")

        # Union of splits + buffers must equal all 366 dates
        all_allocated = train_ts | val_ts | test_ts | p1_dates | p2_dates
        self.assertEqual(all_allocated, set(self.dates))

    def test_configurable_purge_buffer_14_days(self):
        """
        3. Configurable purge buffers: 14-day buffer enforces >= 14 days separation.
        """
        dataset = load_tabular_dataset(
            surf_zarr=self.surf_zarr,
            targ_zarr=self.targ_zarr,
            purge_buffer_days=14,
            scaler_path=self.scaler_json
        )

        meta = dataset["split_metadata"]
        self.assertEqual(meta["purge_buffer_days"], 14)

        t_train_end = pd.to_datetime(meta["train"]["end_date"])
        t_val_start = pd.to_datetime(meta["val"]["start_date"])
        t_val_end = pd.to_datetime(meta["val"]["end_date"])
        t_test_start = pd.to_datetime(meta["test"]["start_date"])

        self.assertGreaterEqual((t_val_start - t_train_end).days - 1, 14)
        self.assertGreaterEqual((t_test_start - t_val_end).days - 1, 14)

if __name__ == "__main__":
    unittest.main()
