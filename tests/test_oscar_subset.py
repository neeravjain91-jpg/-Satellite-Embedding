import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import numpy as np
import pandas as pd
import xarray as xr

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.download_oscar import subset_oscar_dataset, download_and_subset_oscar
from scripts.manifest_manager import ManifestManager

def create_synthetic_oscar_v2_dataset():
    """
    Creates an in-memory synthetic xarray Dataset reproducing the official OSCAR v2.0 schema:
        dimensions:
            latitude = 719
            longitude = 1440
            time = 1
        coordinate variables:
            lat(latitude)
            lon(longitude)
            time(time)
        velocity variables:
            u(time, longitude, latitude)
            v(time, longitude, latitude)
    """
    lat_arr = np.linspace(-89.75, 89.75, 719)
    lon_arr = np.linspace(0.0, 359.75, 1440)
    time_arr = [pd.to_datetime("2020-01-01")]

    # Deterministic ocean current data
    np.random.seed(42)
    u_data = np.random.uniform(-1.5, 1.5, size=(1, 1440, 719)).astype(np.float32)
    v_data = np.random.uniform(-1.5, 1.5, size=(1, 1440, 719)).astype(np.float32)

    # Some land mask NaNs
    u_data[:, :50, :50] = np.nan
    v_data[:, :50, :50] = np.nan

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "longitude", "latitude"], u_data),
            "v": (["time", "longitude", "latitude"], v_data)
        },
        coords={
            "time": time_arr,
            "lat": ("latitude", lat_arr),
            "lon": ("longitude", lon_arr)
        },
        attrs={
            "title": "Ocean Surface Current Analyses Real-time (OSCAR) - Synthetic Test",
            "source": "OSCAR_L4_OC_FINAL_V2.0"
        }
    )
    return ds

class TestOscarSubsetting(unittest.TestCase):
    """
    Regression tests for OSCAR v2.0 schema-aware spatial subsetting.
    """

    def test_legacy_sel_fails_with_root_cause(self):
        """
        Verify that attempting generic ds.sel({'lat': ...}) fails with
        'no index found for coordinate' on official OSCAR v2.0 schema.
        """
        ds = create_synthetic_oscar_v2_dataset()
        with self.assertRaises(KeyError) as cm:
            ds[["u", "v"]].sel({"lat": slice(5.0, 30.0), "lon": slice(45.0, 105.0)})
        self.assertIn("no index found for coordinate", str(cm.exception))

    def test_schema_aware_subset_succeeds(self):
        """
        Verify that subset_oscar_dataset correctly subsets the OSCAR v2.0 dataset:
        - 101 latitude points (5.0° to 30.0°)
        - 241 longitude points (45.0° to 105.0°)
        - 1 time point
        - Preserves coordinate variables lat/lon and data variables u/v
        """
        ds = create_synthetic_oscar_v2_dataset()
        sub = subset_oscar_dataset(ds, bbox=(5.0, 30.0, 45.0, 105.0))

        # Check dimensions
        self.assertEqual(len(sub["latitude"]), 101)
        self.assertEqual(len(sub["longitude"]), 241)
        self.assertEqual(len(sub["time"]), 1)

        # Check coordinates and bounds
        self.assertAlmostEqual(float(sub["lat"].min()), 5.0, places=4)
        self.assertAlmostEqual(float(sub["lat"].max()), 30.0, places=4)
        self.assertAlmostEqual(float(sub["lon"].min()), 45.0, places=4)
        self.assertAlmostEqual(float(sub["lon"].max()), 105.0, places=4)

        # Check variables and non-emptiness
        self.assertIn("u", sub)
        self.assertIn("v", sub)
        self.assertEqual(sub["u"].shape, (1, 241, 101))
        self.assertEqual(sub["v"].shape, (1, 241, 101))
        self.assertGreater(np.sum(~np.isnan(sub["u"].values)), 0)

    def test_download_and_subset_pipeline_with_mock_fetch(self):
        """
        End-to-end integration test of download_and_subset_oscar using
        a temporary NetCDF file and mock fetch.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = os.path.join(tmp_dir, "raw", "oscar")
            os.makedirs(out_dir, exist_ok=True)

            test_manifest = ManifestManager(
                json_path=os.path.join(tmp_dir, "manifest.json"),
                csv_path=os.path.join(tmp_dir, "manifest.csv"),
                checksums_path=os.path.join(tmp_dir, "checksums.csv")
            )

            ds_full = create_synthetic_oscar_v2_dataset()
            granule_info = {
                "title": "oscar_currents_final_20200101",
                "url": "https://example.com/fake_oscar.nc",
                "time_start": "2020-01-01T00:00:00Z"
            }

            # Intercept _fetch inside download_and_subset_oscar by saving ds_full to tmp_raw
            def fake_fetch_side_effect(func, max_retries=3):
                # Save synthetic dataset to tmp_raw
                tmp_raw = os.path.join(out_dir, "oscar_2020-01-01.nc.tmp.nc")
                ds_full.to_netcdf(tmp_raw)
                return True

            with patch("scripts.download_oscar.retry_with_backoff", side_effect=fake_fetch_side_effect), \
                 patch("scripts.download_oscar.check_free_disk"):

                out_file = download_and_subset_oscar(
                    granule_info,
                    bbox=(5.0, 30.0, 45.0, 105.0),
                    output_dir=out_dir,
                    manifest_mgr=test_manifest
                )

            self.assertTrue(os.path.exists(out_file))
            self.assertFalse(os.path.exists(out_file + ".tmp.nc"), "tmp file must be cleaned up")

            # Validate saved file
            with xr.open_dataset(out_file) as ds_saved:
                self.assertEqual(len(ds_saved["latitude"]), 101)
                self.assertEqual(len(ds_saved["longitude"]), 241)
                self.assertEqual(len(ds_saved["time"]), 1)
                self.assertIn("u", ds_saved)
                self.assertIn("v", ds_saved)

if __name__ == "__main__":
    unittest.main()
