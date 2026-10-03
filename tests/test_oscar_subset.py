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

from scripts.download_oscar import (
    subset_oscar_dataset,
    download_and_subset_oscar,
    build_podaac_downloader_cmd,
    ensure_earthdata_netrc,
    COLLECTION_SHORTNAME
)
from scripts.manifest_manager import ManifestManager, compute_sha256

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

class TestOscarSubsettingAndDownloader(unittest.TestCase):
    """
    Comprehensive regression test suite for OSCAR v2.0 acquisition:
    1. Downloader command construction
    2. Authentication failure handling
    3. Successful mocked PO.DAAC downloader execution
    4. Real schema-aware spatial subsetting
    5. Checksum and manifest completion
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

    def test_podaac_downloader_command_construction(self):
        """
        Verify that build_podaac_downloader_cmd constructs compliant arguments:
        - Target collection: OSCAR_L4_OC_FINAL_V2.0
        - Target granule: oscar_currents_final_20200101*
        - Date bounds: start and end ISO strings
        - Extension: .nc
        - NO '--subset' flag (must download authentic granule to preserve source provenance)
        """
        cmd = build_podaac_downloader_cmd(
            collection=COLLECTION_SHORTNAME,
            output_dir="data/raw/oscar/_staging",
            start_date="2020-01-01",
            end_date="2020-01-01",
            granule_name="oscar_currents_final_20200101*"
        )

        self.assertIn("-c", cmd)
        c_idx = cmd.index("-c")
        self.assertEqual(cmd[c_idx + 1], COLLECTION_SHORTNAME)

        self.assertIn("-d", cmd)
        d_idx = cmd.index("-d")
        self.assertEqual(cmd[d_idx + 1], "data/raw/oscar/_staging")

        self.assertIn("-sd", cmd)
        sd_idx = cmd.index("-sd")
        self.assertEqual(cmd[sd_idx + 1], "2020-01-01T00:00:00Z")

        self.assertIn("-ed", cmd)
        ed_idx = cmd.index("-ed")
        self.assertEqual(cmd[ed_idx + 1], "2020-01-01T23:59:59Z")

        self.assertIn("-gr", cmd)
        gr_idx = cmd.index("-gr")
        self.assertEqual(cmd[gr_idx + 1], "oscar_currents_final_20200101*")

        self.assertIn("-e", cmd)
        e_idx = cmd.index("-e")
        self.assertEqual(cmd[e_idx + 1], ".nc")

        # Crucial provenance rule: no Harmony server-side subsetting
        self.assertNotIn("--subset", cmd, "Must download full granule to preserve provenance")

    def test_authentication_failure_handling(self):
        """
        Verify that if Earthdata authentication/netrc is missing or invalid:
        - PermissionError is raised
        - Manifest chunk status is updated to FAILED with descriptive error
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = os.path.join(tmp_dir, "raw", "oscar")
            os.makedirs(out_dir, exist_ok=True)

            test_manifest = ManifestManager(
                json_path=os.path.join(tmp_dir, "manifest.json"),
                csv_path=os.path.join(tmp_dir, "manifest.csv"),
                checksums_path=os.path.join(tmp_dir, "checksums.csv")
            )

            granule_info = {
                "title": "oscar_currents_final_20200101",
                "time_start": "2020-01-01T00:00:00Z"
            }

            with patch("scripts.download_oscar.ensure_earthdata_netrc", return_value=(False, "No credentials in ~/.netrc")), \
                 patch("scripts.download_oscar.check_free_disk"):

                with self.assertRaises(PermissionError) as cm:
                    download_and_subset_oscar(
                        granule_info,
                        output_dir=out_dir,
                        manifest_mgr=test_manifest
                    )

                self.assertIn("NASA Earthdata authentication required", str(cm.exception))

            # Verify manifest recorded failure
            chunk_key = test_manifest.get_chunk_key("OSCAR", "u_v", "2020-01-01", "2020-01-01")
            self.assertIn(chunk_key, test_manifest.manifest)
            rec = test_manifest.manifest[chunk_key]
            self.assertEqual(rec["status"], "FAILED")
            self.assertIn("No credentials", rec["error"])

    def test_successful_mocked_downloader_execution_and_checksum_completion(self):
        """
        End-to-end integration test of podaac-data-downloader execution:
        - Downloader command executed
        - Granule staged and schema-aware subsetting executed
        - Staging directory cleaned up
        - Manifest & checksum updated with status COMPLETE, non-zero size, matching SHA-256
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
                "time_start": "2020-01-01T00:00:00Z"
            }

            # Side-effect for subprocess.run: writes full granule into staging_dir
            def fake_subprocess_run(cmd, *args, **kwargs):
                d_idx = cmd.index("-d")
                target_staging = cmd[d_idx + 1]
                os.makedirs(target_staging, exist_ok=True)
                full_nc = os.path.join(target_staging, "oscar_currents_final_20200101.nc")
                ds_full.to_netcdf(full_nc)
                res = MagicMock()
                res.returncode = 0
                res.stdout = "Downloaded oscar_currents_final_20200101.nc"
                res.stderr = ""
                return res

            with patch("scripts.download_oscar.ensure_earthdata_netrc", return_value=(True, "Credentials verified")), \
                 patch("scripts.download_oscar.subprocess.run", side_effect=fake_subprocess_run), \
                 patch("scripts.download_oscar.check_free_disk"):

                out_file = download_and_subset_oscar(
                    granule_info,
                    bbox=(5.0, 30.0, 45.0, 105.0),
                    output_dir=out_dir,
                    manifest_mgr=test_manifest
                )

            # 1. Output file exists and staging directory is cleaned up
            self.assertTrue(os.path.exists(out_file))
            staging_check = os.path.join(out_dir, "_staging_2020-01-01")
            self.assertFalse(os.path.exists(staging_check), "Staging directory must be cleaned up")

            # 2. Validate saved dataset properties
            with xr.open_dataset(out_file) as ds_saved:
                self.assertEqual(len(ds_saved["latitude"]), 101)
                self.assertEqual(len(ds_saved["longitude"]), 241)
                self.assertEqual(len(ds_saved["time"]), 1)
                self.assertIn("u", ds_saved)
                self.assertIn("v", ds_saved)
                self.assertAlmostEqual(float(ds_saved["lat"].min()), 5.0, places=4)
                self.assertAlmostEqual(float(ds_saved["lat"].max()), 30.0, places=4)
                self.assertAlmostEqual(float(ds_saved["lon"].min()), 45.0, places=4)
                self.assertAlmostEqual(float(ds_saved["lon"].max()), 105.0, places=4)

            # 3. Validate manifest completion & checksum matching
            chunk_key = test_manifest.get_chunk_key("OSCAR", "u_v", "2020-01-01", "2020-01-01")
            self.assertIn(chunk_key, test_manifest.manifest)
            rec = test_manifest.manifest[chunk_key]
            self.assertEqual(rec["status"], "COMPLETE")
            self.assertGreater(rec["size"], 0)
            self.assertEqual(rec["size"], os.path.getsize(out_file))

            real_sha = compute_sha256(out_file)
            self.assertEqual(rec["checksum"], real_sha)

            # 4. Resumable skip check: calling again immediately returns existing file
            with patch("scripts.download_oscar.subprocess.run") as mock_sub:
                out_again = download_and_subset_oscar(
                    granule_info,
                    bbox=(5.0, 30.0, 45.0, 105.0),
                    output_dir=out_dir,
                    manifest_mgr=test_manifest
                )
                self.assertEqual(out_again, out_file)
                mock_sub.assert_not_called()

if __name__ == "__main__":
    unittest.main()
