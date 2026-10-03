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

from scripts.manifest_manager import ManifestManager, compute_sha256
from scripts.harmonize_and_validate import (
    resolve_source_file_for_date,
    check_provenance_and_lineage,
    load_real_surface_and_target_day,
    find_time_index_in_dataset,
    ResolvedSource
)
from preprocessing.depth_interpolation import interpolate_glorys_to_canonical_depths
from preprocessing.regrid import regrid_glorys_thetao

class TestHarmonizationChunkResolution(unittest.TestCase):
    """
    Regression test suite for harmonization pipeline's temporal chunk resolution:
    1. Daily file resolution.
    2. Multi-day chunk resolution.
    3. Date missing from a multi-day file => hard failure.
    4. Multi-day OSTIA selects the correct time index.
    5. Multi-day SSS selects the correct time index.
    6. Multi-day DUACS selects the correct time index.
    7. Multi-day GLORYS selects the correct time index before depth interpolation.
    8. Daily OSCAR resolution.
    9. Daily CCMP resolution.
    10. Checksum mismatch => provenance failure.
    11. Manifest status != COMPLETE => provenance failure.
    """

    def test_daily_file_resolution(self):
        """1. Daily file resolution prefers and returns exact daily file."""
        res = resolve_source_file_for_date("CCMP", "2020-01-02")
        self.assertIsInstance(res, ResolvedSource)
        self.assertIn("ccmp_daily_2020-01-02.nc", res.file_path)
        self.assertEqual(res.time_index, 0)
        self.assertEqual(res.date, "2020-01-02")

    def test_multiday_chunk_resolution(self):
        """2. Multi-day chunk resolution correctly locates containing multi-day chunk."""
        res = resolve_source_file_for_date("OSTIA", "2020-01-03")
        self.assertIn("ostia_sst_2020-01-01_2020-01-07.nc", res.file_path)
        self.assertEqual(res.time_index, 2)
        self.assertEqual(res.date, "2020-01-03")

    def test_date_missing_from_multiday_file_fails(self):
        """3. Date missing from multi-day file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as cm:
            resolve_source_file_for_date("OSTIA", "2020-01-25")
        self.assertIn("Missing authentic NetCDF file", str(cm.exception))
        self.assertIn("2020-01-25", str(cm.exception))

    def test_multiday_ostia_selects_correct_time_index(self):
        """4. Multi-day OSTIA selects the exact time coordinate for all pilot days."""
        expected_indices = {
            "2020-01-01": 0,
            "2020-01-02": 1,
            "2020-01-03": 2,
            "2020-01-04": 3,
            "2020-01-05": 4,
            "2020-01-06": 5,
            "2020-01-07": 6,
        }
        for d_str, expected_idx in expected_indices.items():
            res = resolve_source_file_for_date("OSTIA", d_str)
            self.assertEqual(res.time_index, expected_idx)
            with xr.open_dataset(res.file_path) as ds:
                t_val = str(ds.time.values[res.time_index])[:10]
                self.assertEqual(t_val, d_str)

    def test_multiday_sss_selects_correct_time_index(self):
        """5. Multi-day SSS selects exact time index matching requested date."""
        for idx, day in enumerate(range(1, 8)):
            d_str = f"2020-01-{day:02d}"
            res = resolve_source_file_for_date("SSS", d_str)
            self.assertEqual(res.time_index, idx)
            with xr.open_dataset(res.file_path) as ds:
                t_val = str(ds.time.values[res.time_index])[:10]
                self.assertEqual(t_val, d_str)

    def test_multiday_duacs_selects_correct_time_index(self):
        """6. Multi-day DUACS selects exact time index matching requested date."""
        for idx, day in enumerate(range(1, 8)):
            d_str = f"2020-01-{day:02d}"
            res = resolve_source_file_for_date("DUACS", d_str)
            self.assertEqual(res.time_index, idx)
            with xr.open_dataset(res.file_path) as ds:
                t_val = str(ds.time.values[res.time_index])[:10]
                self.assertEqual(t_val, d_str)

    def test_multiday_glorys_selects_correct_time_index_before_depth_interpolation(self):
        """7. Multi-day GLORYS selects time index first, producing 3D (15, 101, 241)."""
        res = resolve_source_file_for_date("GLORYS", "2020-01-04")
        self.assertEqual(res.time_index, 3)
        with xr.open_dataset(res.file_path) as ds:
            da_day = ds["thetao"].isel(time=res.time_index).squeeze()
            # Dimensions before depth interp: (depth: 35, lat: 301, lon: 721)
            self.assertEqual(da_day.dims, ("depth", "latitude", "longitude"))
            da_interp = interpolate_glorys_to_canonical_depths(da_day)
            thetao_3d = regrid_glorys_thetao(da_interp)
            self.assertEqual(thetao_3d.shape, (15, 101, 241))

    def test_daily_oscar_resolution(self):
        """8. Daily OSCAR resolution resolves file and regrids to (101, 241)."""
        res = resolve_source_file_for_date("OSCAR", "2020-01-01")
        self.assertIn("oscar_2020-01-01.nc", res.file_path)
        self.assertEqual(res.time_index, 0)
        daily_surf, _ = load_real_surface_and_target_day("2020-01-01")
        self.assertEqual(daily_surf["current_u"].shape, (101, 241))
        self.assertEqual(daily_surf["current_v"].shape, (101, 241))

    def test_daily_ccmp_resolution(self):
        """9. Daily CCMP resolution resolves file and regrids to (101, 241)."""
        res = resolve_source_file_for_date("CCMP", "2020-01-01")
        self.assertIn("ccmp_daily_2020-01-01.nc", res.file_path)
        self.assertEqual(res.time_index, 0)
        daily_surf, _ = load_real_surface_and_target_day("2020-01-01")
        self.assertEqual(daily_surf["wind_u"].shape, (101, 241))
        self.assertEqual(daily_surf["wind_v"].shape, (101, 241))

    def test_checksum_mismatch_fails_provenance(self):
        """10. Checksum mismatch in provenance check raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_chk = os.path.join(tmp_dir, "bad_checksums.csv")
            # Write invalid checksum for OSTIA
            with open(bad_chk, "w") as f:
                f.write("timestamp,filepath,sha256,size_bytes\n")
                f.write("2020-01-01T00:00:00,data/raw/ostia/ostia_sst_2020-01-01_2020-01-07.nc,0000000000000000000000000000000000000000000000000000000000000000,8432372\n")

            with self.assertRaises(ValueError) as cm:
                check_provenance_and_lineage(
                    "2020-01-01", "2020-01-01",
                    checksums_path=bad_chk
                )
            self.assertIn("SHA-256 mismatch", str(cm.exception))

    def test_manifest_not_complete_fails_provenance(self):
        """11. Manifest chunk status != COMPLETE raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_p = os.path.join(tmp_dir, "manifest.json")
            csv_p = os.path.join(tmp_dir, "manifest.csv")
            chk_p = os.path.join(tmp_dir, "checksums.csv")

            test_mgr = ManifestManager(json_path=json_p, csv_path=csv_p, checksums_path=chk_p)
            test_mgr.record_chunk(
                dataset="OSTIA",
                dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
                variable="analysed_sst",
                start_datetime="2020-01-01",
                end_datetime="2020-01-07",
                bbox=(5.0, 30.0, 45.0, 105.0),
                depth_range=(0, 0),
                output_file="data/raw/ostia/ostia_sst_2020-01-01_2020-01-07.nc",
                status="DOWNLOADING" # Not complete!
            )

            with self.assertRaises(ValueError) as cm:
                check_provenance_and_lineage(
                    "2020-01-01", "2020-01-01",
                    manifest_mgr=test_mgr
                )
            self.assertIn("expected 'COMPLETE'", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
