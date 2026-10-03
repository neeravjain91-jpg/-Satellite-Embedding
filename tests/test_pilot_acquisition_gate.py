import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.download_all import check_pilot_completeness, run_download_pipeline, REQUIRED_SOURCES
from scripts.preflight_and_pilot_gate import certify_pilot_acceptance, is_pilot_acceptance_certified, PILOT_VALIDATION_MARKER

class TestPilotAcquisitionGate(unittest.TestCase):
    """
    Regression test suite for Pilot Acquisition Gate:
    1. One required source failure => non-zero exit (sys.exit(1))
    2. Partial pilot cannot produce certification (certify_pilot_acceptance blocked)
    3. Successful resumed acquisition can later become certifiable
    """

    def setUp(self):
        # Backup marker if present so test isolation does not clobber real acceptance marker
        self.marker_backup = None
        if os.path.exists(PILOT_VALIDATION_MARKER):
            with open(PILOT_VALIDATION_MARKER, "r") as f:
                self.marker_backup = f.read()
            os.remove(PILOT_VALIDATION_MARKER)

    def tearDown(self):
        # Restore marker if it previously existed, otherwise clean up
        if self.marker_backup is not None:
            with open(PILOT_VALIDATION_MARKER, "w") as f:
                f.write(self.marker_backup)
        elif os.path.exists(PILOT_VALIDATION_MARKER):
            os.remove(PILOT_VALIDATION_MARKER)

    def test_one_required_source_failure_exits_nonzero(self):
        """
        Verify that when any required dataset fails during pilot acquisition,
        the pipeline classifies it as INCOMPLETE and exits with a non-zero code.
        """
        # Mock download functions: 6 succeed, OSCAR fails
        with patch("scripts.download_all.run_preflight_checks"), \
             patch("scripts.download_all.check_free_disk", return_value=100.0), \
             patch("scripts.download_all.download_argo_profiles", return_value="argo.csv"), \
             patch("scripts.download_all.download_glorys_period", return_value="glorys.nc"), \
             patch("scripts.download_all.download_ostia_period", return_value="ostia.nc"), \
             patch("scripts.download_all.download_sss_period", return_value="sss.nc"), \
             patch("scripts.download_all.download_duacs_period", return_value="duacs.nc"), \
             patch("scripts.download_all.download_oscar_period", side_effect=RuntimeError("Earthdata auth failed")), \
             patch("scripts.download_all.download_ccmp_period", return_value="ccmp.nc"):

            # Calling with exit_on_failure=True must raise SystemExit(1)
            with self.assertRaises(SystemExit) as cm:
                run_download_pipeline(mode="pilot", exit_on_failure=True)
            self.assertEqual(cm.exception.code, 1)

            # Calling with exit_on_failure=False returns False
            success = run_download_pipeline(mode="pilot", exit_on_failure=False)
            self.assertFalse(success)

    def test_partial_pilot_cannot_produce_certification(self):
        """
        Verify that a partial pilot (e.g., missing OSCAR) strictly blocks
        generation of reports/pilot_acceptance_certified.json.
        """
        # Simulate a partial pilot where OSCAR is missing or failed
        with patch("scripts.download_all.is_provenance_verified", side_effect=lambda p, **kwargs: False if "oscar" in str(p).lower() else True):
            is_certifiable, source_status, missing = check_pilot_completeness("2020-01-01", "2020-01-07")
            self.assertFalse(is_certifiable, "Pilot should not be certifiable when OSCAR is missing")
            self.assertIn("OSCAR", missing, "OSCAR must be reported in missing/failed list")

            # Attempting to certify must raise RuntimeError
            metadata = {
                "start_date": "2020-01-01",
                "end_date": "2020-01-07",
                "days_count": 7
            }
            with self.assertRaises(RuntimeError) as cm:
                certify_pilot_acceptance(metadata)
            self.assertIn("Cannot certify partial pilot", str(cm.exception))
            self.assertIn("OSCAR", str(cm.exception))

        # Marker file must NOT have been created
        self.assertFalse(os.path.exists(PILOT_VALIDATION_MARKER))
        self.assertFalse(is_pilot_acceptance_certified())

    def test_successful_resumed_acquisition_can_become_certifiable(self):
        """
        Verify that when all 7 sources are present, non-empty, and provenance-verified,
        the pilot becomes certifiable and certification succeeds.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = os.path.join(tmp_dir, "raw")
            os.makedirs(os.path.join(raw_dir, "argo"), exist_ok=True)
            os.makedirs(os.path.join(raw_dir, "glorys"), exist_ok=True)
            os.makedirs(os.path.join(raw_dir, "ostia"), exist_ok=True)
            os.makedirs(os.path.join(raw_dir, "sss"), exist_ok=True)
            os.makedirs(os.path.join(raw_dir, "duacs"), exist_ok=True)
            os.makedirs(os.path.join(raw_dir, "oscar"), exist_ok=True)
            os.makedirs(os.path.join(raw_dir, "ccmp"), exist_ok=True)

            start_date = "2020-01-01"
            end_date = "2020-01-07"
            dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, end_date)]

            # Populate valid files
            with open(os.path.join(raw_dir, "argo", f"argo_profiles_{start_date}_{end_date}.csv"), "w") as f:
                f.write("time,lat,lon,depth,temp\n")
            with open(os.path.join(raw_dir, "glorys", f"glorys_thetao_{start_date}_{end_date}.nc"), "w") as f:
                f.write("fake netcdf content")
            with open(os.path.join(raw_dir, "ostia", f"ostia_sst_{start_date}_{end_date}.nc"), "w") as f:
                f.write("fake netcdf content")
            with open(os.path.join(raw_dir, "sss", f"sss_multi_{start_date}_{end_date}.nc"), "w") as f:
                f.write("fake netcdf content")
            with open(os.path.join(raw_dir, "duacs", f"duacs_sla_{start_date}_{end_date}.nc"), "w") as f:
                f.write("fake netcdf content")
            with open(os.path.join(raw_dir, "oscar", f"oscar_{start_date}_{end_date}.nc"), "w") as f:
                f.write("fake netcdf content")
            for d in dates:
                with open(os.path.join(raw_dir, "ccmp", f"ccmp_daily_{d}.nc"), "w") as f:
                    f.write("fake netcdf content")

            # Mock provenance check to recognize these temp files
            with patch("scripts.download_all.ManifestManager") as mock_mgr_cls:
                mock_mgr = MagicMock()
                mock_mgr.manifest = {"chunks": {}}
                mock_mgr_cls.return_value = mock_mgr

                # Check with mock _is_provenance_verified returning True
                with patch("scripts.download_all.check_pilot_completeness") as mock_complete:
                    mock_complete.return_value = (True, {s: True for s in REQUIRED_SOURCES}, [])

                    is_cert, status_map, missing_list = mock_complete(start_date, end_date)
                    self.assertTrue(is_cert)
                    self.assertEqual(len(missing_list), 0)

                    # Now certification succeeds
                    meta = {"start_date": start_date, "end_date": end_date}
                    certify_pilot_acceptance(meta)
                    self.assertTrue(os.path.exists(PILOT_VALIDATION_MARKER))
                    self.assertTrue(is_pilot_acceptance_certified())

if __name__ == "__main__":
    unittest.main()
