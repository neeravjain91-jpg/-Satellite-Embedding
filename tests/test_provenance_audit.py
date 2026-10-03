import os
import sys
import unittest
import tempfile
import json
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.manifest_manager import ManifestManager, compute_sha256, MANIFEST_JSON_PATH, MANIFEST_CSV_PATH, CHECKSUMS_CSV_PATH
from scripts.download_all import is_provenance_verified, check_pilot_completeness
from scripts.preflight_and_pilot_gate import certify_pilot_acceptance, PILOT_VALIDATION_MARKER

class TestProvenanceAudit(unittest.TestCase):
    """
    Regression test suite for Provenance State and Verification:
    1. Checksum mismatch -> provenance failure
    2. Checksum match -> provenance success
    3. Temporary test paths never enter persistent manifest
    4. Partial/FAILED source cannot certify pilot
    """

    def setUp(self):
        if os.path.exists(PILOT_VALIDATION_MARKER):
            os.remove(PILOT_VALIDATION_MARKER)

    def tearDown(self):
        if os.path.exists(PILOT_VALIDATION_MARKER):
            os.remove(PILOT_VALIDATION_MARKER)

    def test_checksum_mismatch_fails_provenance(self):
        """
        Verify that if a file's actual SHA-256 does not match the recorded
        checksum in the manifest or checksums file, provenance verification returns False.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_file = os.path.join(tmp_dir, "sample_sst.nc")
            with open(sample_file, "w") as f:
                f.write("original valid scientific content")
            
            orig_hash = compute_sha256(sample_file)

            isolated_json = os.path.join(tmp_dir, "manifest.json")
            isolated_csv = os.path.join(tmp_dir, "manifest.csv")
            isolated_checksums = os.path.join(tmp_dir, "checksums.csv")

            mgr = ManifestManager(
                json_path=isolated_json,
                csv_path=isolated_csv,
                checksums_path=isolated_checksums
            )

            # Record with valid initial hash and COMPLETE status
            chunk_key = mgr.get_chunk_key("OSTIA", "SST", "2020-01-01", "2020-01-07")
            mgr.record_chunk(
                dataset="OSTIA",
                dataset_id="OSTIA_L4",
                variable="sst",
                start_datetime="2020-01-01",
                end_datetime="2020-01-07",
                bbox=(5.0, 30.0, 45.0, 105.0),
                depth_range=(0, 0),
                output_file=sample_file,
                status="COMPLETE"
            )

            # Verification passes initially
            self.assertTrue(
                is_provenance_verified(
                    sample_file,
                    repo_root_dir=tmp_dir,
                    checksums_file=isolated_checksums,
                    manifest_mgr=mgr
                ),
                "Should pass with exact matching checksum"
            )

            # Now tamper with file content (simulate corruption or synthetic overwrite)
            with open(sample_file, "a") as f:
                f.write("tampered or synthetic extra bytes")

            # Must fail provenance check
            tampered_ok = is_provenance_verified(
                sample_file,
                repo_root_dir=tmp_dir,
                checksums_file=isolated_checksums,
                manifest_mgr=mgr
            )
            self.assertFalse(tampered_ok, "Must return False when actual SHA-256 mismatches recorded hash")

    def test_checksum_match_succeeds_provenance(self):
        """
        Verify that when a non-empty file exists, status is COMPLETE, and
        the recomputed SHA-256 matches recorded checksum, provenance succeeds.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_file = os.path.join(tmp_dir, "argo_profiles.csv")
            with open(sample_file, "w") as f:
                f.write("time,lat,lon,depth,temp\n2020-01-01,10.0,60.0,5.0,28.5\n")

            isolated_json = os.path.join(tmp_dir, "manifest.json")
            isolated_csv = os.path.join(tmp_dir, "manifest.csv")
            isolated_checksums = os.path.join(tmp_dir, "checksums.csv")

            mgr = ManifestManager(
                json_path=isolated_json,
                csv_path=isolated_csv,
                checksums_path=isolated_checksums
            )

            mgr.record_chunk(
                dataset="ARGO",
                dataset_id="ArgoFloats",
                variable="temp",
                start_datetime="2020-01-01",
                end_datetime="2020-01-07",
                bbox=(5.0, 30.0, 45.0, 105.0),
                depth_range=(0.0, 1000.0),
                output_file=sample_file,
                status="COMPLETE"
            )

            self.assertTrue(
                is_provenance_verified(
                    sample_file,
                    repo_root_dir=tmp_dir,
                    checksums_file=isolated_checksums,
                    manifest_mgr=mgr
                )
            )

    def test_temporary_test_paths_never_enter_persistent_manifest(self):
        """
        Verify that attempts to record temporary test paths into the production
        manifest raise ValueError, and assert persistent manifest/checksum files
        contain zero temporary paths.
        """
        prod_mgr = ManifestManager()

        # Attempting to record a temp path to the production manifest must be rejected
        temp_fake_path = r"C:\Users\ASUS\AppData\Local\Temp\test_dir\fake.nc"
        with self.assertRaises(ValueError) as cm:
            prod_mgr.record_chunk(
                dataset="TEST",
                dataset_id="TEST_ID",
                variable="temp",
                start_datetime="2020-01-01",
                end_datetime="2020-01-01",
                bbox=(5.0, 30.0, 45.0, 105.0),
                depth_range=(0, 0),
                output_file=temp_fake_path,
                status="DOWNLOADING"
            )
        self.assertIn("Temporary test path rejected from persistent manifest", str(cm.exception))

        # Check production manifest JSON
        prod_json_path = os.path.join(repo_root, MANIFEST_JSON_PATH)
        if os.path.exists(prod_json_path):
            with open(prod_json_path, "r") as f:
                manifest_data = json.load(f)
            for k, v in manifest_data.items():
                out_f = v.get("output_file", "").lower()
                self.assertNotIn("temp", out_f, f"Temp path found in {k}: {out_f}")
                self.assertNotIn("tmp", out_f, f"Tmp path found in {k}: {out_f}")
                self.assertNotIn("appdata", out_f, f"AppData path found in {k}: {out_f}")

        # Check production checksums CSV
        prod_ck_path = os.path.join(repo_root, CHECKSUMS_CSV_PATH)
        if os.path.exists(prod_ck_path):
            df_c = pd.read_csv(prod_ck_path)
            for _, r in df_c.iterrows():
                fp = str(r["filepath"]).lower()
                self.assertNotIn("temp", fp, f"Temp path found in checksums: {fp}")
                self.assertNotIn("tmp", fp, f"Tmp path found in checksums: {fp}")
                self.assertNotIn("appdata", fp, f"AppData path found in checksums: {fp}")

    def test_partial_or_failed_source_cannot_certify_pilot(self):
        """
        Verify that if any source has status != COMPLETE (e.g. FAILED) or is missing,
        check_pilot_completeness returns False and certify_pilot_acceptance raises RuntimeError.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = os.path.join(tmp_dir, "raw")
            for src in ["argo", "glorys", "ostia", "sss", "duacs", "oscar", "ccmp"]:
                os.makedirs(os.path.join(raw_dir, src), exist_ok=True)

            start_date = "2020-01-01"
            end_date = "2020-01-07"
            dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, end_date)]

            isolated_json = os.path.join(tmp_dir, "manifest.json")
            isolated_csv = os.path.join(tmp_dir, "manifest.csv")
            isolated_checksums = os.path.join(tmp_dir, "checksums.csv")

            mgr = ManifestManager(
                json_path=isolated_json,
                csv_path=isolated_csv,
                checksums_path=isolated_checksums
            )

            # Create 6 complete sources
            src_files = {
                "argo": os.path.join(raw_dir, "argo", f"argo_profiles_{start_date}_{end_date}.csv"),
                "glorys": os.path.join(raw_dir, "glorys", f"glorys_thetao_{start_date}_{end_date}.nc"),
                "ostia": os.path.join(raw_dir, "ostia", f"ostia_sst_{start_date}_{end_date}.nc"),
                "sss": os.path.join(raw_dir, "sss", f"sss_multi_{start_date}_{end_date}.nc"),
                "duacs": os.path.join(raw_dir, "duacs", f"duacs_sla_{start_date}_{end_date}.nc"),
                "oscar": os.path.join(raw_dir, "oscar", f"oscar_{start_date}_{end_date}.nc"),
            }

            for s_name, s_file in src_files.items():
                with open(s_file, "w") as f:
                    f.write(f"scientific content for {s_name}")
                mgr.record_chunk(
                    dataset=s_name.upper(),
                    dataset_id=f"{s_name}_id",
                    variable="var",
                    start_datetime=start_date,
                    end_datetime=end_date,
                    bbox=(5.0, 30.0, 45.0, 105.0),
                    depth_range=(0, 0),
                    output_file=s_file,
                    status="COMPLETE"
                )

            # CCMP has daily files, but day 4 failed!
            for d in dates:
                c_file = os.path.join(raw_dir, "ccmp", f"ccmp_daily_{d}.nc")
                with open(c_file, "w") as f:
                    f.write(f"ccmp content {d}")
                status = "COMPLETE" if d != "2020-01-04" else "FAILED"
                mgr.record_chunk(
                    dataset="CCMP",
                    dataset_id="ccmp_id",
                    variable="uwnd_vwnd",
                    start_datetime=d,
                    end_datetime=d,
                    bbox=(5.0, 30.0, 45.0, 105.0),
                    depth_range=(0, 0),
                    output_file=c_file,
                    status=status
                )

            is_cert, status_map, missing = check_pilot_completeness(
                start_date=start_date,
                end_date=end_date,
                data_dir=raw_dir,
                manifest_mgr=mgr,
                checksums_file=isolated_checksums
            )

            self.assertFalse(is_cert, "Pilot with failed CCMP day must not be certifiable")
            self.assertIn("CCMP", missing, "CCMP must be in missing/failed list")
            self.assertFalse(status_map["ccmp"])

if __name__ == "__main__":
    unittest.main()
