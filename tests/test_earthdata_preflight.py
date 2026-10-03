import os
import sys
import unittest
from unittest.mock import patch, MagicMock

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.preflight_and_pilot_gate import (
    check_earthdata_credentials,
    PreflightStatus,
    run_preflight_checks
)

class TestEarthdataPreflight(unittest.TestCase):
    """
    Regression test suite for NASA Earthdata preflight/downloader consistency:
    1. Preflight reports PASS only when credentials are fully usable by current OSCAR downloader.
    2. When EARTHDATA_TOKEN is present without usable .netrc or username/password,
       reports [WARN] with exact warning message:
       'Token detected, but current PO.DAAC downloader path requires Earthdata username/password or .netrc.'
    3. Valid tokens are not rejected when a verified token-capable downloader path is enabled.
    4. Reports [BLOCKED] when no credentials exist at all.
    """

    def test_token_only_reports_warn(self):
        """
        Verify that having EARTHDATA_TOKEN but no usable netrc/credentials reports [WARN].
        """
        with patch.dict(os.environ, {"EARTHDATA_TOKEN": "valid_token_xyz"}, clear=False), \
             patch("scripts.download_oscar.ensure_earthdata_netrc", return_value=(False, "No netrc file found")):
            
            # Remove any username/password from env for this test
            if "EARTHDATA_USERNAME" in os.environ:
                del os.environ["EARTHDATA_USERNAME"]
            if "EARTHDATA_PASSWORD" in os.environ:
                del os.environ["EARTHDATA_PASSWORD"]
            if "EARTHDATA_TOKEN_DOWNLOADER_SUPPORTED" in os.environ:
                del os.environ["EARTHDATA_TOKEN_DOWNLOADER_SUPPORTED"]

            status, msg = check_earthdata_credentials()

            self.assertEqual(status, "WARN")
            self.assertFalse(bool(status), "PreflightStatus('WARN') must evaluate to falsy")
            self.assertEqual(f"[{status}]", "[WARN]")
            self.assertEqual(
                msg,
                "Token detected, but current PO.DAAC downloader path requires Earthdata username/password or .netrc."
            )

    def test_usable_netrc_reports_pass(self):
        """
        Verify that having usable netrc credentials reports [PASS].
        """
        with patch("scripts.download_oscar.ensure_earthdata_netrc", return_value=(True, "Credentials verified in ~/.netrc")):
            status, msg = check_earthdata_credentials()

            self.assertEqual(status, "PASS")
            self.assertTrue(bool(status), "PreflightStatus('PASS') must evaluate to truthy")
            self.assertEqual(f"[{status}]", "[PASS]")
            self.assertIn("verified and usable by PO.DAAC downloader", msg)

    def test_token_with_token_capable_downloader_reports_pass(self):
        """
        Verify that a valid token is NOT rejected if a verified token-capable
        downloader path is enabled via parameter or environment variable.
        """
        with patch.dict(os.environ, {"EARTHDATA_TOKEN": "valid_token_xyz"}, clear=False), \
             patch("scripts.download_oscar.ensure_earthdata_netrc", return_value=(False, "No netrc")):
            
            if "EARTHDATA_USERNAME" in os.environ:
                del os.environ["EARTHDATA_USERNAME"]
            if "EARTHDATA_PASSWORD" in os.environ:
                del os.environ["EARTHDATA_PASSWORD"]

            # Test 1: Passed via explicit parameter
            status, msg = check_earthdata_credentials(downloader_supports_token=True)
            self.assertEqual(status, "PASS")
            self.assertTrue(bool(status))
            self.assertIn("token-capable downloader", msg)

            # Test 2: Enabled via environment variable
            with patch.dict(os.environ, {"EARTHDATA_TOKEN_DOWNLOADER_SUPPORTED": "1"}):
                status2, msg2 = check_earthdata_credentials()
                self.assertEqual(status2, "PASS")
                self.assertTrue(bool(status2))
                self.assertIn("token-capable downloader", msg2)

    def test_no_credentials_reports_blocked(self):
        """
        Verify that missing credentials report [BLOCKED].
        """
        with patch.dict(os.environ, {}, clear=False), \
             patch("scripts.download_oscar.ensure_earthdata_netrc", return_value=(False, "No netrc")):
            
            for key in ["EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN_DOWNLOADER_SUPPORTED"]:
                if key in os.environ:
                    del os.environ[key]

            status, msg = check_earthdata_credentials()

            self.assertEqual(status, "BLOCKED")
            self.assertFalse(bool(status))
            self.assertEqual(f"[{status}]", "[BLOCKED]")
            self.assertIn("No NASA Earthdata credentials found", msg)

    def test_preflight_checks_runner_records_warn_state(self):
        """
        Verify that run_preflight_checks records the correct status and valid flag
        when token mismatch occurs.
        """
        with patch("scripts.preflight_and_pilot_gate.check_copernicus_credentials_cli", return_value=(True, "OK")), \
             patch("scripts.preflight_and_pilot_gate.run_copernicus_dry_run", return_value=(True, "OK")), \
             patch("scripts.preflight_and_pilot_gate.check_public_sources", return_value={"ccmp": (True, "OK"), "argo": (True, "OK")}), \
             patch("scripts.preflight_and_pilot_gate.check_earthdata_credentials", return_value=(
                 PreflightStatus("WARN"),
                 "Token detected, but current PO.DAAC downloader path requires Earthdata username/password or .netrc."
             )):
            
            results = run_preflight_checks()
            earthdata_result = results.get("earthdata_auth")

            self.assertIsNotNone(earthdata_result)
            self.assertFalse(earthdata_result["valid"])
            self.assertEqual(earthdata_result["status"], "WARN")
            self.assertEqual(
                earthdata_result["detail"],
                "Token detected, but current PO.DAAC downloader path requires Earthdata username/password or .netrc."
            )

if __name__ == "__main__":
    unittest.main()
