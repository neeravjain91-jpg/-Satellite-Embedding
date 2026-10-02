import unittest
from unittest.mock import patch, MagicMock
import requests
from scripts.preflight_and_pilot_gate import check_public_sources

class TestArgoPreflightCheck(unittest.TestCase):
    """
    Regression tests for ARGO public-endpoint preflight check:
    - Follows HTTP redirects automatically
    - Reports '[PASS] HTTP 200 after redirect (initial 301)' when redirected to 200
    - Reports '[PASS] HTTP Status 200' when reached directly
    - Reports '[FAIL] HTTP <status> at <url>' when response is non-200
    - Handles exceptions cleanly
    """

    def test_argo_redirect_301_to_200_mocked(self):
        """Verify redirect from 301 to 200 is properly detected and reported."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.html"
        
        mock_redirect = MagicMock()
        mock_redirect.status_code = 301
        mock_resp.history = [mock_redirect]

        with patch("requests.head") as mock_head:
            def side_effect(url, *args, **kwargs):
                if "argo" in url.lower() or "ifremer" in url.lower():
                    # Assert allow_redirects=True was passed
                    self.assertTrue(kwargs.get("allow_redirects", False), "requests.head must be called with allow_redirects=True")
                    return mock_resp
                res = MagicMock()
                res.status_code = 200
                res.history = []
                return res

            mock_head.side_effect = side_effect
            status = check_public_sources()

        self.assertIn("argo", status)
        ok, msg = status["argo"]
        self.assertTrue(ok, "Status should be True for 200 after redirect")
        self.assertEqual(msg, "HTTP 200 after redirect (initial 301)")

    def test_argo_direct_200_mocked(self):
        """Verify direct 200 (no redirects) reports 'HTTP Status 200'."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.history = []
        mock_resp.url = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.html"

        with patch("requests.head") as mock_head:
            def side_effect(url, *args, **kwargs):
                return mock_resp

            mock_head.side_effect = side_effect
            status = check_public_sources()

        ok, msg = status["argo"]
        self.assertTrue(ok)
        self.assertEqual(msg, "HTTP Status 200")

    def test_argo_failure_after_redirect_mocked(self):
        """Verify that a redirect leading to non-200 reports clear FAIL with status and URL."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.url = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats_missing.html"
        mock_redirect = MagicMock()
        mock_redirect.status_code = 301
        mock_resp.history = [mock_redirect]

        with patch("requests.head") as mock_head:
            mock_head.return_value = mock_resp
            status = check_public_sources()

        ok, msg = status["argo"]
        self.assertFalse(ok)
        self.assertEqual(msg, "HTTP 404 at https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats_missing.html")

    def test_argo_connection_error_mocked(self):
        """Verify that connection failures are reported cleanly."""
        with patch("requests.head", side_effect=requests.ConnectionError("Name resolution failure")):
            status = check_public_sources()

        ok, msg = status["argo"]
        self.assertFalse(ok)
        self.assertIn("Name resolution failure", msg)

    def test_argo_live_endpoint(self):
        """Integration test against the live IFREMER ARGO endpoint."""
        status = check_public_sources()
        self.assertIn("argo", status)
        ok, msg = status["argo"]
        self.assertTrue(ok, f"Live endpoint check failed: {msg}")
        self.assertEqual(msg, "HTTP 200 after redirect (initial 301)")

if __name__ == "__main__":
    unittest.main()
