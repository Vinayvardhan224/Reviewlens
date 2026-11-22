"""
Unit and regression tests for ReviewLens FastAPI endpoints (api.py).
Tests:
  - Public endpoints (/api/v1/health, /api/v1/aspects) accessible without credentials.
  - Enforced API key authentication (401 when missing or invalid, 200 when valid).
  - Both 'X-API-Key' and 'Authorization: Bearer <key>' headers.
  - Sliding-window rate limiting (429 when threshold exceeded).
  - Payload size and batch size protection.
"""

import os
import unittest
from fastapi.testclient import TestClient
from api import app, _client_requests, _rate_limit_lock


class TestFastAPISecurityAndEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        # Clear rate limiter state between tests
        with _rate_limit_lock:
            _client_requests.clear()
        # Clean environment
        os.environ.pop("REVIEWLENS_API_KEY", None)
        os.environ.pop("REVIEWLENS_REQUIRE_API_KEY", None)

    def tearDown(self):
        os.environ.pop("REVIEWLENS_API_KEY", None)
        os.environ.pop("REVIEWLENS_REQUIRE_API_KEY", None)
        with _rate_limit_lock:
            _client_requests.clear()

    def test_health_check_public(self):
        """Health check endpoint must be publicly accessible."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_aspects_taxonomy_public(self):
        """Aspects taxonomy endpoint must be publicly accessible."""
        response = self.client.get("/api/v1/aspects")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("Battery & Power", data)
        self.assertIn("Camera & Optics", data)

    def test_about_landing_page(self):
        """Landing page endpoint /about serves landing.html with status 200."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ReviewLens", response.text)

    def test_analyze_without_auth_when_not_enforced(self):
        """When auth is not enforced, analyze endpoint allows requests."""
        payload = {"text": "Great battery life and crystal clear display.", "rating": 5.0}
        response = self.client.post("/api/v1/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("sentiment", data)
        self.assertEqual(data["sentiment"]["label"], "POSITIVE")

    def test_analyze_requires_key_when_configured(self):
        """When REVIEWLENS_API_KEY is configured, unauthenticated requests must be rejected with 401."""
        os.environ["REVIEWLENS_API_KEY"] = "secret-production-token-12345"
        payload = {"text": "Great sound quality and deep bass.", "rating": 5.0}

        # 1. No key provided
        resp_no_key = self.client.post("/api/v1/analyze", json=payload)
        self.assertEqual(resp_no_key.status_code, 401)
        self.assertIn("Invalid or missing API Key", resp_no_key.json()["detail"])

        # 2. Wrong key provided
        resp_wrong_key = self.client.post(
            "/api/v1/analyze",
            json=payload,
            headers={"X-API-Key": "wrong-token"}
        )
        self.assertEqual(resp_wrong_key.status_code, 401)

        # 3. Valid key via X-API-Key header
        resp_valid_header = self.client.post(
            "/api/v1/analyze",
            json=payload,
            headers={"X-API-Key": "secret-production-token-12345"}
        )
        self.assertEqual(resp_valid_header.status_code, 200)
        self.assertEqual(resp_valid_header.json()["sentiment"]["label"], "POSITIVE")

        # 4. Valid key via Authorization: Bearer header
        resp_bearer = self.client.post(
            "/api/v1/analyze",
            json=payload,
            headers={"Authorization": "Bearer secret-production-token-12345"}
        )
        self.assertEqual(resp_bearer.status_code, 200)

    def test_batch_analyze_limit(self):
        """Batch analysis must reject requests exceeding maximum batch limit."""
        reviews = [{"text": f"Review number {i}", "rating": 4.0} for i in range(105)]
        response = self.client.post("/api/v1/analyze-batch", json={"reviews": reviews})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Batch size exceeds maximum limit", response.json()["detail"])

    def test_rate_limiting_enforcement(self):
        """Making more than 60 requests within a minute must trigger 429 Too Many Requests."""
        payload = {"text": "Short review test.", "rating": 4.0}
        headers = {"X-API-Key": "rate-limit-test-user"}

        # Fire 60 requests successfully
        for _ in range(60):
            resp = self.client.post("/api/v1/analyze", json=payload, headers=headers)
            self.assertEqual(resp.status_code, 200)

        # The 61st request must trigger 429
        resp_blocked = self.client.post("/api/v1/analyze", json=payload, headers=headers)
        self.assertEqual(resp_blocked.status_code, 429)
        self.assertIn("Rate limit exceeded", resp_blocked.json()["detail"])
        self.assertEqual(resp_blocked.headers.get("Retry-After"), "60")


    def test_require_key_with_empty_server_key_rejects_all(self):
        """When REVIEWLENS_REQUIRE_API_KEY=true but server key is empty, all requests must be rejected."""
        os.environ["REVIEWLENS_REQUIRE_API_KEY"] = "true"
        os.environ["REVIEWLENS_API_KEY"] = ""
        payload = {"text": "Test review text.", "rating": 5.0}

        # Any non-empty key must be rejected with 500 (misconfiguration)
        resp = self.client.post("/api/v1/analyze", json=payload, headers={"X-API-Key": "some-random-key"})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Server configuration error", resp.json()["detail"])

    def test_rate_limiting_not_bypassable_with_random_keys(self):
        """Rate limit is keyed on client IP, so rotating random API keys cannot bypass the limit."""
        payload = {"text": "Short review test.", "rating": 4.0}

        # Send 60 requests from the same client with different random API keys
        for i in range(60):
            headers = {"X-API-Key": f"random-key-{i}"}
            resp = self.client.post("/api/v1/analyze", json=payload, headers=headers)
            self.assertEqual(resp.status_code, 200)

        # 61st request with yet another key must be blocked by IP rate limit
        resp_blocked = self.client.post(
            "/api/v1/analyze",
            json=payload,
            headers={"X-API-Key": "yet-another-random-key"}
        )
        self.assertEqual(resp_blocked.status_code, 429)
        self.assertIn("Rate limit exceeded", resp_blocked.json()["detail"])

    def test_rate_limiter_stale_client_cleanup(self):
        """Inactive client entries must be pruned to prevent unbounded memory growth."""
        from api import check_rate_limit, _client_requests, _rate_limit_lock
        import time

        with _rate_limit_lock:
            # Insert a client request older than 60 seconds
            _client_requests["old-stale-ip"].append(time.time() - 120.0)

        self.assertIn("old-stale-ip", _client_requests)

        # Trigger rate limit check for another client with cleanup enabled
        check_rate_limit("active-new-ip", force_cleanup=True)

        # Stale client must have been pruned from memory
        self.assertNotIn("old-stale-ip", _client_requests)
        self.assertIn("active-new-ip", _client_requests)


if __name__ == "__main__":
    unittest.main()
