"""Tests for the GET /health endpoint."""

from fastapi.testclient import TestClient

from src.api.app import app

_client = TestClient(app)


class TestHealthEndpoint:
    """GET /health returns service liveness information."""

    def test_returns_200(self):
        """Health check responds with HTTP 200."""
        response = _client.get("/health")
        assert response.status_code == 200

    def test_status_is_ok(self):
        """Response body contains status 'ok'."""
        response = _client.get("/health")
        assert response.json()["status"] == "ok"

    def test_version_present(self):
        """Response body contains a version key."""
        response = _client.get("/health")
        assert "version" in response.json()

    def test_version_value(self):
        """Version matches the application release string."""
        response = _client.get("/health")
        assert response.json()["version"] == "0.1.0"

    def test_content_type_json(self):
        """Response Content-Type is application/json."""
        response = _client.get("/health")
        assert "application/json" in response.headers["content-type"]
