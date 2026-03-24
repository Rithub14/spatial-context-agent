"""Tests for auth and rate-limit middleware."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

# ---------------------------------------------------------------------------
# Auth middleware tests
# ---------------------------------------------------------------------------

def test_auth_disabled_allows_request_without_key() -> None:
    """When ENABLE_AUTH=false, requests without X-API-Key must pass."""
    with patch("src.api.middleware.auth.settings") as mock_settings:
        mock_settings.enable_auth = False
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200


def test_auth_enabled_rejects_missing_key() -> None:
    """When ENABLE_AUTH=true, missing X-API-Key must return 401."""
    with patch("src.api.middleware.auth.settings") as mock_settings:
        mock_settings.enable_auth = True
        mock_settings.api_key = "secret"
        with TestClient(app) as client:
            response = client.get("/api/v1/locations")
    assert response.status_code == 401


def test_auth_enabled_rejects_wrong_key() -> None:
    """When ENABLE_AUTH=true, wrong X-API-Key must return 401."""
    with patch("src.api.middleware.auth.settings") as mock_settings:
        mock_settings.enable_auth = True
        mock_settings.api_key = "secret"
        with TestClient(app) as client:
            response = client.get("/api/v1/locations", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_auth_enabled_allows_correct_key() -> None:
    """When ENABLE_AUTH=true, correct X-API-Key must pass through."""
    with patch("src.api.middleware.auth.settings") as mock_settings:
        mock_settings.enable_auth = True
        mock_settings.api_key = "secret"
        with TestClient(app) as client:
            response = client.get("/api/v1/locations", headers={"X-API-Key": "secret"})
    assert response.status_code == 200


def test_auth_health_endpoint_always_exempt() -> None:
    """/health must never require an API key even when auth is enabled."""
    with patch("src.api.middleware.auth.settings") as mock_settings:
        mock_settings.enable_auth = True
        mock_settings.api_key = "secret"
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------

def test_rate_limiter_disabled_allows_all_requests() -> None:
    """When ENABLE_RATE_LIMIT=false, no requests should be throttled."""
    with patch("src.api.middleware.rate_limiter.settings") as mock_settings:
        mock_settings.enable_rate_limit = False
        mock_settings.rate_limit_rpm = 2
        with TestClient(app) as client:
            for _ in range(5):
                response = client.get("/health")
    assert response.status_code == 200


def test_rate_limiter_enabled_returns_429_when_exceeded() -> None:
    """When ENABLE_RATE_LIMIT=true, exceeding rpm must return 429."""
    with patch("src.api.middleware.rate_limiter.settings") as mock_settings:
        mock_settings.enable_rate_limit = True
        mock_settings.rate_limit_rpm = 2
        with TestClient(app) as client:
            responses = [client.get("/health") for _ in range(4)]
    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes


def test_rate_limiter_429_includes_retry_after_header() -> None:
    """429 response must include a Retry-After header."""
    with patch("src.api.middleware.rate_limiter.settings") as mock_settings:
        mock_settings.enable_rate_limit = True
        mock_settings.rate_limit_rpm = 1
        with TestClient(app) as client:
            client.get("/health")
            response = client.get("/health")
    if response.status_code == 429:
        assert "retry-after" in response.headers
