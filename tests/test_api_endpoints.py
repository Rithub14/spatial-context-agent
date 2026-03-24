"""Tests for FastAPI endpoints — CLIP and DB are mocked."""

import base64
import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.db.models import Base
from src.db.session import get_db

# ---------------------------------------------------------------------------
# Test DB (SQLite in-memory, StaticPool so all connections share one database)
# ---------------------------------------------------------------------------

TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


def override_get_db():
    """Override DB dependency with in-memory SQLite."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


app.dependency_overrides[get_db] = override_get_db

# Patch functions that reference the PostgreSQL engine directly.
# Must patch at the *importing* module's namespace, not the source module.
_patches = [
    patch("src.api.main.create_tables"),                        # called in lifespan
    patch("src.api.routes.health.check_db_connection", return_value=True),  # health endpoint
]
for p in _patches:
    p.start()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient with CLIP model mocked."""
    mock_classify = {
        "primary": {"category": "monument", "confidence": 0.75},
        "alternatives": [{"category": "historic building", "confidence": 0.15}],
    }
    with patch("src.api.routes.agent._classifier") as mock_clf:
        mock_clf.classify.return_value = mock_classify
        with TestClient(app) as c:
            yield c


@pytest.fixture
def sample_b64_image() -> str:
    """Base64-encoded 64x64 solid JPEG."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def sample_b64_image_with_gps() -> str:
    """Base64-encoded JPEG with Berlin GPS EXIF embedded."""
    import piexif

    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLatitude: ((52, 1), (30, 1), (5868, 100)),
        piexif.GPSIFD.GPSLongitudeRef: b"E",
        piexif.GPSIFD.GPSLongitude: ((13, 1), (22, 1), (3972, 100)),
    }
    exif_bytes = piexif.dump({"GPS": gps_ifd})
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif_bytes)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Tests — /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client: TestClient) -> None:
    """GET /health must return 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_structure(client: TestClient) -> None:
    """GET /health must return expected fields."""
    response = client.get("/health")
    data = response.json()
    for field in ("status", "model_loaded", "db_connected", "uptime_seconds"):
        assert field in data


# ---------------------------------------------------------------------------
# Tests — POST /api/v1/analyze
# ---------------------------------------------------------------------------

def test_analyze_with_explicit_gps_returns_200(
    client: TestClient, sample_b64_image: str
) -> None:
    """Valid image + explicit GPS must return 200 with correct structure."""
    response = client.post(
        "/api/v1/analyze",
        json={"image": sample_b64_image, "latitude": 52.5163, "longitude": 13.3777},
    )
    assert response.status_code == 200
    data = response.json()
    assert "scene" in data
    assert "location" in data
    assert "narration" in data
    assert "metadata" in data


def test_analyze_scene_result_structure(
    client: TestClient, sample_b64_image: str
) -> None:
    """scene field must have primary, confidence, and alternatives."""
    response = client.post(
        "/api/v1/analyze",
        json={"image": sample_b64_image, "latitude": 52.5163, "longitude": 13.3777},
    )
    scene = response.json()["scene"]
    assert "primary" in scene
    assert "confidence" in scene
    assert "alternatives" in scene
    assert scene["primary"] == "monument"
    assert scene["confidence"] == pytest.approx(0.75)


def test_analyze_metadata_contains_required_fields(
    client: TestClient, sample_b64_image: str
) -> None:
    """metadata must include model_version, inference_time_ms, timestamp, coordinates."""
    response = client.post(
        "/api/v1/analyze",
        json={"image": sample_b64_image, "latitude": 52.5163, "longitude": 13.3777},
    )
    meta = response.json()["metadata"]
    assert "model_version" in meta
    assert "inference_time_ms" in meta
    assert "timestamp" in meta
    assert "coordinates" in meta


def test_analyze_with_exif_gps_returns_200(
    client: TestClient, sample_b64_image_with_gps: str
) -> None:
    """Image with EXIF GPS and no explicit coords must succeed."""
    response = client.post(
        "/api/v1/analyze",
        json={"image": sample_b64_image_with_gps},
    )
    assert response.status_code == 200


def test_analyze_without_gps_returns_400(
    client: TestClient, sample_b64_image: str
) -> None:
    """Image with no GPS EXIF and no explicit coords must return 400."""
    response = client.post(
        "/api/v1/analyze",
        json={"image": sample_b64_image},
    )
    assert response.status_code == 400
    assert "GPS" in response.json()["detail"]


def test_analyze_with_invalid_base64_returns_400(client: TestClient) -> None:
    """Malformed base64 image must return 400."""
    response = client.post(
        "/api/v1/analyze",
        json={"image": "not_valid_base64!!!", "latitude": 52.5, "longitude": 13.4},
    )
    assert response.status_code == 400


def test_analyze_logs_inference_to_db(
    client: TestClient, sample_b64_image: str
) -> None:
    """Each successful analyze call must write one row to inference_logs."""
    from src.db.models import InferenceLog

    client.post(
        "/api/v1/analyze",
        json={"image": sample_b64_image, "latitude": 52.5163, "longitude": 13.3777},
    )
    db = TestingSessionLocal()
    count = db.query(InferenceLog).count()
    db.close()
    assert count == 1


# ---------------------------------------------------------------------------
# Tests — GET /api/v1/locations
# ---------------------------------------------------------------------------

def test_list_locations_returns_200(client: TestClient) -> None:
    """GET /api/v1/locations must return 200 with expected structure."""
    response = client.get("/api/v1/locations")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)
