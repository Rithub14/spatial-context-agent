"""Shared pytest fixtures."""

from unittest.mock import patch

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.db.models import Base
from src.db.session import get_db

# ---------------------------------------------------------------------------
# Shared in-memory SQLite test DB (all API/middleware tests share this)
# ---------------------------------------------------------------------------

TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


def override_get_db():
    """FastAPI dependency override — yields a test DB session."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Patch module-level side-effects that hit the real PostgreSQL engine.
_global_patches = [
    patch("src.api.main.create_tables"),
    patch("src.api.routes.health.check_db_connection", return_value=True),
]
for p in _global_patches:
    p.start()


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


# ---------------------------------------------------------------------------
# Pipeline fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image() -> Image.Image:
    """A solid-colour 224x224 RGB image for use in pipeline tests."""
    return Image.new("RGB", (224, 224), color=(120, 80, 60))


@pytest.fixture
def scene_categories() -> list[str]:
    """Small category list for fast unit tests (subset of SceneClassifier.CATEGORIES)."""
    return [
        "historic building",
        "monument",
        "park or garden",
        "bridge",
        "waterfront",
    ]
