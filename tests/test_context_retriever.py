"""Tests for ContextRetriever — Haversine nearest-landmark lookup."""

import uuid
from unittest.mock import MagicMock

import pytest

from src.pipeline.context_retriever import ContextRetriever

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmark(
    name: str,
    lat: float,
    lng: float,
    district: str = "Mitte",
    category: str = "monument",
) -> MagicMock:
    """Return a mock Landmark ORM object."""
    lm = MagicMock()
    lm.id = uuid.uuid4()
    lm.name = name
    lm.description = f"Description of {name}"
    lm.latitude = lat
    lm.longitude = lng
    lm.city = "Berlin"
    lm.district = district
    lm.category = category
    lm.historical_period = "Test period"
    lm.narration_template = f"You're at {name}."
    lm.image_url = None
    return lm


def _make_db(landmarks: list) -> MagicMock:
    """Return a mock DB session whose query().all() returns landmarks."""
    db = MagicMock()
    db.query.return_value.all.return_value = landmarks
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_find_nearest_returns_closest_landmark() -> None:
    """find_nearest_landmark must return the landmark closest to the query point."""
    gate = _make_landmark("Brandenburg Gate", 52.5163, 13.3777)
    tower = _make_landmark("TV Tower", 52.5208, 13.4094)
    db = _make_db([gate, tower])

    result = ContextRetriever(db).find_nearest_landmark(52.5163, 13.3777)

    assert result is not None
    assert result["name"] == "Brandenburg Gate"


def test_find_nearest_returns_distance_meters() -> None:
    """Result must include distance_meters as an integer."""
    gate = _make_landmark("Brandenburg Gate", 52.5163, 13.3777)
    db = _make_db([gate])

    result = ContextRetriever(db).find_nearest_landmark(52.5163, 13.3777)

    assert result is not None
    assert isinstance(result["distance_meters"], int)
    assert result["distance_meters"] == 0  # same coords → 0 m


def test_find_nearest_returns_none_outside_radius() -> None:
    """find_nearest_landmark must return None when all landmarks are outside radius."""
    # Coordinates in the middle of the ocean, far from any landmark
    gate = _make_landmark("Brandenburg Gate", 52.5163, 13.3777)
    db = _make_db([gate])

    result = ContextRetriever(db).find_nearest_landmark(0.0, 0.0, radius_km=5.0)

    assert result is None


def test_find_nearest_returns_none_when_db_empty() -> None:
    """find_nearest_landmark must return None when no landmarks exist."""
    db = _make_db([])
    result = ContextRetriever(db).find_nearest_landmark(52.5163, 13.3777)
    assert result is None


def test_find_nearest_within_custom_radius() -> None:
    """A landmark 3 km away must be found with radius_km=5 but not radius_km=1."""
    # Reichstag is ~350 m from Brandenburg Gate — both should be within 5 km
    reichstag = _make_landmark("Reichstag", 52.5186, 13.3762)
    db = _make_db([reichstag])

    # Query from a point ~4 km away (near Tiergarten station)
    lat, lng = 52.5145, 13.3301

    found = ContextRetriever(db).find_nearest_landmark(lat, lng, radius_km=5.0)
    not_found = ContextRetriever(db).find_nearest_landmark(lat, lng, radius_km=1.0)

    assert found is not None
    assert not_found is None


def test_find_nearest_result_contains_all_fields() -> None:
    """Result dict must contain all expected landmark fields."""
    gate = _make_landmark("Brandenburg Gate", 52.5163, 13.3777)
    db = _make_db([gate])

    result = ContextRetriever(db).find_nearest_landmark(52.5163, 13.3777)

    assert result is not None
    for field in ("id", "name", "description", "latitude", "longitude",
                  "city", "district", "category", "narration_template", "distance_meters"):
        assert field in result, f"Missing field: {field}"


def test_haversine_distance_same_point_is_zero() -> None:
    """Haversine distance from a point to itself must be 0."""
    dist = ContextRetriever.haversine_distance_m(52.5163, 13.3777, 52.5163, 13.3777)
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_haversine_distance_gate_to_reichstag() -> None:
    """Brandenburg Gate to Reichstag is approximately 275 m."""
    dist = ContextRetriever.haversine_distance_m(
        52.5163, 13.3777,   # Brandenburg Gate
        52.5186, 13.3762,   # Reichstag
    )
    assert 200 < dist < 400


def test_haversine_distance_gate_to_tv_tower() -> None:
    """Brandenburg Gate to TV Tower is approximately 2.2 km."""
    dist = ContextRetriever.haversine_distance_m(
        52.5163, 13.3777,   # Brandenburg Gate
        52.5208, 13.4094,   # TV Tower
    )
    assert 2000 < dist < 2500
