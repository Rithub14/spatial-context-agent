"""Tests for NarrationEngine."""

from src.pipeline.narration_engine import NarrationEngine

COORDS = (52.5163, 13.3777)

SCENE_RESULT = {
    "primary": {"category": "monument", "confidence": 0.75},
    "alternatives": [{"category": "historic building", "confidence": 0.15}],
}

LANDMARK = {
    "name": "Brandenburg Gate",
    "narration_template": "You're standing before the Brandenburg Gate.",
    "historical_period": "18th century",
    "district": "Mitte",
    "city": "Berlin",
    "distance_meters": 50.0,
    "id": "00000000-0000-0000-0000-000000000001",
}


def test_generate_with_landmark_contains_scene_type() -> None:
    """Narration should mention the detected scene type."""
    engine = NarrationEngine()
    result = engine.generate(SCENE_RESULT, LANDMARK, COORDS)
    assert "monument" in result


def test_generate_with_landmark_contains_template() -> None:
    """Narration should include the DB narration template."""
    engine = NarrationEngine()
    result = engine.generate(SCENE_RESULT, LANDMARK, COORDS)
    assert "Brandenburg Gate" in result


def test_generate_with_landmark_contains_historical_period() -> None:
    """Narration should append the historical period in parentheses."""
    engine = NarrationEngine()
    result = engine.generate(SCENE_RESULT, LANDMARK, COORDS)
    assert "18th century" in result


def test_generate_without_landmark_mentions_coordinates() -> None:
    """Unknown-location narration should mention the GPS coordinates."""
    engine = NarrationEngine()
    result = engine.generate(SCENE_RESULT, None, COORDS)
    assert "52.5163" in result
    assert "13.3777" in result


def test_generate_without_landmark_mentions_scene_type() -> None:
    """Unknown-location narration should still mention the scene type."""
    engine = NarrationEngine()
    result = engine.generate(SCENE_RESULT, None, COORDS)
    assert "monument" in result


def test_generate_respects_max_length() -> None:
    """Narration must not exceed max_length characters."""
    engine = NarrationEngine()
    result = engine.generate(SCENE_RESULT, LANDMARK, COORDS, max_length=50)
    assert len(result) <= 50


def test_generate_landmark_without_historical_period() -> None:
    """No historical period suffix when field is empty."""
    lm = {**LANDMARK, "historical_period": None}
    engine = NarrationEngine()
    result = engine.generate(SCENE_RESULT, lm, COORDS)
    assert "(" not in result


def test_generate_returns_string() -> None:
    """generate() must always return a str."""
    engine = NarrationEngine()
    assert isinstance(engine.generate(SCENE_RESULT, None, COORDS), str)
    assert isinstance(engine.generate(SCENE_RESULT, LANDMARK, COORDS), str)
