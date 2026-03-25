"""Tests for NarrationEngine — always uses template path (OpenAI mocked out)."""

from unittest.mock import patch

import pytest

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


@pytest.fixture(autouse=True)
def force_template_path():
    """Patch settings so NarrationEngine always uses the template fallback."""
    with patch("src.pipeline.narration_engine.settings") as mock_settings:
        mock_settings.openai_api_key = ""  # no key → template path
        mock_settings.gpt_model = "gpt-4o-mini"
        mock_settings.default_persona = "historian"
        yield mock_settings


def test_generate_with_landmark_contains_scene_type() -> None:
    """Narration should mention the detected scene type."""
    result = NarrationEngine().generate(SCENE_RESULT, LANDMARK, COORDS)
    assert "monument" in result


def test_generate_with_landmark_contains_template() -> None:
    """Narration should include the DB narration template."""
    result = NarrationEngine().generate(SCENE_RESULT, LANDMARK, COORDS)
    assert "Brandenburg Gate" in result


def test_generate_with_landmark_contains_historical_period() -> None:
    """Narration should append the historical period."""
    result = NarrationEngine().generate(SCENE_RESULT, LANDMARK, COORDS)
    assert "18th century" in result


def test_generate_without_landmark_mentions_coordinates() -> None:
    """Unknown-location narration should mention the GPS coordinates."""
    result = NarrationEngine().generate(SCENE_RESULT, None, COORDS)
    assert "52.5163" in result
    assert "13.3777" in result


def test_generate_without_landmark_mentions_scene_type() -> None:
    """Unknown-location narration should still mention the scene type."""
    result = NarrationEngine().generate(SCENE_RESULT, None, COORDS)
    assert "monument" in result


def test_generate_respects_max_length() -> None:
    """Template narration must not exceed max_length characters."""
    result = NarrationEngine().generate(SCENE_RESULT, LANDMARK, COORDS, max_length=50)
    assert len(result) <= 50


def test_generate_landmark_without_historical_period() -> None:
    """No historical period suffix when field is empty."""
    lm = {**LANDMARK, "historical_period": None}
    result = NarrationEngine().generate(SCENE_RESULT, lm, COORDS)
    assert "18th century" not in result


def test_generate_returns_string() -> None:
    """generate() must always return a str."""
    engine = NarrationEngine()
    assert isinstance(engine.generate(SCENE_RESULT, None, COORDS), str)
    assert isinstance(engine.generate(SCENE_RESULT, LANDMARK, COORDS), str)
