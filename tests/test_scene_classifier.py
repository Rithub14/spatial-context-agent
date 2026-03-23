"""Tests for SceneClassifier — CLIPInference is mocked."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.pipeline.scene_classifier import SceneClassifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset the SceneClassifier singleton between tests to avoid state leakage."""
    SceneClassifier._clip = None
    yield
    SceneClassifier._clip = None


def _mock_clip(results: list[dict]) -> MagicMock:
    """Return a mock CLIPInference whose classify_scene returns `results`."""
    mock = MagicMock()
    mock.classify_scene.return_value = results
    return mock


def _patch_clip(results: list[dict]):
    """Patch CLIPInference in scene_classifier with a mock returning `results`."""
    return patch(
        "src.pipeline.scene_classifier.CLIPInference",
        return_value=_mock_clip(results),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_classify_returns_primary_and_alternatives(sample_image: Image.Image) -> None:
    """classify must return a dict with 'primary' and 'alternatives' keys."""
    mock_results = [
        {"category": "monument", "confidence": 0.60},
        {"category": "historic building", "confidence": 0.25},
        {"category": "plaza or square", "confidence": 0.15},
    ]
    with _patch_clip(mock_results):
        result = SceneClassifier().classify(sample_image, top_k=3)

    assert "primary" in result
    assert "alternatives" in result


def test_primary_has_highest_confidence(sample_image: Image.Image) -> None:
    """Primary result must have the highest confidence among all returned results."""
    mock_results = [
        {"category": "bridge", "confidence": 0.70},
        {"category": "waterfront", "confidence": 0.20},
        {"category": "park or garden", "confidence": 0.10},
    ]
    with _patch_clip(mock_results):
        result = SceneClassifier().classify(sample_image, top_k=3)

    primary_conf = result["primary"]["confidence"]
    for alt in result["alternatives"]:
        assert primary_conf >= alt["confidence"]


def test_alternatives_count_matches_top_k_minus_one(sample_image: Image.Image) -> None:
    """alternatives must contain top_k - 1 items."""
    mock_results = [
        {"category": "museum", "confidence": 0.50},
        {"category": "church or cathedral", "confidence": 0.30},
        {"category": "memorial", "confidence": 0.20},
    ]
    with _patch_clip(mock_results):
        result = SceneClassifier().classify(sample_image, top_k=3)

    assert len(result["alternatives"]) == 2


def test_top_k_one_returns_empty_alternatives(sample_image: Image.Image) -> None:
    """When top_k=1, alternatives must be an empty list."""
    with _patch_clip([{"category": "monument", "confidence": 1.0}]):
        result = SceneClassifier().classify(sample_image, top_k=1)

    assert result["alternatives"] == []


def test_primary_structure(sample_image: Image.Image) -> None:
    """primary must be a dict with 'category' (str) and 'confidence' (float) keys."""
    with _patch_clip([{"category": "waterfront", "confidence": 0.80}]):
        result = SceneClassifier().classify(sample_image, top_k=1)

    primary = result["primary"]
    assert isinstance(primary["category"], str)
    assert isinstance(primary["confidence"], float)


def test_singleton_clip_model_loaded_once(sample_image: Image.Image) -> None:
    """CLIPInference must be instantiated only once across multiple classify calls."""
    mock_clip_instance = _mock_clip([{"category": "monument", "confidence": 1.0}])

    with patch(
        "src.pipeline.scene_classifier.CLIPInference",
        return_value=mock_clip_instance,
    ) as mock_cls:
        classifier = SceneClassifier()
        classifier.classify(sample_image, top_k=1)
        classifier.classify(sample_image, top_k=1)
        classifier.classify(sample_image, top_k=1)

    mock_cls.assert_called_once()


def test_categories_constant_is_non_empty() -> None:
    """CATEGORIES must be a non-empty list of strings."""
    assert len(SceneClassifier.CATEGORIES) > 0
    assert all(isinstance(c, str) for c in SceneClassifier.CATEGORIES)
