"""Shared pytest fixtures."""

import pytest
from PIL import Image


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
