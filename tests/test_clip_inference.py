"""Tests for CLIPInference — CLIP model is mocked to avoid slow loading."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.pipeline.clip_inference import CLIPInference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clip_mocks(num_categories: int = 5) -> tuple[MagicMock, MagicMock]:
    """Return (mock_model, mock_preprocess) that produce deterministic outputs."""
    mock_model = MagicMock()

    # encode_image / encode_text return unit tensors of shape (1, 512) / (N, 512)
    image_feat = torch.ones(1, 512)
    text_feat = torch.ones(num_categories, 512)
    mock_model.encode_image.return_value = image_feat
    mock_model.encode_text.return_value = text_feat

    mock_preprocess = MagicMock()
    # preprocess(image) should return a tensor that supports .unsqueeze(0)
    mock_preprocess.return_value = torch.zeros(3, 224, 224)

    return mock_model, mock_preprocess


@pytest.fixture
def clip_inference(scene_categories: list[str]) -> CLIPInference:
    """CLIPInference with the CLIP library patched out."""
    mock_model, mock_preprocess = _make_clip_mocks(len(scene_categories))
    with (
        patch("src.pipeline.clip_inference.clip.load", return_value=(mock_model, mock_preprocess)),
        patch("src.pipeline.clip_inference.clip.tokenize", return_value=torch.zeros(len(scene_categories), 77, dtype=torch.long)),
    ):
        instance = CLIPInference()
    # Attach mocks so individual tests can inspect calls
    instance._mock_model = mock_model  # type: ignore[attr-defined]
    return instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_sets_device(clip_inference: CLIPInference) -> None:
    """CLIPInference.device must be 'cpu' or 'cuda'."""
    assert clip_inference.device in {"cpu", "cuda"}


def test_classify_scene_returns_list_of_dicts(
    clip_inference: CLIPInference,
    sample_image: Image.Image,
    scene_categories: list[str],
) -> None:
    """classify_scene must return a list of dicts with 'category' and 'confidence'."""
    with patch("src.pipeline.clip_inference.clip.tokenize", return_value=torch.zeros(len(scene_categories), 77, dtype=torch.long)):
        results = clip_inference.classify_scene(sample_image, scene_categories)

    assert isinstance(results, list)
    assert len(results) > 0
    for item in results:
        assert "category" in item
        assert "confidence" in item
        assert isinstance(item["category"], str)
        assert isinstance(item["confidence"], float)


def test_classify_scene_confidence_sums_to_one(
    clip_inference: CLIPInference,
    sample_image: Image.Image,
    scene_categories: list[str],
) -> None:
    """Confidence values for all top-k results must sum to ≈1 when top_k == len(categories)."""
    with patch("src.pipeline.clip_inference.clip.tokenize", return_value=torch.zeros(len(scene_categories), 77, dtype=torch.long)):
        results = clip_inference.classify_scene(
            sample_image, scene_categories, top_k=len(scene_categories)
        )

    total = sum(r["confidence"] for r in results)
    assert abs(total - 1.0) < 1e-3, f"Confidences sum to {total}, expected ~1.0"


def test_classify_scene_top_k_limits_results(
    clip_inference: CLIPInference,
    sample_image: Image.Image,
    scene_categories: list[str],
) -> None:
    """classify_scene must return exactly top_k items when top_k < len(categories)."""
    with patch("src.pipeline.clip_inference.clip.tokenize", return_value=torch.zeros(len(scene_categories), 77, dtype=torch.long)):
        results = clip_inference.classify_scene(sample_image, scene_categories, top_k=2)

    assert len(results) == 2


def test_classify_scene_top_k_clamped_to_category_count(
    clip_inference: CLIPInference,
    sample_image: Image.Image,
    scene_categories: list[str],
) -> None:
    """Requesting more results than categories must not raise an error."""
    with patch("src.pipeline.clip_inference.clip.tokenize", return_value=torch.zeros(len(scene_categories), 77, dtype=torch.long)):
        results = clip_inference.classify_scene(
            sample_image, scene_categories, top_k=999
        )

    assert len(results) == len(scene_categories)


def test_classify_scene_results_sorted_descending(
    clip_inference: CLIPInference,
    sample_image: Image.Image,
    scene_categories: list[str],
) -> None:
    """Results must be ordered from highest to lowest confidence."""
    with patch("src.pipeline.clip_inference.clip.tokenize", return_value=torch.zeros(len(scene_categories), 77, dtype=torch.long)):
        results = clip_inference.classify_scene(
            sample_image, scene_categories, top_k=len(scene_categories)
        )

    confidences = [r["confidence"] for r in results]
    assert confidences == sorted(confidences, reverse=True)


def test_classify_scene_empty_categories_raises(
    clip_inference: CLIPInference,
    sample_image: Image.Image,
) -> None:
    """Passing an empty category list must raise ValueError."""
    with pytest.raises(ValueError, match="categories list must not be empty"):
        clip_inference.classify_scene(sample_image, [])


def test_resolve_device_returns_cpu_when_cuda_unavailable() -> None:
    """_resolve_device must fall back to 'cpu' when CUDA is not available."""
    with patch("src.pipeline.clip_inference.torch.cuda.is_available", return_value=False):
        device = CLIPInference._resolve_device("cuda")
    assert device == "cpu"


def test_resolve_device_returns_cuda_when_available() -> None:
    """_resolve_device must return 'cuda' when CUDA is available and requested."""
    with patch("src.pipeline.clip_inference.torch.cuda.is_available", return_value=True):
        device = CLIPInference._resolve_device("cuda")
    assert device == "cuda"
