"""Scene classification using CLIP with predefined spatial categories."""

import logging
from typing import Any

from PIL import Image

from src.pipeline.clip_inference import CLIPInference

logger = logging.getLogger(__name__)


class SceneClassifier:
    """Classifies a scene image against predefined spatial categories.

    Uses a singleton CLIPInference instance so the model is loaded once
    per process regardless of how many times SceneClassifier is instantiated.
    """

    CATEGORIES: list[str] = [
        "historic building",
        "monument",
        "park or garden",
        "church or cathedral",
        "museum",
        "street scene",
        "bridge",
        "plaza or square",
        "waterfront",
        "modern architecture",
        "memorial",
        "market or shopping area",
    ]

    _clip: CLIPInference | None = None

    @classmethod
    def _get_clip(cls) -> CLIPInference:
        """Return the shared CLIPInference instance, creating it on first call."""
        if cls._clip is None:
            logger.info("Initialising CLIPInference singleton.")
            cls._clip = CLIPInference()
        return cls._clip

    def classify(self, image: Image.Image, top_k: int = 3) -> dict[str, Any]:
        """Classify an image and return the primary scene type plus alternatives.

        Args:
            image: Input image as a PIL Image object.
            top_k: Total number of results to retrieve (1 primary + top_k-1 alternatives).

        Returns:
            {
                "primary":      {"category": str, "confidence": float},
                "alternatives": [{"category": str, "confidence": float}, ...],
            }
        """
        top_k = max(1, top_k)
        results = self._get_clip().classify_scene(image, self.CATEGORIES, top_k=top_k)
        return {
            "primary": results[0],
            "alternatives": results[1:],
        }
