"""CLIP model loading and zero-shot image classification."""

import logging
from typing import Any

import clip
import torch
from PIL import Image

from src.config import settings

logger = logging.getLogger(__name__)


class CLIPInference:
    """Wraps OpenAI CLIP for zero-shot scene classification."""

    def __init__(self) -> None:
        """Load the CLIP model and preprocessing transform onto the configured device."""
        self.device = self._resolve_device(settings.device)
        logger.info("Loading CLIP model '%s' on device '%s'", settings.clip_model_name, self.device)
        self.model, self.preprocess = clip.load(settings.clip_model_name, device=self.device)
        self.model.eval()
        logger.info("CLIP model loaded successfully.")

    def classify_scene(
        self,
        image: Image.Image,
        categories: list[str],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Classify a PIL image against a list of text category labels.

        Args:
            image: Input image as a PIL Image object.
            categories: Text labels to score the image against.
            top_k: Number of top results to return.

        Returns:
            List of dicts sorted by descending confidence, e.g.:
            [{"category": "monument", "confidence": 0.87}, ...]
        """
        if not categories:
            raise ValueError("categories list must not be empty.")

        top_k = min(top_k, len(categories))

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(categories).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

            # Normalise and compute scaled cosine similarity.
            # logit_scale is CLIP's learned temperature — without it softmax
            # produces near-uniform scores because raw cosine values are too close.
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * (image_features @ text_features.T).squeeze(0)
            probs = logits.softmax(dim=-1)

        top_probs, top_indices = probs.topk(top_k)

        results = [
            {
                "category": categories[idx.item()],
                "confidence": round(prob.item(), 4),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        logger.debug("classify_scene top-%d: %s", top_k, results)
        return results

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Return 'cuda' if available and requested, otherwise 'cpu'."""
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested == "cuda":
            logger.warning("CUDA requested but not available — falling back to CPU.")
        return "cpu"
