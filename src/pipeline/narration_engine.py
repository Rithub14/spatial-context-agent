"""Tour guide narration engine — template-based, no LLM dependency."""

import logging

logger = logging.getLogger(__name__)


class NarrationEngine:
    """Generates contextual tour guide narrations from scene + landmark data.

    Two modes:
    - Known landmark: DB template + dynamic scene intro + historical period suffix.
    - Unknown location: Generic narration from scene classification alone.
    """

    def generate(
        self,
        scene_result: dict,
        landmark: dict | None,
        coordinates: tuple[float, float],
        max_length: int = 300,
    ) -> str:
        """Generate a tour guide narration string.

        Args:
            scene_result: Output from SceneClassifier.classify() —
                          {'primary': {'category': str, 'confidence': float}, 'alternatives': [...]}
            landmark: Nearest landmark dict from ContextRetriever, or None.
            coordinates: (latitude, longitude) of the query point.
            max_length: Maximum character length of the returned narration.

        Returns:
            Narration string, truncated to max_length.
        """
        scene_type = scene_result["primary"]["category"]

        if landmark:
            narration = self._known_landmark_narration(scene_type, landmark)
        else:
            narration = self._unknown_location_narration(scene_type, coordinates)

        logger.info(
            "Narration generated (landmark=%s, length=%d)",
            landmark["name"] if landmark else "none",
            len(narration),
        )
        return narration[:max_length]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _known_landmark_narration(self, scene_type: str, landmark: dict) -> str:
        """Build narration for a recognised landmark."""
        intro = f"I can see you're looking at a {scene_type}. "
        body = landmark["narration_template"]
        period = landmark.get("historical_period") or ""
        suffix = f" ({period})" if period else ""
        return intro + body + suffix

    def _unknown_location_narration(
        self, scene_type: str, coordinates: tuple[float, float]
    ) -> str:
        """Build a generic narration when no DB landmark matched."""
        lat, lng = coordinates
        return (
            f"You appear to be looking at a {scene_type}. "
            f"This location ({lat:.4f}, {lng:.4f}) doesn't match a known landmark "
            "in our database, but the scene suggests an interesting urban environment "
            "worth exploring."
        )
