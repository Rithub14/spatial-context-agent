"""Template-based narration engine for the spatial tour guide pipeline."""

import logging

logger = logging.getLogger(__name__)


class NarrationEngine:
    """Generates tour guide narrations from scene classification and landmark data.

    Operates in two modes:
    - Known landmark: uses the DB narration_template, enriched with scene context.
    - Unknown location: builds a generic narration from scene type and coordinates.
    """

    def generate(
        self,
        scene_result: dict,
        landmark: dict | None,
        coordinates: tuple[float, float],
    ) -> str:
        """Generate a narration string for the given scene and location.

        Args:
            scene_result: Output from SceneClassifier.classify(), with 'primary'
                          key containing 'category' and 'confidence'.
            landmark: Nearest landmark dict from ContextRetriever, or None.
            coordinates: (latitude, longitude) tuple for the current position.

        Returns:
            A tour guide narration string.
        """
        scene_type = scene_result["primary"]["category"]
        confidence = scene_result["primary"]["confidence"]

        if landmark:
            narration = self._known_landmark_narration(scene_type, confidence, landmark)
        else:
            narration = self._unknown_location_narration(scene_type, coordinates)

        logger.info(
            "Generated narration — scene: %s, landmark: %s",
            scene_type,
            landmark.get("name") if landmark else "unknown",
        )
        return narration

    def _known_landmark_narration(
        self,
        scene_type: str,
        confidence: float,
        landmark: dict,
    ) -> str:
        """Build narration for a known landmark using its DB template.

        Args:
            scene_type: Primary scene category from CLIP.
            confidence: Confidence score for the primary scene.
            landmark: Landmark dict including name, narration_template,
                      historical_period, district, city.

        Returns:
            Enriched narration string.
        """
        parts: list[str] = []

        # Dynamic intro from CLIP classification
        if confidence >= 0.5:
            parts.append(f"I can see you're looking at a {scene_type}.")
        else:
            parts.append(f"This appears to be a {scene_type}.")

        # Core narration from DB template
        template = landmark.get("narration_template", "")
        if template:
            parts.append(template)

        # Historical context suffix
        period = landmark.get("historical_period", "")
        if period:
            parts.append(f"This site dates from the {period}.")

        return " ".join(parts)

    def _unknown_location_narration(
        self,
        scene_type: str,
        coordinates: tuple[float, float],
    ) -> str:
        """Build a generic narration when no landmark is found in the DB.

        Args:
            scene_type: Primary scene category from CLIP.
            coordinates: (latitude, longitude) of the current position.

        Returns:
            Generic location-aware narration string.
        """
        lat, lng = coordinates
        lat_dir = "N" if lat >= 0 else "S"
        lng_dir = "E" if lng >= 0 else "W"

        return (
            f"You're exploring a {scene_type} at coordinates "
            f"{abs(lat):.4f}°{lat_dir}, {abs(lng):.4f}°{lng_dir}. "
            "This location isn't in our landmarks database yet, but keep exploring — "
            "every great city has hidden gems waiting to be discovered."
        )
