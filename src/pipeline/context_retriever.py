"""Nearest landmark retrieval using the Haversine formula."""

import logging
import math
from typing import Any

from sqlalchemy.orm import Session

from src.db.models import Landmark

logger = logging.getLogger(__name__)

_EARTH_RADIUS_KM = 6371.0


class ContextRetriever:
    """Finds the nearest landmark to a given GPS coordinate using Haversine distance."""

    def __init__(self, db: Session) -> None:
        """Initialise with an active SQLAlchemy session.

        Args:
            db: Active database session (injected via FastAPI dependency).
        """
        self.db = db

    def find_nearest_landmark(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
    ) -> dict[str, Any] | None:
        """Return the nearest landmark within radius_km, or None if none found.

        Fetches all landmarks and computes Haversine distance in Python.
        Sufficient for a city-scale dataset (~20 landmarks).

        Args:
            latitude:  Query latitude in decimal degrees.
            longitude: Query longitude in decimal degrees.
            radius_km: Maximum search radius in kilometres (default 5 km).

        Returns:
            Dict with all landmark fields plus ``distance_meters`` (int),
            or None if no landmark falls within the radius.
        """
        landmarks: list[Landmark] = self.db.query(Landmark).all()
        if not landmarks:
            logger.debug("No landmarks in database.")
            return None

        nearest: Landmark | None = None
        nearest_dist_m: float = float("inf")

        for lm in landmarks:
            dist_m = self.haversine_distance_m(
                latitude, longitude, lm.latitude, lm.longitude
            )
            if dist_m < nearest_dist_m:
                nearest_dist_m = dist_m
                nearest = lm

        if nearest is None or nearest_dist_m > radius_km * 1000:
            logger.debug(
                "No landmark within %.1f km of (%.5f, %.5f).", radius_km, latitude, longitude
            )
            return None

        logger.debug(
            "Nearest landmark: %r at %.0f m.", nearest.name, nearest_dist_m
        )
        return {
            "id": str(nearest.id),
            "name": nearest.name,
            "description": nearest.description,
            "latitude": nearest.latitude,
            "longitude": nearest.longitude,
            "city": nearest.city,
            "district": nearest.district,
            "category": nearest.category,
            "historical_period": nearest.historical_period,
            "narration_template": nearest.narration_template,
            "image_url": nearest.image_url,
            "distance_meters": int(nearest_dist_m),
        }

    @staticmethod
    def haversine_distance_m(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Compute the great-circle distance in metres between two GPS points.

        Args:
            lat1, lon1: First point in decimal degrees.
            lat2, lon2: Second point in decimal degrees.

        Returns:
            Distance in metres.
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return _EARTH_RADIUS_KM * c * 1000
