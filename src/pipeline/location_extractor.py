"""EXIF GPS extraction from PIL images."""

import logging
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# EXIF tag IDs
_TAG_GPS_INFO = 34853   # GPSInfo IFD pointer
_GPS_LATITUDE = 2
_GPS_LATITUDE_REF = 1
_GPS_LONGITUDE = 4
_GPS_LONGITUDE_REF = 3


class LocationExtractor:
    """Extracts GPS coordinates from image EXIF metadata."""

    @staticmethod
    def extract_gps(image: Image.Image) -> tuple[float, float] | None:
        """Read GPS latitude and longitude from a PIL image's EXIF data.

        Args:
            image: PIL Image object (JPEG with embedded EXIF GPS tags).

        Returns:
            (latitude, longitude) as decimal degrees, or None if unavailable.
        """
        try:
            exif_data = image._getexif()  # type: ignore[attr-defined]
        except AttributeError:
            logger.debug("Image format does not support EXIF (_getexif missing).")
            return None

        if not exif_data:
            logger.debug("No EXIF data found in image.")
            return None

        gps_info: dict[int, Any] | None = exif_data.get(_TAG_GPS_INFO)
        if not gps_info:
            logger.debug("EXIF data present but no GPSInfo tag.")
            return None

        try:
            lat = LocationExtractor._dms_to_decimal(
                gps_info[_GPS_LATITUDE],
                gps_info[_GPS_LATITUDE_REF],
            )
            lng = LocationExtractor._dms_to_decimal(
                gps_info[_GPS_LONGITUDE],
                gps_info[_GPS_LONGITUDE_REF],
            )
        except (KeyError, TypeError, ZeroDivisionError) as exc:
            logger.warning("Corrupted GPS EXIF tags: %s", exc)
            return None

        if lat is None or lng is None:
            return None

        logger.debug("Extracted GPS from EXIF: (%.6f, %.6f)", lat, lng)
        return lat, lng

    @staticmethod
    def _dms_to_decimal(dms: tuple, ref: str) -> float | None:
        """Convert DMS (degrees/minutes/seconds) to decimal degrees.

        Handles both raw rational tuples ``(numerator, denominator)`` (from
        piexif / manual mocks) and Pillow ``IFDRational`` objects (returned by
        ``_getexif()`` in modern Pillow builds).

        Args:
            dms: Three-element sequence of either (num, denom) tuples or IFDRational.
            ref: Hemisphere — 'N', 'S', 'E', or 'W'.

        Returns:
            Decimal degrees (negative for S/W), or None on invalid input.
        """
        try:
            degrees = LocationExtractor._rational_to_float(dms[0])
            minutes = LocationExtractor._rational_to_float(dms[1])
            seconds = LocationExtractor._rational_to_float(dms[2])
        except (IndexError, TypeError, ZeroDivisionError) as exc:
            logger.warning("Invalid DMS tuple: %s — %s", dms, exc)
            return None

        if degrees is None or minutes is None or seconds is None:
            return None

        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal

    @staticmethod
    def _rational_to_float(value: object) -> float | None:
        """Convert a rational value to float.

        Accepts:
        - ``(numerator, denominator)`` tuple (piexif / manual)
        - Pillow ``IFDRational`` (supports ``float()`` directly)
        """
        if isinstance(value, tuple):
            num, denom = value
            if denom == 0:
                return None
            return num / denom
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            logger.warning("Cannot convert rational value %r to float: %s", value, exc)
            return None
