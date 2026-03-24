"""Tests for LocationExtractor — EXIF GPS parsing."""

import io
from unittest.mock import patch

import piexif
from PIL import Image

from src.pipeline.location_extractor import LocationExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decimal_to_dms_rational(
    decimal: float,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Convert decimal degrees to rational DMS tuples for piexif."""
    decimal = abs(decimal)
    degrees = int(decimal)
    minutes_float = (decimal - degrees) * 60
    minutes = int(minutes_float)
    seconds = int((minutes_float - minutes) * 60 * 10000)
    return ((degrees, 1), (minutes, 1), (seconds, 10000))


def _make_jpeg_with_gps(lat: float, lng: float) -> Image.Image:
    """Create a minimal JPEG in memory with GPS EXIF tags embedded."""
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S",
        piexif.GPSIFD.GPSLatitude: _decimal_to_dms_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: b"E" if lng >= 0 else b"W",
        piexif.GPSIFD.GPSLongitude: _decimal_to_dms_rational(lng),
    }
    exif_bytes = piexif.dump({"GPS": gps_ifd})
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif_bytes)
    buf.seek(0)
    return Image.open(buf)


def _make_plain_jpeg() -> Image.Image:
    """Create a minimal JPEG with no EXIF (but has _getexif method)."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(50, 50, 50)).save(buf, format="JPEG")
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extract_gps_returns_tuple_for_valid_image() -> None:
    """extract_gps must return a (lat, lng) tuple for a JPEG with GPS EXIF."""
    image = _make_jpeg_with_gps(52.5163, 13.3777)
    result = LocationExtractor.extract_gps(image)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_extract_gps_berlin_coordinates_accuracy() -> None:
    """Extracted Berlin coordinates must be accurate to within 0.001 degrees."""
    lat, lng = 52.5163, 13.3777
    image = _make_jpeg_with_gps(lat, lng)
    result = LocationExtractor.extract_gps(image)
    assert result is not None
    assert abs(result[0] - lat) < 0.001
    assert abs(result[1] - lng) < 0.001


def test_extract_gps_returns_none_for_image_without_exif() -> None:
    """extract_gps must return None for a plain PNG with no EXIF."""
    image = Image.new("RGB", (64, 64))
    result = LocationExtractor.extract_gps(image)
    assert result is None


def test_extract_gps_returns_none_when_getexif_returns_none() -> None:
    """extract_gps must return None when _getexif() returns None."""
    image = _make_plain_jpeg()
    with patch.object(image, "_getexif", return_value=None):
        result = LocationExtractor.extract_gps(image)
    assert result is None


def test_extract_gps_returns_none_for_exif_without_gps() -> None:
    """extract_gps must return None when EXIF has no GPSInfo tag."""
    image = _make_plain_jpeg()
    with patch.object(image, "_getexif", return_value={0x010F: "TestCamera"}):
        result = LocationExtractor.extract_gps(image)
    assert result is None


def test_extract_gps_returns_none_for_corrupted_gps_tags() -> None:
    """extract_gps must return None when GPSInfo values are malformed."""
    image = _make_plain_jpeg()
    corrupt_gps = {2: "bad", 1: "N", 4: "bad", 3: "E"}
    with patch.object(image, "_getexif", return_value={34853: corrupt_gps}):
        result = LocationExtractor.extract_gps(image)
    assert result is None


def test_extract_gps_returns_none_for_no_getexif_method() -> None:
    """extract_gps must return None when the image has no _getexif method at all."""
    image = Image.new("RGB", (64, 64))   # no _getexif on plain Image
    result = LocationExtractor.extract_gps(image)
    assert result is None


def test_dms_to_decimal_north() -> None:
    """DMS conversion must produce correct positive decimal for North latitude."""
    dms = ((52, 1), (30, 1), (5868, 100))   # 52° 30' 58.68" N ≈ 52.5163
    result = LocationExtractor._dms_to_decimal(dms, "N")
    assert result is not None
    assert abs(result - 52.5163) < 0.001


def test_dms_to_decimal_south_is_negative() -> None:
    """DMS conversion must produce negative decimal for South latitude."""
    dms = ((33, 1), (52, 1), (0, 1))
    result = LocationExtractor._dms_to_decimal(dms, "S")
    assert result is not None
    assert result < 0


def test_dms_to_decimal_west_is_negative() -> None:
    """DMS conversion must produce negative decimal for West longitude."""
    dms = ((73, 1), (59, 1), (0, 1))
    result = LocationExtractor._dms_to_decimal(dms, "W")
    assert result is not None
    assert result < 0


def test_dms_to_decimal_zero_denominator_returns_none() -> None:
    """_dms_to_decimal must return None when a denominator is zero."""
    dms = ((52, 0), (30, 1), (0, 1))
    result = LocationExtractor._dms_to_decimal(dms, "N")
    assert result is None
