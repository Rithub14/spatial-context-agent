"""Smoke test — validates a deployed Spatial Context Agent endpoint.

Usage:
    python scripts/smoke_test.py --url http://localhost:8000
    python scripts/smoke_test.py --url https://my-api.azurecontainer.io:8000 --api-key secret

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed (safe for CI/CD integration).
"""

import argparse
import base64
import io
import sys

import httpx
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Print PASS/FAIL for a named check and record failures."""
    status = PASS if condition else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if not condition:
        _failures.append(name)


def _make_sample_b64_image() -> str:
    """Create a minimal valid base64-encoded JPEG."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 149, 237)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_health(client: httpx.Client) -> None:
    print("\n── /health ──────────────────────────────────────────────────────")
    try:
        r = client.get("/health", timeout=10)
        check("status 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            for field in ("status", "model_loaded", "db_connected", "uptime_seconds"):
                check(f"field '{field}' present", field in data)
    except Exception as exc:
        check("health reachable", False, str(exc))


def check_analyze(client: httpx.Client) -> None:
    print("\n── POST /api/v1/analyze ─────────────────────────────────────────")
    payload = {
        "image": _make_sample_b64_image(),
        "latitude": 52.5163,   # Brandenburg Gate
        "longitude": 13.3777,
    }
    try:
        r = client.post("/api/v1/analyze", json=payload, timeout=60)
        check("status 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            check("field 'scene' present", "scene" in data)
            check("field 'location' present", "location" in data)
            check("field 'narration' present", "narration" in data)
            check("field 'metadata' present", "metadata" in data)

            scene = data.get("scene", {})
            check("scene.primary is str", isinstance(scene.get("primary"), str))
            check(
                "scene.confidence in [0, 1]",
                isinstance(scene.get("confidence"), float)
                and 0.0 <= scene["confidence"] <= 1.0,
            )
            check("narration is non-empty str", bool(data.get("narration")))

            meta = data.get("metadata", {})
            check("metadata.inference_time_ms present", "inference_time_ms" in meta)
            check("metadata.coordinates present", "coordinates" in meta)
    except Exception as exc:
        check("/analyze reachable", False, str(exc))


def check_analyze_no_gps(client: httpx.Client) -> None:
    print("\n── POST /api/v1/analyze (no GPS → expect 400) ───────────────────")
    try:
        r = client.post(
            "/api/v1/analyze",
            json={"image": _make_sample_b64_image()},
            timeout=60,
        )
        check("status 400", r.status_code == 400, f"got {r.status_code}")
        check("error mentions GPS", "GPS" in r.json().get("detail", ""))
    except Exception as exc:
        check("/analyze no-gps reachable", False, str(exc))


def check_locations(client: httpx.Client) -> None:
    print("\n── GET /api/v1/locations ────────────────────────────────────────")
    try:
        r = client.get("/api/v1/locations", timeout=10)
        check("status 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            check("field 'total' present", "total" in data)
            check("field 'items' is list", isinstance(data.get("items"), list))
    except Exception as exc:
        check("/locations reachable", False, str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for Spatial Context Agent.")
    parser.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--api-key", default="", help="X-API-Key header value (if auth enabled)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    print("\nSpatial Context Agent — Smoke Test")
    print(f"Target: {base_url}")

    with httpx.Client(base_url=base_url, headers=headers) as client:
        check_health(client)
        check_analyze(client)
        check_analyze_no_gps(client)
        check_locations(client)

    print("\n" + "─" * 65)
    if _failures:
        print(f"  RESULT: {len(_failures)} check(s) FAILED: {', '.join(_failures)}")
        sys.exit(1)
    else:
        print("  RESULT: All checks passed ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
