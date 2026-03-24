"""Manual smoke test for SceneClassifier.

Run from project root:
    python scripts/test_clip.py
    python scripts/test_clip.py --image path/to/your/photo.jpg
"""

import argparse
import sys
import time
from pathlib import Path

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.pipeline.scene_classifier import SceneClassifier


def create_test_image() -> Image.Image:
    """Create a simple gradient image as a stand-in test input."""
    import urllib.request

    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Brandenburger_Tor_abends.jpg/320px-Brandenburger_Tor_abends.jpg"
    print(f"Downloading test image from Wikipedia... ({url})")
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            from io import BytesIO
            image = Image.open(BytesIO(response.read())).convert("RGB")
        print(f"Downloaded image: {image.size[0]}x{image.size[1]} px")
        return image
    except Exception as e:
        print(f"Download failed ({e}), generating synthetic image instead.")
        # Simple gradient so CLIP gets something non-trivial
        import numpy as np
        arr = np.zeros((224, 224, 3), dtype="uint8")
        arr[:, :, 0] = np.linspace(50, 200, 224)   # red gradient
        arr[:, :, 2] = np.linspace(200, 50, 224)   # blue gradient
        return Image.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual CLIP scene classifier test")
    parser.add_argument("--image", type=str, default=None, help="Path to a local image file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to show")
    args = parser.parse_args()

    # --- Load image ---
    if args.image:
        path = Path(args.image)
        if not path.exists():
            print(f"Error: file not found: {path}")
            sys.exit(1)
        image = Image.open(path).convert("RGB")
        print(f"Loaded local image: {path.name} ({image.size[0]}x{image.size[1]} px)")
    else:
        image = create_test_image()

    # --- Load model ---
    print("\nLoading CLIP model (this may take a moment on first run)...")
    t0 = time.perf_counter()
    classifier = SceneClassifier()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_ms:.0f} ms")

    # --- Run inference ---
    print(f"\nRunning inference (top_k={args.top_k})...")
    t1 = time.perf_counter()
    result = classifier.classify(image, top_k=args.top_k)
    infer_ms = (time.perf_counter() - t1) * 1000

    # --- Print results ---
    primary = result["primary"]
    print(f"\n{'='*40}")
    print(f"  PRIMARY:  {primary['category']!r}  ({primary['confidence']:.1%})")
    print(f"{'='*40}")
    if result["alternatives"]:
        print("  Alternatives:")
        for alt in result["alternatives"]:
            bar = "█" * int(alt["confidence"] * 30)
            print(f"    {alt['category']:<30} {alt['confidence']:.1%}  {bar}")
    print(f"\nInference time: {infer_ms:.1f} ms")


if __name__ == "__main__":
    main()
