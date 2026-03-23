# Spatial Context Agent

AI backend for location-aware spatial experiences. Combines CLIP (zero-shot computer vision) with geospatial context to classify what a user is looking at and generate contextual narrations — the core intelligence behind AR tour guides and spatial storytelling platforms.

**Pipeline:** Upload photo → EXIF GPS extracted → CLIP classifies scene → nearest landmark fetched from DB → narration returned as structured JSON.

## Tech Stack

Python 3.11 · FastAPI · PyTorch + CLIP (ViT-B/32) · PostgreSQL · Docker · Azure · Terraform

## Quickstart

```bash
# 1. Create and activate environment
uv venv --python 3.11 && source .venv/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Copy env vars
cp .env.example .env

# 4. Start API + database
docker-compose up -d

# 5. Seed Berlin landmarks
python -m src.db.seed
```

API available at `http://localhost:8000`. Health check at `http://localhost:8000/health`.

## Usage

```bash
# Analyze an image (with explicit coordinates)
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "<base64>", "latitude": 52.5163, "longitude": 13.3777}'

# Manual CLIP smoke test
python scripts/test_clip.py --image path/to/photo.jpg
```

## Development

```bash
pytest --cov=src tests/   # run tests
ruff check src/ tests/    # lint
```

## Deployment

Infrastructure is managed with Terraform targeting Azure (Container Instances + PostgreSQL Flexible Server). See `terraform/` and `.github/workflows/ci-cd.yml`.
