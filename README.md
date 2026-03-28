# Spatial Context Agent

An autonomous AI tour guide for cultural and heritage sites — point your phone at any landmark and get a real-time, persona-driven narration with conversational follow-ups.

![CI](https://github.com/Rithub14/spatial-context-agent/actions/workflows/ci-cd.yml/badge.svg)

---

## What it does

A tourist photographs the Brandenburg Gate. The system extracts GPS from the photo's EXIF metadata, classifies the scene with CLIP, finds the nearest landmark in its knowledge base, retrieves contextual facts via RAG, and streams a narration through GPT-4o-mini — all in one API call. The user can then ask follow-up questions in natural language; the agent detects intent, queries live web results when needed, and responds in character.

**Core loop:**

```
Photo → EXIF GPS → CLIP scene → nearest landmark → RAG context → GPT narration (streamed)
                                                                       ↓
                                              follow-up chat with intent detection + web search
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│   Upload photo  ──►  SSE stream narration  ──►  Chat follow-ups │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────────┐
│                     FastAPI  (port 8001)                        │
│                                                                 │
│  POST /agent/stream          POST /agent/followup               │
│         │                           │                           │
│  ┌──────▼──────────────────┐  ┌─────▼──────────────────────┐   │
│  │    LangGraph Pipeline   │  │   Intent Detector          │   │
│  │  vision → context →     │  │   keyword + GPT-4o-mini    │   │
│  │  narration (streamed)   │  │   routing to tools         │   │
│  └──────┬──────────────────┘  └─────┬──────────────────────┘   │
│         │                           │                           │
│  ┌──────▼────────┐  ┌───────────────▼──────────────────────┐   │
│  │  CLIP ViT-B/32│  │  Tools                               │   │
│  │  scene class. │  │  • RAG retriever (pgvector)          │   │
│  └───────────────┘  │  • Foursquare reverse geocoder       │   │
│                     │  • DuckDuckGo web search (live)      │   │
│  ┌──────────────┐   │  • Nearby places (DB)                │   │
│  │  PostgreSQL  │   └──────────────────────────────────────┘   │
│  │  landmarks   │                                               │
│  │  pgvector    │   Session memory (DB-backed)                  │
│  │  conv. turns │   Location context injected per request       │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Vision | OpenAI CLIP (ViT-B/32) via PyTorch — zero-shot scene classification |
| Orchestration | LangGraph `StateGraph` — vision → context → narration nodes |
| LLM | GPT-4o-mini via LangChain (`langchain-openai`) |
| RAG | pgvector + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Geolocation | EXIF GPS extraction + Foursquare Places API v3 |
| Web search | DuckDuckGo (`duckduckgo-search`) — live results, no API key |
| API | FastAPI + Uvicorn, SSE via `StreamingResponse` |
| Database | PostgreSQL 16 + pgvector extension |
| ORM | SQLAlchemy 2.0 with `Mapped` / `mapped_column` |
| Session memory | DB-backed (`UserSession` + `ConversationTurn` models) |
| Containerisation | Docker multi-stage build + docker-compose |
| CI | GitHub Actions — ruff lint + 84 pytest tests on every push |
| Demo UI | Streamlit with live SSE token streaming |
| Testing | pytest, SQLite `StaticPool`, all CLIP calls mocked |
| Linting | Ruff |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Rithub14/spatial-context-agent.git
cd spatial-context-agent

# 2. Copy env and fill in your keys
cp .env.example .env
# Required: OPENAI_API_KEY
# Optional: FOURSQUARE_API_KEY (falls back to DB-only landmarks without it)

# 3. Start API + PostgreSQL (pgvector image)
docker-compose up --build -d

# 4. Seed 18 Berlin landmarks
docker-compose exec api python -m src.db.seed

# 5. (Optional) Build the RAG knowledge base from Wikipedia
docker-compose exec api python -m src.db.build_knowledge_base

# 6. Launch Streamlit
pip install streamlit httpx pandas Pillow
streamlit run streamlit_app/app.py
```

Or run the API locally without Docker:

```bash
uv venv && uv pip install -r requirements.txt
uvicorn src.api.main:app --port 8001 --reload
```

---

## API Endpoints

### `POST /api/v1/agent/stream` — main endpoint (SSE)

Runs the full LangGraph pipeline and streams narration token-by-token.

**Request**
```json
{
  "image": "<base64 JPEG/PNG>",
  "latitude": 52.5163,
  "longitude": 13.3777,
  "persona": "historian",
  "session_id": null
}
```
`latitude` / `longitude` are optional if the image contains GPS EXIF metadata.
`persona` options: `historian` · `storyteller` · `local` · `child_friendly`

**SSE event stream**
```
data: {"type": "step",  "content": "👁️ Scene identified: monument (87% confidence)"}
data: {"type": "step",  "content": "📍 Landmark found: Brandenburg Gate (42m away)"}
data: {"type": "step",  "content": "📚 Retrieved knowledge context (1842 chars)"}
data: {"type": "token", "content": "Standing before"}
data: {"type": "token", "content": " the iconic"}
...
data: {"type": "done",  "session_id": "...", "scene": {...}, "location": {...}, "metadata": {...}}
```

---

### `POST /api/v1/agent/followup` — conversational follow-ups

```json
{
  "session_id": "abc123",
  "question": "any exhibitions happening nearby?",
  "persona": "historian"
}
```

**Response**
```json
{
  "session_id": "abc123",
  "answer": "As of March 2026, the Pergamon Museum is...",
  "intent": "current_events",
  "intent_confidence": 1.0
}
```

**Detected intents:**

| Intent | Triggered by | Action |
|---|---|---|
| `nearby_places` | "what else is nearby?" | DB proximity search |
| `historical_facts` | "when was it built?" | RAG knowledge retrieval |
| `current_events` | "any exhibitions?" | Live DuckDuckGo web search |
| `opening_hours` | "when does it open?" | LLM with advisory note |
| `directions` | "how do I get there?" | LLM using last known GPS |
| `translation` | "say that in German" | LLM |
| `photo_tip` | "best angle for a photo?" | LLM |
| `moved` | "I've moved" | Prompt to upload new photo |
| `general` | anything else | LLM |

The user's GPS coordinates from their last uploaded photo are injected into every follow-up so distance questions ("how far is X?") are answered relative to their actual location.

Today's date is always injected so the LLM cannot hallucinate stale event information.

---

### `POST /api/v1/analyze` — non-streaming (legacy)

Same pipeline as `/agent/stream` but returns a single JSON response. Useful for testing or non-streaming clients.

### `GET /api/v1/locations`

Paginated list of all landmarks in the database.

### `POST /api/v1/locations`

Add a landmark (requires `X-API-Key` header when `ENABLE_AUTH=true`).

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "db_connected": true, "uptime_seconds": 142.7}
```

---

## Project Structure

```
spatial-context-agent/
├── src/
│   ├── api/
│   │   ├── main.py                  # FastAPI app, lifespan, middleware wiring
│   │   ├── routes/
│   │   │   ├── agent.py             # All agent endpoints + intent routing helpers
│   │   │   └── health.py            # GET /health
│   │   ├── middleware/
│   │   │   ├── auth.py              # API key validation (toggleable)
│   │   │   └── rate_limiter.py      # Per-IP sliding window (toggleable)
│   │   └── schemas/
│   │       ├── request.py           # Pydantic request models
│   │       └── response.py          # Pydantic response models
│   ├── agent/
│   │   ├── graph.py                 # LangGraph StateGraph (vision→context→narration)
│   │   ├── tools.py                 # LangChain @tool wrappers for pipeline components
│   │   └── memory.py                # DB-backed session memory (UserSession, ConversationTurn)
│   ├── pipeline/
│   │   ├── clip_inference.py        # CLIP model loading + logit-scaled inference
│   │   ├── scene_classifier.py      # Zero-shot classification (12 categories)
│   │   ├── location_extractor.py    # EXIF GPS extraction (IFDRational-safe)
│   │   ├── context_retriever.py     # Haversine nearest-landmark lookup
│   │   ├── narration_engine.py      # GPT-4o-mini narration + template fallback
│   │   ├── embedder.py              # sentence-transformers singleton
│   │   ├── rag_retriever.py         # pgvector cosine similarity retrieval
│   │   └── intent_detector.py       # Intent classification (LLM + keyword fallback)
│   ├── db/
│   │   ├── models.py                # Landmark, InferenceLog, LandmarkChunk,
│   │   │                            # UserSession, ConversationTurn
│   │   ├── seed.py                  # 18 Berlin landmarks with GPS + narration templates
│   │   ├── build_knowledge_base.py  # Embeds DB text + Wikipedia into pgvector
│   │   └── session.py               # Engine, SessionLocal, get_db
│   └── config.py                    # pydantic-settings (all config from env vars)
├── tests/                           # 84 tests — CLIP always mocked, SQLite StaticPool
├── streamlit_app/
│   └── app.py                       # Streamlit demo: SSE streaming + chat interface
├── scripts/
│   └── smoke_test.py                # Manual health + analyze + locations check
├── docker/
│   └── init-pgvector.sql            # CREATE EXTENSION vector (runs on DB init)
├── .github/workflows/
│   └── ci-cd.yml                    # CI: ruff + pytest on every push/PR
├── Dockerfile                       # Multi-stage: builder (gcc, git) + runtime
├── docker-compose.yml               # pgvector/pgvector:pg16 + FastAPI
├── requirements.txt
└── .env.example
```

---

## Key Design Decisions

**CLIP for zero-shot scene classification**
No labelled training data or model retraining needed per city. New landmark categories = update a Python list. ViT-B/32 balances accuracy (~87% on landmark scenes) and CPU inference speed (~300 ms).

**LangGraph for orchestration**
Three-node `StateGraph` (vision → context → narration) gives a clear audit trail via `step_trace`, makes each stage independently testable, and mirrors the autonomous agent pattern used in production spatial platforms.

**pgvector RAG over a pure LLM approach**
Wikipedia summaries and DB descriptions are chunked, embedded with `all-MiniLM-L6-v2`, and stored as 384-dim vectors. Cosine similarity retrieval grounds the narration in factual content, reducing hallucinations without expensive fine-tuning.

**Intent detection before every follow-up**
Classifying the user's question first (8 intents) lets the system call the right tool — web search for current events, RAG for history, DB proximity for nearby places — rather than throwing everything at the LLM and hoping.

**DuckDuckGo for live web search**
Zero cost, no API key, no rate-limit account required. Today's date is always injected into the system prompt so the LLM cannot answer "any exhibitions?" with stale 2023 training data.

**SSE streaming over a single JSON response**
Step trace events appear as each node completes (giving the user visual feedback that the agent is working), then narration tokens stream in character-by-character. This matches the real-time feel of production spatial agents.

**GPS from EXIF, not typed input**
Every smartphone photo already has GPS metadata. Auto-extraction removes friction for the tourist use case. Explicit lat/lng is the fallback; the last-known GPS from an uploaded photo is reused for all follow-up distance/direction questions.

**Toggleable auth + rate limiting**
`ENABLE_AUTH=false` + `ENABLE_RATE_LIMIT=false` for local dev, flip to `true` in production via env vars — no code changes needed.

---

## Testing

```bash
# Run all 84 tests
PYTHONPATH=. pytest tests/ -v

# With coverage
PYTHONPATH=. pytest --cov=src --cov-report=term-missing tests/

# Smoke test against a running instance
python scripts/smoke_test.py --url http://localhost:8001
```

CLIP is always mocked (slow to load). The test database uses SQLite in-memory with `StaticPool` so all connections share one instance. pgvector tests are skipped in SQLite mode.

---

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/spatial_agent

# Model
CLIP_MODEL_NAME=ViT-B/32
DEVICE=cpu

# API
API_HOST=0.0.0.0
API_PORT=8000

# Security (toggleable)
ENABLE_AUTH=false
ENABLE_RATE_LIMIT=false
API_KEY=dev-key-change-in-production
RATE_LIMIT_RPM=60

# Agentic system (required for LLM narration, intent detection, web search)
OPENAI_API_KEY=sk-...

# Foursquare (optional — enables worldwide reverse geocoding)
FOURSQUARE_API_KEY=...
```

---

## License

MIT
