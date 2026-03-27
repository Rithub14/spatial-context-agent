"""Analyze endpoint — core spatial context pipeline + LangGraph agentic endpoint."""

import base64
import io
import logging
import time
import uuid as _uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.schemas.request import AnalyzeRequest, LandmarkCreateRequest
from src.api.schemas.response import AnalyzeResponse, LocationResult, SceneResult
from src.config import settings
from src.db.models import InferenceLog, Landmark
from src.db.session import get_db
from src.pipeline.context_retriever import ContextRetriever
from src.pipeline.location_extractor import LocationExtractor
from src.pipeline.narration_engine import NarrationEngine
from src.pipeline.rag_retriever import RAGRetriever
from src.pipeline.scene_classifier import SceneClassifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

_classifier = SceneClassifier()

# In-memory session trace store (session_id → step_trace list)
_session_traces: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Pydantic models for agentic endpoints
# ---------------------------------------------------------------------------

class AgentAnalyzeRequest(BaseModel):
    """Request for the LangGraph agentic analyze endpoint."""

    image: str
    latitude: float | None = None
    longitude: float | None = None
    persona: str = "historian"
    session_id: str | None = None


class FollowupRequest(BaseModel):
    """Request for conversational follow-up questions."""

    session_id: str
    question: str
    persona: str = "historian"


# ---------------------------------------------------------------------------
# LangGraph agentic endpoint
# ---------------------------------------------------------------------------

@router.post("/agent/analyze")
def agent_analyze(
    request: AgentAnalyzeRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Run the full LangGraph multi-agent pipeline on an image + location.

    Returns narration, scene result, landmark, step trace, and session_id.
    """
    from src.agent.graph import get_graph

    t_start = time.perf_counter()

    # Resolve GPS
    try:
        image_bytes = base64.b64decode(request.image)
        raw_image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    lat = request.latitude
    lng = request.longitude
    if lat is None or lng is None:
        coords = LocationExtractor.extract_gps(raw_image)
        if coords:
            lat, lng = coords
        else:
            raise HTTPException(
                status_code=400,
                detail="No GPS data. Provide latitude/longitude or an image with EXIF GPS.",
            )

    session_id = _ensure_session(request.session_id, request.persona, db)
    history = _get_session_history(session_id, db)

    state = {
        "image_b64": request.image,
        "latitude": lat,
        "longitude": lng,
        "persona": request.persona,
        "session_id": session_id,
        "conversation_history": history,
        "scene_result": {},
        "landmark": {},
        "knowledge_chunks": "",
        "nearby_places": "",
        "narration": "",
        "step_trace": [],
    }

    result = get_graph().invoke(state)
    inference_ms = int((time.perf_counter() - t_start) * 1000)

    # Persist trace and conversation turn
    _session_traces[session_id] = result["step_trace"]
    lm = result.get("landmark", {})
    landmark_name = lm.get("name") if lm.get("found") else None
    _add_session_turn(session_id, "assistant", result["narration"], landmark_name, db)

    landmark = result.get("landmark", {})
    return {
        "session_id": session_id,
        "narration": result["narration"],
        "scene": result["scene_result"],
        "location": {
            "nearest_landmark": landmark.get("name") if landmark.get("found") else None,
            "distance_meters": landmark.get("distance_meters"),
            "district": landmark.get("district"),
            "city": landmark.get("city"),
        },
        "step_trace": result["step_trace"],
        "metadata": {
            "model_version": settings.clip_model_name,
            "inference_time_ms": inference_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "coordinates": {"latitude": lat, "longitude": lng},
            "persona": request.persona,
        },
    }


@router.post("/agent/followup")
def agent_followup(
    request: FollowupRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Answer a follow-up question using conversation history (no image needed)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from src.pipeline.narration_engine import PERSONA_PROMPTS

    history = _get_session_history(request.session_id, db)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    system_prompt = PERSONA_PROMPTS.get(request.persona, PERSONA_PROMPTS["historian"])
    system_prompt += " Answer the user's follow-up question concisely."

    messages: list = [SystemMessage(content=system_prompt)]
    for turn in history[-6:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=request.question))

    if settings.openai_api_key:
        llm = ChatOpenAI(model=settings.gpt_model, api_key=settings.openai_api_key,
                         max_tokens=300)
        answer = llm.invoke(messages).content
    else:
        answer = "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."

    _add_session_turn(request.session_id, "user", request.question, None, db)
    _add_session_turn(request.session_id, "assistant", answer, None, db)

    return {"session_id": request.session_id, "answer": answer}


@router.post("/agent/speak")
def agent_speak(body: dict) -> Response:
    """Convert a narration string to MP3 audio and return it as a binary response.

    Request body: {"text": "...", "language": "en"}
    Returns: audio/mpeg binary stream ready for playback.
    """
    from src.pipeline.tts_engine import TTSEngine

    text = body.get("text", "").strip()
    language = body.get("language", "en")

    if not text:
        raise HTTPException(status_code=400, detail="'text' field is required.")

    try:
        audio_bytes = TTSEngine().synthesize(text, language)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/agent/trace/{session_id}")
def get_trace(session_id: str) -> dict:
    """Return the step trace for a given session (agent thinking steps)."""
    trace = _session_traces.get(session_id, [])
    return {"session_id": session_id, "step_trace": trace}


# ---------------------------------------------------------------------------
# Session helpers — DB-backed via SessionMemory
# ---------------------------------------------------------------------------

def _get_session_history(session_id: str, db: Session) -> list[dict]:
    from src.agent.memory import SessionMemory
    return SessionMemory(db).get_history(session_id)


def _ensure_session(session_id: str | None, persona: str, db: Session) -> str:
    """Return existing session_id or create a new one in the DB."""
    from src.agent.memory import SessionMemory
    mem = SessionMemory(db)
    if session_id:
        existing = mem.get_session(session_id)
        if existing:
            return session_id
    return mem.create_session(persona=persona)


def _add_session_turn(
    session_id: str, role: str, content: str, landmark_name: str | None, db: Session
) -> None:
    from src.agent.memory import SessionMemory
    SessionMemory(db).add_turn(session_id, role, content, landmark_name)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    db: Annotated[Session, Depends(get_db)],
) -> AnalyzeResponse:
    """Classify scene, find nearest landmark, and return narration.

    GPS resolution order:
    1. Explicit lat/lng in request body
    2. EXIF GPS extracted from image
    3. 400 error
    """
    t_start = time.perf_counter()

    # --- Decode image ---
    try:
        image_bytes = base64.b64decode(request.image)
        raw_image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    # --- Resolve GPS (before convert("RGB") which strips EXIF) ---
    lat: float | None = request.latitude
    lng: float | None = request.longitude

    if lat is None or lng is None:
        coords = LocationExtractor.extract_gps(raw_image)
        if coords:
            lat, lng = coords
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No GPS data available. Provide latitude/longitude or "
                    "upload an image with GPS EXIF metadata."
                ),
            )

    # Convert to RGB for CLIP (strips EXIF, but GPS already resolved)
    image = raw_image.convert("RGB")

    # --- Scene classification ---
    scene_result = _classifier.classify(image, top_k=5)
    primary = scene_result["primary"]

    # --- Nearest landmark ---
    retriever = ContextRetriever(db)
    landmark = retriever.find_nearest_landmark(lat, lng)

    # --- RAG context retrieval ---
    context_chunks: list[str] = []
    if landmark:
        try:
            rag = RAGRetriever(db)
            context_chunks = rag.retrieve(
                query=f"{landmark['name']} {scene_result['primary']['category']}",
                landmark_id=landmark["id"],
                top_k=3,
            )
        except Exception as exc:
            logger.warning("RAG retrieval failed, continuing without context: %s", exc)

    # --- Narration ---
    narration = NarrationEngine().generate(
        scene_result,
        landmark,
        (lat, lng),
        request.max_narration_length,
        persona=request.persona,
        context_chunks=context_chunks,
    )

    # --- Inference time ---
    inference_ms = int((time.perf_counter() - t_start) * 1000)

    # --- Log to DB ---
    _log_inference(db, lat, lng, primary, inference_ms, landmark)

    logger.info(
        "analyze lat=%.4f lng=%.4f scene=%s landmark=%s inference_ms=%d",
        lat, lng, primary["category"],
        landmark["name"] if landmark else "none",
        inference_ms,
    )

    return AnalyzeResponse(
        scene=SceneResult(
            primary=primary["category"],
            confidence=primary["confidence"],
            alternatives=scene_result["alternatives"],
        ),
        location=LocationResult(
            nearest_landmark=landmark["name"] if landmark else None,
            distance_meters=landmark["distance_meters"] if landmark else None,
            district=landmark["district"] if landmark else None,
            city=landmark["city"] if landmark else None,
        ),
        narration=narration,
        metadata={
            "model_version": settings.clip_model_name,
            "inference_time_ms": inference_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "coordinates": {"latitude": lat, "longitude": lng},
            "persona": request.persona,
            "rag_chunks_used": len(context_chunks),
        },
    )


@router.get("/locations")
def list_locations(
    db: Annotated[Session, Depends(get_db)],
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """List all landmarks with pagination."""
    total = db.query(Landmark).count()
    landmarks = db.query(Landmark).offset(offset).limit(limit).all()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "id": str(lm.id),
                "name": lm.name,
                "latitude": lm.latitude,
                "longitude": lm.longitude,
                "district": lm.district,
                "category": lm.category,
            }
            for lm in landmarks
        ],
    }


@router.post("/locations", status_code=201)
def create_location(
    body: LandmarkCreateRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Add a new landmark to the database."""
    lm = Landmark(
        name=body.name,
        description=body.description,
        latitude=body.latitude,
        longitude=body.longitude,
        city=body.city,
        district=body.district,
        category=body.category,
        historical_period=body.historical_period,
        narration_template=body.narration_template,
        image_url=body.image_url,
    )
    db.add(lm)
    db.commit()
    db.refresh(lm)
    logger.info("Created landmark %r (id=%s)", lm.name, lm.id)
    return {"id": str(lm.id), "name": lm.name}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_inference(
    db: Session,
    lat: float,
    lng: float,
    primary: dict,
    inference_ms: int,
    landmark: dict | None,
) -> None:
    """Write an inference log entry to the database."""
    try:
        log = InferenceLog(
            latitude=lat,
            longitude=lng,
            predicted_scene=primary["category"],
            confidence=primary["confidence"],
            matched_landmark_id=_uuid.UUID(landmark["id"]) if landmark else None,
            inference_time_ms=inference_ms,
            model_version=settings.clip_model_name,
        )
        db.add(log)
        db.commit()
    except Exception as exc:
        logger.error("Failed to write inference log: %s", exc)
        db.rollback()
