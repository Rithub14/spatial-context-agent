"""Analyze endpoint — core spatial context pipeline + LangGraph agentic endpoint."""

import base64
import io
import json
import logging
import time
import uuid as _uuid
from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
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

# In-memory location store (session_id → last known location context)
_session_coords: dict[str, dict] = {}
# shape: {"lat": float, "lng": float, "landmark": str | None, "scene": str}


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
    _session_coords[session_id] = {
        "lat": lat, "lng": lng,
        "landmark": landmark_name,
        "scene": result["scene_result"]["primary"]["category"],
    }

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


@router.post("/agent/stream")
def agent_stream(
    request: AgentAnalyzeRequest,
    db: Annotated[Session, Depends(get_db)],
) -> StreamingResponse:
    """Stream narration token-by-token via Server-Sent Events.

    Runs vision + context nodes synchronously, then streams GPT narration.

    SSE event types:
      {"type": "step",  "content": "👁️ Scene identified: ..."}
      {"type": "token", "content": "partial narration text"}
      {"type": "done",  "session_id": "...", "scene": {...}, "location": {...}, ...}
    """
    from src.agent.tools import (
        classify_scene_tool,
        find_landmark_tool,
        retrieve_knowledge_tool,
        reverse_geocode_tool,
    )

    # --- Decode image + GPS (must happen before StreamingResponse) ---
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

    # --- Run vision node synchronously ---
    t_start = time.perf_counter()
    scene_result = classify_scene_tool.invoke({"image_b64": request.image})
    step_trace: list[str] = [
        f"👁️ Scene identified: {scene_result['primary']['category']} "
        f"({scene_result['primary']['confidence']:.0%} confidence)"
    ]

    # --- Run context node synchronously ---
    landmark = find_landmark_tool.invoke({"latitude": lat, "longitude": lng})
    if landmark.get("found"):
        step_trace.append(
            f"📍 Landmark found: {landmark['name']} ({landmark['distance_meters']:.0f}m away)"
        )
        knowledge = retrieve_knowledge_tool.invoke({
            "landmark_name": landmark["name"],
            "query": f"{landmark['name']} {scene_result['primary']['category']} history",
            "landmark_id": landmark["id"],
        })
        step_trace.append(f"📚 Retrieved knowledge context ({len(knowledge)} chars)")
    else:
        step_trace.append("📍 No local landmark — querying Foursquare Places API...")
        knowledge = reverse_geocode_tool.invoke({"latitude": lat, "longitude": lng})
        step_trace.append(f"🌍 Foursquare: {knowledge[:80]}...")

    # --- Build narration prompt (same as narration_node) ---
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from src.pipeline.narration_engine import PERSONA_PROMPTS

    system_prompt = PERSONA_PROMPTS.get(request.persona, PERSONA_PROMPTS["historian"])
    system_prompt += "\n\nKeep your response under 300 characters. Be vivid but concise."

    context_parts: list[str] = []
    if landmark.get("found"):
        context_parts.append(
            f"Landmark: {landmark['name']}, {landmark.get('district', '')}, "
            f"{landmark.get('city', 'Berlin')}. "
            f"Category: {landmark.get('category', '')}. "
            f"Historical period: {landmark.get('historical_period', '')}. "
            f"Distance: {landmark.get('distance_meters', 0):.0f}m from user."
        )
    if knowledge:
        context_parts.append(f"Knowledge base context:\n{knowledge[:800]}")

    scene_type = scene_result["primary"]["category"]
    confidence = scene_result["primary"]["confidence"]
    user_content = (
        f"Location: ({lat:.4f}, {lng:.4f})\n"
        f"Scene: {scene_type} ({confidence:.0%} confidence)\n"
        + ("\n".join(context_parts) if context_parts else "No landmark data available.")
        + "\n\nGenerate a tour guide narration."
    )

    messages: list = [SystemMessage(content=system_prompt)]
    for turn in history[-4:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=user_content))

    # Snapshot values for use inside the generator (avoids closure over mutable db)
    _session_id = session_id
    _lat, _lng = lat, lng
    _landmark = landmark
    _scene_result = scene_result
    _step_trace = list(step_trace)
    _inference_start = t_start
    _persona = request.persona

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    def generate() -> Generator[str, None, None]:
        from langchain_openai import ChatOpenAI

        from src.db.session import SessionLocal

        # 1 — emit step trace
        for step in _step_trace:
            yield _sse({"type": "step", "content": step})

        # 2 — stream narration tokens
        full_narration = ""
        if settings.openai_api_key:
            llm = ChatOpenAI(
                model=settings.gpt_model,
                api_key=settings.openai_api_key,
                max_tokens=300,
                streaming=True,
            )
            for chunk in llm.stream(messages):
                token = chunk.content
                if token:
                    full_narration += token
                    yield _sse({"type": "token", "content": token})
            _step_trace.append(f"🎙️ Narration streamed by {settings.gpt_model} ({_persona} persona)")
        else:
            from src.pipeline.narration_engine import NarrationEngine
            lm = _landmark if _landmark.get("found") else None
            full_narration = NarrationEngine()._template_narration(
                _scene_result, lm, (_lat, _lng), 300
            )
            yield _sse({"type": "token", "content": full_narration})
            _step_trace.append("🎙️ Narration generated via template (no OpenAI key)")

        inference_ms = int((time.perf_counter() - _inference_start) * 1000)

        # 3 — persist to DB using a fresh session
        try:
            fresh_db = SessionLocal()
            lm_name = _landmark.get("name") if _landmark.get("found") else None
            _add_session_turn(_session_id, "assistant", full_narration, lm_name, fresh_db)
            fresh_db.close()
        except Exception as exc:
            logger.error("stream: failed to persist session turn: %s", exc)

        _session_traces[_session_id] = _step_trace
        _session_coords[_session_id] = {
            "lat": _lat, "lng": _lng,
            "landmark": _landmark.get("name") if _landmark.get("found") else None,
            "scene": _scene_result["primary"]["category"],
        }

        # 4 — emit done event with full metadata
        lm_data = _landmark if _landmark.get("found") else {}
        yield _sse({
            "type": "done",
            "session_id": _session_id,
            "narration": full_narration,
            "scene": _scene_result,
            "location": {
                "nearest_landmark": lm_data.get("name"),
                "distance_meters": lm_data.get("distance_meters"),
                "district": lm_data.get("district"),
                "city": lm_data.get("city"),
            },
            "step_trace": _step_trace,
            "metadata": {
                "model_version": settings.clip_model_name,
                "inference_time_ms": inference_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "coordinates": {"latitude": _lat, "longitude": _lng},
                "persona": _persona,
            },
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/agent/followup")
def agent_followup(
    request: FollowupRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Answer a follow-up question with intent detection and routing.

    Detects user intent (nearby_places, historical_facts, directions, etc.) and
    routes to the appropriate tool or LLM strategy before answering.
    """
    import datetime

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from src.pipeline.intent_detector import (
        INTENT_CURRENT_EVENTS,
        INTENT_HISTORICAL_FACTS,
        INTENT_MOVED,
        INTENT_NEARBY_PLACES,
        INTENT_OPENING_HOURS,
        IntentDetector,
    )
    from src.pipeline.narration_engine import PERSONA_PROMPTS

    history = _get_session_history(request.session_id, db)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    today = datetime.date.today().strftime("%B %d, %Y")  # e.g. "March 28, 2026"

    # --- Detect intent ---
    intent_result = IntentDetector().detect(request.question)
    intent = intent_result.intent
    logger.info(
        "followup session=%s intent=%s method=%s",
        request.session_id[:8],
        intent,
        intent_result.method,
    )

    # --- Short-circuit: user has moved, ask for new image ---
    if intent == INTENT_MOVED:
        answer = (
            "Got it — upload a new photo from your current location and click "
            "Analyze so I can orient myself. I'll pick up the tour from there!"
        )
        _add_session_turn(request.session_id, "user", request.question, None, db)
        _add_session_turn(request.session_id, "assistant", answer, None, db)
        return {
            "session_id": request.session_id,
            "answer": answer,
            "intent": intent,
            "intent_confidence": round(intent_result.confidence, 2),
        }

    # --- Inject last known location into system context ---
    loc_ctx = _session_coords.get(request.session_id)
    location_block = ""
    if loc_ctx:
        lm_line = f"near {loc_ctx['landmark']}" if loc_ctx["landmark"] else ""
        location_block = (
            f"The user's current location (from their last uploaded photo): "
            f"({loc_ctx['lat']:.5f}, {loc_ctx['lng']:.5f}) {lm_line}. "
            f"Scene type: {loc_ctx['scene']}. "
            "Use this as 'the user's location' when answering distance or direction questions. "
            "If asked how far something is, estimate relative to these coordinates."
        )

    # --- Tool-augmented context for specific intents ---
    tool_context = ""
    if intent == INTENT_NEARBY_PLACES:
        tool_context = _get_nearby_context(history)
    elif intent == INTENT_HISTORICAL_FACTS:
        tool_context = _get_historical_context(request.session_id, history, db)
    elif intent == INTENT_OPENING_HOURS:
        tool_context = (
            "Note: I don't have real-time opening hours. Advise the user to check "
            "the official website or Google Maps for current hours and ticket prices."
        )
    elif intent == INTENT_CURRENT_EVENTS:
        tool_context = _get_current_events_context(loc_ctx, today)

    # --- Build LLM messages ---
    system_prompt = PERSONA_PROMPTS.get(request.persona, PERSONA_PROMPTS["historian"])
    intent_instruction = _intent_instruction(intent)
    # Always ground the LLM with today's date so it can't hallucinate stale events
    system_prompt = (
        f"{system_prompt}\n\n"
        f"Today's date is {today}. "
        "Only reference events, exhibitions, or information that are current as of this date. "
        "Never invent or guess event dates — use the web search context provided.\n\n"
        f"{intent_instruction}"
    )
    if location_block:
        system_prompt += f"\n\n{location_block}"
    if tool_context:
        system_prompt += "\n\nLive web search results (use these, they are current):\n"
        system_prompt += tool_context

    messages: list = [SystemMessage(content=system_prompt)]
    for turn in history[-6:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
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

    return {
        "session_id": request.session_id,
        "answer": answer,
        "intent": intent,
        "intent_confidence": round(intent_result.confidence, 2),
    }


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
# Intent routing helpers
# ---------------------------------------------------------------------------

def _intent_instruction(intent: str) -> str:
    """Return a concise instruction suffix that steers the LLM for this intent."""
    from src.pipeline.intent_detector import (
        INTENT_CURRENT_EVENTS,
        INTENT_DIRECTIONS,
        INTENT_HISTORICAL_FACTS,
        INTENT_MOVED,
        INTENT_NEARBY_PLACES,
        INTENT_OPENING_HOURS,
        INTENT_PHOTO_TIP,
        INTENT_TELL_MORE,
        INTENT_TRANSLATION,
    )

    instructions = {
        INTENT_NEARBY_PLACES: (
            "The user wants to know what else is nearby. "
            "Suggest 2-3 specific nearby attractions with brief descriptions."
        ),
        INTENT_HISTORICAL_FACTS: (
            "The user wants historical facts. "
            "Focus on dates, key figures, and significant events. Be precise."
        ),
        INTENT_TELL_MORE: (
            "The user wants more detail about the current location. "
            "Expand on what was just said with interesting additional facts."
        ),
        INTENT_OPENING_HOURS: (
            "The user is asking about visiting logistics (hours, tickets, fees). "
            "Provide what you know and advise checking official sources for current info."
        ),
        INTENT_DIRECTIONS: (
            "The user wants directions or transit information. "
            "Describe how to reach the location from the city centre, "
            "mentioning relevant public transport options."
        ),
        INTENT_TRANSLATION: (
            "The user wants a translation. Provide the translation clearly "
            "and, if relevant, note any interesting linguistic context."
        ),
        INTENT_PHOTO_TIP: (
            "The user wants photography advice for this spot. "
            "Suggest the best angles, lighting conditions, and times of day."
        ),
        INTENT_MOVED: (
            "The user has moved to a new location. "
            "Ask them to upload a new photo so you can orient yourself."
        ),
        INTENT_CURRENT_EVENTS: (
            "The user wants to know about current or upcoming events and exhibitions. "
            "Use ONLY the live web search results provided — do not invent or recall events "
            "from training data. Summarise what is actually happening now, with dates."
        ),
    }
    return instructions.get(intent, "Answer the user's question concisely and engagingly.")


def _get_nearby_context(history: list[dict]) -> str:
    """Pull lat/lng from session trace and call get_nearby_places_tool."""
    # Extract the last known landmark from conversation history
    landmark_name = next(
        (t.get("landmark_name") for t in reversed(history) if t.get("landmark_name")),
        None,
    )
    if not landmark_name:
        return ""
    try:
        from src.agent.tools import get_nearby_places_tool
        from src.db.models import Landmark
        from src.db.session import SessionLocal

        db = SessionLocal()
        try:
            lm = db.query(Landmark).filter(Landmark.name == landmark_name).first()
            if lm:
                result = get_nearby_places_tool.invoke({
                    "latitude": lm.latitude,
                    "longitude": lm.longitude,
                    "category": "",
                })
                return result
        finally:
            db.close()
    except Exception as exc:
        logger.warning("_get_nearby_context failed: %s", exc)
    return ""


def _get_current_events_context(loc_ctx: dict | None, today: str) -> str:
    """Run a live web search for exhibitions/events near the user's location."""
    from src.agent.tools import web_search_tool

    location = "Berlin"
    if loc_ctx and loc_ctx.get("landmark"):
        location = loc_ctx["landmark"]
    elif loc_ctx:
        location = f"Berlin ({loc_ctx['lat']:.3f}, {loc_ctx['lng']:.3f})"

    query = f"{location} exhibitions events happening {today}"
    try:
        return web_search_tool.invoke({"query": query})
    except Exception as exc:
        logger.warning("_get_current_events_context failed: %s", exc)
        return ""


def _get_historical_context(session_id: str, history: list[dict], db) -> str:
    """Retrieve RAG knowledge for the landmark in the current session."""
    from src.agent.memory import SessionMemory

    visited = SessionMemory(db).get_visited_landmarks(session_id)
    if not visited:
        return ""
    try:
        from src.db.models import Landmark
        from src.pipeline.rag_retriever import RAGRetriever

        landmark_name = visited[-1]
        lm = db.query(Landmark).filter(Landmark.name == landmark_name).first()
        if lm:
            chunks = RAGRetriever(db).retrieve(
                query=f"{landmark_name} history construction architecture",
                landmark_id=str(lm.id),
                top_k=2,
            )
            return "\n\n".join(chunks) if chunks else ""
    except Exception as exc:
        logger.warning("_get_historical_context failed: %s", exc)
    return ""


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
