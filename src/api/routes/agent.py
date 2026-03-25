"""Analyze endpoint — core spatial context pipeline."""

import base64
import io
import logging
import time
import uuid as _uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
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
