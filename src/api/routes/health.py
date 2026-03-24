"""Health check endpoint."""

import logging
import time

from fastapi import APIRouter

from src.api.schemas.response import HealthResponse
from src.db.session import check_db_connection
from src.pipeline.scene_classifier import SceneClassifier

logger = logging.getLogger(__name__)
router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return model status, DB connectivity, and uptime."""
    model_loaded = SceneClassifier._clip is not None
    db_ok = check_db_connection()

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        model_loaded=model_loaded,
        db_connected=db_ok,
        uptime_seconds=round(time.time() - _start_time, 1),
    )
