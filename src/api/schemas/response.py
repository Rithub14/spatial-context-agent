"""Pydantic response models."""


from pydantic import BaseModel


class SceneResult(BaseModel):
    """CLIP scene classification result."""

    primary: str
    confidence: float
    alternatives: list[dict]


class LocationResult(BaseModel):
    """Nearest landmark lookup result."""

    nearest_landmark: str | None
    distance_meters: int | None
    district: str | None
    city: str | None


class AnalyzeResponse(BaseModel):
    """Full response from POST /api/v1/analyze."""

    scene: SceneResult
    location: LocationResult
    narration: str
    metadata: dict


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str
    model_loaded: bool
    db_connected: bool
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str
    detail: str | None = None
