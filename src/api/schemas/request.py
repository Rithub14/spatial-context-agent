"""Pydantic request models."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for POST /api/v1/analyze."""

    image: str = Field(..., description="Base64-encoded image (JPEG or PNG).")
    latitude: float | None = Field(None, description="Explicit latitude in decimal degrees.")
    longitude: float | None = Field(None, description="Explicit longitude in decimal degrees.")
    language: str = Field("en", description="Response language code (reserved for future use).")
    max_narration_length: int = Field(
        200, ge=50, le=1000, description="Maximum narration length in characters."
    )
    persona: str = Field(
        "historian",
        description="Tour guide persona: historian | storyteller | local | child_friendly.",
    )
    session_id: str | None = Field(None, description="Session ID for conversation continuity.")


class LandmarkCreateRequest(BaseModel):
    """Request body for POST /api/v1/locations."""

    name: str = Field(..., max_length=255)
    description: str
    latitude: float
    longitude: float
    city: str = Field("Berlin", max_length=100)
    district: str = Field(..., max_length=100)
    category: str = Field(..., max_length=100)
    historical_period: str | None = Field(None, max_length=100)
    narration_template: str
    image_url: str | None = Field(None, max_length=500)
