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
