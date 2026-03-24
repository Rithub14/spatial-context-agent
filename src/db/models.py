"""SQLAlchemy ORM models for landmarks and inference_logs tables."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Shared declarative base for all models."""


class Landmark(Base):
    """A real-world point of interest used for spatial context matching."""

    __tablename__ = "landmarks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False, default="Berlin")
    district: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    historical_period: Mapped[str] = mapped_column(String(100), nullable=True)
    narration_template: Mapped[str] = mapped_column(Text, nullable=False)
    image_url: Mapped[str] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    inference_logs: Mapped[list["InferenceLog"]] = relationship(
        "InferenceLog", back_populates="matched_landmark"
    )

    # Composite index for proximity queries (lat/lng range scans)
    __table_args__ = (
        Index("ix_landmarks_lat_lng", "latitude", "longitude"),
    )

    def __repr__(self) -> str:
        return f"<Landmark {self.name!r} ({self.latitude}, {self.longitude})>"


class InferenceLog(Base):
    """Audit log of every /analyze request for monitoring and analytics."""

    __tablename__ = "inference_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_scene: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    matched_landmark_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("landmarks.id"), nullable=True
    )
    inference_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    matched_landmark: Mapped["Landmark | None"] = relationship(
        "Landmark", back_populates="inference_logs"
    )

    def __repr__(self) -> str:
        return f"<InferenceLog {self.predicted_scene!r} @ ({self.latitude}, {self.longitude})>"
