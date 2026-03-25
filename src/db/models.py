"""SQLAlchemy ORM models for landmarks and inference_logs tables."""

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
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


class LandmarkChunk(Base):
    """A text chunk from a landmark's knowledge base, stored with its embedding."""

    __tablename__ = "landmark_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    landmark_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("landmarks.id"), nullable=False
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list] = mapped_column(Vector(384), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="db")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    landmark: Mapped["Landmark"] = relationship("Landmark")

    def __repr__(self) -> str:
        return f"<LandmarkChunk landmark_id={self.landmark_id} source={self.source!r}>"


class UserSession(Base):
    """A tour session for a user — tracks visited landmarks and persona."""

    __tablename__ = "user_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    persona: Mapped[str] = mapped_column(String(50), nullable=False, default="historian")
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    turns: Mapped[list["ConversationTurn"]] = relationship(
        "ConversationTurn", back_populates="session", order_by="ConversationTurn.created_at"
    )

    def __repr__(self) -> str:
        return f"<UserSession {self.id} persona={self.persona!r}>"


class ConversationTurn(Base):
    """A single message turn in a user session."""

    __tablename__ = "conversation_turns"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("user_sessions.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    landmark_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    session: Mapped["UserSession"] = relationship("UserSession", back_populates="turns")

    def __repr__(self) -> str:
        return f"<ConversationTurn session={self.session_id} role={self.role!r}>"
