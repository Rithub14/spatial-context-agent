"""Database engine, session factory, and FastAPI dependency."""

import logging

from pgvector.sqlalchemy import Vector  # noqa: F401 — registers vector type with SQLAlchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.db.models import Base

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,   # reconnect on stale connections
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """Create all tables if they do not already exist."""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")


def get_db():
    """FastAPI dependency that yields a database session and closes it after use."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """Return True if the database is reachable, False otherwise."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return False
