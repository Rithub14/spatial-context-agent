"""Build the RAG knowledge base by embedding landmark text + Wikipedia summaries.

Run once (or re-run to refresh):
    python -m src.db.build_knowledge_base
"""

import logging
import sys

import wikipediaapi

from src.db.models import Landmark, LandmarkChunk
from src.db.session import SessionLocal, create_tables
from src.pipeline.embedder import EmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_CHUNK_SIZE = 400  # max chars per text chunk


def _chunk_text(text: str, max_chars: int = _CHUNK_SIZE) -> list[str]:
    """Split text into chunks of roughly max_chars, respecting sentence boundaries."""
    sentences = text.replace("\n", " ").split(". ")
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = (current + ". " + sentence).strip(". ")
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]


def _fetch_wikipedia_summary(landmark_name: str, city: str = "Berlin") -> str | None:
    """Fetch the first section of a Wikipedia article for a landmark."""
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="SpatialContextAgent/1.0 (portfolio project)",
    )
    # Try exact name first, then "Name, City"
    for query in [landmark_name, f"{landmark_name}, {city}"]:
        page = wiki.page(query)
        if page.exists():
            summary = page.summary
            if summary:
                logger.info("Wikipedia found: %r (%d chars)", query, len(summary))
                return summary[:2000]  # cap at 2000 chars
    logger.warning("Wikipedia: no article found for %r", landmark_name)
    return None


def build(force: bool = False) -> None:
    """Embed all landmarks and their Wikipedia summaries into landmark_chunks."""
    create_tables()
    db = SessionLocal()
    embedder = EmbeddingModel()

    try:
        landmarks = db.query(Landmark).all()
        if not landmarks:
            logger.error("No landmarks in DB. Run `python -m src.db.seed` first.")
            sys.exit(1)

        logger.info("Building knowledge base for %d landmarks...", len(landmarks))

        for landmark in landmarks:
            # Skip if already embedded (unless force)
            existing = (
                db.query(LandmarkChunk)
                .filter(LandmarkChunk.landmark_id == landmark.id)
                .count()
            )
            if existing > 0 and not force:
                logger.info("Skipping %r — already embedded (%d chunks)", landmark.name, existing)
                continue

            if force:
                db.query(LandmarkChunk).filter(
                    LandmarkChunk.landmark_id == landmark.id
                ).delete()

            chunks_added = 0

            # 1. Embed DB content (description + narration template)
            db_text = f"{landmark.name}. {landmark.description} {landmark.narration_template}"
            for chunk in _chunk_text(db_text):
                db.add(LandmarkChunk(
                    landmark_id=landmark.id,
                    chunk_text=chunk,
                    embedding=embedder.encode(chunk),
                    source="db",
                ))
                chunks_added += 1

            # 2. Embed Wikipedia content
            wiki_text = _fetch_wikipedia_summary(landmark.name, landmark.city)
            if wiki_text:
                for chunk in _chunk_text(wiki_text):
                    db.add(LandmarkChunk(
                        landmark_id=landmark.id,
                        chunk_text=chunk,
                        embedding=embedder.encode(chunk),
                        source="wikipedia",
                    ))
                    chunks_added += 1

            db.commit()
            logger.info("✓ %r — %d chunks embedded", landmark.name, chunks_added)

        total = db.query(LandmarkChunk).count()
        logger.info("Knowledge base complete. Total chunks: %d", total)

    finally:
        db.close()


if __name__ == "__main__":
    force = "--force" in sys.argv
    build(force=force)
