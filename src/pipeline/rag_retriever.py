"""RAG retriever — semantic search over landmark knowledge chunks via pgvector."""

import logging

from sqlalchemy.orm import Session

from src.db.models import LandmarkChunk
from src.pipeline.embedder import EmbeddingModel

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant knowledge chunks for a landmark using vector similarity.

    Uses pgvector cosine distance to find the most relevant text chunks
    from the landmark_chunks table given a natural language query.
    """

    def __init__(self, db: Session) -> None:
        """Initialise with a database session.

        Args:
            db: SQLAlchemy session.
        """
        self.db = db
        self._embedder = EmbeddingModel()

    def retrieve(
        self,
        query: str,
        landmark_id: str | None = None,
        top_k: int = 3,
    ) -> list[str]:
        """Retrieve the most relevant text chunks for a query.

        Args:
            query: Natural language query (e.g. "history of the gate").
            landmark_id: If provided, restrict search to chunks for this landmark.
            top_k: Number of chunks to return.

        Returns:
            List of chunk text strings, most relevant first.
        """
        query_embedding = self._embedder.encode(query)

        q = self.db.query(LandmarkChunk)
        if landmark_id:
            q = q.filter(LandmarkChunk.landmark_id == landmark_id)

        # pgvector cosine distance operator: <=>
        results = (
            q.order_by(LandmarkChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
            .all()
        )

        chunks = [r.chunk_text for r in results]
        logger.debug(
            "RAG retrieved %d chunks for query %r (landmark_id=%s)",
            len(chunks), query, landmark_id,
        )
        return chunks
