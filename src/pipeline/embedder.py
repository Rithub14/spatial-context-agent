"""Local sentence-transformer embedding model — free, no API key needed."""

import logging

from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

_DIMS = 384  # all-MiniLM-L6-v2 output dimensions


class EmbeddingModel:
    """Wraps sentence-transformers for text embedding.

    Singleton — model loaded once per process.
    Uses all-MiniLM-L6-v2 (384 dims, ~80MB, runs on CPU in <50ms per text).
    """

    _instance: "EmbeddingModel | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "EmbeddingModel":
        """Return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self) -> None:
        """Load the model on first use."""
        if self._model is None:
            logger.info("Loading embedding model %r...", settings.embedding_model)
            self._model = SentenceTransformer(settings.embedding_model)
            logger.info("Embedding model loaded.")

    def encode(self, text: str) -> list[float]:
        """Encode a single text string into a 384-dim embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            List of 384 floats.
        """
        self._load()
        return self._model.encode(text, normalize_embeddings=True).tolist()  # type: ignore[union-attr]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts in one forward pass.

        Args:
            texts: List of strings to embed.

        Returns:
            List of 384-dim float vectors.
        """
        self._load()
        return self._model.encode(  # type: ignore[union-attr]
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    @property
    def dims(self) -> int:
        """Embedding dimensionality."""
        return _DIMS
