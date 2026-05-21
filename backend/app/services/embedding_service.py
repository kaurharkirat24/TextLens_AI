"""Embedding services for Phase 3 semantic retrieval."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from app.core.config import settings


logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when embeddings cannot be generated."""


class EmbeddingService(Protocol):
    """Common interface for document and query embedding providers."""

    provider: str
    model_name: str
    batch_size: int

    def embed_texts(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        """Embed document chunks."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a search or QA query."""

    def get_dimension(self) -> int:
        """Return the configured/provider embedding dimension."""


class SentenceTransformerEmbeddingService:
    """Generate embeddings locally using SentenceTransformers."""

    provider = "sentence_transformer"
    _model_cache: dict[tuple[str, str], Any] = {}

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        device: str = "cpu",
    ) -> None:
        if SentenceTransformer is None:
            raise EmbeddingServiceError(
                "sentence-transformers not installed. Run 'pip install sentence-transformers'"
            )
        
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.device = device
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model to save memory until needed."""
        if self._model is None:
            cache_key = (self.model_name, self.device)
            if cache_key in self._model_cache:
                self._model = self._model_cache[cache_key]
                return self._model

            try:
                logger.info("Loading SentenceTransformer model: %s on %s", self.model_name, self.device)
                try:
                    self._model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
                except Exception:
                    logger.info("Model %s was not fully available locally; retrying with remote lookup", self.model_name)
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                self._model_cache[cache_key] = self._model
            except Exception as exc:
                logger.exception("Failed to load SentenceTransformer model")
                raise EmbeddingServiceError(f"Failed to load model {self.model_name}: {exc}") from exc
        return self._model

    def embed_texts(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        """Embed texts in batches using the local model."""
        if not texts:
            return []

        start = time.perf_counter()
        try:
            # sentence-transformers encode handles batching internally
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Convert numpy array to list of lists for compatibility
            result = embeddings.tolist()
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Generated %s embeddings with model %s in %.1f ms",
                len(result),
                self.model_name,
                elapsed_ms,
            )
            return result
        except Exception as exc:
            logger.exception("Embedding generation failed")
            raise EmbeddingServiceError(f"Embedding generation failed: {exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search or QA query."""
        cleaned = query.strip()
        if not cleaned:
            raise EmbeddingServiceError("Query cannot be empty")
        
        # For a single query, we don't need progress tracking
        embeddings = self.embed_texts([cleaned], show_progress=False)
        return embeddings[0]

    def get_dimension(self) -> int:
        """Return the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()


def get_embedding_service() -> EmbeddingService:
    """Create the configured vector embedding service.

    Retrieval must use the same MiniLM/SentenceTransformer family for both stored chunks
    and user queries. Gemini is intentionally not supported here; it belongs in QA
    answer generation after Pinecone retrieval.
    """
    provider = settings.EMBEDDING_PROVIDER
    if provider in {"sentence_transformer", "sentence-transformer", "local"}:
        return SentenceTransformerEmbeddingService(
            model_name=settings.EMBEDDING_MODEL_NAME,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    if provider == "gemini":
        raise EmbeddingServiceError(
            "Gemini embeddings are disabled for retrieval. Use "
            "EMBEDDING_PROVIDER=sentence_transformer so user queries and Pinecone vectors "
            "use the same MiniLM embedding space."
        )
    raise EmbeddingServiceError(f"Unsupported EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'")


def _batches(items: list[str], size: int):
    batch_size = max(1, size)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
