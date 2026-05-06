"""Ollama embedding client for Phase 3 semantic retrieval."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import httpx

from app.core.config import settings


logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when embeddings cannot be generated."""


class OllamaEmbeddingService:
    """Generate embeddings through Ollama's /api/embed endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.timeout = timeout

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches while preserving input order."""
        if not texts:
            return []

        start = time.perf_counter()
        embeddings: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for batch in _batches(texts, self.batch_size):
                embeddings.extend(self._embed_batch(client, batch))

        if len(embeddings) != len(texts):
            raise EmbeddingServiceError(
                f"Embedding count mismatch: generated {len(embeddings)} for {len(texts)} texts"
            )

        dimensions = {len(vector) for vector in embeddings}
        if len(dimensions) != 1:
            raise EmbeddingServiceError(f"Inconsistent embedding dimensions: {sorted(dimensions)}")

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Generated %s embeddings with model %s in %.1f ms",
            len(embeddings),
            self.model,
            elapsed_ms,
        )
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search or QA query."""
        cleaned = query.strip()
        if not cleaned:
            raise EmbeddingServiceError("Query cannot be empty")
        embeddings = self.embed_texts([cleaned])
        return embeddings[0]

    def _embed_batch(self, client: httpx.Client, batch: list[str]) -> list[list[float]]:
        try:
            response = client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.exception("Ollama embedding request failed")
            raise EmbeddingServiceError(f"Ollama embedding request failed: {exc}") from exc

        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingServiceError("Ollama response did not contain embeddings")
        return embeddings


def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + size]
