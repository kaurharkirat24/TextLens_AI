"""Pinecone vector store integration for dataset-aware semantic search."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

from app.core.config import settings


logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when Pinecone operations fail."""


class PineconeVectorStore:
    """Single-index Pinecone client using dataset_id as the namespace."""

    def __init__(self) -> None:
        self.index_name = settings.PINECONE_INDEX_NAME
        self.api_key = settings.PINECONE_API_KEY
        self.region = settings.PINECONE_REGION
        self.cloud = settings.PINECONE_CLOUD
        self._pc = None
        self._index = None

    @property
    def client(self):
        if not self.api_key:
            raise VectorStoreError("Missing Pinecone API key")
        if self._pc is None:
            try:
                from pinecone import Pinecone
            except ImportError as exc:
                raise VectorStoreError("Pinecone SDK is not installed") from exc
            self._pc = Pinecone(api_key=self.api_key)
        return self._pc

    @property
    def index(self):
        if self._index is None:
            self._index = self.client.Index(self.index_name)
        return self._index

    def ensure_index(self, dimension: int) -> None:
        """Create the configured single index if missing, validating dimension if present."""
        if not self.index_name:
            raise VectorStoreError("Missing Pinecone index name")
        if not self.region:
            raise VectorStoreError("Missing Pinecone region")

        try:
            existing_names = set(self.client.list_indexes().names())
            if self.index_name not in existing_names:
                from pinecone import ServerlessSpec

                logger.info("Creating Pinecone index %s with dimension %s", self.index_name, dimension)
                self.client.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                    deletion_protection="disabled",
                )
                self._wait_until_ready()
            else:
                self._validate_existing_dimension(dimension)
        except Exception as exc:
            logger.exception("Pinecone index setup failed")
            raise VectorStoreError(f"Pinecone index setup failed: {exc}") from exc

    def has_index(self) -> bool:
        """Return whether the configured Pinecone index already exists."""
        try:
            return self.index_name in set(self.client.list_indexes().names())
        except Exception as exc:
            logger.exception("Pinecone index lookup failed")
            raise VectorStoreError(f"Pinecone index lookup failed: {exc}") from exc

    def describe_dimension(self) -> int | None:
        """Return the configured index dimension when the index exists."""
        try:
            if not self.has_index():
                return None
            description = self.client.describe_index(self.index_name)
            dimension = _response_value(description, "dimension", None)
            return int(dimension) if dimension else None
        except Exception as exc:
            logger.exception("Pinecone index describe failed")
            raise VectorStoreError(f"Pinecone index describe failed: {exc}") from exc

    def existing_ids(self, ids: list[str], namespace: str) -> set[str]:
        """Fetch IDs already present in the dataset namespace."""
        existing: set[str] = set()
        try:
            for batch in _batches(ids, 100):
                response = self.index.fetch(ids=batch, namespace=namespace)
                vectors = _response_value(response, "vectors", {})
                existing.update(vectors.keys())
            return existing
        except Exception as exc:
            logger.exception("Pinecone fetch failed")
            raise VectorStoreError(f"Pinecone fetch failed: {exc}") from exc

    def upsert_vectors(
        self,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
        namespace: str,
        batch_size: int | None = None,
    ) -> int:
        """Upsert vectors into the dataset namespace in batches."""
        if not vectors:
            return 0
        count = 0
        size = batch_size or settings.VECTOR_UPSERT_BATCH_SIZE
        try:
            for batch in _batches(vectors, size):
                payload = [
                    {"id": vector_id, "values": values, "metadata": metadata}
                    for vector_id, values, metadata in batch
                ]
                self.index.upsert(vectors=payload, namespace=namespace)
                count += len(payload)
            return count
        except Exception as exc:
            logger.exception("Pinecone upsert failed")
            raise VectorStoreError(f"Pinecone upsert failed: {exc}") from exc

    def query(self, vector: list[float], namespace: str, top_k: int) -> list[dict[str, Any]]:
        """Query a single dataset namespace and return normalized matches."""
        try:
            start = time.perf_counter()
            response = self.index.query(
                vector=vector,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Pinecone query completed for namespace %s top_k=%s in %.1f ms",
                namespace,
                top_k,
                elapsed_ms,
            )
        except Exception as exc:
            logger.exception("Pinecone query failed")
            raise VectorStoreError(f"Pinecone query failed: {exc}") from exc

        matches = _response_value(response, "matches", []) or []
        normalized: list[dict[str, Any]] = []
        for match in matches:
            metadata = _response_value(match, "metadata", {}) or {}
            normalized.append(
                {
                    "id": str(_response_value(match, "id", "")),
                    "score": float(_response_value(match, "score", 0.0) or 0.0),
                    "metadata": metadata,
                    "text": str(metadata.get("text", "")),
                }
            )
        return normalized

    def _validate_existing_dimension(self, expected_dimension: int) -> None:
        description = self.client.describe_index(self.index_name)
        actual_dimension = _response_value(description, "dimension", None)
        if actual_dimension and int(actual_dimension) != expected_dimension:
            raise VectorStoreError(
                f"Existing Pinecone index dimension {actual_dimension} does not match "
                f"embedding dimension {expected_dimension}"
            )

    def _wait_until_ready(self, timeout_seconds: int = 120) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            description = self.client.describe_index(self.index_name)
            status = _response_value(description, "status", {}) or {}
            ready = _response_value(status, "ready", False)
            if ready:
                self._index = None
                return
            time.sleep(2)
        raise VectorStoreError(f"Pinecone index '{self.index_name}' was not ready in time")


def _batches(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + size]


def _response_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
