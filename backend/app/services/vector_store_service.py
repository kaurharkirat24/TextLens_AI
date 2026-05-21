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

    _client_cache = None
    _index_cache: dict[str, Any] = {}

    def __init__(self, index_name: str | None = None) -> None:
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.api_key = settings.PINECONE_API_KEY
        self.region = settings.PINECONE_REGION
        self.cloud = settings.PINECONE_CLOUD
        self._pc = None
        self._index = None

    @classmethod
    def for_dimension(cls, dimension: int, base_name: str | None = None) -> "PineconeVectorStore":
        """Return a store bound to a dimension-specific index name."""
        return cls(index_name=_dimension_index_name(base_name or settings.PINECONE_INDEX_NAME, dimension))

    @property
    def client(self):
        if not self.api_key:
            raise VectorStoreError("Missing Pinecone API key")
        if self._pc is None:
            if PineconeVectorStore._client_cache is not None:
                self._pc = PineconeVectorStore._client_cache
                return self._pc

            try:
                from pinecone import Pinecone
            except ImportError as exc:
                raise VectorStoreError("Pinecone SDK is not installed") from exc
            self._pc = Pinecone(api_key=self.api_key)
            PineconeVectorStore._client_cache = self._pc
        return self._pc

    @property
    def index(self):
        if self._index is None:
            if self.index_name not in PineconeVectorStore._index_cache:
                PineconeVectorStore._index_cache[self.index_name] = self.client.Index(self.index_name)
            self._index = PineconeVectorStore._index_cache[self.index_name]
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
                self._create_index(dimension)
            else:
                self._validate_or_select_dimension_index(dimension, existing_names)
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
        max_retries: int = 3,
    ) -> int:
        """Upsert vectors into the dataset namespace in batches with retry logic."""
        if not vectors:
            return 0
        count = 0
        size = batch_size or settings.PINECONE_UPSERT_BATCH_SIZE
        
        for batch in _batches(vectors, size):
            payload = [
                {"id": vector_id, "values": values, "metadata": metadata}
                for vector_id, values, metadata in batch
            ]
            
            attempt = 0
            while attempt < max_retries:
                try:
                    self.index.upsert(vectors=payload, namespace=namespace)
                    count += len(payload)
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt == max_retries:
                        logger.exception("Pinecone upsert failed after %s attempts", max_retries)
                        raise VectorStoreError(f"Pinecone upsert failed after {max_retries} attempts: {exc}") from exc
                    
                    wait_time = 2 ** attempt
                    logger.warning("Pinecone upsert failed (attempt %s/%s). Retrying in %s seconds...", 
                                   attempt, max_retries, wait_time)
                    time.sleep(wait_time)
            
        return count

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

    def _create_index(self, dimension: int) -> None:
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

    def _validate_or_select_dimension_index(self, expected_dimension: int, existing_names: set[str]) -> None:
        description = self.client.describe_index(self.index_name)
        actual_dimension = _response_value(description, "dimension", None)
        if not actual_dimension or int(actual_dimension) == expected_dimension:
            return

        base_name = self.index_name
        dimension_index_name = _dimension_index_name(base_name, expected_dimension)
        logger.warning(
            "Pinecone index %s has dimension %s, using dimension-specific index %s for %s-dimensional vectors",
            base_name,
            actual_dimension,
            dimension_index_name,
            expected_dimension,
        )

        self.index_name = dimension_index_name
        self._index = None
        PineconeVectorStore._index_cache.pop(base_name, None)
        if dimension_index_name not in existing_names:
            self._create_index(expected_dimension)
        else:
            description = self.client.describe_index(dimension_index_name)
            actual_dimension = _response_value(description, "dimension", None)
            if actual_dimension and int(actual_dimension) != expected_dimension:
                raise VectorStoreError(
                    f"Dimension-specific Pinecone index {dimension_index_name} has dimension "
                    f"{actual_dimension}, expected {expected_dimension}"
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


def _dimension_index_name(base_name: str, dimension: int) -> str:
    suffix = f"-{dimension}"
    if base_name.endswith(suffix):
        return base_name
    return f"{base_name[: 45 - len(suffix)]}{suffix}".strip("-")
