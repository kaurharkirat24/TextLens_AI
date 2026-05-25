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
    _grpc_available: bool | None = None
    _index_cache: dict[str, Any] = {}
    _index_exists_cache: set[str] = set()
    _index_dimension_cache: dict[str, int] = {}

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
                PineconeVectorStore._index_cache[self.index_name] = self._create_index_client()
            self._index = PineconeVectorStore._index_cache[self.index_name]
        return self._index

    @property
    def transport(self) -> str:
        return "grpc" if PineconeVectorStore._grpc_available else "rest"

    def _create_index_client(self):
        """Create a reusable data-plane index client, preferring Pinecone gRPC."""
        if PineconeVectorStore._grpc_available is not False:
            try:
                index_factory = getattr(self.client, "index", None)
                if callable(index_factory):
                    index = index_factory(name=self.index_name, grpc=True)
                else:
                    from pinecone.grpc import PineconeGRPC

                    grpc_client = PineconeGRPC(api_key=self.api_key)
                    index = grpc_client.Index(self.index_name)
                PineconeVectorStore._grpc_available = True
                logger.info("Initialized Pinecone gRPC index client for %s", self.index_name)
                return index
            except Exception as exc:
                PineconeVectorStore._grpc_available = False
                logger.warning(
                    "Pinecone gRPC index client unavailable for %s; falling back to REST client: %s",
                    self.index_name,
                    exc,
                )

        logger.info("Initialized Pinecone REST index client for %s", self.index_name)
        return self.client.Index(self.index_name)

    def ensure_index(self, dimension: int) -> None:
        """Create the configured single index if missing, validating dimension if present."""
        if not self.index_name:
            raise VectorStoreError("Missing Pinecone index name")
        if not self.region:
            raise VectorStoreError("Missing Pinecone region")

        try:
            cached_dimension = PineconeVectorStore._index_dimension_cache.get(self.index_name)
            if cached_dimension and cached_dimension == dimension:
                return

            if self.index_name in PineconeVectorStore._index_exists_cache:
                existing_names = set(PineconeVectorStore._index_exists_cache)
            else:
                existing_names = set(self.client.list_indexes().names())
                PineconeVectorStore._index_exists_cache.update(existing_names)

            if self.index_name not in existing_names:
                self._create_index(dimension)
            else:
                self._validate_or_select_dimension_index(dimension, existing_names)
        except Exception as exc:
            logger.exception("Pinecone index setup failed")
            raise VectorStoreError(f"Pinecone index setup failed: {exc}") from exc

    def has_index(self) -> bool:
        """Return whether the configured Pinecone index already exists."""
        if self.index_name in PineconeVectorStore._index_exists_cache:
            return True
        try:
            existing_names = set(self.client.list_indexes().names())
            PineconeVectorStore._index_exists_cache.update(existing_names)
            return self.index_name in existing_names
        except Exception as exc:
            logger.exception("Pinecone index lookup failed")
            raise VectorStoreError(f"Pinecone index lookup failed: {exc}") from exc

    def describe_dimension(self) -> int | None:
        """Return the configured index dimension when the index exists."""
        try:
            if self.index_name in PineconeVectorStore._index_dimension_cache:
                return PineconeVectorStore._index_dimension_cache[self.index_name]
            if not self.has_index():
                return None
            description = self.client.describe_index(self.index_name)
            dimension = _response_value(description, "dimension", None)
            if not dimension:
                return None
            dimension = int(dimension)
            PineconeVectorStore._index_dimension_cache[self.index_name] = dimension
            return dimension
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
        max_retries: int | None = None,
    ) -> int:
        """Upsert vectors into the dataset namespace in batches with retry logic."""
        if not vectors:
            return 0
        count = 0
        size = batch_size or settings.PINECONE_UPSERT_BATCH_SIZE
        retries = max_retries or settings.PINECONE_UPSERT_MAX_RETRIES
        total_vectors = len(vectors)
        start = time.perf_counter()
        batch_count = 0
        
        for batch in _batches(vectors, size):
            batch_count += 1
            payload_start = time.perf_counter()
            payload = [
                {"id": vector_id, "values": values, "metadata": metadata}
                for vector_id, values, metadata in batch
            ]
            payload_ms = (time.perf_counter() - payload_start) * 1000
            upsert_start = time.perf_counter()
            upserted = self._upsert_payload_with_retry(payload, namespace, retries)
            upsert_ms = (time.perf_counter() - upsert_start) * 1000
            count += upserted
            logger.info(
                "Pinecone %s upsert batch completed: vectors=%s namespace=%s payload_ms=%.1f upsert_ms=%.1f vectors_per_sec=%.1f",
                self.transport,
                upserted,
                namespace,
                payload_ms,
                upsert_ms,
                upserted / max(upsert_ms / 1000, 0.001),
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Pinecone upsert completed: vectors=%s batches=%s batch_size=%s namespace=%s elapsed_ms=%.1f vectors_per_sec=%.1f transport=%s",
            total_vectors,
            batch_count,
            size,
            namespace,
            elapsed_ms,
            total_vectors / max(elapsed_ms / 1000, 0.001),
            self.transport,
        )
        return count

    def _upsert_payload_with_retry(
        self,
        payload: list[dict[str, Any]],
        namespace: str,
        max_retries: int,
    ) -> int:
        """Upsert one payload, splitting it if writes repeatedly time out."""
        attempt = 0
        while attempt < max_retries:
            try:
                self.index.upsert(
                    vectors=payload,
                    namespace=namespace,
                    timeout=settings.PINECONE_UPSERT_TIMEOUT_SECONDS,
                )
                return len(payload)
            except Exception as exc:
                attempt += 1
                if attempt == max_retries:
                    if len(payload) > settings.PINECONE_MIN_UPSERT_BATCH_SIZE:
                        midpoint = max(1, len(payload) // 2)
                        logger.warning(
                            "Pinecone upsert failed after %s attempts for %s vectors; splitting into %s and %s vectors.",
                            max_retries,
                            len(payload),
                            midpoint,
                            len(payload) - midpoint,
                        )
                        return self._upsert_payload_with_retry(
                            payload[:midpoint],
                            namespace,
                            max_retries,
                        ) + self._upsert_payload_with_retry(
                            payload[midpoint:],
                            namespace,
                            max_retries,
                        )

                    logger.exception("Pinecone upsert failed after %s attempts", max_retries)
                    raise VectorStoreError(f"Pinecone upsert failed after {max_retries} attempts: {exc}") from exc

                wait_time = min(30, 2 ** attempt)
                logger.warning(
                    "Pinecone upsert failed for %s vectors (attempt %s/%s). Retrying in %s seconds...",
                    len(payload),
                    attempt,
                    max_retries,
                    wait_time,
                )
                time.sleep(wait_time)

        return 0

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
        PineconeVectorStore._index_exists_cache.add(self.index_name)
        PineconeVectorStore._index_dimension_cache[self.index_name] = int(dimension)
        self._wait_until_ready()

    def _validate_or_select_dimension_index(self, expected_dimension: int, existing_names: set[str]) -> None:
        cached_dimension = PineconeVectorStore._index_dimension_cache.get(self.index_name)
        if cached_dimension:
            if cached_dimension == expected_dimension:
                return
            actual_dimension = cached_dimension
        else:
            description = self.client.describe_index(self.index_name)
            actual_dimension = _response_value(description, "dimension", None)
            if actual_dimension:
                PineconeVectorStore._index_dimension_cache[self.index_name] = int(actual_dimension)
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
            cached_dimension = PineconeVectorStore._index_dimension_cache.get(dimension_index_name)
            if cached_dimension:
                actual_dimension = cached_dimension
            else:
                description = self.client.describe_index(dimension_index_name)
                actual_dimension = _response_value(description, "dimension", None)
                if actual_dimension:
                    PineconeVectorStore._index_dimension_cache[dimension_index_name] = int(actual_dimension)
            if actual_dimension and int(actual_dimension) != expected_dimension:
                raise VectorStoreError(
                    f"Dimension-specific Pinecone index {dimension_index_name} has dimension "
                    f"{actual_dimension}, expected {expected_dimension}"
                )
        PineconeVectorStore._index_exists_cache.add(dimension_index_name)

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
