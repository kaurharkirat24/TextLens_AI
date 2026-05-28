"""In-memory semantic cache for search and QA responses."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from app.core.config import settings


@dataclass
class CacheEntry:
    dataset_id: str
    route: str
    top_k: int
    embedding: list[float]
    payload: Any
    created_at: float


class SemanticQueryCache:
    """Embedding-similarity cache with TTL and bounded size."""

    def __init__(self) -> None:
        self._entries: deque[CacheEntry] = deque()

    def get(self, dataset_id: str, route: str, top_k: int, embedding: list[float]) -> Any | None:
        if not settings.SEMANTIC_CACHE_ENABLED:
            return None
        self._prune_expired()
        best_payload = None
        best_similarity = -1.0
        for entry in self._entries:
            if entry.dataset_id != dataset_id or entry.route != route:
                continue
            if route == "search" and entry.top_k != top_k:
                continue
            similarity = _cosine_similarity(entry.embedding, embedding)
            if similarity >= settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD and similarity > best_similarity:
                best_similarity = similarity
                best_payload = entry.payload
        return best_payload

    def put(self, dataset_id: str, route: str, top_k: int, embedding: list[float], payload: Any) -> None:
        if not settings.SEMANTIC_CACHE_ENABLED:
            return
        self._prune_expired()
        self._entries.appendleft(
            CacheEntry(
                dataset_id=dataset_id,
                route=route,
                top_k=top_k,
                embedding=embedding,
                payload=payload,
                created_at=time.time(),
            )
        )
        max_entries = max(16, settings.SEMANTIC_CACHE_MAX_ENTRIES)
        while len(self._entries) > max_entries:
            self._entries.pop()

    def _prune_expired(self) -> None:
        ttl = max(1, settings.SEMANTIC_CACHE_TTL_SECONDS)
        now = time.time()
        kept = deque(entry for entry in self._entries if (now - entry.created_at) <= ttl)
        self._entries = kept


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return -1.0
    return dot / denom


_CACHE = SemanticQueryCache()


def get_semantic_cache() -> SemanticQueryCache:
    return _CACHE
