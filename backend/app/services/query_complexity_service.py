"""Estimate query complexity for adaptive top-k selection."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryComplexity:
    level: str
    score: float
    rationale: str


class QueryComplexityService:
    """Simple deterministic complexity scoring for retrieval sizing."""

    def assess(self, query: str, intent: str) -> QueryComplexity:
        normalized = query.strip().lower()
        if not normalized:
            return QueryComplexity(level="simple", score=0.0, rationale="Empty query.")

        tokens = re.findall(r"[a-z0-9_'-]+", normalized)
        token_count = len(tokens)
        signals = 0
        if token_count >= 14:
            signals += 1
        if token_count >= 22:
            signals += 1
        if re.search(r"\b(and|or|compare|versus|vs|trend|over time|by month|by year|distribution)\b", normalized):
            signals += 1
        if re.search(r"[?].*[?]|[,;:].*[,;:]", query):
            signals += 1
        if intent in {"trend", "comparison", "summarization"}:
            signals += 1

        if signals >= 3:
            return QueryComplexity(level="complex", score=0.9, rationale="Multiple complexity signals detected.")
        if signals >= 1:
            return QueryComplexity(level="medium", score=0.6, rationale="Some complexity signals detected.")
        return QueryComplexity(level="simple", score=0.3, rationale="Short/direct question.")
