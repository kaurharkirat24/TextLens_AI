"""Classify user questions before retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryIntent:
    intent: str
    confidence: float
    rationale: str
    signals: list[str] = field(default_factory=list)


class QueryIntentClassifier:
    """Lightweight deterministic classifier for routing QA requests."""

    AGGREGATION_PATTERNS = (
        r"\bmost (common|frequent|mentioned|repeated)\b",
        r"\btop\s+\d*\s*(topics|issues|complaints|themes|keywords|countries|sentiments)\b",
        r"\bfrequency\b",
        r"\bdistribution\b",
        r"\bpercentage\b",
        r"\bhow many\b",
        r"\bcount\b",
    )
    TREND_PATTERNS = (
        r"\bover time\b",
        r"\btrend\b",
        r"\bchanged?\b",
        r"\bby month\b",
        r"\bby year\b",
        r"\btimeline\b",
    )
    COMPARISON_PATTERNS = (
        r"\bcompare\b",
        r"\bcontrast\b",
        r"\bcompared to\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference between\b",
    )
    SUMMARY_PATTERNS = (
        r"\bsummarize\b",
        r"\boverview\b",
        r"\bmainly discussing\b",
        r"\bmain themes\b",
        r"\bwhat are people saying\b",
    )
    EXPLORATION_PATTERNS = (
        r"\bcomplain",
        r"\bissues?\b",
        r"\bproblems?\b",
        r"\bwhy\b",
        r"\breasons?\b",
        r"\bfeedback\b",
    )
    FACTUAL_PATTERNS = (
        r"^what is\b",
        r"^who\b",
        r"^when\b",
        r"^where\b",
        r"^how\b",
        r"\bfind comments? (about|mentioning)\b",
    )

    def classify(self, query: str) -> QueryIntent:
        normalized = _normalize(query)
        if not normalized:
            return QueryIntent("unsupported", 0.0, "Empty question.")

        checks = [
            ("trend", self.TREND_PATTERNS, 0.88, "Time-based wording asks for trend analysis."),
            ("comparison", self.COMPARISON_PATTERNS, 0.86, "Comparison wording asks for grouped analysis."),
            ("aggregation", self.AGGREGATION_PATTERNS, 0.9, "Frequency/count wording asks for aggregate analysis."),
            ("summarization", self.SUMMARY_PATTERNS, 0.82, "Summary wording asks for broad dataset context."),
            ("semantic_exploration", self.EXPLORATION_PATTERNS, 0.78, "Exploratory wording asks for themes with evidence."),
            ("factual", self.FACTUAL_PATTERNS, 0.72, "Specific wording asks for direct evidence."),
        ]
        for intent, patterns, confidence, rationale in checks:
            signals = _matching_patterns(normalized, patterns)
            if signals:
                return QueryIntent(intent, confidence, rationale, signals)

        return QueryIntent(
            "semantic_exploration",
            0.55,
            "No strong aggregate or factual signal; defaulting to semantic exploration.",
            [],
        )


def _matching_patterns(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text)]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
