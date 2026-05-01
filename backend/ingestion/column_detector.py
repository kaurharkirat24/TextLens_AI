"""
Local text column detection.

The upload flow used to call Gemini when a simple keyword heuristic was
uncertain. That made Phase 1 dependent on API quota for something we can do
deterministically from the CSV itself. This detector ranks columns with a
small local pipeline:

1. column-name signals
2. sampled content shape
3. penalties for IDs, dates, numeric/rating fields, URLs, and sparse columns
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from ingestion.config import IngestionConfig, TEXT_COLUMN_KEYWORDS
from ingestion.models import ColumnDetectionResult


NEGATIVE_COLUMN_KEYWORDS = {
    "id",
    "uuid",
    "index",
    "row",
    "date",
    "time",
    "timestamp",
    "created",
    "updated",
    "rating",
    "score",
    "stars",
    "price",
    "amount",
    "quantity",
    "qty",
    "count",
    "email",
    "phone",
    "url",
    "link",
    "source",
    "author",
    "user",
    "username",
    "name",
}

DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$")
URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z]{2,}")


@dataclass
class ColumnScore:
    column_name: str
    score: float
    reasons: list[str]


def _clean_values(values: Iterable) -> list[str]:
    cleaned = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null", "na", "n/a"}:
            cleaned.append(text)
    return cleaned


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _name_score(column: str) -> tuple[float, list[str]]:
    normalized = column.lower().strip()
    tokens = set(re.split(r"[^a-z0-9]+", normalized))
    score = 0.0
    reasons: list[str] = []

    if normalized in TEXT_COLUMN_KEYWORDS:
        score += 45
        reasons.append("exact text-column name match")
    elif any(keyword in normalized for keyword in TEXT_COLUMN_KEYWORDS):
        score += 30
        reasons.append("text-like column name")

    negative_hits = tokens & NEGATIVE_COLUMN_KEYWORDS
    if negative_hits:
        score -= 25
        reasons.append(f"metadata-like name ({', '.join(sorted(negative_hits))})")

    return score, reasons


def _content_score(column: str, values: list[str], total_rows: int) -> ColumnScore:
    name_score, reasons = _name_score(column)
    if not values:
        return ColumnScore(column, name_score - 40, reasons + ["no non-empty sampled values"])

    lengths = [len(value) for value in values]
    word_counts = [len(WORD_RE.findall(value)) for value in values]
    alpha_counts = [sum(ch.isalpha() for ch in value) for value in values]
    digit_counts = [sum(ch.isdigit() for ch in value) for value in values]

    avg_len = sum(lengths) / len(lengths)
    median_len = sorted(lengths)[len(lengths) // 2]
    avg_words = sum(word_counts) / len(word_counts)
    non_empty_ratio = _safe_ratio(len(values), total_rows)
    unique_ratio = _safe_ratio(len({value.lower() for value in values}), len(values))
    alpha_ratio = _safe_ratio(sum(alpha_counts), sum(lengths))
    digit_ratio = _safe_ratio(sum(digit_counts), sum(lengths))
    sentenceish_ratio = _safe_ratio(
        sum(1 for value in values if len(WORD_RE.findall(value)) >= 3 or len(value) >= 30),
        len(values),
    )
    numeric_ratio = _safe_ratio(sum(1 for value in values if _is_numeric_like(value)), len(values))
    date_ratio = _safe_ratio(sum(1 for value in values if DATE_RE.match(value)), len(values))
    url_ratio = _safe_ratio(sum(1 for value in values if URL_RE.match(value)), len(values))

    score = name_score

    score += min(avg_len, 120) * 0.35
    score += min(median_len, 100) * 0.15
    score += min(avg_words, 20) * 2.5
    score += non_empty_ratio * 15
    score += unique_ratio * 10
    score += alpha_ratio * 20
    score += sentenceish_ratio * 20

    score -= numeric_ratio * 65
    score -= date_ratio * 55
    score -= url_ratio * 35
    score -= digit_ratio * 20

    if avg_len >= 25:
        reasons.append(f"average sampled length is {avg_len:.0f} chars")
    if avg_words >= 4:
        reasons.append(f"average sampled text has {avg_words:.1f} words")
    if sentenceish_ratio >= 0.5:
        reasons.append("values look like natural-language comments")
    if numeric_ratio >= 0.5:
        reasons.append("mostly numeric values")
    if date_ratio >= 0.5:
        reasons.append("mostly date-like values")
    if url_ratio >= 0.5:
        reasons.append("mostly URL-like values")

    if math.isnan(score):
        score = -100.0

    return ColumnScore(column, score, reasons)


def _is_numeric_like(value: str) -> bool:
    compact = value.replace(",", "").replace("%", "").strip()
    if not compact:
        return False
    try:
        float(compact)
        return True
    except ValueError:
        return False


def _confidence(best: ColumnScore, runner_up: ColumnScore | None) -> str:
    margin = best.score - runner_up.score if runner_up else best.score
    if best.score >= 75 and margin >= 12:
        return "high"
    if best.score >= 45 and margin >= 5:
        return "medium"
    return "low"


def _heuristic_detect(columns: list[str], sample_values: dict[str, list]) -> ColumnDetectionResult | None:
    """
    Rank columns with a local scoring pipeline.

    Returns None when no column has enough textual evidence. This preserves the
    old unit-testable helper contract while removing all external API fallback.
    """
    total_rows = max((len(values) for values in sample_values.values()), default=0)
    scores = [
        _content_score(column, _clean_values(sample_values.get(column, [])), total_rows)
        for column in columns
    ]
    scores.sort(key=lambda item: item.score, reverse=True)

    if not scores or scores[0].score < 35:
        return None

    best = scores[0]
    runner_up = scores[1] if len(scores) > 1 else None
    confidence = _confidence(best, runner_up)
    top_candidates = [score.column_name for score in scores[:3] if score.score >= 25]
    reason = "; ".join(best.reasons[:3]) or "Selected from sampled text-column scoring."

    return ColumnDetectionResult(
        column_name=best.column_name,
        method="local_pipeline",
        confidence=confidence,
        candidates=top_candidates or [best.column_name],
        reasoning=f"Local detector selected '{best.column_name}': {reason}.",
    )


def detect_text_column(df, config: IngestionConfig) -> ColumnDetectionResult:
    """
    Detect the primary unstructured text column without using an external API.
    Priority: user_specified > local scoring pipeline.
    """
    columns = df.columns.tolist()

    if config.text_column:
        if config.text_column not in columns:
            raise ValueError(
                f"Specified text column '{config.text_column}' not found. "
                f"Available columns: {columns}"
            )
        return ColumnDetectionResult(
            column_name=config.text_column,
            method="user_specified",
            confidence="high",
            candidates=[config.text_column],
            reasoning="Column specified explicitly by the user.",
        )

    sample_values = {col: df[col].dropna().head(250).tolist() for col in columns}
    result = _heuristic_detect(columns, sample_values)
    if result:
        return result

    raise ValueError(
        "Cannot detect a text column from this CSV. Provide text_column explicitly "
        f"or upload a file with a natural-language column. Available columns: {columns}"
    )
