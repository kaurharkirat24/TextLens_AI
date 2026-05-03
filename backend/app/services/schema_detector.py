"""
Schema detection for Phase 2 analytics.

The detector infers one of four analysis types for every DataFrame column:
text, numeric, categorical, or datetime. It uses data-shape heuristics only;
column names are intentionally ignored so uploaded datasets do not need a
fixed schema.
"""

from __future__ import annotations

import re
import warnings
from typing import Literal

import pandas as pd

ColumnType = Literal["text", "numeric", "categorical", "datetime"]

NUMERIC_PARSE_THRESHOLD = 0.90
DATETIME_PARSE_THRESHOLD = 0.80
CATEGORICAL_MAX_UNIQUE_RATIO = 0.30
CATEGORICAL_MAX_UNIQUE_ABS = 50
TEXT_MIN_AVG_CHARS = 24
TEXT_MIN_AVG_WORDS = 3

_DATE_SHAPE_RE = re.compile(
    r"(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4})|"
    r"(?:\d{1,2}:\d{2})|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*\b")

PRIMARY_TEXT_NAME_HINTS = (
    "comment",
    "commenttext",
    "review",
    "feedback",
    "message",
    "body",
    "description",
    "text",
    "content",
)
SECONDARY_TEXT_NAME_HINTS = (
    "title",
    "name",
    "author",
    "channel",
    "country",
    "category",
    "tag",
    "url",
    "link",
    "id",
)
CONTENT_TITLE_HINTS = ("videotitle", "video_title", "title", "content_title", "post_title")
CONTENT_ID_HINTS = ("videoid", "video_id", "content_id", "post_id", "item_id")
DATETIME_HINTS = ("published", "created", "date", "time", "timestamp", "posted")
ENGAGEMENT_HINTS = {
    "likes": ("likes", "like_count", "likecount", "upvotes", "upvote_count"),
    "replies": ("replies", "reply_count", "replycount", "responses", "response_count"),
    "views": ("views", "view_count", "viewcount", "impressions"),
    "comments": ("comments", "comment_count", "commentcount"),
    "shares": ("shares", "share_count", "sharecount"),
}


def detect_schema(df: pd.DataFrame, sample_size: int = 5000) -> dict[str, ColumnType]:
    """Return ``{column_name: detected_type}`` for every DataFrame column."""
    sample = df.head(sample_size) if len(df) > sample_size else df
    return {column: _classify_column(sample[column]) for column in sample.columns}


def detect_column_roles(df: pd.DataFrame, schema: dict[str, ColumnType], sample_size: int = 5000) -> dict:
    """Detect semantic roles used by insight-driven analytics.

    The schema remains shape-based for compatibility. Roles add product-level
    meaning, most importantly selecting exactly one primary free-text column for
    sentiment and keyword enrichment.
    """
    sample = df.head(sample_size) if len(df) > sample_size else df
    text_cols = [col for col, kind in schema.items() if kind == "text" and col in sample.columns]
    numeric_cols = [col for col, kind in schema.items() if kind == "numeric" and col in sample.columns]
    datetime_cols = [col for col, kind in schema.items() if kind == "datetime" and col in sample.columns]

    primary_text = _choose_primary_text(sample, text_cols)
    secondary_text = [col for col in text_cols if col != primary_text]

    roles = {
        "primary_text": primary_text,
        "secondary_text": secondary_text,
        "content": {
            "id": _first_matching_column(sample.columns, CONTENT_ID_HINTS, exclude={primary_text}),
            "title": _first_matching_column(sample.columns, CONTENT_TITLE_HINTS, exclude={primary_text}),
        },
        "engagement": _detect_engagement_columns(numeric_cols),
        "time": {
            "primary_datetime": _choose_datetime_column(datetime_cols),
        },
    }
    return roles


def get_schema_summary(schema: dict[str, ColumnType]) -> dict:
    """Group detected columns by type for API consumers."""
    summary: dict[str, list[str]] = {
        "text": [],
        "numeric": [],
        "categorical": [],
        "datetime": [],
    }
    for column, column_type in schema.items():
        summary[column_type].append(column)
    return {
        "columns": schema,
        "summary": summary,
        "total_columns": len(schema),
    }


def _choose_primary_text(df: pd.DataFrame, text_cols: list[str]) -> str | None:
    if not text_cols:
        return None
    scored = [(_primary_text_score(df[col], col), col) for col in text_cols]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][1] if scored else None


def _primary_text_score(series: pd.Series, column: str) -> float:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return 0.0

    normalized = _normalize_name(column)
    avg_chars = float(values.str.len().mean())
    word_counts = values.str.count(_WORD_RE.pattern)
    avg_words = float(word_counts.mean())
    unique_ratio = float(values.nunique(dropna=True) / len(values))
    non_empty_ratio = float(len(values) / max(len(series), 1))

    score = min(avg_chars, 240) / 12
    score += min(avg_words, 40) * 1.8
    score += unique_ratio * 8
    score += non_empty_ratio * 5

    if any(hint in normalized for hint in PRIMARY_TEXT_NAME_HINTS):
        score += 30
    if any(hint in normalized for hint in SECONDARY_TEXT_NAME_HINTS):
        score -= 24
    return score


def _first_matching_column(
    columns: list[str] | pd.Index,
    hints: tuple[str, ...],
    exclude: set[str | None] | None = None,
) -> str | None:
    excluded = {value for value in (exclude or set()) if value}
    for column in columns:
        if column in excluded:
            continue
        normalized = _normalize_name(column)
        if any(_normalize_name(hint) in normalized or hint == str(column).lower() for hint in hints):
            return column
    return None


def _detect_engagement_columns(numeric_cols: list[str]) -> dict[str, str]:
    engagement: dict[str, str] = {}
    for metric, hints in ENGAGEMENT_HINTS.items():
        match = _first_matching_column(numeric_cols, hints)
        if match:
            engagement[metric] = match
    return engagement


def _choose_datetime_column(datetime_cols: list[str]) -> str | None:
    if not datetime_cols:
        return None
    for column in datetime_cols:
        normalized = _normalize_name(column)
        if any(hint in normalized for hint in DATETIME_HINTS):
            return column
    return datetime_cols[0]


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _classify_column(series: pd.Series) -> ColumnType:
    values = series.dropna()
    values = values[values.astype(str).str.strip() != ""]

    if values.empty:
        return "categorical"

    if pd.api.types.is_datetime64_any_dtype(values):
        return "datetime"

    if pd.api.types.is_numeric_dtype(values):
        return "numeric"

    as_text = values.astype(str).str.strip()

    if _looks_datetime(as_text):
        return "datetime"

    if _looks_numeric(as_text):
        return "numeric"

    unique_count = int(as_text.nunique(dropna=True))
    total_count = int(len(as_text))
    unique_ratio = unique_count / total_count if total_count else 1.0
    avg_chars = float(as_text.str.len().mean())
    word_counts = as_text.str.count(_WORD_RE.pattern)
    avg_words = float(word_counts.mean())
    single_token_ratio = float((~as_text.str.contains(r"\s", regex=True)).mean())
    multi_word_ratio = float((word_counts >= 2).mean())
    has_sentence_marks = float(as_text.str.contains(r"[.!?;,]", regex=True).mean())

    if single_token_ratio >= 0.90 and has_sentence_marks == 0:
        return "categorical"

    if (
        avg_words >= TEXT_MIN_AVG_WORDS
        or avg_chars >= TEXT_MIN_AVG_CHARS
        or (multi_word_ratio >= 0.50 and has_sentence_marks >= 0.10)
    ):
        return "text"

    if unique_ratio <= CATEGORICAL_MAX_UNIQUE_RATIO or unique_count <= CATEGORICAL_MAX_UNIQUE_ABS:
        return "categorical"

    # High-cardinality short strings are usually identifiers/codes. Treating
    # them as categorical prevents bogus sentiment and keyword analysis.
    return "categorical"


def _looks_numeric(values: pd.Series) -> bool:
    parsed = pd.to_numeric(values.str.replace(",", "", regex=False), errors="coerce")
    return float(parsed.notna().mean()) >= NUMERIC_PARSE_THRESHOLD


def _looks_datetime(values: pd.Series) -> bool:
    likely_dates = values[values.str.contains(_DATE_SHAPE_RE, regex=True, na=False)]
    if likely_dates.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(likely_dates, errors="coerce")
    return float(parsed.notna().mean()) >= DATETIME_PARSE_THRESHOLD
