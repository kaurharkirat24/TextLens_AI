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


def detect_schema(df: pd.DataFrame, sample_size: int = 5000) -> dict[str, ColumnType]:
    """Return ``{column_name: detected_type}`` for every DataFrame column."""
    sample = df.head(sample_size) if len(df) > sample_size else df
    return {column: _classify_column(sample[column]) for column in sample.columns}


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
