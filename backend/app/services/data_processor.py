"""
Data processing for Phase 2 analytics.

This module owns cleaning and enrichment. It avoids dataset-specific
assumptions and uses pandas string/numeric operations for the expensive parts.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from app.services.schema_detector import ColumnType

MAX_KEYWORD_TEXT_CHARS = 2_000_000

CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
    "you're": "you are",
    "you've": "you have",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "luv": "love",
    "cuz": "because",
    "cause": "because",
    "u": "you",
    "r": "are",
    "thx": "thanks",
}

EMOJI_MAP = {
    "😀": "happy",
    "😃": "happy",
    "😄": "happy",
    "😁": "grinning",
    "😆": "laughing",
    "😂": "laughing",
    "🤣": "laughing",
    "😊": "smiling",
    "🙂": "smiling",
    "😉": "winking",
    "😍": "love",
    "🥰": "love",
    "😎": "cool",
    "😐": "neutral",
    "😑": "neutral",
    "😒": "negative",
    "🙄": "negative",
    "😤": "frustrated",
    "😠": "angry",
    "😡": "angry",
    "😢": "sad",
    "😭": "crying",
    "😰": "anxious",
    "😱": "shocked",
    "🤔": "thinking",
    "👍": "thumbs up",
    "👎": "thumbs down",
    "❤️": "love",
    "❤": "love",
    "💔": "heartbreak",
    "🔥": "fire",
    "💯": "perfect",
    "⭐": "star",
    "🌟": "star",
    "✨": "sparkle",
}

POSITIVE_WORDS = set(
    """
    good great excellent amazing awesome wonderful fantastic superb outstanding
    brilliant perfect love lovely beautiful best happy pleased satisfied
    impressive remarkable exceptional delightful nice fine terrific fabulous
    recommend recommended enjoy enjoyed helpful quality premium comfortable
    reliable efficient elegant smooth fast friendly polished clean solid
    flawless superior stunning incredible phenomenal spectacular thanks thank
    """
    .split()
)

NEGATIVE_WORDS = set(
    """
    bad terrible awful horrible worst poor disappointing disappointed frustrated
    frustrating annoying annoyed angry hate hated broken useless waste garbage
    trash defective faulty slow sluggish ugly cheap weak unreliable painful
    difficult confusing misleading scam overpriced expensive regret complaint
    fail failed failure missing damaged crashed error problem issue bug
    unacceptable pathetic dreadful mediocre subpar inferior nightmare disaster
    """
    .split()
)

STOP_WORDS = set(
    """
    the a an is are was were be been being have has had do does did will would
    could should may might can to of in for on with at by from as into through
    during before after above below between out off over under again then once
    here there when where why how all each every both few more most other some
    such no nor not only own same so than too very just because but and or if
    while about up it its this that these those i me my we our you your he him
    his she her they them their what which who whom am also get got one two
    really even much still well back going thing
    """
    .split()
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*\b")
_CONTRACTION_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b", re.I)
_EMOJI_RE = re.compile("|".join(re.escape(k) for k in EMOJI_MAP))


def process_dataframe(
    df: pd.DataFrame,
    schema: dict[str, ColumnType],
    column_roles: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Clean and enrich a DataFrame according to the detected schema."""
    cleaned, cleaning_report = clean_dataframe(df, schema)
    enriched, enrichment_report = enrich_dataframe(cleaned, schema, column_roles)
    return enriched, {
        "cleaning": cleaning_report,
        "enrichment": enrichment_report,
    }


def clean_dataframe(df: pd.DataFrame, schema: dict[str, ColumnType]) -> tuple[pd.DataFrame, dict]:
    """Normalize rows/cells, remove empty rows, clean text, and deduplicate."""
    clean = df.copy()
    original_rows = len(clean)
    original_null_cells = int(clean.isna().sum().sum())

    clean = clean.replace(r"^\s*$", np.nan, regex=True)
    before = len(clean)
    clean = clean.dropna(how="all")
    null_rows_removed = before - len(clean)

    text_cols = _columns_of_type(schema, "text", clean)
    categorical_cols = _columns_of_type(schema, "categorical", clean)
    numeric_cols = _columns_of_type(schema, "numeric", clean)
    datetime_cols = _columns_of_type(schema, "datetime", clean)

    text_cells_cleaned = 0
    for col in text_cols:
        non_null = clean[col].notna()
        text_cells_cleaned += int(non_null.sum())
        clean.loc[non_null, col] = _clean_text_series(clean.loc[non_null, col])

    # Fill remaining nulls so processed samples and analytics payloads are JSON-safe.
    null_cells_before_fill = int(clean.isna().sum().sum())
    for col in text_cols + categorical_cols:
        clean[col] = clean[col].fillna("").astype(str).str.strip()

    for col in numeric_cols:
        numeric = pd.to_numeric(clean[col], errors="coerce")
        fill_value = numeric.median() if numeric.notna().any() else 0
        clean[col] = numeric.fillna(fill_value)

    for col in datetime_cols:
        parsed = pd.to_datetime(clean[col], errors="coerce")
        if parsed.notna().any():
            fill_value = parsed.dropna().median()
            clean[col] = parsed.fillna(fill_value)
        else:
            clean[col] = parsed

    before = len(clean)
    clean = clean.drop_duplicates()
    duplicate_rows_removed = before - len(clean)

    return clean.reset_index(drop=True), {
        "original_rows": original_rows,
        "final_rows": len(clean),
        "rows_removed": original_rows - len(clean),
        "null_rows_removed": null_rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "original_null_cells": original_null_cells,
        "null_cells_filled": null_cells_before_fill,
        "remaining_null_cells": int(clean.isna().sum().sum()),
        "text_cells_cleaned": text_cells_cleaned,
    }


def enrich_dataframe(
    df: pd.DataFrame,
    schema: dict[str, ColumnType],
    column_roles: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Add sentiment, word counts, and type-level summary stats."""
    enriched = df.copy()
    report: dict[str, Any] = {
        "primary_text": (column_roles or {}).get("primary_text"),
        "secondary_text": (column_roles or {}).get("secondary_text", []),
        "text_enrichments": [],
        "numeric_stats": {},
        "categorical_stats": {},
        "datetime_stats": {},
    }

    for col in _text_columns_for_enrichment(schema, enriched, column_roles):
        text = enriched[col].fillna("").astype(str)
        words = text.str.findall(_WORD_RE)
        enriched[f"{col}__word_count"] = words.str.len().astype(int)

        pos_counts = _count_terms(text, POSITIVE_WORDS)
        neg_counts = _count_terms(text, NEGATIVE_WORDS)
        total = pos_counts + neg_counts
        score = ((pos_counts - neg_counts) / total.replace(0, np.nan)).fillna(0).clip(-1, 1).round(3)
        enriched[f"{col}__sentiment_score"] = score
        enriched[f"{col}__sentiment_label"] = np.select(
            [score > 0.10, score < -0.10],
            ["positive", "negative"],
            default="neutral",
        )
        report["text_enrichments"].append(
            {
                "column": col,
                "added_columns": [
                    f"{col}__word_count",
                    f"{col}__sentiment_score",
                    f"{col}__sentiment_label",
                ],
            }
        )

    for col in _columns_of_type(schema, "numeric", enriched):
        numeric = pd.to_numeric(enriched[col], errors="coerce")
        valid = numeric.dropna()
        report["numeric_stats"][col] = _numeric_stats(valid)

    for col in _columns_of_type(schema, "categorical", enriched):
        vc = enriched[col].astype(str).replace("", "(blank)").value_counts().head(20)
        report["categorical_stats"][col] = {
            "unique_count": int(enriched[col].nunique(dropna=False)),
            "top_values": {str(k): int(v) for k, v in vc.items()},
        }

    for col in _columns_of_type(schema, "datetime", enriched):
        parsed = pd.to_datetime(enriched[col], errors="coerce")
        valid = parsed.dropna()
        report["datetime_stats"][col] = {
            "earliest": valid.min().isoformat() if not valid.empty else None,
            "latest": valid.max().isoformat() if not valid.empty else None,
            "span_days": int((valid.max() - valid.min()).days) if len(valid) > 1 else 0,
            "valid_count": int(len(valid)),
        }

    return enriched, report


def extract_keywords(df: pd.DataFrame, text_columns: list[str], top_n: int = 20) -> dict[str, list[dict]]:
    """Return top frequency keywords for each text column."""
    results: dict[str, list[dict]] = {}
    for col in text_columns:
        if col not in df.columns:
            continue
        text = " ".join(df[col].dropna().astype(str).head(MAX_KEYWORD_TEXT_CHARS).tolist())
        words = [w.lower() for w in _WORD_RE.findall(text)]
        filtered = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
        results[col] = [
            {"word": word, "count": int(count)}
            for word, count in Counter(filtered).most_common(top_n)
        ]
    return results


def _clean_text_series(series: pd.Series) -> pd.Series:
    text = series.astype(str)
    text = text.str.replace(_HTML_TAG_RE, " ", regex=True)
    text = text.str.replace(_URL_RE, " ", regex=True)
    text = text.str.replace(_EMOJI_RE, lambda match: f" {EMOJI_MAP[match.group(0)]} ", regex=True)
    text = text.str.replace(_CONTRACTION_RE, lambda match: CONTRACTIONS[match.group(0).lower()], regex=True)
    text = text.str.replace(_MULTI_SPACE_RE, " ", regex=True).str.strip()
    return text


def _count_terms(series: pd.Series, terms: set[str]) -> pd.Series:
    if not terms:
        return pd.Series(0, index=series.index)
    pattern = r"\b(" + "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)) + r")\b"
    return series.str.lower().str.count(pattern).astype(float)


def _numeric_stats(series: pd.Series) -> dict[str, float | int | None]:
    if series.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
        }
    return {
        "count": int(len(series)),
        "mean": round(float(series.mean()), 3),
        "median": round(float(series.median()), 3),
        "std": round(float(series.std()), 3) if len(series) > 1 else 0.0,
        "min": round(float(series.min()), 3),
        "max": round(float(series.max()), 3),
        "q25": round(float(series.quantile(0.25)), 3),
        "q75": round(float(series.quantile(0.75)), 3),
    }


def _columns_of_type(schema: dict[str, ColumnType], column_type: ColumnType, df: pd.DataFrame) -> list[str]:
    return [col for col, detected in schema.items() if detected == column_type and col in df.columns]


def _text_columns_for_enrichment(
    schema: dict[str, ColumnType],
    df: pd.DataFrame,
    column_roles: dict | None,
) -> list[str]:
    if column_roles and column_roles.get("primary_text") in df.columns:
        return [column_roles["primary_text"]]
    return _columns_of_type(schema, "text", df)
