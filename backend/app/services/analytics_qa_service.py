"""Dataset-level analytics used by Phase 4 QA routing."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import pandas as pd

from app.core.config import settings
from app.models.schemas import DatasetMeta
from app.services.data_processor import extract_keywords
from app.services.semantic_dataset_service import SemanticDatasetError


MAX_ANALYTICS_ROWS = 50_000
MAX_EXAMPLES = 5


class AnalyticsQAService:
    """Compute compact aggregate facts from the cleaned dataset."""

    def build_context(self, meta: DatasetMeta, question: str, intent: str) -> dict[str, Any]:
        df = _load_clean_dataframe(meta)
        analysis = _load_analysis(meta)
        primary_text = _primary_text_column(meta, analysis, df)

        if len(df) > MAX_ANALYTICS_ROWS:
            df = df.sample(MAX_ANALYTICS_ROWS, random_state=42).reset_index(drop=True)

        context: dict[str, Any] = {
            "row_count_analyzed": int(len(df)),
            "primary_text_column": primary_text,
            "keywords": [],
            "sentiment_distribution": {},
            "representative_rows": [],
            "notes": [],
        }
        if not primary_text:
            context["notes"].append("No primary text column was available for text analytics.")
            return context

        keywords = extract_keywords(df, [primary_text], top_n=12).get(primary_text, [])
        context["keywords"] = keywords
        context["sentiment_distribution"] = _sentiment_distribution(df)
        context["representative_rows"] = _representative_rows(df, primary_text, keywords, question)

        if intent in {"trend", "comparison"}:
            context["time_summary"] = _time_summary(df, analysis)
        return context


def analytics_rows(analytics: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert representative analytics rows to the SearchResult-compatible shape."""
    rows = []
    for item in analytics.get("representative_rows") or []:
        row_id = item.get("row_id", "")
        rows.append(
            {
                "id": f"analytics_{row_id}",
                "text": item.get("text", ""),
                "metadata": {
                    "row_id": row_id,
                    "sentiment": item.get("sentiment", ""),
                    "engagement": item.get("engagement", 0),
                    "timestamp": item.get("timestamp", ""),
                    "source": "analytics",
                },
                "score": 0.0,
            }
        )
    return rows


def _load_clean_dataframe(meta: DatasetMeta) -> pd.DataFrame:
    if not meta.clean_csv_path or not os.path.exists(meta.clean_csv_path):
        raise SemanticDatasetError("Clean dataset not found. Run analysis before analytics QA.")
    return pd.read_csv(meta.clean_csv_path)


def _load_analysis(meta: DatasetMeta) -> dict[str, Any]:
    candidates = [
        getattr(meta, "analysis_path", "") or "",
        os.path.join(settings.OUTPUT_DIR, f"{meta.id}_analysis.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                continue
    return {}


def _primary_text_column(meta: DatasetMeta, analysis: dict[str, Any], df: pd.DataFrame) -> str:
    column = (analysis.get("column_roles") or {}).get("primary_text") or meta.text_column or ""
    return column if column in df.columns else ""


def _sentiment_distribution(df: pd.DataFrame) -> dict[str, int]:
    sentiment_col = _first_present(df, ["Sentiment", "sentiment", "sentiment_label"])
    if not sentiment_col:
        return {}
    counts = df[sentiment_col].fillna("unknown").astype(str).str.strip().replace("", "unknown").value_counts()
    return {str(label): int(count) for label, count in counts.head(10).items()}


def _representative_rows(
    df: pd.DataFrame,
    primary_text: str,
    keywords: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    terms = [str(item.get("word") or item.get("keyword") or "").lower() for item in keywords[:5]]
    terms.extend(_query_terms(question))
    terms = [term for term in dict.fromkeys(terms) if term]

    if not terms:
        sample = df.head(MAX_EXAMPLES)
    else:
        pattern = re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
        mask = df[primary_text].fillna("").astype(str).str.contains(pattern, na=False)
        sample = df[mask].head(MAX_EXAMPLES)
        if sample.empty:
            sample = df.head(MAX_EXAMPLES)

    sentiment_col = _first_present(df, ["Sentiment", "sentiment", "sentiment_label"])
    timestamp_col = _first_present(df, ["PublishedAt", "published_at", "timestamp", "Timestamp", "date", "Date"])
    rows = []
    for idx, row in sample.iterrows():
        rows.append(
            {
                "row_id": int(idx),
                "text": str(row.get(primary_text, "")),
                "sentiment": str(row.get(sentiment_col, "")) if sentiment_col else "",
                "engagement": _engagement(row),
                "timestamp": str(row.get(timestamp_col, "")) if timestamp_col else "",
            }
        )
    return rows


def _time_summary(df: pd.DataFrame, analysis: dict[str, Any]) -> dict[str, Any]:
    time_col = ((analysis.get("column_roles") or {}).get("time") or {}).get("primary_datetime")
    if not time_col or time_col not in df.columns:
        time_col = _first_present(df, ["PublishedAt", "published_at", "timestamp", "Timestamp", "date", "Date"])
    if not time_col:
        return {"available": False}

    dates = pd.to_datetime(df[time_col], errors="coerce").dropna()
    if dates.empty:
        return {"available": False, "column": time_col}
    buckets = dates.dt.to_period("M").astype(str).value_counts().sort_index()
    return {
        "available": True,
        "column": time_col,
        "first": str(dates.min()),
        "last": str(dates.max()),
        "monthly_counts": {str(label): int(count) for label, count in buckets.tail(12).items()},
    }


def _query_terms(question: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{3,}", question.lower())
    stop = {"what", "most", "common", "frequent", "users", "people", "comments", "discussed"}
    return [word for word in words if word not in stop]


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    return ""


def _engagement(row: pd.Series) -> float:
    total = 0.0
    for column in ("engagement", "Engagement", "Likes", "likes", "Replies", "replies"):
        if column in row:
            try:
                total += float(row[column])
            except (TypeError, ValueError):
                pass
    return total
