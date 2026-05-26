"""Build and load compact dataset profiles for retrieval intelligence."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import pandas as pd

from app.core.config import settings
from app.services.pipeline import sanitize_for_json


MAX_PROFILE_VALUES_PER_COLUMN = 25
MAX_PROFILE_TEXT_EXAMPLES = 8


class DatasetProfileService:
    """Create a small, durable summary of a CSV for routing and guardrails."""

    def build(self, dataset_id: str, analysis: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Any]:
        schema_columns = _schema_columns(analysis)
        roles = analysis.get("column_roles") if isinstance(analysis, dict) else {}
        roles = roles or {}
        profile: dict[str, Any] = {
            "dataset_id": dataset_id,
            "row_count": _row_count(analysis, df),
            "columns": [],
            "column_roles": roles,
            "keywords": _keywords(analysis),
            "supported_topics": [],
            "text_profile": {},
        }

        if df is not None and not df.empty:
            profile["columns"] = _column_profiles(df, schema_columns)
            profile["text_profile"] = _text_profile(df, roles, analysis)
        else:
            profile["columns"] = [
                {"name": column, "type": kind, "sample_values": []}
                for column, kind in schema_columns.items()
            ]

        profile["supported_topics"] = _supported_topics(profile)
        return sanitize_for_json(profile)

    def save(self, dataset_id: str, profile: dict[str, Any], output_dir: str | None = None) -> str:
        path = profile_path(dataset_id, output_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(sanitize_for_json(profile), handle, indent=2)
        return path

    def load(self, dataset_id: str, output_dir: str | None = None) -> dict[str, Any]:
        path = profile_path(dataset_id, output_dir)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


def profile_path(dataset_id: str, output_dir: str | None = None) -> str:
    return os.path.join(output_dir or settings.OUTPUT_DIR, f"{dataset_id}_profile.json")


def _schema_columns(analysis: dict[str, Any]) -> dict[str, str]:
    schema = analysis.get("schema") if isinstance(analysis, dict) else {}
    columns = schema.get("columns") if isinstance(schema, dict) else {}
    if not isinstance(columns, dict):
        return {}
    return {str(column): str(kind) for column, kind in columns.items()}


def _row_count(analysis: dict[str, Any], df: pd.DataFrame | None) -> int:
    if df is not None:
        return int(len(df))
    stats = analysis.get("stats") if isinstance(analysis, dict) else {}
    return int(stats.get("total_rows_after_cleaning") or stats.get("total_rows_analyzed") or 0)


def _column_profiles(df: pd.DataFrame, schema_columns: dict[str, str]) -> list[dict[str, Any]]:
    profiles = []
    for column in df.columns:
        series = df[column]
        kind = schema_columns.get(str(column)) or _infer_kind(series)
        item: dict[str, Any] = {
            "name": str(column),
            "type": kind,
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": _sample_values(series, kind),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                item["numeric_summary"] = {
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "mean": float(numeric.mean()),
                    "median": float(numeric.median()),
                }
        profiles.append(item)
    return profiles


def _infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    sample = series.dropna().astype(str).str.strip().head(200)
    if _looks_date_like(sample):
        parsed_dates = pd.to_datetime(sample, errors="coerce")
        if not parsed_dates.empty and parsed_dates.notna().mean() >= 0.8:
            return "datetime"
    if series.nunique(dropna=True) <= min(50, max(10, int(len(series) * 0.2))):
        return "categorical"
    return "text"


def _looks_date_like(sample: pd.Series) -> bool:
    if sample.empty:
        return False
    dateish = sample.str.contains(r"\d{4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", regex=True)
    return float(dateish.mean()) >= 0.5


def _sample_values(series: pd.Series, kind: str) -> list[str]:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return []
    if kind in {"categorical", "datetime"} or values.nunique(dropna=True) <= MAX_PROFILE_VALUES_PER_COLUMN * 4:
        counts = values.value_counts().head(MAX_PROFILE_VALUES_PER_COLUMN)
        return [str(value) for value in counts.index.tolist()]
    return [str(value)[:300] for value in values.drop_duplicates().head(MAX_PROFILE_TEXT_EXAMPLES).tolist()]


def _text_profile(df: pd.DataFrame, roles: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    primary_text = roles.get("primary_text") or ""
    if primary_text not in df.columns:
        return {}
    sample = df[primary_text].dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(MAX_PROFILE_TEXT_EXAMPLES).tolist()
    return {
        "primary_text_column": primary_text,
        "examples": [text[:500] for text in sample],
        "top_terms": _keywords(analysis)[:20],
    }


def _keywords(analysis: dict[str, Any]) -> list[str]:
    raw_keywords = analysis.get("keywords") if isinstance(analysis, dict) else {}
    items: list[Any] = []
    if isinstance(raw_keywords, dict):
        for values in raw_keywords.values():
            if isinstance(values, list):
                items.extend(values)
    elif isinstance(raw_keywords, list):
        items = raw_keywords

    words = []
    for item in items:
        value = (item.get("word") or item.get("keyword")) if isinstance(item, dict) else item
        normalized = str(value or "").strip()
        if normalized:
            words.append(normalized)
    return list(dict.fromkeys(words))


def _supported_topics(profile: dict[str, Any]) -> list[str]:
    topics = []
    roles = profile.get("column_roles") or {}
    for value in _flatten_role_values(roles):
        topics.append(_display_label(value))
    for column in profile.get("columns") or []:
        topics.append(_display_label(column.get("name", "")))
    for keyword in profile.get("keywords") or []:
        topics.append(str(keyword).lower())
    return [topic for topic, _ in Counter(topic for topic in topics if topic).most_common(12)]


def _flatten_role_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        result = []
        for nested in value.values():
            result.extend(_flatten_role_values(nested))
        return result
    if isinstance(value, list):
        result = []
        for nested in value:
            result.extend(_flatten_role_values(nested))
        return result
    return [str(value)] if value else []


def _display_label(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text).lower()
