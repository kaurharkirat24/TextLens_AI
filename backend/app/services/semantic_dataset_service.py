"""Dataset loading and vector payload preparation for Phase 3."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import settings
from app.models.schemas import DatasetMeta
from app.services.dataset_manager import get_dataset, update_dataset


class SemanticDatasetError(RuntimeError):
    """Raised when a dataset is not ready for semantic indexing."""


def load_semantic_dataset(dataset_id: str) -> tuple[DatasetMeta, pd.DataFrame, dict[str, Any]]:
    """Load dataset metadata, cleaned CSV, and Phase 2 analysis roles."""
    meta = get_dataset(dataset_id)
    if not meta:
        raise SemanticDatasetError(f"Dataset '{dataset_id}' not found")

    if not meta.clean_csv_path or not os.path.exists(meta.clean_csv_path):
        raise SemanticDatasetError("Clean dataset not found. Run ingestion/analysis before embedding.")

    try:
        df = pd.read_csv(meta.clean_csv_path)
    except Exception as exc:
        raise SemanticDatasetError(f"Failed to load clean dataset: {exc}") from exc

    analysis = _load_analysis(meta, dataset_id)
    return meta, df, analysis


def get_primary_text_column(meta: DatasetMeta, df: pd.DataFrame, analysis: dict[str, Any]) -> str:
    """Resolve the Phase 2 primary_text role without rerunning Phase 2."""
    column_roles = analysis.get("column_roles") if isinstance(analysis, dict) else {}
    primary_text = (column_roles or {}).get("primary_text") or meta.text_column
    if not primary_text or primary_text not in df.columns:
        raise SemanticDatasetError("Primary text column not found in clean dataset")
    return str(primary_text)


def extract_text_rows(df: pd.DataFrame, primary_text: str) -> list[tuple[int, str]]:
    """Return non-empty row IDs and text from the cleaned dataset."""
    rows: list[tuple[int, str]] = []
    for row_id, value in enumerate(df[primary_text].fillna("").astype(str).tolist()):
        text = value.strip()
        if text:
            rows.append((row_id, text))
    if not rows:
        raise SemanticDatasetError("Clean dataset has no non-empty primary text values")
    return rows


def build_metadata(
    dataset_id: str,
    row_id: int,
    text: str,
    row: pd.Series,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Build Pinecone-compatible metadata with the required Phase 3 shape."""
    roles = analysis.get("column_roles") if isinstance(analysis, dict) else {}
    roles = roles or {}
    engagement_roles = roles.get("engagement") or {}
    time_roles = roles.get("time") or {}

    sentiment_col = _first_present(row, ["Sentiment", "sentiment", "sentiment_label"])
    timestamp_col = time_roles.get("primary_datetime") or _first_present(
        row,
        ["PublishedAt", "published_at", "timestamp", "Timestamp", "created_at", "CreatedAt", "date", "Date"],
    )

    engagement = 0.0
    for column in engagement_roles.values():
        if column in row:
            engagement += _to_number(row[column])
    if not engagement:
        for fallback in ("engagement", "Engagement", "Likes", "likes", "Replies", "replies"):
            if fallback in row:
                engagement += _to_number(row[fallback])

    return {
        "dataset_id": dataset_id,
        "row_id": int(row_id),
        "text": text,
        "sentiment": _clean_scalar(row.get(sentiment_col, "")) if sentiment_col else "",
        "engagement": engagement,
        "timestamp": _clean_scalar(row.get(timestamp_col, "")) if timestamp_col else "",
    }


def mark_embedding_completed(
    dataset_id: str,
    *,
    model: str,
    dimension: int,
    count: int,
    index_name: str,
) -> None:
    update_dataset(
        dataset_id,
        embedding_status="completed",
        embedding_model=model,
        embedding_dimension=dimension,
        embedding_count=count,
        embedding_index_name=index_name,
        embedded_at=datetime.now(timezone.utc).isoformat(),
        error=None,
    )


def mark_embedding_failed(dataset_id: str, error: str) -> None:
    update_dataset(dataset_id, embedding_status="failed", error=error)


def _load_analysis(meta: DatasetMeta, dataset_id: str) -> dict[str, Any]:
    candidates = [
        getattr(meta, "analysis_path", "") or "",
        os.path.join(settings.OUTPUT_DIR, f"{dataset_id}_analysis.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def _first_present(row: pd.Series, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in row:
            return column
    return None


def _to_number(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clean_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
