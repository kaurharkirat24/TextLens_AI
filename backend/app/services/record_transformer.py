"""Canonical JSON record transformation for retrieval-optimized RAG ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.services.retrieval_text_builder import RETRIEVAL_TEXT_VERSION, build_retrieval_text


RECORD_SCHEMA_VERSION = "record_v1"
CLEANING_VERSION = "clean_v1"
MAX_METADATA_VALUE_CHARS = 1000

TECHNICAL_COLUMNS = {
    "original_row_index",
    "duplicate_frequency",
}


def transform_row_to_record(
    row: pd.Series | dict[str, Any],
    *,
    dataset_id: str,
    row_index: int,
    source_file: str = "",
    uploaded_at: Any = None,
    analysis: dict[str, Any] | None = None,
    retrieval_strategy: str = "hybrid",
) -> dict[str, Any]:
    """Convert one cleaned CSV row into a canonical retrieval record."""
    row_dict = dict(row)
    original_row_index = _original_row_index(row_dict, row_index)
    business_fields = _business_fields(row_dict)
    retrieval_text = build_retrieval_text({"business_fields": business_fields}, strategy=retrieval_strategy)
    language = _detect_language(retrieval_text)
    word_count = len(re.findall(r"\b\w+\b", retrieval_text))
    text_length = len(retrieval_text)
    duplicate_frequency = _to_int(row_dict.get("duplicate_frequency"), default=1)
    quality_score = _quality_score(retrieval_text, duplicate_frequency)
    primary_text_column = _primary_text_column(analysis)

    hash_input = {
        "business_fields": business_fields,
        "cleaning_version": CLEANING_VERSION,
        "retrieval_text_version": RETRIEVAL_TEXT_VERSION,
    }
    content_hash = _content_hash(hash_input)

    record = {
        "row_id": f"{dataset_id}:{original_row_index}",
        "dataset_id": dataset_id,
        "original_row_index": original_row_index,
        "content_hash": content_hash,
        "language": language,
        "quality_score": quality_score,
        "source_file": source_file,
        "ingestion_timestamp": _timestamp(uploaded_at),
        "cleaning_version": CLEANING_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "retrieval_text_version": RETRIEVAL_TEXT_VERSION,
        "word_count": word_count,
        "text_length": text_length,
        "duplicate_frequency": duplicate_frequency,
        "primary_text_column": primary_text_column,
        "business_fields": business_fields,
        "retrieval_text": retrieval_text,
    }
    record["retrieval_text_hash"] = _content_hash({"retrieval_text": retrieval_text})
    return record


def build_vector_metadata(record: dict[str, Any], chunk_text: str | None = None) -> dict[str, Any]:
    """Build vector DB metadata from a canonical record."""
    text = chunk_text if chunk_text is not None else record.get("retrieval_text", "")
    metadata: dict[str, Any] = {
        "dataset_id": record.get("dataset_id", ""),
        "row_id": record.get("row_id", ""),
        "original_row_index": int(record.get("original_row_index", 0) or 0),
        "content_hash": record.get("content_hash", ""),
        "retrieval_text_hash": record.get("retrieval_text_hash", ""),
        "source": record.get("dataset_id", ""),
        "source_file": record.get("source_file", ""),
        "language": record.get("language", ""),
        "quality_score": float(record.get("quality_score", 0.0) or 0.0),
        "word_count": int(record.get("word_count", 0) or 0),
        "text_length": int(record.get("text_length", 0) or 0),
        "duplicate_frequency": int(record.get("duplicate_frequency", 1) or 1),
        "cleaning_version": record.get("cleaning_version", ""),
        "record_schema_version": record.get("record_schema_version", ""),
        "retrieval_text_version": record.get("retrieval_text_version", ""),
        "primary_text_column": record.get("primary_text_column", ""),
        "ingestion_timestamp": record.get("ingestion_timestamp", ""),
        "text": str(text or ""),
    }

    for column, value in (record.get("business_fields") or {}).items():
        cleaned = _clean_scalar(value)
        if cleaned == "":
            continue
        key = _metadata_key(column)
        metadata[key[:64]] = _metadata_value(value)
        metadata[f"col_{key}"[:64]] = cleaned[:MAX_METADATA_VALUE_CHARS]

    return metadata


def _business_fields(row: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for column, value in row.items():
        column_name = str(column)
        normalized = _metadata_key(column_name)
        if normalized in TECHNICAL_COLUMNS or "__" in column_name:
            continue
        cleaned = _clean_scalar(value)
        if cleaned != "":
            fields[column_name] = _json_safe_value(value)
    return fields


def _content_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _quality_score(text: str, duplicate_frequency: int) -> float:
    words = re.findall(r"\b\w+\b", text)
    if not text.strip():
        return 0.0
    score = 0.35
    score += min(len(words) / 30, 0.35)
    score += min(len(text) / 500, 0.20)
    if duplicate_frequency > 1:
        score += 0.05
    if len(words) < 3:
        score -= 0.20
    return round(max(0.0, min(score, 1.0)), 3)


def _detect_language(text: str) -> str:
    if not text.strip():
        return "unknown"
    ascii_letters = sum(1 for char in text if char.isascii() and char.isalpha())
    letters = sum(1 for char in text if char.isalpha())
    if letters and ascii_letters / letters > 0.8:
        return "en"
    return "unknown"


def _timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value
    elif value:
        try:
            dt = pd.to_datetime(value).to_pydatetime()
        except Exception:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _primary_text_column(analysis: dict[str, Any] | None) -> str:
    roles = (analysis or {}).get("column_roles") or {}
    return str(roles.get("primary_text") or "")


def _original_row_index(row: dict[str, Any], fallback: int) -> int:
    return _to_int(row.get("original_row_index"), default=fallback)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _metadata_value(value: Any) -> str | int | float | bool:
    safe = _json_safe_value(value)
    if isinstance(safe, (int, float, bool)):
        return safe
    return str(safe)[:MAX_METADATA_VALUE_CHARS]


def _json_safe_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _clean_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _metadata_key(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
