"""
Dataset manager — tracks uploaded datasets using a JSON-based registry.

Stores metadata in  uploads/registry.json  so we can list, fetch, and
update dataset records without a full database for the MVP.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.models.schemas import DatasetMeta, DatasetStatus


REGISTRY_PATH = os.path.join(settings.UPLOAD_DIR, "registry.json")


def _load_registry() -> list[dict]:
    """Load the registry from disk, returning an empty list if missing."""
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save_registry(records: list[dict]) -> None:
    """Persist the registry to disk."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)


def create_dataset(original_filename: str, file_path: str) -> DatasetMeta:
    """Register a new dataset and return its metadata."""
    meta = DatasetMeta(
        id=uuid.uuid4().hex[:12],
        original_filename=original_filename,
        uploaded_at=datetime.now(timezone.utc),
        status=DatasetStatus.UPLOADED,
        file_path=file_path,
    )
    records = _load_registry()
    records.append(meta.model_dump(mode="json"))
    _save_registry(records)
    return meta


def get_dataset(dataset_id: str) -> Optional[DatasetMeta]:
    """Fetch a single dataset by ID, or None if not found."""
    for record in _load_registry():
        if record["id"] == dataset_id:
            return DatasetMeta(**record)
    return None


def list_datasets() -> list[DatasetMeta]:
    """Return all datasets, newest first."""
    records = _load_registry()
    metas = [DatasetMeta(**r) for r in records]
    metas.sort(key=lambda m: m.uploaded_at, reverse=True)
    return metas


def update_dataset(dataset_id: str, **fields) -> Optional[DatasetMeta]:
    """Update specific fields on a dataset record."""
    records = _load_registry()
    for i, record in enumerate(records):
        if record["id"] == dataset_id:
            record.update(fields)
            records[i] = record
            _save_registry(records)
            return DatasetMeta(**record)
    return None
