"""
Dataset manager — tracks uploaded datasets using SQLite.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings
from app.core.database import get_db
from app.models.schemas import DatasetMeta, DatasetStatus


REGISTRY_PATH = f"{settings.UPLOAD_DIR}/registry.json"


def create_dataset(original_filename: str, file_path: str) -> DatasetMeta:
    """Register a new dataset in SQLite and return its metadata."""
    meta = DatasetMeta(
        id=uuid.uuid4().hex[:12],
        original_filename=original_filename,
        uploaded_at=datetime.now(timezone.utc),
        status=DatasetStatus.UPLOADED,
        file_path=file_path,
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO datasets (id, original_filename, uploaded_at, status, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                meta.id,
                meta.original_filename,
                meta.uploaded_at.isoformat(),
                meta.status.value,
                meta.file_path,
            ),
        )
        conn.commit()
    return meta


def get_dataset(dataset_id: str) -> Optional[DatasetMeta]:
    """Fetch a single dataset from SQLite by ID, or None if not found."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row:
            return DatasetMeta(**dict(row))
    return None


def list_datasets() -> list[DatasetMeta]:
    """Return all datasets from SQLite, newest first."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM datasets ORDER BY uploaded_at DESC").fetchall()
        return [DatasetMeta(**dict(row)) for row in rows]


def update_dataset(dataset_id: str, **fields) -> Optional[DatasetMeta]:
    """Update specific fields on a dataset record in SQLite."""
    if not fields:
        return get_dataset(dataset_id)

    # Handle enums and datetimes for SQLite
    processed_fields = {}
    for k, v in fields.items():
        if hasattr(v, "value"):  # Enum
            processed_fields[k] = v.value
        elif hasattr(v, "isoformat"):  # datetime
            processed_fields[k] = v.isoformat()
        else:
            processed_fields[k] = v

    set_clause = ", ".join([f"{k} = ?" for k in processed_fields.keys()])
    values = list(processed_fields.values())
    values.append(dataset_id)

    with get_db() as conn:
        conn.execute(f"UPDATE datasets SET {set_clause} WHERE id = ?", values)
        conn.commit()

    return get_dataset(dataset_id)
