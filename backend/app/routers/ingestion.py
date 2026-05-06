"""
Ingestion router - file upload, dataset listing, and preview endpoints.
"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.models.schemas import (
    ColumnDetectionSchema,
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetStatsSchema,
    DatasetStatus,
    IngestionReportSchema,
    SeverityLevel,
    UploadResponse,
    ValidationIssueSchema,
)
from app.services.dataset_manager import (
    create_dataset,
    get_dataset,
    list_datasets,
    update_dataset,
)

# ── Import the existing ingestion pipeline ────────────────────────────────────
from ingestion.config import IngestionConfig
from ingestion.pipeline import ingest

router = APIRouter(prefix="/api", tags=["ingestion"])
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by keeping only alphanumeric, dots, dashes and underscores.
    """
    path = Path(filename)
    name = path.stem
    ext = path.suffix.lower()
    # Remove everything except alphanumeric, dots, dashes, underscores
    clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    # Ensure it's not too long and doesn't start with dots/dashes
    clean_name = clean_name.strip('._-')
    if not clean_name:
        clean_name = "dataset"
    return f"{clean_name[:100]}{ext}"


def _report_to_schema(report, filename: str) -> IngestionReportSchema:
    """Convert the dataclass-based IngestionReport → Pydantic schema."""
    text_col = None
    if report.text_column:
        text_col = ColumnDetectionSchema(
            column_name=report.text_column.column_name,
            method=report.text_column.method,
            confidence=report.text_column.confidence,
            candidates=report.text_column.candidates,
            reasoning=report.text_column.reasoning,
        )

    stats = None
    if report.stats:
        stats = DatasetStatsSchema(
            total_rows=report.stats.total_rows,
            total_columns=report.stats.total_columns,
            text_column=report.stats.text_column,
            null_count=report.stats.null_count,
            null_ratio=report.stats.null_ratio,
            empty_count=report.stats.empty_count,
            too_short_count=report.stats.too_short_count,
            too_long_count=report.stats.too_long_count,
            duplicate_count=report.stats.duplicate_count,
            clean_count=report.stats.clean_count,
            avg_text_length=report.stats.avg_text_length,
            median_text_length=report.stats.median_text_length,
        )

    issues = [
        ValidationIssueSchema(
            severity=SeverityLevel(issue.severity.value),
            category=issue.category,
            count=issue.count,
            message=issue.message,
        )
        for issue in report.issues
    ]

    return IngestionReportSchema(
        success=report.success,
        dataset_id=report.dataset_id,
        file_name=filename,
        text_column=text_col,
        stats=stats,
        issues=issues,
        error=report.error,
    )


# ── POST /api/upload ─────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    text_column: Optional[str] = Query(None, description="Specify the text column name (auto-detected if omitted)"),
):
    """
    Upload a file, run the ingestion pipeline, and return a structured report.
    Currently supports CSV files.
    """
    logger.info("Upload requested for filename=%s text_column=%s", file.filename, text_column)
    # Validate file type
    allowed_extensions = {".csv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_extensions)}",
        )

    # Enforce file size limit
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    # Sanitize filename
    safe_name = sanitize_filename(file.filename)

    # Save uploaded file to disk
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    dataset_meta = create_dataset(file.filename, "")
    logger.info("Created dataset registry entry dataset_id=%s", dataset_meta.id)

    upload_path = os.path.join(settings.UPLOAD_DIR, f"{dataset_meta.id}_{safe_name}")
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    update_dataset(dataset_meta.id, file_path=upload_path)

    # Run the existing ingestion pipeline
    config = IngestionConfig(
        dataset_id=dataset_meta.id,
        text_column=text_column,
        output_dir=settings.OUTPUT_DIR,
        max_null_ratio=settings.MAX_NULL_RATIO,
        min_text_length=settings.MIN_TEXT_LENGTH,
        max_text_length=settings.MAX_TEXT_LENGTH,
    )

    report = ingest(upload_path, config)

    # Update dataset metadata with results
    update_fields = {
        "status": DatasetStatus.INGESTED.value if report.success else DatasetStatus.FAILED.value,
        "total_rows": report.stats.total_rows if report.stats else 0,
        "clean_rows": report.stats.clean_count if report.stats else 0,
        "text_column": report.text_column.column_name if report.text_column else None,
        "clean_csv_path": report.clean_csv_path or "",
        "report_json_path": report.report_json_path or "",
        "embedding_status": "not_started" if report.success else None,
        "embedding_model": None,
        "embedding_dimension": None,
        "embedding_count": 0,
        "embedding_index_name": None,
        "embedded_at": None,
        "error": report.error,
    }
    update_dataset(dataset_meta.id, **update_fields)
    logger.info(
        "Upload pipeline finished for dataset_id=%s success=%s clean_rows=%s",
        dataset_meta.id,
        report.success,
        report.stats.clean_count if report.stats else 0,
    )

    report_schema = _report_to_schema(report, file.filename)

    return UploadResponse(dataset_id=dataset_meta.id, report=report_schema)


# ── GET /api/datasets ────────────────────────────────────────────────────────

@router.get("/datasets", response_model=DatasetListResponse)
async def get_datasets():
    """List all uploaded datasets, newest first."""
    datasets = list_datasets()
    return DatasetListResponse(datasets=datasets, total=len(datasets))


# ── GET /api/datasets/{id}/preview ───────────────────────────────────────────

@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def preview_dataset(
    dataset_id: str,
    limit: int = Query(50, ge=1, le=500, description="Number of rows to return"),
):
    """
    Return the first N rows of the clean CSV for a dataset.
    Falls back to the original upload if no clean CSV exists yet.
    """
    meta = get_dataset(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    # Prefer clean CSV, fallback to original
    csv_path = meta.clean_csv_path if meta.clean_csv_path and os.path.exists(meta.clean_csv_path) else meta.file_path

    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = pd.read_csv(csv_path, dtype=str, nrows=limit)
    total = meta.clean_rows or meta.total_rows

    return DatasetPreviewResponse(
        columns=df.columns.tolist(),
        rows=df.fillna("").to_dict(orient="records"),
        total_rows=total,
        showing=len(df),
    )
