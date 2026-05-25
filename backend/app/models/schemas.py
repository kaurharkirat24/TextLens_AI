"""
Pydantic schemas for API request / response bodies.

These mirror the dataclass models in ingestion/models.py but are
designed for FastAPI serialisation and OpenAPI docs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class SeverityLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DatasetStatus(str, Enum):
    UPLOADED = "uploaded"
    INGESTED = "ingested"
    PREPROCESSED = "preprocessed"
    ANALYZED = "analyzed"
    PROCESSING = "processing"
    EMBEDDED = "embedded"
    FAILED = "failed"


# ── Column Detection ─────────────────────────────────────────────────────────

class ColumnDetectionSchema(BaseModel):
    column_name: str
    method: str
    confidence: str
    candidates: list[str] = Field(default_factory=list)
    reasoning: str = ""


# ── Validation ────────────────────────────────────────────────────────────────

class ValidationIssueSchema(BaseModel):
    severity: SeverityLevel
    category: str
    count: int = 0
    message: str = ""


class DatasetStatsSchema(BaseModel):
    total_rows: int = 0
    total_columns: int = 0
    text_column: str = ""
    null_count: int = 0
    null_ratio: float = 0.0
    empty_count: int = 0
    too_short_count: int = 0
    too_long_count: int = 0
    duplicate_count: int = 0
    clean_count: int = 0
    avg_text_length: float = 0.0
    median_text_length: float = 0.0


# ── Ingestion Report ─────────────────────────────────────────────────────────

class IngestionReportSchema(BaseModel):
    success: bool
    dataset_id: Optional[str] = None
    file_name: str
    text_column: Optional[ColumnDetectionSchema] = None
    stats: Optional[DatasetStatsSchema] = None
    issues: list[ValidationIssueSchema] = Field(default_factory=list)
    error: Optional[str] = None


# ── Dataset ───────────────────────────────────────────────────────────────────

class DatasetMeta(BaseModel):
    """Metadata record for a dataset stored on disk."""
    id: str
    original_filename: str
    uploaded_at: datetime
    status: DatasetStatus = DatasetStatus.UPLOADED
    text_column: Optional[str] = None
    total_rows: int = 0
    clean_rows: int = 0
    file_path: Optional[str] = ""          # path to the uploaded file
    clean_csv_path: Optional[str] = ""     # path to the cleaned file
    report_json_path: Optional[str] = ""   # path to the persisted ingestion report
    analysis_path: Optional[str] = ""      # path to the analysis results JSON
    embedding_status: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    embedding_count: int = 0
    embedding_index_name: Optional[str] = None
    embedded_at: Optional[datetime] = None
    embedding_progress: float = 0.0      # 0.0 to 1.0
    error: Optional[str] = None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetMeta] = Field(default_factory=list)
    total: int = 0


class DatasetPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    showing: int


# ── Upload response ──────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    dataset_id: str
    report: IngestionReportSchema


# Semantic Search + QA

class EmbedRequest(BaseModel):
    dataset_id: str


class EmbedResponse(BaseModel):
    status: str = "success"
    message: str
    embedding_status: str
    dataset_id: str
    embedded_count: int
    skipped_existing: int = 0
    dimension: int
    index_name: str
    namespace: str
    model: str
    embedding_progress: float = 0.0


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    dataset_id: Optional[str] = None


class SearchResult(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class SearchResponse(BaseModel):
    dataset_id: str
    query: str
    top_k: int
    results: list[SearchResult]


class QARequest(BaseModel):
    dataset_id: str
    question: str
    top_k: int = 5


class QAResponse(BaseModel):
    answer: str
    supporting_rows: list[SearchResult]
    mode: str


class EmbeddingWorkerClaimResponse(BaseModel):
    job_available: bool = False
    dataset_id: Optional[str] = None
    namespace: Optional[str] = None
    index_name: Optional[str] = None
    model: Optional[str] = None
    dimension: Optional[int] = None
    total_chunks: int = 0
    start_index: int = 0
    chunk_download_url: Optional[str] = None
    message: str = ""


class EmbeddingWorkerProgressRequest(BaseModel):
    processed_chunks: int
    total_chunks: int
    dimension: int
    index_name: str
    message: Optional[str] = None


class EmbeddingWorkerCompleteRequest(BaseModel):
    processed_chunks: int
    total_chunks: int
    dimension: int
    index_name: str
    model: str


class EmbeddingWorkerFailRequest(BaseModel):
    error: str
