"""
Data models for ingestion results and validation reports.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation finding for a row or column."""
    severity: Severity
    category: str          # e.g. "null_value", "empty_text", "too_short"
    row_indices: List[int] = field(default_factory=list)
    count: int = 0
    message: str = ""

    def __post_init__(self):
        if not self.count:
            self.count = len(self.row_indices)


@dataclass
class ColumnDetectionResult:
    """Result of text-column detection (heuristic or Gemini-assisted)."""
    column_name: str
    method: str            # "user_specified" | "heuristic" | "gemini"
    confidence: str        # "high" | "medium" | "low"
    candidates: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class DatasetStats:
    """Summary statistics for the loaded dataset."""
    total_rows: int = 0
    total_columns: int = 0
    text_column: str = ""
    null_count: int = 0
    null_ratio: float = 0.0
    empty_count: int = 0          # non-null but whitespace-only
    too_short_count: int = 0
    too_long_count: int = 0
    duplicate_count: int = 0
    clean_count: int = 0          # rows that passed all checks
    avg_text_length: float = 0.0
    median_text_length: float = 0.0
    sources: dict = field(default_factory=dict)   # value_counts of a 'source' column if present


@dataclass
class IngestionReport:
    """Full report returned after ingestion + validation."""
    success: bool = False
    dataset_id: Optional[str] = None
    file_path: str = ""
    text_column: ColumnDetectionResult = None
    stats: DatasetStats = None
    issues: List[ValidationIssue] = field(default_factory=list)
    clean_csv_path: Optional[str] = None
    report_json_path: Optional[str] = None
    error: Optional[str] = None

    # Convenience helpers
    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    def issues_by_severity(self, severity: Severity) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == severity]
