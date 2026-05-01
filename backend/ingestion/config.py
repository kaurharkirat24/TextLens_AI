"""
Configuration for the TextLens ingestion pipeline.
"""

from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class IngestionConfig:
    # Dataset metadata
    dataset_id: Optional[str] = None

    # Column detection
    # If None, the system auto-detects the text column locally.
    text_column: Optional[str] = None

    # Validation thresholds
    max_null_ratio: float = 0.30          # warn if > 30% of text rows are null
    min_text_length: int = 3              # flag rows where stripped text is shorter
    max_text_length: int = 10_000        # flag suspiciously long rows

    # Output
    output_dir: str = "output"
    save_clean_csv: bool = True
    save_report_json: bool = True


# Candidate column name keywords (used in heuristic detection before calling Gemini)
TEXT_COLUMN_KEYWORDS = [
    "text", "review", "comment", "feedback", "description",
    "message", "content", "body", "note", "response", "opinion",
]
