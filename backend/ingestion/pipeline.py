"""
Main ingestion pipeline.

Usage
-----
from ingestion.pipeline import ingest
from ingestion.config import IngestionConfig

report = ingest("data/reviews.csv", IngestionConfig())
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ingestion.column_detector import detect_text_column
from ingestion.config import IngestionConfig
from ingestion.models import IngestionReport
from ingestion.validator import validate


# ── CSV loader ────────────────────────────────────────────────────────────────

def _load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV with sensible defaults:
    - UTF-8 first, fall back to latin-1
    - Skip completely blank lines
    - Strip whitespace from column names
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.suffix}")

    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                skip_blank_lines=True,      # pandas skips lines that are entirely empty
                dtype=str,                   # keep everything as string to avoid type coercion surprises
                na_values=["", "N/A", "NA", "null", "NULL", "None", "none", "NaN"],
                keep_default_na=True,
            )
            df.columns = df.columns.str.strip()
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode {file_path} as UTF-8 or latin-1.")


# ── clean DataFrame builder ───────────────────────────────────────────────────

def _build_clean_df(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Return a copy of df with:
    - Entirely empty rows removed
    - Rows with null/empty text removed
    - A row_index column added (original row number for traceability)
    - Text column whitespace-normalised
    """
    clean = df.copy()
    clean.insert(0, "original_row_index", clean.index)

    # Drop all-null rows
    clean = clean.dropna(how="all")

    # Drop rows where text is null or whitespace-only
    clean = clean[clean[text_column].notna()]
    clean = clean[clean[text_column].str.strip() != ""].copy()

    # Normalise whitespace in the text column
    clean[text_column] = clean[text_column].str.strip()
    normalised_text = clean[text_column].str.lower().str.replace(r"\s+", " ", regex=True)
    clean["duplicate_frequency"] = normalised_text.map(normalised_text.value_counts()).astype(int)

    return clean.reset_index(drop=True)


# ── output writers ────────────────────────────────────────────────────────────

def _save_outputs(
    clean_df: pd.DataFrame,
    report: IngestionReport,
    config: IngestionConfig,
    base_name: str,
) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    if config.save_clean_csv:
        csv_path = os.path.join(config.output_dir, f"{base_name}_clean.csv")
        clean_df.to_csv(csv_path, index=False)
        report.clean_csv_path = csv_path

    if config.save_report_json:
        report_dict = {
            "success": report.success,
            "dataset_id": report.dataset_id,
            "file_path": report.file_path,
            "column_detection": asdict(report.text_column) if report.text_column else None,
            "stats": asdict(report.stats) if report.stats else None,
            "issues": [_issue_to_report_dict(i) for i in report.issues],
            "clean_csv_path": report.clean_csv_path,
            "error": report.error,
        }
        json_path = os.path.join(config.output_dir, f"{base_name}_report.json")
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        report.report_json_path = json_path


def _issue_to_report_dict(issue) -> dict:
    return {
        "severity": issue.severity.value,
        "category": issue.category,
        "count": issue.count,
        "message": issue.message,
        "row_indices_count": len(issue.row_indices),
        "row_indices_sample": issue.row_indices[:25],
    }


# ── public API ────────────────────────────────────────────────────────────────

def ingest(file_path: str, config: IngestionConfig | None = None) -> IngestionReport:
    """
    Full ingestion pipeline:
      1. Load CSV
      2. Detect text column
      3. Validate
      4. Build clean DataFrame
      5. Save outputs
      6. Return IngestionReport

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    config : IngestionConfig, optional
        Pipeline configuration. Defaults to IngestionConfig().

    Returns
    -------
    IngestionReport
        Structured report with stats, issues, and paths to saved artifacts.
    """
    if config is None:
        config = IngestionConfig()

    report = IngestionReport(dataset_id=config.dataset_id, file_path=file_path)

    try:
        # ── Step 1: Load ──────────────────────────────────────────────────────
        df = _load_csv(file_path)

        # ── Step 2: Detect text column ────────────────────────────────────────
        detection = detect_text_column(df, config)
        report.text_column = detection

        # ── Step 3: Validate ──────────────────────────────────────────────────
        issues, stats = validate(df, detection.column_name, config)
        report.issues = issues
        report.stats = stats

        # ── Step 4: Build clean DataFrame ─────────────────────────────────────
        clean_df = _build_clean_df(df, detection.column_name)
        report.stats.clean_count = len(clean_df)
        report.success = True

        # ── Step 5: Save outputs ──────────────────────────────────────────────
        base_name = Path(file_path).stem
        _save_outputs(clean_df, report, config, base_name)

    except Exception as exc:
        report.success = False
        report.error = str(exc)

    return report
