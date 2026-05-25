"""End-to-end Phase 2 data processing and auto-analytics pipeline."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.services.chart_generator import generate_charts
from app.services.data_processor import clean_dataframe, enrich_dataframe, extract_keywords
from app.services.insight_engine import generate_insights
from app.services.schema_detector import detect_column_roles, detect_schema, get_schema_summary

MAX_ANALYSIS_ROWS = 10_000
SAMPLE_ROWS = 50


def run_analysis(
    csv_path: str,
    dataset_id: str | None = None,
    clean_output_dir: str | None = None,
    clean_filename: str | None = None,
    clean_columns: list[str] | None = None,
) -> dict:
    """Run schema detection, processing, insights, and chart generation."""
    raw_df = load_csv(csv_path)
    original_rows = len(raw_df)

    schema = detect_schema(raw_df)
    column_roles = detect_column_roles(raw_df, schema)
    cleaned_df, cleaning_report = clean_dataframe(raw_df, schema)
    processed_df, enrichment_report = enrich_dataframe(cleaned_df, schema, column_roles)
    processing_report = {
        "cleaning": cleaning_report,
        "enrichment": enrichment_report,
    }

    clean_dataset_path = None
    if clean_output_dir and clean_filename:
        clean_dataset_path = save_clean_dataset(cleaned_df, clean_output_dir, clean_filename, clean_columns)

    sampled = False
    analysis_df = processed_df
    if len(processed_df) > MAX_ANALYSIS_ROWS:
        analysis_df = processed_df.sample(n=MAX_ANALYSIS_ROWS, random_state=42).reset_index(drop=True)
        sampled = True

    primary_text = column_roles.get("primary_text")
    text_cols = [primary_text] if primary_text in analysis_df.columns else [
        col for col, kind in schema.items() if kind == "text" and col in analysis_df.columns
    ]
    keywords = extract_keywords(analysis_df, text_cols, top_n=20)
    insights = generate_insights(analysis_df, schema, processing_report, keywords, column_roles)
    charts = generate_charts(insights, schema)

    sample_columns = [
        col
        for col in analysis_df.columns
        if col in raw_df.columns
        or col.endswith("__word_count")
        or col.endswith("__sentiment_score")
        or col.endswith("__sentiment_label")
    ]
    processed_sample = analysis_df[sample_columns].head(SAMPLE_ROWS).to_dict(orient="records")

    response = {
        "dataset_id": dataset_id,
        "schema": get_schema_summary(schema),
        "column_roles": column_roles,
        "processed_data_sample": processed_sample,
        "insights": insights,
        "charts": charts,
        "keywords": keywords,
        "cleaning_report": processing_report["cleaning"],
        "clean_dataset_path": clean_dataset_path,
        "enrichment_report": processing_report["enrichment"],
        "stats": {
            "total_rows_original": original_rows,
            "total_rows_after_cleaning": processing_report["cleaning"]["final_rows"],
            "total_rows_analyzed": len(analysis_df),
            "sampled": sampled,
            "columns_analyzed": len(schema),
            "total_charts_generated": len(charts),
        },
    }
    return sanitize_for_json(response)


def save_clean_dataset(
    cleaned_df: pd.DataFrame,
    output_dir: str,
    original_filename: str,
    columns: list[str] | None = None,
) -> str:
    """Persist the cleaned, pre-enrichment dataset as UTF-8 CSV."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(original_filename).stem or "dataset"
    path = os.path.join(output_dir, f"clean_{stem}.csv")
    output_df = cleaned_df

    if columns:
        preserved = [column for column in columns if column in output_df.columns]
        if preserved:
            output_df = output_df[preserved]

    output_df.to_csv(path, index=False, encoding="utf-8")
    return path


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file with common encoding fallbacks."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                skip_blank_lines=True,
                na_values=["", "N/A", "NA", "null", "NULL", "None", "none", "NaN"],
                keep_default_na=True,
            )
            df.columns = df.columns.astype(str).str.strip()
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} as UTF-8, UTF-8-SIG, or latin-1.")


def save_analysis_results(results: dict, output_dir: str, dataset_id: str) -> str:
    """Persist analysis JSON and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dataset_id}_analysis.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(results), handle, indent=2)
    return path


def sanitize_for_json(value: Any) -> Any:
    """Convert pandas/numpy scalars, timestamps, NaN, and Inf into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value) and value is not None:
        return None
    return value
