"""Compatibility wrapper for the Phase 2 pipeline module."""

from app.services.pipeline import (
    load_csv,
    run_analysis,
    sanitize_for_json,
    save_analysis_results,
    save_clean_dataset,
)

__all__ = ["load_csv", "run_analysis", "sanitize_for_json", "save_analysis_results", "save_clean_dataset"]
