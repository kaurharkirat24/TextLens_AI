"""
Analysis router — endpoints for data processing and auto-analytics.

Phase 2 endpoints:
  POST /api/datasets/{id}/analyze   — Run the full analysis pipeline
  GET  /api/datasets/{id}/analysis  — Retrieve saved analysis results
"""

import json
import os

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.services.dataset_manager import get_dataset, update_dataset
from app.services.analysis_pipeline import run_analysis, save_analysis_results

router = APIRouter(prefix="/api", tags=["analysis"])


# ── POST /api/datasets/{id}/analyze ──────────────────────────────────────────

@router.post("/datasets/{dataset_id}/analyze")
async def analyze_dataset(dataset_id: str):
    """
    Run the full analysis pipeline on an ingested dataset.

    Steps: schema detection → cleaning → enrichment → insights → charts.
    Returns the complete analysis response.
    """
    meta = get_dataset(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    # Prefer the clean CSV from ingestion, fall back to original upload
    csv_path = (
        meta.clean_csv_path
        if meta.clean_csv_path and os.path.exists(meta.clean_csv_path)
        else meta.file_path
    )

    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    try:
        results = run_analysis(csv_path, dataset_id=dataset_id)

        # Save results to disk for later retrieval
        analysis_path = save_analysis_results(results, settings.OUTPUT_DIR, dataset_id)

        # Update dataset status
        update_dataset(
            dataset_id,
            status="analyzed",
            analysis_path=analysis_path,
        )

        return results

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


# ── GET /api/datasets/{id}/analysis ──────────────────────────────────────────

@router.get("/datasets/{dataset_id}/analysis")
async def get_analysis(dataset_id: str):
    """
    Retrieve previously saved analysis results for a dataset.
    """
    meta = get_dataset(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    # Check for saved analysis file
    analysis_path = getattr(meta, "analysis_path", None) or ""

    # Fallback: look for the file by convention
    if not analysis_path or not os.path.exists(analysis_path):
        fallback = os.path.join(settings.OUTPUT_DIR, f"{dataset_id}_analysis.json")
        if os.path.exists(fallback):
            analysis_path = fallback
        else:
            raise HTTPException(
                status_code=404,
                detail="No analysis results found. Run POST /api/datasets/{id}/analyze first.",
            )

    try:
        with open(analysis_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load analysis: {str(exc)}")
