import pandas as pd

from app.services.chart_generator import generate_charts
from app.services.data_processor import process_dataframe
from app.services.insight_engine import generate_insights
from app.services.pipeline import run_analysis
from app.services.schema_detector import detect_schema


def test_phase2_pipeline_detects_processes_insights_and_charts():
    df = pd.DataFrame(
        {
            "row_id": ["A-001-x", "A-002-x", "A-003-x", "A-003-x", None],
            "comment": [
                "<b>Great product!</b> https://example.com",
                "Terrible support and slow response",
                "Great product!",
                "Great product!",
                None,
            ],
            "rating": [5, 1, 4, 4, None],
            "segment": ["pro", "free", "pro", "pro", None],
            "created_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-03", None],
        }
    )

    schema = detect_schema(df)
    assert schema["comment"] == "text"
    assert schema["rating"] == "numeric"
    assert schema["segment"] == "categorical"
    assert schema["created_at"] == "datetime"
    assert schema["row_id"] == "categorical"

    processed, report = process_dataframe(df, schema)
    assert report["cleaning"]["null_rows_removed"] == 1
    assert report["cleaning"]["duplicate_rows_removed"] == 1
    assert "comment__word_count" in processed.columns
    assert "comment__sentiment_score" in processed.columns
    assert "comment__sentiment_label" in processed.columns
    assert processed["comment"].str.contains("http", case=False).sum() == 0
    assert int(processed.isna().sum().sum()) == 0

    insights = generate_insights(processed, schema, report, {"comment": [{"word": "great", "count": 2}]})
    assert "text" in insights
    assert "numeric" in insights
    assert "categorical" in insights
    assert "datetime" in insights

    charts = generate_charts(insights, schema)
    assert charts
    assert all({"type", "x", "y", "data"}.issubset(chart.keys()) for chart in charts)


def test_run_analysis_returns_json_safe_payload():
    result = run_analysis("data/sample_youtube_comments.csv", dataset_id="unit")

    assert result["dataset_id"] == "unit"
    assert result["processed_data_sample"]
    assert result["insights"]
    assert result["charts"]
    assert result["stats"]["total_charts_generated"] == len(result["charts"])
