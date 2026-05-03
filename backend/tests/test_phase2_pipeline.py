from collections import Counter

import pandas as pd

from app.services.chart_generator import generate_charts
from app.services.data_processor import process_dataframe
from app.services.insight_engine import generate_insights
from app.services.pipeline import run_analysis
from app.services.schema_detector import detect_column_roles, detect_schema


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
    assert len(result["charts"]) <= 8
    assert max(Counter(chart.get("section", "other") for chart in result["charts"]).values()) <= 2


def test_role_detection_limits_sentiment_to_primary_text():
    df = pd.DataFrame(
        {
            "VideoTitle": [
                "Full launch recap and release notes",
                "Full launch recap and release notes",
                "Detailed pricing update and roadmap discussion",
            ],
            "CommentText": [
                "I love the new release and the walkthrough was helpful",
                "Terrible support experience after the release",
                "Great update, the pricing explanation was clear",
            ],
            "Likes": [12, 1, 8],
            "Replies": [2, 0, 1],
            "PublishedAt": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )

    schema = detect_schema(df)
    roles = detect_column_roles(df, schema)
    processed, report = process_dataframe(df, schema, roles)
    insights = generate_insights(processed, schema, report, {"CommentText": []}, roles)
    charts = generate_charts(insights, schema)

    assert roles["primary_text"] == "CommentText"
    assert "VideoTitle" in roles["secondary_text"]
    assert "CommentText__sentiment_label" in processed.columns
    assert "VideoTitle__sentiment_label" not in processed.columns
    assert sum(chart["title"] == "Sentiment Distribution" for chart in charts) == 1
    assert all(chart["title"] != "Column Type Distribution" for chart in charts)
    assert len(charts) <= 8
