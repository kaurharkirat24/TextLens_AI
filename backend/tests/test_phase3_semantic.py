import pytest
import pandas as pd
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.routers.semantic import _top_k
from app.services.qa_service import QAService
from app.services.semantic_dataset_service import build_metadata


def test_build_metadata_uses_required_shape_and_roles():
    row = pd.Series(
        {
            "CommentText": "Great tutorial",
            "Sentiment": "Positive",
            "Likes": 4,
            "Replies": 2,
            "PublishedAt": "2026-05-01 10:00:00",
        }
    )
    analysis = {
        "column_roles": {
            "engagement": {"likes": "Likes", "replies": "Replies"},
            "time": {"primary_datetime": "PublishedAt"},
        }
    }

    metadata = build_metadata("dataset123", 7, "Great tutorial", row, analysis)

    assert metadata == {
        "dataset_id": "dataset123",
        "row_id": 7,
        "text": "Great tutorial",
        "sentiment": "Positive",
        "engagement": 6.0,
        "timestamp": "2026-05-01 10:00:00",
    }


def test_top_k_is_capped_at_ten():
    assert _top_k(10) == 10

    with pytest.raises(HTTPException):
        _top_k(11)

    with pytest.raises(HTTPException):
        _top_k(0)


def test_qa_fallback_returns_supporting_rows_and_mode(monkeypatch):
    monkeypatch.setattr("app.services.qa_service.settings.LLM_PROVIDER", "")
    monkeypatch.setattr("app.services.qa_service.settings.LLM_API_KEY", "")

    rows = [
        {
            "id": "dataset_1",
            "text": "Users like the automation tutorial and examples",
            "metadata": {"sentiment": "Positive", "row_id": 1},
            "score": 0.91,
        },
        {
            "id": "dataset_2",
            "text": "Automation examples need more setup detail",
            "metadata": {"sentiment": "Neutral", "row_id": 2},
            "score": 0.82,
        },
    ]

    response = QAService().answer("What do users say about automation?", rows)

    assert response["mode"] == "fallback"
    assert response["supporting_rows"] == rows
    assert "Sentiment distribution" in response["answer"]
    assert "automation" in response["answer"]


def test_search_contract_requires_dataset_id():
    client = TestClient(app)

    response = client.post("/api/search", json={"query": "automation", "top_k": 5})

    assert response.status_code == 400
    assert response.json()["detail"] == "dataset_id is required"
