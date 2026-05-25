import pytest
import pandas as pd
import asyncio
from fastapi import HTTPException
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from pathlib import Path
from uuid import uuid4

from app.main import app
from app.routers.semantic import _ACTIVE_EMBEDDING_JOBS, _start_embedding_job, _top_k
from app.core.config import settings
from app.services import dataset_manager
from app.services.dataset_manager import create_dataset, update_dataset
from app.services.embedding_service import EmbeddingServiceError, get_embedding_service
from app.services.qa_service import QAService
from app.services.semantic_dataset_service import build_metadata, mark_embedding_started
from app.services.vector_store_service import PineconeVectorStore


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
        "source": "dataset123",
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


def test_external_embedding_completion_endpoint_is_removed():
    client = TestClient(app)

    response = client.post(
        "/api/datasets/example/embeddings/external-complete",
        json={"model": "all-MiniLM-L6-v2", "dimension": 384, "count": 10, "index_name": "textlens-ai-384"},
    )

    assert response.status_code == 404


def test_mark_embedding_started_resume_preserves_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(dataset_manager, "REGISTRY_PATH", str(tmp_path / "uploads" / "registry.json"))

    dataset = create_dataset("feedback.csv", str(tmp_path / "feedback.csv"))
    update_dataset(dataset.id, embedding_status="processing", embedding_count=42, embedding_progress=0.42)

    mark_embedding_started(dataset.id, resume=True)
    resumed = dataset_manager.get_dataset(dataset.id)

    assert resumed.embedding_status == "processing"
    assert resumed.embedding_count == 42
    assert resumed.embedding_progress == 0.42


def test_processing_dataset_without_active_job_is_resumed(monkeypatch, tmp_path):
    base_dir = tmp_path / uuid4().hex
    upload_dir = base_dir / "uploads"
    upload_dir.mkdir(parents=True)
    clean_csv = base_dir / "clean.csv"
    analysis_json = base_dir / "analysis.json"
    clean_csv.write_text("text\nhello world with enough words\n", encoding="utf-8")
    analysis_json.write_text('{"column_roles": {"primary_text": "text"}}', encoding="utf-8")

    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(dataset_manager, "REGISTRY_PATH", str(upload_dir / "registry.json"))

    dataset = create_dataset("feedback.csv", str(base_dir / "feedback.csv"))
    update_dataset(
        dataset.id,
        status="analyzed",
        clean_csv_path=str(clean_csv),
        analysis_path=str(analysis_json),
        text_column="text",
        embedding_status="processing",
        embedding_count=12,
        embedding_progress=0.5,
    )

    class FakeEmbeddingService:
        model_name = "all-MiniLM-L6-v2"
        batch_size = 128

        def get_dimension(self):
            return 384

    class FakeVectorStore:
        index_name = "textlens-ai-384"

    _ACTIVE_EMBEDDING_JOBS.discard(dataset.id)
    monkeypatch.setattr("app.routers.semantic.settings.EMBEDDING_EXECUTION_MODE", "local")
    monkeypatch.setattr("app.routers.semantic.get_embedding_service", lambda: FakeEmbeddingService())
    monkeypatch.setattr("app.routers.semantic.PineconeVectorStore", FakeVectorStore)

    response = asyncio.run(_start_embedding_job(dataset.id, BackgroundTasks()))

    assert response.message == "Embedding job resumed in background."
    assert response.embedded_count == 12
    assert response.embedding_progress == 0.5
    assert dataset.id in _ACTIVE_EMBEDDING_JOBS
    _ACTIVE_EMBEDDING_JOBS.discard(dataset.id)


def test_dimension_specific_index_name_is_used_for_minilm_dimension(monkeypatch):
    monkeypatch.setattr("app.services.vector_store_service.settings.PINECONE_INDEX_NAME", "textlens-ai")

    store = PineconeVectorStore.for_dimension(384)

    assert store.index_name == "textlens-ai-384"


def test_pinecone_vector_store_reuses_grpc_index_client(monkeypatch):
    class FakeIndex:
        def __init__(self):
            self.upserts = 0

        def upsert(self, vectors, namespace, timeout):
            self.upserts += 1
            return {"upserted_count": len(vectors)}

    class FakeClient:
        def __init__(self):
            self.index_calls = []
            self.rest_index_calls = []
            self.fake_index = FakeIndex()

        def index(self, name, grpc=False):
            self.index_calls.append((name, grpc))
            return self.fake_index

        def Index(self, name):
            self.rest_index_calls.append(name)
            return self.fake_index

    fake_client = FakeClient()
    monkeypatch.setattr("app.services.vector_store_service.settings.PINECONE_API_KEY", "test-key")
    monkeypatch.setattr("app.services.vector_store_service.settings.PINECONE_INDEX_NAME", "textlens-ai-384")
    monkeypatch.setattr(PineconeVectorStore, "_client_cache", fake_client)
    monkeypatch.setattr(PineconeVectorStore, "_grpc_available", None)
    monkeypatch.setattr(PineconeVectorStore, "_index_cache", {})

    store = PineconeVectorStore()
    store.upsert_vectors([("id-1", [0.1, 0.2], {"text": "hello"})], namespace="dataset-1")
    store.upsert_vectors([("id-2", [0.2, 0.3], {"text": "world"})], namespace="dataset-1")

    assert fake_client.index_calls == [("textlens-ai-384", True)]
    assert fake_client.rest_index_calls == []
    assert fake_client.fake_index.upserts == 2


def test_gemini_embedding_provider_is_rejected_for_retrieval(monkeypatch):
    monkeypatch.setattr("app.services.embedding_service.settings.EMBEDDING_PROVIDER", "gemini")

    with pytest.raises(EmbeddingServiceError) as exc_info:
        get_embedding_service()

    assert "Gemini embeddings are disabled for retrieval" in str(exc_info.value)
