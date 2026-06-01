import json
from datetime import datetime, timezone

import pandas as pd

from app.core.config import settings
from app.models.schemas import DatasetMeta
from app.services.record_transformer import build_vector_metadata, transform_row_to_record
from app.services.retrieval_text_builder import build_retrieval_text
from app.services.semantic_dataset_service import IngestionPipeline


def test_transform_row_to_record_builds_stable_canonical_record():
    row = {
        "original_row_index": 42,
        "duplicate_frequency": 3,
        "product": "iPhone",
        "rating": 5,
        "review": "Great battery life",
    }
    uploaded_at = datetime(2026, 6, 1, tzinfo=timezone.utc)

    first = transform_row_to_record(
        row,
        dataset_id="dataset123",
        row_index=0,
        source_file="reviews.csv",
        uploaded_at=uploaded_at,
        analysis={"column_roles": {"primary_text": "review"}},
    )
    second = transform_row_to_record(
        dict(row),
        dataset_id="dataset123",
        row_index=99,
        source_file="reviews.csv",
        uploaded_at=uploaded_at,
        analysis={"column_roles": {"primary_text": "review"}},
    )

    assert first["row_id"] == "dataset123:42"
    assert first["original_row_index"] == 42
    assert first["duplicate_frequency"] == 3
    assert first["language"] == "en"
    assert first["primary_text_column"] == "review"
    assert first["business_fields"] == {
        "product": "iPhone",
        "rating": 5,
        "review": "Great battery life",
    }
    assert "Customer reviewed iPhone with a rating of 5." in first["retrieval_text"]
    assert "Review:\nGreat battery life" in first["retrieval_text"]
    assert first["content_hash"] == second["content_hash"]


def test_build_retrieval_text_supports_label_value_and_hybrid():
    record = {
        "business_fields": {
            "product": "iPhone",
            "rating": 5,
            "review": "Great battery life",
        }
    }

    label_value = build_retrieval_text(record, strategy="label_value")
    hybrid = build_retrieval_text(record)

    assert label_value == "Product: iPhone\nRating: 5\nReview: Great battery life"
    assert "Customer reviewed iPhone" in hybrid
    assert "Fields:" in hybrid
    assert "Product: iPhone" in hybrid


def test_build_vector_metadata_flattens_business_fields_for_filters():
    record = transform_row_to_record(
        {
            "original_row_index": 7,
            "product": "iPhone",
            "rating": 5,
            "review": "Great battery life",
        },
        dataset_id="dataset123",
        row_index=7,
        source_file="reviews.csv",
        uploaded_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        analysis={"column_roles": {"primary_text": "review"}},
    )

    metadata = build_vector_metadata(record, "Review chunk")

    assert metadata["dataset_id"] == "dataset123"
    assert metadata["row_id"] == "dataset123:7"
    assert metadata["original_row_index"] == 7
    assert metadata["content_hash"].startswith("sha256:")
    assert metadata["quality_score"] > 0
    assert metadata["product"] == "iPhone"
    assert metadata["rating"] == 5
    assert metadata["col_review"] == "Great battery life"
    assert metadata["text"] == "Review chunk"


def test_process_chunks_writes_records_and_canonical_chunk_metadata(monkeypatch, tmp_path):
    clean_csv = tmp_path / "clean_reviews.csv"
    pd.DataFrame(
        [
            {
                "original_row_index": 2,
                "duplicate_frequency": 2,
                "product": "iPhone",
                "rating": 5,
                "review": "Great battery life and the camera is excellent.",
            }
        ]
    ).to_csv(clean_csv, index=False)

    meta = DatasetMeta(
        id="dataset123",
        original_filename="reviews.csv",
        uploaded_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        clean_csv_path=str(clean_csv),
        text_column="review",
    )

    class FakeEmbeddingService:
        provider = "sentence_transformer"
        model_name = "all-MiniLM-L6-v2"
        batch_size = 128

        def get_dimension(self):
            return 384

    monkeypatch.setattr("app.services.semantic_dataset_service.get_dataset", lambda dataset_id: meta)
    monkeypatch.setattr("app.services.semantic_dataset_service.get_embedding_service", lambda: FakeEmbeddingService())
    monkeypatch.setattr(settings, "DATA_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(settings, "DATA_CLEANED_DIR", str(tmp_path / "cleaned"))
    monkeypatch.setattr(settings, "DATA_RECORDS_DIR", str(tmp_path / "records"))
    monkeypatch.setattr(settings, "DATA_CHUNKS_DIR", str(tmp_path / "chunks"))
    monkeypatch.setattr(settings, "DATA_EMBEDDINGS_DIR", str(tmp_path / "embeddings"))
    monkeypatch.setattr(settings, "DATA_TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setattr(settings, "LOGS_DIR", str(tmp_path / "logs"))

    chunks_path, _ = IngestionPipeline("dataset123")._process_chunks()
    records_path = tmp_path / "records" / "dataset123_records.jsonl"

    assert records_path.exists()
    record = json.loads(records_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["row_id"] == "dataset123:2"
    assert record["business_fields"]["product"] == "iPhone"
    assert "retrieval_text" in record

    chunk = json.loads(chunks_path.read_text(encoding="utf-8").splitlines()[0])
    metadata = json.loads(chunk["metadata"])
    assert chunk["text"] == metadata["text"]
    assert metadata["row_id"] == "dataset123:2"
    assert metadata["content_hash"] == record["content_hash"]
    assert metadata["product"] == "iPhone"
    assert metadata["rating"] == 5
