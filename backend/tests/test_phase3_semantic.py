import pytest
import pandas as pd
import asyncio
from fastapi import HTTPException
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.main import app
from app.models.schemas import DatasetMeta
from app.routers.semantic import _ACTIVE_EMBEDDING_JOBS, _start_embedding_job, _top_k
from app.core.config import settings
from app.services import dataset_manager
from app.services.dataset_relevance_service import DatasetRelevanceService
from app.services.dataset_manager import create_dataset, update_dataset
from app.services.dataset_profile_service import DatasetProfileService
from app.services.embedding_service import EmbeddingServiceError, get_embedding_service
from app.services.qa_service import QAService
from app.services.query_intent_service import QueryIntentClassifier
from app.services.query_router import QueryRouter
from app.services.rag_service import DatasetRAGPipeline
from app.services.retrieval_context import RetrievalContext
from app.services.retrieval_planner import RetrievalPlanner
from app.services.semantic_dataset_service import build_metadata, build_row_text, mark_embedding_started
from app.services.structured_query_service import StructuredQueryService
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

    assert {
        key: metadata[key]
        for key in ("dataset_id", "row_id", "text", "source", "sentiment", "engagement", "timestamp")
    } == {
        "dataset_id": "dataset123",
        "row_id": 7,
        "text": "Great tutorial",
        "source": "dataset123",
        "sentiment": "Positive",
        "engagement": 6.0,
        "timestamp": "2026-05-01 10:00:00",
    }
    assert metadata["col_commenttext"] == "Great tutorial"
    assert metadata["col_likes"] == "4"


def test_build_row_text_preserves_structured_movie_fields():
    row = {
        "title": "Sankofa",
        "director": "Haile Gerima",
        "rating": "TV-MA",
        "duration": "125 min",
        "description": "An American model slips back in time.",
    }

    text = build_row_text(row, {"column_roles": {"primary_text": "description"}}, "description")

    assert "Title: Sankofa" in text
    assert "Director: Haile Gerima" in text
    assert "Rating: TV-MA" in text
    assert "Duration: 125 min" in text
    assert "Description: An American model slips back in time." in text


def test_structured_query_answers_movie_lookup_and_filters(tmp_path):
    csv_path = tmp_path / "movies.csv"
    pd.DataFrame(
        [
            {
                "show_id": "s1",
                "type": "Movie",
                "title": "Dick Johnson Is Dead",
                "director": "Kirsten Johnson",
                "rating": "PG-13",
                "duration": "90 min",
                "description": "A filmmaker stages her father's death.",
            },
            {
                "show_id": "s8",
                "type": "Movie",
                "title": "Sankofa",
                "director": "Haile Gerima",
                "rating": "TV-MA",
                "duration": "125 min",
                "description": "An American model slips back in time.",
            },
            {
                "show_id": "s10",
                "type": "Movie",
                "title": "The Starling",
                "director": "Theodore Melfi",
                "rating": "PG-13",
                "duration": "104 min",
                "description": "A woman adjusts to life after loss.",
            },
        ]
    ).to_csv(csv_path, index=False)
    meta = DatasetMeta(
        id="movies",
        original_filename="movies.csv",
        uploaded_at=datetime.now(timezone.utc),
        clean_csv_path=str(csv_path),
    )

    service = StructuredQueryService()

    directed = service.answer(meta, "Who directed Sankofa?", top_k=5)
    assert directed is not None
    assert directed.answer == "Sankofa's director is Haile Gerima."
    assert directed.plan["strategy"] == "dataframe_lookup"

    pg13 = service.answer(meta, "Find PG-13 movies.", top_k=5)
    assert pg13 is not None
    assert "Found 2 row(s)" in pg13.answer
    assert "PG-13" in pg13.answer
    assert [row["metadata"]["title"] for row in pg13.rows] == ["Dick Johnson Is Dead", "The Starling"]

    under_100 = service.answer(meta, "Which movies are under 100 minutes?", top_k=5)
    assert under_100 is not None
    assert "Found 1 row(s) with duration < 100 minutes." in under_100.answer
    assert under_100.rows[0]["metadata"]["title"] == "Dick Johnson Is Dead"


def test_structured_query_answers_generic_analytics_over_catalog(tmp_path):
    csv_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        [
            {
                "title": "Fast Night",
                "type": "Movie",
                "country": "India",
                "listed_in": "Thrillers, Crime",
                "rating": "PG-13",
                "duration": "90 min",
                "release_year": 2020,
                "description": "A dark thriller about a robbery.",
            },
            {
                "title": "Quiet Road",
                "type": "Movie",
                "country": "India",
                "listed_in": "Dramas",
                "rating": "PG-13",
                "duration": "110 min",
                "release_year": 2021,
                "description": "An emotional family drama.",
            },
            {
                "title": "City Cells",
                "type": "TV Show",
                "country": "United States",
                "listed_in": "Crime, TV Dramas",
                "rating": "TV-MA",
                "duration": "2 Seasons",
                "release_year": 2021,
                "description": "A prison show about gangs.",
            },
            {
                "title": "Street Files",
                "type": "TV Show",
                "country": "India",
                "listed_in": "Crime, Thrillers",
                "rating": "TV-MA",
                "duration": "4 Seasons",
                "release_year": 2022,
                "description": "Investigators follow organized crime.",
            },
        ]
    ).to_csv(csv_path, index=False)
    meta = DatasetMeta(
        id="catalog",
        original_filename="catalog.csv",
        uploaded_at=datetime.now(timezone.utc),
        clean_csv_path=str(csv_path),
    )
    analysis = {
        "schema": {
            "columns": {
                "title": "categorical",
                "type": "categorical",
                "country": "categorical",
                "listed_in": "categorical",
                "rating": "categorical",
                "duration": "categorical",
                "release_year": "numeric",
                "description": "text",
            }
        },
        "column_roles": {"primary_text": "description", "content": {"title": "title"}},
    }
    service = StructuredQueryService()

    average_duration = service.answer(meta, "Average movie duration?", top_k=5, analysis=analysis)
    country_most = service.answer(meta, "Which country has the most titles?", top_k=5, analysis=analysis)
    genre_most = service.answer(meta, "Which genre is most common?", top_k=5, analysis=analysis)
    type_counts = service.answer(meta, "Movies vs TV Shows count?", top_k=5, analysis=analysis)
    pg13_count = service.answer(meta, "How many PG-13 movies exist?", top_k=5, analysis=analysis)
    release_year = service.answer(meta, "Which release year has most titles?", top_k=5, analysis=analysis)
    seasons = service.answer(meta, "Average seasons per TV Show?", top_k=5, analysis=analysis)

    assert average_duration is not None
    assert "average duration where type is Movie is 100.0 minutes" in average_duration.answer
    assert country_most is not None
    assert "India with 3 row(s)" in country_most.answer
    assert genre_most is not None
    assert "Crime with 3 row(s)" in genre_most.answer
    assert type_counts is not None
    assert "Movie: 2" in type_counts.answer
    assert "TV Show: 2" in type_counts.answer
    assert pg13_count is not None
    assert "Found 2 row(s)" in pg13_count.answer
    assert release_year is not None
    assert "2021 with 2 row(s)" in release_year.answer
    assert seasons is not None
    assert "average duration where type is TV Show is 3.0 seasons" in seasons.answer


def test_structured_query_answers_hybrid_filter_recommendations(tmp_path):
    csv_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        [
            {
                "title": "Short Fright",
                "type": "Movie",
                "country": "United States",
                "listed_in": "Horror, Thrillers",
                "rating": "PG-13",
                "duration": "88 min",
                "release_year": 2019,
                "description": "A tense haunted house story.",
            },
            {
                "title": "Long Fright",
                "type": "Movie",
                "country": "United States",
                "listed_in": "Horror",
                "rating": "PG-13",
                "duration": "130 min",
                "release_year": 2018,
                "description": "A long supernatural mystery.",
            },
            {
                "title": "Gang Unit",
                "type": "TV Show",
                "country": "India",
                "listed_in": "Crime, TV Dramas",
                "rating": "TV-MA",
                "duration": "1 Season",
                "release_year": 2021,
                "description": "Police track gangs after a prison escape.",
            },
            {
                "title": "Old Gang Unit",
                "type": "TV Show",
                "country": "India",
                "listed_in": "Crime",
                "rating": "TV-MA",
                "duration": "1 Season",
                "release_year": 2019,
                "description": "Gangs fight for territory.",
            },
        ]
    ).to_csv(csv_path, index=False)
    meta = DatasetMeta(
        id="catalog",
        original_filename="catalog.csv",
        uploaded_at=datetime.now(timezone.utc),
        clean_csv_path=str(csv_path),
    )
    analysis = {
        "schema": {
            "columns": {
                "title": "categorical",
                "type": "categorical",
                "country": "categorical",
                "listed_in": "categorical",
                "rating": "categorical",
                "duration": "categorical",
                "release_year": "numeric",
                "description": "text",
            }
        },
        "column_roles": {"primary_text": "description", "content": {"title": "title"}},
    }
    service = StructuredQueryService()

    horror = service.answer(meta, "Recommend horror movies under 100 mins.", top_k=5, analysis=analysis)
    gangs = service.answer(meta, "Find TV Shows about gangs released after 2020.", top_k=5, analysis=analysis)

    assert horror is not None
    assert "Short Fright" in horror.answer
    assert "Long Fright" not in horror.answer
    assert gangs is not None
    assert "Gang Unit" in gangs.answer
    assert "Old Gang Unit" not in gangs.answer


def test_structured_query_uses_numeric_metric_for_highest_questions(tmp_path):
    csv_path = tmp_path / "catalog.csv"
    pd.DataFrame(
        [
            {"title": "Quiet Road", "type": "Movie", "IMDb Rating": 7.1, "description": "Drama."},
            {"title": "Bright Night", "type": "Movie", "IMDb Rating": 8.4, "description": "Thriller."},
            {"title": "Long Arc", "type": "TV Show", "IMDb Rating": 8.9, "description": "Series."},
        ]
    ).to_csv(csv_path, index=False)
    meta = DatasetMeta(
        id="catalog",
        original_filename="catalog.csv",
        uploaded_at=datetime.now(timezone.utc),
        clean_csv_path=str(csv_path),
    )
    analysis = {
        "schema": {
            "columns": {
                "title": "categorical",
                "type": "categorical",
                "IMDb Rating": "numeric",
                "description": "text",
            }
        },
        "column_roles": {"primary_text": "description", "content": {"title": "title"}},
    }

    response = StructuredQueryService().answer(meta, "Which movie has highest IMDb rating?", top_k=5, analysis=analysis)

    assert response is not None
    assert response.plan["strategy"] == "dataframe_extreme"
    assert "bright night has the highest imdb rating" in response.answer.lower()


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


def test_dataset_relevance_flags_general_knowledge_question_for_arbitrary_dataset(tmp_path):
    csv_path = tmp_path / "support_tickets.csv"
    pd.DataFrame(
        [
            {
                "TicketId": "A-100",
                "CustomerSegment": "Enterprise",
                "IssueCategory": "Billing",
                "ResolutionTimeHours": 6,
                "AgentNotes": "Customer was charged twice after plan upgrade.",
            }
        ]
    ).to_csv(csv_path, index=False)
    meta = DatasetMeta(
        id="tickets",
        original_filename="support_tickets.csv",
        uploaded_at=datetime.now(timezone.utc),
        clean_csv_path=str(csv_path),
    )
    analysis = {
        "schema": {
            "columns": {
                "TicketId": "categorical",
                "CustomerSegment": "categorical",
                "IssueCategory": "categorical",
                "ResolutionTimeHours": "numeric",
                "AgentNotes": "text",
            }
        },
        "column_roles": {"primary_text": "AgentNotes"},
    }

    result = DatasetRelevanceService().assess(meta, analysis, "What is photosynthesis?")

    assert result.is_related is False
    assert "agent notes" in result.supported_topics
    assert "customer segment" in result.supported_topics


def test_dataset_relevance_allows_question_that_matches_arbitrary_schema_and_values(tmp_path):
    csv_path = tmp_path / "support_tickets.csv"
    pd.DataFrame(
        [
            {
                "TicketId": "A-100",
                "CustomerSegment": "Enterprise",
                "IssueCategory": "Billing",
                "ResolutionTimeHours": 6,
                "AgentNotes": "Customer was charged twice after plan upgrade.",
            }
        ]
    ).to_csv(csv_path, index=False)
    meta = DatasetMeta(
        id="tickets",
        original_filename="support_tickets.csv",
        uploaded_at=datetime.now(timezone.utc),
        clean_csv_path=str(csv_path),
    )
    analysis = {
        "schema": {
            "columns": {
                "TicketId": "categorical",
                "CustomerSegment": "categorical",
                "IssueCategory": "categorical",
                "ResolutionTimeHours": "numeric",
                "AgentNotes": "text",
            }
        },
        "column_roles": {"primary_text": "AgentNotes"},
    }

    result = DatasetRelevanceService().assess(meta, analysis, "How many Enterprise billing tickets are there?")

    assert result.is_related is True
    assert any(signal.startswith("column:") or signal.startswith("value:") for signal in result.matched_signals)


def test_dataset_relevance_matches_title_value_beyond_sample_limit(tmp_path):
    csv_path = tmp_path / "catalog.csv"
    rows = [
        {
            "show_id": f"s{i}",
            "type": "Movie",
            "title": f"Catalog Item {i:03d}",
            "director": f"Person {i:03d}",
            "cast": "",
            "country": "Unknown",
            "date_added": "January 1, 2020",
            "release_year": 2020,
            "rating": "TV-PG",
            "duration": "90 min",
            "listed_in": "Example",
            "description": "Synthetic catalog row for relevance testing.",
        }
        for i in range(180)
    ]
    rows.append(
        {
            "show_id": "s999",
            "type": "Movie",
            "title": "Sankofa",
            "director": "Haile Gerima",
            "cast": "",
            "country": "United States",
            "date_added": "September 24, 2021",
            "release_year": 1993,
            "rating": "TV-MA",
            "duration": "125 min",
            "listed_in": "Drama",
            "description": "A model slips back in time.",
        }
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    meta = DatasetMeta(
        id="catalog",
        original_filename="catalog.csv",
        uploaded_at=datetime.now(timezone.utc),
        clean_csv_path=str(csv_path),
    )
    analysis = {
        "schema": {
            "columns": {
                "show_id": "categorical",
                "type": "categorical",
                "title": "categorical",
                "director": "categorical",
                "cast": "text",
                "country": "categorical",
                "date_added": "datetime",
                "release_year": "numeric",
                "rating": "categorical",
                "duration": "categorical",
                "listed_in": "categorical",
                "description": "text",
            }
        },
        "column_roles": {
            "primary_text": "description",
            "content": {"id": "show_id", "title": "title"},
            "time": {"primary_datetime": "date_added"},
        },
    }

    result = DatasetRelevanceService().assess(meta, analysis, "Who directed Sankofa?")

    assert result.is_related is True
    assert "column:director" in result.matched_signals
    assert "value:title=Sankofa" in result.matched_signals


def test_dataset_relevance_allows_dataset_level_question_only_when_dataset_is_referenced(tmp_path):
    csv_path = tmp_path / "inventory.csv"
    pd.DataFrame(
        [
            {
                "Sku": "SKU-1",
                "WarehouseZone": "North",
                "StockCount": 42,
                "RestockDate": "2026-05-20",
            }
        ]
    ).to_csv(csv_path, index=False)
    meta = DatasetMeta(
        id="inventory",
        original_filename="inventory.csv",
        uploaded_at=datetime.now(timezone.utc),
        clean_csv_path=str(csv_path),
    )
    analysis = {
        "schema": {
            "columns": {
                "Sku": "categorical",
                "WarehouseZone": "categorical",
                "StockCount": "numeric",
                "RestockDate": "datetime",
            }
        },
        "column_roles": {},
    }
    service = DatasetRelevanceService()

    unrelated_aggregate = service.assess(meta, analysis, "What are the most common symptoms of flu?")
    dataset_summary = service.assess(meta, analysis, "Summarize this uploaded dataset.")

    assert unrelated_aggregate.is_related is False
    assert dataset_summary.is_related is True


def test_dataset_profile_captures_schema_values_and_supported_topics(tmp_path):
    df = pd.DataFrame(
        [
            {"IssueCategory": "Billing", "CustomerSegment": "Enterprise", "ResolutionTimeHours": 6},
            {"IssueCategory": "Login", "CustomerSegment": "SMB", "ResolutionTimeHours": 2},
            {"IssueCategory": "Billing", "CustomerSegment": "SMB", "ResolutionTimeHours": 4},
        ]
    )
    analysis = {
        "schema": {
            "columns": {
                "IssueCategory": "categorical",
                "CustomerSegment": "categorical",
                "ResolutionTimeHours": "numeric",
            }
        },
        "column_roles": {"engagement": {"resolution_time": "ResolutionTimeHours"}},
    }

    profile = DatasetProfileService().build("tickets", analysis, df)

    assert profile["row_count"] == 3
    assert "issue category" in profile["supported_topics"]
    issue_column = next(column for column in profile["columns"] if column["name"] == "IssueCategory")
    assert "Billing" in issue_column["sample_values"]


def test_query_router_uses_profile_backed_context_for_dataset_summary(tmp_path):
    csv_path = tmp_path / "tickets.csv"
    pd.DataFrame(
        [
            {"IssueCategory": "Billing", "AgentNotes": "Customer was charged twice."},
            {"IssueCategory": "Login", "AgentNotes": "User could not reset password."},
        ]
    ).to_csv(csv_path, index=False)
    analysis = {
        "schema": {"columns": {"IssueCategory": "categorical", "AgentNotes": "text"}},
        "column_roles": {"primary_text": "AgentNotes"},
        "keywords": {"AgentNotes": [{"word": "customer", "count": 1}]},
    }
    meta = DatasetMeta(
        id="tickets",
        original_filename="tickets.csv",
        uploaded_at=datetime.now(timezone.utc),
        clean_csv_path=str(csv_path),
    )
    profile = DatasetProfileService().build("tickets", analysis, pd.read_csv(csv_path))
    context = RetrievalContext(dataset_id="tickets", meta=meta, analysis=analysis, _profile=profile)

    routed = QueryRouter().route(context, "Summarize this uploaded dataset.", requested_top_k=5)

    assert routed.plan is not None
    assert routed.plan.intent == "summarization"
    assert routed.plan.strategy == "hybrid"
    assert routed.analytics is not None


def test_rag_pipeline_returns_polite_out_of_scope_answer(monkeypatch, tmp_path):
    csv_path = tmp_path / "support_tickets.csv"
    pd.DataFrame(
        [
            {
                "TicketId": "A-100",
                "CustomerSegment": "Enterprise",
                "IssueCategory": "Billing",
                "ResolutionTimeHours": 6,
                "AgentNotes": "Customer was charged twice after plan upgrade.",
            }
        ]
    ).to_csv(csv_path, index=False)
    meta = DatasetMeta(
        id="tickets",
        original_filename="support_tickets.csv",
        uploaded_at=datetime.now(timezone.utc),
        clean_csv_path=str(csv_path),
        embedding_status="completed",
        embedding_index_name="textlens-ai-384",
        embedding_dimension=384,
    )
    analysis = {
        "schema": {
            "columns": {
                "TicketId": "categorical",
                "CustomerSegment": "categorical",
                "IssueCategory": "categorical",
                "ResolutionTimeHours": "numeric",
                "AgentNotes": "text",
            }
        },
        "column_roles": {"primary_text": "AgentNotes"},
    }

    monkeypatch.setattr("app.services.rag_service.load_semantic_dataset", lambda dataset_id: (meta, pd.DataFrame(), analysis))

    response = DatasetRAGPipeline().answer("tickets", "What is photosynthesis?", top_k=5)

    assert response["mode"] == "out_of_scope"
    assert response["strategy"] == "guardrail"
    assert response["supporting_rows"] == []
    assert "does not appear related to the uploaded dataset" in response["answer"]
    assert "agent notes" in response["answer"]


def test_query_intent_classifier_detects_aggregation():
    intent = QueryIntentClassifier().classify("What are the most frequent topics discussed in the comments?")

    assert intent.intent == "aggregation"
    assert intent.confidence >= 0.8


def test_retrieval_planner_routes_aggregation_to_analytics():
    intent = QueryIntentClassifier().classify("Most common issues?")
    plan = RetrievalPlanner().plan(intent, requested_top_k=10)

    assert plan.strategy == "analytics"
    assert plan.use_analytics is True
    assert plan.use_semantic is False
    assert plan.top_k == 5


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
