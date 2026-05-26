"""Dataset RAG orchestration: MiniLM retrieval, Gemini grounded generation."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.core.config import settings
from app.models.schemas import DatasetMeta
from app.services.analytics_qa_service import AnalyticsQAService, analytics_rows
from app.services.embedding_service import EmbeddingServiceError, get_embedding_service
from app.services.qa_service import QAService
from app.services.query_intent_service import QueryIntentClassifier
from app.services.retrieval_planner import RetrievalPlanner
from app.services.semantic_dataset_service import SemanticDatasetError, load_semantic_dataset
from app.services.structured_query_service import StructuredQueryService
from app.services.vector_store_service import PineconeVectorStore, VectorStoreError


logger = logging.getLogger(__name__)


class DatasetRAGPipeline:
    """Run the final RAG pipeline for one uploaded dataset.

    Pipeline:
    User query -> MiniLM embedding -> Pinecone search -> retrieved chunks ->
    Gemini grounded answer generation.
    """

    def __init__(self) -> None:
        self.embedding_service = None
        self.qa_service = QAService()
        self.intent_classifier = QueryIntentClassifier()
        self.retrieval_planner = RetrievalPlanner()
        self.analytics_service = AnalyticsQAService()
        self.structured_query_service = StructuredQueryService()

    def search(self, dataset_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """Embed the user query with MiniLM and retrieve matching Pinecone chunks."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise SemanticDatasetError("Query cannot be empty")

        meta = self._load_ready_dataset(dataset_id)
        self._validate_embedding_contract(meta)

        start = time.perf_counter()
        embedding_service = self._embedding_service()
        query_embedding = embedding_service.embed_query(cleaned_query)
        self._validate_query_dimension(meta.embedding_dimension, len(query_embedding))

        results = PineconeVectorStore(index_name=meta.embedding_index_name).query(
            query_embedding,
            namespace=dataset_id,
            top_k=top_k,
        )
        filtered = self._filter_results(results)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Dataset RAG retrieval completed dataset_id=%s model=%s dimension=%s results=%s in %.1f ms",
            dataset_id,
            embedding_service.model_name,
            len(query_embedding),
            len(filtered),
            elapsed_ms,
        )
        return filtered

    def answer(self, dataset_id: str, question: str, top_k: int) -> dict[str, Any]:
        """Route the question to semantic, analytics, or hybrid QA."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise SemanticDatasetError("Question cannot be empty")

        meta, _, _ = load_semantic_dataset(dataset_id)

        structured = self.structured_query_service.answer(meta, cleaned_question, top_k)
        if structured:
            logger.info(
                "QA answered from structured dataframe dataset_id=%s plan=%s",
                dataset_id,
                structured.plan,
            )
            return {
                "answer": structured.answer,
                "supporting_rows": structured.rows,
                "mode": "structured",
                "intent": "structured_query",
                "strategy": structured.plan.get("strategy", "dataframe"),
                "analytics": None,
                "retrieval_plan": {
                    "intent": "structured_query",
                    "strategy": structured.plan.get("strategy", "dataframe"),
                    "top_k": top_k,
                    "rationale": "Exact lookup/filter question answered directly from the cleaned dataframe.",
                    "structured_plan": structured.plan,
                },
            }

        self._ensure_ready_for_semantic(meta)
        intent = self.intent_classifier.classify(cleaned_question)
        plan = self.retrieval_planner.plan(intent, top_k)
        logger.info(
            "QA retrieval plan dataset_id=%s intent=%s strategy=%s top_k=%s rationale=%s",
            dataset_id,
            plan.intent,
            plan.strategy,
            plan.top_k,
            plan.rationale,
        )

        analytics = self.analytics_service.build_context(meta, cleaned_question, plan.intent) if plan.use_analytics else None
        rows: list[dict[str, Any]] = []
        if plan.use_semantic:
            rows.extend(self._semantic_search_ready(meta, dataset_id, cleaned_question, plan.top_k))
        if analytics and not rows:
            rows.extend(analytics_rows(analytics))

        answer = self.qa_service.answer(
            cleaned_question,
            rows,
            intent=plan.intent,
            strategy=plan.strategy,
            prompt_style=plan.prompt_style,
            analytics=analytics,
        )
        answer["retrieval_plan"] = {
            "intent": plan.intent,
            "strategy": plan.strategy,
            "top_k": plan.top_k,
            "rationale": plan.rationale,
            "classifier_confidence": intent.confidence,
            "classifier_rationale": intent.rationale,
        }
        return answer

    def _load_ready_dataset(self, dataset_id: str) -> DatasetMeta:
        meta, _, _ = load_semantic_dataset(dataset_id)
        self._ensure_ready_for_semantic(meta)
        return meta

    def _ensure_ready_for_semantic(self, meta: DatasetMeta) -> None:
        if meta.embedding_status != "completed":
            raise SemanticDatasetError("Embeddings are not completed for this dataset. Run /api/embed first.")
        if not meta.embedding_index_name:
            raise SemanticDatasetError("Dataset embedding metadata is missing the Pinecone index name.")

    def _semantic_search_ready(
        self,
        meta: DatasetMeta,
        dataset_id: str,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        self._validate_embedding_contract(meta)
        embedding_service = self._embedding_service()
        query_embedding = embedding_service.embed_query(query)
        self._validate_query_dimension(meta.embedding_dimension, len(query_embedding))
        results = PineconeVectorStore(index_name=meta.embedding_index_name).query(
            query_embedding,
            namespace=dataset_id,
            top_k=top_k,
        )
        return self._filter_results(results)

    def _embedding_service(self):
        if self.embedding_service is None:
            self.embedding_service = get_embedding_service()
        return self.embedding_service

    def _validate_embedding_contract(self, meta: DatasetMeta) -> None:
        embedding_service = self._embedding_service()
        expected_model = embedding_service.model_name
        expected_dimension = embedding_service.get_dimension()
        if meta.embedding_model and not _same_sentence_transformer_model(meta.embedding_model, expected_model):
            raise VectorStoreError(
                f"Dataset was embedded with {meta.embedding_model}, but query embedding model is "
                f"{expected_model}. Re-embed the dataset with the configured MiniLM model."
            )
        if meta.embedding_dimension and meta.embedding_dimension != expected_dimension:
            raise VectorStoreError(
                f"Dataset vectors are {meta.embedding_dimension}-dimensional, but query embeddings are "
                f"{expected_dimension}-dimensional. Do not mix embedding models or dimensions."
            )

    def _validate_query_dimension(self, expected: int | None, actual: int) -> None:
        if expected and expected != actual:
            raise VectorStoreError(
                f"Query embedding dimension {actual} does not match dataset embedding dimension {expected}"
            )

    def _filter_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        threshold = settings.RAG_MIN_RELEVANCE_SCORE
        if threshold <= 0:
            return results
        return [row for row in results if float(row.get("score") or 0.0) >= threshold]


def _same_sentence_transformer_model(stored_model: str, query_model: str) -> bool:
    stored = stored_model.strip().lower()
    query = query_model.strip().lower()
    if stored == query:
        return True
    return stored.endswith(f"/{query}") or query.endswith(f"/{stored}")
