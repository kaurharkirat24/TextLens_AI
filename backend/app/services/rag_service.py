"""Dataset RAG orchestration: MiniLM retrieval, Gemini grounded generation."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.models.schemas import DatasetMeta
from app.services.qa_service import QAService
from app.services.query_router import QueryRouter
from app.services.retrieval_context import RetrievalContext
from app.services.semantic_retriever import SemanticRetriever
from app.services.semantic_cache import get_semantic_cache
from app.services.semantic_dataset_service import SemanticDatasetError, load_semantic_dataset
from app.core.config import settings


logger = logging.getLogger(__name__)


class DatasetRAGPipeline:
    """Run the final RAG pipeline for one uploaded dataset.

    Pipeline:
    User query -> MiniLM embedding -> Pinecone search -> retrieved chunks ->
    Gemini grounded answer generation.
    """

    def __init__(self) -> None:
        self.qa_service = QAService()
        self.query_router = QueryRouter()
        self.semantic_retriever = SemanticRetriever()
        self.semantic_cache = get_semantic_cache()

    def search(self, dataset_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """Embed the user query with MiniLM and retrieve matching Pinecone chunks."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise SemanticDatasetError("Query cannot be empty")

        meta = self._load_ready_dataset(dataset_id)
        self.semantic_retriever.validate_embedding_contract(meta)
        query_embedding = self.semantic_retriever.embed_query(cleaned_query)
        cached = self.semantic_cache.get(dataset_id, "search", top_k, query_embedding)
        if cached is not None:
            logger.info("Semantic cache hit for search dataset_id=%s", dataset_id)
            return cached

        start = time.perf_counter()
        filtered = self.semantic_retriever.search_by_embedding(meta, dataset_id, query_embedding, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Dataset RAG retrieval completed dataset_id=%s model=%s dimension=%s results=%s in %.1f ms",
            dataset_id,
            self.semantic_retriever.embedding_service.model_name,
            meta.embedding_dimension or self.semantic_retriever.embedding_service.get_dimension(),
            len(filtered),
            elapsed_ms,
        )
        self.semantic_cache.put(dataset_id, "search", top_k, query_embedding, filtered)
        return filtered

    def answer(self, dataset_id: str, question: str, top_k: int) -> dict[str, Any]:
        """Route the question to semantic, analytics, or hybrid QA."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise SemanticDatasetError("Question cannot be empty")
        question_embedding = self.semantic_retriever.embed_query(cleaned_question)
        cached_answer = self.semantic_cache.get(dataset_id, "qa", top_k, question_embedding)
        if cached_answer is not None:
            cached_copy = dict(cached_answer)
            plan = dict(cached_copy.get("retrieval_plan") or {})
            plan["semantic_cache"] = {"hit": True}
            cached_copy["retrieval_plan"] = plan
            return cached_copy

        context = RetrievalContext.load(dataset_id)
        routed = self.query_router.route(context, cleaned_question, top_k)

        if routed.structured:
            return {
                "answer": routed.structured.answer,
                "supporting_rows": routed.structured.rows,
                "mode": "structured",
                "intent": "structured_query",
                "strategy": routed.structured.plan.get("strategy", "dataframe"),
                "analytics": None,
                "retrieval_plan": routed.retrieval_plan,
            }

        if routed.stop:
            answer = self.qa_service.unrelated_answer(routed.retrieval_plan.get("supported_topics"))
            answer["retrieval_plan"] = routed.retrieval_plan
            return answer

        if not routed.plan:
            raise SemanticDatasetError("Could not produce a retrieval plan for this question.")

        rows: list[dict[str, Any]] = list(routed.rows or [])
        plan = routed.plan
        if plan.use_semantic:
            self._ensure_ready_for_semantic(context.meta)
            self.semantic_retriever.validate_embedding_contract(context.meta)
            semantic_rows = self.semantic_retriever.search_by_embedding(
                context.meta,
                dataset_id,
                question_embedding,
                plan.top_k,
            )
            semantic_ids = {item.get("id") for item in semantic_rows}
            rows = semantic_rows + [row for row in rows if row.get("id") not in semantic_ids]

        answer = self.qa_service.answer(
            cleaned_question,
            rows,
            intent=plan.intent,
            strategy=plan.strategy,
            prompt_style=plan.prompt_style,
            analytics=routed.analytics,
        )
        retrieval_plan = dict(routed.retrieval_plan)
        retrieval_plan["semantic_cache"] = {"hit": False}
        retrieval_plan["self_rag"] = {
            "enabled": settings.SELF_RAG_ENABLED,
            "passes": 1,
            "rewritten_query": None,
            "initial_confidence": answer.get("confidence"),
            "final_confidence": answer.get("confidence"),
        }

        if self._should_retry_with_self_rag(answer, plan, rows):
            expanded_top_k = min(plan.top_k + 3, 10)
            rewritten_query = self.qa_service.rewrite_query(cleaned_question, rows, plan.intent)
            refined_rows = self.semantic_retriever.search(context.meta, dataset_id, rewritten_query, expanded_top_k)
            refined_answer = self.qa_service.answer(
                cleaned_question,
                refined_rows,
                intent=plan.intent,
                strategy=f"{plan.strategy}_self_rag",
                prompt_style=plan.prompt_style,
                analytics=routed.analytics,
            )
            if float(refined_answer.get("confidence") or 0.0) >= float(answer.get("confidence") or 0.0):
                answer = refined_answer
                rows = refined_rows
            retrieval_plan["self_rag"] = {
                "enabled": True,
                "passes": 2,
                "rewritten_query": rewritten_query,
                "expanded_top_k": expanded_top_k,
                "initial_confidence": retrieval_plan["self_rag"]["initial_confidence"],
                "final_confidence": answer.get("confidence"),
            }

        answer["supporting_rows"] = answer.get("supporting_rows") or rows
        answer["retrieval_plan"] = retrieval_plan
        self.semantic_cache.put(dataset_id, "qa", top_k, question_embedding, dict(answer))
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

    def _should_retry_with_self_rag(self, answer: dict[str, Any], plan, rows: list[dict[str, Any]]) -> bool:
        if not settings.SELF_RAG_ENABLED:
            return False
        if settings.SELF_RAG_MAX_RETRIES < 1:
            return False
        if not plan.use_semantic:
            return False
        if not rows:
            return False
        return float(answer.get("confidence") or 0.0) < settings.SELF_RAG_CONFIDENCE_THRESHOLD
