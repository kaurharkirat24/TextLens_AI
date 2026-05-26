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
from app.services.semantic_dataset_service import SemanticDatasetError, load_semantic_dataset


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

    def search(self, dataset_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """Embed the user query with MiniLM and retrieve matching Pinecone chunks."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise SemanticDatasetError("Query cannot be empty")

        meta = self._load_ready_dataset(dataset_id)
        self.semantic_retriever.validate_embedding_contract(meta)

        start = time.perf_counter()
        filtered = self.semantic_retriever.search(meta, dataset_id, cleaned_query, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Dataset RAG retrieval completed dataset_id=%s model=%s dimension=%s results=%s in %.1f ms",
            dataset_id,
            self.semantic_retriever.embedding_service.model_name,
            meta.embedding_dimension or self.semantic_retriever.embedding_service.get_dimension(),
            len(filtered),
            elapsed_ms,
        )
        return filtered

    def answer(self, dataset_id: str, question: str, top_k: int) -> dict[str, Any]:
        """Route the question to semantic, analytics, or hybrid QA."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise SemanticDatasetError("Question cannot be empty")

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
            semantic_rows = self.semantic_retriever.search(context.meta, dataset_id, cleaned_question, plan.top_k)
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
        answer["retrieval_plan"] = routed.retrieval_plan
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
