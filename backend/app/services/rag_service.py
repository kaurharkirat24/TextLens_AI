"""Dataset RAG orchestration: MiniLM retrieval, Gemini grounded generation."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.core.config import settings
from app.models.schemas import DatasetMeta
from app.services.embedding_service import EmbeddingServiceError, get_embedding_service
from app.services.qa_service import QAService
from app.services.semantic_dataset_service import SemanticDatasetError, load_semantic_dataset
from app.services.vector_store_service import PineconeVectorStore, VectorStoreError


logger = logging.getLogger(__name__)


class DatasetRAGPipeline:
    """Run the final RAG pipeline for one uploaded dataset.

    Pipeline:
    User query -> MiniLM embedding -> Pinecone search -> retrieved chunks ->
    Gemini grounded answer generation.
    """

    def __init__(self) -> None:
        self.embedding_service = get_embedding_service()
        self.qa_service = QAService()

    def search(self, dataset_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """Embed the user query with MiniLM and retrieve matching Pinecone chunks."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise SemanticDatasetError("Query cannot be empty")

        meta = self._load_ready_dataset(dataset_id)
        self._validate_embedding_contract(meta)

        start = time.perf_counter()
        query_embedding = self.embedding_service.embed_query(cleaned_query)
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
            self.embedding_service.model_name,
            len(query_embedding),
            len(filtered),
            elapsed_ms,
        )
        return filtered

    def answer(self, dataset_id: str, question: str, top_k: int) -> dict[str, Any]:
        """Retrieve chunks with MiniLM, then answer with Gemini/fallback over those chunks."""
        rows = self.search(dataset_id, question, top_k)
        return self.qa_service.answer(question, rows)

    def _load_ready_dataset(self, dataset_id: str) -> DatasetMeta:
        meta, _, _ = load_semantic_dataset(dataset_id)
        if meta.embedding_status != "completed":
            raise SemanticDatasetError("Embeddings are not completed for this dataset. Run /api/embed first.")
        if not meta.embedding_index_name:
            raise SemanticDatasetError("Dataset embedding metadata is missing the Pinecone index name.")
        return meta

    def _validate_embedding_contract(self, meta: DatasetMeta) -> None:
        expected_model = self.embedding_service.model_name
        expected_dimension = self.embedding_service.get_dimension()
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
