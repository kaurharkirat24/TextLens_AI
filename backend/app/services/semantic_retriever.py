"""Semantic retrieval over the dataset Pinecone namespace."""

from __future__ import annotations

from app.core.config import settings
from app.models.schemas import DatasetMeta
from app.services.embedding_service import get_embedding_service
from app.services.vector_store_service import PineconeVectorStore, VectorStoreError


class SemanticRetriever:
    """Keep query embedding, Pinecone lookup, and score filtering in one place."""

    def __init__(self) -> None:
        self._embedding_service = None

    def search(self, meta: DatasetMeta, dataset_id: str, query: str, top_k: int) -> list[dict]:
        query_embedding = self.embedding_service.embed_query(query)
        self._validate_query_dimension(meta.embedding_dimension, len(query_embedding))
        results = PineconeVectorStore(index_name=meta.embedding_index_name).query(
            query_embedding,
            namespace=dataset_id,
            top_k=top_k,
        )
        return self._filter_results(results)

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def validate_embedding_contract(self, meta: DatasetMeta) -> None:
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

    def _filter_results(self, results: list[dict]) -> list[dict]:
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
