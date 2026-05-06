"""Phase 3 semantic search and QA endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    EmbedRequest,
    EmbedResponse,
    QARequest,
    QAResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.embedding_service import EmbeddingServiceError, OllamaEmbeddingService
from app.services.qa_service import QAService
from app.services.semantic_dataset_service import (
    SemanticDatasetError,
    build_metadata,
    extract_text_rows,
    get_primary_text_column,
    load_semantic_dataset,
    mark_embedding_completed,
    mark_embedding_failed,
)
from app.services.vector_store_service import PineconeVectorStore, VectorStoreError


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["semantic"])


@router.post("/datasets/{dataset_id}/embed", response_model=EmbedResponse)
async def embed_dataset(dataset_id: str):
    """Embed the cleaned dataset primary_text column into Pinecone."""
    return await _embed_dataset(dataset_id)


@router.post("/embed", response_model=EmbedResponse)
async def embed_dataset_contract(request: EmbedRequest):
    """Contract-first embedding endpoint used by the frontend."""
    dataset_id = request.dataset_id.strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    return await _embed_dataset(dataset_id)


async def _embed_dataset(dataset_id: str) -> EmbedResponse:
    """Embed one dataset namespace, overwriting stale vectors when the model changes."""
    logger.info("Embedding requested for dataset_id=%s", dataset_id)
    embedding_service = OllamaEmbeddingService()
    vector_store = PineconeVectorStore()

    try:
        meta, df, analysis = load_semantic_dataset(dataset_id)
        primary_text = get_primary_text_column(meta, df, analysis)
        text_rows = extract_text_rows(df, primary_text)
        vector_ids = [f"{dataset_id}_{row_id}" for row_id, _ in text_rows]
        sample_dimension = len(embedding_service.embed_query(text_rows[0][1]))
        live_dimension = vector_store.describe_dimension()
        if live_dimension and live_dimension != sample_dimension:
            raise VectorStoreError(
                f"Existing Pinecone index dimension {live_dimension} does not match "
                f"Ollama embedding dimension {sample_dimension} for model {embedding_service.model}"
            )
        dimension = sample_dimension
        vector_store.ensure_index(dimension)

        if (
            meta.embedding_status == "completed"
            and meta.embedding_model == embedding_service.model
            and meta.embedding_index_name == vector_store.index_name
            and meta.embedding_count >= len(text_rows)
            and meta.embedding_dimension == dimension
        ):
            return EmbedResponse(
                status="success",
                message="Embeddings already completed for this dataset.",
                dataset_id=dataset_id,
                embedding_status="completed",
                embedded_count=0,
                skipped_existing=len(text_rows),
                dimension=dimension,
                index_name=vector_store.index_name,
                namespace=dataset_id,
                model=embedding_service.model,
            )

        same_embedding_contract = (
            meta.embedding_model == embedding_service.model
            and meta.embedding_index_name == vector_store.index_name
            and meta.embedding_dimension == dimension
        )
        existing_ids = vector_store.existing_ids(vector_ids, namespace=dataset_id) if same_embedding_contract else set()
        rows_to_embed = [
            (row_id, text)
            for row_id, text in text_rows
            if f"{dataset_id}_{row_id}" not in existing_ids
        ]

        if not rows_to_embed:
            mark_embedding_completed(
                dataset_id,
                model=embedding_service.model,
                dimension=dimension,
                count=len(text_rows),
                index_name=vector_store.index_name,
            )
            return EmbedResponse(
                status="success",
                message="Embeddings already completed for this dataset.",
                dataset_id=dataset_id,
                embedding_status="completed",
                embedded_count=0,
                skipped_existing=len(text_rows),
                dimension=dimension,
                index_name=vector_store.index_name,
                namespace=dataset_id,
                model=embedding_service.model,
            )

        texts = [text for _, text in rows_to_embed]
        embeddings = embedding_service.embed_texts(texts)
        if len(embeddings) != len(texts):
            raise EmbeddingServiceError("Embedding count does not match text count")
        if any(len(vector) != dimension for vector in embeddings):
            raise EmbeddingServiceError("Embedding dimension mismatch detected")

        vectors = []
        for (row_id, text), embedding in zip(rows_to_embed, embeddings):
            metadata = build_metadata(dataset_id, row_id, text, df.iloc[row_id], analysis)
            vectors.append((f"{dataset_id}_{row_id}", embedding, metadata))

        upserted = vector_store.upsert_vectors(vectors, namespace=dataset_id)
        total_embedded = len(existing_ids) + upserted
        mark_embedding_completed(
            dataset_id,
            model=embedding_service.model,
            dimension=dimension,
            count=total_embedded,
            index_name=vector_store.index_name,
        )

        return EmbedResponse(
            status="success",
            message=f"Generated embeddings for {upserted} rows.",
            dataset_id=dataset_id,
            embedding_status="completed",
            embedded_count=upserted,
            skipped_existing=len(existing_ids),
            dimension=dimension,
            index_name=vector_store.index_name,
            namespace=dataset_id,
            model=embedding_service.model,
        )
    except SemanticDatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (EmbeddingServiceError, VectorStoreError) as exc:
        mark_embedding_failed(dataset_id, str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Dataset embedding failed")
        mark_embedding_failed(dataset_id, str(exc))
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc


@router.post("/datasets/{dataset_id}/search", response_model=SearchResponse)
async def search_dataset(dataset_id: str, request: SearchRequest):
    """Search semantically within one dataset namespace."""
    return await _search_dataset(dataset_id, request)


@router.post("/search", response_model=SearchResponse)
async def search_dataset_contract(request: SearchRequest):
    """Contract-first search endpoint used by the frontend."""
    dataset_id = (request.dataset_id or "").strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    return await _search_dataset(dataset_id, request)


async def _search_dataset(dataset_id: str, request: SearchRequest) -> SearchResponse:
    """Search semantically within one dataset namespace."""
    top_k = _top_k(request.top_k)
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    logger.info("Semantic search requested for dataset_id=%s top_k=%s", dataset_id, top_k)

    meta, _, _ = _load_searchable_dataset(dataset_id)
    if meta.embedding_status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Embeddings are not completed for this dataset. Run /api/embed first.",
        )

    try:
        embedding_service = OllamaEmbeddingService()
        vector_store = PineconeVectorStore()
        embedding = embedding_service.embed_query(query)
        _validate_query_dimension(meta.embedding_dimension, len(embedding))
        results = vector_store.query(embedding, namespace=dataset_id, top_k=top_k)
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return SearchResponse(dataset_id=dataset_id, query=query, top_k=top_k, results=results)


@router.post("/qa", response_model=QAResponse)
async def answer_question(request: QARequest):
    """Answer a question using retrieved rows, with deterministic fallback."""
    top_k = _top_k(request.top_k)
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    logger.info("QA requested for dataset_id=%s top_k=%s", request.dataset_id, top_k)

    meta, _, _ = _load_searchable_dataset(request.dataset_id)
    if meta.embedding_status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Embeddings are not completed for this dataset. Run /api/embed first.",
        )

    try:
        embedding = OllamaEmbeddingService().embed_query(question)
        _validate_query_dimension(meta.embedding_dimension, len(embedding))
        rows = PineconeVectorStore().query(embedding, namespace=request.dataset_id, top_k=top_k)
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return QAService().answer(question, rows)


def _load_searchable_dataset(dataset_id: str):
    try:
        return load_semantic_dataset(dataset_id)
    except SemanticDatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _validate_query_dimension(expected: int | None, actual: int) -> None:
    if expected and expected != actual:
        raise VectorStoreError(
            f"Query embedding dimension {actual} does not match dataset embedding dimension {expected}"
        )


def _top_k(value: int) -> int:
    if value < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")
    if value > 10:
        raise HTTPException(status_code=400, detail="top_k must be <= 10")
    return value
