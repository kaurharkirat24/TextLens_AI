"""Phase 3 semantic search and QA endpoints."""

from __future__ import annotations

import logging
from app.core.config import settings

from fastapi import APIRouter, HTTPException, BackgroundTasks

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
async def embed_dataset(dataset_id: str, background_tasks: BackgroundTasks):
    """Embed the cleaned dataset primary_text column into Pinecone."""
    return await _start_embedding_job(dataset_id, background_tasks)


@router.post("/embed", response_model=EmbedResponse)
async def embed_dataset_contract(request: EmbedRequest, background_tasks: BackgroundTasks):
    """Contract-first embedding endpoint used by the frontend."""
    dataset_id = request.dataset_id.strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    return await _start_embedding_job(dataset_id, background_tasks)


async def _start_embedding_job(dataset_id: str, background_tasks: BackgroundTasks) -> EmbedResponse:
    """Validate prerequisites and queue the embedding job."""
    embedding_service = OllamaEmbeddingService()
    vector_store = PineconeVectorStore()

    try:
        meta, df, analysis = load_semantic_dataset(dataset_id)
        
        # Immediate check: Is it already done?
        if (
            meta.embedding_status == "completed"
            and meta.embedding_model == embedding_service.model
            and meta.embedding_index_name == vector_store.index_name
        ):
            return EmbedResponse(
                status="success",
                message="Embeddings already completed.",
                dataset_id=dataset_id,
                embedding_status="completed",
                embedded_count=meta.embedding_count,
                dimension=meta.embedding_dimension or 0,
                index_name=vector_store.index_name,
                namespace=dataset_id,
                model=embedding_service.model,
            )

        # Check if already processing
        if meta.embedding_status == "processing":
             return EmbedResponse(
                status="success",
                message="Embedding job is already in progress.",
                dataset_id=dataset_id,
                embedding_status="processing",
                embedded_count=meta.embedding_count,
                dimension=meta.embedding_dimension or 0,
                index_name=vector_store.index_name,
                namespace=dataset_id,
                model=embedding_service.model,
            )

        # Start background job
        from app.services.semantic_dataset_service import mark_embedding_started
        mark_embedding_started(dataset_id)
        
        background_tasks.add_task(run_background_embedding, dataset_id)

        return EmbedResponse(
            status="success",
            message="Embedding job started in background.",
            dataset_id=dataset_id,
            embedding_status="processing",
            embedded_count=0,
            dimension=0,
            index_name=vector_store.index_name,
            namespace=dataset_id,
            model=embedding_service.model,
        )
    except SemanticDatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to start embedding job")
        raise HTTPException(status_code=500, detail=f"Failed to start job: {exc}") from exc


async def run_background_embedding(dataset_id: str):
    """Long-running background task to embed and upsert vectors in batches."""
    logger.info("Background embedding job started for %s", dataset_id)
    embedding_service = OllamaEmbeddingService()
    vector_store = PineconeVectorStore()

    try:
        from app.services.semantic_dataset_service import (
            mark_embedding_progress,
            mark_embedding_started,
        )
        
        meta, df, analysis = load_semantic_dataset(dataset_id)
        primary_text = get_primary_text_column(meta, df, analysis)
        text_rows = extract_text_rows(df, primary_text)
        total_rows = len(text_rows)

        # Dimension validation (using first row)
        sample_embedding = embedding_service.embed_query(text_rows[0][1])
        dimension = len(sample_embedding)
        vector_store.ensure_index(dimension)

        # Check existing to skip work
        vector_ids = [f"{dataset_id}_{row_id}" for row_id, _ in text_rows]
        existing_ids = vector_store.existing_ids(vector_ids, namespace=dataset_id)
        
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
                count=total_rows,
                index_name=vector_store.index_name,
            )
            return

        # Process in batches
        from app.services.embedding_service import _batches
        batch_size = settings.EMBEDDING_BATCH_SIZE
        processed_count = len(existing_ids)

        for i, batch in enumerate(_batches(rows_to_embed, batch_size)):
            batch_texts = [text for _, text in batch]
            batch_embeddings = embedding_service.embed_texts(batch_texts)
            
            vectors = []
            for (row_id, text), emb in zip(batch, batch_embeddings):
                metadata = build_metadata(dataset_id, row_id, text, df.iloc[row_id], analysis)
                vectors.append((f"{dataset_id}_{row_id}", emb, metadata))
            
            vector_store.upsert_vectors(vectors, namespace=dataset_id)
            
            processed_count += len(batch)
            mark_embedding_progress(dataset_id, processed_count / total_rows)
            logger.info("Dataset %s embedding progress: %.1f%%", dataset_id, (processed_count/total_rows)*100)

        mark_embedding_completed(
            dataset_id,
            model=embedding_service.model,
            dimension=dimension,
            count=total_rows,
            index_name=vector_store.index_name,
        )
        logger.info("Background embedding job completed for %s", dataset_id)

    except Exception as exc:
        logger.exception("Background embedding job failed for %s", dataset_id)
        mark_embedding_failed(dataset_id, str(exc))
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
