"""Phase 3 semantic search and QA endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.models.schemas import (
    EmbedRequest,
    EmbedResponse,
    ExternalEmbeddingCompleteRequest,
    QARequest,
    QAResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.embedding_service import EmbeddingServiceError, get_embedding_service
from app.services.rag_service import DatasetRAGPipeline
from app.services.semantic_dataset_service import (
    IngestionPipeline,
    SemanticDatasetError,
    load_semantic_dataset,
    mark_embedding_completed,
    mark_embedding_failed,
    mark_embedding_started,
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


@router.post("/datasets/{dataset_id}/embeddings/external-complete", response_model=EmbedResponse)
async def mark_external_embeddings_complete(dataset_id: str, request: ExternalEmbeddingCompleteRequest):
    """Mark a dataset ready after external embedding generation/upsert, such as Google Colab."""
    if request.dimension < 1:
        raise HTTPException(status_code=400, detail="dimension must be greater than zero")
    if request.count < 1:
        raise HTTPException(status_code=400, detail="count must be greater than zero")
    if not request.index_name.strip():
        raise HTTPException(status_code=400, detail="index_name is required")

    try:
        load_semantic_dataset(dataset_id)
    except SemanticDatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    mark_embedding_completed(
        dataset_id,
        model=request.model.strip() or "all-MiniLM-L6-v2",
        dimension=request.dimension,
        count=request.count,
        index_name=request.index_name.strip(),
    )
    return EmbedResponse(
        message="External embeddings marked as completed.",
        embedding_status="completed",
        dataset_id=dataset_id,
        embedded_count=request.count,
        dimension=request.dimension,
        index_name=request.index_name.strip(),
        namespace=dataset_id,
        model=request.model.strip() or "all-MiniLM-L6-v2",
        embedding_progress=1.0,
    )


async def _start_embedding_job(dataset_id: str, background_tasks: BackgroundTasks) -> EmbedResponse:
    """Validate prerequisites and queue the embedding job."""
    embedding_service = get_embedding_service()
    vector_store = PineconeVectorStore()

    try:
        meta, _, _ = load_semantic_dataset(dataset_id)
        
        # Immediate check: Is it already done?
        if (
            meta.embedding_status == "completed"
            and meta.embedding_model == embedding_service.model_name
            and meta.embedding_dimension == embedding_service.get_dimension()
            and meta.embedding_index_name
        ):
            return EmbedResponse(
                status="success",
                message="Embeddings already completed.",
                dataset_id=dataset_id,
                embedding_status="completed",
                embedded_count=meta.embedding_count,
                dimension=meta.embedding_dimension or 0,
                index_name=meta.embedding_index_name,
                namespace=dataset_id,
                model=embedding_service.model_name,
                embedding_progress=meta.embedding_progress,
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
                index_name=meta.embedding_index_name or vector_store.index_name,
                namespace=dataset_id,
                model=embedding_service.model_name,
                embedding_progress=meta.embedding_progress,
            )

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
            model=embedding_service.model_name,
        )
    except SemanticDatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to start embedding job")
        raise HTTPException(status_code=500, detail=f"Failed to start job: {exc}") from exc


async def run_background_embedding(dataset_id: str):
    """Long-running background task to embed and upsert vectors in batches."""
    logger.info("Background embedding job started for %s using new pipeline", dataset_id)
    try:
        pipeline = IngestionPipeline(dataset_id)
        pipeline.run()
        logger.info("Background embedding job completed for %s", dataset_id)
    except Exception as exc:
        logger.exception("Background embedding job failed for %s", dataset_id)
        mark_embedding_failed(dataset_id, str(exc))


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

    try:
        results = DatasetRAGPipeline().search(dataset_id, query, top_k)
    except SemanticDatasetError as exc:
        status_code = 409 if "Embeddings are not completed" in str(exc) else 404
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
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

    try:
        answer = DatasetRAGPipeline().answer(request.dataset_id, question, top_k)
    except SemanticDatasetError as exc:
        status_code = 409 if "Embeddings are not completed" in str(exc) else 404
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return answer


def _load_searchable_dataset(dataset_id: str):
    try:
        return load_semantic_dataset(dataset_id)
    except SemanticDatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _top_k(value: int) -> int:
    if value < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")
    if value > 10:
        raise HTTPException(status_code=400, detail="top_k must be <= 10")
    return value
