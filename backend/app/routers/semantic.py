"""Phase 3 semantic search and QA endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.schemas import (
    EmbedRequest,
    EmbedResponse,
    EmbeddingWorkerClaimResponse,
    EmbeddingWorkerCompleteRequest,
    EmbeddingWorkerFailRequest,
    EmbeddingWorkerProgressRequest,
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
from app.services.dataset_manager import list_datasets, update_dataset
from app.services.vector_store_service import PineconeVectorStore, VectorStoreError


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["semantic"])
_ACTIVE_EMBEDDING_JOBS: set[str] = set()


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

        resume = meta.embedding_status == "processing" and dataset_id not in _ACTIVE_EMBEDDING_JOBS
        if meta.embedding_status == "processing" and not resume:
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

        if settings.EMBEDDING_EXECUTION_MODE in {"remote", "remote_colab", "colab"}:
            return _queue_remote_embedding_job(dataset_id, meta, embedding_service, vector_store, resume)

        mark_embedding_started(dataset_id, resume=resume)
        _ACTIVE_EMBEDDING_JOBS.add(dataset_id)
        
        background_tasks.add_task(run_background_embedding, dataset_id)

        return EmbedResponse(
            status="success",
            message="Embedding job resumed in background." if resume else "Embedding job started in background.",
            dataset_id=dataset_id,
            embedding_status="processing",
            embedded_count=meta.embedding_count if resume else 0,
            dimension=meta.embedding_dimension or 0,
            index_name=meta.embedding_index_name or vector_store.index_name,
            namespace=dataset_id,
            model=embedding_service.model_name,
            embedding_progress=meta.embedding_progress if resume else 0.0,
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
        await pipeline.run_async()
        logger.info("Background embedding job completed for %s", dataset_id)
    except Exception as exc:
        logger.exception("Background embedding job failed for %s", dataset_id)
        mark_embedding_failed(dataset_id, str(exc))
    finally:
        _ACTIVE_EMBEDDING_JOBS.discard(dataset_id)


def _queue_remote_embedding_job(dataset_id: str, meta, embedding_service, vector_store, resume: bool) -> EmbedResponse:
    """Prepare chunks locally, then leave embedding/upsert to an authenticated GPU worker."""
    if not settings.EMBEDDING_WORKER_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="EMBEDDING_WORKER_TOKEN must be set when EMBEDDING_EXECUTION_MODE=remote_colab.",
        )

    if meta.embedding_status == "queued":
        return EmbedResponse(
            status="success",
            message="Embedding job is queued for the remote GPU worker.",
            dataset_id=dataset_id,
            embedding_status="queued",
            embedded_count=meta.embedding_count,
            dimension=meta.embedding_dimension or embedding_service.get_dimension(),
            index_name=meta.embedding_index_name or PineconeVectorStore.for_dimension(embedding_service.get_dimension()).index_name,
            namespace=dataset_id,
            model=embedding_service.model_name,
            embedding_progress=meta.embedding_progress,
        )

    try:
        pipeline = IngestionPipeline(dataset_id)
        _, total_chunks, dimension, index_name = pipeline.prepare_remote_embedding_job()
    except Exception as exc:
        mark_embedding_failed(dataset_id, str(exc))
        raise

    update_dataset(
        dataset_id,
        embedding_status="queued",
        embedding_model=embedding_service.model_name,
        embedding_dimension=dimension,
        embedding_count=meta.embedding_count if resume else 0,
        embedding_index_name=index_name,
        embedding_progress=meta.embedding_progress if resume else 0.0,
        error=None,
    )

    return EmbedResponse(
        status="success",
        message="Embedding job queued for the remote Colab GPU worker.",
        dataset_id=dataset_id,
        embedding_status="queued",
        embedded_count=meta.embedding_count if resume else 0,
        dimension=dimension,
        index_name=index_name,
        namespace=dataset_id,
        model=embedding_service.model_name,
        embedding_progress=meta.embedding_progress if resume else 0.0,
    )


def _require_worker_token(authorization: str | None) -> None:
    if not settings.EMBEDDING_WORKER_TOKEN:
        raise HTTPException(status_code=503, detail="Remote embedding worker mode is not configured.")
    expected = f"Bearer {settings.EMBEDDING_WORKER_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid embedding worker token.")


@router.post("/workers/embedding/jobs/claim", response_model=EmbeddingWorkerClaimResponse)
async def claim_embedding_job(authorization: str | None = Header(default=None)):
    """Claim the next queued embedding job for an external GPU worker."""
    _require_worker_token(authorization)

    queued = [dataset for dataset in list_datasets() if dataset.embedding_status == "queued"]
    if not queued:
        return EmbeddingWorkerClaimResponse(message="No queued embedding jobs.")

    meta = queued[0]
    pipeline = IngestionPipeline(meta.id)
    chunks_path = pipeline._existing_chunks_path()
    if not chunks_path.exists():
        chunks_path, total_chunks, dimension, index_name = pipeline.prepare_remote_embedding_job()
    else:
        total_chunks = pipeline._count_chunks(chunks_path)
        dimension = meta.embedding_dimension or pipeline.embedding_service.get_dimension()
        index_name = meta.embedding_index_name or PineconeVectorStore.for_dimension(dimension).index_name

    start_index = max(0, int(meta.embedding_count or 0))
    update_dataset(meta.id, embedding_status="processing", error=None)

    logger.info(
        "Remote embedding worker claimed dataset=%s chunks=%s start_index=%s index=%s",
        meta.id,
        total_chunks,
        start_index,
        index_name,
    )
    return EmbeddingWorkerClaimResponse(
        job_available=True,
        dataset_id=meta.id,
        namespace=meta.id,
        index_name=index_name,
        model=meta.embedding_model or settings.EMBEDDING_MODEL_NAME,
        dimension=dimension,
        total_chunks=total_chunks,
        start_index=start_index,
        chunk_download_url=f"/api/workers/embedding/jobs/{meta.id}/chunks",
        message="Embedding job claimed.",
    )


@router.get("/workers/embedding/jobs/{dataset_id}/chunks")
async def download_embedding_chunks(dataset_id: str, authorization: str | None = Header(default=None)):
    """Download the prepared JSONL chunk file for a remote embedding worker."""
    _require_worker_token(authorization)
    try:
        pipeline = IngestionPipeline(dataset_id)
        chunks_path = pipeline._existing_chunks_path()
        if not chunks_path.exists():
            chunks_path, _, _, _ = pipeline.prepare_remote_embedding_job()
    except SemanticDatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(
        path=chunks_path,
        media_type="application/x-ndjson",
        filename=f"{dataset_id}_chunks.jsonl",
    )


@router.post("/workers/embedding/jobs/{dataset_id}/progress")
async def update_embedding_job_progress(
    dataset_id: str,
    request: EmbeddingWorkerProgressRequest,
    authorization: str | None = Header(default=None),
):
    """Update progress for a remote embedding worker."""
    _require_worker_token(authorization)
    processed = max(0, min(request.processed_chunks, request.total_chunks))
    progress = processed / max(request.total_chunks, 1)
    update_dataset(
        dataset_id,
        embedding_status="processing",
        embedding_dimension=request.dimension,
        embedding_count=processed,
        embedding_index_name=request.index_name,
        embedding_progress=progress,
        error=None,
    )
    logger.info(
        "Remote embedding progress dataset=%s processed=%s/%s message=%s",
        dataset_id,
        processed,
        request.total_chunks,
        request.message or "",
    )
    return {"status": "success", "embedding_progress": round(progress, 3), "embedding_count": processed}


@router.post("/workers/embedding/jobs/{dataset_id}/complete")
async def complete_embedding_job(
    dataset_id: str,
    request: EmbeddingWorkerCompleteRequest,
    authorization: str | None = Header(default=None),
):
    """Mark a remote embedding job complete after the worker upserts all vectors."""
    _require_worker_token(authorization)
    mark_embedding_completed(
        dataset_id,
        model=request.model,
        dimension=request.dimension,
        count=request.processed_chunks,
        index_name=request.index_name,
    )
    logger.info("Remote embedding job completed dataset=%s vectors=%s", dataset_id, request.processed_chunks)
    return {"status": "success", "embedding_status": "completed"}


@router.post("/workers/embedding/jobs/{dataset_id}/fail")
async def fail_embedding_job(
    dataset_id: str,
    request: EmbeddingWorkerFailRequest,
    authorization: str | None = Header(default=None),
):
    """Mark a remote embedding job failed."""
    _require_worker_token(authorization)
    mark_embedding_failed(dataset_id, request.error)
    logger.error("Remote embedding job failed dataset=%s error=%s", dataset_id, request.error)
    return {"status": "success", "embedding_status": "failed"}


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
