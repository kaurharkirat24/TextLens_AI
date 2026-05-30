"""Colab GPU worker for TextLens_AI remote embeddings.

Run this from a Google Colab GPU runtime after exposing the local backend with
ngrok or Cloudflare Tunnel. The worker polls the backend, downloads prepared
chunks, embeds them on CUDA, upserts directly to Pinecone, and reports progress.
"""

from __future__ import annotations

import json
import os
import time
import subprocess
import concurrent.futures
from pathlib import Path
from urllib.parse import urljoin

import requests
from sentence_transformers import SentenceTransformer


BACKEND_URL = os.environ.get("TEXTLENS_BACKEND_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("TEXTLENS_EMBEDDING_WORKER_TOKEN", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "512"))
PINECONE_UPSERT_BATCH_SIZE = int(os.environ.get("PINECONE_UPSERT_BATCH_SIZE", "500"))
PROGRESS_EVERY_BATCHES = int(os.environ.get("PROGRESS_EVERY_BATCHES", "2"))
POST_RETRY_ATTEMPTS = int(os.environ.get("TEXTLENS_POST_RETRY_ATTEMPTS", "5"))
POST_RETRY_BASE_DELAY_SECONDS = float(os.environ.get("TEXTLENS_POST_RETRY_BASE_DELAY_SECONDS", "2"))
TRANSIENT_STATUS_CODES = {429, 502, 503, 504}

UPSERT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=5)
pending_futures = []


def main() -> None:
    _validate_config()
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {WORKER_TOKEN}"})

    claim = _post(session, "/workers/embedding/jobs/claim", {}).json()
    if not claim.get("job_available"):
        print(claim.get("message") or "No queued embedding jobs.")
        return

    dataset_id = claim["dataset_id"]
    namespace = claim["namespace"]
    index_name = claim["index_name"]
    total_chunks = int(claim["total_chunks"])
    start_index = int(claim.get("start_index") or 0)
    model_name = claim.get("model") or EMBEDDING_MODEL_NAME
    dimension = int(claim.get("dimension") or 0)
    chunk_url = claim["chunk_download_url"]

    print(
        f"Claimed dataset={dataset_id} chunks={total_chunks} "
        f"start_index={start_index} index={index_name}"
    )

    download_start = time.perf_counter()
    chunks_path = _download_chunks(session, dataset_id, chunk_url)
    print(f"Download phase took {time.perf_counter() - download_start:.1f} seconds.")
    
    model_start = time.perf_counter()
    model = SentenceTransformer(model_name, device="cuda")
    print(f"Model load phase took {time.perf_counter() - model_start:.1f} seconds.")
    
    actual_dimension = int(model.get_embedding_dimension())
    if dimension and actual_dimension != dimension:
        raise RuntimeError(f"Model dimension {actual_dimension} does not match backend dimension {dimension}")

    index = _pinecone_index(index_name)
    processed = start_index
    pending_vectors = []
    batch_texts = []
    batch_rows = []
    batch_number = 0
    start = time.perf_counter()

    try:
        for row_index, row in _iter_rows(chunks_path):
            if row_index < start_index:
                continue

            batch_texts.append(row["text"])
            batch_rows.append(row)
            if len(batch_texts) < EMBEDDING_BATCH_SIZE:
                continue

            batch_number += 1
            processed = _embed_and_buffer(
                model,
                index,
                namespace,
                dataset_id,
                batch_texts,
                batch_rows,
                pending_vectors,
                processed,
            )
            batch_texts.clear()
            batch_rows.clear()

            if batch_number % PROGRESS_EVERY_BATCHES == 0:
                _flush(index, namespace, pending_vectors)
                _progress(session, dataset_id, processed, total_chunks, actual_dimension, index_name)

        if batch_texts:
            batch_number += 1
            processed = _embed_and_buffer(
                model,
                index,
                namespace,
                dataset_id,
                batch_texts,
                batch_rows,
                pending_vectors,
                processed,
            )

        _flush(index, namespace, pending_vectors, force=True)
        _complete(session, dataset_id, processed, total_chunks, actual_dimension, index_name, model_name)
        elapsed = time.perf_counter() - start
        print(f"Completed dataset={dataset_id} vectors={processed} elapsed_seconds={elapsed:.1f}")
    except Exception as exc:
        _fail(session, dataset_id, str(exc))
        raise


def _validate_config() -> None:
    missing = [
        name
        for name, value in {
            "TEXTLENS_BACKEND_URL": BACKEND_URL,
            "TEXTLENS_EMBEDDING_WORKER_TOKEN": WORKER_TOKEN,
            "PINECONE_API_KEY": PINECONE_API_KEY,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def _pinecone_index(index_name: str):
    from pinecone.grpc import PineconeGRPC

    pc = PineconeGRPC(api_key=PINECONE_API_KEY)
    return pc.Index(name=index_name)


def _download_chunks(session: requests.Session, dataset_id: str, chunk_url: str) -> Path:
    url = chunk_url if chunk_url.startswith("http") else urljoin(f"{BACKEND_URL}/", chunk_url.lstrip("/"))
    target = Path(f"/content/{dataset_id}_chunks.jsonl")
    
    print(f"Downloading chunks from {url}...")
    cmd = [
        "curl", "-s", "-S", "-L", "--fail", "--compressed",
        "--retry", "5",
        "--retry-delay", "2",
        "--retry-connrefused",
        "-H", f"Authorization: Bearer {WORKER_TOKEN}",
        "-o", str(target),
        url
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Downloaded chunks to {target}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Download failed. curl error: {exc.stderr}")
        
    return target


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                yield index, json.loads(line)


def _embed_and_buffer(
    model,
    index,
    namespace: str,
    dataset_id: str,
    texts: list[str],
    rows: list[dict],
    pending_vectors: list[dict],
    processed: int,
) -> int:
    embed_start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).tolist()
    embed_ms = (time.perf_counter() - embed_start) * 1000

    for embedding, row in zip(embeddings, rows):
        metadata = json.loads(row["metadata"])
        vector_id = f"{dataset_id}_{metadata['row_id']}_{metadata['chunk_id']}"
        pending_vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})

    processed += len(embeddings)
    print(f"Embedded batch vectors={len(embeddings)} processed={processed} embed_ms={embed_ms:.1f}")
    if len(pending_vectors) >= PINECONE_UPSERT_BATCH_SIZE:
        _flush(index, namespace, pending_vectors)
    return processed


def _flush(index, namespace: str, pending_vectors: list[dict], force: bool = False) -> None:
    while len(pending_vectors) >= PINECONE_UPSERT_BATCH_SIZE:
        batch = pending_vectors[:PINECONE_UPSERT_BATCH_SIZE]
        del pending_vectors[:PINECONE_UPSERT_BATCH_SIZE]
        future = UPSERT_EXECUTOR.submit(_upsert, index, namespace, batch)
        pending_futures.append(future)
        
    if force and pending_vectors:
        batch = list(pending_vectors)
        pending_vectors.clear()
        future = UPSERT_EXECUTOR.submit(_upsert, index, namespace, batch)
        pending_futures.append(future)
        
    if force:
        # Wait for all final upserts
        concurrent.futures.wait(pending_futures)
        for f in pending_futures:
            f.result()  # Propagate exceptions
        pending_futures.clear()
    else:
        # Propagate exceptions from completed futures without blocking
        done = [f for f in pending_futures if f.done()]
        for f in done:
            f.result()
            pending_futures.remove(f)


def _upsert(index, namespace: str, vectors: list[dict]) -> None:
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            start = time.perf_counter()
            index.upsert(vectors=vectors, namespace=namespace, timeout=60)
            elapsed = time.perf_counter() - start
            print(f"Upserted vectors={len(vectors)} seconds={elapsed:.2f} vectors_per_sec={len(vectors) / max(elapsed, 0.001):.1f}")
            return
        except Exception as exc:
            if attempt == max_retries:
                raise
            delay = 2 * attempt
            print(f"Warning: Upsert failed (attempt {attempt}/{max_retries}), retrying in {delay}s: {exc}")
            time.sleep(delay)


def _progress(
    session: requests.Session,
    dataset_id: str,
    processed: int,
    total: int,
    dimension: int,
    index_name: str,
) -> None:
    try:
        _post(
            session,
            f"/workers/embedding/jobs/{dataset_id}/progress",
            {
                "processed_chunks": processed,
                "total_chunks": total,
                "dimension": dimension,
                "index_name": index_name,
                "message": "Colab GPU worker progress",
            },
        )
    except requests.RequestException as exc:
        print(f"Warning: progress update failed for dataset={dataset_id} processed={processed}: {exc}")


def _complete(
    session: requests.Session,
    dataset_id: str,
    processed: int,
    total: int,
    dimension: int,
    index_name: str,
    model_name: str,
) -> None:
    _post(
        session,
        f"/workers/embedding/jobs/{dataset_id}/complete",
        {
            "processed_chunks": processed,
            "total_chunks": total,
            "dimension": dimension,
            "index_name": index_name,
            "model": model_name,
        },
    )


def _fail(session: requests.Session, dataset_id: str, error: str) -> None:
    try:
        _post(session, f"/workers/embedding/jobs/{dataset_id}/fail", {"error": error})
    except requests.RequestException as exc:
        print(f"Warning: failed to report worker error for dataset={dataset_id}: {exc}")


def _post(session: requests.Session, path: str, payload: dict):
    url = f"{BACKEND_URL}/api{path}"
    last_error: Exception | None = None
    for attempt in range(1, POST_RETRY_ATTEMPTS + 1):
        try:
            response = session.post(url, json=payload, timeout=120)
            if response.status_code in TRANSIENT_STATUS_CODES and attempt < POST_RETRY_ATTEMPTS:
                delay = POST_RETRY_BASE_DELAY_SECONDS * attempt
                print(
                    f"Transient backend response status={response.status_code} "
                    f"path={path} attempt={attempt}/{POST_RETRY_ATTEMPTS} retry_in={delay:.1f}s"
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= POST_RETRY_ATTEMPTS:
                break
            delay = POST_RETRY_BASE_DELAY_SECONDS * attempt
            print(
                f"Transient backend request failure path={path} "
                f"attempt={attempt}/{POST_RETRY_ATTEMPTS} retry_in={delay:.1f}s error={exc}"
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


if __name__ == "__main__":
    main()
