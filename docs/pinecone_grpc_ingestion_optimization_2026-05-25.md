# Pinecone gRPC Ingestion Optimization

Date: 2026-05-25

## Goal

Reduce Pinecone ingestion latency in the semantic pipeline without redesigning the current upload, checkpoint, namespace, or batching flow.

## Changes Made

### `backend/app/services/vector_store_service.py`

- Migrated Pinecone data-plane index operations to prefer the gRPC client.
- Reused one global Pinecone control client per process.
- Reused one global index client per Pinecone index name.
- Added caches for known index names and index dimensions to avoid repeated `list_indexes()` and `describe_index()` calls after validation.
- Kept index creation and dimension validation on the control-plane client.
- Preserved namespace behavior by continuing to pass `namespace=dataset_id` to fetch, query, and upsert calls.
- Preserved retry behavior and adaptive split-on-failure behavior for large upsert payloads.
- Added per-upsert timing logs with:
  - payload creation time
  - upsert time
  - vectors/sec
  - selected transport, `grpc` or `rest`
- Added a REST fallback if gRPC initialization is unavailable in the installed Pinecone SDK.

### `backend/app/services/semantic_dataset_service.py`

- Kept the existing embedding batch and Pinecone upsert batch separation.
- Kept resumable checkpoint behavior, including saved dimension and index name.
- Reused the resolved Pinecone store in async ingestion once the embedding dimension is known.
- Avoided recreating a new vector store wrapper for every async batch after index initialization.
- Added clearer ingestion logs for:
  - embed time
  - payload creation time
  - upsert time
  - flushed vector count
  - upsert vectors/sec
  - total embed, payload, and upsert seconds in the summary

### `backend/requirements.txt`

- Pinned Pinecone to `pinecone>=9.0.0` so the project targets the current SDK path where `Pinecone.index(..., grpc=True)` is available.

### `backend/tests/test_phase3_semantic.py`

- Added coverage to verify that `PineconeVectorStore` initializes the gRPC index client once and reuses it across upserts.

## Why

The observed ingestion bottleneck is mostly network/upload overhead after chunking. The previous REST path created avoidable HTTP overhead and logs showed repeated index discovery/setup activity during ingestion. gRPC reduces data-plane overhead through a persistent HTTP/2-backed transport, while process-level index caching prevents repeated client and index handle creation.

## Expected Impact

- Lower Pinecone upload latency per batch.
- Higher vectors/sec throughput for buffered upserts.
- Less repeated control-plane traffic during ingestion.
- Cleaner ingestion logs for diagnosing whether time is spent in embedding, payload creation, or Pinecone upsert.
- Minimal pipeline disruption: existing namespaces, checkpointing, dimension-specific indexes, batching, and retry behavior remain intact.

## Pinecone gRPC Limitation Impact Check

- `GrpcIndex` is sync-only.
  - No issue for the current codebase.
  - The synchronous ingestion path calls `PineconeVectorStore.upsert_vectors()` directly.
  - The async ingestion path calls the same sync method through `asyncio.to_thread(...)`, so the event loop does not directly await or block on `GrpcIndex`.

- `pinecone._grpc` is a platform-specific native extension.
  - No code-path issue was found.
  - `PineconeVectorStore` attempts to initialize a gRPC index client and falls back to the standard REST index client if gRPC initialization fails at runtime.
  - If the Pinecone package itself cannot be installed on a platform, that remains an environment/dependency compatibility issue rather than an ingestion-code issue.

- `upsert_records` and `search` on `GrpcIndex` route over REST.
  - No issue for the current ingestion pipeline.
  - TextLens_AI uses precomputed local embeddings and calls `index.upsert(...)` for vector upload.
  - Retrieval uses `index.query(...)`, not integrated-inference `search(...)`.
  - No `upsert_records(...)` or `search(...)` calls are used in the backend Pinecone integration.

## Verification

- `.\venv\Scripts\python.exe -m pytest backend\tests\test_phase3_semantic.py`
  - Result: `10 passed`
- `python -m compileall backend/app/services/vector_store_service.py backend/app/services/semantic_dataset_service.py`
  - Result: compiled successfully
- Static scan:
  - `rg -n "await .*\\.index|await .*upsert|upsert_async|query_async|PineconeFuture|upsert_records|\\.search\\(|search_records|\\.query\\(|\\.upsert\\(" backend/app backend/tests`
  - Result: only `asyncio.to_thread(...)` wrappers call sync upsert from async ingestion; backend uses `upsert(...)` and `query(...)`, with no `upsert_records(...)` or integrated-inference `search(...)` usage.
