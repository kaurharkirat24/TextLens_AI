# Urgent Issue: Q&A Embedding Timeout

Date: 2026-05-07

## Symptom

The frontend logs this from `QAPage.jsx`:

```text
[TextLens Phase 3] timeout of 300000ms exceeded
```

This means Axios waited 300,000 ms, or 5 minutes, and then cancelled the request.

The timeout is configured in:

```text
frontend/src/services/api.js
```

Current API client timeout:

```js
timeout: 300_000
```

The failure most likely happens when Q&A calls:

```js
embedDataset(selectedId)
```

from:

```text
frontend/src/pages/QAPage/QAPage.jsx
```

## Why It Happens

The current embedding flow is only partially batched.

What is batched:

- Texts are sent to Ollama in batches.
- Vectors are upserted to Pinecone in batches.

What is not chunked safely:

- The full cleaned CSV is loaded into memory.
- All text rows are extracted before embedding.
- The whole embedding job runs inside one HTTP request.
- The frontend waits for that one request to finish.

For a large CSV, embedding can take longer than 5 minutes. The browser request times out even if the backend or external services are still working.

## Why This Is Risky

- Large datasets can timeout.
- Backend memory can spike before batching begins.
- Ollama can be slow for large batches.
- Pinecone upsert can fail midway.
- Some vectors may already be written to Pinecone while dataset metadata says embedding failed.
- Retrying may be slow and inconsistent.
- Refreshing the browser loses the visible progress.

## Quick Mitigations

These help for local testing, but they do not fully solve the design problem.

### 1. Verify Ollama Is Ready

Run:

```powershell
ollama list
ollama ps
```

Make sure the embedding model exists locally:

```powershell
ollama pull qwen3-embedding:0.6b
```

### 2. Reduce Embedding Batch Size

In `.env`, try:

```env
EMBEDDING_BATCH_SIZE=8
VECTOR_UPSERT_BATCH_SIZE=50
```

This can reduce pressure on Ollama and Pinecone. It may increase total runtime, but each batch is less likely to fail.

### 3. Increase Frontend Timeout Temporarily

In `frontend/src/services/api.js`, increase:

```js
timeout: 300_000
```

to something larger for local testing, for example:

```js
timeout: 900_000
```

This only gives the request more time. It does not fix memory usage, partial writes, or lack of progress recovery.

### 4. Test With A Smaller CSV

Use a smaller sample to confirm the pipeline works:

- 100 rows
- 1,000 rows
- 5,000 rows

If small datasets work and large datasets timeout, the issue is the current long-running embedding design.

## Proper Fix

Convert embedding into a background job.

### Target Flow

1. Frontend starts embedding:

```text
POST /api/datasets/{dataset_id}/embedding-jobs
```

2. Backend immediately returns:

```json
{
  "job_id": "abc123",
  "status": "queued"
}
```

3. Backend processes embedding in the background:

- read clean CSV in chunks
- embed a batch
- upsert that batch to Pinecone
- save progress
- repeat until complete

4. Frontend polls:

```text
GET /api/embedding-jobs/{job_id}
```

5. Backend returns progress:

```json
{
  "job_id": "abc123",
  "status": "running",
  "total_rows": 50000,
  "completed_rows": 12400,
  "failed_rows": 0,
  "current_batch": 388,
  "message": "Embedding batch 388"
}
```

6. When complete, metadata is updated:

```json
{
  "embedding_status": "completed",
  "embedding_count": 50000,
  "embedding_model": "qwen3-embedding:0.6b",
  "embedding_index_name": "textlens-ai"
}
```

## Backend Requirements For The Fix

- Add an embedding job model or JSON/SQLite table.
- Track:
  - `job_id`
  - `dataset_id`
  - `status`
  - `total_rows`
  - `completed_rows`
  - `failed_rows`
  - `current_batch`
  - `started_at`
  - `finished_at`
  - `error`
  - `clean_dataset_hash`
- Read CSV using chunks instead of full-file load.
- Save progress after every batch.
- Mark dataset embedding as `running`, `completed`, or `failed`.
- Detect stale jobs when the clean CSV changes.

## Frontend Requirements For The Fix

- Replace one long `embedDataset()` request with:
  - start job request
  - polling request
  - progress UI
- Show:
  - rows embedded
  - total rows
  - current status
  - failure reason
  - retry action
- Allow page refresh without losing job status.

## Priority

High.

This issue blocks large CSV support and can leave partial vectors in Pinecone. The quick timeout increase is acceptable for local debugging, but production or serious demo usage needs background chunked embedding.

