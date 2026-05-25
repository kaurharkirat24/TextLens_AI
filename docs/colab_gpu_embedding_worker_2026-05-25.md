# Colab GPU Embedding Worker

Date: 2026-05-25

## Purpose

Move TextLens_AI embedding generation off the local CPU and onto a temporary Google Colab GPU runtime while keeping the app dashboard workflow automated.

Users still click **Embed** in the app. They do not manually use Colab. Colab is a private GPU worker controlled by the developer/operator.

## Architecture

```text
User clicks Embed in TextLens_AI
        |
Local backend cleans/chunks dataset
        |
Backend marks dataset embedding_status=queued
        |
Colab GPU worker polls tunneled backend
        |
Colab downloads chunk JSONL
        |
Colab embeds with all-MiniLM-L6-v2 on cuda
        |
Colab upserts vectors directly to Pinecone over gRPC
        |
Colab reports progress/completion to backend
        |
Dashboard polling shows progress/completed
```

## Why Colab Calls The Backend

The local backend cannot directly call a Colab runtime in a reliable production-style way. Instead, the Colab worker makes outbound HTTPS requests to your backend.

For local development, expose the backend with a tunnel:

- ngrok: `ngrok http 8000`
- Cloudflare Tunnel: `cloudflared tunnel --url http://localhost:8000`

Use the generated HTTPS URL as `TEXTLENS_BACKEND_URL` in Colab.

## Backend Settings

`.env` now includes:

```env
EMBEDDING_EXECUTION_MODE=remote_colab
EMBEDDING_WORKER_TOKEN=textlens-local-colab-worker-token
PINECONE_UPSERT_BATCH_SIZE=500
```

Restart the backend after changing these values.

## Backend Endpoints Added

All worker endpoints require:

```http
Authorization: Bearer <EMBEDDING_WORKER_TOKEN>
```

Endpoints:

- `POST /api/workers/embedding/jobs/claim`
- `GET /api/workers/embedding/jobs/{dataset_id}/chunks`
- `POST /api/workers/embedding/jobs/{dataset_id}/progress`
- `POST /api/workers/embedding/jobs/{dataset_id}/complete`
- `POST /api/workers/embedding/jobs/{dataset_id}/fail`

## Colab Setup

1. Open Google Colab.
2. Runtime -> Change runtime type -> GPU.
3. Run:

```python
!pip install -q sentence-transformers pinecone requests
```

4. Set runtime variables:

```python
import os

os.environ["TEXTLENS_BACKEND_URL"] = "https://YOUR-TUNNEL-URL"
os.environ["TEXTLENS_EMBEDDING_WORKER_TOKEN"] = "textlens-local-colab-worker-token"
os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"
os.environ["EMBEDDING_MODEL_NAME"] = "all-MiniLM-L6-v2"
os.environ["EMBEDDING_BATCH_SIZE"] = "512"
os.environ["PINECONE_UPSERT_BATCH_SIZE"] = "500"
```

5. Upload or copy `colab/textlens_colab_embedding_worker.py` into Colab.
6. Run:

```python
!python textlens_colab_embedding_worker.py
```

## Local Workflow

1. Start backend locally.
2. Start frontend locally.
3. Start tunnel to backend port `8000`.
4. Start Colab GPU runtime and worker.
5. In TextLens_AI dashboard, select dataset and click **Embed**.
6. The backend queues the job.
7. The Colab worker claims the job and processes it.
8. The dashboard shows queued/processing/completed as it polls `/api/datasets`.

## Notes And Limitations

- Free Colab is not guaranteed infrastructure. It can disconnect or lose GPU availability.
- This is good for development and demos, not a final production worker system.
- The production version should reuse the same worker protocol on RunPod, Modal, AWS GPU EC2, SageMaker, or another managed GPU service.
- Worker progress is resumable by vector ID and stored `embedding_count`, but if Colab dies mid-batch, the next run may repeat some upserts. Pinecone upserts are idempotent for the same vector IDs.
- Do not commit real API keys or worker tokens to a shared repository.
