# TextLens AI Project Health Report

Date: 2026-05-21

## Scope

Reviewed the implemented upload, ingestion, analysis/dashboard, and started semantic QA/RAG pipeline against `implementation_plan.md`. Ran backend tests, frontend lint, and frontend production build after fixes.

## Bugs Found And Fixed

### 1. SQLite used a stale upload directory

`backend/app/core/database.py` computed `DB_PATH` once at import time. Tests and runtime overrides that changed `settings.UPLOAD_DIR` still wrote to the old database location, which could mix datasets across environments.

Fix:
- Resolve the SQLite path dynamically from the active `settings.UPLOAD_DIR`.
- Initialize the dataset schema whenever a connection is opened so isolated runtime directories work correctly.

### 2. Legacy registry hook broke upload tests

The project moved from `registry.json` to SQLite, but one compatibility hook expected by tests no longer existed.

Fix:
- Restored a harmless `REGISTRY_PATH` compatibility constant in `dataset_manager.py`.

### 3. Removed config names still used by health and QA code

The semantic refactor introduced `EMBEDDING_MODEL_NAME` and `PINECONE_UPSERT_BATCH_SIZE`, but `/api/system/status` and Ollama QA still referenced `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL`, and the old vector batch setting.

Fix:
- Added backward-compatible settings aliases.
- Let `PINECONE_UPSERT_BATCH_SIZE` honor either the new env var or the older `VECTOR_UPSERT_BATCH_SIZE`.

### 4. Semantic sentence splitting could fail at import/runtime

`semantic_dataset_service.py` tried to rely on NLTK sentence data during import. On a clean machine this can fail or trigger unexpected dependency setup.

Fix:
- Removed import-time NLTK data setup.
- Added a regex sentence splitter fallback when NLTK punkt data is unavailable.
- Added explicit errors for datasets that produce zero chunks, avoiding divide-by-zero and empty embedding artifacts.

### 5. Clean CSV download contract drifted

The Phase 2 clean download had started preserving ingestion metadata columns (`original_row_index`, `duplicate_frequency`), but the documented current contract and tests expect the downloaded analysis-clean CSV to preserve the original uploaded schema.

Fix:
- `save_clean_dataset()` now limits columns to the provided original upload schema when available.

### 6. Frontend lint failure

`Layout.jsx` caught an unused `err` variable, causing ESLint to fail.

Fix:
- Switched to a bare `catch`.

### 7. Semantic metadata test was stale

The refactored RAG metadata now includes `source`, matching the refactor report and retrieval metadata requirements.

Fix:
- Updated the test expectation to include `source`.

## Verification

All checks pass:

```text
python -m pytest backend\tests
26 passed

npm.cmd run lint
passed

npm.cmd run build
passed
```

Build note:
- Vite still warns that the main JS chunk is larger than 500 kB after minification. This is not a functional failure, but future code splitting would improve load performance.

## Remaining Risks

- Phase 3 embedding now runs in a background task, but the frontend still needs stronger polling/progress UX for long jobs.
- Pinecone and SentenceTransformer behavior were covered by unit/contract tests, not by a live external-service integration run.
- Several files still contain mojibake in comments/docs. It does not break tests, but it should be cleaned up for maintainability.
- The clean-data contract should be documented explicitly: ingestion preview includes traceability columns; analysis clean download currently preserves the original uploaded schema.

## RAG End-To-End Test Update

Date: 2026-05-21

### Scope

Tested the Phase 3 RAG flow with a disposable 4-row dataset:

1. Upload CSV.
2. Run analysis.
3. Start embedding.
4. Generate local sentence-transformer embeddings.
5. Upsert vectors to Pinecone.
6. Run semantic search.
7. Run QA through the frontend-facing `/api/qa` contract.
8. Review and fix QA page state handling.

### Issues Found And Fixed

#### 1. Embedding failed without Parquet dependencies

The RAG chunk store wrote only `.parquet` files. The active venv did not have `pyarrow` or `fastparquet`, so embedding failed before reaching Pinecone.

Fix:
- Added a JSONL chunk-store fallback when no Parquet engine is installed.
- Reading now supports both `.parquet` and `.jsonl` chunk files.

#### 2. RAG intermediate storage wrote to OneDrive-backed `data/`

`DATA_CHUNKS_DIR`, `DATA_EMBEDDINGS_DIR`, and related RAG paths wrote into the OneDrive workspace and hit permission errors.

Fix:
- Reused the existing runtime-directory fallback pattern for all RAG storage paths.
- RAG intermediates now use `backend/.runtime/data/...` unless explicitly overridden by env vars.

#### 3. Pinecone index dimension mismatch blocked embedding

The configured `textlens-ai` Pinecone index already existed with dimension `1024`, while `all-MiniLM-L6-v2` produces `384`-dimensional embeddings. Upsert failed with a dimension mismatch.

Fix:
- `PineconeVectorStore.ensure_index()` now detects incompatible existing index dimensions.
- It automatically selects/creates a dimension-specific index, e.g. `textlens-ai-384`.
- Dataset metadata records the actual index used.
- Search and QA now query `meta.embedding_index_name`, not only the base configured index.

#### 4. Repeated model and Pinecone setup added latency

Search and QA created fresh embedding/vector clients, causing repeated model and Pinecone SDK setup.

Fix:
- Added in-process SentenceTransformer model caching.
- Added in-process Pinecone client/index caching.
- Model loading now tries local cached files first, then falls back to remote lookup only when needed.

#### 5. QA waited too long on slow/unavailable Ollama

The retrieval step was healthy, but `/api/qa` tried Ollama first because `LLM_PROVIDER=ollama`; fallback only happened after a long timeout.

Fix:
- Added `LLM_ENABLED`, defaulting to `false`.
- QA now uses fast deterministic grounded fallback unless `LLM_ENABLED=true` is explicitly set.
- Lowered default `LLM_TIMEOUT_SECONDS` from `20` to `5` for opt-in LLM mode.

#### 6. QA page progress could display stale embedding state

`QAPage.jsx` preferred the immediate `/api/embed` response over refreshed dataset metadata. That could pin progress at `0%` and show the base index instead of the final dimension-specific index.

Fix:
- QA page derived status/dimension/model/index/progress now prefer refreshed dataset metadata.
- The operation log now describes embedding as an accepted background job rather than completed work.

### Live Smoke Result

The RAG flow completed successfully after fixes.

```text
upload: 200, ~49 ms
analyze: 200, ~63 ms
embed start/background completion under TestClient: completed, 4 vectors
actual Pinecone index used: textlens-ai-384
search: 200, 3 results
qa: 200, fallback mode, 3 supporting rows
```

Final measured warm QA request after model/index caching:

```text
search: 200, ~5.4 s on a fresh process with model load + Pinecone setup
qa immediately after search: 200, ~0.43 s
```

Cold-start note:
- First ever model download took about 268 s because Hugging Face model files had to be downloaded.
- After the model was cached locally, model load dropped to about 0.5 s in a fresh process and query embedding dropped to milliseconds within the same process.

Pinecone note:
- Creating the new `textlens-ai-384` index added one-time latency during the first successful embedding.
- Subsequent datasets using the same model/dimension reuse that index.

### Verification After Fixes

```text
.\venv\Scripts\python.exe -m pytest backend\tests
26 passed

npm.cmd run lint
passed

npm.cmd run build
passed
```

Build note:
- Vite still warns that the main JS chunk is larger than 500 kB after minification. This does not break RAG, but route-level code splitting would improve frontend load time.

### Remaining RAG Notes

- For production, pre-cache the embedding model during setup/deploy so the first user does not pay the model download cost.
- The checked-in/local `.env` contains a Pinecone API key. Rotate it if this repo has been shared or committed anywhere outside your machine.
- The QA page was validated through API contracts, lint, and production build. A browser automation pass with Playwright would be useful for visual/polling-state QA once Playwright is added.

## RAG Performance Optimization Update

Date: 2026-05-21

### Bottlenecks Found

The Phase 3 embedding pipeline had several avoidable costs that could explain long embedding runs on larger datasets:

- Every text row went through sentence tokenization, even short comments/reviews that should become one chunk.
- Chunk generation accumulated all chunks in memory before writing.
- Embedding generated all vectors before Pinecone upload, causing large memory use and delaying network upload until the end.
- The previous checkpoint approach saved a growing `.npy` partial embedding file every batch, creating O(n²)-style disk churn on large jobs.
- Pinecone upload built one large vector list in memory.
- Re-analysis could leave stale chunk/embedding artifacts that were reused later.
- Search/QA loaded dataset metadata through a function that read the full clean CSV even when the dataframe was ignored.

### Optimizations Applied

- Added smart chunk fast path:
  - Short texts between `MIN_CHUNK_WORDS` and `MAX_CHUNK_WORDS` now become one chunk without sentence tokenization.
  - Long text still uses sentence-aware splitting and word-window fallback.
- Added configurable chunk controls:
  - `MAX_CHUNK_WORDS`
  - `CSV_READ_CHUNK_SIZE`
  - `EMBEDDING_CHECKPOINT_EVERY_BATCHES`
  - `EMBEDDING_SAVE_VECTORS`
- Changed new chunk artifacts to streaming JSONL writes.
  - Existing Parquet chunk files remain readable for compatibility.
- Changed embedding flow to embed and upsert each batch immediately.
  - No full vector set is held in memory.
  - No growing partial `.npy` file is rewritten each batch.
  - Optional vector persistence is off by default via `EMBEDDING_SAVE_VECTORS=false`.
- Added artifact manifests tied to:
  - clean CSV path, size, mtime
  - embedding model
  - chunking settings
  This prevents stale chunks from being reused after re-analysis or config changes.
- Fixed completed-embedding detection for dimension-specific Pinecone indexes.
- Search/QA metadata loading no longer reads the full clean CSV.

### Verification

Automated checks after optimization:

```text
.\venv\Scripts\python.exe -m pytest backend\tests
26 passed

npm.cmd run lint
passed

npm.cmd run build
passed
```

Local streaming smoke with fake embedding/vector services:

```text
embedding_status: completed
embedding_count: 3
embedding_dimension: 4
upsert batches: 2
elapsed: ~166 ms
```

Live Pinecone smoke after optimization:

```text
embedding_status: completed
embedding_count: 4
index: textlens-ai-384
search_status: 200
search_results: 3
```

Observed live timings on the tiny smoke dataset:

```text
chunk write: ~10 ms
embedding generation for 4 chunks: ~611 ms
Pinecone index validation + upsert dominated total runtime
search query after embedding: ~456 ms Pinecone query time
```

### Expected Impact

For large datasets, the main improvement is not changing the embedding model itself; it is removing avoidable work around it:

- Fewer chunker CPU cycles for short text-heavy datasets.
- Bounded memory during chunk creation.
- Bounded memory during embedding.
- Upsert overlaps the job progression instead of waiting for all embeddings to finish first.
- Checkpoint writes are tiny JSON files instead of repeated full vector dumps.
- Failed/retried jobs avoid stale artifact reuse.

### Remaining Performance Considerations

- If a dataset truly has hundreds of thousands of text chunks, CPU embedding with `all-MiniLM-L6-v2` can still be slow. The next production option is a dedicated embedding worker or GPU-backed embedding service.
- Pinecone upsert/query latency depends on network and index readiness. The current code batches and retries, but queue-backed jobs would be safer for very large uploads.
- Browser polling still reads dataset metadata; a dedicated embedding job/status endpoint would be cleaner for production.
