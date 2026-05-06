

### 4. Dataset Registry Is A JSON File With No Locking

Dataset metadata is stored in `settings.UPLOAD_DIR/registry.json`.

Risk:

- Concurrent uploads or analysis requests can overwrite each other.
- A partial write can corrupt the whole registry.
- Multiple backend workers will race on the same file.

Likely breakpoints:

- Two browser tabs upload at the same time.
- Uvicorn/Gunicorn runs multiple workers.
- Process crashes while writing `registry.json`.

Recommended fix:

- Use SQLite for local MVP metadata.
- At minimum, use atomic writes and file locks.

### 5. Upload Filenames Are Used In Disk Paths Without Sanitization

`upload_file` builds:

```python
upload_path = os.path.join(settings.UPLOAD_DIR, f"{dataset_meta.id}_{file.filename}")
```

Risk:

- Unexpected path separators or special characters in filenames can cause bad paths.
- Windows-reserved names or very long filenames can fail.
- User-controlled filenames leak into output paths and download names.

Recommended fix:

- Sanitize uploaded filenames with a strict basename function.
- Store original display filename separately from safe disk filename.

### 6. Ingestion And Analysis Both Clean Data Differently

Phase 1 ingestion creates a clean CSV with `original_row_index` and `duplicate_frequency`. Phase 2 analysis loads that clean CSV, cleans again, and writes another clean CSV with only original upload columns preserved.

Risk:

- Row identity can change between upload preview, analysis, embedding, and QA.
- `original_row_index` and `duplicate_frequency` can disappear after analysis.
- Duplicate handling is inconsistent: ingestion preserves duplicate rows with frequency; analysis may drop duplicate rows.

Likely breakpoints:

- User previews clean data after upload and sees different row shape after analysis.
- Semantic metadata `row_id` points to the analysis-cleaned row position, not the original CSV row.
- Duplicate-heavy datasets produce analytics/semantic results that do not match ingestion stats.

Recommended fix:

- Define one canonical cleaned dataset contract.
- Preserve `original_row_index` through analysis and embedding.
- Decide whether duplicates are preserved, deduplicated, or frequency-weighted, and apply that consistently.

### 7. Semantic Embedding Requires Analysis Metadata But Does Not Force Analysis

The Q&A page lets users click Embed after upload. The semantic loader can work with ingestion metadata, but richer role information only exists after analysis.

Risk:

- Embedding can run before analysis, then use weaker or different primary text resolution.
- Later analysis can change `clean_csv_path`, reset embedding status, and invalidate old vectors.

Likely breakpoints:

- User uploads, embeds, then analyzes. Existing Pinecone vectors may remain but app marks embeddings as `not_started`.
- User analyzes, then embeds. Results are based on a different clean CSV than upload preview.

Recommended fix:

- Make analysis an explicit prerequisite for embedding.
- Hide or disable Embed until dataset status is `analyzed`.
- Store a content hash for the clean CSV and embedding contract.

### 8. Pinecone Index Dimension Contract Is Global

The app uses a single Pinecone index and `dataset_id` namespaces. Dimension is validated against the existing index.

Risk:

- Changing the Ollama embedding model can break all future embeddings if dimensions differ.
- One existing Pinecone index dimension controls every dataset.
- The error is surfaced at runtime during embedding/search.

Recommended fix:

- Include embedding model/dimension in index naming, or enforce one model permanently.
- Add a startup/config validation endpoint that checks Pinecone dimension before users start embedding.

### 9. Ollama API Endpoint Difference Between App And `ro.py`

The app uses `/api/embed` with:

```json
{"model": "...", "input": [...]}
```

`ro.py` uses `/api/embeddings` with:

```json
{"model": "...", "prompt": "..."}
```

Risk:

- Manual connection test may pass while production code fails, or the reverse.
- Different endpoints may return different response shapes (`embedding` vs `embeddings`).

Recommended fix:

- Keep one connection test that uses the exact same `OllamaEmbeddingService` class as the app.

### 10. Frontend Assumes API Connectivity Without Checking Health

The sidebar always shows `API Connected`, but no health check drives that status.

Risk:

- User sees "API Connected" while backend is offline.
- Errors only appear after a page action fails.

Recommended fix:

- Call `/api/health` on app load.
- Show disconnected/retry state when health check fails.

## Medium-Risk Issues

### 11. CORS Is Hardcoded To Local Development Origins

`settings.CORS_ORIGINS` includes local Vite/dev origins only.

Risk:

- Deployed frontend cannot call deployed backend without code changes.

Recommended fix:

- Load CORS origins from `.env`.
- Parse comma-separated values into a list.

### 12. Large CSV Handling Is Memory-Heavy

Both ingestion and analysis load full CSV files into pandas memory.

Risk:

- Large datasets can crash the backend process.
- Frontend timeout is five minutes, but backend memory may fail earlier.

Recommended fix:

- Enforce upload file size and row count limits.
- Add chunked ingestion for large files.
- Move long analysis/embedding tasks to background jobs.

### 13. Long-Running API Calls Are Synchronous

Upload, analysis, embedding, search, and QA are synchronous request/response flows.

Risk:

- Browser requests can timeout.
- Backend workers are blocked.
- User cannot reliably resume after refresh.

Recommended fix:

- Add job records for analysis and embedding.
- Return job IDs and poll progress.

### 14. External Service Failures Are Mostly Runtime-Only

Missing Pinecone key, wrong region, Ollama down, missing model, or dimension mismatch are detected only when a user triggers embedding/search/QA.

Risk:

- App appears functional until late in the workflow.
- Errors are harder to diagnose for users.

Recommended fix:

- Add `/api/system/status` that checks:
  - upload/output directory writeability
  - Pinecone config presence
  - Pinecone index/dimension
  - Ollama availability
  - embedding model availability
  - LLM model availability

### 15. Embedding Is Batched But Still Runs As One Long Request

The semantic embedding flow batches Ollama embedding requests and Pinecone upserts, but it still loads the full cleaned CSV into memory and performs the whole embedding job inside one API request.

Risk:

- Large datasets can timeout in the browser or API server.
- The backend process can run out of memory before batching starts.
- Ollama or Pinecone can fail midway after some vectors have already been written.
- Dataset metadata may show `failed` while Pinecone contains partial vectors.
- Retrying can be slow and may duplicate work unless existing IDs are fetched successfully.

Likely breakpoints:

- Uploading and embedding a CSV with tens or hundreds of thousands of text rows.
- Running embedding while Ollama is slow or the selected embedding model is large.
- Network interruption during Pinecone upsert.
- Frontend refresh or request timeout while the backend continues or crashes.

Recommended fix:

- Convert embedding into a background job.
- Read the clean CSV in chunks instead of loading all rows at once.
- Persist progress after every batch.
- Track total rows, completed rows, failed rows, current batch, and error details.
- Let the frontend poll job status instead of waiting for one long request.
- Store a clean dataset hash so stale/partial embeddings can be detected safely.

### 16. QA LLM Provider Is Named Generic But Only Supports Ollama

`LLM_PROVIDER` exists, but `QAService` only supports `ollama`.

Risk:

- Setting another provider silently falls back.
- Environment config suggests broader provider support than exists.

Recommended fix:

- Rename config to `ollama`-specific fields, or implement provider adapters.
- Surface unsupported provider as a startup warning.

### 17. Pydantic Models Use Mutable Defaults

Several schemas use defaults like:

```python
candidates: list[str] = []
datasets: list[DatasetMeta] = []
issues: list[ValidationIssueSchema] = []
```

Pydantic v2 usually handles this safely, but it is still a risky pattern and can confuse future changes.

Recommended fix:

- Use `Field(default_factory=list)`.

### 18. Encoding/Mojibake Exists In Source And UI Text

Several files contain corrupted dash, ellipsis, and emoji sequences.

Risk:

- UI text looks broken.
- Emoji normalization maps likely do not match real emoji input.
- Comments and docs are harder to maintain.

Recommended fix:

- Normalize files to UTF-8.
- Replace mojibake text with plain ASCII or correct Unicode.
- For emoji handling, use a tested emoji library instead of a corrupted manual map.

### 19. Clean Dataset Download Depends On Analysis

The frontend lets the user download clean data from Dashboard, but this endpoint requires the analysis-cleaned dataset.

Risk:

- User has an ingested clean CSV, but download fails until analysis runs.
- Product terminology "clean data" is ambiguous across ingestion and analysis.

Recommended fix:

- Expose separate downloads:
  - ingestion-clean CSV
  - analysis-clean CSV
- Label them clearly.

### 20. LocalStorage Can Point To Deleted Or Failed Datasets

Frontend stores `textlens:selectedDatasetId` and `textlens:lastUploadResult`.

Risk:

- Cached IDs can refer to files deleted from disk.
- Upload page can show a stale result even if backend registry is cleared.

Recommended fix:

- Validate cached dataset IDs against `/api/datasets`.
- Clear stale cache when dataset is missing.

### 21. Analysis Cleans From Original Columns But Reads Ingestion-Clean CSV

`analysis.py` passes the ingestion clean CSV as `csv_path`, but computes `clean_columns` from the original upload. Then `save_clean_dataset` preserves only original columns that still exist.

Risk:

- Ingestion-added columns are dropped.
- If original and clean files diverge, analysis output may lose traceability fields.

Recommended fix:

- Decide whether Phase 2 should analyze original upload or Phase 1 clean output.
- If it analyzes Phase 1 output, preserve Phase 1 metadata columns intentionally.

## Lower-Risk Issues And Maintainability Notes

### 22. Compatibility Wrapper Modules Add Indirection

Files such as `analysis_pipeline.py`, `data_cleaner.py`, `data_enrichment.py`, and `insight_generator.py` mostly re-export newer modules.

Risk:

- New contributors may edit the wrapper instead of the real implementation.

Recommended fix:

- Keep wrappers only if needed for imports/tests.
- Add deprecation comments or remove them after import paths are migrated.

### 23. Semantic Search Has Both Dataset Path And Contract-First Routes

Routes exist for both:

- `POST /api/datasets/{dataset_id}/search`
- `POST /api/search`

Risk:

- Contracts can drift over time.

Recommended fix:

- Keep one public route style.
- If both remain, test both.

### 24. Upload Size Is Mentioned In UI But Not Enforced In Backend

The upload page says CSV files up to 50MB, but the backend does not enforce that limit.

Risk:

- Users can upload larger files and crash or slow the service.

Recommended fix:

- Enforce file size on backend.
- Return a clear 413-style error.

### 25. Dataset Status And Embedding Status Are Separate But Not Modeled As A State Machine

Dataset status can be `uploaded`, `ingested`, `analyzed`, `failed`, while embedding status can be `not_started`, `completed`, or `failed`.

Risk:

- Invalid state combinations can occur.
- UI may enable actions in the wrong order.

Recommended fix:

- Define valid transitions.
- Centralize status updates in service functions.

## Suggested Fix Order

1. Remove and rotate the hardcoded Pinecone key in `ro.py`.
2. Install dependencies and run backend tests.
3. Add a repeatable setup/check command for contributors.
4. Add a system status endpoint for Ollama, Pinecone, and runtime directories.
5. Make analysis a prerequisite for embedding.
6. Replace JSON registry with SQLite or add file locking and atomic writes.
7. Unify the clean dataset contract across ingestion, analysis, and semantic indexing.
8. Add filename sanitization and backend upload size limits.
9. Fix mojibake/encoding issues.
10. Add frontend health status instead of static "API Connected".
