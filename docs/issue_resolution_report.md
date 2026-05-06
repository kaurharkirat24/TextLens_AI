# TextLens AI: Technical Debt & Security Resolution Report

This document summarizes the changes made to resolve identified issues, improve system security, and harden the architecture of the TextLens AI platform.

## Summary of Changes

### 1. Security Hardening & Configuration

- **Dynamic Pinecone Configuration:** Removed hardcoded Pinecone API keys and index names from `ro.py`. The script now dynamically loads credentials from the `.env` file.
- **Dynamic CORS Origins:** Updated `backend/app/core/config.py` to parse `CORS_ORIGINS` from environment variables, allowing for secure cross-origin configuration in different environments.
- **Filename Sanitization:** Implemented a strict `sanitize_filename` helper in `backend/app/routers/ingestion.py` to prevent path traversal and ensure consistent file naming.
- **Upload Size Enforcement:** Added backend enforcement for a 50MB file size limit (configurable via `MAX_UPLOAD_SIZE_MB`), preventing potential disk exhaustion.

### 2. Architectural Improvements (Registry & Persistence)

- **SQLite Database Integration:** Replaced the fragile `registry.json` file with a robust SQLite database. This resolves concurrency issues, prevents data corruption during simultaneous writes, and provides a scalable foundation for future features.
- **Pydantic Schema Fixes:** Resolved a common Pydantic pitfall by replacing mutable default lists (`[]`) with `Field(default_factory=list)`. This prevents shared state bugs between different instances of the same model.
- **Automatic Database Initialization:** Added a startup hook in `backend/app/main.py` to ensure the SQLite schema is automatically created or migrated upon application start.

### 3. System Health & Observability

- **System Status API:** Introduced a new `/api/system/status` endpoint that proactively verifies:
  - Write permissions for upload and output directories.
  - Connectivity and configuration status for Pinecone.
  - Availability of Ollama and presence of required models (embedding/LLM).
- **Frontend Connectivity Indicator:** Refactored the sidebar status in the React frontend to perform real-time health checks against the backend, replacing the hardcoded "API Connected" text with accurate status feedback.

### 4. Data Pipeline & Contract Unification

- **Canonical Clean Dataset:** Refined the data processing contract to ensure that `original_row_index` and `duplicate_frequency` (added during ingestion) are preserved throughout the analysis and enrichment phases.
- **Consistent Deduplication:** Updated the analysis pipeline to perform deduplication based on content while protecting metadata columns, ensuring that traceability is maintained.
- **Embedding Prerequisite Enforcement:** Enforced that datasets must be analyzed before they can be embedded. This ensures that the vector store is populated with enriched data and correct column roles.

## Impact of Changes

| Area                      | Impact                                                                                                        |
| :------------------------ | :------------------------------------------------------------------------------------------------------------ |
| **Security**        | Eliminated credential exposure and potential file system vulnerabilities (sanitization/size limits).          |
| **Stability**       | Eliminated "silent failures" and data corruption risks caused by JSON file locking issues.                    |
| **UX**              | Improved transparency by providing real-time system status and clear feedback when prerequisites are not met. |
| **Maintainability** | Standardized Pydantic usage and unified the data contract between ingestion and analysis.                     |

## Next Steps

1. **Background Processing (Async Jobs):** Implement a background task queue (e.g., using FastAPI's `BackgroundTasks` or a simple job manager) to handle long-running embeddings and prevent browser timeouts.
2. **Dataset Deletion:** Implement the missing `/api/datasets/{id}` DELETE endpoint to allow users to clean up their workspace.
3. **Atomic File Operations:** Transition existing pandas CSV writes to a "write-then-rename" pattern for maximum reliability.
