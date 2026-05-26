# Adaptive Top-K + Self-RAG + HyDE + Semantic Cache (2026-05-27)

## Summary

This update adds four QA retrieval improvements:

1. Adaptive `top_k`: retrieval window size is adjusted based on query intent and complexity.
2. Self-RAG confidence gate: after generating an answer, the system scores confidence and may run one rewritten second-pass retrieval when confidence is low.
3. HyDE query expansion: short/vague questions can be expanded into a hypothetical answer passage before embedding.
4. Semantic query cache: query embeddings are matched against recent queries to skip repeated Pinecone/LLM work.

Both features are controlled by environment flags.

## What Changed

### 1) Adaptive Top-K

- Added query complexity scoring (`simple`, `medium`, `complex`) in:
  - `backend/app/services/query_complexity_service.py`
- Routed effective retrieval size in:
  - `backend/app/services/query_router.py`
- Planner now stores complexity in:
  - `backend/app/services/retrieval_planner.py`

Behavior:

- Simple factual queries tend to use smaller windows (`~3-5`).
- Trend/comparison/summary and complex queries use larger windows (`~8-10`).
- Safety bounds remain enforced (`1..10`).
- Response `retrieval_plan` now includes:
  - `requested_top_k`
  - `effective_top_k`
  - `query_complexity`
  - `query_complexity_score`
  - `query_complexity_rationale`

### 2) Self-RAG Confidence Gate

- Confidence scoring and query rewrite support added in:
  - `backend/app/services/qa_service.py`
- Retry orchestration added in:
- `backend/app/services/rag_service.py`

Behavior:

1. First answer is generated as usual.
2. Confidence is computed (LLM-based when available; heuristic fallback otherwise).
3. If confidence is below threshold, semantic retrieval runs again once with:
   - rewritten query
   - expanded `top_k` (`+3`, capped at `10`)
4. The higher-confidence answer is returned.

Response `retrieval_plan.self_rag` now includes:

- `enabled`
- `passes`
- `rewritten_query`
- `expanded_top_k` (when second pass is used)
- `initial_confidence`
- `final_confidence`

### 3) HyDE Query Expansion

- Added in:
  - `backend/app/services/hyde_service.py`
- Wired into semantic embedding path in:
  - `backend/app/services/semantic_retriever.py`

Behavior:

- For short/vague questions, the system asks the configured LLM for a short hypothetical answer passage.
- That hypothetical passage is embedded instead of the raw query.
- HyDE is gated and not applied for clear aggregate/trend/count wording.

### 4) Semantic Query Cache

- Added in:
  - `backend/app/services/semantic_cache.py`
- Used by:
  - `backend/app/services/rag_service.py` (`search` and `qa`)

Behavior:

- Incoming query is embedded once.
- Cache searches same-dataset prior queries using cosine similarity.
- On hit:
  - `search` can skip Pinecone query.
  - `qa` can skip full routing/retrieval/generation and return cached answer.
- `retrieval_plan.semantic_cache.hit` indicates cache behavior.

## New Config Flags

In `backend/app/core/config.py`:

- `ADAPTIVE_TOP_K_ENABLED` (default: `true`)
- `ADAPTIVE_TOP_K_MIN` (default: `3`)
- `ADAPTIVE_TOP_K_MAX` (default: `10`)
- `SELF_RAG_ENABLED` (default: `true`)
- `SELF_RAG_CONFIDENCE_THRESHOLD` (default: `0.65`)
- `SELF_RAG_MAX_RETRIES` (default: `1`)
- `HYDE_ENABLED` (default: `true`)
- `HYDE_MAX_QUERY_TOKENS` (default: `8`)
- `HYDE_PROMPT_MAX_CHARS` (default: `500`)
- `SEMANTIC_CACHE_ENABLED` (default: `true`)
- `SEMANTIC_CACHE_MAX_ENTRIES` (default: `512`)
- `SEMANTIC_CACHE_TTL_SECONDS` (default: `900`)
- `SEMANTIC_CACHE_SIMILARITY_THRESHOLD` (default: `0.94`)

## API Schema Update

`QAResponse` now optionally includes:

- `confidence`
- `confidence_rationale`

File:
- `backend/app/models/schemas.py`

## Notes

- Structured-query path is still preferred when exact dataframe answers are possible.
- Out-of-scope guardrails are unchanged.
- Self-RAG retry currently applies to semantic-capable plans only.
- Semantic cache is in-memory (process-local) and resets on service restart.
