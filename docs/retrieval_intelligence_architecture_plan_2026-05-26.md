# Retrieval Intelligence Architecture Plan

Date: 2026-05-26

## Purpose

TextLens needs a retrieval pipeline that works for any uploaded CSV, not only comment-like datasets. The current RAG flow has useful foundations, but it still mixes routing, dataframe lookups, semantic retrieval, analytics, and response generation inside one orchestration path. That makes it harder to tune, harder to test, and more expensive than necessary because the same dataset artifacts can be loaded multiple times during one QA request.

This plan defines the next architecture before deeper implementation.

## Current Findings

The current backend already has several good pieces:

- `semantic_dataset_service.py` builds schema-aware row text and Pinecone chunks.
- `structured_query_service.py` can answer exact dataframe questions such as lookup, filters, averages, counts, and extrema.
- `dataset_relevance_service.py` acts as an out-of-scope guardrail.
- `query_intent_service.py` and `retrieval_planner.py` classify and map query intent to semantic, analytics, or hybrid retrieval.
- `qa_service.py` and `prompt_builder.py` separate answer generation from retrieval enough to keep Gemini as the final wording layer.

The main problems are architectural rather than only algorithmic:

- One QA request can reload the clean CSV in relevance checking, structured QA, and analytics QA.
- `rag_service.py` is doing too much orchestration and owns too many decisions.
- The pipeline has no durable dataset profile artifact, so relevance and routing must infer context repeatedly.
- Aggregation and global questions still depend too much on heuristic logic and representative rows.
- The code needs an explicit evaluation suite for factual, structured, aggregate, semantic, off-topic, and ambiguous questions.

## Review Of The Proposed Workflow Images

The proposed flow is directionally right:

- Upload dataset.
- Analyze schema and column types.
- Generate a dataset profile with summary and topics.
- Store structured rows and semantic representations.
- Use the profile to check whether a question belongs to the dataset.
- Route relevant questions into retrieval.
- Use the LLM mainly to format the final grounded answer.

One adjustment is important: the LLM should not be the mandatory first step for every question. A deterministic relevance check using schema, column names, sample values, keywords, and cached dataset profile should run first. An LLM relevance check should be used only when confidence is ambiguous. This keeps the system faster, cheaper, and more reliable.

## Target Runtime Flow

```text
QA request
-> RetrievalContext loads metadata + analysis + dataframe once
-> DatasetRelevanceService checks schema/profile/value relevance
-> QueryRouter selects route
   -> structured route for exact table questions
   -> analytics route for global counts/summaries/trends
   -> semantic route for evidence lookup
   -> hybrid route for analytics plus examples
-> ResponseGenerator/QAService formats final answer from computed facts and evidence
-> API returns answer, rows, intent, strategy, and retrieval_plan
```

## Target Upload/Analysis Flow

```text
Upload CSV
-> existing ingestion and cleaning
-> schema + role detection
-> DatasetProfileService creates dataset_profile.json
   -> columns, roles, row count, sample values
   -> numeric/categorical/time summaries
   -> primary text summary, keywords, topic hints
   -> supported question types
-> semantic chunking embeds rows/chunks
-> optional profile summary embedding stored separately for relevance/search expansion
```

## Proposed Modules

### `retrieval_context.py`

What: Request-scoped object that loads dataset metadata, analysis JSON, and clean dataframe lazily.

Why: Avoid repeated disk reads and give every retrieval service the same dataset view.

Impact: Lower latency per QA request and simpler service contracts.

### `dataset_profile_service.py`

What: Builds and reads a durable `dataset_profile.json` artifact during analysis.

Why: The relevance checker and router need compact knowledge of the dataset without scanning the CSV every time.

Impact: Better out-of-scope decisions, better starter suggestions, and better prompts for any CSV shape.

### `query_router.py`

What: Central coordinator for relevance, structured route, intent classification, plan creation, analytics context, and semantic retrieval selection.

Why: `rag_service.py` should orchestrate the request, not own every decision.

Impact: Cleaner architecture, easier tests, easier strategy tuning.

### `structured_query_service.py`

What: Continue using pandas now, but keep the service contract isolated so DuckDB can be added later.

Why: Many CSV questions are table questions, not vector questions.

Impact: More exact answers for counts, filters, max/min, averages, group-by, and lookup questions.

### `semantic_retriever.py`

What: Own query embedding, embedding contract validation, Pinecone query, score thresholding, and metadata filter support.

Why: Retrieval mechanics should be separate from routing and answer generation.

Impact: Safer vector retrieval and easier addition of query expansion or reranking.

### `analytics_qa_service.py`

What: Compute aggregate facts from the dataframe and cached profile.

Why: Questions like "top topics", "sentiment distribution", and "summarize all comments" should not be answered from top-5 semantic rows.

Impact: More accurate global answers and fewer unsupported LLM claims.

### `qa_service.py` / optional `response_generator.py`

What: Keep `QAService` as the response generator for now, with `PromptBuilder` handling intent-specific prompts.

Why: Renaming immediately adds churn without improving reliability. A `response_generator.py` wrapper can be introduced later if the naming becomes confusing.

Impact: Smaller, safer refactor.

## Framework Recommendation

Do not add LlamaIndex or LangChain as a core dependency right now. The project has a custom CSV ingestion pipeline, Pinecone namespace model, pandas analytics path, and Gemini generation path already working. A framework would add abstraction weight before the architecture is stable.

Use targeted libraries only where they clearly help:

- DuckDB later for scalable SQL-style analytics over larger CSVs.
- spaCy later for noun phrases/entities if keyword/topic quality becomes a bottleneck.
- A lightweight reranker later if semantic retrieval quality is the main failure mode.

## Implementation Phases

### Phase 0: Stabilize Current Partial Refactor

What:

- Keep `DatasetRelevanceService` backward-compatible while moving to `RetrievalContext`.
- Fix method placement and syntax issues in `structured_query_service.py`.
- Leave newly added router/retriever modules unwired until tests are ready.

Why:

- The codebase should remain runnable before larger architecture work.

Impact:

- Reduces risk before the next implementation pass.

### Phase 1: Shared Retrieval Context

What:

- Wire `DatasetRAGPipeline.answer()` to create one `RetrievalContext`.
- Pass context to relevance, structured, analytics, and router services.
- Remove repeated CSV and analysis loading from those services.

Why:

- This is the simplest high-impact performance improvement.

Impact:

- One dataframe load per QA request instead of multiple loads.

### Phase 2: Dataset Profile Artifact

What:

- Add `DatasetProfileService`.
- Generate profile after analysis succeeds.
- Store profile path in dataset metadata or derive path from dataset ID.
- Include schema, roles, row counts, sample values, summary stats, keywords, and supported topics.

Why:

- Relevance and routing need stable context that is cheaper than scanning the dataframe.

Impact:

- Better relevance checking and faster routing.

### Phase 3: Query Router

What:

- Make `QueryRouter` the only component that decides the route.
- Route order:
  1. Relevance guardrail.
  2. Structured dataframe answer if exact.
  3. Intent classification and retrieval plan.
  4. Analytics, semantic, or hybrid execution.

Why:

- A central router makes the RAG pipeline observable and testable.

Impact:

- Cleaner `rag_service.py` and easier debugging through `retrieval_plan`.

### Phase 4: Analytics And Hybrid Retrieval

What:

- Expand analytics context for group counts, text keyword/topic summaries, sentiment distribution, and time buckets when available.
- For hybrid answers, use analytics for global claims and semantic rows only as examples.

Why:

- Global dataset answers should come from computed facts, not a small vector sample.

Impact:

- More accurate and honest answers.

### Phase 5: Reliability And Evaluation

What:

- Add a retrieval evaluation test suite with fixtures for generic CSVs.
- Cover off-topic questions, exact lookup, count/group-by, semantic evidence, summary, trend, and hybrid answers.
- Assert the selected intent, strategy, and whether semantic retrieval was used.

Why:

- Retrieval changes can silently regress quality without tests.

Impact:

- Safer iteration and clearer confidence before UI changes.

## Key Design Rules

- Use deterministic checks before LLM checks.
- Use dataframe/analytics for exact or global answers.
- Use semantic retrieval for evidence and fuzzy text questions.
- Use hybrid retrieval when global facts need examples.
- Keep final LLM generation grounded in computed facts and retrieved rows.
- Return `retrieval_plan` for every QA response.
- Cache durable dataset understanding at analysis time.
- Avoid adding a large RAG framework until the project has a stable internal contract.

## Expected Impact

- Performance: fewer duplicate CSV reads and fewer unnecessary LLM calls.
- Accuracy: aggregate questions answered from dataframe facts instead of top-k chunks.
- Reliability: out-of-scope questions blocked with clearer supported topics.
- Maintainability: routing logic moves out of `rag_service.py`.
- Scalability: future DuckDB/reranking/profile embeddings can be added without rewriting API contracts.
- Debuggability: every answer exposes intent, strategy, top-k, rationale, and evidence source.

## Immediate Next Step

Before deeper implementation, finish Phase 0 and Phase 1 with tests. After that, add the dataset profile artifact and wire the router. This keeps the project moving in small reliable steps while matching the architecture in the workflow diagrams.


**Implemented Stages**


Added profile-backed retrieval architecture:

* dataset_profile_service.py (line 21) builds compact dataset profiles with schema, roles, sample values, summaries, keywords, and supported topics.
* retrieval_context.py (line 20) loads metadata, analysis, dataframe, and profile once per request.
* query_router.py (line 30) centralizes relevance, structured routing, intent planning, and analytics context.
* semantic_retriever.py (line 11) owns embedding contract validation, Pinecone search, and score filtering.

* Rewired rag_service.py (line 55) so it is now mostly orchestration glue: context -> router -> optional semantic retrieval -> QA response.
* Updated analysis.py (line 29) to generate and persist **OUTPUT_DIR/{dataset_id}_profile.json** after analysis.
* Improved analytics prompts and fallback answers so global answers can include categorical distributions and numeric summaries, not just keywords/sentiment.
