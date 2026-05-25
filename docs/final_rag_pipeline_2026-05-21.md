# Final RAG Pipeline

Date: 2026-05-21

## Correct Runtime Flow

```text
User Query
-> MiniLM embedding
-> Pinecone similarity search
-> Retrieved chunks
-> Gemini grounded answer generation
```

This is intentionally not:

```text
User Query
-> Gemini embedding
-> Pinecone search with MiniLM vectors
```

The query embedding and stored chunk embeddings must live in the same vector space. For the current project that means `all-MiniLM-L6-v2`, 384 dimensions, and a Pinecone index such as `textlens-ai-384`.

## Notebook Design Adapted

The reference notebook `backend/rag_pipeline.ipynb` uses this useful structure:

- chunk documents
- embed chunks
- retrieve relevant chunks
- generate an answer from context
- optionally verify grounding/usefulness and retry

For TextLens, I adapted the structure without pulling in LangChain/LangGraph runtime dependencies:

- CSV analysis produces the primary text role.
- `SentenceSplitter` chunks dataset rows.
- `SentenceTransformerEmbeddingService` embeds chunks and user queries with MiniLM.
- `PineconeVectorStore` stores/retrieves vectors under the dataset namespace.
- `DatasetRAGPipeline` orchestrates query embedding, Pinecone retrieval, and answer generation.
- `QAService` uses Gemini only after retrieval, with a strict prompt that limits answers to retrieved rows and asks for row citations.

## Files Added Or Changed

- `backend/app/services/rag_service.py`
  - New orchestration service for the final RAG flow.
  - Enforces MiniLM query embedding before Pinecone retrieval.
  - Validates dataset embedding model and dimension before search/QA.

- `backend/app/services/embedding_service.py`
  - Gemini embedding is no longer a supported retrieval provider.
  - `get_embedding_service()` raises if `EMBEDDING_PROVIDER=gemini`.

- `backend/app/routers/semantic.py`
  - Search and QA endpoints now call `DatasetRAGPipeline`.

- `backend/app/services/qa_service.py`
  - Gemini remains available for grounded answer generation over retrieved chunks.

- `backend/app/routers/system.py`
  - System status now reports SentenceTransformer/MiniLM as the query embedding provider and says Gemini embeddings are disabled.

## Environment

```env
EMBEDDING_PROVIDER=sentence_transformer
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=128

LLM_ENABLED=true
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your_gemini_api_key_here
```
