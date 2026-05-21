# Google Colab SentenceTransformer Embedding Workflow

Date: 2026-05-21

## Target Architecture

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Pinecone index: `textlens-ai-384`
- Pinecone namespace: the local TextLens `dataset_id`
- Query embeddings: generated locally by the backend with the same SentenceTransformer model
- Answer generation: Gemini, grounded only on rows retrieved from Pinecone

This avoids Gemini embedding API rate/batch limits while keeping answer generation fast and grounded.

## Local App Configuration

Keep these values in the project root `.env`:

```env
EMBEDDING_PROVIDER=sentence_transformer
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=128

LLM_ENABLED=true
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your_gemini_api_key_here
```

`GEMINI_API_KEY` is used for answer generation only. Gemini embeddings are not used in this workflow.

## Colab Steps

1. Upload and analyze the CSV in TextLens locally.
2. Copy the dataset ID from the app.
3. Find the cleaned CSV path from the dataset metadata or backend output folder.
4. Open a Google Colab notebook.
5. Runtime -> Change runtime type -> select GPU.
6. Upload the cleaned CSV to Colab.
7. Run the notebook code below.
8. Back on your local machine, call the external-complete endpoint so TextLens knows the vectors are ready.

## Colab Notebook Code

```python
!pip install -q sentence-transformers pinecone pandas tqdm

import os
import re
import time
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

DATASET_ID = "paste_dataset_id_here"
CLEAN_CSV_PATH = "/content/your_clean_file.csv"
PRIMARY_TEXT_COLUMN = "paste_primary_text_column_here"

PINECONE_API_KEY = "paste_pinecone_api_key_here"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
BASE_INDEX_NAME = "textlens-ai"
MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSION = 384
INDEX_NAME = f"{BASE_INDEX_NAME}-{DIMENSION}"

CHUNK_WORD_LIMIT = 180
MIN_CHUNK_WORDS = 5
BATCH_SIZE = 512
UPSERT_BATCH_SIZE = 100

def chunk_text(text):
    text = str(text or "").strip()
    words = text.split()
    if len(words) < MIN_CHUNK_WORDS:
        return []
    if len(words) <= CHUNK_WORD_LIMIT:
        return [text]

    chunks = []
    step = CHUNK_WORD_LIMIT - 30
    for start in range(0, len(words), step):
        chunk_words = words[start:start + CHUNK_WORD_LIMIT]
        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk_words))
        if start + CHUNK_WORD_LIMIT >= len(words):
            break
    return chunks

def clean_scalar(value):
    if pd.isna(value):
        return ""
    return str(value)

pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = set(pc.list_indexes().names())
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        deletion_protection="disabled",
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)

index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME, device="cuda")
df = pd.read_csv(CLEAN_CSV_PATH)

texts = []
metadatas = []
ids = []

for row_id, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
    for chunk_id, text in enumerate(chunk_text(row.get(PRIMARY_TEXT_COLUMN, ""))):
        ids.append(f"{DATASET_ID}_{row_id}_{chunk_id}")
        texts.append(text)
        metadatas.append({
            "dataset_id": DATASET_ID,
            "row_id": int(row_id),
            "chunk_id": int(chunk_id),
            "text": text,
            "source": DATASET_ID,
            "sentiment": clean_scalar(row.get("Sentiment", row.get("sentiment", ""))),
            "engagement": float(row.get("Likes", row.get("likes", 0)) or 0) + float(row.get("Replies", row.get("replies", 0)) or 0),
            "timestamp": clean_scalar(row.get("PublishedAt", row.get("published_at", row.get("timestamp", "")))),
        })

print(f"Prepared {len(texts)} chunks")
if not texts:
    raise ValueError("No chunks found. Check PRIMARY_TEXT_COLUMN.")

embedded_count = 0
for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding + upserting"):
    batch_texts = texts[start:start + BATCH_SIZE]
    batch_ids = ids[start:start + BATCH_SIZE]
    batch_meta = metadatas[start:start + BATCH_SIZE]

    vectors = model.encode(
        batch_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).tolist()

    payload = [
        {"id": vector_id, "values": values, "metadata": metadata}
        for vector_id, values, metadata in zip(batch_ids, vectors, batch_meta)
    ]

    for upsert_start in range(0, len(payload), UPSERT_BATCH_SIZE):
        index.upsert(
            vectors=payload[upsert_start:upsert_start + UPSERT_BATCH_SIZE],
            namespace=DATASET_ID,
        )

    embedded_count += len(batch_texts)

print({
    "dataset_id": DATASET_ID,
    "model": MODEL_NAME,
    "dimension": DIMENSION,
    "count": embedded_count,
    "index_name": INDEX_NAME,
    "namespace": DATASET_ID,
})
```

## Mark The Dataset Ready Locally

After Colab finishes, keep your local backend running and call:

```powershell
$datasetId = "paste_dataset_id_here"
$body = @{
  model = "all-MiniLM-L6-v2"
  dimension = 384
  count = 12345
  index_name = "textlens-ai-384"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/datasets/$datasetId/embeddings/external-complete" `
  -ContentType "application/json" `
  -Body $body
```

Replace `count` with the `embedded_count` printed by Colab.

## Important Notes

- Do not use `textlens-ai-768` for this workflow. That index is for 768-dimensional Gemini embeddings.
- Do not mix dimensions in one Pinecone index.
- Search and QA will use local `all-MiniLM-L6-v2` query embeddings, so Colab and local backend must use the same model.
- Gemini sees only the retrieved rows, not the full uploaded dataset.
