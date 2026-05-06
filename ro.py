import requests
from pinecone import Pinecone

# =========================
# CONFIG
# =========================

PINECONE_API_KEY = "pcsk_JoH5v_NetMMNbQPwUyNLarWNLiSobp3fAEvGHA1AwyUjnZPdNZAM61f6xEPbKczFuzbaz"
INDEX_NAME = "lolo"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "qwen3-embedding:0.6b"

TEXT = "This is a test document from Ollama to Pinecone."


# =========================
# GENERATE EMBEDDING
# =========================

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL_NAME,
        "prompt": TEXT
    }
)

data = response.json()

embedding = data["embedding"]

print(f"Embedding generated")
print(f"Vector dimension: {len(embedding)}")


# =========================
# CONNECT TO PINECONE
# =========================

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

print("Connected to Pinecone")


# =========================
# UPSERT VECTOR
# =========================

index.upsert(
    vectors=[
        {
            "id": "doc1",
            "values": embedding,
            "metadata": {
                "text": TEXT
            }
        }
    ]
)

print("Vector inserted into Pinecone")


# =========================
# QUERY VECTOR
# =========================

results = index.query(
    vector=embedding,
    top_k=1,
    include_metadata=True
)

print("\nQuery Results:")
print(results)