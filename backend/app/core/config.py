"""
Application-level configuration.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_ROOT = Path(__file__).resolve().parents[2]

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")
logger = logging.getLogger(__name__)


def _resolve_runtime_dir(env_key: str, preferred: Path, fallback_leaf: str) -> str:
    """Use env override when present; avoid OneDrive-backed runtime writes."""
    configured = os.getenv(env_key)
    if configured:
        return configured

    if "OneDrive" not in preferred.as_posix():
        return str(preferred)

    fallback = _BACKEND_ROOT / ".runtime" / fallback_leaf
    fallback.mkdir(parents=True, exist_ok=True)
    logger.warning("Using fallback runtime directory for %s: %s", env_key, fallback)
    return str(fallback)


class Settings:
    """Central configuration for the FastAPI application."""

    UPLOAD_DIR: str = _resolve_runtime_dir("UPLOAD_DIR", _PROJECT_ROOT / "uploads", "uploads")
    OUTPUT_DIR: str = _resolve_runtime_dir("OUTPUT_DIR", _PROJECT_ROOT / "output", "output")
    DATA_DIR: str = str(_PROJECT_ROOT / "data")

    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]

    # Log the origins on startup to help debug CORS issues
    if not os.getenv("PYTEST_CURRENT_TEST"):
        print(f"INFO:  CORS Origins: {CORS_ORIGINS}")

    MAX_NULL_RATIO: float = 0.30
    MIN_TEXT_LENGTH: int = 3
    MAX_TEXT_LENGTH: int = 10_000
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "textlens-ai")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_UPSERT_BATCH_SIZE: int = int(
        os.getenv("PINECONE_UPSERT_BATCH_SIZE", os.getenv("VECTOR_UPSERT_BATCH_SIZE", "250"))
    )
    PINECONE_MIN_UPSERT_BATCH_SIZE: int = int(os.getenv("PINECONE_MIN_UPSERT_BATCH_SIZE", "50"))
    PINECONE_UPSERT_TIMEOUT_SECONDS: int = int(os.getenv("PINECONE_UPSERT_TIMEOUT_SECONDS", "120"))
    PINECONE_UPSERT_MAX_RETRIES: int = int(os.getenv("PINECONE_UPSERT_MAX_RETRIES", "5"))
    VECTOR_UPSERT_BATCH_SIZE: int = PINECONE_UPSERT_BATCH_SIZE

    # Embedding settings
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "sentence_transformer").strip().lower()
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", EMBEDDING_MODEL_NAME)
    
    # Chunking settings
    CHUNK_SIZE_SENTENCES: int = int(os.getenv("CHUNK_SIZE_SENTENCES", "8"))
    CHUNK_OVERLAP_SENTENCES: int = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "1"))
    MIN_CHUNK_WORDS: int = int(os.getenv("MIN_CHUNK_WORDS", "5"))
    MAX_CHUNK_WORDS: int = int(os.getenv("MAX_CHUNK_WORDS", "180"))
    CSV_READ_CHUNK_SIZE: int = int(os.getenv("CSV_READ_CHUNK_SIZE", "5000"))
    EMBEDDING_CHECKPOINT_EVERY_BATCHES: int = int(os.getenv("EMBEDDING_CHECKPOINT_EVERY_BATCHES", "10"))
    EMBEDDING_WORKERS: int = int(os.getenv("EMBEDDING_WORKERS", "1"))
    EMBEDDING_SAVE_VECTORS: bool = os.getenv("EMBEDDING_SAVE_VECTORS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    EMBEDDING_EXECUTION_MODE: str = os.getenv("EMBEDDING_EXECUTION_MODE", "local").strip().lower()
    EMBEDDING_WORKER_TOKEN: str = os.getenv("EMBEDDING_WORKER_TOKEN", "")
    RAG_MIN_RELEVANCE_SCORE: float = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0"))

    # Structured Storage Paths
    DATA_RAW_DIR: str = _resolve_runtime_dir("DATA_RAW_DIR", _PROJECT_ROOT / "data" / "raw", "data/raw")
    DATA_CLEANED_DIR: str = _resolve_runtime_dir("DATA_CLEANED_DIR", _PROJECT_ROOT / "data" / "cleaned", "data/cleaned")
    DATA_CHUNKS_DIR: str = _resolve_runtime_dir("DATA_CHUNKS_DIR", _PROJECT_ROOT / "data" / "chunks", "data/chunks")
    DATA_EMBEDDINGS_DIR: str = _resolve_runtime_dir(
        "DATA_EMBEDDINGS_DIR", _PROJECT_ROOT / "data" / "embeddings", "data/embeddings"
    )
    DATA_TEMP_DIR: str = _resolve_runtime_dir("DATA_TEMP_DIR", _PROJECT_ROOT / "data" / "temp", "data/temp")
    LOGS_DIR: str = _resolve_runtime_dir("LOGS_DIR", _PROJECT_ROOT / "logs", "logs")

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    LLM_ENABLED: bool = os.getenv("LLM_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "5"))

    ADAPTIVE_TOP_K_ENABLED: bool = os.getenv("ADAPTIVE_TOP_K_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    ADAPTIVE_TOP_K_MIN: int = int(os.getenv("ADAPTIVE_TOP_K_MIN", "3"))
    ADAPTIVE_TOP_K_MAX: int = int(os.getenv("ADAPTIVE_TOP_K_MAX", "10"))
    SELF_RAG_ENABLED: bool = os.getenv("SELF_RAG_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    SELF_RAG_CONFIDENCE_THRESHOLD: float = float(os.getenv("SELF_RAG_CONFIDENCE_THRESHOLD", "0.65"))
    SELF_RAG_MAX_RETRIES: int = int(os.getenv("SELF_RAG_MAX_RETRIES", "1"))


settings = Settings()
