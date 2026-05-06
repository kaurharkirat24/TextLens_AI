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

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    VECTOR_UPSERT_BATCH_SIZE: int = int(os.getenv("VECTOR_UPSERT_BATCH_SIZE", "100"))

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "phi3:latest")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))


settings = Settings()
