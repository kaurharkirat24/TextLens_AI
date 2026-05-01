"""
Application-level configuration.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_ROOT = Path(__file__).resolve().parents[2]

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")


class Settings:
    """Central configuration for the FastAPI application."""

    UPLOAD_DIR: str = str(_PROJECT_ROOT / "uploads")
    OUTPUT_DIR: str = str(_PROJECT_ROOT / "output")
    DATA_DIR: str = str(_PROJECT_ROOT / "data")

    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    MAX_NULL_RATIO: float = 0.30
    MIN_TEXT_LENGTH: int = 3
    MAX_TEXT_LENGTH: int = 10_000


settings = Settings()
