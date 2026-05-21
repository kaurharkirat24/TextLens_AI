import os
import sqlite3

from app.core.config import settings


def _db_path() -> str:
    return os.path.join(settings.UPLOAD_DIR, "textlens.db")


DB_PATH = _db_path()


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                status TEXT NOT NULL,
                text_column TEXT,
                total_rows INTEGER DEFAULT 0,
                clean_rows INTEGER DEFAULT 0,
                file_path TEXT,
                clean_csv_path TEXT,
                report_json_path TEXT,
                analysis_path TEXT,
                embedding_status TEXT,
                embedding_model TEXT,
                embedding_dimension INTEGER,
                embedding_count INTEGER DEFAULT 0,
                embedding_index_name TEXT,
                embedded_at TEXT,
                embedding_progress REAL DEFAULT 0.0,
                error TEXT
            )
        """
    )

    # Migration: Add embedding_progress if missing.
    try:
        conn.execute("ALTER TABLE datasets ADD COLUMN embedding_progress REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass

    conn.commit()


def get_db():
    db_path = _db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _initialize_schema(conn)
    return conn


def init_db():
    with get_db():
        pass
