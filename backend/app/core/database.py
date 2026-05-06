import sqlite3
import os
from app.core.config import settings

DB_PATH = os.path.join(settings.UPLOAD_DIR, "textlens.db")

def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
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
        """)
        
        # Migration: Add embedding_progress if missing
        try:
            conn.execute("ALTER TABLE datasets ADD COLUMN embedding_progress REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            pass # Already exists
            
        conn.commit()
