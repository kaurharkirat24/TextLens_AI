"""Scalable, resumable, and CPU-efficient ingestion pipeline for Phase 3."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    sent_tokenize = None

from app.core.config import settings
from app.models.schemas import DatasetMeta
from app.services.dataset_manager import get_dataset, update_dataset
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticDatasetError(RuntimeError):
    """Raised when a dataset is not ready for semantic indexing."""


class SentenceSplitter:
    """Implement efficient sentence-based chunking."""

    def __init__(
        self,
        chunk_size: int = 8,
        overlap: int = 1,
        min_words: int = 5,
        max_words: int = 180,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_words = min_words
        self.max_words = max_words

    def split(self, text: str) -> list[str]:
        """Split text into overlapping sentence chunks."""
        if not text or not isinstance(text, str):
            return []

        words = text.split()
        word_count = len(words)
        if word_count < self.min_words:
            return []
        if word_count <= self.max_words:
            return [text.strip()]

        sentences = self._sentences(text)
        if not sentences:
            return []

        chunks = []
        # Sliding window over sentences
        for i in range(0, len(sentences), self.chunk_size - self.overlap):
            chunk_sentences = sentences[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_sentences).strip()
            
            # Skip chunks with fewer than X words
            chunks.extend(self._split_by_words(chunk_text))
                
            # If we reached the end, stop
            if i + self.chunk_size >= len(sentences):
                break
 
        return chunks

    def _sentences(self, text: str) -> list[str]:
        if sent_tokenize is not None:
            try:
                return [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]
            except LookupError:
                logger.warning("NLTK punkt data is unavailable; using regex sentence splitting.")

        return [sentence.strip() for sentence in SENTENCE_BOUNDARY_RE.split(text) if sentence.strip()]

    def _split_by_words(self, text: str) -> list[str]:
        words = text.split()
        if len(words) < self.min_words:
            return []
        if len(words) <= self.max_words:
            return [text]

        step = max(self.min_words, self.max_words - 30)
        chunks = []
        for start in range(0, len(words), step):
            chunk_words = words[start : start + self.max_words]
            if len(chunk_words) >= self.min_words:
                chunks.append(" ".join(chunk_words))
            if start + self.max_words >= len(words):
                break
        return chunks


class IngestionPipeline:
    """Manages the full RAG ingestion lifecycle with resumability."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self.meta = self._get_meta()
        self.storage = self._ensure_storage()
        self.embedding_service = get_embedding_service()
        self.embedding_index_name = settings.PINECONE_INDEX_NAME
        self.splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE_SENTENCES,
            overlap=settings.CHUNK_OVERLAP_SENTENCES,
            min_words=settings.MIN_CHUNK_WORDS,
            max_words=settings.MAX_CHUNK_WORDS,
        )

    def _get_meta(self) -> DatasetMeta:
        meta = get_dataset(self.dataset_id)
        if not meta:
            raise SemanticDatasetError(f"Dataset '{self.dataset_id}' not found")
        return meta

    def _ensure_storage(self) -> dict[str, Path]:
        """Create and return paths for structured pipeline storage."""
        paths = {
            "raw": Path(settings.DATA_RAW_DIR),
            "cleaned": Path(settings.DATA_CLEANED_DIR),
            "chunks": Path(settings.DATA_CHUNKS_DIR),
            "embeddings": Path(settings.DATA_EMBEDDINGS_DIR),
            "temp": Path(settings.DATA_TEMP_DIR),
            "logs": Path(settings.LOGS_DIR)
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def run(self) -> None:
        """Execute the ingestion pipeline."""
        try:
            mark_embedding_started(self.dataset_id)
            
            # 1. Chunking phase
            chunks_file, _ = self._process_chunks()
            
            # 2. Embed and upload in batches to avoid large in-memory/vector files.
            embedded_count, dimension = self._embed_and_upload(chunks_file)
            
            # 3. Cleanup
            self._cleanup()
            
            # 4. Finalize
            mark_embedding_completed(
                self.dataset_id,
                model=self.embedding_service.model_name,
                dimension=dimension,
                count=embedded_count,
                index_name=self.embedding_index_name
            )
            
        except Exception as exc:
            logger.exception("Ingestion pipeline failed for %s", self.dataset_id)
            mark_embedding_failed(self.dataset_id, str(exc))
            raise

    def _process_chunks(self) -> tuple[Path, Path]:
        """Stream CSV and split into chunks, saving to Parquet for persistence."""
        chunks_path = self._existing_chunks_path()
        manifest_path = self._manifest_path()

        # Check if already done
        if chunks_path.exists() and self._manifest_matches(manifest_path):
            logger.info("Found existing chunks at %s, skipping chunking phase.", chunks_path)
            return chunks_path, chunks_path

        self._delete_stale_artifacts()

        if not self.meta.clean_csv_path or not os.path.exists(self.meta.clean_csv_path):
            raise SemanticDatasetError("Clean dataset not found.")

        analysis = _load_analysis(self.meta, self.dataset_id)
        primary_text_col = (analysis.get("column_roles") or {}).get("primary_text") or self.meta.text_column
        
        if not primary_text_col:
            raise SemanticDatasetError("Primary text column not identified.")

        logger.info("Starting chunking for dataset %s", self.dataset_id)
        chunks_path = self.storage["chunks"] / f"{self.dataset_id}_chunks.jsonl"
        temp_chunks_path = chunks_path.with_suffix(".jsonl.tmp")
        chunk_count = 0
        
        chunk_iter = pd.read_csv(self.meta.clean_csv_path, chunksize=settings.CSV_READ_CHUNK_SIZE)
        total_rows_processed = 0

        with open(temp_chunks_path, "w", encoding="utf-8") as handle:
            for df_chunk in chunk_iter:
                records = df_chunk.to_dict("records")
                for idx, row in zip(df_chunk.index, records):
                    text = str(row.get(primary_text_col, "") or "").strip()
                    if not text:
                        continue

                    sentence_chunks = self.splitter.split(text)
                    for chunk_id, chunk_text in enumerate(sentence_chunks):
                        meta = build_metadata(self.dataset_id, idx, chunk_text, row, analysis)
                        meta["chunk_id"] = chunk_id
                        record = {"text": chunk_text, "metadata": json.dumps(meta)}
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        chunk_count += 1

                total_rows_processed += len(df_chunk)
                logger.debug("Processed %s rows into chunks", total_rows_processed)

        if chunk_count == 0:
            temp_chunks_path.unlink(missing_ok=True)
            raise SemanticDatasetError("Clean dataset has no text chunks suitable for embedding.")

        temp_chunks_path.replace(chunks_path)
        self._write_manifest(manifest_path)
        logger.info("Saved %s chunks to %s", chunk_count, chunks_path)
        
        return chunks_path, chunks_path

    def _embed_and_upload(self, chunks_path: Path) -> tuple[int, int]:
        """Embed chunks and upsert each batch immediately."""
        from app.services.vector_store_service import PineconeVectorStore

        total_chunks = self._count_chunks(chunks_path)
        if total_chunks == 0:
            raise SemanticDatasetError("No chunks found for embedding.")

        checkpoint_path = self.storage["temp"] / f"{self.dataset_id}_embedding_upload_checkpoint.json"
        start_idx = self._load_checkpoint(checkpoint_path)
        if start_idx >= total_chunks:
            start_idx = 0

        vector_store = PineconeVectorStore()
        dimension = 0
        embedded_count = start_idx
        saved_embeddings: list[list[float]] = []
        batch_size = self.embedding_service.batch_size
        checkpoint_every = max(1, settings.EMBEDDING_CHECKPOINT_EVERY_BATCHES)

        logger.info(
            "Embedding and uploading %s chunks for %s in batches of %s starting at %s",
            total_chunks - start_idx,
            self.dataset_id,
            batch_size,
            start_idx,
        )

        for batch_number, (i, batch) in enumerate(
            self._iter_chunk_batches(chunks_path, start_idx, batch_size),
            start=1,
        ):
            end_idx = min(i + len(batch), total_chunks)
            batch_texts = batch["text"].tolist()
            batch_embeddings = self.embedding_service.embed_texts(batch_texts, show_progress=False)
            if not batch_embeddings:
                continue

            if not dimension:
                dimension = len(batch_embeddings[0])
                vector_store = PineconeVectorStore.for_dimension(dimension)
                vector_store.ensure_index(dimension)
                self.embedding_index_name = vector_store.index_name

            vectors = []
            for embedding, row in zip(batch_embeddings, batch.to_dict("records")):
                meta = json.loads(row["metadata"])
                vector_id = f"{self.dataset_id}_{meta['row_id']}_{meta['chunk_id']}"
                vectors.append((vector_id, embedding, meta))

            vector_store.upsert_vectors(vectors, namespace=self.dataset_id)
            if settings.EMBEDDING_SAVE_VECTORS:
                saved_embeddings.extend(batch_embeddings)

            embedded_count = end_idx
            if batch_number % checkpoint_every == 0 or embedded_count == total_chunks:
                self._save_checkpoint(checkpoint_path, embedded_count)
                mark_embedding_progress(self.dataset_id, embedded_count / total_chunks)

            logger.info(
                "Embedding upload progress: %.1f%% (%s/%s)",
                (embedded_count / total_chunks) * 100,
                embedded_count,
                total_chunks,
            )

        if settings.EMBEDDING_SAVE_VECTORS and saved_embeddings:
            embeddings_path = self.storage["embeddings"] / f"{self.dataset_id}_embeddings.npy"
            np.save(embeddings_path, np.array(saved_embeddings, dtype=np.float32))
            logger.info("Saved %s embeddings to %s", len(saved_embeddings), embeddings_path)

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        if not dimension:
            dimension = self.embedding_service.get_dimension()

        return embedded_count, dimension

    def _cleanup(self) -> None:
        """Delete temporary files after successful processing."""
        # Requirements: Delete temporary chunk files and intermediate cleaned files after successful Pinecone upload verification
        # Keep raw uploads, embeddings, metadata, and logs.
        
        # We don't delete the chunks parquet because it's our "metadata" for embeddings.
        # But we should delete files in temp.
        temp_dir = self.storage["temp"]
        for file in temp_dir.glob(f"{self.dataset_id}_*"):
            try:
                file.unlink()
            except Exception:
                pass
        
        # Remove stale temp files (> 24h)
        now = time.time()
        for file in temp_dir.glob("*"):
            if file.is_file() and now - file.stat().st_mtime > 86400:
                try:
                    file.unlink()
                except Exception:
                    pass
        
        logger.info("Cleanup completed for %s", self.dataset_id)

    def _manifest_path(self) -> Path:
        return self.storage["chunks"] / f"{self.dataset_id}_chunks.manifest.json"

    def _artifact_signature(self) -> dict[str, Any]:
        clean_path = Path(self.meta.clean_csv_path or "")
        stat = clean_path.stat() if clean_path.exists() else None
        return {
            "dataset_id": self.dataset_id,
            "clean_csv_path": str(clean_path),
            "clean_csv_size": stat.st_size if stat else None,
            "clean_csv_mtime_ns": stat.st_mtime_ns if stat else None,
            "embedding_provider": self.embedding_service.provider,
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.get_dimension(),
            "chunk_size_sentences": settings.CHUNK_SIZE_SENTENCES,
            "chunk_overlap_sentences": settings.CHUNK_OVERLAP_SENTENCES,
            "min_chunk_words": settings.MIN_CHUNK_WORDS,
            "max_chunk_words": settings.MAX_CHUNK_WORDS,
        }

    def _manifest_matches(self, manifest_path: Path) -> bool:
        if not manifest_path.exists():
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                return json.load(handle) == self._artifact_signature()
        except Exception:
            return False

    def _write_manifest(self, manifest_path: Path) -> None:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(self._artifact_signature(), handle, indent=2)

    def _delete_stale_artifacts(self) -> None:
        for directory in ("chunks", "embeddings", "temp"):
            for file in self.storage[directory].glob(f"{self.dataset_id}_*"):
                try:
                    file.unlink()
                except OSError:
                    logger.warning("Could not delete stale semantic artifact %s", file)

    def _load_checkpoint(self, checkpoint_path: Path) -> int:
        if not checkpoint_path.exists():
            return 0
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            return max(0, int(checkpoint.get("last_index", 0)))
        except Exception:
            logger.warning("Failed to read embedding checkpoint %s; restarting from zero", checkpoint_path)
            return 0

    def _save_checkpoint(self, checkpoint_path: Path, last_index: int) -> None:
        with open(checkpoint_path, "w", encoding="utf-8") as handle:
            json.dump({"last_index": int(last_index)}, handle)

    def _existing_chunks_path(self) -> Path:
        parquet_path = self.storage["chunks"] / f"{self.dataset_id}_chunks.parquet"
        jsonl_path = self.storage["chunks"] / f"{self.dataset_id}_chunks.jsonl"
        if jsonl_path.exists():
            return jsonl_path
        if parquet_path.exists():
            return parquet_path
        return jsonl_path

    def _read_chunks(self, chunks_path: Path) -> pd.DataFrame:
        if chunks_path.suffix == ".parquet":
            try:
                return pd.read_parquet(chunks_path)
            except ImportError as exc:
                raise SemanticDatasetError(
                    "Chunk file is Parquet but no Parquet engine is installed. "
                    "Delete the stale chunk file or install pyarrow/fastparquet."
                ) from exc

        return pd.read_json(chunks_path, orient="records", lines=True)

    def _count_chunks(self, chunks_path: Path) -> int:
        if chunks_path.suffix == ".jsonl":
            with open(chunks_path, "r", encoding="utf-8") as handle:
                return sum(1 for _ in handle)
        return len(self._read_chunks(chunks_path))

    def _iter_chunk_batches(self, chunks_path: Path, start_idx: int, batch_size: int):
        if chunks_path.suffix == ".jsonl":
            reader = pd.read_json(chunks_path, orient="records", lines=True, chunksize=batch_size)
            current_idx = 0
            for batch in reader:
                next_idx = current_idx + len(batch)
                if next_idx <= start_idx:
                    current_idx = next_idx
                    continue
                if start_idx > current_idx:
                    batch = batch.iloc[start_idx - current_idx :]
                    current_idx = start_idx
                yield current_idx, batch
                current_idx = next_idx
            return

        df_chunks = self._read_chunks(chunks_path)
        for i in range(start_idx, len(df_chunks), batch_size):
            yield i, df_chunks.iloc[i : i + batch_size]


def load_semantic_dataset(dataset_id: str) -> tuple[DatasetMeta, pd.DataFrame, dict[str, Any]]:
    """Load dataset metadata and analysis without reading the full clean CSV."""
    meta = get_dataset(dataset_id)
    if not meta:
        raise SemanticDatasetError(f"Dataset '{dataset_id}' not found")

    if not meta.clean_csv_path or not os.path.exists(meta.clean_csv_path):
        raise SemanticDatasetError("Clean dataset not found. Run analysis before embedding or search.")

    analysis = _load_analysis(meta, dataset_id)
    return meta, pd.DataFrame(), analysis


def build_metadata(
    dataset_id: str,
    row_id: int,
    text: str,
    row: pd.Series | dict,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Build Pinecone-compatible metadata with the required Phase 3 shape."""
    roles = analysis.get("column_roles") if isinstance(analysis, dict) else {}
    roles = roles or {}
    engagement_roles = roles.get("engagement") or {}
    time_roles = roles.get("time") or {}

    sentiment_col = _first_present(row, ["Sentiment", "sentiment", "sentiment_label"])
    timestamp_col = time_roles.get("primary_datetime") or _first_present(
        row,
        ["PublishedAt", "published_at", "timestamp", "Timestamp", "created_at", "CreatedAt", "date", "Date"],
    )

    engagement = 0.0
    for column in engagement_roles.values():
        if column in row:
            engagement += _to_number(row[column])
    if not engagement:
        for fallback in ("engagement", "Engagement", "Likes", "likes", "Replies", "replies"):
            if fallback in row:
                engagement += _to_number(row[fallback])

    return {
        "dataset_id": dataset_id,
        "row_id": int(row_id),
        "text": text,
        "source": dataset_id,
        "sentiment": _clean_scalar(row.get(sentiment_col, "")) if sentiment_col else "",
        "engagement": engagement,
        "timestamp": _clean_scalar(row.get(timestamp_col, "")) if timestamp_col else "",
    }


def mark_embedding_completed(
    dataset_id: str,
    *,
    model: str,
    dimension: int,
    count: int,
    index_name: str,
) -> None:
    update_dataset(
        dataset_id,
        embedding_status="completed",
        embedding_model=model,
        embedding_dimension=dimension,
        embedding_count=count,
        embedding_index_name=index_name,
        embedded_at=datetime.now(timezone.utc).isoformat(),
        embedding_progress=1.0,
        error=None,
    )


def mark_embedding_failed(dataset_id: str, error: str) -> None:
    update_dataset(dataset_id, embedding_status="failed", embedding_progress=0.0, error=error)


def mark_embedding_progress(dataset_id: str, progress: float) -> None:
    """Update embedding progress (0.0 to 1.0)."""
    update_dataset(dataset_id, embedding_progress=round(progress, 3))


def mark_embedding_started(dataset_id: str) -> None:
    """Set initial processing state."""
    update_dataset(dataset_id, embedding_status="processing", embedding_progress=0.0)


def _load_analysis(meta: DatasetMeta, dataset_id: str) -> dict[str, Any]:
    candidates = [
        getattr(meta, "analysis_path", "") or "",
        os.path.join(settings.OUTPUT_DIR, f"{dataset_id}_analysis.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                continue
    return {}


def _first_present(row: pd.Series | dict, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in row:
            return column
    return None


def _to_number(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clean_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
