"""Scalable, resumable, and CPU-efficient ingestion pipeline for Phase 3."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import asyncio
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
        pipeline_start = time.perf_counter()
        try:
            latest_meta = self._get_meta()
            mark_embedding_started(self.dataset_id, resume=latest_meta.embedding_status == "processing")
            
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
            logger.info(
                "Embedding pipeline completed for %s in %.1f seconds",
                self.dataset_id,
                time.perf_counter() - pipeline_start,
            )
            
        except Exception as exc:
            logger.exception("Ingestion pipeline failed for %s", self.dataset_id)
            mark_embedding_failed(self.dataset_id, str(exc))
            raise

    async def run_async(self) -> None:
        """Execute the ingestion pipeline without blocking the event loop."""
        pipeline_start = time.perf_counter()
        try:
            latest_meta = self._get_meta()
            mark_embedding_started(self.dataset_id, resume=latest_meta.embedding_status == "processing")

            chunks_file, _ = await asyncio.to_thread(self._process_chunks)
            embedded_count, dimension = await self._embed_and_upload_async(chunks_file)

            await asyncio.to_thread(self._cleanup)
            await asyncio.to_thread(
                mark_embedding_completed,
                self.dataset_id,
                model=self.embedding_service.model_name,
                dimension=dimension,
                count=embedded_count,
                index_name=self.embedding_index_name,
            )
            logger.info(
                "Async embedding pipeline completed for %s in %.1f seconds",
                self.dataset_id,
                time.perf_counter() - pipeline_start,
            )
        except Exception as exc:
            logger.exception("Ingestion pipeline failed for %s", self.dataset_id)
            mark_embedding_failed(self.dataset_id, str(exc))
            raise

    def _process_chunks(self) -> tuple[Path, Path]:
        """Stream CSV and split into chunks, saving to Parquet for persistence."""
        start = time.perf_counter()
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
        elapsed = time.perf_counter() - start
        logger.info(
            "Chunking completed for %s: rows_processed=%s chunks=%s elapsed_seconds=%.1f rows_per_sec=%.1f chunks_per_sec=%.1f",
            self.dataset_id,
            total_rows_processed,
            chunk_count,
            elapsed,
            total_rows_processed / max(elapsed, 0.001),
            chunk_count / max(elapsed, 0.001),
        )
        
        return chunks_path, chunks_path

    def prepare_remote_embedding_job(self) -> tuple[Path, int, int, str]:
        """Prepare chunks and index metadata for an external embedding worker."""
        from app.services.vector_store_service import PineconeVectorStore

        chunks_path, _ = self._process_chunks()
        total_chunks = self._count_chunks(chunks_path)
        if total_chunks == 0:
            raise SemanticDatasetError("No chunks found for embedding.")

        dimension = self.embedding_service.get_dimension()
        vector_store = PineconeVectorStore.for_dimension(dimension)
        vector_store.ensure_index(dimension)
        self.embedding_index_name = vector_store.index_name
        return chunks_path, total_chunks, dimension, vector_store.index_name

    def _embed_and_upload(self, chunks_path: Path) -> tuple[int, int]:
        """Embed chunks and upsert each batch immediately."""
        from app.services.vector_store_service import PineconeVectorStore

        total_chunks = self._count_chunks(chunks_path)
        if total_chunks == 0:
            raise SemanticDatasetError("No chunks found for embedding.")

        checkpoint_path = self._embedding_checkpoint_path()
        start_idx = self._load_checkpoint(checkpoint_path, chunks_path, total_chunks)
        if start_idx >= total_chunks:
            start_idx = 0

        vector_store = PineconeVectorStore()
        dimension = self._load_checkpoint_dimension(checkpoint_path)
        checkpoint_index_name = self._load_checkpoint_index_name(checkpoint_path)
        if dimension:
            vector_store = PineconeVectorStore.for_dimension(dimension)
            self.embedding_index_name = checkpoint_index_name or vector_store.index_name
            vector_store.ensure_index(dimension)
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

        embed_ms_total = 0.0
        payload_ms_total = 0.0
        upsert_ms_total = 0.0
        loop_start = time.perf_counter()
        pending_vectors: list[tuple[str, list[float], dict[str, Any]]] = []

        def flush_vectors(force: bool = False) -> tuple[int, float]:
            nonlocal upsert_ms_total
            if not pending_vectors:
                return 0, 0.0
            if not force and len(pending_vectors) < settings.PINECONE_UPSERT_BATCH_SIZE:
                return 0, 0.0

            vectors_to_upsert = list(pending_vectors)
            pending_vectors.clear()
            upsert_start = time.perf_counter()
            vector_store.upsert_vectors(vectors_to_upsert, namespace=self.dataset_id)
            upsert_ms = (time.perf_counter() - upsert_start) * 1000
            upsert_ms_total += upsert_ms
            logger.info(
                "Embedding upload flush: dataset=%s vectors=%s upsert_ms=%.1f vectors_per_sec=%.1f",
                self.dataset_id,
                len(vectors_to_upsert),
                upsert_ms,
                len(vectors_to_upsert) / max(upsert_ms / 1000, 0.001),
            )
            return len(vectors_to_upsert), upsert_ms

        for batch_number, (i, batch) in enumerate(
            self._iter_chunk_batches(chunks_path, start_idx, batch_size),
            start=1,
        ):
            end_idx = min(i + len(batch), total_chunks)
            batch_texts = batch["text"].tolist()
            embed_start = time.perf_counter()
            batch_embeddings = self.embedding_service.embed_texts(batch_texts, show_progress=False)
            embed_ms = (time.perf_counter() - embed_start) * 1000
            embed_ms_total += embed_ms
            if not batch_embeddings:
                continue

            if not dimension:
                dimension = len(batch_embeddings[0])
                vector_store = PineconeVectorStore.for_dimension(dimension)
                vector_store.ensure_index(dimension)
                self.embedding_index_name = vector_store.index_name

            payload_start = time.perf_counter()
            vectors = []
            for embedding, row in zip(batch_embeddings, batch.to_dict("records")):
                meta = json.loads(row["metadata"])
                vector_id = f"{self.dataset_id}_{meta['row_id']}_{meta['chunk_id']}"
                vectors.append((vector_id, embedding, meta))
            payload_ms = (time.perf_counter() - payload_start) * 1000
            payload_ms_total += payload_ms

            pending_vectors.extend(vectors)
            flushed_count, upsert_ms = flush_vectors()
            if settings.EMBEDDING_SAVE_VECTORS:
                saved_embeddings.extend(batch_embeddings)

            embedded_count = end_idx
            if batch_number % checkpoint_every == 0 or embedded_count == total_chunks:
                checkpoint_flushed_count, checkpoint_upsert_ms = flush_vectors(force=True)
                if checkpoint_flushed_count:
                    flushed_count += checkpoint_flushed_count
                    upsert_ms += checkpoint_upsert_ms
                self._save_checkpoint(checkpoint_path, chunks_path, total_chunks, embedded_count, dimension, self.embedding_index_name)
                mark_embedding_progress(self.dataset_id, embedded_count / total_chunks, embedded_count)

            logger.info(
                "Embedding upload progress: %.1f%% (%s/%s) batch=%s embed_ms=%.1f payload_ms=%.1f upsert_ms=%.1f flushed_vectors=%s upsert_vectors_per_sec=%.1f",
                (embedded_count / total_chunks) * 100,
                embedded_count,
                total_chunks,
                batch_number,
                embed_ms,
                payload_ms,
                upsert_ms,
                flushed_count,
                flushed_count / max(upsert_ms / 1000, 0.001) if flushed_count else 0.0,
            )

        flush_vectors(force=True)

        if settings.EMBEDDING_SAVE_VECTORS and saved_embeddings:
            embeddings_path = self.storage["embeddings"] / f"{self.dataset_id}_embeddings.npy"
            np.save(embeddings_path, np.array(saved_embeddings, dtype=np.float32))
            logger.info("Saved %s embeddings to %s", len(saved_embeddings), embeddings_path)

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        if not dimension:
            dimension = self.embedding_service.get_dimension()

        elapsed = time.perf_counter() - loop_start
        processed = max(0, embedded_count - start_idx)
        logger.info(
            "Embedding upload summary for %s: processed=%s total=%s elapsed_seconds=%.1f vectors_per_sec=%.1f embed_seconds=%.1f payload_seconds=%.1f upsert_seconds=%.1f",
            self.dataset_id,
            processed,
            total_chunks,
            elapsed,
            processed / max(elapsed, 0.001),
            embed_ms_total / 1000,
            payload_ms_total / 1000,
            upsert_ms_total / 1000,
        )
        return embedded_count, dimension

    async def _embed_and_upload_async(self, chunks_path: Path) -> tuple[int, int]:
        """Embed chunks and upsert batches with async workers and durable checkpoints."""
        from app.services.vector_store_service import PineconeVectorStore

        total_chunks = await asyncio.to_thread(self._count_chunks, chunks_path)
        if total_chunks == 0:
            raise SemanticDatasetError("No chunks found for embedding.")

        checkpoint_path = self._embedding_checkpoint_path()
        start_idx = await asyncio.to_thread(self._load_checkpoint, checkpoint_path, chunks_path, total_chunks)
        if start_idx >= total_chunks:
            start_idx = 0

        dimension = await asyncio.to_thread(self._load_checkpoint_dimension, checkpoint_path)
        checkpoint_index_name = await asyncio.to_thread(self._load_checkpoint_index_name, checkpoint_path)
        embedded_count = start_idx
        batch_size = self.embedding_service.batch_size
        checkpoint_every = max(1, settings.EMBEDDING_CHECKPOINT_EVERY_BATCHES)
        worker_count = max(1, settings.EMBEDDING_WORKERS)
        pending: dict[int, tuple[int, int, int]] = {}
        next_checkpoint_batch = 1
        index_ready = False
        index_lock = asyncio.Lock()
        embed_ms_total = 0.0
        payload_ms_total = 0.0
        upsert_ms_total = 0.0
        pending_vectors: list[tuple[str, list[float], dict[str, Any]]] = []
        upsert_lock = asyncio.Lock()
        loop_start = time.perf_counter()

        logger.info(
            "Async embedding upload for %s: %s chunks, batch_size=%s, workers=%s, start=%s",
            self.dataset_id,
            total_chunks,
            batch_size,
            worker_count,
            start_idx,
        )

        async def add_and_flush_vectors(vectors: list[tuple[str, list[float], dict[str, Any]]]) -> tuple[int, float]:
            nonlocal upsert_ms_total
            async with upsert_lock:
                pending_vectors.extend(vectors)
                if len(pending_vectors) < settings.PINECONE_UPSERT_BATCH_SIZE:
                    return 0, 0.0
                vectors_to_upsert = list(pending_vectors)
                pending_vectors.clear()
            upsert_start = time.perf_counter()
            await asyncio.to_thread(batch_store.upsert_vectors, vectors_to_upsert, self.dataset_id)
            upsert_ms = (time.perf_counter() - upsert_start) * 1000
            upsert_ms_total += upsert_ms
            logger.info(
                "Async embedding upload flush: dataset=%s vectors=%s upsert_ms=%.1f vectors_per_sec=%.1f",
                self.dataset_id,
                len(vectors_to_upsert),
                upsert_ms,
                len(vectors_to_upsert) / max(upsert_ms / 1000, 0.001),
            )
            return len(vectors_to_upsert), upsert_ms

        async def flush_vectors() -> tuple[int, float]:
            nonlocal upsert_ms_total
            async with upsert_lock:
                if not pending_vectors:
                    return 0, 0.0
                vectors_to_upsert = list(pending_vectors)
                pending_vectors.clear()
            upsert_start = time.perf_counter()
            await asyncio.to_thread(batch_store.upsert_vectors, vectors_to_upsert, self.dataset_id)
            upsert_ms = (time.perf_counter() - upsert_start) * 1000
            upsert_ms_total += upsert_ms
            logger.info(
                "Async embedding upload flush: dataset=%s vectors=%s upsert_ms=%.1f vectors_per_sec=%.1f",
                self.dataset_id,
                len(vectors_to_upsert),
                upsert_ms,
                len(vectors_to_upsert) / max(upsert_ms / 1000, 0.001),
            )
            return len(vectors_to_upsert), upsert_ms

        batch_store = PineconeVectorStore()
        if dimension:
            batch_store = PineconeVectorStore(index_name=checkpoint_index_name) if checkpoint_index_name else PineconeVectorStore.for_dimension(dimension)
            self.embedding_index_name = batch_store.index_name
            await asyncio.to_thread(batch_store.ensure_index, dimension)
            index_ready = True

        async def process_batch(batch_number: int, i: int, batch: pd.DataFrame) -> tuple[int, int, int, int, float, float, int, float]:
            nonlocal dimension, index_ready, batch_store
            batch_texts = batch["text"].tolist()
            embed_start = time.perf_counter()
            batch_embeddings = await asyncio.to_thread(
                self.embedding_service.embed_texts,
                batch_texts,
                False,
            )
            embed_ms = (time.perf_counter() - embed_start) * 1000
            if not batch_embeddings:
                return batch_number, i, i, 0, embed_ms, 0.0, 0, 0.0

            batch_dimension = len(batch_embeddings[0])
            async with index_lock:
                if dimension and dimension != batch_dimension:
                    raise SemanticDatasetError(
                        f"Embedding dimension changed from {dimension} to {batch_dimension} during ingestion."
                    )
                if not dimension:
                    dimension = batch_dimension
                if not index_ready:
                    batch_store = PineconeVectorStore.for_dimension(dimension)
                    await asyncio.to_thread(batch_store.ensure_index, dimension)
                    self.embedding_index_name = batch_store.index_name
                    index_ready = True

            payload_start = time.perf_counter()
            vectors = []
            for embedding, row in zip(batch_embeddings, batch.to_dict("records")):
                meta = json.loads(row["metadata"])
                vector_id = f"{self.dataset_id}_{meta['row_id']}_{meta['chunk_id']}"
                vectors.append((vector_id, embedding, meta))
            payload_ms = (time.perf_counter() - payload_start) * 1000

            flushed_count, upsert_ms = await add_and_flush_vectors(vectors)
            self.embedding_index_name = batch_store.index_name
            return batch_number, i, min(i + len(batch), total_chunks), batch_dimension, embed_ms, payload_ms, flushed_count, upsert_ms

        semaphore = asyncio.Semaphore(worker_count)

        async def guarded_process(batch_number: int, i: int, batch: pd.DataFrame) -> tuple[int, int, int, int, float, float, int, float]:
            async with semaphore:
                return await process_batch(batch_number, i, batch)

        batch_iter = enumerate(
            self._iter_chunk_batches(chunks_path, start_idx, batch_size),
            start=1,
        )
        active_tasks: set[asyncio.Task] = set()
        max_queued = worker_count * 2

        def queue_next_batches() -> None:
            while len(active_tasks) < max_queued:
                try:
                    batch_number, (i, batch) = next(batch_iter)
                except StopIteration:
                    return
                active_tasks.add(asyncio.create_task(guarded_process(batch_number, i, batch)))

        queue_next_batches()
        try:
            while active_tasks:
                done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                queue_next_batches()
                for task in done:
                    batch_number, _start, end_idx, batch_dimension, embed_ms, payload_ms, flushed_count, upsert_ms = await task
                    embed_ms_total += embed_ms
                    payload_ms_total += payload_ms
                    if batch_dimension and not dimension:
                        dimension = batch_dimension
                    pending[batch_number] = (end_idx, batch_dimension, batch_number)

                    while next_checkpoint_batch in pending:
                        contiguous_end, contiguous_dimension, _ = pending.pop(next_checkpoint_batch)
                        embedded_count = contiguous_end
                        if contiguous_dimension:
                            dimension = contiguous_dimension
                        if next_checkpoint_batch % checkpoint_every == 0 or embedded_count == total_chunks:
                            checkpoint_flushed_count, checkpoint_upsert_ms = await flush_vectors()
                            if checkpoint_flushed_count:
                                flushed_count += checkpoint_flushed_count
                                upsert_ms += checkpoint_upsert_ms
                            await asyncio.to_thread(
                                self._save_checkpoint,
                                checkpoint_path,
                                chunks_path,
                                total_chunks,
                                embedded_count,
                                dimension,
                                self.embedding_index_name,
                            )
                            await asyncio.to_thread(
                                mark_embedding_progress,
                                self.dataset_id,
                                embedded_count / total_chunks,
                                embedded_count,
                            )
                        next_checkpoint_batch += 1

                    logger.info(
                        "Async embedding upload progress: %.1f%% (%s/%s) batch=%s embed_ms=%.1f payload_ms=%.1f upsert_ms=%.1f flushed_vectors=%s upsert_vectors_per_sec=%.1f",
                        (embedded_count / total_chunks) * 100,
                        embedded_count,
                        total_chunks,
                        batch_number,
                        embed_ms,
                        payload_ms,
                        upsert_ms,
                        flushed_count,
                        flushed_count / max(upsert_ms / 1000, 0.001) if flushed_count else 0.0,
                    )
        except Exception:
            for task in active_tasks:
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            raise

        await flush_vectors()

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        if not dimension:
            dimension = await asyncio.to_thread(self.embedding_service.get_dimension)

        elapsed = time.perf_counter() - loop_start
        processed = max(0, embedded_count - start_idx)
        logger.info(
            "Async embedding upload summary for %s: processed=%s total=%s elapsed_seconds=%.1f vectors_per_sec=%.1f embed_seconds=%.1f payload_seconds=%.1f upsert_seconds=%.1f workers=%s",
            self.dataset_id,
            processed,
            total_chunks,
            elapsed,
            processed / max(elapsed, 0.001),
            embed_ms_total / 1000,
            payload_ms_total / 1000,
            upsert_ms_total / 1000,
            worker_count,
        )
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

    def _embedding_checkpoint_path(self) -> Path:
        return self.storage["temp"] / f"{self.dataset_id}_embedding_upload_checkpoint.json"

    def _checkpoint_signature(self, chunks_path: Path, total_chunks: int) -> dict[str, Any]:
        stat = chunks_path.stat() if chunks_path.exists() else None
        return {
            "dataset_id": self.dataset_id,
            "chunks_path": str(chunks_path),
            "chunks_size": stat.st_size if stat else None,
            "chunks_mtime_ns": stat.st_mtime_ns if stat else None,
            "total_chunks": total_chunks,
            "embedding_provider": self.embedding_service.provider,
            "embedding_model": self.embedding_service.model_name,
        }

    def _load_checkpoint(self, checkpoint_path: Path, chunks_path: Path, total_chunks: int) -> int:
        if not checkpoint_path.exists():
            return 0
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            if checkpoint.get("signature") != self._checkpoint_signature(chunks_path, total_chunks):
                logger.info("Ignoring stale embedding checkpoint %s", checkpoint_path)
                return 0
            return max(0, int(checkpoint.get("last_index", 0)))
        except Exception:
            logger.warning("Failed to read embedding checkpoint %s; restarting from zero", checkpoint_path)
            return 0

    def _load_checkpoint_dimension(self, checkpoint_path: Path) -> int:
        if not checkpoint_path.exists():
            return 0
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            return max(0, int(checkpoint.get("dimension", 0) or 0))
        except Exception:
            return 0

    def _load_checkpoint_index_name(self, checkpoint_path: Path) -> str:
        if not checkpoint_path.exists():
            return ""
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            return str(checkpoint.get("index_name") or "")
        except Exception:
            return ""

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        chunks_path: Path,
        total_chunks: int,
        last_index: int,
        dimension: int,
        index_name: str,
    ) -> None:
        checkpoint = {
            "signature": self._checkpoint_signature(chunks_path, total_chunks),
            "last_index": int(last_index),
            "total_chunks": int(total_chunks),
            "progress": round(last_index / total_chunks, 3) if total_chunks else 0.0,
            "dimension": int(dimension or 0),
            "index_name": index_name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temp_path = checkpoint_path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle)
        temp_path.replace(checkpoint_path)

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


def mark_embedding_progress(dataset_id: str, progress: float, count: int | None = None) -> None:
    """Update embedding progress (0.0 to 1.0)."""
    fields: dict[str, Any] = {"embedding_progress": round(progress, 3)}
    if count is not None:
        fields["embedding_count"] = int(count)
    update_dataset(dataset_id, **fields)


def mark_embedding_started(dataset_id: str, resume: bool = False) -> None:
    """Set initial processing state."""
    fields: dict[str, Any] = {"embedding_status": "processing", "error": None}
    if not resume:
        fields.update(embedding_progress=0.0, embedding_count=0)
    update_dataset(dataset_id, **fields)


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
