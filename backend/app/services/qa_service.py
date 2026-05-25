"""Question answering over retrieved dataset rows with deterministic fallback."""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import Any

import httpx

from app.core.config import settings


logger = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "with",
}


class QAService:
    """Answer questions from retrieved rows, using an LLM when configured."""

    _gemini_client_cache = None
    _gemini_client_key = ""

    def answer(self, question: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        supporting_rows = _supporting_rows(rows)
        try:
            llm_answer = self._llm_answer(question, rows)
            if llm_answer:
                logger.info("QA completed in llm mode using %s", settings.LLM_MODEL)
                return {"answer": llm_answer, "supporting_rows": supporting_rows, "mode": "llm"}
        except Exception:
            logger.exception("LLM QA failed; using deterministic fallback")

        logger.info("QA completed in fallback mode")
        return {
            "answer": self._fallback_answer(question, rows),
            "supporting_rows": supporting_rows,
            "mode": "fallback",
        }

    def _llm_answer(self, question: str, rows: list[dict[str, Any]]) -> str | None:
        if not settings.LLM_ENABLED or not settings.LLM_PROVIDER:
            return None

        provider = settings.LLM_PROVIDER.strip().lower()
        if provider == "gemini":
            return self._gemini_answer(question, rows)
        if provider != "ollama":
            logger.warning("Unsupported LLM provider '%s'; using fallback", settings.LLM_PROVIDER)
            return None

        base_url = (settings.LLM_BASE_URL or settings.OLLAMA_BASE_URL).rstrip("/")
        prompt = _build_prompt(question, rows)
        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }

        with httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            start = time.perf_counter()
            response = client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("Ollama QA generation completed in %.1f ms", elapsed_ms)

        if data.get("done") is False:
            return None
        content = (data.get("response") or "").strip()
        return content or None

    def _gemini_answer(self, question: str, rows: list[dict[str, Any]]) -> str | None:
        if not settings.LLM_API_KEY:
            logger.warning("Missing Gemini API key for QA; using fallback")
            return None

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is not installed. Run 'pip install google-genai'") from exc

        prompt = _build_prompt(question, rows)
        client = self._gemini_client(genai)
        start = time.perf_counter()
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=(
                    "You answer only from the supplied dataset rows. "
                    "If the rows do not support an answer, say that the dataset evidence is insufficient. "
                    "Cite row IDs when making claims."
                ),
            ),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Gemini QA generation completed in %.1f ms", elapsed_ms)
        content = (getattr(response, "text", "") or "").strip()
        return content or None

    def _gemini_client(self, genai_module):
        if self._gemini_client_cache is None or self._gemini_client_key != settings.LLM_API_KEY:
            self._gemini_client_cache = genai_module.Client(api_key=settings.LLM_API_KEY)
            self._gemini_client_key = settings.LLM_API_KEY
        return self._gemini_client_cache

    def _fallback_answer(self, question: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No relevant rows were retrieved for this question."

        sentiments = Counter(
            str((row.get("metadata") or {}).get("sentiment") or "unknown").strip() or "unknown"
            for row in rows
        )
        keywords = Counter()
        for row in rows:
            keywords.update(_keywords(str(row.get("text") or "")))

        sentiment_summary = ", ".join(f"{label}: {count}" for label, count in sentiments.most_common())
        keyword_summary = ", ".join(word for word, _ in keywords.most_common(8)) or "no dominant keywords"
        row_count = len(rows)
        top_score = max(float(row.get("score") or 0.0) for row in rows)

        return (
            f"Summary based on {row_count} retrieved rows. "
            f"Sentiment distribution: {sentiment_summary}. "
            f"Frequent keywords: {keyword_summary}. "
            f"The strongest semantic match scored {top_score:.3f}."
        )


def _build_prompt(question: str, rows: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"Row {row.get('metadata', {}).get('row_id', row.get('id'))}: {row.get('text', '')}"
        for row in rows
    )
    return (
        "You are analyzing dataset entries.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Instructions:\n"
        "- Answer clearly and concisely\n"
        "- Use ONLY the provided context\n"
        "- Cite row IDs for concrete claims\n"
        "- Identify key patterns or trends"
    )


def _supporting_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("id", ""),
            "text": row.get("text", ""),
            "metadata": row.get("metadata") or {},
            "score": float(row.get("score") or 0.0),
        }
        for row in rows
    ]


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", text.lower())
    return [word for word in words if word not in STOPWORDS]
