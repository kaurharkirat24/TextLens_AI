"""Question answering over retrieved dataset rows with deterministic fallback."""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import Any

import httpx

from app.core.config import settings
from app.services.prompt_builder import PromptBuilder


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

    def __init__(self) -> None:
        self.prompt_builder = PromptBuilder()

    def answer(
        self,
        question: str,
        rows: list[dict[str, Any]],
        *,
        intent: str = "semantic_exploration",
        strategy: str = "semantic",
        prompt_style: str = "exploration",
        analytics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        supporting_rows = _supporting_rows(rows)
        try:
            llm_answer = self._llm_answer(question, rows, prompt_style, analytics)
            if llm_answer:
                logger.info("QA completed in llm mode using %s", settings.LLM_MODEL)
                return {
                    "answer": llm_answer,
                    "supporting_rows": supporting_rows,
                    "mode": "llm",
                    "intent": intent,
                    "strategy": strategy,
                    "analytics": analytics,
                }
        except Exception:
            logger.exception("LLM QA failed; using deterministic fallback")

        logger.info("QA completed in fallback mode")
        return {
            "answer": self._fallback_answer(question, rows, analytics),
            "supporting_rows": supporting_rows,
            "mode": "fallback",
            "intent": intent,
            "strategy": strategy,
            "analytics": analytics,
        }

    def unrelated_answer(self, supported_topics: list[str] | None = None) -> dict[str, Any]:
        answer = "This question does not appear related to the uploaded dataset."
        topics = [topic for topic in (supported_topics or []) if topic][:6]
        if topics:
            answer += "\n\nI can answer questions about: " + ", ".join(topics) + "."
        return {
            "answer": answer,
            "supporting_rows": [],
            "mode": "out_of_scope",
            "intent": "dataset_relevance",
            "strategy": "guardrail",
            "analytics": None,
        }

    def _llm_answer(
        self,
        question: str,
        rows: list[dict[str, Any]],
        prompt_style: str,
        analytics: dict[str, Any] | None,
    ) -> str | None:
        if not settings.LLM_ENABLED or not settings.LLM_PROVIDER:
            return None

        provider = settings.LLM_PROVIDER.strip().lower()
        if provider == "gemini":
            return self._gemini_answer(question, rows, prompt_style, analytics)
        if provider != "ollama":
            logger.warning("Unsupported LLM provider '%s'; using fallback", settings.LLM_PROVIDER)
            return None

        base_url = (settings.LLM_BASE_URL or settings.OLLAMA_BASE_URL).rstrip("/")
        prompt = self.prompt_builder.build(question, rows, prompt_style, analytics)
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

    def _gemini_answer(
        self,
        question: str,
        rows: list[dict[str, Any]],
        prompt_style: str,
        analytics: dict[str, Any] | None,
    ) -> str | None:
        if not settings.LLM_API_KEY:
            logger.warning("Missing Gemini API key for QA; using fallback")
            return None

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is not installed. Run 'pip install google-genai'") from exc

        prompt = self.prompt_builder.build(question, rows, prompt_style, analytics)
        client = self._gemini_client(genai)
        start = time.perf_counter()
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=(
                    "You answer only from the supplied dataset rows. "
                    "When aggregate facts are supplied, use them for global claims. "
                    "If the rows do not support an answer, say that the dataset evidence is insufficient. "
                    "Cite row IDs when making claims. Use plain text, not markdown."
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

    def _fallback_answer(
        self,
        question: str,
        rows: list[dict[str, Any]],
        analytics: dict[str, Any] | None = None,
    ) -> str:
        if analytics:
            keywords = analytics.get("keywords") or []
            sentiment = analytics.get("sentiment_distribution") or {}
            parts = [f"Analyzed {analytics.get('row_count_analyzed', 0)} rows."]
            if keywords:
                parts.append(
                    "Top terms: "
                    + ", ".join(
                        f"{item.get('word') or item.get('keyword')} ({int(item.get('count', 0))})"
                        for item in keywords[:8]
                    )
                    + "."
                )
            if sentiment:
                parts.append("Sentiment distribution: " + ", ".join(f"{k}: {v}" for k, v in sentiment.items()) + ".")
            categorical = analytics.get("categorical_distributions") or []
            if categorical:
                first = categorical[0]
                values = ", ".join(
                    f"{item.get('value')}: {item.get('count')}"
                    for item in (first.get("top_values") or [])[:5]
                )
                if values:
                    parts.append(f"Top {first.get('column')} values: {values}.")
            return " ".join(parts)

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
