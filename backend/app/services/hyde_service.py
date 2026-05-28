"""HyDE query expansion for vague/short questions."""

from __future__ import annotations

import logging
import re

import httpx

from app.core.config import settings


logger = logging.getLogger(__name__)


class HyDEService:
    """Generate a hypothetical answer and use it for query embedding."""

    def expand(self, query: str) -> str:
        if not settings.HYDE_ENABLED:
            return query
        if not self._should_apply(query):
            return query
        hypothetical = self._generate_hypothetical_answer(query)
        if not hypothetical:
            return query
        return hypothetical

    def _should_apply(self, query: str) -> bool:
        tokens = re.findall(r"[a-zA-Z0-9_'-]+", query.strip().lower())
        if not tokens:
            return False
        if len(tokens) > settings.HYDE_MAX_QUERY_TOKENS:
            return False
        if re.search(r"\b(compare|trend|distribution|percentage|count|how many)\b", query.lower()):
            return False
        return True

    def _generate_hypothetical_answer(self, query: str) -> str | None:
        if not settings.LLM_ENABLED or not settings.LLM_PROVIDER:
            return None
        prompt = (
            "Write a short hypothetical dataset-grounded answer passage for retrieval.\n"
            "2-4 sentences, plain text, no markdown.\n"
            f"Question: {query[:settings.HYDE_PROMPT_MAX_CHARS]}"
        )
        provider = settings.LLM_PROVIDER.strip().lower()
        try:
            if provider == "gemini":
                return self._gemini_text(prompt)
            if provider == "ollama":
                return self._ollama_text(prompt)
        except Exception:
            logger.exception("HyDE expansion failed; using raw query")
        return None

    def _gemini_text(self, prompt: str) -> str | None:
        if not settings.LLM_API_KEY:
            return None
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.LLM_API_KEY)
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        text = (getattr(response, "text", "") or "").strip()
        return text or None

    def _ollama_text(self, prompt: str) -> str | None:
        base_url = (settings.LLM_BASE_URL or settings.OLLAMA_BASE_URL).rstrip("/")
        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        }
        with httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        content = (data.get("response") or "").strip()
        return content or None
