"""Build retrieval-optimized text from canonical row records."""

from __future__ import annotations

import re
from typing import Any


RETRIEVAL_TEXT_VERSION = "retrieval_text_v1"

TEXT_FIELD_HINTS = {
    "review",
    "comment",
    "feedback",
    "description",
    "text",
    "body",
    "message",
    "agent_notes",
    "notes",
}
TITLE_FIELD_HINTS = {"title", "name", "product", "item", "movie", "show"}
RATING_FIELD_HINTS = {"rating", "score", "stars", "star_rating"}


def build_retrieval_text(record: dict[str, Any], strategy: str = "hybrid") -> str:
    """Render a canonical record into text for embedding."""
    fields = record.get("business_fields") or {}
    if strategy == "label_value":
        return build_label_value_text(fields)
    if strategy == "natural_language":
        return build_natural_language_text(fields)
    if strategy == "markdown":
        return build_markdown_text(fields)
    return build_hybrid_text(fields)


def build_hybrid_text(fields: dict[str, Any]) -> str:
    """Combine a sentence-like summary with exact field labels."""
    natural = build_natural_language_text(fields)
    label_value = build_label_value_text(fields)
    if natural and label_value and natural != label_value:
        return f"{natural}\n\nFields:\n{label_value}"
    return natural or label_value


def build_label_value_text(fields: dict[str, Any]) -> str:
    """Generic renderer that works for arbitrary CSV schemas."""
    lines = []
    for column, value in fields.items():
        cleaned = _clean_value(value)
        if cleaned:
            lines.append(f"{_humanize(column)}: {cleaned}")
    return "\n".join(lines)


def build_natural_language_text(fields: dict[str, Any]) -> str:
    """Domain-aware sentence renderer with a generic fallback."""
    if not fields:
        return ""

    title_column = _first_matching_field(fields, TITLE_FIELD_HINTS)
    rating_column = _first_matching_field(fields, RATING_FIELD_HINTS)
    text_column = _first_matching_field(fields, TEXT_FIELD_HINTS)

    title = _clean_value(fields.get(title_column)) if title_column else ""
    rating = _clean_value(fields.get(rating_column)) if rating_column else ""
    primary_text = _clean_value(fields.get(text_column)) if text_column else ""

    sentences = []
    if title and rating:
        sentences.append(f"Customer reviewed {title} with a rating of {rating}.")
    elif title:
        sentences.append(f"Record about {title}.")
    elif rating:
        sentences.append(f"Record with rating {rating}.")

    if primary_text:
        label = _humanize(text_column or "text")
        sentences.append(f"{label}:\n{primary_text}")

    if sentences:
        return "\n\n".join(sentences)
    return build_label_value_text(fields)


def build_markdown_text(fields: dict[str, Any]) -> str:
    """Markdown renderer for prompt display or experiments."""
    label_value = build_label_value_text(fields)
    if not label_value:
        return ""
    return f"# Record\n\n{label_value}"


def _first_matching_field(fields: dict[str, Any], hints: set[str]) -> str | None:
    normalized = {_metadata_key(column): column for column in fields}
    for hint in hints:
        if hint in normalized:
            return normalized[hint]
    for key, column in normalized.items():
        if any(hint in key for hint in hints):
            return column
    return None


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _humanize(column: str) -> str:
    return str(column).replace("_", " ").strip().title()


def _metadata_key(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
