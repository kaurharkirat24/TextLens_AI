"""Prompt construction for grounded QA modes."""

from __future__ import annotations

from typing import Any


class PromptBuilder:
    """Build compact, intent-aware prompts."""

    def build(self, question: str, rows: list[dict[str, Any]], prompt_style: str, analytics: dict[str, Any] | None) -> str:
        row_context = _row_context(rows)
        analytics_context = _analytics_context(analytics)

        style_instruction = {
            "factual": "Answer directly in 2-4 sentences. Cite only rows that support the answer.",
            "exploration": "Identify 2-4 patterns from the retrieved rows. Cite row IDs for examples.",
            "summary": "Give a concise dataset summary using aggregate facts first, then representative examples.",
            "aggregation": "Use the aggregate facts as the source of truth. Do not infer global frequencies from examples.",
            "trend": "Describe time patterns only when time data is provided. Mention if time evidence is insufficient.",
            "comparison": "Compare groups using provided aggregate facts. Cite examples only as illustrations.",
        }.get(prompt_style, "Answer clearly using only the supplied evidence.")

        return (
            "You are analyzing a user-uploaded dataset.\n"
            "Use only the supplied evidence. If evidence is insufficient, say so.\n"
            "Write plain text only; do not use markdown formatting.\n"
            "Use row citations like Row 123, never Row 123.0.\n\n"
            f"Question: {question}\n\n"
            f"Answer style: {style_instruction}\n\n"
            f"Aggregate facts:\n{analytics_context}\n\n"
            f"Evidence rows:\n{row_context}\n"
        )


def _row_context(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No evidence rows supplied."
    lines = []
    for row in rows:
        metadata = row.get("metadata") or {}
        row_id = metadata.get("row_id", row.get("id", "unknown"))
        lines.append(f"Row {_clean_row_id(row_id)}: {row.get('text', '')}")
    return "\n".join(lines)


def _analytics_context(analytics: dict[str, Any] | None) -> str:
    if not analytics:
        return "No aggregate facts supplied."

    lines = [f"Rows analyzed: {analytics.get('row_count_analyzed', 0)}"]
    keywords = analytics.get("keywords") or []
    if keywords:
        rendered = ", ".join(
            f"{item.get('word') or item.get('keyword')} ({int(item.get('count', 0))})"
            for item in keywords[:10]
        )
        lines.append(f"Top keywords/topics: {rendered}")

    sentiment = analytics.get("sentiment_distribution") or {}
    if sentiment:
        lines.append(
            "Sentiment distribution: "
            + ", ".join(f"{label}: {count}" for label, count in sentiment.items())
        )

    categorical = analytics.get("categorical_distributions") or []
    for item in categorical[:4]:
        values = ", ".join(
            f"{entry.get('value')}: {entry.get('count')}" for entry in (item.get("top_values") or [])[:5]
        )
        if values:
            lines.append(f"Top values for {item.get('column')}: {values}")

    numeric = analytics.get("numeric_summaries") or []
    for item in numeric[:4]:
        lines.append(
            f"{item.get('column')} summary: min {item.get('min')}, max {item.get('max')}, "
            f"mean {item.get('mean')}, median {item.get('median')}"
        )

    time_summary = analytics.get("time_summary") or {}
    if time_summary.get("available"):
        lines.append(
            f"Time range: {time_summary.get('first')} to {time_summary.get('last')} "
            f"using {time_summary.get('column')}"
        )
    for note in analytics.get("notes") or []:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def _clean_row_id(value: Any) -> str:
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)
