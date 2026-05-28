"""Detect whether a question is meaningfully related to an uploaded dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.models.schemas import DatasetMeta
from app.services.query_intent_service import QueryIntent, QueryIntentClassifier
from app.services.retrieval_context import RetrievalContext


_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "be",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "should",
    "tell",
    "that",
    "the",
    "these",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}

_DATASET_REFERENCE_TERMS = {
    "csv",
    "data",
    "dataset",
    "file",
    "records",
    "rows",
    "table",
    "uploaded",
}
MAX_EXACT_VALUE_CANDIDATES_PER_COLUMN = 50_000


@dataclass(frozen=True)
class DatasetRelevanceResult:
    is_related: bool
    confidence: float
    rationale: str
    supported_topics: list[str]
    matched_signals: list[str]


class DatasetRelevanceService:
    """Use detected schema, column roles, keywords, and sample values as guardrails."""

    def __init__(self) -> None:
        self.intent_classifier = QueryIntentClassifier()

    def assess(
        self,
        context_or_meta: RetrievalContext | DatasetMeta,
        analysis_or_question: dict[str, Any] | str,
        question: str | None = None,
    ) -> DatasetRelevanceResult:
        if isinstance(context_or_meta, RetrievalContext):
            context = context_or_meta
            analysis = context.analysis
            question_text = str(analysis_or_question)
        else:
            if not isinstance(analysis_or_question, dict) or question is None:
                raise TypeError("Expected RetrievalContext + question or DatasetMeta + analysis + question.")
            context = RetrievalContext(
                dataset_id=context_or_meta.id,
                meta=context_or_meta,
                analysis=analysis_or_question,
            )
            analysis = context.analysis
            question_text = question

        normalized_question = _normalize(question_text)
        if not normalized_question:
            return DatasetRelevanceResult(False, 0.0, "Question is empty.", [], [])

        profile = context.profile
        schema = _schema_columns(analysis)
        roles = analysis.get("column_roles") if isinstance(analysis, dict) else {}
        roles = roles or {}
        column_names = context.column_names
        supported_topics = profile.get("supported_topics") or _supported_topics(column_names, schema, roles)
        intent = self.intent_classifier.classify(question_text)

        matched_signals = []
        matched_signals.extend(_matched_column_signals(column_names, normalized_question))
        matched_signals.extend(_matched_keyword_signals(analysis, profile, normalized_question))
        matched_signals.extend(_matched_profile_value_signals(profile, normalized_question))
        # Fall back to an exact dataframe scan for values that were not captured in the compact profile.
        if _should_scan_dataframe_values(intent, matched_signals):
            matched_signals.extend(_matched_value_signals(context.dataframe, schema, roles, normalized_question))

        if matched_signals:
            return DatasetRelevanceResult(
                True,
                0.95,
                "Question references detected dataset schema, keywords, or representative values.",
                supported_topics,
                _dedupe(matched_signals)[:8],
            )

        if _is_dataset_level_question(normalized_question, intent):
            return DatasetRelevanceResult(
                True,
                0.7,
                "Question asks about the uploaded dataset as a whole.",
                supported_topics,
                [],
            )

        return DatasetRelevanceResult(
            False,
            0.9,
            "Question does not match the detected dataset schema, keywords, or representative values.",
            supported_topics,
            [],
        )


def _matched_column_signals(columns: Any, normalized_question: str) -> list[str]:
    signals = []
    question_terms = _expanded_terms(_tokens(normalized_question))
    for column in columns:
        label = _display_label(column)
        phrases = {label}
        terms = _column_terms(column)
        phrases.update(terms)
        if _contains_phrase(normalized_question, label) or question_terms.intersection(terms):
            signals.append(f"column:{column}")
    return signals


def _matched_keyword_signals(analysis: dict[str, Any], profile: dict[str, Any], normalized_question: str) -> list[str]:
    if not isinstance(analysis, dict):
        analysis = {}

    question_terms = set(_tokens(normalized_question))
    signals = []
    keywords = analysis.get("keywords") or {}
    if isinstance(keywords, dict):
        keyword_items = []
        for values in keywords.values():
            if isinstance(values, list):
                keyword_items.extend(values)
    elif isinstance(keywords, list):
        keyword_items = keywords
    else:
        keyword_items = []
    keyword_items.extend(profile.get("keywords") or [])

    for item in keyword_items:
        word = ""
        if isinstance(item, dict):
            word = str(item.get("word") or item.get("keyword") or "")
        else:
            word = str(item)
        normalized = _normalize(word)
        if normalized and (normalized in question_terms or _contains_phrase(normalized_question, normalized)):
            signals.append(f"keyword:{word}")
    return signals


def _matched_profile_value_signals(profile: dict[str, Any], normalized_question: str) -> list[str]:
    signals = []
    for column in profile.get("columns") or []:
        column_name = str(column.get("name") or "")
        for value in column.get("sample_values") or []:
            normalized_value = _normalize(str(value))
            if normalized_value and _contains_phrase(normalized_question, normalized_value):
                signals.append(f"value:{column_name}={value}")
                break
    return signals


def _should_scan_dataframe_values(intent: QueryIntent, matched_signals: list[str]) -> bool:
    # Exact lookup and filter questions often mention values beyond the profile sample.
    if intent.intent in {"factual", "aggregation", "comparison"}:
        return True
    return any(signal.startswith("column:") for signal in matched_signals)


def _matched_value_signals(
    df: pd.DataFrame,
    schema: dict[str, str],
    roles: dict[str, Any],
    normalized_question: str,
) -> list[str]:
    signals: list[str] = []
    question_terms = set(_tokens(normalized_question))
    value_columns = _value_match_columns(df, schema, roles)
    for column in value_columns:
        values = df[column].dropna().astype(str).str.strip()
        values = values[values != ""].drop_duplicates()
        if len(values) > MAX_EXACT_VALUE_CANDIDATES_PER_COLUMN:
            values = values.head(MAX_EXACT_VALUE_CANDIDATES_PER_COLUMN)
        for value in values:
            normalized_value = _normalize(value)
            value_terms = set(_tokens(normalized_value))
            if not normalized_value or not value_terms.intersection(question_terms):
                continue
            if _contains_phrase(normalized_question, normalized_value):
                signals.append(f"value:{column}={value}")
                break
    return signals


def _value_match_columns(df: pd.DataFrame, schema: dict[str, str], roles: dict[str, Any]) -> list[str]:
    primary_text = roles.get("primary_text")
    secondary_text = set(roles.get("secondary_text") or [])
    content_roles = roles.get("content") or {}
    role_columns = {value for value in content_roles.values() if value}
    role_columns.update((roles.get("geo") or {}).values())
    role_columns.update((roles.get("time") or {}).values())

    columns = []
    for column in df.columns:
        kind = schema.get(str(column))
        if column == primary_text:
            continue
        if column in secondary_text and _high_cardinality(df[column]):
            continue
        if kind == "text" and column not in role_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]) and column not in role_columns:
            continue
        columns.append(column)
    return columns


def _supported_topics(columns: Any, schema: dict[str, str], roles: dict[str, Any]) -> list[str]:
    prioritized: list[str] = []
    primary_text = roles.get("primary_text")
    if primary_text:
        prioritized.append(primary_text)

    for value in (roles.get("content") or {}).values():
        if value:
            prioritized.append(value)
    for value in (roles.get("geo") or {}).values():
        if value:
            prioritized.append(value)
    for value in (roles.get("time") or {}).values():
        if value:
            prioritized.append(value)
    for value in (roles.get("engagement") or {}).values():
        if value:
            prioritized.append(value)

    for kind in ("categorical", "numeric", "datetime", "text"):
        prioritized.extend(column for column, detected in schema.items() if detected == kind)
    prioritized.extend(str(column) for column in columns)

    topics = []
    seen = set()
    for column in prioritized:
        label = _display_label(column)
        key = label.lower()
        if not label or key in seen or _is_generated_column(column):
            continue
        topics.append(label)
        seen.add(key)
        if len(topics) >= 6:
            break
    return topics


def _schema_columns(analysis: dict[str, Any]) -> dict[str, str]:
    if not isinstance(analysis, dict):
        return {}
    schema = analysis.get("schema") or {}
    columns = schema.get("columns") if isinstance(schema, dict) else {}
    if isinstance(columns, dict):
        return {str(column): str(kind) for column, kind in columns.items()}
    return {}


def _is_dataset_level_question(normalized_question: str, intent: QueryIntent) -> bool:
    terms = set(_tokens(normalized_question))
    if not terms.intersection(_DATASET_REFERENCE_TERMS):
        return False
    return intent.intent in {"summarization", "aggregation", "trend", "comparison", "semantic_exploration"}


def _high_cardinality(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return False
    return values.nunique(dropna=True) / max(len(values), 1) > 0.5


def _column_terms(column: str) -> set[str]:
    label = _display_label(column)
    return _expanded_terms(_tokens(label))


def _expanded_terms(tokens: list[str]) -> set[str]:
    terms: set[str] = set()
    for token in tokens:
        terms.update(_term_variants(token))
    return terms


def _term_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) < 5:
        return variants

    suffixes = (
        ("ed", 2),
        ("ers", 3),
        ("er", 2),
        ("ors", 3),
        ("or", 2),
        ("ies", 3),
        ("s", 1),
    )
    for suffix, trim in suffixes:
        if token.endswith(suffix) and len(token) - trim >= 4:
            base = token[:-trim]
            variants.add(base)
            if suffix == "ies":
                variants.add(f"{base}y")
    return variants


def _display_label(column: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(column))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text).lower()


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOPWORDS]


def _is_generated_column(column: str) -> bool:
    return str(column).endswith(("__word_count", "__sentiment_score", "__sentiment_label"))


def _dedupe(values: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped
