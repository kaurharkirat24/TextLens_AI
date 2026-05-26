"""Safe dataframe-backed QA for exact lookup and filtering questions."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.models.schemas import DatasetMeta
from app.services.semantic_dataset_service import SemanticDatasetError, build_row_text


MAX_STRUCTURED_ROWS = 25


@dataclass(frozen=True)
class StructuredResult:
    answer: str
    rows: list[dict[str, Any]]
    plan: dict[str, Any]


class StructuredQueryService:
    """Translate common natural-language table questions into safe pandas plans."""

    def answer(self, meta: DatasetMeta, question: str, top_k: int) -> StructuredResult | None:
        df = _load_clean_dataframe(meta)
        if df.empty:
            return None

        columns = _ColumnIndex(df)
        normalized_question = _normalize(question)

        result = self._answer_title_lookup(df, columns, normalized_question, top_k)
        if result:
            return result

        result = self._answer_duration_filter(df, columns, normalized_question, top_k)
        if result:
            return result

        result = self._answer_categorical_filter(df, columns, normalized_question, top_k)
        if result:
            return result

        return None

    def _answer_title_lookup(
        self,
        df: pd.DataFrame,
        columns: "_ColumnIndex",
        normalized_question: str,
        top_k: int,
    ) -> StructuredResult | None:
        title_col = columns.first("title", "name", "movie_title", "show_title")
        if not title_col:
            return None

        matched = _match_cell_value(df, title_col, normalized_question)
        if matched is None:
            return None

        lookup_targets = [
            ("director", ("director", "directed", "directed by")),
            ("cast", ("cast", "actor", "actors", "star", "stars")),
            ("rating", ("rating", "rated")),
            ("duration", ("duration", "runtime", "long")),
            ("release_year", ("release year", "released", "year")),
            ("country", ("country", "where")),
        ]
        target_col = ""
        target_label = ""
        for canonical, signals in lookup_targets:
            candidate = columns.first(canonical, *signals)
            if candidate and any(signal in normalized_question for signal in signals):
                target_col = candidate
                target_label = canonical.replace("_", " ")
                break
        if not target_col:
            return None

        rows_df = df[df[title_col].astype(str).map(_normalize) == matched]
        if rows_df.empty:
            return None

        first = rows_df.iloc[0]
        title = _clean(first.get(title_col))
        value = _clean(first.get(target_col))
        if not value:
            return None

        answer = f"{title}'s {target_label} is {value}."
        rows = _rows(rows_df.head(top_k), score=1.0)
        return StructuredResult(
            answer=answer,
            rows=rows,
            plan={
                "strategy": "dataframe_lookup",
                "lookup_column": title_col,
                "lookup_value": title,
                "return_column": target_col,
            },
        )

    def _answer_duration_filter(
        self,
        df: pd.DataFrame,
        columns: "_ColumnIndex",
        normalized_question: str,
        top_k: int,
    ) -> StructuredResult | None:
        duration_col = columns.first("duration", "runtime", "length")
        if not duration_col or not any(term in normalized_question for term in ("minute", "minutes", "min", "duration", "runtime")):
            return None

        match = re.search(r"\b(under|less than|below|shorter than|over|more than|above|longer than)\s+(\d+)", normalized_question)
        if not match:
            return None

        operator_text, raw_limit = match.groups()
        limit = int(raw_limit)
        minutes = df[duration_col].map(_duration_minutes)
        if minutes.notna().sum() == 0:
            return None

        if operator_text in {"under", "less than", "below", "shorter than"}:
            mask = minutes < limit
            operator = "<"
        else:
            mask = minutes > limit
            operator = ">"

        type_col = columns.first("type")
        if type_col and "movie" in normalized_question:
            mask = mask & df[type_col].astype(str).map(_normalize).eq("movie")

        filtered = df[mask].copy()
        if filtered.empty:
            return StructuredResult(
                answer=f"No rows matched duration {operator} {limit} minutes.",
                rows=[],
                plan={"strategy": "dataframe_filter", "column": duration_col, "operator": operator, "value": limit},
            )

        rows = _rows(filtered.head(top_k), score=1.0)
        examples = _titles(filtered, columns, top_k=5)
        answer = f"Found {len(filtered)} row(s) with {duration_col} {operator} {limit} minutes."
        if examples:
            answer += " Examples: " + ", ".join(examples) + "."
        return StructuredResult(
            answer=answer,
            rows=rows,
            plan={"strategy": "dataframe_filter", "column": duration_col, "operator": operator, "value": limit},
        )

    def _answer_categorical_filter(
        self,
        df: pd.DataFrame,
        columns: "_ColumnIndex",
        normalized_question: str,
        top_k: int,
    ) -> StructuredResult | None:
        matches: list[tuple[int, str, str]] = []
        for column in df.columns:
            if _column_key(column) in {"title", "name", "movietitle", "showtitle"}:
                continue
            if column.endswith(("__word_count", "__sentiment_score", "__sentiment_label")):
                continue
            if df[column].nunique(dropna=True) > 100:
                continue
            matched = _match_cell_value(df, column, normalized_question)
            if matched is None:
                continue
            specificity = len(matched)
            if _column_key(column) == "type":
                specificity -= 10
            matches.append((specificity, column, matched))

        for _, column, matched in sorted(matches, reverse=True):
            filtered = df[df[column].astype(str).map(_normalize) == matched]
            if filtered.empty:
                continue
            rows = _rows(filtered.head(top_k), score=1.0)
            examples = _titles(filtered, columns, top_k=5)
            value = _clean(filtered.iloc[0].get(column))
            answer = f"Found {len(filtered)} row(s) where {column} is {value}."
            if examples:
                answer += " Examples: " + ", ".join(examples) + "."
            return StructuredResult(
                answer=answer,
                rows=rows,
                plan={"strategy": "dataframe_filter", "column": column, "operator": "=", "value": value},
            )
        return None


class _ColumnIndex:
    def __init__(self, df: pd.DataFrame) -> None:
        self._columns = list(df.columns)
        self._normalized = {_column_key(column): column for column in self._columns}

    def first(self, *candidates: str) -> str:
        for candidate in candidates:
            key = _column_key(candidate)
            if key in self._normalized:
                return self._normalized[key]
        for candidate in candidates:
            key = _column_key(candidate)
            for normalized, column in self._normalized.items():
                if key and (key in normalized or normalized in key):
                    return column
        return ""


def _load_clean_dataframe(meta: DatasetMeta) -> pd.DataFrame:
    if not meta.clean_csv_path or not os.path.exists(meta.clean_csv_path):
        raise SemanticDatasetError("Clean dataset not found. Run analysis before structured QA.")
    return pd.read_csv(meta.clean_csv_path)


def _match_cell_value(df: pd.DataFrame, column: str, normalized_question: str) -> str | None:
    values = df[column].dropna().astype(str).str.strip()
    values = values[values != ""].drop_duplicates()
    candidates = sorted((value for value in values if len(value) <= 80), key=len, reverse=True)
    for value in candidates:
        normalized_value = _normalize(value)
        if normalized_value and _contains_phrase(normalized_question, normalized_value):
            return normalized_value
    return None


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text):
        return True
    return phrase in text


def _rows(df: pd.DataFrame, score: float) -> list[dict[str, Any]]:
    rows = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        metadata = {str(key): _clean(value) for key, value in row_dict.items()}
        metadata["row_id"] = int(idx)
        metadata["source"] = "dataframe"
        rows.append(
            {
                "id": f"dataframe_{idx}",
                "text": build_row_text(row_dict, {}, None),
                "metadata": metadata,
                "score": score,
            }
        )
    return rows


def _titles(df: pd.DataFrame, columns: _ColumnIndex, top_k: int) -> list[str]:
    title_col = columns.first("title", "name", "movie_title", "show_title")
    if not title_col:
        return []
    return [_clean(value) for value in df[title_col].head(top_k).tolist() if _clean(value)]


def _duration_minutes(value: Any) -> float:
    text = _clean(value).lower()
    match = re.search(r"\b(\d+)\s*min", text)
    if not match:
        return float("nan")
    return float(match.group(1))


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
