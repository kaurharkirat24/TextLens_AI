"""Safe dataframe-backed QA for exact lookup, filtering, and aggregates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.models.schemas import DatasetMeta
from app.services.retrieval_context import RetrievalContext
from app.services.semantic_dataset_service import build_row_text


MAX_STRUCTURED_ROWS = 25
MAX_VALUE_SCAN = 50_000


@dataclass(frozen=True)
class StructuredResult:
    answer: str
    rows: list[dict[str, Any]]
    plan: dict[str, Any]


class StructuredQueryService:
    """Translate common natural-language table questions into safe pandas plans."""

    def answer(
        self,
        meta: DatasetMeta | RetrievalContext,
        question: str,
        top_k: int,
        analysis: dict[str, Any] | None = None,
    ) -> StructuredResult | None:
        context = _context_from_inputs(meta, analysis)
        df = context.dataframe
        if df.empty:
            return None

        columns = _ColumnIndex(df, context.analysis)
        normalized_question = _normalize(question)

        result = self._answer_title_lookup(df, columns, normalized_question, top_k)
        if result:
            return result

        result = self._answer_numeric_extreme(df, columns, normalized_question, top_k)
        if result:
            return result

        result = self._answer_average(df, columns, normalized_question)
        if result:
            return result

        result = self._answer_group_count(df, columns, normalized_question, top_k)
        if result:
            return result

        result = self._answer_filtered_count(df, columns, normalized_question, top_k)
        if result:
            return result

        result = self._answer_filtered_rows(df, columns, normalized_question, top_k)
        if result:
            return result

        result = self._answer_duration_filter(df, columns, normalized_question, top_k)
        if result:
            return result

        result = self._answer_categorical_filter(df, columns, normalized_question, top_k)
        if result:
            return result

        return None

    def _answer_average(self, df: pd.DataFrame, columns: "_ColumnIndex", normalized_question: str) -> StructuredResult | None:
        if not re.search(r"\b(avg|average|mean)\b", normalized_question):
            return None

        measure = _measure_series(df, columns, normalized_question)
        if not measure:
            return None
        label, series, unit = measure
        mask, filters = _filter_mask(df, columns, normalized_question)
        values = series[mask].dropna()
        if values.empty:
            return StructuredResult(
                answer=f"No rows had usable {label} values for that question.",
                rows=[],
                plan={"strategy": "dataframe_average", "measure": label, "filters": filters},
            )

        average = float(values.mean())
        suffix = f" {unit}" if unit else ""
        filter_text = _filter_text(filters)
        answer = f"The average {label}{filter_text} is {average:.1f}{suffix} across {len(values)} row(s)."
        return StructuredResult(
            answer=answer,
            rows=_rows(df[mask].head(5), score=1.0),
            plan={"strategy": "dataframe_average", "measure": label, "unit": unit, "filters": filters},
        )

    def _answer_numeric_extreme(
        self,
        df: pd.DataFrame,
        columns: "_ColumnIndex",
        normalized_question: str,
        top_k: int,
    ) -> StructuredResult | None:
        match = re.search(r"\b(highest|largest|biggest|lowest|smallest)\b", normalized_question)
        if not match:
            return None

        direction = "lowest" if match.group(1) in {"lowest", "smallest"} else "highest"
        measure = _measure_series(df, columns, normalized_question)
        if not measure:
            return None

        label, series, unit = measure
        mask, filters = _filter_mask(df, columns, normalized_question)
        candidates = df[mask].copy()
        candidates["__measure"] = series[mask]
        candidates = candidates.dropna(subset=["__measure"])
        if candidates.empty:
            return None

        candidates = candidates.sort_values("__measure", ascending=(direction == "lowest"))
        first = candidates.iloc[0]
        title = _clean(first.get(columns.title_column)) if columns.title_column else ""
        value = float(first["__measure"])
        suffix = f" {unit}" if unit else ""
        subject = title or f"row {int(candidates.index[0])}"
        answer = f"{subject} has the {direction} {label}{_filter_text(filters)}: {value:g}{suffix}."
        return StructuredResult(
            answer=answer,
            rows=_rows(candidates.head(top_k).drop(columns=["__measure"]), score=1.0),
            plan={"strategy": "dataframe_extreme", "measure": label, "direction": direction, "filters": filters},
        )

    def _answer_group_count(
        self,
        df: pd.DataFrame,
        columns: "_ColumnIndex",
        normalized_question: str,
        top_k: int,
    ) -> StructuredResult | None:
        wants_group_count = (
            re.search(r"\b(most|least|common|frequent|distribution|vs|versus|by)\b", normalized_question)
            and not re.search(r"\b(best|loved|successful|prefer)\b", normalized_question)
        )
        if not wants_group_count:
            return None

        group_col = _question_column(df, columns, normalized_question)
        if not group_col:
            return None

        mask, filters = _filter_mask(df, columns, normalized_question, exclude_columns={group_col})
        filtered = df[mask]
        if filtered.empty:
            return StructuredResult(
                answer="No rows matched the requested filters.",
                rows=[],
                plan={"strategy": "dataframe_group_count", "group_column": group_col, "filters": filters},
            )

        counts = _value_counts(filtered[group_col])
        if not counts:
            return None

        if re.search(r"\b(vs|versus|distribution|count|counts|by)\b", normalized_question):
            rendered = ", ".join(f"{label}: {count}" for label, count in counts[: min(top_k, 10)])
            answer = f"Counts by {group_col}: {rendered}."
        else:
            label, count = counts[0]
            answer = f"The most common {group_col} is {label} with {count} row(s)."

        return StructuredResult(
            answer=answer,
            rows=_rows(filtered.head(top_k), score=1.0),
            plan={"strategy": "dataframe_group_count", "group_column": group_col, "filters": filters},
        )

    def _answer_filtered_count(
        self,
        df: pd.DataFrame,
        columns: "_ColumnIndex",
        normalized_question: str,
        top_k: int,
    ) -> StructuredResult | None:
        if not re.search(r"\b(how many|count|number of)\b", normalized_question):
            return None

        mask, filters = _filter_mask(df, columns, normalized_question)
        if not filters:
            return None

        filtered = df[mask]
        filter_text = _filter_text(filters)
        answer = f"Found {len(filtered)} row(s){filter_text}."
        examples = _titles(filtered, columns, top_k=5)
        if examples:
            answer += " Examples: " + ", ".join(examples) + "."
        return StructuredResult(
            answer=answer,
            rows=_rows(filtered.head(top_k), score=1.0),
            plan={"strategy": "dataframe_count", "filters": filters},
        )

    def _answer_filtered_rows(
        self,
        df: pd.DataFrame,
        columns: "_ColumnIndex",
        normalized_question: str,
        top_k: int,
    ) -> StructuredResult | None:
        if not re.search(r"\b(recommend|suggest|find|show|list)\b", normalized_question):
            return None

        mask, filters = _filter_mask(df, columns, normalized_question)
        text_terms = _text_search_terms(normalized_question, filters)
        text_filter_applied = False
        if columns.primary_text_column and text_terms:
            text_mask = _text_match_mask(df[columns.primary_text_column], text_terms)
            if text_mask.any():
                mask = mask & text_mask
                text_filter_applied = True

        if not filters and not text_filter_applied:
            return None

        filtered = df[mask]
        if filtered.empty:
            return StructuredResult(
                answer="No rows matched the requested filters.",
                rows=[],
                plan={"strategy": "dataframe_filtered_rows", "filters": filters, "text_terms": text_terms},
            )

        examples = _titles(filtered, columns, top_k=top_k) or [
            _clean(row.get(columns.primary_text_column, "")) for _, row in filtered.head(top_k).iterrows() if columns.primary_text_column
        ]
        answer = f"Found {len(filtered)} row(s){_filter_text(filters)}."
        if examples:
            answer += " Suggestions: " + ", ".join(examples[:top_k]) + "."
        return StructuredResult(
            answer=answer,
            rows=_rows(filtered.head(top_k), score=1.0),
            plan={"strategy": "dataframe_filtered_rows", "filters": filters, "text_terms": text_terms},
        )

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

        filter_mask, filters = _filter_mask(df, columns, normalized_question, exclude_columns={duration_col})
        mask = mask & filter_mask

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
            plan={"strategy": "dataframe_filter", "column": duration_col, "operator": operator, "value": limit, "filters": filters},
        )

    def _answer_categorical_filter(
        self,
        df: pd.DataFrame,
        columns: "_ColumnIndex",
        normalized_question: str,
        top_k: int,
    ) -> StructuredResult | None:
        if re.search(r"\b(best|loved|favorite|successful|prefer|highest|lowest|largest|smallest)\b", normalized_question):
            return None

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


def _context_from_inputs(meta: DatasetMeta | RetrievalContext, analysis: dict[str, Any] | None) -> RetrievalContext:
    if isinstance(meta, RetrievalContext):
        return meta
    return RetrievalContext(dataset_id=meta.id, meta=meta, analysis=analysis or {})


class _ColumnIndex:
    def __init__(self, df: pd.DataFrame, analysis: dict[str, Any] | None = None) -> None:
        self._columns = list(df.columns)
        self._normalized = {_column_key(column): column for column in self._columns}
        roles = (analysis or {}).get("column_roles") or {}
        self.title_column = ((roles.get("content") or {}).get("title") or self.first("title", "name", "movie_title", "show_title"))
        self.id_column = ((roles.get("content") or {}).get("id") or self.first("id", "item_id", "show_id"))
        primary_text = roles.get("primary_text")
        self.primary_text_column = primary_text if primary_text in self._columns else self.first("description", "text", "summary", "review", "comment")

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


def _match_cell_value(df: pd.DataFrame, column: str, normalized_question: str) -> str | None:
    values = df[column].dropna().astype(str).str.strip()
    values = values[values != ""].drop_duplicates()
    if len(values) > MAX_VALUE_SCAN:
        values = values.head(MAX_VALUE_SCAN)
    for value in values:
        normalized_value = _normalize(value)
        if normalized_value and _contains_phrase(normalized_question, normalized_value):
            return normalized_value
    return None


def _filter_mask(
    df: pd.DataFrame,
    columns: _ColumnIndex,
    normalized_question: str,
    exclude_columns: set[str] | None = None,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    exclude = exclude_columns or set()
    mask = pd.Series(True, index=df.index)
    filters: list[dict[str, Any]] = []

    for column in df.columns:
        if column in exclude or column.endswith(("__word_count", "__sentiment_score", "__sentiment_label")):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        matched = _match_cell_value(df, column, normalized_question)
        if matched is None:
            continue
        value_mask = df[column].fillna("").astype(str).map(_normalize).map(
            lambda text: _contains_split_value(text, matched)
        )
        if value_mask.any():
            mask = mask & value_mask
            filters.append({"column": column, "operator": "=", "value": _first_display_value(df, column, matched)})

    for column in df.columns:
        if column in exclude:
            continue
        series = _numeric_or_parsed_series(df[column], normalized_question)
        if series is None:
            continue
        comparison = _comparison_from_question(normalized_question, column)
        if not comparison:
            continue
        operator, limit = comparison
        value_mask = _compare(series, operator, limit)
        if value_mask.notna().any():
            mask = mask & value_mask.fillna(False)
            filters.append({"column": column, "operator": operator, "value": limit})
            break

    return mask, filters


def _question_column(df: pd.DataFrame, columns: _ColumnIndex, normalized_question: str) -> str:
    terms = set(_tokens(normalized_question))
    best: tuple[int, str] | None = None
    for column in df.columns:
        if column.endswith(("__word_count", "__sentiment_score", "__sentiment_label")):
            continue
        label_terms = set(_tokens(_display_label(column)))
        score = len(terms.intersection(label_terms)) * 10
        value_match_count = _matched_value_count(df, column, normalized_question)
        if value_match_count >= 2:
            score += value_match_count * 12
        if columns.title_column and column == columns.title_column and terms.intersection({"title", "titles", "item", "items", "record", "records"}):
            score += 8
        if _is_multi_value_column(df[column]) and terms.intersection({"category", "categories", "genre", "genres", "tag", "tags", "topic", "topics"}):
            score += 7
        if score and (best is None or score > best[0]):
            best = (score, column)
    return best[1] if best else ""


def _matched_value_count(df: pd.DataFrame, column: str, normalized_question: str) -> int:
    values = df[column].dropna().astype(str).str.strip()
    values = values[values != ""].drop_duplicates()
    if len(values) > MAX_VALUE_SCAN:
        values = values.head(MAX_VALUE_SCAN)
    count = 0
    for value in values:
        normalized_value = _normalize(value)
        if normalized_value and _contains_phrase(normalized_question, normalized_value):
            count += 1
    return count


def _measure_series(df: pd.DataFrame, columns: _ColumnIndex, normalized_question: str) -> tuple[str, pd.Series, str] | None:
    terms = set(_tokens(normalized_question))
    candidates: list[tuple[int, str, pd.Series, str]] = []
    for column in df.columns:
        series = _numeric_or_parsed_series(df[column], normalized_question)
        if series is None or series.dropna().empty:
            continue
        label_terms = set(_tokens(_display_label(column)))
        score = len(terms.intersection(label_terms)) * 10
        unit = ""
        if series.attrs.get("unit"):
            unit = str(series.attrs["unit"])
            if unit in terms or (unit == "minutes" and terms.intersection({"duration", "runtime", "minute", "minutes", "mins"})):
                score += 8
            if unit == "seasons" and terms.intersection({"season", "seasons"}):
                score += 8
        if score:
            candidates.append((score, column, series, unit))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, column, series, unit = candidates[0]
    return _display_label(column), series, unit


def _numeric_or_parsed_series(series: pd.Series, normalized_question: str) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.fillna("").astype(str).str.lower()
    if re.search(r"\b(duration|runtime|minute|minutes|mins|under|over|less than|more than)\b", normalized_question):
        minutes = text.str.extract(r"\b(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)\b", expand=False)
        parsed = pd.to_numeric(minutes, errors="coerce")
        if parsed.notna().any():
            parsed.attrs["unit"] = "minutes"
            return parsed
    if re.search(r"\b(season|seasons)\b", normalized_question):
        seasons = text.str.extract(r"\b(\d+(?:\.\d+)?)\s*(?:season|seasons)\b", expand=False)
        parsed = pd.to_numeric(seasons, errors="coerce")
        if parsed.notna().any():
            parsed.attrs["unit"] = "seasons"
            return parsed
    return None


def _comparison_from_question(normalized_question: str, column: str) -> tuple[str, float] | None:
    label = _display_label(column)
    if label and label not in normalized_question and not re.search(r"\b(under|less than|below|shorter than|over|more than|above|after|before)\b", normalized_question):
        return None
    match = re.search(r"\b(under|less than|below|shorter than|before|over|more than|above|longer than|after)\s+(\d+(?:\.\d+)?)", normalized_question)
    if not match:
        return None
    raw_operator, raw_value = match.groups()
    operator = "<" if raw_operator in {"under", "less than", "below", "shorter than", "before"} else ">"
    return operator, float(raw_value)


def _compare(series: pd.Series, operator: str, value: float) -> pd.Series:
    if operator == "<":
        return series < value
    return series > value


def _value_counts(series: pd.Series) -> list[tuple[str, int]]:
    counter: dict[str, int] = {}
    for value in series.dropna().astype(str):
        parts = _split_multi_value(value)
        for part in parts:
            counter[part] = counter.get(part, 0) + 1
    return sorted(counter.items(), key=lambda item: (-item[1], item[0].lower()))


def _contains_split_value(text: str, value: str) -> bool:
    if _contains_phrase(text, value):
        return True
    return any(_normalize(part) == value for part in _split_multi_value(text))


def _split_multi_value(value: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[,;/|]", str(value)) if part.strip()]
    return parts or [str(value).strip()]


def _is_multi_value_column(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    return float(sample.str.contains(r"[,;/|]", regex=True).mean()) >= 0.2


def _first_display_value(df: pd.DataFrame, column: str, normalized_value: str) -> str:
    for value in df[column].dropna().astype(str):
        if _contains_split_value(_normalize(value), normalized_value):
            return _clean(value)
    return normalized_value


def _filter_text(filters: list[dict[str, Any]]) -> str:
    if not filters:
        return ""
    rendered = []
    for item in filters:
        if item["operator"] == "=":
            rendered.append(f"{item['column']} is {item['value']}")
        else:
            rendered.append(f"{item['column']} {item['operator']} {item['value']}")
    return " where " + " and ".join(rendered)


def _text_search_terms(normalized_question: str, filters: list[dict[str, Any]]) -> list[str]:
    filter_terms: set[str] = set()
    for item in filters:
        filter_terms.update(_tokens(str(item.get("column", ""))))
        filter_terms.update(_tokens(str(item.get("value", ""))))
    generic = {
        "about",
        "after",
        "before",
        "find",
        "list",
        "movie",
        "movies",
        "recommend",
        "released",
        "show",
        "shows",
        "suggest",
        "title",
        "titles",
        "under",
        "over",
    }
    terms = []
    for term in _tokens(normalized_question):
        if len(term) < 4 or term in generic or term in filter_terms:
            continue
        terms.append(term)
    return list(dict.fromkeys(terms))[:6]


def _text_match_mask(series: pd.Series, terms: list[str]) -> pd.Series:
    if not terms:
        return pd.Series(False, index=series.index)
    pattern = re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
    return series.fillna("").astype(str).str.contains(pattern, na=False)


def _display_label(column: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(column))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text).lower()


def _tokens(value: str) -> list[str]:
    stop = {"a", "an", "and", "are", "by", "for", "has", "have", "how", "is", "of", "the", "to", "what", "which", "with"}
    return [token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in stop]


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
