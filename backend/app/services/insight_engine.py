"""Insight generation for processed TextLens datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.schema_detector import ColumnType


def generate_insights(
    df: pd.DataFrame,
    schema: dict[str, ColumnType],
    processing_report: dict | None = None,
    keywords: dict[str, list[dict]] | None = None,
    column_roles: dict | None = None,
) -> dict[str, Any]:
    """Generate schema-driven insights from a processed DataFrame."""
    text_cols = _columns_of_type(schema, "text", df)
    primary_text_cols = _primary_text_columns(text_cols, df, column_roles)
    numeric_cols = _columns_of_type(schema, "numeric", df)
    categorical_cols = _columns_of_type(schema, "categorical", df)
    datetime_cols = _columns_of_type(schema, "datetime", df)

    insights: dict[str, Any] = {
        "summary": {
            "total_rows": int(len(df)),
            "total_columns": int(len(schema)),
            "column_types": {
                "text": len(text_cols),
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "datetime": len(datetime_cols),
            },
            "column_roles": column_roles or {},
        }
    }

    dataset_insights = _dataset_insights(df, schema, column_roles or {}, primary_text_cols)
    if dataset_insights:
        insights["dataset"] = dataset_insights

    if primary_text_cols:
        insights["text"] = _text_insights(df, primary_text_cols, keywords or {})
    if numeric_cols:
        insights["numeric"] = _numeric_insights(df, numeric_cols)
    if categorical_cols:
        insights["categorical"] = _categorical_insights(df, categorical_cols)
    if datetime_cols:
        insights["datetime"] = _datetime_insights(df, datetime_cols, primary_text_cols)

    combinations = _combination_insights(df, primary_text_cols, numeric_cols, categorical_cols)
    if combinations:
        insights["combinations"] = combinations

    if processing_report:
        insights["processing"] = {
            "cleaning": processing_report.get("cleaning", {}),
            "enrichment": processing_report.get("enrichment", {}),
        }

    return insights


def _dataset_insights(
    df: pd.DataFrame,
    schema: dict[str, ColumnType],
    column_roles: dict,
    primary_text_cols: list[str],
) -> dict[str, Any]:
    dataset: dict[str, Any] = {}
    content_roles = column_roles.get("content", {})
    engagement_roles = column_roles.get("engagement", {})
    time_roles = column_roles.get("time", {})
    primary_text = primary_text_cols[0] if primary_text_cols else None

    content_col = content_roles.get("title") or content_roles.get("id")
    if content_col in df.columns:
        counts = df[content_col].astype(str).replace("", "(blank)").value_counts().head(10)
        if not counts.empty:
            dataset["top_content_by_comments"] = {
                "content_col": content_col,
                "items": [
                    {"content": str(label), "comments": int(count)}
                    for label, count in counts.items()
                ],
            }

    datetime_col = time_roles.get("primary_datetime")
    if datetime_col in df.columns:
        comments_over_time = _volume_over_time(df[datetime_col])
        if comments_over_time:
            comments_over_time["datetime_col"] = datetime_col
            dataset["comments_over_time"] = comments_over_time

    distributions: dict[str, Any] = {}
    for metric in ("likes", "replies"):
        col = engagement_roles.get(metric)
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                distributions[metric] = {
                    "column": col,
                    "summary_stats": _series_stats(values),
                    "histogram": _histogram(values),
                }
    if distributions:
        dataset["engagement_distributions"] = distributions

    if primary_text:
        score_col = f"{primary_text}__sentiment_score"
        metric_col = _preferred_engagement_metric(engagement_roles, df)
        if score_col in df.columns and metric_col:
            valid = pd.DataFrame(
                {
                    "x": pd.to_numeric(df[metric_col], errors="coerce"),
                    "y": pd.to_numeric(df[score_col], errors="coerce"),
                }
            ).dropna()
            if len(valid) >= 5 and valid["x"].nunique() >= 2:
                correlation = valid["x"].corr(valid["y"])
                dataset["engagement_vs_sentiment"] = {
                    "engagement_col": metric_col,
                    "sentiment_col": score_col,
                    "correlation": round(float(correlation), 3) if not pd.isna(correlation) else None,
                    "points": _sample_points(valid),
                }

    key_insights = _key_insights(df, schema, column_roles, dataset, primary_text)
    if key_insights:
        dataset["key_insights"] = key_insights

    return dataset


def _text_insights(df: pd.DataFrame, text_cols: list[str], keywords: dict[str, list[dict]]) -> dict:
    results: dict[str, dict] = {}
    for col in text_cols:
        item: dict[str, Any] = {}
        label_col = f"{col}__sentiment_label"
        score_col = f"{col}__sentiment_score"
        word_col = f"{col}__word_count"

        if label_col in df.columns:
            counts = df[label_col].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
            total = int(counts.sum())
            item["sentiment_distribution"] = {
                "counts": {str(k): int(v) for k, v in counts.items()},
                "percentages": {
                    str(k): round((int(v) / total * 100), 1) if total else 0.0
                    for k, v in counts.items()
                },
                "total": total,
            }

        if score_col in df.columns:
            scores = pd.to_numeric(df[score_col], errors="coerce").dropna()
            item["sentiment_score"] = _series_stats(scores)
            item["sentiment_score_histogram"] = _histogram(scores, bins=10, value_range=(-1, 1))

        if word_col in df.columns:
            word_counts = pd.to_numeric(df[word_col], errors="coerce").dropna()
            item["word_count_stats"] = _series_stats(word_counts)

        item["keywords"] = keywords.get(col, [])
        results[col] = item
    return results


def _numeric_insights(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    results = {}
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        results[col] = {
            "summary_stats": _series_stats(values),
            "histogram": _histogram(values),
        }
    return results


def _categorical_insights(df: pd.DataFrame, categorical_cols: list[str]) -> dict:
    results = {}
    for col in categorical_cols:
        values = df[col].astype(str).replace("", "(blank)")
        counts = values.value_counts().head(15)
        total = int(values.shape[0])
        results[col] = {
            "value_counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {
                str(k): round(int(v) / total * 100, 1) if total else 0.0
                for k, v in counts.items()
            },
            "unique_values": int(values.nunique(dropna=False)),
            "total": total,
            "top_value": str(counts.index[0]) if not counts.empty else None,
            "top_value_count": int(counts.iloc[0]) if not counts.empty else 0,
        }
    return results


def _datetime_insights(df: pd.DataFrame, datetime_cols: list[str], text_cols: list[str]) -> dict:
    results = {}
    for col in datetime_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid = parsed.dropna()
        if valid.empty:
            continue

        freq, freq_label = _time_frequency(valid)
        periods = valid.dt.to_period(freq)
        counts = periods.value_counts().sort_index()
        item: dict[str, Any] = {
            "range": {
                "earliest": valid.min().isoformat(),
                "latest": valid.max().isoformat(),
                "span_days": int((valid.max() - valid.min()).days) if len(valid) > 1 else 0,
            },
            "volume_over_time": {
                "labels": [str(period) for period in counts.index],
                "values": [int(v) for v in counts.values],
                "frequency": freq_label,
            },
        }

        for text_col in text_cols:
            score_col = f"{text_col}__sentiment_score"
            if score_col not in df.columns:
                continue
            temp = pd.DataFrame(
                {
                    "period": parsed.dt.to_period(freq),
                    "score": pd.to_numeric(df[score_col], errors="coerce"),
                }
            ).dropna()
            if temp.empty:
                continue
            grouped = temp.groupby("period")["score"].mean().sort_index()
            item[f"sentiment_trend_{text_col}"] = {
                "labels": [str(period) for period in grouped.index],
                "values": [round(float(v), 3) for v in grouped.values],
                "frequency": freq_label,
            }
        results[col] = item
    return results


def _volume_over_time(series: pd.Series) -> dict[str, Any] | None:
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return None
    freq, freq_label = _time_frequency(valid)
    counts = valid.dt.to_period(freq).value_counts().sort_index()
    return {
        "labels": [str(period) for period in counts.index],
        "values": [int(v) for v in counts.values],
        "frequency": freq_label,
        "range": {
            "earliest": valid.min().isoformat(),
            "latest": valid.max().isoformat(),
            "span_days": int((valid.max() - valid.min()).days) if len(valid) > 1 else 0,
        },
    }


def _combination_insights(
    df: pd.DataFrame,
    text_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict:
    results: dict[str, dict] = {}

    for text_col in text_cols:
        score_col = f"{text_col}__sentiment_score"
        label_col = f"{text_col}__sentiment_label"
        if score_col not in df.columns:
            continue

        for numeric_col in numeric_cols:
            valid = pd.DataFrame(
                {
                    "x": pd.to_numeric(df[numeric_col], errors="coerce"),
                    "y": pd.to_numeric(df[score_col], errors="coerce"),
                }
            ).dropna()
            if len(valid) < 5 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
                continue
            correlation = valid["x"].corr(valid["y"])
            if pd.isna(correlation) or abs(float(correlation)) < 0.05:
                continue
            key = f"{text_col}_sentiment_vs_{numeric_col}"
            results[key] = {
                "type": "correlation",
                "x_col": numeric_col,
                "y_col": f"{text_col} sentiment score",
                "correlation": round(float(correlation), 3),
                "interpretation": _interpret_correlation(float(correlation)),
                "points": _sample_points(valid),
            }

        if label_col not in df.columns:
            continue
        for category_col in categorical_cols:
            values = df[category_col].astype(str).replace("", "(blank)")
            if values.nunique(dropna=False) > 10:
                continue
            cross = pd.crosstab(values, df[label_col])
            if cross.empty:
                continue
            key = f"{text_col}_sentiment_by_{category_col}"
            results[key] = {
                "type": "sentiment_by_category",
                "category_col": category_col,
                "data": {
                    str(index): {str(col): int(value) for col, value in row.items()}
                    for index, row in cross.iterrows()
                },
            }

    return results


def _series_stats(series: pd.Series) -> dict[str, float | int | None]:
    if series.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
        }
    return {
        "count": int(len(series)),
        "mean": round(float(series.mean()), 3),
        "median": round(float(series.median()), 3),
        "std": round(float(series.std()), 3) if len(series) > 1 else 0.0,
        "min": round(float(series.min()), 3),
        "max": round(float(series.max()), 3),
        "q25": round(float(series.quantile(0.25)), 3),
        "q75": round(float(series.quantile(0.75)), 3),
    }


def _histogram(
    series: pd.Series,
    bins: int | None = None,
    value_range: tuple[float, float] | None = None,
) -> dict:
    if series.empty:
        return {"counts": [], "bin_edges": [], "labels": []}
    bin_count = bins or min(20, max(5, int(np.sqrt(len(series)))))
    counts, edges = np.histogram(series, bins=bin_count, range=value_range)
    rounded_edges = [round(float(edge), 3) for edge in edges.tolist()]
    return {
        "counts": [int(v) for v in counts.tolist()],
        "bin_edges": rounded_edges,
        "labels": [
            f"{rounded_edges[i]} to {rounded_edges[i + 1]}"
            for i in range(len(rounded_edges) - 1)
        ],
    }


def _time_frequency(valid: pd.Series) -> tuple[str, str]:
    span_days = int((valid.max() - valid.min()).days) if len(valid) > 1 else 0
    if span_days <= 31:
        return "D", "day"
    if span_days <= 365:
        return "W", "week"
    if span_days <= 1825:
        return "M", "month"
    return "Y", "year"


def _sample_points(data: pd.DataFrame, max_points: int = 200) -> list[dict]:
    sample = data if len(data) <= max_points else data.sample(n=max_points, random_state=42)
    return [
        {"x": round(float(row.x), 3), "y": round(float(row.y), 3)}
        for row in sample.itertuples(index=False)
    ]


def _interpret_correlation(value: float) -> str:
    strength = "very weak"
    if abs(value) >= 0.8:
        strength = "very strong"
    elif abs(value) >= 0.6:
        strength = "strong"
    elif abs(value) >= 0.4:
        strength = "moderate"
    elif abs(value) >= 0.2:
        strength = "weak"
    direction = "positive" if value > 0 else "negative"
    return f"{strength} {direction} correlation"


def _columns_of_type(schema: dict[str, ColumnType], column_type: ColumnType, df: pd.DataFrame) -> list[str]:
    return [col for col, detected in schema.items() if detected == column_type and col in df.columns]


def _primary_text_columns(text_cols: list[str], df: pd.DataFrame, column_roles: dict | None) -> list[str]:
    primary = (column_roles or {}).get("primary_text")
    if primary in df.columns:
        return [primary]
    return text_cols


def _preferred_engagement_metric(engagement_roles: dict, df: pd.DataFrame) -> str | None:
    for metric in ("likes", "replies", "views", "shares", "comments"):
        col = engagement_roles.get(metric)
        if col in df.columns:
            return col
    return None


def _key_insights(
    df: pd.DataFrame,
    schema: dict[str, ColumnType],
    column_roles: dict,
    dataset: dict[str, Any],
    primary_text: str | None,
) -> list[dict[str, str]]:
    insights = [
        {
            "label": "Primary text",
            "value": _prettify(primary_text) if primary_text else "Not detected",
            "detail": "Sentiment and keywords are generated from this column only.",
        },
        {
            "label": "Rows analyzed",
            "value": f"{len(df):,}",
            "detail": f"{len(schema):,} columns scanned for dataset-level signals.",
        },
    ]

    sentiment = {}
    if primary_text:
        label_col = f"{primary_text}__sentiment_label"
        if label_col in df.columns:
            sentiment = df[label_col].value_counts().to_dict()
    if sentiment:
        dominant, count = max(sentiment.items(), key=lambda item: item[1])
        pct = round((count / len(df) * 100), 1) if len(df) else 0
        insights.append(
            {
                "label": "Dominant sentiment",
                "value": str(dominant).capitalize(),
                "detail": f"{pct}% of analyzed rows are {dominant}.",
            }
        )

    top_content = dataset.get("top_content_by_comments", {}).get("items", [])
    if top_content:
        insights.append(
            {
                "label": "Most discussed content",
                "value": str(top_content[0]["content"]),
                "detail": f"{top_content[0]['comments']:,} comments in the analyzed dataset.",
            }
        )

    time_range = dataset.get("comments_over_time", {}).get("range")
    if time_range:
        insights.append(
            {
                "label": "Time span",
                "value": f"{time_range.get('span_days', 0):,} days",
                "detail": "Based on the primary datetime column.",
            }
        )

    return insights[:5]


def _prettify(value: str | None) -> str:
    if not value:
        return ""
    return str(value).replace("_", " ").replace("-", " ").title()
