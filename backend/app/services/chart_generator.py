"""Chart configuration generation for Phase 2 insights."""

from __future__ import annotations

import hashlib
from typing import Any

from app.services.schema_detector import ColumnType

SENTIMENT_COLORS = {
    "positive": "#34d399",
    "neutral": "#60a5fa",
    "negative": "#f87171",
}

CHART_PALETTE = [
    "#4f6ef7",
    "#a855f7",
    "#34d399",
    "#fbbf24",
    "#f87171",
    "#60a5fa",
    "#f472b6",
    "#38bdf8",
    "#fb923c",
    "#a3e635",
]


def generate_charts(
    insights: dict,
    schema: dict[str, ColumnType] | None = None,
    keywords: dict | None = None,
    enrichment_report: dict | None = None,
) -> list[dict]:
    """Convert insights into chart configs with ``type``, ``x``, ``y``, and ``data``."""
    charts: list[dict] = []

    for col, item in insights.get("text", {}).items():
        charts.extend(_text_charts(col, item))

    for col, item in insights.get("numeric", {}).items():
        charts.extend(_numeric_charts(col, item))

    for col, item in insights.get("categorical", {}).items():
        charts.extend(_categorical_charts(col, item))

    for col, item in insights.get("datetime", {}).items():
        charts.extend(_datetime_charts(col, item))

    for key, item in insights.get("combinations", {}).items():
        charts.extend(_combination_charts(key, item))

    summary = insights.get("summary", {})
    column_types = summary.get("column_types", {})
    if any(column_types.values()):
        labels = [key for key, value in column_types.items() if value]
        values = [int(column_types[key]) for key in labels]
        charts.append(
            _chart(
                seed="column_types",
                chart_type="donut",
                title="Column Type Distribution",
                subtitle=f"{summary.get('total_columns', 0)} columns detected",
                x=[label.capitalize() for label in labels],
                y=values,
                data=[
                    {
                        "name": label.capitalize(),
                        "value": value,
                        "color": CHART_PALETTE[i % len(CHART_PALETTE)],
                    }
                    for i, (label, value) in enumerate(zip(labels, values))
                ],
            )
        )

    return charts


def _text_charts(col: str, item: dict) -> list[dict]:
    charts = []
    sentiment = item.get("sentiment_distribution")
    if sentiment:
        counts = sentiment.get("counts", {})
        labels = list(counts.keys())
        values = [int(counts[label]) for label in labels]
        data = [
            {
                "name": label.capitalize(),
                "value": int(counts[label]),
                "percentage": sentiment.get("percentages", {}).get(label, 0),
                "color": SENTIMENT_COLORS.get(label, "#8b90a8"),
            }
            for label in labels
        ]
        charts.append(
            _chart(
                seed=f"sentiment_donut_{col}",
                chart_type="donut",
                title="Sentiment Distribution",
                subtitle=_prettify(col),
                x=[label.capitalize() for label in labels],
                y=values,
                data=data,
            )
        )
        charts.append(
            _chart(
                seed=f"sentiment_bar_{col}",
                chart_type="bar",
                title="Sentiment Breakdown",
                subtitle=_prettify(col),
                x=[label.capitalize() for label in labels],
                y=values,
                data=data,
                colors=[SENTIMENT_COLORS.get(label, "#8b90a8") for label in labels],
            )
        )

    histogram = item.get("sentiment_score_histogram")
    if histogram and histogram.get("counts"):
        charts.append(
            _chart(
                seed=f"sentiment_hist_{col}",
                chart_type="histogram",
                title="Sentiment Score Distribution",
                subtitle=_prettify(col),
                x=histogram["labels"],
                y=histogram["counts"],
                color="#a855f7",
            )
        )

    keywords = item.get("keywords", [])
    if keywords:
        top = keywords[:15]
        charts.append(
            _chart(
                seed=f"keywords_{col}",
                chart_type="horizontal_bar",
                title="Top Keywords",
                subtitle=_prettify(col),
                x=[entry["word"] for entry in top],
                y=[int(entry["count"]) for entry in top],
                data=top,
                color="#4f6ef7",
            )
        )

    word_count = item.get("word_count_stats")
    if word_count:
        labels = ["Mean", "Median", "Min", "Max"]
        values = [word_count.get("mean"), word_count.get("median"), word_count.get("min"), word_count.get("max")]
        charts.append(
            _chart(
                seed=f"word_count_{col}",
                chart_type="stat_card",
                title="Word Count Statistics",
                subtitle=_prettify(col),
                x=labels,
                y=values,
                data=[{"label": label, "value": value} for label, value in zip(labels, values)],
            )
        )

    return charts


def _numeric_charts(col: str, item: dict) -> list[dict]:
    charts = []
    histogram = item.get("histogram")
    if histogram and histogram.get("counts"):
        charts.append(
            _chart(
                seed=f"numeric_hist_{col}",
                chart_type="histogram",
                title=f"Distribution of {_prettify(col)}",
                subtitle="Frequency histogram",
                x=histogram["labels"],
                y=histogram["counts"],
                color="#34d399",
            )
        )

    stats = item.get("summary_stats")
    if stats:
        labels = ["Mean", "Median", "Std", "Min", "Max", "Q25", "Q75"]
        keys = ["mean", "median", "std", "min", "max", "q25", "q75"]
        values = [stats.get(key) for key in keys]
        charts.append(
            _chart(
                seed=f"numeric_stats_{col}",
                chart_type="stat_card",
                title=f"{_prettify(col)} Summary",
                subtitle=f"{stats.get('count', 0)} values",
                x=labels,
                y=values,
                data=[{"label": label, "value": value} for label, value in zip(labels, values)],
            )
        )
    return charts


def _categorical_charts(col: str, item: dict) -> list[dict]:
    counts = item.get("value_counts", {})
    if not counts:
        return []
    labels = list(counts.keys())
    values = [int(counts[label]) for label in labels]
    data = [
        {
            "name": label,
            "value": value,
            "percentage": item.get("percentages", {}).get(label, 0),
            "color": CHART_PALETTE[i % len(CHART_PALETTE)],
        }
        for i, (label, value) in enumerate(zip(labels, values))
    ]
    charts = [
        _chart(
            seed=f"category_bar_{col}",
            chart_type="bar",
            title=f"Value Counts - {_prettify(col)}",
            subtitle=f"{item.get('unique_values', 0)} unique values",
            x=labels,
            y=values,
            data=data,
            colors=[entry["color"] for entry in data],
        )
    ]
    if len(labels) <= 8:
        charts.append(
            _chart(
                seed=f"category_pie_{col}",
                chart_type="pie",
                title=f"{_prettify(col)} Distribution",
                subtitle=f"{item.get('total', 0)} rows",
                x=labels,
                y=values,
                data=data,
            )
        )
    return charts


def _datetime_charts(col: str, item: dict) -> list[dict]:
    charts = []
    volume = item.get("volume_over_time")
    if volume:
        charts.append(
            _chart(
                seed=f"datetime_volume_{col}",
                chart_type="area",
                title="Volume Over Time",
                subtitle=f"{_prettify(col)} by {volume.get('frequency', 'period')}",
                x=volume.get("labels", []),
                y=volume.get("values", []),
                color="#38bdf8",
            )
        )

    for key, trend in item.items():
        if not key.startswith("sentiment_trend_"):
            continue
        text_col = key.replace("sentiment_trend_", "")
        charts.append(
            _chart(
                seed=f"sentiment_trend_{col}_{text_col}",
                chart_type="line",
                title="Sentiment Trend",
                subtitle=f"{_prettify(text_col)} by {trend.get('frequency', 'period')}",
                x=trend.get("labels", []),
                y=trend.get("values", []),
                color="#a855f7",
                options={"y_label": "Average sentiment score", "reference_line": 0},
            )
        )
    return charts


def _combination_charts(key: str, item: dict) -> list[dict]:
    if item.get("type") == "correlation":
        points = item.get("points", [])
        return [
            _chart(
                seed=f"correlation_{key}",
                chart_type="scatter",
                title=f"Correlation: {_prettify(item.get('x_col', ''))} vs Sentiment",
                subtitle=item.get("interpretation", ""),
                x=[point["x"] for point in points],
                y=[point["y"] for point in points],
                data=points,
                x_label=item.get("x_col", "x"),
                y_label=item.get("y_col", "sentiment score"),
                color="#f472b6",
            )
        ]

    if item.get("type") == "sentiment_by_category":
        data = item.get("data", {})
        categories = list(data.keys())
        sentiment_keys = sorted({sentiment for row in data.values() for sentiment in row.keys()})
        stacked_data = []
        for category in categories:
            row = {"category": category}
            for sentiment in sentiment_keys:
                row[sentiment] = int(data[category].get(sentiment, 0))
            stacked_data.append(row)
        return [
            _chart(
                seed=f"sentiment_by_category_{key}",
                chart_type="stacked_bar",
                title=f"Sentiment by {_prettify(item.get('category_col', 'Category'))}",
                subtitle="Distribution across categories",
                x=categories,
                y=[sum(row.get(sentiment, 0) for sentiment in sentiment_keys) for row in stacked_data],
                data=stacked_data,
                keys=sentiment_keys,
                colors=[SENTIMENT_COLORS.get(key, "#8b90a8") for key in sentiment_keys],
            )
        ]

    return []


def _chart(
    *,
    seed: str,
    chart_type: str,
    title: str,
    subtitle: str,
    x: list,
    y: list,
    data: list[dict] | None = None,
    **extra: Any,
) -> dict:
    payload = {
        "id": "chart_" + hashlib.md5(seed.encode("utf-8")).hexdigest()[:10],
        "type": chart_type,
        "title": title,
        "subtitle": subtitle,
        "x": x,
        "y": y,
        "data": data if data is not None else [{"x": xv, "y": yv} for xv, yv in zip(x, y)],
    }
    payload.update(extra)
    return payload


def _prettify(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()
