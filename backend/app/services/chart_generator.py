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

MAX_CHARTS = 8
MAX_CHARTS_PER_SECTION = 2
SECTION_PRIORITY = {
    "key": 0,
    "time": 1,
    "content": 2,
    "engagement": 3,
    "geo": 4,
    "other": 5,
}


def generate_charts(
    insights: dict,
    schema: dict[str, ColumnType] | None = None,
    keywords: dict | None = None,
    enrichment_report: dict | None = None,
) -> list[dict]:
    """Convert insights into curated analytics chart configs."""
    charts: list[dict] = []

    charts.extend(_dataset_charts(insights.get("dataset", {})))

    column_roles = insights.get("summary", {}).get("column_roles", {})
    primary_text = column_roles.get("primary_text")
    text_insights = insights.get("text", {})
    if primary_text and primary_text in text_insights:
        charts.extend(_primary_text_charts(primary_text, text_insights[primary_text]))
    elif len(text_insights) == 1:
        col, item = next(iter(text_insights.items()))
        charts.extend(_primary_text_charts(col, item))

    charts = _prioritize_charts(_dedupe_charts(charts))

    return charts


def _dataset_charts(dataset: dict) -> list[dict]:
    charts: list[dict] = []

    top_content = dataset.get("top_content_by_comments", {})
    items = top_content.get("items", [])
    if items:
        labels = [entry["content"] for entry in items]
        values = [int(entry["comments"]) for entry in items]
        charts.append(
            _chart(
                seed="top_content_by_comments",
                chart_type="horizontal_bar",
                title="Top Videos by Comment Count",
                subtitle=f"Grouped by {_prettify(top_content.get('content_col', 'content'))}",
                x=labels,
                y=values,
                data=[
                    {"label": label, "value": value, "color": CHART_PALETTE[i % len(CHART_PALETTE)]}
                    for i, (label, value) in enumerate(zip(labels, values))
                ],
                color="#4f6ef7",
                section="content",
            )
        )

    engagement = dataset.get("engagement_vs_sentiment", {})
    points = engagement.get("points", [])
    if points:
        charts.append(
            _chart(
                seed="engagement_vs_sentiment",
                chart_type="scatter",
                title="Engagement vs Sentiment",
                subtitle=f"{_prettify(engagement.get('engagement_col', 'engagement'))} compared with sentiment score",
                x=[point["x"] for point in points],
                y=[point["y"] for point in points],
                data=points,
                x_label=_prettify(engagement.get("engagement_col", "Engagement")),
                y_label="Sentiment score",
                color="#f472b6",
                section="engagement",
            )
        )

    comments_over_time = dataset.get("comments_over_time", {})
    if comments_over_time.get("values"):
        charts.append(
            _chart(
                seed="comments_over_time",
                chart_type="area",
                title="Comments Over Time",
                subtitle=f"Comment volume by {comments_over_time.get('frequency', 'period')}",
                x=comments_over_time.get("labels", []),
                y=comments_over_time.get("values", []),
                color="#38bdf8",
                section="time",
            )
        )

    distributions = dataset.get("engagement_distributions", {})
    for metric, item in distributions.items():
        histogram = item.get("histogram", {})
        if not histogram.get("counts"):
            continue
        charts.append(
            _chart(
                seed=f"{metric}_distribution",
                chart_type="histogram",
                title=f"{_prettify(metric)} Distribution",
                subtitle=f"Distribution of {_prettify(item.get('column', metric))}",
                x=histogram.get("labels", []),
                y=histogram.get("counts", []),
                color="#34d399" if metric == "likes" else "#fbbf24",
                section="engagement",
            )
        )

    top_geo = dataset.get("top_geo", {})
    geo_items = top_geo.get("items", [])
    if geo_items:
        labels = [entry["location"] for entry in geo_items]
        values = [int(entry["comments"]) for entry in geo_items]
        charts.append(
            _chart(
                seed="top_geo",
                chart_type="bar",
                title="Top Locations",
                subtitle=f"Grouped by {_prettify(top_geo.get('geo_col', 'location'))}",
                x=labels,
                y=values,
                data=[
                    {"label": label, "value": value, "color": CHART_PALETTE[i % len(CHART_PALETTE)]}
                    for i, (label, value) in enumerate(zip(labels, values))
                ],
                section="geo",
            )
        )

    return charts


def _primary_text_charts(col: str, item: dict) -> list[dict]:
    charts = []
    sentiment = item.get("sentiment_distribution")
    if sentiment:
        counts = sentiment.get("counts", {})
        labels = [label for label in ("positive", "neutral", "negative") if label in counts]
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
                seed=f"primary_sentiment_{col}",
                chart_type="donut",
                title="Sentiment Distribution",
                subtitle=f"Primary text: {_prettify(col)}",
                x=[label.capitalize() for label in labels],
                y=values,
                data=data,
                section="key",
            )
        )

    keywords = item.get("keywords", [])
    if keywords:
        top = keywords[:12]
        charts.append(
            _chart(
                seed=f"primary_keywords_{col}",
                chart_type="horizontal_bar",
                title="Top Keywords",
                subtitle=f"Primary text: {_prettify(col)}",
                x=[entry["word"] for entry in top],
                y=[int(entry["count"]) for entry in top],
                data=[
                    {"label": entry["word"], "value": int(entry["count"])}
                    for entry in top
                ],
                color="#4f6ef7",
                section="content",
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


def _dedupe_charts(charts: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for chart in charts:
        key = (chart.get("section", ""), chart.get("title", ""), tuple(chart.get("x", [])[:5]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(chart)
    return unique


def _prioritize_charts(charts: list[dict]) -> list[dict]:
    ordered = sorted(
        enumerate(charts),
        key=lambda item: (
            SECTION_PRIORITY.get(item[1].get("section", "other"), SECTION_PRIORITY["other"]),
            item[0],
        ),
    )
    selected: list[dict] = []
    section_counts: dict[str, int] = {}
    for _, chart in ordered:
        section = chart.get("section", "other")
        if section_counts.get(section, 0) >= MAX_CHARTS_PER_SECTION:
            continue
        selected.append(chart)
        section_counts[section] = section_counts.get(section, 0) + 1
        if len(selected) >= MAX_CHARTS:
            break
    return selected


def _prettify(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()
