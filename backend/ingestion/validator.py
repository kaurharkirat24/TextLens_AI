"""
Validation engine: runs all checks on the text column and returns
a list of ValidationIssue objects + a populated DatasetStats object.
"""

import numpy as np
import pandas as pd

from ingestion.config import IngestionConfig
from ingestion.models import DatasetStats, Severity, ValidationIssue


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_blank(val) -> bool:
    """True if val is NaN, None, or a whitespace-only string."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return str(val).strip() == ""


# ── individual checks ─────────────────────────────────────────────────────────

def _check_nulls(series: pd.Series, config: IngestionConfig) -> list[ValidationIssue]:
    issues = []
    null_mask = series.isna()
    null_indices = series.index[null_mask].tolist()

    if null_indices:
        null_ratio = len(null_indices) / len(series)
        severity = Severity.ERROR if null_ratio > config.max_null_ratio else Severity.WARNING
        issues.append(ValidationIssue(
            severity=severity,
            category="null_value",
            row_indices=null_indices,
            count=len(null_indices),
            message=(
                f"{len(null_indices)} null values found in text column "
                f"({null_ratio:.1%} of rows). "
                + ("Exceeds allowed threshold." if severity == Severity.ERROR else "Within threshold.")
            ),
        ))
    return issues


def _check_empty_strings(series: pd.Series) -> list[ValidationIssue]:
    """Catch non-null but whitespace-only values."""
    issues = []
    non_null = series.dropna()
    empty_mask = non_null.apply(lambda v: str(v).strip() == "")
    empty_indices = non_null.index[empty_mask].tolist()

    if empty_indices:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="empty_text",
            row_indices=empty_indices,
            count=len(empty_indices),
            message=f"{len(empty_indices)} rows contain whitespace-only text (non-null but effectively empty).",
        ))
    return issues


def _check_too_short(series: pd.Series, min_len: int) -> list[ValidationIssue]:
    issues = []
    valid = series.dropna()
    short_mask = valid.apply(lambda v: 0 < len(str(v).strip()) < min_len)
    short_indices = valid.index[short_mask].tolist()

    if short_indices:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="too_short",
            row_indices=short_indices,
            count=len(short_indices),
            message=f"{len(short_indices)} rows have text shorter than {min_len} characters after stripping.",
        ))
    return issues


def _check_too_long(series: pd.Series, max_len: int) -> list[ValidationIssue]:
    issues = []
    valid = series.dropna()
    long_mask = valid.apply(lambda v: len(str(v).strip()) > max_len)
    long_indices = valid.index[long_mask].tolist()

    if long_indices:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="too_long",
            row_indices=long_indices,
            count=len(long_indices),
            message=f"{len(long_indices)} rows exceed {max_len} characters — may contain malformed data.",
        ))
    return issues


def _check_duplicates(series: pd.Series) -> list[ValidationIssue]:
    issues = []
    valid = series.dropna()
    # Normalise for comparison: lowercase + strip
    normalised = valid.apply(lambda v: str(v).strip().lower())
    dup_mask = normalised.duplicated(keep="first")
    dup_indices = valid.index[dup_mask].tolist()

    if dup_indices:
        issues.append(ValidationIssue(
            severity=Severity.INFO,
            category="duplicate_text",
            row_indices=dup_indices,
            count=len(dup_indices),
            message=f"{len(dup_indices)} rows appear to be exact duplicates of an earlier row.",
        ))
    return issues


def _check_entirely_empty_rows(df: pd.DataFrame) -> list[ValidationIssue]:
    """Flag rows where EVERY column is null/empty."""
    issues = []
    all_null_mask = df.isnull().all(axis=1)
    all_null_indices = df.index[all_null_mask].tolist()

    if all_null_indices:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            category="empty_row",
            row_indices=all_null_indices,
            count=len(all_null_indices),
            message=f"{len(all_null_indices)} rows are entirely empty (all columns null).",
        ))
    return issues


# ── stats ─────────────────────────────────────────────────────────────────────

def _compute_stats(
    df: pd.DataFrame,
    series: pd.Series,
    text_column: str,
    issues: list[ValidationIssue],
    config: IngestionConfig,
) -> DatasetStats:
    null_count = series.isna().sum()
    null_ratio = null_count / len(series) if len(series) else 0.0

    non_null = series.dropna()
    empty_count = non_null.apply(lambda v: str(v).strip() == "").sum()
    too_short_count = non_null.apply(
        lambda v: 0 < len(str(v).strip()) < config.min_text_length
    ).sum()
    too_long_count = non_null.apply(
        lambda v: len(str(v).strip()) > config.max_text_length
    ).sum()

    dup_issues = [i for i in issues if i.category == "duplicate_text"]
    dup_count = dup_issues[0].count if dup_issues else 0

    usable = non_null.apply(lambda v: str(v).strip()).replace("", np.nan).dropna()
    lengths = usable.apply(len)

    sources = {}
    if "source" in df.columns:
        sources = df["source"].value_counts().to_dict()

    clean_count = len(usable)

    return DatasetStats(
        total_rows=len(df),
        total_columns=len(df.columns),
        text_column=text_column,
        null_count=int(null_count),
        null_ratio=round(float(null_ratio), 4),
        empty_count=int(empty_count),
        too_short_count=int(too_short_count),
        too_long_count=int(too_long_count),
        duplicate_count=int(dup_count),
        clean_count=int(clean_count),
        avg_text_length=round(float(lengths.mean()), 1) if len(lengths) else 0.0,
        median_text_length=round(float(lengths.median()), 1) if len(lengths) else 0.0,
        sources=sources,
    )


# ── main entry point ──────────────────────────────────────────────────────────

def validate(
    df: pd.DataFrame,
    text_column: str,
    config: IngestionConfig,
) -> tuple[list[ValidationIssue], DatasetStats]:
    """
    Run all validation checks on df[text_column].
    Returns (issues, stats).
    """
    series = df[text_column]
    issues: list[ValidationIssue] = []

    issues += _check_entirely_empty_rows(df)
    issues += _check_nulls(series, config)
    issues += _check_empty_strings(series)
    issues += _check_too_short(series, config.min_text_length)
    issues += _check_too_long(series, config.max_text_length)
    issues += _check_duplicates(series)

    stats = _compute_stats(df, series, text_column, issues, config)
    return issues, stats
