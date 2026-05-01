"""
Tests for the ingestion pipeline.
Run: python -m pytest tests/ -v
"""

import pandas as pd
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.config import IngestionConfig
from ingestion.validator import validate, _check_nulls, _check_empty_strings, _check_duplicates
from ingestion.column_detector import _heuristic_detect
from ingestion.models import Severity


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return IngestionConfig(text_column=None)


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "review_id": [1, 2, 3],
        "text": ["Great product!", "Terrible experience.", "Average quality."],
    })


@pytest.fixture
def df_with_nulls():
    # 2 nulls out of 10 rows = 20%, below the 30% threshold → WARNING not ERROR
    return pd.DataFrame({
        "text": [
            "Good product", None, "Bad experience", None, "Okay",
            "Great!", "Decent", "Not bad", "Loved it", "Meh",
        ],
    })


@pytest.fixture
def df_with_empties():
    return pd.DataFrame({
        "text": ["Hello world", "   ", "\t\n", "Valid text"],
    })


@pytest.fixture
def df_with_duplicates():
    return pd.DataFrame({
        "text": ["Great product", "great product", "Bad service", "Bad service"],
    })


# ── Null checks ───────────────────────────────────────────────────────────────

class TestNullChecks:
    def test_no_nulls(self, clean_df, config):
        issues = _check_nulls(clean_df["text"], config)
        assert len(issues) == 0

    def test_nulls_below_threshold_are_warnings(self, df_with_nulls, config):
        issues = _check_nulls(df_with_nulls["text"], config)
        assert len(issues) == 1
        assert issues[0].severity == Severity.WARNING
        assert issues[0].count == 2

    def test_nulls_above_threshold_are_errors(self, config):
        df = pd.DataFrame({"text": [None] * 8 + ["valid"] * 2})
        issues = _check_nulls(df["text"], config)
        assert issues[0].severity == Severity.ERROR

    def test_null_row_indices_are_correct(self, df_with_nulls, config):
        issues = _check_nulls(df_with_nulls["text"], config)
        assert 1 in issues[0].row_indices
        assert 3 in issues[0].row_indices


# ── Empty string checks ───────────────────────────────────────────────────────

class TestEmptyStringChecks:
    def test_detects_whitespace_only(self, df_with_empties):
        issues = _check_empty_strings(df_with_empties["text"])
        assert len(issues) == 1
        assert issues[0].count == 2
        assert issues[0].category == "empty_text"

    def test_clean_series_no_issues(self, clean_df):
        issues = _check_empty_strings(clean_df["text"])
        assert len(issues) == 0


# ── Duplicate checks ──────────────────────────────────────────────────────────

class TestDuplicateChecks:
    def test_detects_duplicates(self, df_with_duplicates):
        issues = _check_duplicates(df_with_duplicates["text"])
        assert len(issues) == 1
        assert issues[0].count == 2  # one "great product" dupe + one "bad service" dupe

    def test_case_insensitive(self):
        df = pd.DataFrame({"text": ["Hello World", "hello world", "unique"]})
        issues = _check_duplicates(df["text"])
        assert issues[0].count == 1

    def test_no_duplicates(self, clean_df):
        issues = _check_duplicates(clean_df["text"])
        assert len(issues) == 0


# ── Heuristic column detection ────────────────────────────────────────────────

class TestHeuristicDetection:
    def test_detects_text_column(self):
        cols = ["id", "author", "text", "rating"]
        sample = {"id": [1], "author": ["A"], "text": ["hello"], "rating": [5]}
        result = _heuristic_detect(cols, sample)
        assert result is not None
        assert result.column_name == "text"
        assert result.confidence == "high"
        assert result.method == "local_pipeline"

    def test_detects_review_column(self):
        cols = ["review_id", "review_text", "score"]
        sample = {"review_id": [1], "review_text": ["Great"], "score": [5]}
        result = _heuristic_detect(cols, sample)
        assert result is not None
        assert result.column_name == "review_text"

    def test_no_match_returns_none(self):
        cols = ["id", "price", "quantity", "date"]
        sample = {"id": [1], "price": [9.99], "quantity": [2], "date": ["2024-01-01"]}
        result = _heuristic_detect(cols, sample)
        assert result is None

    def test_multiple_candidates_picks_longest(self):
        cols = ["comment", "text"]
        sample = {
            "comment": ["short"],
            "text": ["This is a much longer text that should win the selection"],
        }
        result = _heuristic_detect(cols, sample)
        assert result is not None
        assert result.column_name == "text"

    def test_detects_textual_column_without_keyword(self):
        cols = ["id", "stars", "verbatim"]
        sample = {
            "id": [1, 2, 3],
            "stars": [5, 2, 4],
            "verbatim": [
                "The checkout page was confusing and slow.",
                "Support answered quickly and solved my issue.",
                "I wish the mobile app had better filtering.",
            ],
        }
        result = _heuristic_detect(cols, sample)
        assert result is not None
        assert result.column_name == "verbatim"


# ── Full validation pipeline ──────────────────────────────────────────────────

class TestValidationPipeline:
    def test_clean_data_has_no_issues(self, clean_df, config):
        issues, stats = validate(clean_df, "text", config)
        assert all(i.severity != Severity.ERROR for i in issues)
        assert stats.total_rows == 3
        assert stats.clean_count == 3

    def test_stats_accuracy(self, df_with_nulls, config):
        issues, stats = validate(df_with_nulls, "text", config)
        assert stats.null_count == 2
        assert stats.null_ratio == pytest.approx(0.2, rel=0.01)

    def test_empty_rows_flagged(self, config):
        df = pd.DataFrame({"text": [None, None], "id": [None, None]})
        issues, _ = validate(df, "text", config)
        cats = [i.category for i in issues]
        assert "empty_row" in cats

    def test_too_short_flagged(self, config):
        df = pd.DataFrame({"text": ["Hi", "Good product", "OK"]})
        issues, _ = validate(df, "text", config)
        cats = [i.category for i in issues]
        assert "too_short" in cats
