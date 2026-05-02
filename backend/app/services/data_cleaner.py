"""Compatibility wrapper for the Phase 2 data processor."""

from __future__ import annotations

import pandas as pd

from app.services.data_processor import clean_dataframe as _clean_dataframe
from app.services.schema_detector import ColumnType


def clean_dataframe(
    df: pd.DataFrame,
    text_columns: list[str] | dict[str, ColumnType],
    **_: object,
) -> tuple[pd.DataFrame, dict]:
    """Clean a DataFrame using either the new schema map or old text-column list."""
    if isinstance(text_columns, dict):
        schema = text_columns
    else:
        schema = {col: "text" for col in text_columns}
        for col in df.columns:
            schema.setdefault(col, "categorical")
    return _clean_dataframe(df, schema)
