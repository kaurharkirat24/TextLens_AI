"""Shared request-scoped context for retrieval services."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.core.config import settings
from app.models.schemas import DatasetMeta
from app.services.dataset_manager import get_dataset
from app.services.dataset_profile_service import DatasetProfileService
from app.services.semantic_dataset_service import SemanticDatasetError


@dataclass
class RetrievalContext:
    """Cache dataset state so one QA request does not reload the same artifacts repeatedly."""

    dataset_id: str
    meta: DatasetMeta
    analysis: dict[str, Any]
    _dataframe: pd.DataFrame | None = field(default=None, repr=False)
    _profile: dict[str, Any] | None = field(default=None, repr=False)

    @classmethod
    def load(cls, dataset_id: str) -> "RetrievalContext":
        meta = get_dataset(dataset_id)
        if not meta:
            raise SemanticDatasetError(f"Dataset '{dataset_id}' not found")
        if not meta.clean_csv_path or not os.path.exists(meta.clean_csv_path):
            raise SemanticDatasetError("Clean dataset not found. Run analysis before retrieval.")
        return cls(dataset_id=dataset_id, meta=meta, analysis=_load_analysis(meta, dataset_id))

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            self._dataframe = pd.read_csv(self.meta.clean_csv_path)
        return self._dataframe

    @property
    def profile(self) -> dict[str, Any]:
        if self._profile is None:
            service = DatasetProfileService()
            profile = service.load(self.dataset_id)
            if not profile:
                profile = service.build(self.dataset_id, self.analysis, self.dataframe)
                service.save(self.dataset_id, profile)
            self._profile = profile
        return self._profile

    @property
    def column_names(self) -> list[str]:
        profile_columns = self.profile.get("columns") or []
        names = [str(column.get("name", "")) for column in profile_columns if column.get("name")]
        if names:
            return names
        if self._dataframe is not None:
            return [str(column) for column in self._dataframe.columns]
        return list(self.schema_columns.keys())

    @property
    def column_roles(self) -> dict[str, Any]:
        roles = self.analysis.get("column_roles")
        return roles if isinstance(roles, dict) else {}

    @property
    def schema_columns(self) -> dict[str, str]:
        schema = self.analysis.get("schema") or {}
        columns = schema.get("columns") if isinstance(schema, dict) else {}
        if not isinstance(columns, dict):
            return {}
        return {str(column): str(kind) for column, kind in columns.items()}

    @property
    def primary_text_column(self) -> str:
        column = self.column_roles.get("primary_text") or self.meta.text_column or ""
        return column if column in self.dataframe.columns else ""


def _load_analysis(meta: DatasetMeta, dataset_id: str) -> dict[str, Any]:
    candidates = [
        getattr(meta, "analysis_path", "") or "",
        os.path.join(settings.OUTPUT_DIR, f"{dataset_id}_analysis.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                continue
    return {}
