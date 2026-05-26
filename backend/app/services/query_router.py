"""Central routing for dataset-grounded QA requests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.services.analytics_qa_service import AnalyticsQAService, analytics_rows
from app.services.dataset_relevance_service import DatasetRelevanceService
from app.services.query_intent_service import QueryIntentClassifier
from app.services.query_complexity_service import QueryComplexityService
from app.services.retrieval_context import RetrievalContext
from app.services.retrieval_planner import RetrievalPlan, RetrievalPlanner
from app.services.structured_query_service import StructuredQueryService, StructuredResult
from app.core.config import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutedQuery:
    plan: RetrievalPlan | None
    retrieval_plan: dict[str, Any]
    structured: StructuredResult | None = None
    analytics: dict[str, Any] | None = None
    rows: list[dict[str, Any]] | None = None
    stop: bool = False


class QueryRouter:
    """Own the non-semantic routing decisions for one QA request."""

    def __init__(self) -> None:
        self.intent_classifier = QueryIntentClassifier()
        self.relevance_service = DatasetRelevanceService()
        self.retrieval_planner = RetrievalPlanner()
        self.complexity_service = QueryComplexityService()
        self.analytics_service = AnalyticsQAService()
        self.structured_query_service = StructuredQueryService()

    def route(self, context: RetrievalContext, question: str, requested_top_k: int) -> RoutedQuery:
        relevance = self.relevance_service.assess(context, question)
        if not relevance.is_related:
            logger.info(
                "QA blocked as out-of-scope dataset_id=%s confidence=%.2f rationale=%s",
                context.dataset_id,
                relevance.confidence,
                relevance.rationale,
            )
            return RoutedQuery(
                plan=None,
                stop=True,
                retrieval_plan={
                    "intent": "dataset_relevance",
                    "strategy": "guardrail",
                    "top_k": 0,
                    "rationale": relevance.rationale,
                    "classifier_confidence": relevance.confidence,
                    "matched_signals": relevance.matched_signals,
                    "supported_topics": relevance.supported_topics,
                },
            )

        structured = self.structured_query_service.answer(context, question, requested_top_k)
        if structured:
            logger.info("QA answered from structured dataframe dataset_id=%s plan=%s", context.dataset_id, structured.plan)
            return RoutedQuery(
                plan=None,
                stop=True,
                structured=structured,
                retrieval_plan={
                    "intent": "structured_query",
                    "strategy": structured.plan.get("strategy", "dataframe"),
                    "top_k": requested_top_k,
                    "rationale": "Exact lookup/filter question answered directly from the cleaned dataframe.",
                    "structured_plan": structured.plan,
                },
            )

        intent = self.intent_classifier.classify(question)
        complexity = self.complexity_service.assess(question, intent.intent)
        effective_top_k = _effective_top_k(requested_top_k, intent.intent, complexity.level)
        plan = self.retrieval_planner.plan(intent, effective_top_k, complexity.level)
        analytics = self.analytics_service.build_context(context, question, plan.intent) if plan.use_analytics else None
        rows = analytics_rows(analytics) if analytics else []
        logger.info(
            "QA retrieval plan dataset_id=%s intent=%s strategy=%s top_k=%s rationale=%s",
            context.dataset_id,
            plan.intent,
            plan.strategy,
            plan.top_k,
            plan.rationale,
        )
        return RoutedQuery(
            plan=plan,
            analytics=analytics,
            rows=rows,
            retrieval_plan={
                "intent": plan.intent,
                "strategy": plan.strategy,
                "top_k": plan.top_k,
                "rationale": plan.rationale,
                "classifier_confidence": intent.confidence,
                "classifier_rationale": intent.rationale,
                "requested_top_k": requested_top_k,
                "effective_top_k": plan.top_k,
                "query_complexity": complexity.level,
                "query_complexity_score": complexity.score,
                "query_complexity_rationale": complexity.rationale,
            },
        )


def _effective_top_k(requested_top_k: int, intent: str, complexity: str) -> int:
    bounded = max(1, min(requested_top_k, 10))
    if not settings.ADAPTIVE_TOP_K_ENABLED:
        return bounded

    min_k = max(1, min(settings.ADAPTIVE_TOP_K_MIN, 10))
    max_k = max(min_k, min(settings.ADAPTIVE_TOP_K_MAX, 10))
    if complexity == "simple" and intent in {"factual", "structured_query"}:
        return min(max(min_k, 3), bounded)
    if complexity == "complex" or intent in {"trend", "comparison", "summarization"}:
        return min(max_k, max(bounded, 8))
    return min(max_k, max(min_k, bounded))
