"""Map query intent to retrieval and answer strategy."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.query_intent_service import QueryIntent


@dataclass(frozen=True)
class RetrievalPlan:
    intent: str
    strategy: str
    top_k: int
    prompt_style: str
    use_semantic: bool
    use_analytics: bool
    rationale: str
    complexity: str = "medium"


class RetrievalPlanner:
    """Keep routing choices centralized and easy to tune."""

    def plan(self, intent: QueryIntent, effective_top_k: int, complexity: str = "medium") -> RetrievalPlan:
        top_k = max(1, min(effective_top_k, 10))

        if intent.intent == "aggregation":
            return RetrievalPlan(
                intent=intent.intent,
                strategy="analytics",
                top_k=min(top_k, 5),
                prompt_style="aggregation",
                use_semantic=False,
                use_analytics=True,
                rationale="Aggregate query should be answered from dataset-level counts and representative examples.",
                complexity=complexity,
            )
        if intent.intent in {"trend", "comparison"}:
            return RetrievalPlan(
                intent=intent.intent,
                strategy="hybrid",
                top_k=max(top_k, 8),
                prompt_style=intent.intent,
                use_semantic=True,
                use_analytics=True,
                rationale="Grouped query benefits from analytics plus semantic examples.",
                complexity=complexity,
            )
        if intent.intent == "summarization":
            return RetrievalPlan(
                intent=intent.intent,
                strategy="hybrid",
                top_k=max(top_k, 10),
                prompt_style="summary",
                use_semantic=True,
                use_analytics=True,
                rationale="Broad summary should combine aggregate facts with representative rows.",
                complexity=complexity,
            )
        if intent.intent == "factual":
            return RetrievalPlan(
                intent=intent.intent,
                strategy="semantic",
                top_k=min(top_k, 5),
                prompt_style="factual",
                use_semantic=True,
                use_analytics=False,
                rationale="Specific query should use concise semantic evidence.",
                complexity=complexity,
            )

        return RetrievalPlan(
            intent="semantic_exploration",
            strategy="semantic",
            top_k=max(top_k, 8),
            prompt_style="exploration",
            use_semantic=True,
            use_analytics=False,
            rationale="Exploratory query should retrieve more examples for pattern finding.",
            complexity=complexity,
        )
