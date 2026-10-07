"""Transactional, gate-protected persistence of Day-08 Recommendation outputs."""

from __future__ import annotations

from dataclasses import dataclass

from django.db import transaction

from planning.models import DecisionOption, OptimizationResult, Recommendation
from planning.services.explanations import ExplanationResult, generate_explanations_for_optimization_result
from planning.services.publication_gate import require_ready_for_action_generation


RESEARCH_MODE = "RESEARCH_SYNTHETIC"


class RecommendationPersistenceError(ValueError):
    """Raised when a checked selected plan cannot be durably represented safely."""


@dataclass(frozen=True)
class RecommendationPersistenceResult:
    """Idempotent result for one exact OptimizationResult recommendation batch."""

    optimization_result_id: int
    created_count: int
    existing_count: int
    recommendation_ids: tuple[int, ...]


def _reason_payload(explanation: ExplanationResult, option: DecisionOption) -> list[dict[str, str]]:
    """Store ordered deterministic reasons plus durable synthetic-mode context."""
    return [
        *[{"code": reason.code, "text": reason.text} for reason in explanation.reasons],
        {
            "code": RESEARCH_MODE,
            "text": explanation.status_text,
            "parameter_source": option.parameter_source,
        },
    ]


def _validate_selected_options(result: OptimizationResult, options: tuple[DecisionOption, ...]) -> None:
    """Refuse missing, foreign, duplicate, or inconsistent persisted selections."""
    if not options:
        raise RecommendationPersistenceError("A usable result needs persisted selected DecisionOptions.")
    option_ids = [option.pk for option in options]
    if len(option_ids) != len(set(option_ids)):
        raise RecommendationPersistenceError("Selected DecisionOption IDs must be unique.")
    pairs = [(option.branch_id, option.product_id) for option in options]
    if len(pairs) != len(set(pairs)):
        raise RecommendationPersistenceError("Selected DecisionOptions must contain exactly one branch-product pair each.")
    for option in options:
        if (
            option.planning_month != result.planning_month
            or option.scenario_version != result.scenario_version
            or not result.selected_options.filter(pk=option.pk).exists()
        ):
            raise RecommendationPersistenceError(f"DecisionOption {option.pk} does not belong to this exact OptimizationResult scenario.")


def _save_new_recommendation(recommendation: Recommendation) -> Recommendation:
    """Small seam for controlled transaction tests; validates before the write."""
    recommendation.full_clean()
    recommendation.save()
    return recommendation


def _same_persisted_output(
    recommendation: Recommendation,
    *,
    option: DecisionOption,
    explanation: ExplanationResult,
    reasons: list[dict[str, str]],
) -> bool:
    """Keep completed result rows immutable on idempotent re-execution."""
    return (
        recommendation.selected_option_id == option.pk
        and recommendation.branch_id == option.branch_id
        and recommendation.product_id == option.product_id
        and recommendation.action == explanation.action
        and recommendation.status == explanation.action_status
        and recommendation.reasons == reasons
    )


def create_recommendations_for_result(optimization_result_id: int) -> RecommendationPersistenceResult:
    """Persist one output per checked selected option, or raise without partial rows.

    The server-side Step-13 publication gate is required here rather than being
    trusted to callers.  Action and explanation services remain the single
    source of their respective results; this service only persists them.
    """
    require_ready_for_action_generation(optimization_result_id)

    with transaction.atomic():
        result = OptimizationResult.objects.select_for_update().get(pk=optimization_result_id)
        # Re-evaluate after obtaining the batch lock so unsafe transitions cannot publish.
        require_ready_for_action_generation(result.pk)
        options = tuple(
            result.selected_options.select_related("branch", "product").order_by(
                "branch__branch_code", "product__canonical_product_id", "pk"
            )
        )
        _validate_selected_options(result, options)

        explanations = generate_explanations_for_optimization_result(result.pk)
        explanations_by_option = {explanation.decision_option_id: explanation for explanation in explanations}
        if set(explanations_by_option) != {option.pk for option in options}:
            raise RecommendationPersistenceError("Action/explanation output does not cover the exact persisted selected option set.")

        created_count = 0
        existing_count = 0
        persisted: list[Recommendation] = []
        for option in options:
            explanation = explanations_by_option[option.pk]
            reasons = _reason_payload(explanation, option)
            existing = Recommendation.objects.select_for_update().filter(
                optimization_result=result,
                branch=option.branch,
                product=option.product,
            ).first()
            if existing is not None:
                if not _same_persisted_output(existing, option=option, explanation=explanation, reasons=reasons):
                    raise RecommendationPersistenceError(
                        f"Recommendation {existing.pk} already exists with different immutable output for DecisionOption {option.pk}."
                    )
                existing_count += 1
                persisted.append(existing)
                continue

            persisted.append(
                _save_new_recommendation(
                    Recommendation(
                        optimization_result=result,
                        selected_option=option,
                        branch=option.branch,
                        product=option.product,
                        action=explanation.action,
                        status=explanation.action_status,
                        reasons=reasons,
                    )
                )
            )
            created_count += 1

        return RecommendationPersistenceResult(
            optimization_result_id=result.pk,
            created_count=created_count,
            existing_count=existing_count,
            recommendation_ids=tuple(recommendation.pk for recommendation in persisted),
        )
