"""Deterministic, read-only explanations for checked Day-08 selected states."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from django.conf import settings

from analytics.models import CandidateResult
from planning.models import DecisionOption, OptimizationResult, Recommendation
from planning.services.action_engine import DerivedAction, derive_actions_for_optimization_result
from planning.services.synthetic_state import load_synthetic_options


SYNTHETIC_STATUS_TEXT = "SYNTHETIC DECISION DEMO - not operationally approved."


@dataclass(frozen=True)
class ExplanationReason:
    """One short, stored-evidence-based explanation fact."""

    code: str
    text: str


@dataclass(frozen=True)
class ConstraintContext:
    """Read-only branch facts from the selected plan and active synthetic config."""

    selected_listed_skus: int
    max_skus_per_branch: int | None
    selected_introductions: int
    max_introductions_per_branch: int | None
    checker_passed: bool


@dataclass(frozen=True)
class ExplanationResult:
    """Ordered deterministic explanation text for one selected decision state."""

    decision_option_id: int
    action: str | None
    action_status: str
    reasons: tuple[ExplanationReason, ...]
    status_text: str

    @property
    def reason_texts(self) -> tuple[str, ...]:
        return tuple(reason.text for reason in self.reasons)


WITHHELD_TEXT = {
    "MISSING_LISTING_STATE": "No action was generated because the required listing state L0 is unavailable.",
    "MISSING_PROPOSED_STATE": "No action was generated because the proposed listing state L1 is unavailable.",
    "MISSING_REFERENCE_STATE": "No action was generated because the required reference target q0 is unavailable.",
    "MISSING_PROPOSED_TARGET": "No action was generated because the proposed target q1 is unavailable.",
    "INSUFFICIENT_EVIDENCE": "No action was generated because the required evidence is insufficient.",
    "UNKNOWN_ELIGIBILITY": "No action was generated because candidate eligibility is not confirmed.",
    "INVALID_STATE": "No action was generated because the selected decision state is invalid.",
}


def ordinal(value: int) -> str:
    """Format a positive rank without recomputing any candidate ordering."""
    suffix = "th" if 10 <= value % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _number(value: object) -> str:
    """Stable concise formatting for stored numeric evidence scores."""
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _selected_state_reason(option: object, action: str) -> ExplanationReason:
    q0 = getattr(option, "q0", None)
    q1 = getattr(option, "q1", None)
    templates = {
        Recommendation.Action.INCREASE: ("TARGET_INCREASE", f"Selected target increases from {q0} to {q1} units."),
        Recommendation.Action.KEEP: ("TARGET_KEEP", f"Selected target remains at {q1} units."),
        Recommendation.Action.REDUCE: ("TARGET_REDUCE", f"Selected target decreases from {q0} to {q1} units."),
        Recommendation.Action.REMOVE: ("SELECTED_REMOVE", "Selected option changes the product from listed to unlisted with target 0."),
        Recommendation.Action.INTRODUCE: ("SELECTED_INTRODUCE", f"Selected option introduces the product with a target of {q1} unit{'s' if q1 != 1 else ''}."),
        Recommendation.Action.DO_NOT_INTRODUCE: ("SELECTED_UNLISTED", "Selected feasible option keeps the candidate unlisted in this scenario."),
    }
    code, text = templates[action]
    return ExplanationReason(code=code, text=text)


def _track_a_reasons(option: object) -> list[ExplanationReason]:
    score = getattr(option, "evidence_score", None)
    if score is None:
        return [ExplanationReason("HISTORICAL_EVIDENCE_UNAVAILABLE", "Stored historical evidence is unavailable for this selected option.")]
    return [
        ExplanationReason(
            "HISTORICAL_LAG_1_EVIDENCE",
            f"Historical lag-1 normalized evidence score is {_number(score)}.",
        )
    ]


def _track_b_reasons(candidate: object | None, branch_code: str) -> list[ExplanationReason]:
    if candidate is None:
        return [ExplanationReason("CANDIDATE_EVIDENCE_UNAVAILABLE", "Stored candidate evidence is unavailable for this selected option.")]

    reasons: list[ExplanationReason] = []
    rank = getattr(candidate, "rank", None)
    if isinstance(rank, int) and rank > 0:
        reasons.append(
            ExplanationReason(
                "CANDIDATE_RANK",
                f"Ranked {ordinal(rank)} among research-shortlisted candidates for {branch_code}.",
            )
        )

    for field, code, label in (
        ("peer_score", "PEER_PRODUCT_SCORE", "Peer-product research score"),
        ("family_score", "BRANCH_FAMILY_SCORE", "Branch-family research score"),
        ("similar_branch_score", "SIMILAR_BRANCH_SCORE", "Comparable-branch research score"),
    ):
        value = getattr(candidate, field, None)
        if value is not None:
            reasons.append(ExplanationReason(code, f"{label} is {_number(value)}."))
            break
    else:
        score = getattr(candidate, "candidate_score", None)
        if score is not None:
            reasons.append(ExplanationReason("CANDIDATE_SCORE", f"Candidate research score is {_number(score)}."))

    evidence_class = getattr(candidate, "evidence_class", None)
    if evidence_class:
        reasons.append(ExplanationReason("CANDIDATE_EVIDENCE_CLASS", f"Candidate evidence class is {evidence_class}."))
    return reasons or [ExplanationReason("CANDIDATE_EVIDENCE_UNAVAILABLE", "Stored candidate evidence is unavailable for this selected option.")]


def _constraint_reasons(context: ConstraintContext | None, *, is_candidate: bool) -> list[ExplanationReason]:
    if context is None:
        return []
    reasons: list[ExplanationReason] = []
    if is_candidate and context.max_introductions_per_branch is not None:
        reasons.append(
            ExplanationReason(
                "INTRODUCTION_LIMIT_CONTEXT",
                f"The selected plan uses {context.selected_introductions} of the synthetic {context.max_introductions_per_branch} introduction slots.",
            )
        )
    elif not is_candidate and context.max_skus_per_branch is not None:
        reasons.append(
            ExplanationReason(
                "SKU_LIMIT_CONTEXT",
                f"The selected plan uses {context.selected_listed_skus} of the synthetic {context.max_skus_per_branch} listed-SKU capacity.",
            )
        )
    if context.checker_passed:
        reasons.append(
            ExplanationReason("CHECKER_PASS", "The complete selected plan passed the independent feasibility checker.")
        )
    return reasons


def generate_explanation(
    *,
    selected_option: object,
    action_result: object,
    candidate_result: object | None = None,
    constraint_context: ConstraintContext | None = None,
) -> ExplanationResult:
    """Format 2-4 factual reasons from persisted evidence and rule outcomes.

    This pure function has no database writes, solver calls, raw-data reads, or
    score recalculation. A WITHHELD result deliberately has only its truthful
    availability reason instead of invented positive recommendation evidence.
    """
    option_id = getattr(selected_option, "pk", None)
    action = getattr(action_result, "action", None)
    action_status = getattr(action_result, "status", None)
    if action_status == Recommendation.Status.WITHHELD:
        reason_code = getattr(action_result, "reason_code", None) or "INVALID_STATE"
        return ExplanationResult(
            decision_option_id=option_id,
            action=None,
            action_status=Recommendation.Status.WITHHELD,
            reasons=(ExplanationReason(reason_code, WITHHELD_TEXT.get(reason_code, WITHHELD_TEXT["INVALID_STATE"])),),
            status_text=SYNTHETIC_STATUS_TEXT,
        )

    if action_status != Recommendation.Status.READY or action not in {value for value, _label in Recommendation.Action.choices}:
        raise ValueError("Explanation generation requires a READY six-action result or a WITHHELD result.")

    is_candidate = getattr(selected_option, "evidence_source", None) == DecisionOption.EvidenceSource.TRACK_B_CANDIDATE_SCORE
    branch_code = getattr(getattr(selected_option, "branch", None), "branch_code", "this branch")
    reasons: list[ExplanationReason] = [_selected_state_reason(selected_option, action)]
    reasons.extend(_track_b_reasons(candidate_result, branch_code) if is_candidate else _track_a_reasons(selected_option))
    for reason in _constraint_reasons(constraint_context, is_candidate=is_candidate):
        if len(reasons) >= 4:
            break
        reasons.append(reason)

    return ExplanationResult(
        decision_option_id=option_id,
        action=action,
        action_status=Recommendation.Status.READY,
        reasons=tuple(reasons[:4]),
        status_text=SYNTHETIC_STATUS_TEXT,
    )


def _constraint_contexts(result: OptimizationResult, options: Iterable[DecisionOption]) -> dict[int, ConstraintContext]:
    """Build branch facts without rechecking, solving, or changing the selected plan."""
    path = Path(settings.BASE_DIR) / "config" / "synthetic" / "options.yaml"
    config = load_synthetic_options(path)
    constraints = config.get("constraints", {})
    grouped: dict[int, list[DecisionOption]] = defaultdict(list)
    for option in options:
        grouped[option.branch_id].append(option)
    contexts: dict[int, ConstraintContext] = {}
    for branch_id, branch_options in grouped.items():
        contexts[branch_id] = ConstraintContext(
            selected_listed_skus=sum(option.l1 for option in branch_options),
            max_skus_per_branch=constraints.get("max_skus_per_branch"),
            selected_introductions=sum(not option.l0 and option.l1 for option in branch_options),
            max_introductions_per_branch=constraints.get("max_introductions_per_branch"),
            checker_passed=result.feasibility_status == OptimizationResult.FeasibilityStatus.PASS,
        )
    return contexts


def _candidate_evidence_by_option(options: Iterable[DecisionOption]) -> dict[int, CandidateResult]:
    """Return only unambiguous persisted Day-07 candidate evidence by option ID."""
    candidate_options = [
        option
        for option in options
        if option.evidence_source == DecisionOption.EvidenceSource.TRACK_B_CANDIDATE_SCORE
    ]
    if not candidate_options:
        return {}
    candidate_rows = CandidateResult.objects.filter(
        planning_month__in={option.planning_month for option in candidate_options},
        branch_id__in={option.branch_id for option in candidate_options},
        product_id__in={option.product_id for option in candidate_options},
    ).select_related("branch", "product")
    grouped: dict[tuple[int, int, object], list[CandidateResult]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(row.branch_id, row.product_id, row.planning_month)].append(row)
    evidence: dict[int, CandidateResult] = {}
    for option in candidate_options:
        matches = grouped[(option.branch_id, option.product_id, option.planning_month)]
        if len(matches) == 1:
            evidence[option.pk] = matches[0]
    return evidence


def generate_explanations_for_optimization_result(optimization_result_id: int) -> tuple[ExplanationResult, ...]:
    """Generate read-only explanations after action orchestration passes its gate."""
    derived_actions = derive_actions_for_optimization_result(optimization_result_id)
    result = OptimizationResult.objects.get(pk=optimization_result_id)
    options = list(
        result.selected_options.select_related("branch", "product").order_by(
            "branch__branch_code", "product__canonical_product_id", "pk"
        )
    )
    actions_by_option = {action.decision_option_id: action for action in derived_actions}
    contexts = _constraint_contexts(result, options)
    candidate_evidence = _candidate_evidence_by_option(options)
    return tuple(
        generate_explanation(
            selected_option=option,
            action_result=actions_by_option[option.pk],
            candidate_result=candidate_evidence.get(option.pk),
            constraint_context=contexts[option.branch_id],
        )
        for option in options
    )
