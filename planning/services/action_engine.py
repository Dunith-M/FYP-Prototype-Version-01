"""Derive six actions, or safely withhold an action for missing decision state."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from django.conf import settings

from planning.models import DecisionOption, OptimizationResult, Recommendation
from planning.services.publication_gate import require_ready_for_action_generation
from planning.services.synthetic_state import load_synthetic_options


class ActionDerivationError(ValueError):
    """Raised when an excluded item reaches the action engine unexpectedly."""


class WithheldReason:
    """Stable machine-readable reasons for an unavailable action decision."""

    MISSING_LISTING_STATE = "MISSING_LISTING_STATE"
    MISSING_PROPOSED_STATE = "MISSING_PROPOSED_STATE"
    MISSING_REFERENCE_STATE = "MISSING_REFERENCE_STATE"
    MISSING_PROPOSED_TARGET = "MISSING_PROPOSED_TARGET"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNKNOWN_ELIGIBILITY = "UNKNOWN_ELIGIBILITY"
    INVALID_STATE = "INVALID_STATE"


@dataclass(frozen=True)
class ActionDecision:
    """One action decision; WITHHELD is a status, never an action label."""

    action: str | None
    status: str
    reason_code: str | None = None

    def __post_init__(self) -> None:
        actions = {value for value, _label in Recommendation.Action.choices}
        if self.status == Recommendation.Status.WITHHELD:
            if self.action is not None or not self.reason_code:
                raise ValueError("WITHHELD decisions require no action and one reason code.")
            return
        if self.status == Recommendation.Status.READY:
            if self.action not in actions or self.reason_code is not None:
                raise ValueError("READY decisions require exactly one of the six actions.")
            return
        raise ValueError(f"Unsupported action-decision status: {self.status!r}.")


@dataclass(frozen=True)
class DerivedAction:
    """Read-only in-memory interpretation of one exact selected DecisionOption."""

    decision_option_id: int
    branch_code: str
    canonical_product_id: str
    l0: bool | None
    q0: int | None
    l1: bool | None
    q1: int | None
    delta: int | Decimal
    action: str | None
    status: str
    reason_code: str | None


def _is_nonnegative_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        return False
    if isinstance(value, Decimal) and not value.is_finite():
        return False
    return value >= 0


def _is_positive_number(value: object) -> bool:
    return _is_nonnegative_number(value) and value > 0


def _withheld(reason_code: str) -> ActionDecision:
    return ActionDecision(
        action=None,
        status=Recommendation.Status.WITHHELD,
        reason_code=reason_code,
    )


def _ready(action: str) -> ActionDecision:
    return ActionDecision(action=action, status=Recommendation.Status.READY)


def load_action_delta(config_path: Path | None = None) -> int | Decimal:
    """Load the visible synthetic-demo delta; it is never a hard-coded policy."""
    path = config_path or Path(settings.BASE_DIR) / "config" / "synthetic" / "options.yaml"
    options = load_synthetic_options(path)
    if options.get("mode") != "RESEARCH_SYNTHETIC":
        raise ActionDerivationError("Action derivation requires RESEARCH_SYNTHETIC configuration.")
    delta = options.get("delta")
    if not _is_nonnegative_number(delta):
        raise ActionDerivationError("Configured delta must be a finite non-negative integer or Decimal.")
    return delta


def derive_action(
    *,
    l0: object,
    l1: object,
    q0: object,
    q1: object,
    delta: object,
    evidence_available: bool | None = True,
    eligibility: bool | None = True,
    assessed: bool | None = True,
) -> ActionDecision:
    """Interpret one selected state without guessing missing evidence or targets.

    Listing transitions are evaluated before target comparison. The function
    does not read a database, use an objective/forecast score, or call CP-SAT.
    ``q0`` is required only for the listed-to-listed quantity comparison.
    """
    if l0 is None:
        return _withheld(WithheldReason.MISSING_LISTING_STATE)
    if not isinstance(l0, bool):
        return _withheld(WithheldReason.INVALID_STATE)
    if l1 is None:
        return _withheld(WithheldReason.MISSING_PROPOSED_STATE)
    if not isinstance(l1, bool):
        return _withheld(WithheldReason.INVALID_STATE)
    if q1 is None:
        return _withheld(WithheldReason.MISSING_PROPOSED_TARGET)
    if not _is_nonnegative_number(q1) or not _is_nonnegative_number(delta):
        return _withheld(WithheldReason.INVALID_STATE)
    if not l1 and q1 != 0:
        return _withheld(WithheldReason.INVALID_STATE)
    if l1 and not _is_positive_number(q1):
        return _withheld(WithheldReason.INVALID_STATE)
    if not l0 and q0 is not None:
        return _withheld(WithheldReason.INVALID_STATE)
    if evidence_available is not True:
        return _withheld(WithheldReason.INSUFFICIENT_EVIDENCE)

    # Track-B candidates do not need q0, but they must be known eligible and assessed.
    if not l0:
        if eligibility is None:
            return _withheld(WithheldReason.UNKNOWN_ELIGIBILITY)
        if eligibility is False:
            raise ActionDerivationError("Excluded/ineligible candidates must not reach action derivation.")
        if assessed is not True:
            return _withheld(WithheldReason.INSUFFICIENT_EVIDENCE)

    if l0 and l1:
        if q0 is None:
            return _withheld(WithheldReason.MISSING_REFERENCE_STATE)
        if not _is_positive_number(q0):
            return _withheld(WithheldReason.INVALID_STATE)

    # Resolve listing transitions before comparing quantities.
    if l0 and not l1:
        return _ready(Recommendation.Action.REMOVE)
    if not l0 and l1:
        return _ready(Recommendation.Action.INTRODUCE)
    if not l0 and not l1:
        return _ready(Recommendation.Action.DO_NOT_INTRODUCE)

    assert q0 is not None
    if q1 > q0 + delta:
        return _ready(Recommendation.Action.INCREASE)
    if q1 < q0 - delta:
        return _ready(Recommendation.Action.REDUCE)
    return _ready(Recommendation.Action.KEEP)


def derive_actions_for_optimization_result(optimization_result_id: int) -> tuple[DerivedAction, ...]:
    """Derive in-memory decisions only after the mandatory publication gate passes."""
    require_ready_for_action_generation(optimization_result_id)
    result = OptimizationResult.objects.get(pk=optimization_result_id)
    delta = load_action_delta()
    selected_options = result.selected_options.select_related("branch", "product").order_by(
        "branch__branch_code", "product__canonical_product_id", "pk"
    )
    decisions: list[DerivedAction] = []
    for option in selected_options:
        # A selected, validated Day-07 option is the assessed finite decision set.
        decision = derive_action(
            l0=option.l0,
            l1=option.l1,
            q0=option.q0,
            q1=option.q1,
            delta=delta,
            evidence_available=(
                option.evidence_score is not None
                and option.evidence_source != DecisionOption.EvidenceSource.UNASSIGNED
            ),
            eligibility=True,
            assessed=True,
        )
        decisions.append(
            DerivedAction(
                decision_option_id=option.pk,
                branch_code=option.branch.branch_code,
                canonical_product_id=option.product.canonical_product_id,
                l0=option.l0,
                q0=option.q0,
                l1=option.l1,
                q1=option.q1,
                delta=delta,
                action=decision.action,
                status=decision.status,
                reason_code=decision.reason_code,
            )
        )
    return tuple(decisions)
