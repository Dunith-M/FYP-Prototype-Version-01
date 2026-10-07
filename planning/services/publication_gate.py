"""Server-side release gate between independent checking and action derivation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

from planning.models import DecisionOption, OptimizationResult
from planning.services.plan_identity import build_selected_plan_fingerprint, synthetic_constraint_signature
from planning.services.synthetic_state import load_synthetic_options


USABLE_SOLVER_STATUSES = frozenset(
    {OptimizationResult.SolverStatus.OPTIMAL, OptimizationResult.SolverStatus.FEASIBLE}
)


class PublicationGateBlocked(PermissionError):
    """Raised when a result is not permitted to reach the action engine."""


@dataclass(frozen=True)
class GateResult:
    """Read-only service-level decision about action-generation eligibility."""

    optimization_result_id: int
    allowed: bool
    code: str
    reasons: tuple[str, ...]
    plan_fingerprint: str | None = None


def _blocked(result: OptimizationResult, code: str, reason: str, *, fingerprint: str | None = None) -> GateResult:
    return GateResult(result.pk, False, code, (reason,), fingerprint)


def evaluate_publication_gate(
    optimization_result_id: int,
    *,
    config_path: Path | None = None,
) -> GateResult:
    """Allow only a usable solver result checked against this exact saved plan.

    The gate is intentionally read-only: it neither recomputes feasibility nor
    changes selections, solver state, checker state, or recommendations.
    """
    result = OptimizationResult.objects.get(pk=optimization_result_id)
    if result.solver_status not in USABLE_SOLVER_STATUSES:
        return _blocked(result, "PUB-GATE-001", f"Solver status {result.solver_status} is not usable.")

    selected_options = tuple(result.selected_options.select_related("branch", "product").all())
    if not selected_options:
        return _blocked(result, "PUB-GATE-006", "No persisted selected DecisionOptions exist for this usable solver result.")
    if any(option.parameter_source != DecisionOption.ParameterSource.SYNTHETIC_DEMO for option in selected_options):
        return _blocked(
            result,
            "PUB-GATE-007",
            "Selected DecisionOptions are not explicitly marked SYNTHETIC_DEMO.",
        )

    if result.feasibility_status == OptimizationResult.FeasibilityStatus.NOT_RUN:
        return _blocked(result, "PUB-GATE-002", "Independent checker has not completed for this result.")
    if result.feasibility_status != OptimizationResult.FeasibilityStatus.PASS:
        return _blocked(result, "PUB-GATE-003", f"Independent checker status is {result.feasibility_status}, not PASS.")

    checker_record = (result.solver_metadata or {}).get("independent_checker")
    if not isinstance(checker_record, dict):
        return _blocked(result, "PUB-GATE-002", "Checker evidence/identity is missing for this result.")
    if checker_record.get("status") != "PASS":
        return _blocked(result, "PUB-GATE-003", "Persisted checker evidence does not record PASS.")
    if checker_record.get("optimization_result_id") != result.pk:
        return _blocked(result, "PUB-GATE-005", "Checker evidence belongs to another OptimizationResult.")

    path = config_path or Path(settings.BASE_DIR) / "config" / "synthetic" / "options.yaml"
    options = load_synthetic_options(path)
    current_fingerprint = build_selected_plan_fingerprint(
        result,
        selected_options,
        constraint_signature=synthetic_constraint_signature(options),
    )
    if checker_record.get("plan_fingerprint") != current_fingerprint:
        return _blocked(result, "PUB-GATE-004", "Checker PASS is stale for the current selected plan or configuration.", fingerprint=current_fingerprint)

    return GateResult(
        optimization_result_id=result.pk,
        allowed=True,
        code="READY_FOR_ACTION_ENGINE",
        reasons=("Usable solver state, checker PASS, and exact selected-plan identity match.",),
        plan_fingerprint=current_fingerprint,
    )


def require_ready_for_action_generation(optimization_result_id: int) -> GateResult:
    """Return READY or raise, for mandatory use by future action publishing code."""
    gate = evaluate_publication_gate(optimization_result_id)
    if not gate.allowed:
        raise PublicationGateBlocked(f"{gate.code}: {' '.join(gate.reasons)}")
    return gate
