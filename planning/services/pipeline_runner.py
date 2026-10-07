"""Orchestrate existing persisted Day-06--Day-08 research components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging

from analytics.models import CandidateResult
from analytics.services.forecast_service import ForecastArtifactsUnavailable, get_forecast_page_data
from catalog.models import Branch
from optimization.checker import CheckerInputError, check_and_persist_feasibility
from optimization.runner import SolverRunError, USABLE_SOLVER_STATUSES, run_and_persist_optimization
from planning.models import DecisionOption, OptimizationResult
from planning.modes import ScenarioMode
from planning.services.recommendation_service import (
    RecommendationPersistenceError,
    create_recommendations_for_result,
)
from planning.services.publication_gate import PublicationGateBlocked
from planning.services.run_provenance import RunProvenanceCaptureError, capture_run_provenance


logger = logging.getLogger(__name__)

HISTORICAL_MODE = ScenarioMode.HISTORICAL
SYNTHETIC_MODE = ScenarioMode.RESEARCH_SYNTHETIC


class PipelineRunError(RuntimeError):
    """Controlled failure that must not be represented as a completed run."""


@dataclass(frozen=True)
class PipelineRunResult:
    """Small, UI-safe summary of a traceable orchestration attempt."""

    mode: str
    planning_month: date
    branch_code: str
    candidate_count: int
    optimization_result_id: int | None = None
    run_version: str | None = None
    solver_status: str | None = None
    feasibility_status: str | None = None
    recommendation_count: int = 0
    reused_existing_result: bool = False

    @property
    def completed(self) -> bool:
        return (
            self.mode == SYNTHETIC_MODE
            and self.solver_status in USABLE_SOLVER_STATUSES
            and self.feasibility_status == OptimizationResult.FeasibilityStatus.PASS
        )


def _require_persisted_evidence(*, planning_month: date, branch: Branch) -> int:
    """Confirm frozen Day-06 and persisted Day-07 evidence without recalculating it."""

    try:
        get_forecast_page_data(branch=branch.branch_code, month=planning_month.isoformat())
    except ForecastArtifactsUnavailable as error:
        raise PipelineRunError(str(error)) from error

    candidate_count = CandidateResult.objects.filter(
        branch=branch,
        planning_month=planning_month,
    ).count()
    if not candidate_count:
        raise PipelineRunError(
            f"No persisted Day-07 candidate evidence is available for {branch.branch_code} "
            f"in {planning_month:%B %Y}."
        )
    return candidate_count


def _resolve_synthetic_scenario(*, planning_month: date, branch: Branch) -> str:
    """Find the one existing finite option scenario that exactly matches the context."""

    scenarios = list(
        DecisionOption.objects.filter(branch=branch, planning_month=planning_month)
        .order_by("scenario_version")
        .values_list("scenario_version", flat=True)
        .distinct()
    )
    if not scenarios:
        raise PipelineRunError(
            f"No persisted synthetic decision-option scenario is configured for "
            f"{branch.branch_code} in {planning_month:%B %Y}."
        )
    if len(scenarios) != 1:
        raise PipelineRunError(
            f"Multiple persisted synthetic scenarios exist for {branch.branch_code} "
            f"in {planning_month:%B %Y}; select an unambiguous scenario before running."
        )

    scenario_version = scenarios[0]
    scenario_options = DecisionOption.objects.filter(
        planning_month=planning_month,
        scenario_version=scenario_version,
    )
    if scenario_options.exclude(branch=branch).exists():
        raise PipelineRunError(
            "The persisted synthetic scenario contains another branch and cannot be "
            "safely used for this branch-specific run."
        )
    if scenario_options.exclude(
        parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO
    ).exists():
        raise PipelineRunError(
            "The persisted decision options are not explicitly marked SYNTHETIC_DEMO; "
            "the synthetic planning path is blocked."
        )
    return scenario_version


def _run_version(*, planning_month: date, branch: Branch, scenario_version: str) -> str:
    """Stable browser-run identity prevents refreshes from creating duplicate plans."""

    return f"day09_web_{branch.branch_code.lower()}_{planning_month:%Y%m}_{scenario_version}"


def _existing_result(*, planning_month: date, scenario_version: str, run_version: str):
    return OptimizationResult.objects.filter(
        planning_month=planning_month,
        scenario_version=scenario_version,
        run_version=run_version,
    ).first()


def _persist_synthetic_run_context(
    result: OptimizationResult,
    *,
    planning_month: date,
    branch: Branch,
    candidate_count: int,
) -> None:
    """Persist explicit mode/provenance beside one existing solver result."""

    metadata = dict(result.solver_metadata or {})
    existing_context = metadata.get("pipeline_context")
    if isinstance(existing_context, dict) and existing_context.get("mode") not in {
        None,
        SYNTHETIC_MODE,
    }:
        raise PipelineRunError(
            f"OptimizationResult {result.pk} is already marked for a different mode and cannot be reused."
        )
    metadata["pipeline_context"] = {
        "mode": SYNTHETIC_MODE,
        "parameter_source": DecisionOption.ParameterSource.SYNTHETIC_DEMO,
        "planning_month": planning_month.isoformat(),
        "branch_code": branch.branch_code,
        "candidate_evidence_count": candidate_count,
    }
    result.solver_metadata = metadata
    result.save(update_fields=("solver_metadata",))


def run_pipeline(*, planning_month: date, branch: Branch, mode: str) -> PipelineRunResult:
    """Coordinate only the existing specialist stages for one validated context.

    Historical mode verifies frozen evidence and deliberately stops before the
    synthetic CP-SAT/action workflow.  Synthetic mode consumes the existing
    finite option scenario and enforces solver and independent-checker gates
    before recommendation persistence.
    """

    if mode not in {HISTORICAL_MODE, SYNTHETIC_MODE}:
        raise PipelineRunError(f"Unsupported prototype mode: {mode!r}.")

    logger.info("Scenario %s/%s/%s: STARTED", planning_month, branch.branch_code, mode)
    candidate_count = _require_persisted_evidence(planning_month=planning_month, branch=branch)
    logger.info("Scenario %s/%s: FORECAST AND CANDIDATE EVIDENCE AVAILABLE", planning_month, branch.branch_code)

    if mode == HISTORICAL_MODE:
        logger.info("Scenario %s/%s: HISTORICAL EVIDENCE COMPLETE", planning_month, branch.branch_code)
        return PipelineRunResult(
            mode=mode,
            planning_month=planning_month,
            branch_code=branch.branch_code,
            candidate_count=candidate_count,
        )

    scenario_version = _resolve_synthetic_scenario(planning_month=planning_month, branch=branch)
    run_version = _run_version(
        planning_month=planning_month,
        branch=branch,
        scenario_version=scenario_version,
    )
    existing = _existing_result(
        planning_month=planning_month,
        scenario_version=scenario_version,
        run_version=run_version,
    )
    if existing is not None and existing.recommendations.exists():
        _persist_synthetic_run_context(
            existing,
            planning_month=planning_month,
            branch=branch,
            candidate_count=candidate_count,
        )
        logger.info("Run %s: REUSED COMPLETED RESULT", existing.pk)
        return PipelineRunResult(
            mode=mode,
            planning_month=planning_month,
            branch_code=branch.branch_code,
            candidate_count=candidate_count,
            optimization_result_id=existing.pk,
            run_version=run_version,
            solver_status=existing.solver_status,
            feasibility_status=existing.feasibility_status,
            recommendation_count=existing.recommendations.count(),
            reused_existing_result=True,
        )

    try:
        outcome = run_and_persist_optimization(
            planning_month=planning_month,
            scenario_version=scenario_version,
            run_version=run_version,
        )
    except SolverRunError as error:
        raise PipelineRunError(f"CP-SAT execution could not complete: {error}") from error
    result = outcome.optimization_result
    _persist_synthetic_run_context(
        result,
        planning_month=planning_month,
        branch=branch,
        candidate_count=candidate_count,
    )
    try:
        capture_run_provenance(
            optimization_result=result,
            branch=branch,
            mode=SYNTHETIC_MODE,
        )
    except RunProvenanceCaptureError as error:
        raise PipelineRunError(f"Run provenance could not be captured: {error}") from error
    logger.info("Run %s: SOLVER %s", result.pk, result.solver_status)
    if result.solver_status not in USABLE_SOLVER_STATUSES:
        return PipelineRunResult(
            mode=mode,
            planning_month=planning_month,
            branch_code=branch.branch_code,
            candidate_count=candidate_count,
            optimization_result_id=result.pk,
            run_version=run_version,
            solver_status=result.solver_status,
            feasibility_status=result.feasibility_status,
        )

    try:
        feasibility_report = check_and_persist_feasibility(result.pk)
    except CheckerInputError as error:
        raise PipelineRunError(f"Independent feasibility checker could not run: {error}") from error
    logger.info("Run %s: FEASIBILITY %s", result.pk, feasibility_report.status)
    if feasibility_report.status != OptimizationResult.FeasibilityStatus.PASS:
        return PipelineRunResult(
            mode=mode,
            planning_month=planning_month,
            branch_code=branch.branch_code,
            candidate_count=candidate_count,
            optimization_result_id=result.pk,
            run_version=run_version,
            solver_status=result.solver_status,
            feasibility_status=feasibility_report.status,
        )

    try:
        recommendations = create_recommendations_for_result(result.pk)
    except (RecommendationPersistenceError, PublicationGateBlocked, ValueError) as error:
        raise PipelineRunError(f"Recommendation publication could not complete: {error}") from error
    recommendation_count = recommendations.created_count + recommendations.existing_count
    logger.info("Run %s: RECOMMENDATIONS CREATED=%s; COMPLETE", result.pk, recommendation_count)
    return PipelineRunResult(
        mode=mode,
        planning_month=planning_month,
        branch_code=branch.branch_code,
        candidate_count=candidate_count,
        optimization_result_id=result.pk,
        run_version=run_version,
        solver_status=result.solver_status,
        feasibility_status=feasibility_report.status,
        recommendation_count=recommendation_count,
    )
