"""Build deterministic CSV exports from one persisted, reviewed run."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
import re

from planning.models import OptimizationResult, Recommendation
from planning.modes import ScenarioMode
from planning.services.publication_gate import evaluate_publication_gate
from reviews.models import ReviewDecision


NOT_AVAILABLE = "NOT_AVAILABLE"
CSV_COLUMNS = (
    "run_version",
    "branch_code",
    "planning_month",
    "mode",
    "product_id",
    "product_desc",
    "family",
    "l0",
    "l1",
    "q0",
    "q1",
    "action_or_status",
    "decision_evidence_score",
    "decision_evidence_source",
    "reason_1",
    "reason_2",
    "solver_status",
    "feasibility_status",
    "model_version",
    "mapping_version",
    "scenario_version",
    "source_manifest_hash",
    "review_status",
    "review_decision",
    "review_comment",
    "limitations",
)
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class RecommendationExportUnavailable(PermissionError):
    """Raised when persisted results cannot safely become a reviewed CSV."""


@dataclass(frozen=True)
class RecommendationCsvExport:
    """Response-ready, in-memory export for one immutable optimization run."""

    filename: str
    content: str
    row_count: int


def _provenance_for(run: OptimizationResult):
    try:
        return run.provenance
    except OptimizationResult.provenance.RelatedObjectDoesNotExist:
        return None


def _saved_reason_texts(recommendation: Recommendation) -> tuple[str, ...]:
    reasons = recommendation.reasons if isinstance(recommendation.reasons, list) else []
    return tuple(
        reason["text"]
        for reason in reasons
        if isinstance(reason, dict) and isinstance(reason.get("text"), str)
    )


def _filename_for(run: OptimizationResult) -> str:
    safe_run_version = _UNSAFE_FILENAME_CHARS.sub("_", run.run_version).strip("._") or "run"
    return f"recommendations_{safe_run_version}.csv"


def _require_exportable_reviewed_run(run: OptimizationResult):
    pipeline_context = (run.solver_metadata or {}).get("pipeline_context")
    if not isinstance(pipeline_context, dict) or pipeline_context.get("mode") != ScenarioMode.RESEARCH_SYNTHETIC:
        raise RecommendationExportUnavailable(
            "CSV export is available only for an identified RESEARCH_SYNTHETIC decision-demo run."
        )
    if pipeline_context.get("planning_month") != run.planning_month.isoformat():
        raise RecommendationExportUnavailable("CSV export is unavailable because persisted run context is inconsistent.")

    gate = evaluate_publication_gate(run.pk)
    if not gate.allowed:
        raise RecommendationExportUnavailable(
            f"Recommendation export unavailable - optimization/checking did not pass. "
            f"{gate.code}: {' '.join(gate.reasons)}"
        )

    review_status, latest_review = ReviewDecision.current_status_for_run(run)
    if latest_review is None:
        raise RecommendationExportUnavailable(
            "Recommendation export unavailable - a persisted human review is required for this exact run."
        )
    return pipeline_context, review_status, latest_review


def build_recommendation_csv(run: OptimizationResult) -> RecommendationCsvExport:
    """Serialize saved outputs for one exact reviewed run without recomputation."""

    pipeline_context, review_status, latest_review = _require_exportable_reviewed_run(run)
    recommendations = list(
        Recommendation.objects.filter(optimization_result=run)
        .select_related("branch", "product", "selected_option")
        .order_by("branch__branch_code", "product__canonical_product_id", "pk")
    )
    if not recommendations:
        raise RecommendationExportUnavailable(
            "Recommendation export unavailable - no saved Recommendation records exist for this exact run."
        )

    provenance = _provenance_for(run)
    model_version = provenance.model_version if provenance is not None else NOT_AVAILABLE
    mapping_version = provenance.mapping_version if provenance is not None else NOT_AVAILABLE
    source_manifest_hash = provenance.source_manifest_hash if provenance is not None else NOT_AVAILABLE
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()

    for recommendation in recommendations:
        option = recommendation.selected_option
        reason_texts = _saved_reason_texts(recommendation)
        writer.writerow({
            "run_version": run.run_version,
            "branch_code": recommendation.branch.branch_code,
            "planning_month": run.planning_month.isoformat(),
            "mode": pipeline_context["mode"],
            "product_id": recommendation.product.canonical_product_id,
            "product_desc": recommendation.product.product_desc,
            "family": recommendation.product.family_desc or NOT_AVAILABLE,
            "l0": option.l0 if option is not None else NOT_AVAILABLE,
            "l1": option.l1 if option is not None else NOT_AVAILABLE,
            "q0": option.q0 if option is not None and option.q0 is not None else NOT_AVAILABLE,
            "q1": option.q1 if option is not None else NOT_AVAILABLE,
            "action_or_status": recommendation.action or recommendation.status,
            "decision_evidence_score": (
                option.evidence_score if option is not None and option.evidence_score is not None else NOT_AVAILABLE
            ),
            "decision_evidence_source": option.evidence_source if option is not None else NOT_AVAILABLE,
            "reason_1": reason_texts[0] if reason_texts else NOT_AVAILABLE,
            "reason_2": reason_texts[1] if len(reason_texts) > 1 else NOT_AVAILABLE,
            "solver_status": run.solver_status,
            "feasibility_status": run.feasibility_status,
            "model_version": model_version,
            "mapping_version": mapping_version,
            "scenario_version": run.scenario_version,
            "source_manifest_hash": source_manifest_hash,
            "review_status": review_status,
            "review_decision": latest_review.decision,
            "review_comment": latest_review.comment,
            "limitations": (
                "SYNTHETIC DECISION DEMO - L0/q0 and constraints are demonstration values; "
                "not Singer-approved operational advice."
            ),
        })

    return RecommendationCsvExport(
        filename=_filename_for(run),
        content=output.getvalue(),
        row_count=len(recommendations),
    )
