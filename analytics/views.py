from datetime import date
from urllib.parse import urlencode

from django.http import Http404
from django.shortcuts import get_object_or_404, render
from django.urls import reverse

from .services.forecast_service import (
    ForecastArtifactsUnavailable,
    get_forecast_evidence_for_context,
    get_forecast_page_data,
)
from .services.data_quality_summary import get_data_quality_page_data
from .models import CandidateResult
from catalog.models import Branch
from planning.models import OptimizationResult
from planning.modes import ScenarioMode
from planning.navigation import workflow_links


def forecast(request):
    """Render persisted historical forecasting evidence without running ML."""
    branch = request.GET.get("branch") or None
    month = request.GET.get("month") or None
    try:
        context = get_forecast_page_data(branch=branch, month=month)
    except ForecastArtifactsUnavailable as error:
        context = {
            "artifact_available": False,
            "artifact_message": str(error),
            "selected_branch": branch or "",
            "selected_month": month or "",
        }
    context["mode"] = "HISTORICAL_RESEARCH"
    return render(request, "analytics/forecast.html", context)


def _validated_evidence_context(request):
    """Resolve a persisted synthetic result or a validated historical context."""

    result_id = request.GET.get("result")
    if result_id:
        result = get_object_or_404(OptimizationResult, pk=result_id)
        persisted_context = (result.solver_metadata or {}).get("pipeline_context")
        if not isinstance(persisted_context, dict) or persisted_context.get("mode") != ScenarioMode.RESEARCH_SYNTHETIC:
            raise Http404("This persisted result has no identified synthetic planning context.")
        if persisted_context.get("planning_month") != result.planning_month.isoformat():
            raise Http404("The persisted planning context does not match the result month.")
        branch = get_object_or_404(Branch, branch_code=persisted_context.get("branch_code"))
        mode = ScenarioMode.RESEARCH_SYNTHETIC
        run_id = result.pk
        run_version = result.run_version
        planning_month = result.planning_month
    else:
        if request.GET.get("mode") != ScenarioMode.HISTORICAL:
            raise Http404("A historical evidence request requires HISTORICAL mode.")
        try:
            planning_month = date.fromisoformat(request.GET["month"])
        except (KeyError, TypeError, ValueError) as error:
            raise Http404("A valid historical planning month is required.") from error
        if planning_month.day != 1:
            raise Http404("The planning month must use the first calendar day.")
        branch = get_object_or_404(Branch, branch_code=request.GET.get("branch"))
        mode = ScenarioMode.HISTORICAL
        run_id = None
        run_version = None

    return branch, mode, {
        "run_id": run_id,
        "run_version": run_version,
        "branch": branch,
        "planning_month": planning_month,
    }


def _contextual_evidence_url(*, name: str, mode: str, run_context: dict) -> str:
    """Build a context-preserving read-only evidence URL."""

    query = (
        {"result": run_context["run_id"]}
        if run_context["run_id"] is not None
        else {
            "mode": mode,
            "month": run_context["planning_month"].isoformat(),
            "branch": run_context["branch"].branch_code,
        }
    )
    return f"{reverse(name)}?{urlencode(query)}"


def forecast_evidence(request):
    """Render saved Day-06 evidence for one validated Day-09 planning context."""

    branch, mode, run_context = _validated_evidence_context(request)
    try:
        context = get_forecast_evidence_for_context(
            branch=branch.branch_code,
            planning_month=run_context["planning_month"].isoformat(),
        )
    except ForecastArtifactsUnavailable as error:
        context = {
            "artifact_available": False,
            "artifact_message": str(error),
            "forecast_rows": [],
            "evidence_counts": {},
        }
    context.update(
        {
            "mode": mode,
            "run_context": run_context,
            "candidate_evidence_url": _contextual_evidence_url(
                name="analytics:candidate_evidence", mode=mode, run_context=run_context
            ),
            "workflow_navigation": workflow_links(
                mode=mode, branch_code=branch.branch_code,
                planning_month=run_context["planning_month"], run_id=run_context["run_id"],
            ),
        }
    )
    return render(request, "analytics/forecast.html", context)


def data_quality(request):
    """Render persisted mapping and DQ summaries without reprocessing source data."""
    return render(request, "analytics/data_quality.html", get_data_quality_page_data())


def _candidate_why(candidate: CandidateResult) -> str:
    """Create concise text from persisted component evidence only."""
    reasons = []
    if candidate.peer_score is not None and candidate.peer_score >= 0.67:
        reasons.append("Strong peer demand")
    if candidate.family_score is not None and candidate.family_score >= 0.67:
        reasons.append("Strong family evidence")
    if candidate.similar_branch_score is not None and candidate.similar_branch_score >= 0.67:
        reasons.append("Similar-branch evidence")
    if candidate.price_fit_score is not None and candidate.price_fit_score >= 0.67:
        reasons.append("Price fit")
    return " + ".join(reasons[:2]) if reasons else "Limited supporting evidence"


def _saved_candidate_rows(*, branch: Branch, planning_month: date, limit: int | None = None):
    """Read persisted research-shortlist rows only; never recalculate scores."""

    queryset = CandidateResult.objects.filter(
        branch=branch,
        planning_month=planning_month,
        status=CandidateResult.CandidateStatus.RESEARCH_SHORTLIST,
    ).select_related("product").order_by("rank", "-candidate_score", "product__canonical_product_id")
    total = queryset.count()
    rows = list(queryset[:limit] if limit is not None else queryset)
    for row in rows:
        row.why_text = _candidate_why(row)
    counts = {
        "shortlist": total,
        "adequate": sum(row.evidence_class == CandidateResult.EvidenceClass.ADEQUATE for row in rows),
        "limited": sum(row.evidence_class == CandidateResult.EvidenceClass.LIMITED for row in rows),
        "insufficient": sum(row.evidence_class == CandidateResult.EvidenceClass.INSUFFICIENT for row in rows),
    }
    return rows, total, counts


def candidate_list(request):
    """Render persisted Track-B research evidence without recalculating scores."""
    base = CandidateResult.objects.select_related("branch", "product")
    branches = list(Branch.objects.filter(candidate_results__isnull=False).distinct().order_by("branch_code"))
    selected_branch = request.GET.get("branch") or (branches[0].branch_code if branches else "")
    branch_months = base.filter(branch__branch_code=selected_branch).values_list("planning_month", flat=True).distinct().order_by("planning_month") if selected_branch else []
    months = list(branch_months)
    default_month = months[-1].strftime("%Y-%m") if months else ""
    selected_month = request.GET.get("month") or default_month
    try:
        year, month = (int(part) for part in selected_month.split("-"))
        branch = Branch.objects.get(branch_code=selected_branch)
        rows, total, counts = _saved_candidate_rows(
            branch=branch, planning_month=date(year, month, 1), limit=20
        )
    except (TypeError, ValueError, Branch.DoesNotExist):
        rows, total, counts = [], 0, {"shortlist": 0, "adequate": 0, "limited": 0, "insufficient": 0}
    return render(request, "analytics/candidates.html", {
        "branches": branches, "months": months, "selected_branch": selected_branch,
        "selected_month": selected_month, "candidate_rows": rows, "counts": counts,
        "total_candidates": total,
        "mode": ScenarioMode.HISTORICAL,
    })


def candidate_evidence(request):
    """Render saved Day-07 research-shortlist evidence for one Day-09 context."""

    branch, mode, run_context = _validated_evidence_context(request)
    rows, total, counts = _saved_candidate_rows(
        branch=branch,
        planning_month=run_context["planning_month"],
        limit=20,
    )
    return render(request, "analytics/candidates.html", {
        "candidate_rows": rows,
        "counts": counts,
        "total_candidates": total,
        "display_limit": 20,
        "candidate_run_versions": sorted({row.run_version for row in rows}),
        "mode": mode,
        "run_context": run_context,
        "recommendation_url": _contextual_evidence_url(
            name="planning:recommendation_table", mode=mode, run_context=run_context
        ),
        "forecast_evidence_url": _contextual_evidence_url(
            name="analytics:forecast_evidence", mode=mode, run_context=run_context
        ),
        "workflow_navigation": workflow_links(
            mode=mode, branch_code=branch.branch_code,
            planning_month=run_context["planning_month"], run_id=run_context["run_id"],
        ),
    })
