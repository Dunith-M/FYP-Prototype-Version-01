from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from analytics.models import CandidateResult
from catalog.models import Branch
from planning.models import OptimizationResult, Recommendation
from planning.modes import ScenarioMode
from planning.navigation import workflow_links
from planning.services.publication_gate import evaluate_publication_gate

from .forms import ReviewDecisionForm
from .models import ReviewDecision
from .services.export_service import RecommendationExportUnavailable, build_recommendation_csv


def _reason_texts(recommendation: Recommendation) -> tuple[str, ...]:
    reasons = recommendation.reasons if isinstance(recommendation.reasons, list) else []
    return tuple(
        reason["text"]
        for reason in reasons
        if isinstance(reason, dict) and isinstance(reason.get("text"), str)
    )


def _review_context(run: OptimizationResult) -> dict[str, object]:
    """Read saved plan evidence for a specific run without recalculating it."""

    pipeline_context = (run.solver_metadata or {}).get("pipeline_context")
    if not isinstance(pipeline_context, dict):
        raise Http404("This run has no persisted planning context.")
    if pipeline_context.get("mode") != ScenarioMode.RESEARCH_SYNTHETIC:
        raise Http404("Only identified synthetic decision-demo runs can be reviewed here.")
    if pipeline_context.get("planning_month") != run.planning_month.isoformat():
        raise Http404("The persisted planning context does not match this run month.")
    branch_code = pipeline_context.get("branch_code")
    if not branch_code:
        raise Http404("The persisted planning context has no branch.")

    try:
        provenance = run.provenance
    except OptimizationResult.provenance.RelatedObjectDoesNotExist:
        provenance = None
    branch = provenance.branch if provenance is not None else Branch.objects.filter(branch_code=branch_code).first()
    gate = evaluate_publication_gate(run.pk)
    review_status, latest_review = ReviewDecision.current_status_for_run(run)
    rows: list[Recommendation] = []
    if gate.allowed:
        rows = list(
            Recommendation.objects.filter(optimization_result=run, branch__branch_code=branch_code)
            .select_related("product", "selected_option")
            .order_by("action", "product__canonical_product_id")
        )
        candidates = CandidateResult.objects.filter(
            branch__branch_code=branch_code,
            planning_month=run.planning_month,
            product_id__in=[row.product_id for row in rows],
        )
        candidate_by_product = {candidate.product_id: candidate for candidate in candidates}
        for row in rows:
            row.reason_texts = _reason_texts(row)
            row.candidate_evidence = candidate_by_product.get(row.product_id)

    return {
        "run": run,
        "mode": ScenarioMode.RESEARCH_SYNTHETIC,
        "branch_code": branch_code,
        "branch": branch,
        "provenance": provenance,
        "recommendation_rows": rows,
        "gate": gate,
        "review_status": review_status,
        "latest_review": latest_review,
        "review_history": run.review_decisions.select_related("reviewer").order_by("-reviewed_at", "-pk"),
        "workflow_navigation": workflow_links(
            mode=ScenarioMode.RESEARCH_SYNTHETIC,
            branch_code=branch_code,
            planning_month=run.planning_month,
            run_id=run.pk,
        ),
    }


@login_required(login_url="admin:login")
def review_run(request, run_id: int):
    """Show one persisted, gate-approved plan before recording a human review."""

    run = get_object_or_404(OptimizationResult.objects.select_related("provenance__branch"), pk=run_id)
    context = _review_context(run)
    if not context["gate"].allowed:
        context.update({
            "form": None,
            "blocked_message": "Review unavailable — this recommendation plan did not pass optimization/feasibility checks.",
        })
        return render(request, "reviews/review.html", context)

    form = ReviewDecisionForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        review = form.save(commit=False)
        review.reviewed_run = run
        review.reviewer = request.user
        review.save()
        messages.success(request, "Review saved successfully.")
        return redirect(f"{reverse('reviews:review_run', args=(run.pk,))}?saved_review={review.pk}")

    context["form"] = form
    context["blocked_message"] = None
    context["saved_review_id"] = request.GET.get("saved_review")
    return render(request, "reviews/review.html", context)


@login_required(login_url="admin:login")
def export_reviewed_run_csv(request, run_id: int):
    """Download a CSV for exactly one persisted, reviewed recommendation run."""

    run = get_object_or_404(OptimizationResult.objects.select_related("provenance__branch"), pk=run_id)
    try:
        export = build_recommendation_csv(run)
    except RecommendationExportUnavailable as error:
        return HttpResponse(str(error), status=409, content_type="text/plain; charset=utf-8")

    response = HttpResponse(export.content, content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{export.filename}"'
    return response
