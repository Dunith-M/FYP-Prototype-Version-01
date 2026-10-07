from urllib.parse import urlencode
from datetime import date
import logging

from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from reviews.models import ReviewDecision

from .forms import RunScenarioForm
from .modes import ScenarioMode
from .models import OptimizationResult, Recommendation
from .navigation import remember_workflow, workflow_links
from .services.pipeline_runner import PipelineRunError, run_pipeline
from .services.publication_gate import evaluate_publication_gate


logger = logging.getLogger(__name__)


def run_scenario(request):
    """Validate browser input and delegate all work to the pipeline runner."""

    form = RunScenarioForm(request.POST if request.method == "POST" else None)
    context = {"form": form, "mode": None, "pipeline_error": None}

    if request.method == "POST" and form.is_valid():
        selected_mode = form.cleaned_data["mode"]
        try:
            outcome = run_pipeline(
                planning_month=form.cleaned_data["planning_month"],
                branch=form.cleaned_data["branch"],
                mode=selected_mode,
            )
        except PipelineRunError as error:
            context.update({"mode": selected_mode, "pipeline_error": str(error)})
        else:
            remember_workflow(
                request,
                mode=selected_mode,
                branch_code=outcome.branch_code,
                planning_month=outcome.planning_month,
                run_id=outcome.optimization_result_id,
            )
            query = {}
            if outcome.optimization_result_id is not None:
                query["result"] = outcome.optimization_result_id
            else:
                query.update(
                    {
                        "mode": selected_mode,
                        "month": outcome.planning_month.isoformat(),
                        "branch": outcome.branch_code,
                    }
                )
            from django.urls import reverse

            return redirect(f"{reverse('analytics:forecast_evidence')}?{urlencode(query)}")

    return render(request, "planning/run_scenario.html", context)


def redirect_to_confirmation() -> str:
    """Keep the redirect destination centralized without exposing pipeline details."""

    from django.urls import reverse

    return reverse("planning:run_confirmation")


def run_confirmation(request):
    """Small PRG destination; it reads a persisted synthetic result when available."""

    result = None
    result_id = request.GET.get("result")
    if result_id:
        try:
            result = OptimizationResult.objects.prefetch_related("recommendations").get(pk=int(result_id))
        except (OptimizationResult.DoesNotExist, TypeError, ValueError):
            result = None
    persisted_context = (result.solver_metadata or {}).get("pipeline_context") if result is not None else None
    mode = (
        persisted_context.get("mode")
        if isinstance(persisted_context, dict) and persisted_context.get("mode") in ScenarioMode.values
        else request.GET.get("mode") if result is None else None
    )
    return render(
        request,
        "planning/run_confirmation.html",
        {
            "mode": mode,
            "result": result,
            "historical_month": request.GET.get("month"),
            "historical_branch": request.GET.get("branch"),
        },
    )


def _recommendation_context(request):
    """Resolve a persisted synthetic run or a validated historical context."""

    result_id = request.GET.get("result")
    if result_id:
        result = get_object_or_404(OptimizationResult.objects.select_related("provenance__branch"), pk=result_id)
        persisted_context = (result.solver_metadata or {}).get("pipeline_context")
        if not isinstance(persisted_context, dict) or persisted_context.get("mode") != ScenarioMode.RESEARCH_SYNTHETIC:
            raise Http404("This persisted result has no identified synthetic planning context.")
        if persisted_context.get("planning_month") != result.planning_month.isoformat():
            raise Http404("The persisted planning context does not match the result month.")
        branch_code = persisted_context.get("branch_code")
        if not branch_code:
            raise Http404("The persisted planning context has no branch.")
        return result, {
            "mode": ScenarioMode.RESEARCH_SYNTHETIC,
            "branch_code": branch_code,
            "planning_month": result.planning_month,
            "run_id": result.pk,
            "run_version": result.run_version,
        }

    if request.GET.get("mode") != ScenarioMode.HISTORICAL:
        raise Http404("A historical recommendation request requires HISTORICAL mode.")
    try:
        planning_month = date.fromisoformat(request.GET["month"])
    except (KeyError, TypeError, ValueError) as error:
        raise Http404("A valid historical planning month is required.") from error
    if planning_month.day != 1 or not request.GET.get("branch"):
        raise Http404("A branch and first-of-month planning date are required.")
    return None, {
        "mode": ScenarioMode.HISTORICAL,
        "branch_code": request.GET["branch"],
        "planning_month": planning_month,
        "run_id": None,
        "run_version": None,
    }


def _stored_reason_texts(recommendation: Recommendation) -> tuple[str, ...]:
    """Extract only durable explanation text; never generate new reasons."""

    reasons = recommendation.reasons if isinstance(recommendation.reasons, list) else []
    return tuple(
        reason["text"]
        for reason in reasons
        if isinstance(reason, dict) and isinstance(reason.get("text"), str)
    )


def recommendation_table(request):
    """Read saved Day-08 outputs without invoking the decision pipeline."""

    result, run_context = _recommendation_context(request)
    rows = []
    counts = {action: 0 for action, _label in Recommendation.Action.choices}
    counts["WITHHELD"] = 0
    recommendation_gate = None
    blocked_message = None

    if result is not None:
        recommendation_gate = evaluate_publication_gate(result.pk)
        if not recommendation_gate.allowed:
            logger.warning(
                "Run %s: recommendation display blocked — %s",
                result.pk,
                " ".join(recommendation_gate.reasons),
            )
            blocked_message = "Recommendation unavailable — optimization/checking did not pass."
        else:
            rows = list(
                Recommendation.objects.filter(
                    optimization_result=result,
                    branch__branch_code=run_context["branch_code"],
                )
                .select_related("product", "selected_option")
                .order_by("action", "product__canonical_product_id")
            )
            for row in rows:
                row.reason_texts = _stored_reason_texts(row)
                row.primary_reason = row.reason_texts[0] if row.reason_texts else "Not recorded"
                counts[row.action or "WITHHELD"] += 1

    candidate_query = urlencode({
        "result": result.pk,
    }) if result is not None else urlencode({
        "mode": run_context["mode"],
        "month": run_context["planning_month"].isoformat(),
        "branch": run_context["branch_code"],
    })
    run_provenance = getattr(result, "provenance", None) if result is not None else None
    review_status, latest_review = (
        ReviewDecision.current_status_for_run(result)
        if result is not None
        else (ReviewDecision.CurrentStatus.PENDING, None)
    )
    return render(request, "planning/recommendations.html", {
        "result": result,
        "run_context": run_context,
        "mode": run_context["mode"],
        "recommendation_rows": rows,
        "counts": counts,
        "recommendation_gate": recommendation_gate,
        "blocked_message": blocked_message,
        "run_provenance": run_provenance,
        "review_status": review_status,
        "latest_review": latest_review,
        "candidate_evidence_url": f"{reverse('analytics:candidate_evidence')}?{candidate_query}",
        "workflow_navigation": workflow_links(
            mode=run_context["mode"],
            branch_code=run_context["branch_code"],
            planning_month=run_context["planning_month"],
            run_id=run_context["run_id"],
        ),
    })
