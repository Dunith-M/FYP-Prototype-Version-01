"""Links for the selected, exact planning context in the prototype workflow."""

from datetime import date
from urllib.parse import urlencode

from django.urls import reverse

from catalog.models import Branch
from planning.models import OptimizationResult
from planning.modes import ScenarioMode


SESSION_KEY = "selected_planning_context"


def workflow_links(*, mode: str, branch_code: str, planning_month: date, run_id: int | None = None):
    """Build existing named routes without changing the selected context."""
    query = (
        {"result": run_id}
        if run_id is not None
        else {"mode": mode, "month": planning_month.isoformat(), "branch": branch_code}
    )
    if mode not in ScenarioMode.values or (mode == ScenarioMode.RESEARCH_SYNTHETIC) != (run_id is not None):
        return None
    suffix = urlencode(query)
    return {
        "mode": mode,
        "branch_code": branch_code,
        "planning_month": planning_month,
        "run_id": run_id,
        "forecast_url": f"{reverse('analytics:forecast_evidence')}?{suffix}",
        "candidate_url": f"{reverse('analytics:candidate_evidence')}?{suffix}",
        "recommendation_url": f"{reverse('planning:recommendation_table')}?{suffix}",
    }


def remember_workflow(request, *, mode: str, branch_code: str, planning_month: date, run_id: int | None):
    """Remember only the selected identity; all synthetic details are read from its saved result."""
    request.session[SESSION_KEY] = (
        {"result": run_id}
        if run_id is not None
        else {"mode": mode, "month": planning_month.isoformat(), "branch": branch_code}
    )


def workflow_navigation(request):
    """Expose links in shared navigation; never substitute the globally latest run."""
    selected = request.session.get(SESSION_KEY)
    if not isinstance(selected, dict):
        return {"workflow_navigation": None}

    if "result" in selected:
        result_id = selected["result"]
        if isinstance(result_id, bool) or not isinstance(result_id, int) or result_id < 1:
            return {"workflow_navigation": None}
        result = OptimizationResult.objects.filter(pk=result_id).first()
        if result is None:
            return {"workflow_navigation": None}
        context = (result.solver_metadata or {}).get("pipeline_context")
        if not isinstance(context, dict) or context.get("mode") != ScenarioMode.RESEARCH_SYNTHETIC:
            return {"workflow_navigation": None}
        if context.get("planning_month") != result.planning_month.isoformat():
            return {"workflow_navigation": None}
        branch_code = context.get("branch_code")
        if not branch_code or not Branch.objects.filter(branch_code=branch_code).exists():
            return {"workflow_navigation": None}
        links = workflow_links(
            mode=ScenarioMode.RESEARCH_SYNTHETIC,
            branch_code=branch_code,
            planning_month=result.planning_month,
            run_id=result.pk,
        )
    else:
        if selected.get("mode") != ScenarioMode.HISTORICAL:
            return {"workflow_navigation": None}
        try:
            planning_month = date.fromisoformat(selected["month"])
        except (KeyError, TypeError, ValueError):
            return {"workflow_navigation": None}
        branch_code = selected.get("branch")
        if planning_month.day != 1 or not branch_code or not Branch.objects.filter(branch_code=branch_code).exists():
            return {"workflow_navigation": None}
        links = workflow_links(
            mode=ScenarioMode.HISTORICAL,
            branch_code=branch_code,
            planning_month=planning_month,
        )
    return {"workflow_navigation": links}
