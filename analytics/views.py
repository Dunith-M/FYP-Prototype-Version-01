from django.shortcuts import render

from .services.forecast_service import ForecastArtifactsUnavailable, get_forecast_page_data


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
    return render(request, "analytics/forecast.html", context)
