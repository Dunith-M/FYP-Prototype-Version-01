from django.urls import path

from . import views


app_name = "analytics"

urlpatterns = [
    path("forecast/", views.forecast, name="forecast"),
    path("forecast/evidence/", views.forecast_evidence, name="forecast_evidence"),
    path("candidates/", views.candidate_list, name="candidate_list"),
    path("candidates/evidence/", views.candidate_evidence, name="candidate_evidence"),
]
