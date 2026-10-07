from django.urls import path

from . import views


app_name = "planning"

urlpatterns = [
    path("run/", views.run_scenario, name="run_scenario"),
    path("run/confirmation/", views.run_confirmation, name="run_confirmation"),
    path("recommendations/", views.recommendation_table, name="recommendation_table"),
]
