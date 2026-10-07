from django.urls import path

from . import views


app_name = "reviews"

urlpatterns = [
    path("review/<int:run_id>/", views.review_run, name="review_run"),
    path("review/<int:run_id>/export/", views.export_reviewed_run_csv, name="export_reviewed_run_csv"),
]
