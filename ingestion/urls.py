from django.urls import path

from . import views


app_name = "ingestion"

urlpatterns = [
    path("data-status/", views.data_status, name="data_status"),
]
