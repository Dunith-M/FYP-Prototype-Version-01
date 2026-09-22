from django.urls import path

from . import views


app_name = "analytics"

urlpatterns = [
    path("forecast/", views.forecast, name="forecast"),
]
