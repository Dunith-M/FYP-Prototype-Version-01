"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import include, path

from analytics import views as analytics_views
from .views import home

urlpatterns = [
    path("", home, name="home"),
    path("", include("ingestion.urls")),
    path("data-quality/", analytics_views.data_quality, name="data_quality"),
    path("", include("planning.urls")),
    path("", include("reviews.urls")),
    path("analytics/", include("analytics.urls")),
    path("admin/", admin.site.urls),
]
