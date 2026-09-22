from django.test import TestCase

# Create your tests here.
from unittest.mock import patch

from django.test import SimpleTestCase
from django.urls import reverse

from .services.forecast_service import ForecastArtifactsUnavailable


class ForecastPageTests(SimpleTestCase):
    @patch("analytics.views.get_forecast_page_data")
    def test_forecast_page_is_research_labelled_and_passes_get_filters(self, page_data):
        page_data.return_value = {
            "artifact_available": True,
            "selected_model": "trailing_mean_3",
            "overall_mae": 2.24,
            "overall_wape_display": "58.82%",
            "branches": ["BIL01"],
            "months": ["2025-12", "2026-01", "2026-02"],
            "selected_branch": "BIL01",
            "selected_month": "2026-02",
            "forecast_rows": [],
            "total_matching_rows": 0,
            "rows_truncated": False,
            "filter_message": None,
        }

        response = self.client.get(reverse("analytics:forecast"), {"branch": "BIL01", "month": "2026-02"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "HISTORICAL / RESEARCH PROTOTYPE")
        self.assertContains(response, "Historical forecasting evidence only. Not current operational advice.")
        self.assertContains(response, "trailing_mean_3")
        self.assertNotContains(response, "2026-03")
        page_data.assert_called_once_with(branch="BIL01", month="2026-02")

    @patch("analytics.views.get_forecast_page_data")
    def test_forecast_page_shows_safe_artifact_message(self, page_data):
        page_data.side_effect = ForecastArtifactsUnavailable(
            "Forecast artifacts are not available. Run the Day-05 forecasting pipeline first."
        )

        response = self.client.get(reverse("analytics:forecast"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Forecast artifacts are not available")
