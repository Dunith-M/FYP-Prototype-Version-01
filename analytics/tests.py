import inspect
from datetime import date
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase, TestCase
from django.template.loader import render_to_string
from django.urls import reverse

from .services.forecast_service import ForecastArtifactsUnavailable, _to_records
from .services.data_quality_summary import get_data_quality_page_data
from catalog.models import Branch
from planning.models import OptimizationResult


class ForecastPageTests(SimpleTestCase):
    @patch("analytics.views.get_forecast_page_data")
    def test_forecast_page_is_research_labelled_and_passes_get_filters(self, page_data):
        page_data.return_value = {
            "artifact_available": True,
            "selected_model": "trailing_mean_3",
            "overall_mae": "2.24",
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
        self.assertContains(response, "HISTORICAL / RESEARCH PROTOTYPE &mdash; NOT CURRENT OPERATIONAL ADVICE")
        self.assertContains(response, "trailing_mean_3")
        self.assertContains(response, "Evidence")
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

    @patch("analytics.views.get_forecast_page_data")
    def test_forecast_page_displays_confidence_without_exposing_withheld_raw_prediction(self, page_data):
        page_data.return_value = {
            "artifact_available": True,
            "selected_model": "trailing_mean_3",
            "overall_mae": "2.24",
            "overall_wape_display": "58.82%",
            "branches": ["BIL01"],
            "months": ["2026-02"],
            "selected_branch": "",
            "selected_month": "",
            "total_matching_rows": 3,
            "rows_truncated": False,
            "filter_message": None,
            "forecast_rows": [
                {"product": "ADEQUATE_PRODUCT", "family": "F1", "actual_units": "7.00", "forecast": "8.20", "model_name": "trailing_mean_3", "evidence_class": "ADEQUATE", "evidence_badge_class": "bg-success", "evidence_reason": "", "absolute_error": "1.20"},
                {"product": "LIMITED_PRODUCT", "family": "F2", "actual_units": "3.00", "forecast": "3.10", "model_name": "trailing_mean_3", "evidence_class": "LIMITED", "evidence_badge_class": "bg-warning text-dark", "evidence_reason": "Sparse historical observations", "absolute_error": "0.10"},
                {"product": "WITHHELD_PRODUCT", "family": "F3", "actual_units": "5.00", "forecast": "WITHHELD", "model_name": "—", "evidence_class": "INSUFFICIENT", "evidence_badge_class": "bg-danger", "evidence_reason": "No usable historical observations", "absolute_error": "—"},
            ],
        }

        response = self.client.get(reverse("analytics:forecast"))

        self.assertContains(response, "ADEQUATE")
        self.assertContains(response, "LIMITED")
        self.assertContains(response, "Sparse historical observations")
        self.assertContains(response, "INSUFFICIENT")
        self.assertContains(response, "WITHHELD")
        self.assertContains(response, "—")
        self.assertNotContains(response, "999.99")
        self.assertContains(response, "trailing_mean_3")

    def test_withheld_record_hides_raw_prediction_model_and_error(self):
        rows = pd.DataFrame(
            [{
                "canonical_product_id": "CP_1", "family_id": "F1", "actual_units": 5.0,
                "raw_prediction": 999.99, "forecast_value": None, "selected_model": "trailing_mean_3",
                "evidence_class": "INSUFFICIENT", "evidence_reason_code": "NO_USABLE_HISTORY",
                "evidence_warning": "Forecast withheld because required supporting evidence is insufficient.",
                "forecast_status": "WITHHELD",
            }]
        )

        record = _to_records(rows)[0]

        self.assertEqual(record["forecast"], "WITHHELD")
        self.assertEqual(record["model_name"], "—")
        self.assertEqual(record["absolute_error"], "—")
        self.assertNotIn("999.99", record.values())

    def test_prototype_banner_maps_existing_historical_and_synthetic_modes(self):
        historical = render_to_string("partials/prototype_banner.html", {"mode": "HISTORICAL_RESEARCH"})
        synthetic = render_to_string("partials/prototype_banner.html", {"mode": "RESEARCH_SYNTHETIC"})
        unidentified = render_to_string("partials/prototype_banner.html")

        self.assertIn("HISTORICAL / RESEARCH PROTOTYPE &mdash; NOT CURRENT OPERATIONAL ADVICE", historical)
        self.assertIn("SYNTHETIC DECISION DEMO &mdash; L0/q0 AND CONSTRAINTS ARE DEMONSTRATION VALUES", synthetic)
        self.assertIn("PROTOTYPE MODE NOT IDENTIFIED &mdash; NOT CURRENT OPERATIONAL ADVICE", unidentified)


class DataQualityPageTests(SimpleTestCase):
    @patch("analytics.views.get_data_quality_page_data")
    def test_page_renders_persisted_mapping_and_unresolved_evidence(self, page_data):
        page_data.return_value = {
            "mode": "HISTORICAL_RESEARCH",
            "data_available": True,
            "branch_mapping": {"coverage": "100.0%", "mapped": "10", "expected": "10", "unmatched": "0", "status": "PASS", "badge_class": "bg-success"},
            "product_mapping": {"coverage": "88.7%", "mapped": "1,406", "total": "1,586", "unresolved": "180"},
            "quality_metrics": [{"label": "Negative TOTAL_UNITS", "value": "9,428", "status": "REVIEW", "badge_class": "bg-warning text-dark", "notes": "Preserved."}],
            "unknown_inventory": "Not available in persisted DQ results.",
            "unresolved_rows": [{"source_product_code": "FREE2YR", "affected_sales_rows": "3273", "affected_signed_units": "2315.0", "confidence_class": "UNRESOLVED", "notes": "Historical sales code absent."}],
            "unresolved_total": 180,
            "safety_checks": [{"label": "Branch mapping", "status": "PASS", "badge_class": "bg-success", "detail": "Confirmed."}],
        }

        response = self.client.get(reverse("data_quality"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Mapping &amp; Data Quality")
        self.assertContains(response, "1,406")
        self.assertContains(response, "FREE2YR")
        self.assertContains(response, "UNRESOLVED")
        self.assertContains(response, "HISTORICAL / RESEARCH PROTOTYPE &mdash; NOT CURRENT OPERATIONAL ADVICE")

    @patch("analytics.views.get_data_quality_page_data")
    def test_page_handles_missing_persisted_reports_and_empty_unresolved_queue(self, page_data):
        page_data.return_value = {
            "mode": "HISTORICAL_RESEARCH", "data_available": False,
            "branch_mapping": {"coverage": "Not available", "mapped": "Not available", "expected": "Not available", "unmatched": "Not available", "status": "NOT AVAILABLE", "badge_class": "bg-secondary"},
            "product_mapping": None, "quality_metrics": [],
            "unknown_inventory": "Not available in persisted DQ results.",
            "unresolved_rows": [], "unresolved_total": 0,
            "safety_checks": [],
        }

        response = self.client.get(reverse("data_quality"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No persisted mapping or data-quality reports are available yet.")
        self.assertContains(response, "No persisted unresolved-product mapping records are available.")

    def test_summary_service_does_not_open_raw_source_datasets(self):
        source = inspect.getsource(get_data_quality_page_data)

        for forbidden_operation in (
            "read_excel",
            "read_parquet",
            "openpyxl",
            "pandas",
            "data/raw",
        ):
            self.assertNotIn(forbidden_operation, source)


class ForecastEvidencePageTests(TestCase):
    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="GALLE-A")
        self.result = OptimizationResult.objects.create(
            planning_month=date(2026, 4, 1),
            scenario_version="day07_synthetic_options_v1",
            run_version="day09_forecast_context",
            solver_status=OptimizationResult.SolverStatus.OPTIMAL,
            solver_metadata={
                "pipeline_context": {
                    "mode": "RESEARCH_SYNTHETIC",
                    "parameter_source": "SYNTHETIC_DEMO",
                    "planning_month": "2026-04-01",
                    "branch_code": "BIL01",
                }
            },
        )
        self.saved_data = {
            "artifact_available": True,
            "selected_model": "trailing_mean_3",
            "overall_mae": "2.24",
            "overall_wape_display": "58.82%",
            "branches": ["BIL01"],
            "months": ["2025-12", "2026-01", "2026-02"],
            "selected_branch": "BIL01",
            "selected_month": "",
            "forecast_rows": [{
                "product": "CP_FORECAST", "product_code": "P001", "product_description": "Saved fridge",
                "family": "REF", "actual_units": "7.00", "forecast": "8.20",
                "model_name": "trailing_mean_3", "evidence_class": "ADEQUATE",
                "evidence_badge_class": "bg-success", "evidence_reason": "", "absolute_error": "1.20",
            }],
            "total_matching_rows": 1,
            "rows_truncated": False,
            "filter_message": None,
            "evidence_counts": {"ADEQUATE": 1, "LIMITED": 0, "INSUFFICIENT": 0},
            "context_forecast_available": False,
            "artifact_validation_months": ["2025-12", "2026-01", "2026-02"],
        }

    @patch("analytics.views.get_forecast_evidence_for_context")
    def test_synthetic_run_reads_saved_forecast_evidence_with_persisted_context(self, reader):
        reader.return_value = self.saved_data

        response = self.client.get(reverse("analytics:forecast_evidence"), {"result": self.result.pk})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "BIL01 - GALLE-A")
        self.assertContains(response, "April 2026")
        self.assertContains(response, "RESEARCH_SYNTHETIC")
        self.assertContains(response, "Saved fridge")
        self.assertContains(response, "8.20")
        self.assertContains(response, "trailing_mean_3")
        self.assertContains(response, "ADEQUATE")
        self.assertContains(response, "SYNTHETIC DECISION DEMO")
        reader.assert_called_once_with(branch="BIL01", planning_month="2026-04-01")

    @patch("analytics.views.get_forecast_evidence_for_context")
    def test_historical_context_is_labelled_without_synthetic_values(self, reader):
        reader.return_value = self.saved_data

        response = self.client.get(
            reverse("analytics:forecast_evidence"),
            {"mode": "HISTORICAL", "branch": "BIL01", "month": "2026-04-01"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "HISTORICAL / RESEARCH PROTOTYPE")
        self.assertNotContains(response, "L0/q0 AND CONSTRAINTS")

    @patch("analytics.views.get_forecast_evidence_for_context")
    def test_empty_saved_forecast_context_is_safe(self, reader):
        reader.return_value = self.saved_data | {"forecast_rows": [], "total_matching_rows": 0}

        response = self.client.get(reverse("analytics:forecast_evidence"), {"result": self.result.pk})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No saved forecast evidence matches the selected context.")

    def test_invalid_or_unidentified_result_is_not_displayed(self):
        response = self.client.get(reverse("analytics:forecast_evidence"), {"result": 999999})
        self.assertEqual(response.status_code, 404)

    def test_forecast_evidence_view_reads_saved_data_without_training_or_pipeline_execution(self):
        source = inspect.getsource(__import__("analytics.views", fromlist=["forecast_evidence"]).forecast_evidence)

        for forbidden_operation in ("fit(", "train", "run_pipeline", "run_and_persist_optimization", "build_feature"):
            self.assertNotIn(forbidden_operation, source)
