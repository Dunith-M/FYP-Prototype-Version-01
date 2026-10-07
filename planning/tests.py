from datetime import date
import inspect
from unittest.mock import patch

from django.test import Client, TestCase
from django.urls import reverse

from analytics.models import CandidateResult
from catalog.models import Branch, Product
from planning.models import DecisionOption, OptimizationResult, Recommendation
from planning.forms import RunScenarioForm
from planning.services.pipeline_runner import PipelineRunResult


class RunScenarioPageTests(TestCase):
    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="GALLE-A")
        self.product = Product.objects.create(
            canonical_product_id="CP_SCENARIO",
            product_code="SCENARIO",
            product_desc="Scenario evidence product",
        )
        CandidateResult.objects.create(
            branch=self.branch,
            product=self.product,
            planning_month=date(2026, 4, 1),
            peer_score=0.5,
            family_score=0.5,
            similar_branch_score=0.5,
            price_fit_score=0.5,
            candidate_score=0.5,
            rank=1,
            evidence_class=CandidateResult.EvidenceClass.ADEQUATE,
            run_version="day07_test",
        )
        self.url = reverse("planning:run_scenario")

    def _valid_data(self, mode="HISTORICAL"):
        return {"planning_month": "2026-04-01", "branch": self.branch.pk, "mode": mode}

    @patch("planning.views.run_pipeline")
    def test_get_displays_persisted_choices_without_creating_pipeline_records(self, pipeline):
        before = (DecisionOption.objects.count(), OptimizationResult.objects.count(), Recommendation.objects.count())

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Planning month")
        self.assertContains(response, "April 2026")
        self.assertContains(response, "BIL01 — GALLE-A")
        self.assertContains(response, "HISTORICAL")
        self.assertContains(response, "RESEARCH_SYNTHETIC")
        self.assertContains(response, "PROTOTYPE MODE NOT IDENTIFIED")
        self.assertEqual(before, (DecisionOption.objects.count(), OptimizationResult.objects.count(), Recommendation.objects.count()))
        pipeline.assert_not_called()

    @patch("planning.views.run_pipeline")
    def test_valid_historical_submission_calls_runner_and_redirects(self, pipeline):
        pipeline.return_value = PipelineRunResult(
            mode="HISTORICAL", planning_month=date(2026, 4, 1), branch_code="BIL01", candidate_count=20
        )

        response = self.client.post(self.url, self._valid_data())

        self.assertEqual(response.status_code, 302)
        self.assertIn("/analytics/forecast/evidence/?mode=HISTORICAL", response["Location"])
        pipeline.assert_called_once_with(planning_month=date(2026, 4, 1), branch=self.branch, mode="HISTORICAL")

    @patch("planning.views.run_pipeline")
    def test_valid_synthetic_submission_calls_runner_and_redirects(self, pipeline):
        pipeline.return_value = PipelineRunResult(
            mode="RESEARCH_SYNTHETIC", planning_month=date(2026, 4, 1), branch_code="BIL01",
            candidate_count=20, optimization_result_id=123, run_version="web", solver_status="OPTIMAL",
            feasibility_status="PASS", recommendation_count=30,
        )

        response = self.client.post(self.url, self._valid_data("RESEARCH_SYNTHETIC"))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/analytics/forecast/evidence/?result=123", response["Location"])
        pipeline.assert_called_once_with(planning_month=date(2026, 4, 1), branch=self.branch, mode="RESEARCH_SYNTHETIC")

    @patch("planning.views.run_pipeline")
    def test_missing_and_tampered_values_are_rejected_without_calling_runner(self, pipeline):
        outside_pilot = Branch.objects.create(branch_code="OUT01", branch_name="Outside pilot")
        invalid_cases = (
            ({}, "This field is required."),
            ({**self._valid_data(), "planning_month": "2026-05-01"}, "Select a valid choice."),
            ({**self._valid_data(), "branch": "FAKE01"}, "Select a valid choice."),
            ({**self._valid_data(), "branch": outside_pilot.pk}, "Select a valid choice."),
            ({**self._valid_data(), "mode": "OPERATIONAL"}, "Select a valid choice."),
        )

        for payload, error_text in invalid_cases:
            with self.subTest(payload=payload):
                response = self.client.post(self.url, payload)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, error_text)
        pipeline.assert_not_called()

    def test_empty_persisted_sources_are_safe(self):
        CandidateResult.objects.all().delete()
        Branch.objects.all().delete()

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No pilot branches are currently registered.")
        self.assertContains(response, "No supported planning months are currently available.")
        self.assertContains(response, "disabled")

    def test_csrf_protection_rejects_post_without_token(self):
        csrf_client = Client(enforce_csrf_checks=True)

        response = csrf_client.post(self.url, self._valid_data())

        self.assertEqual(response.status_code, 403)

    def test_form_does_not_read_raw_datasets(self):
        source = inspect.getsource(RunScenarioForm)
        for forbidden_operation in ("read_excel", "read_parquet", "openpyxl", "pandas", "data/raw"):
            self.assertNotIn(forbidden_operation, source)
