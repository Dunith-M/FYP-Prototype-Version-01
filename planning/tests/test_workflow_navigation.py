from datetime import date
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from analytics.models import CandidateResult
from catalog.models import Branch, Product
from planning.models import OptimizationResult
from planning.navigation import SESSION_KEY
from planning.services.pipeline_runner import PipelineRunResult


class WorkflowNavigationTests(TestCase):
    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="GALLE-A")
        self.product = Product.objects.create(
            canonical_product_id="NAV_PRODUCT", product_code="NAV", product_desc="Navigation product"
        )
        CandidateResult.objects.create(
            branch=self.branch, product=self.product, planning_month=date(2026, 4, 1),
            peer_score=.5, family_score=.5, candidate_score=.5, rank=1,
            evidence_class=CandidateResult.EvidenceClass.ADEQUATE, run_version="nav-test",
        )
        self.run = self._result("first-run")

    def _result(self, version):
        return OptimizationResult.objects.create(
            planning_month=date(2026, 4, 1), scenario_version="nav-synthetic-v1",
            run_version=version, solver_status=OptimizationResult.SolverStatus.UNKNOWN,
            solver_metadata={"pipeline_context": {
                "mode": "RESEARCH_SYNTHETIC", "planning_month": "2026-04-01",
                "branch_code": "BIL01", "parameter_source": "SYNTHETIC_DEMO",
            }},
        )

    def test_global_links_and_disabled_run_links_before_selection(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        for name in ("home", "ingestion:data_status", "data_quality", "planning:run_scenario"):
            self.assertContains(response, f'href="{reverse(name)}"')
        self.assertContains(response, "Run a scenario first to view Forecast Evidence")
        self.assertContains(response, '<span class="nav-link disabled" aria-disabled="true">Recommendations</span>', html=True)
        self.assertNotContains(response, 'href="/recommendations/?result=')

    @patch("planning.views.run_pipeline")
    def test_selected_synthetic_run_stays_available_in_global_navigation(self, pipeline):
        pipeline.return_value = PipelineRunResult(
            mode="RESEARCH_SYNTHETIC", planning_month=date(2026, 4, 1),
            branch_code="BIL01", candidate_count=1, optimization_result_id=self.run.pk,
        )
        response = self.client.post(reverse("planning:run_scenario"), {
            "planning_month": "2026-04-01", "branch": self.branch.pk,
            "mode": "RESEARCH_SYNTHETIC",
        })
        self.assertRedirects(
            response, f"{reverse('analytics:forecast_evidence')}?result={self.run.pk}",
            fetch_redirect_response=False,
        )
        home = self.client.get(reverse("home"))
        self.assertContains(home, f'href="{reverse("analytics:forecast_evidence")}?result={self.run.pk}"')
        self.assertContains(home, f'href="{reverse("analytics:candidate_evidence")}?result={self.run.pk}"')
        self.assertContains(home, f'href="{reverse("planning:recommendation_table")}?result={self.run.pk}"')
        self.assertEqual(self.client.session[SESSION_KEY], {"result": self.run.pk})

    @patch("analytics.views.get_forecast_evidence_for_context")
    def test_forecast_to_candidate_to_recommendations_preserves_exact_run(self, forecast_reader):
        forecast_reader.return_value = {"artifact_available": False, "artifact_message": "Saved evidence unavailable"}
        later_run = self._result("later-run")
        session = self.client.session
        session[SESSION_KEY] = {"result": later_run.pk}
        session.save()

        forecast = self.client.get(reverse("analytics:forecast_evidence"), {"result": self.run.pk})
        self.assertContains(forecast, "BIL01")
        self.assertContains(forecast, "April 2026")
        self.assertContains(forecast, "RESEARCH_SYNTHETIC")
        self.assertContains(forecast, "first-run")
        self.assertContains(forecast, f'href="{reverse("analytics:candidate_evidence")}?result={self.run.pk}"')
        self.assertNotContains(forecast, f'href="{reverse("analytics:candidate_evidence")}?result={later_run.pk}"')

        candidate = self.client.get(reverse("analytics:candidate_evidence"), {"result": self.run.pk})
        self.assertContains(candidate, "RESEARCH SHORTLIST")
        self.assertContains(candidate, "first-run")
        self.assertContains(candidate, f'href="{reverse("planning:recommendation_table")}?result={self.run.pk}"')
        self.assertNotContains(candidate, f'href="{reverse("planning:recommendation_table")}?result={later_run.pk}"')

        recommendations = self.client.get(reverse("planning:recommendation_table"), {"result": self.run.pk})
        self.assertContains(recommendations, "first-run")
        self.assertContains(recommendations, "Recommendation unavailable")
        self.assertNotContains(recommendations, "Review this plan")

    @patch("planning.views.run_pipeline")
    def test_historical_navigation_retains_branch_month_and_mode_without_a_result_id(self, pipeline):
        pipeline.return_value = PipelineRunResult(
            mode="HISTORICAL", planning_month=date(2026, 4, 1),
            branch_code="BIL01", candidate_count=1,
        )
        self.client.post(reverse("planning:run_scenario"), {
            "planning_month": "2026-04-01", "branch": self.branch.pk, "mode": "HISTORICAL",
        })
        response = self.client.get(reverse("home"))
        self.assertContains(response, "mode=HISTORICAL&amp;month=2026-04-01&amp;branch=BIL01")
        self.assertNotContains(response, f"result={self.run.pk}")
        candidate = self.client.get(reverse("analytics:candidate_evidence"), {
            "mode": "HISTORICAL", "month": "2026-04-01", "branch": "BIL01",
        })
        self.assertContains(candidate, "HISTORICAL / RESEARCH PROTOTYPE")
        self.assertContains(candidate, "Historical research context; no decision run is persisted")

    def test_missing_or_tampered_session_result_disables_context_links(self):
        session = self.client.session
        session[SESSION_KEY] = {"result": 999999}
        session.save()
        response = self.client.get(reverse("home"))
        self.assertContains(response, "Run a scenario first to view Forecast Evidence")
        self.assertNotContains(response, "?result=999999")
        session = self.client.session
        session[SESSION_KEY] = {"result": "not-an-id"}
        session.save()
        self.assertEqual(self.client.get(reverse("home")).status_code, 200)

    def test_review_page_navigation_uses_url_run_even_if_session_points_elsewhere(self):
        later_run = self._result("later-run")
        session = self.client.session
        session[SESSION_KEY] = {"result": later_run.pk}
        session.save()
        self.client.force_login(get_user_model().objects.create_user(username="navigation-reviewer"))
        response = self.client.get(reverse("reviews:review_run", args=(self.run.pk,)))
        self.assertContains(response, "first-run")
        self.assertContains(response, f'href="{reverse("planning:recommendation_table")}?result={self.run.pk}"')
        self.assertNotContains(response, f'href="{reverse("planning:recommendation_table")}?result={later_run.pk}"')
        self.assertNotContains(response, "Save Review")
