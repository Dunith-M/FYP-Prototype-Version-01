from datetime import date
import inspect

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from catalog.models import Branch, Product
from optimization.checker import check_and_persist_feasibility
from planning.models import DecisionOption, OptimizationResult, Recommendation, RunProvenance
from reviews.models import ReviewDecision


class RecommendationTablePageTests(TestCase):
    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="GALLE-A")
        self.product = Product.objects.create(
            canonical_product_id="CP_REC", product_code="REC", product_desc="Saved recommendation product",
            family_id="F1", family_desc="Saved family",
        )
        self.option = DecisionOption.objects.create(
            branch=self.branch, product=self.product, planning_month=date(2026, 4, 1),
            l0=True, q0=2, l1=True, q1=4,
            option_type=DecisionOption.OptionType.LISTED_TARGET,
            evidence_score=.75,
            evidence_source=DecisionOption.EvidenceSource.HISTORICAL_LAG_1_NORMALIZED,
            parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
            scenario_version="test_synthetic_options_v1",
        )
        self.result = OptimizationResult.objects.create(
            planning_month=date(2026, 4, 1), scenario_version="test_synthetic_options_v1",
            run_version="day09_web_bil01_202604_test", solver_status=OptimizationResult.SolverStatus.OPTIMAL,
            solver_metadata={"pipeline_context": {
                "mode": "RESEARCH_SYNTHETIC", "parameter_source": "SYNTHETIC_DEMO",
                "planning_month": "2026-04-01", "branch_code": "BIL01",
            }},
        )
        self.result.selected_options.add(self.option)
        check_and_persist_feasibility(self.result.pk)
        self.result.refresh_from_db()
        # Intentionally inconsistent with q0/q1: the page must render stored KEEP,
        # proving it does not independently derive the action.
        self.recommendation = Recommendation.objects.create(
            optimization_result=self.result, selected_option=self.option, branch=self.branch, product=self.product,
            action=Recommendation.Action.KEEP, status=Recommendation.Status.READY,
            reasons=[{"code": "SAVED_REASON", "text": "Saved backend explanation."}],
        )
        RunProvenance.objects.create(
            optimization_result=self.result, branch=self.branch, mode="RESEARCH_SYNTHETIC",
            source_manifest_path="data/artifacts/feature_month_manifest.json", source_manifest_hash="a" * 64,
            source_snapshot={"sales": {"filename": "sales.xlsx", "file_hash": "1" * 64}},
            mapping_version="mapping-test-v1", feature_version="feature-test-v1", feature_artifact_hash="b" * 64,
            model_manifest_path="data/artifacts/day06/model_manifest.json", model_manifest_hash="c" * 64,
            model_type="Baseline: trailing_mean_3", model_version="model-test-v1", selected_model="trailing_mean_3",
            scenario_config_path="config/synthetic/options.yaml", scenario_config_hash="d" * 64,
        )

    def test_page_displays_saved_action_without_rederiving_it(self):
        response = self.client.get(reverse("planning:recommendation_table"), {"result": self.result.pk})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "BIL01")
        self.assertContains(response, "April 2026")
        self.assertContains(response, "RESEARCH_SYNTHETIC")
        self.assertContains(response, "OPTIMAL")
        self.assertContains(response, "PASS")
        self.assertContains(response, "Saved recommendation product")
        self.assertContains(response, "2")
        self.assertContains(response, "4")
        self.assertContains(response, "KEEP")
        self.assertContains(response, "0.750")
        self.assertContains(response, "HISTORICAL_LAG_1_NORMALIZED")
        self.assertContains(response, "Saved backend explanation.")
        self.assertNotContains(response, "INCREASE")
        self.assertContains(response, "SYNTHETIC DECISION DEMO")
        self.assertContains(response, "Run Information")
        self.assertContains(response, "day09_web_bil01_202604_test")
        self.assertContains(response, "GALLE-A")
        self.assertContains(response, "mapping-test-v1")
        self.assertContains(response, "model-test-v1")
        self.assertContains(response, "test_synthetic_options_v1")
        self.assertContains(response, "Current Review Status:")
        self.assertContains(response, "PENDING")
        self.assertContains(response, f'href="{reverse("reviews:review_run", args=(self.result.pk,))}"')

    def test_page_shows_current_human_status_without_changing_machine_action(self):
        reviewer = get_user_model().objects.create_user(username="recommendation-reviewer")
        ReviewDecision.objects.create(
            reviewed_run=self.result,
            reviewer=reviewer,
            decision=ReviewDecision.Decision.REJECT,
            comment="Development test review; not an operational decision.",
        )

        response = self.client.get(reverse("planning:recommendation_table"), {"result": self.result.pk})
        self.recommendation.refresh_from_db()

        self.assertContains(response, "REJECTED")
        self.assertContains(response, "KEEP")
        self.assertEqual(self.recommendation.action, Recommendation.Action.KEEP)

    def test_get_is_read_only_and_does_not_execute_the_pipeline(self):
        before = Recommendation.objects.count()
        provenance = RunProvenance.objects.get(optimization_result=self.result)
        metadata_before = provenance.mapping_version, provenance.model_version

        first = self.client.get(reverse("planning:recommendation_table"), {"result": self.result.pk})
        second = self.client.get(reverse("planning:recommendation_table"), {"result": self.result.pk})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(Recommendation.objects.count(), before)
        provenance.refresh_from_db()
        self.assertEqual((provenance.mapping_version, provenance.model_version), metadata_before)
        from planning.views import recommendation_table

        source = inspect.getsource(recommendation_table)
        for forbidden_operation in (
            "run_and_persist_optimization", "check_and_persist_feasibility",
            "create_recommendations_for_result", "derive_action", "generate_explanations", "run_pipeline",
        ):
            self.assertNotIn(forbidden_operation, source)

    def test_historical_context_never_receives_synthetic_actions(self):
        response = self.client.get(
            reverse("planning:recommendation_table"),
            {"mode": "HISTORICAL", "branch": "BIL01", "month": "2026-04-01"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "HISTORICAL / RESEARCH PROTOTYPE")
        self.assertContains(response, "No decision recommendations were produced")
        self.assertNotContains(response, "Saved backend explanation.")
        self.assertContains(response, "Not applicable for Historical evidence context")
        self.assertNotContains(response, "test_synthetic_options_v1")

    def test_only_usable_solver_states_with_pass_are_published(self):
        for status in (OptimizationResult.SolverStatus.OPTIMAL, OptimizationResult.SolverStatus.FEASIBLE):
            with self.subTest(status=status):
                self.result.solver_status = status
                self.result.feasibility_status = OptimizationResult.FeasibilityStatus.PASS
                self.result.save(update_fields=("solver_status", "feasibility_status"))
                response = self.client.get(reverse("planning:recommendation_table"), {"result": self.result.pk})
                self.assertContains(response, "Saved backend explanation.")

    def test_unusable_solver_statuses_and_stale_rows_are_blocked(self):
        for status in ("INFEASIBLE", "UNKNOWN", "MODEL_INVALID", "ERROR"):
            with self.subTest(status=status):
                self.result.solver_status = status
                self.result.feasibility_status = OptimizationResult.FeasibilityStatus.PASS
                self.result.save(update_fields=("solver_status", "feasibility_status"))
                response = self.client.get(reverse("planning:recommendation_table"), {"result": self.result.pk})
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "Recommendation unavailable — optimization/checking did not pass.")
                self.assertContains(response, status)
                self.assertContains(response, "Run Information")
                self.assertContains(response, "mapping-test-v1")
                self.assertContains(response, "model-test-v1")
                self.assertNotContains(response, "Saved backend explanation.")
                self.assertNotContains(response, "Review this plan")

    def test_checker_fail_or_missing_blocks_existing_recommendation_rows(self):
        for feasibility_status in (OptimizationResult.FeasibilityStatus.FAIL, OptimizationResult.FeasibilityStatus.NOT_RUN):
            with self.subTest(feasibility_status=feasibility_status):
                self.result.solver_status = OptimizationResult.SolverStatus.OPTIMAL
                self.result.feasibility_status = feasibility_status
                self.result.save(update_fields=("solver_status", "feasibility_status"))
                response = self.client.get(reverse("planning:recommendation_table"), {"result": self.result.pk})
                self.assertContains(response, "Recommendation unavailable — optimization/checking did not pass.")
                self.assertContains(response, feasibility_status)
                self.assertNotContains(response, "Saved backend explanation.")

    def test_missing_result_is_safely_blocked_by_404(self):
        self.assertEqual(
            self.client.get(reverse("planning:recommendation_table"), {"result": 999999}).status_code,
            404,
        )
