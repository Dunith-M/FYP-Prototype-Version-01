import csv
from datetime import date
from io import StringIO
import inspect

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from catalog.models import Branch, Product
from optimization.checker import check_and_persist_feasibility
from planning.models import DecisionOption, OptimizationResult, Recommendation, RunProvenance
from reviews.models import ReviewDecision


class ReviewedRecommendationCsvExportTests(TestCase):
    def setUp(self):
        self.reviewer = get_user_model().objects.create_user(username="csv-reviewer", password="test-password")
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="GALLE-A")
        self.product = Product.objects.create(
            canonical_product_id="CP_EXPORT_A",
            product_code="EXPORT-A",
            product_desc='Product, with "quoted" description',
            family_id="F1",
            family_desc="Export family",
        )
        self.option = DecisionOption.objects.create(
            branch=self.branch, product=self.product, planning_month=date(2026, 4, 1),
            l0=True, q0=2, l1=True, q1=5,
            option_type=DecisionOption.OptionType.LISTED_TARGET,
            evidence_score=.75,
            evidence_source=DecisionOption.EvidenceSource.TRACK_B_CANDIDATE_SCORE,
            parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
            scenario_version="export-synthetic-v1",
        )
        self.run = OptimizationResult.objects.create(
            planning_month=date(2026, 4, 1),
            scenario_version="export-synthetic-v1",
            run_version="reviewed-export-run-v1",
            solver_status=OptimizationResult.SolverStatus.OPTIMAL,
            solver_metadata={"pipeline_context": {
                "mode": "RESEARCH_SYNTHETIC",
                "parameter_source": "SYNTHETIC_DEMO",
                "planning_month": "2026-04-01",
                "branch_code": "BIL01",
            }},
        )
        self.run.selected_options.add(self.option)
        check_and_persist_feasibility(self.run.pk)
        self.run.refresh_from_db()
        self.recommendation = Recommendation.objects.create(
            optimization_result=self.run,
            selected_option=self.option,
            branch=self.branch,
            product=self.product,
            action=Recommendation.Action.INCREASE,
            status=Recommendation.Status.READY,
            reasons=[
                {"code": "SAVED_REASON_1", "text": 'Strong demand, review "display capacity".'},
                {"code": "SAVED_REASON_2", "text": "Saved second reason."},
            ],
        )
        RunProvenance.objects.create(
            optimization_result=self.run,
            branch=self.branch,
            mode="RESEARCH_SYNTHETIC",
            source_manifest_path="data/artifacts/feature_month_manifest.json",
            source_manifest_hash="a" * 64,
            source_snapshot={"sales": {"filename": "sales.xlsx", "file_hash": "1" * 64}},
            mapping_version="mapping-export-v1",
            feature_version="feature-export-v1",
            feature_artifact_hash="b" * 64,
            model_manifest_path="data/artifacts/day06/model_manifest.json",
            model_manifest_hash="c" * 64,
            model_type="Baseline: trailing_mean_3",
            model_version="model-export-v1",
            selected_model="trailing_mean_3",
            scenario_config_path="config/synthetic/options.yaml",
            scenario_config_hash="d" * 64,
        )
        self.url = reverse("reviews:export_reviewed_run_csv", args=[self.run.pk])
        self.client.force_login(self.reviewer)

    def _review(self, decision, comment):
        return ReviewDecision.objects.create(
            reviewed_run=self.run,
            reviewer=self.reviewer,
            decision=decision,
            comment=comment,
        )

    def _export_rows(self):
        response = self.client.get(self.url)
        return response, list(csv.DictReader(StringIO(response.content.decode("utf-8"))))

    def test_reviewed_run_exports_saved_rows_with_versioned_filename_and_metadata(self):
        self._review(ReviewDecision.Decision.ACCEPT, "Keep item, but review capacity next month.")

        response, rows = self._export_rows()
        review_page = self.client.get(reverse("reviews:review_run", args=[self.run.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(review_page, "Export CSV")
        self.assertContains(review_page, self.url)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn('attachment; filename="recommendations_reviewed-export-run-v1.csv"', response["Content-Disposition"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["run_version"], self.run.run_version)
        self.assertEqual(row["branch_code"], "BIL01")
        self.assertEqual(row["planning_month"], "2026-04-01")
        self.assertEqual(row["mode"], "RESEARCH_SYNTHETIC")
        self.assertEqual(row["product_id"], "CP_EXPORT_A")
        self.assertEqual(row["q0"], "2")
        self.assertEqual(row["q1"], "5")
        self.assertEqual(row["action_or_status"], "INCREASE")
        self.assertEqual(row["solver_status"], "OPTIMAL")
        self.assertEqual(row["feasibility_status"], "PASS")
        self.assertEqual(row["model_version"], "model-export-v1")
        self.assertEqual(row["mapping_version"], "mapping-export-v1")
        self.assertEqual(row["scenario_version"], "export-synthetic-v1")
        self.assertEqual(row["review_status"], "ACCEPTED")
        self.assertEqual(row["review_decision"], "ACCEPT")
        self.assertIn("not Singer-approved operational advice", row["limitations"])

    def test_export_scopes_rows_to_the_exact_requested_run(self):
        self._review(ReviewDecision.Decision.ACCEPT, "Export exact first run.")
        other_product = Product.objects.create(
            canonical_product_id="CP_EXPORT_B", product_code="EXPORT-B", product_desc="Other run product"
        )
        other_option = DecisionOption.objects.create(
            branch=self.branch, product=other_product, planning_month=date(2026, 5, 1),
            l0=True, q0=1, l1=True, q1=3,
            option_type=DecisionOption.OptionType.LISTED_TARGET,
            parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
            scenario_version="export-synthetic-v2",
        )
        other_run = OptimizationResult.objects.create(
            planning_month=date(2026, 5, 1), scenario_version="export-synthetic-v2",
            run_version="other-export-run-v2", solver_status=OptimizationResult.SolverStatus.OPTIMAL,
            solver_metadata={"pipeline_context": {
                "mode": "RESEARCH_SYNTHETIC", "planning_month": "2026-05-01", "branch_code": "BIL01",
            }},
        )
        other_run.selected_options.add(other_option)
        check_and_persist_feasibility(other_run.pk)
        Recommendation.objects.create(
            optimization_result=other_run, selected_option=other_option, branch=self.branch, product=other_product,
            action=Recommendation.Action.KEEP, status=Recommendation.Status.READY,
        )

        _response, rows = self._export_rows()

        self.assertEqual([row["run_version"] for row in rows], [self.run.run_version])
        self.assertEqual([row["product_id"] for row in rows], ["CP_EXPORT_A"])
        self.assertNotIn("CP_EXPORT_B", _response.content.decode("utf-8"))

    def test_export_keeps_machine_action_separate_from_human_reject(self):
        self._review(ReviewDecision.Decision.REJECT, "Not suitable for this research demonstration.")

        _response, rows = self._export_rows()

        self.assertEqual(rows[0]["action_or_status"], "INCREASE")
        self.assertEqual(rows[0]["review_status"], "REJECTED")
        self.assertEqual(rows[0]["review_decision"], "REJECT")

    def test_modify_export_preserves_original_q1_and_quotes_special_characters(self):
        comment = 'Reduce q1 from 5 to 3, because branch "display capacity" is limited.'
        self._review(ReviewDecision.Decision.MODIFY, comment)

        response, rows = self._export_rows()

        self.assertEqual(rows[0]["q1"], "5")
        self.assertEqual(rows[0]["review_status"], "MODIFICATION_REQUESTED")
        self.assertEqual(rows[0]["review_comment"], comment)
        self.assertEqual(rows[0]["reason_1"], 'Strong demand, review "display capacity".')
        self.assertIn('""display capacity""', response.content.decode("utf-8"))

    def test_unreviewed_run_and_blocked_runs_do_not_export_normal_csv(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 409)
        self.assertIn("persisted human review is required", response.content.decode("utf-8"))

        self._review(ReviewDecision.Decision.ACCEPT, "Review exists but plan later becomes blocked.")
        for solver_status in ("INFEASIBLE", "UNKNOWN", "MODEL_INVALID", "ERROR"):
            with self.subTest(solver_status=solver_status):
                self.run.solver_status = solver_status
                self.run.save(update_fields=("solver_status",))
                response = self.client.get(self.url)
                self.assertEqual(response.status_code, 409)
                self.assertNotEqual(response["Content-Type"], "text/csv; charset=utf-8")

        self.run.solver_status = OptimizationResult.SolverStatus.OPTIMAL
        self.run.feasibility_status = OptimizationResult.FeasibilityStatus.FAIL
        self.run.save(update_fields=("solver_status", "feasibility_status"))
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 409)

    def test_export_is_deterministic_read_only_and_requires_authenticated_user(self):
        self._review(ReviewDecision.Decision.ACCEPT, "Stable export.")
        original_recommendation = (
            self.recommendation.action,
            self.option.q0,
            self.option.q1,
            self.recommendation.reasons,
        )
        review_count = ReviewDecision.objects.count()

        first = self.client.get(self.url)
        second = self.client.get(self.url)
        self.recommendation.refresh_from_db()
        self.option.refresh_from_db()

        self.assertEqual(first.content, second.content)
        self.assertEqual(ReviewDecision.objects.count(), review_count)
        self.assertEqual(
            (self.recommendation.action, self.option.q0, self.option.q1, self.recommendation.reasons),
            original_recommendation,
        )
        self.client.logout()
        anonymous = self.client.get(self.url)
        self.assertEqual(anonymous.status_code, 302)
        self.assertTrue(anonymous.url.startswith("/admin/login/"))

    def test_filename_uses_only_safe_characters_from_the_persisted_run_identifier(self):
        from reviews.services.export_service import _filename_for

        unsafe_identifier_run = OptimizationResult(run_version="../unsafe\\run name")

        self.assertEqual(_filename_for(unsafe_identifier_run), "recommendations_unsafe_run_name.csv")

    def test_export_view_and_service_do_not_execute_the_pipeline(self):
        from reviews.services.export_service import build_recommendation_csv
        from reviews.views import export_reviewed_run_csv

        source = inspect.getsource(build_recommendation_csv) + inspect.getsource(export_reviewed_run_csv)
        for forbidden_operation in (
            "run_pipeline", "run_and_persist_optimization", "check_and_persist_feasibility",
            "create_recommendations_for_result", "derive_action", "generate_explanations", "train_model",
        ):
            self.assertNotIn(forbidden_operation, source)
