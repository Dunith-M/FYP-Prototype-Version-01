from datetime import date
import inspect

from django.test import TestCase
from django.urls import reverse

from analytics.models import CandidateResult
from catalog.models import Branch, Product
from planning.models import OptimizationResult, Recommendation


class CandidateEvidencePageTests(TestCase):
    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="GALLE-A")
        self.other_branch = Branch.objects.create(branch_code="BIN01", branch_name="Other")
        self.product_a = Product.objects.create(
            canonical_product_id="CP_A", product_code="A", product_desc="Alpha", family_id="F1", family_desc="Family One"
        )
        self.product_b = Product.objects.create(
            canonical_product_id="CP_B", product_code="B", product_desc="Beta", family_id="F2", family_desc="Family Two"
        )
        for product, rank, score, similar in ((self.product_b, 2, .5, None), (self.product_a, 1, .8, .7)):
            CandidateResult.objects.create(
                branch=self.branch, product=product, planning_month=date(2026, 4, 1),
                peer_score=.8, family_score=.9, similar_branch_score=similar, price_fit_score=.6,
                candidate_score=score, rank=rank, evidence_class="INSUFFICIENT", run_version="day07_candidate_score_v1",
            )
        CandidateResult.objects.create(
            branch=self.other_branch, product=self.product_a, planning_month=date(2026, 5, 1),
            peer_score=.5, family_score=.5, similar_branch_score=None, price_fit_score=None,
            candidate_score=.5, rank=1, evidence_class="LIMITED", run_version="day07_candidate_score_v1",
        )
        self.result = OptimizationResult.objects.create(
            planning_month=date(2026, 4, 1),
            scenario_version="day07_synthetic_options_v1",
            run_version="day09_web_bil01_202604_day07_synthetic_options_v1",
            solver_status=OptimizationResult.SolverStatus.OPTIMAL,
            solver_metadata={
                "pipeline_context": {
                    "mode": "RESEARCH_SYNTHETIC",
                    "parameter_source": "SYNTHETIC_DEMO",
                    "planning_month": "2026-04-01",
                    "branch_code": "BIL01",
                    "candidate_evidence_count": 2,
                }
            },
        )

    def test_contextual_page_reads_persisted_rows_in_rank_order(self):
        before = (CandidateResult.objects.count(), Recommendation.objects.count())

        response = self.client.get(reverse("analytics:candidate_evidence"), {"result": self.result.pk})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "BIL01 - GALLE-A")
        self.assertContains(response, "April 2026")
        self.assertContains(response, "RESEARCH_SYNTHETIC")
        self.assertContains(response, "RESEARCH SHORTLIST")
        self.assertContains(response, "Alpha")
        self.assertContains(response, "0.800")
        self.assertContains(response, "0.900")
        self.assertContains(response, "0.700")
        self.assertContains(response, "0.600")
        self.assertContains(response, "INSUFFICIENT")
        self.assertContains(response, "SYNTHETIC DECISION DEMO")
        self.assertNotContains(response, "Singer should introduce")
        self.assertLess(response.content.find(b"Alpha"), response.content.find(b"Beta"))
        self.assertEqual(before, (CandidateResult.objects.count(), Recommendation.objects.count()))

    def test_historical_context_is_research_evidence_not_an_action(self):
        response = self.client.get(
            reverse("analytics:candidate_evidence"),
            {"mode": "HISTORICAL", "branch": "BIL01", "month": "2026-04-01"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "HISTORICAL / RESEARCH PROTOTYPE")
        self.assertContains(response, "RESEARCH SHORTLIST")
        self.assertNotContains(response, "Singer should introduce")

    def test_empty_context_is_safe_and_invalid_result_is_404(self):
        CandidateResult.objects.filter(branch=self.branch).delete()

        response = self.client.get(
            reverse("analytics:candidate_evidence"),
            {"mode": "HISTORICAL", "branch": "BIL01", "month": "2026-04-01"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No saved research candidate evidence was found")
        self.assertEqual(
            self.client.get(reverse("analytics:candidate_evidence"), {"result": 999999}).status_code,
            404,
        )

    def test_candidate_page_reads_saved_results_without_recomputing(self):
        from analytics.views import candidate_evidence

        source = inspect.getsource(candidate_evidence)
        for forbidden_operation in (
            "build_candidate_pools", "generate_candidates", "read_excel", "read_parquet",
            "run_pipeline", "run_and_persist_optimization", "fit(",
        ):
            self.assertNotIn(forbidden_operation, source)
