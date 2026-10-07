from datetime import date

import pandas as pd
from django.test import TestCase

from analytics.models import CandidateResult
from analytics.services.candidate_result_loader import load_candidate_results
from catalog.models import Branch, Product


class CandidateResultLoaderTests(TestCase):
    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="Branch")
        self.product = Product.objects.create(
            canonical_product_id="CP_LOAD", product_code="LOAD", product_desc="Loader product", family_id="F1", family_desc="Family"
        )

    def _frame(self):
        return pd.DataFrame([{
            "branch_code": "BIL01", "planning_month": "2026-04", "canonical_product_id": "CP_LOAD",
            "peer_product_demand_score": None, "branch_family_demand_score": 0.8,
            "similar_branch_product_score": None, "price_fit_score": 0.5,
            "candidate_score": 0.7, "candidate_rank": 1, "evidence_class": "LIMITED",
            "status": "RESEARCH_SHORTLIST", "parameter_source": "HISTORICAL_EVIDENCE",
            "score_method_version": "day07_candidate_score_v1",
        }])

    def test_loads_existing_evidence_idempotently(self):
        self.assertEqual(load_candidate_results(self._frame()), (1, 0))
        self.assertEqual(load_candidate_results(self._frame()), (0, 1))
        result = CandidateResult.objects.get()
        self.assertEqual(result.planning_month, date(2026, 4, 1))
        self.assertIsNone(result.peer_score)
        self.assertEqual(result.status, CandidateResult.CandidateStatus.RESEARCH_SHORTLIST)

    def test_rejects_non_research_status(self):
        frame = self._frame()
        frame.loc[0, "status"] = "INTRODUCE"
        with self.assertRaisesRegex(ValueError, "RESEARCH_SHORTLIST"):
            load_candidate_results(frame)
