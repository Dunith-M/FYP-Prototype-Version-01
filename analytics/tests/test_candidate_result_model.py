from datetime import date

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase

from analytics.models import CandidateResult
from catalog.models import Branch, Product


class CandidateResultModelTests(TestCase):
    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="Branch")
        self.product = Product.objects.create(canonical_product_id="CP_TEST", product_code="TEST", product_desc="Test product")

    def _result(self, **overrides):
        values = {
            "branch": self.branch, "product": self.product, "planning_month": date(2026, 4, 1),
            "peer_score": 0.5, "family_score": 0.6, "similar_branch_score": None, "price_fit_score": None,
            "candidate_score": 0.55, "rank": 1, "evidence_class": CandidateResult.EvidenceClass.INSUFFICIENT,
            "run_version": "day07_candidate_score_v1",
        }
        return CandidateResult(**(values | overrides))

    def test_valid_research_result_persists_with_nullable_scores(self):
        result = self._result()
        result.full_clean()
        result.save()
        result.refresh_from_db()
        self.assertEqual(result.status, CandidateResult.CandidateStatus.RESEARCH_SHORTLIST)
        self.assertEqual(result.candidate_score, 0.55)
        self.assertIsNone(result.similar_branch_score)
        self.assertEqual(result.branch, self.branch)
        self.assertEqual(result.product, self.product)

    def test_unique_grain_allows_a_different_run_version(self):
        self._result().save()
        with transaction.atomic():
            with self.assertRaises(IntegrityError):
                self._result().save()
        self._result(run_version="day07_candidate_score_v2").save()
        self.assertEqual(CandidateResult.objects.count(), 2)

    def test_status_choices_do_not_offer_introduce(self):
        values = {value for value, _ in CandidateResult.CandidateStatus.choices}
        self.assertEqual(values, {"RESEARCH_SHORTLIST"})
        with self.assertRaises(ValidationError):
            self._result(status="INTRODUCE").full_clean()
