from datetime import date

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase

from catalog.models import Branch, Product
from planning.models import DecisionOption, OptimizationResult, Recommendation


class Day08OutputModelTests(TestCase):
    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="Branch")
        self.product = Product.objects.create(canonical_product_id="CP_DAY08", product_code="DAY08", product_desc="Day 08 product")
        self.option = DecisionOption.objects.create(
            branch=self.branch,
            product=self.product,
            planning_month=date(2026, 4, 1),
            l0=False,
            q0=None,
            l1=True,
            q1=1,
            option_type=DecisionOption.OptionType.LIST_TRIAL,
            scenario_version="day07_synthetic_options_v1",
        )

    def _result(self, **overrides):
        values = {
            "planning_month": date(2026, 4, 1),
            "scenario_version": "day07_synthetic_options_v1",
            "run_version": "day08_optimizer_v1",
            "solver_status": OptimizationResult.SolverStatus.OPTIMAL,
            "feasibility_status": OptimizationResult.FeasibilityStatus.PASS,
        }
        return OptimizationResult.objects.create(**(values | overrides))

    def test_result_tracks_selected_options_and_distinct_statuses(self):
        result = self._result()
        result.selected_options.add(self.option)
        self.assertEqual(list(result.selected_options.all()), [self.option])
        self.assertEqual(result.solver_status, "OPTIMAL")
        self.assertEqual(result.feasibility_status, "PASS")

    def test_recommendation_keeps_withheld_as_status_not_action(self):
        recommendation = Recommendation(
            optimization_result=self._result(),
            branch=self.branch,
            product=self.product,
            status=Recommendation.Status.WITHHELD,
            action=None,
            reasons=["Required evidence is unavailable."],
        )
        recommendation.full_clean()
        recommendation.save()
        self.assertIsNone(recommendation.action)
        self.assertEqual(recommendation.status, "WITHHELD")
        self.assertNotIn("WITHHELD", {value for value, _ in Recommendation.Action.choices})

    def test_ready_recommendation_requires_action_and_is_unique_per_result_pair(self):
        result = self._result()
        invalid = Recommendation(
            optimization_result=result,
            branch=self.branch,
            product=self.product,
            status=Recommendation.Status.READY,
            action=None,
        )
        with self.assertRaises(ValidationError):
            invalid.full_clean()
        result.selected_options.add(self.option)
        recommendation = Recommendation(
            optimization_result=result,
            selected_option=self.option,
            branch=self.branch,
            product=self.product,
            status=Recommendation.Status.READY,
            action=Recommendation.Action.INTRODUCE,
        )
        recommendation.full_clean()
        recommendation.save()
        with transaction.atomic():
            with self.assertRaises(IntegrityError):
                Recommendation.objects.create(
                    optimization_result=result,
                    selected_option=self.option,
                    branch=self.branch,
                    product=self.product,
                    status=Recommendation.Status.WITHHELD,
                )
