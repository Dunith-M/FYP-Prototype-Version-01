from datetime import date

from django.db import IntegrityError, transaction
from django.test import TestCase

from catalog.models import Branch, Product
from planning.models import DecisionOption


class DecisionOptionModelTests(TestCase):
    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="Branch")
        self.product = Product.objects.create(canonical_product_id="CP_OPT", product_code="OPT", product_desc="Option")

    def _option(self, **overrides):
        values = {"branch": self.branch, "product": self.product, "planning_month": date(2026, 4, 1), "l0": False, "q0": None, "l1": True, "q1": 1, "option_type": DecisionOption.OptionType.LIST_TRIAL, "scenario_version": "day07_v1"}
        return DecisionOption(**(values | overrides))

    def test_valid_option_allows_null_q0_and_synthetic_provenance(self):
        option = self._option()
        option.full_clean(); option.save()
        self.assertEqual(option.parameter_source, "SYNTHETIC_DEMO")
        self.assertIsNone(option.q0)

    def test_constraints_reject_duplicates_and_invalid_l1_q1_state(self):
        self._option().save()
        with transaction.atomic():
            with self.assertRaises(IntegrityError): self._option().save()
        with self.assertRaises(Exception):
            self._option(l1=False, q1=1).full_clean()
        self.assertNotIn("INTRODUCE", {value for value, _ in DecisionOption.OptionType.choices})
