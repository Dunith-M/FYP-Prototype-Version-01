from datetime import date

from django.test import TestCase

from catalog.models import Branch, Product
from planning.models import DecisionOption, OptimizationResult
from planning.services.pipeline_runner import _persist_synthetic_run_context
from planning.services.publication_gate import evaluate_publication_gate


class ModeSeparationPersistenceTests(TestCase):
    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="GALLE-A")
        self.product = Product.objects.create(
            canonical_product_id="CP_MODE", product_code="MODE", product_desc="Mode test product"
        )
        self.result = OptimizationResult.objects.create(
            planning_month=date(2026, 4, 1),
            scenario_version="mode_test", run_version="mode_test_run",
            solver_status=OptimizationResult.SolverStatus.OPTIMAL,
        )

    def test_synthetic_run_context_persists_explicit_mode_and_demo_provenance(self):
        _persist_synthetic_run_context(
            self.result,
            planning_month=date(2026, 4, 1),
            branch=self.branch,
            candidate_count=20,
        )
        self.result.refresh_from_db()

        context = self.result.solver_metadata["pipeline_context"]
        self.assertEqual(context["mode"], "RESEARCH_SYNTHETIC")
        self.assertEqual(context["parameter_source"], "SYNTHETIC_DEMO")
        self.assertEqual(context["branch_code"], "BIL01")

    def test_publication_gate_rejects_non_synthetic_option_provenance(self):
        option = DecisionOption.objects.create(
            branch=self.branch,
            product=self.product,
            planning_month=date(2026, 4, 1),
            l0=False,
            q0=None,
            l1=True,
            q1=1,
            option_type=DecisionOption.OptionType.LIST_TRIAL,
            scenario_version="mode_test",
            parameter_source="HISTORICAL_EVIDENCE",
        )
        self.result.selected_options.add(option)

        gate = evaluate_publication_gate(self.result.pk)

        self.assertFalse(gate.allowed)
        self.assertEqual(gate.code, "PUB-GATE-007")

    def test_confirmation_uses_persisted_mode_not_result_id_inference(self):
        _persist_synthetic_run_context(
            self.result,
            planning_month=date(2026, 4, 1),
            branch=self.branch,
            candidate_count=20,
        )

        response = self.client.get(f"/run/confirmation/?result={self.result.pk}")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "SYNTHETIC DECISION DEMO")
