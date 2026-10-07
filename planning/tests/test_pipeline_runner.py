from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from planning.services.pipeline_runner import PipelineRunError, run_pipeline


class PipelineRunnerTests(SimpleTestCase):
    planning_month = date(2026, 4, 1)
    branch = SimpleNamespace(branch_code="BIL01")

    @patch("planning.services.pipeline_runner._require_persisted_evidence", return_value=20)
    @patch("planning.services.pipeline_runner._resolve_synthetic_scenario")
    @patch("planning.services.pipeline_runner.check_and_persist_feasibility")
    @patch("planning.services.pipeline_runner.create_recommendations_for_result")
    @patch("planning.services.pipeline_runner.run_and_persist_optimization")
    def test_historical_mode_does_not_load_synthetic_inputs_or_actions(
        self, optimizer, recommendations, checker, scenario, evidence
    ):
        outcome = run_pipeline(
            planning_month=self.planning_month,
            branch=self.branch,
            mode="HISTORICAL",
        )

        self.assertIsNone(outcome.optimization_result_id)
        self.assertEqual(outcome.candidate_count, 20)
        evidence.assert_called_once_with(planning_month=self.planning_month, branch=self.branch)
        scenario.assert_not_called()
        optimizer.assert_not_called()
        checker.assert_not_called()
        recommendations.assert_not_called()

    @patch("planning.services.pipeline_runner.create_recommendations_for_result")
    @patch("planning.services.pipeline_runner.check_and_persist_feasibility")
    @patch("planning.services.pipeline_runner.run_and_persist_optimization")
    @patch("planning.services.pipeline_runner._existing_result", return_value=None)
    @patch("planning.services.pipeline_runner._resolve_synthetic_scenario", return_value="day07_synthetic_options_v1")
    @patch("planning.services.pipeline_runner._require_persisted_evidence", return_value=20)
    def test_usable_solver_then_checker_pass_publishes_existing_action_and_explanation_pipeline(
        self, evidence, scenario, existing, optimizer, checker, recommendations
    ):
        result = SimpleNamespace(
            pk=17, solver_status="OPTIMAL", feasibility_status="NOT_RUN", solver_metadata={},
            save=lambda **kwargs: None,
        )
        optimizer.return_value = SimpleNamespace(optimization_result=result)
        checker.return_value = SimpleNamespace(status="PASS")
        recommendations.return_value = SimpleNamespace(created_count=3, existing_count=2)

        with patch("planning.services.pipeline_runner.capture_run_provenance") as provenance:
            outcome = run_pipeline(
                planning_month=self.planning_month,
                branch=self.branch,
                mode="RESEARCH_SYNTHETIC",
            )

        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.optimization_result_id, 17)
        self.assertEqual(outcome.recommendation_count, 5)
        optimizer.assert_called_once()
        checker.assert_called_once_with(17)
        recommendations.assert_called_once_with(17)
        provenance.assert_called_once_with(
            optimization_result=result, branch=self.branch, mode="RESEARCH_SYNTHETIC"
        )

    @patch("planning.services.pipeline_runner.create_recommendations_for_result")
    @patch("planning.services.pipeline_runner.check_and_persist_feasibility")
    @patch("planning.services.pipeline_runner.run_and_persist_optimization")
    @patch("planning.services.pipeline_runner._existing_result", return_value=None)
    @patch("planning.services.pipeline_runner._resolve_synthetic_scenario", return_value="day07_synthetic_options_v1")
    @patch("planning.services.pipeline_runner._require_persisted_evidence", return_value=20)
    def test_solver_failure_blocks_checker_and_recommendations(
        self, evidence, scenario, existing, optimizer, checker, recommendations
    ):
        result = SimpleNamespace(
            pk=18, solver_status="INFEASIBLE", feasibility_status="NOT_RUN", solver_metadata={},
            save=lambda **kwargs: None,
        )
        optimizer.return_value = SimpleNamespace(optimization_result=result)

        with patch("planning.services.pipeline_runner.capture_run_provenance") as provenance:
            outcome = run_pipeline(
                planning_month=self.planning_month,
                branch=self.branch,
                mode="RESEARCH_SYNTHETIC",
            )

        self.assertEqual(outcome.solver_status, "INFEASIBLE")
        provenance.assert_called_once_with(
            optimization_result=result, branch=self.branch, mode="RESEARCH_SYNTHETIC"
        )
        checker.assert_not_called()
        recommendations.assert_not_called()

    @patch("planning.services.pipeline_runner.create_recommendations_for_result")
    @patch("planning.services.pipeline_runner.check_and_persist_feasibility")
    @patch("planning.services.pipeline_runner.run_and_persist_optimization")
    @patch("planning.services.pipeline_runner._existing_result", return_value=None)
    @patch("planning.services.pipeline_runner._resolve_synthetic_scenario", return_value="day07_synthetic_options_v1")
    @patch("planning.services.pipeline_runner._require_persisted_evidence", return_value=20)
    def test_feasibility_fail_blocks_recommendations(
        self, evidence, scenario, existing, optimizer, checker, recommendations
    ):
        result = SimpleNamespace(
            pk=19, solver_status="FEASIBLE", feasibility_status="NOT_RUN", solver_metadata={},
            save=lambda **kwargs: None,
        )
        optimizer.return_value = SimpleNamespace(optimization_result=result)
        checker.return_value = SimpleNamespace(status="FAIL")

        with patch("planning.services.pipeline_runner.capture_run_provenance") as provenance:
            outcome = run_pipeline(
                planning_month=self.planning_month,
                branch=self.branch,
                mode="RESEARCH_SYNTHETIC",
            )

        self.assertEqual(outcome.feasibility_status, "FAIL")
        provenance.assert_called_once_with(
            optimization_result=result, branch=self.branch, mode="RESEARCH_SYNTHETIC"
        )
        recommendations.assert_not_called()

    def test_unsupported_mode_is_rejected(self):
        with self.assertRaises(PipelineRunError):
            run_pipeline(planning_month=self.planning_month, branch=self.branch, mode="OPERATIONAL")
