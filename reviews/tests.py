from datetime import date

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.urls import reverse

from analytics.models import CandidateResult
from catalog.models import Branch, Product
from optimization.checker import check_and_persist_feasibility
from planning.models import DecisionOption, OptimizationResult, Recommendation
from reviews.models import ReviewDecision


class ReviewDecisionModelTests(TestCase):
    def setUp(self):
        self.reviewer = get_user_model().objects.create_user(username="reviewer", password="test-password")
        self.run = OptimizationResult.objects.create(
            planning_month=date(2026, 4, 1),
            scenario_version="review-test-synthetic-v1",
            run_version="review-test-run-v1",
            solver_status=OptimizationResult.SolverStatus.OPTIMAL,
            feasibility_status=OptimizationResult.FeasibilityStatus.PASS,
        )

    def _create(self, decision):
        return ReviewDecision.objects.create(
            reviewed_run=self.run,
            reviewer=self.reviewer,
            decision=decision,
            comment=f"Development verification for {decision}; not a Singer operational approval.",
        )

    def test_accept_reject_and_modify_are_persisted(self):
        decisions = [self._create(choice) for choice in ReviewDecision.Decision.values]

        self.assertEqual([decision.decision for decision in decisions], ["ACCEPT", "REJECT", "MODIFY"])
        self.assertTrue(all(decision.reviewed_at is not None for decision in decisions))

    def test_review_traces_to_the_exact_run_version_and_reviewer(self):
        review = self._create(ReviewDecision.Decision.ACCEPT)

        self.assertEqual(review.reviewed_run, self.run)
        self.assertEqual(review.reviewed_run.run_version, "review-test-run-v1")
        self.assertEqual(review.reviewer, self.reviewer)
        self.assertIn("Development verification", review.comment)

    def test_later_run_does_not_move_an_existing_review_target(self):
        review = self._create(ReviewDecision.Decision.MODIFY)
        later_run = OptimizationResult.objects.create(
            planning_month=date(2026, 5, 1),
            scenario_version="review-test-synthetic-v2",
            run_version="review-test-run-v2",
        )

        review.refresh_from_db()
        self.assertEqual(review.reviewed_run_id, self.run.pk)
        self.assertNotEqual(review.reviewed_run_id, later_run.pk)

    def test_review_creation_does_not_mutate_the_reviewed_run(self):
        original_state = (
            self.run.run_version,
            self.run.solver_status,
            self.run.feasibility_status,
        )

        self._create(ReviewDecision.Decision.REJECT)
        self.run.refresh_from_db()

        self.assertEqual(
            (self.run.run_version, self.run.solver_status, self.run.feasibility_status),
            original_state,
        )

    def test_current_status_is_derived_from_the_newest_review_for_this_run(self):
        status, latest_review = ReviewDecision.current_status_for_run(self.run)
        self.assertEqual(status, ReviewDecision.CurrentStatus.PENDING)
        self.assertIsNone(latest_review)

        accepted = self._create(ReviewDecision.Decision.ACCEPT)
        status, latest_review = ReviewDecision.current_status_for_run(self.run)
        self.assertEqual(status, ReviewDecision.CurrentStatus.ACCEPTED)
        self.assertEqual(latest_review.pk, accepted.pk)

        rejected = self._create(ReviewDecision.Decision.REJECT)
        status, latest_review = ReviewDecision.current_status_for_run(self.run)
        self.assertEqual(status, ReviewDecision.CurrentStatus.REJECTED)
        self.assertEqual(latest_review.pk, rejected.pk)

        modification = self._create(ReviewDecision.Decision.MODIFY)
        status, latest_review = ReviewDecision.current_status_for_run(self.run)
        self.assertEqual(status, ReviewDecision.CurrentStatus.MODIFICATION_REQUESTED)
        self.assertEqual(latest_review.pk, modification.pk)
        self.assertEqual(self.run.review_decisions.count(), 3)

    def test_invalid_decision_and_blank_comment_are_rejected_by_model_validation(self):
        review = ReviewDecision(
            reviewed_run=self.run,
            reviewer=self.reviewer,
            decision="APPROVE",
            comment="",
        )

        with self.assertRaises(ValidationError) as error:
            review.full_clean()
        self.assertIn("decision", error.exception.message_dict)
        self.assertIn("comment", error.exception.message_dict)


class ReviewPageTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.reviewer = user_model.objects.create_user(username="reviewer", password="test-password")
        self.other_user = user_model.objects.create_user(username="other", password="test-password")
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="GALLE-A")
        self.product = Product.objects.create(
            canonical_product_id="CP_REVIEW", product_code="REVIEW", product_desc="Review product",
            family_id="F1", family_desc="Review family",
        )
        self.option = DecisionOption.objects.create(
            branch=self.branch, product=self.product, planning_month=date(2026, 4, 1),
            l0=True, q0=2, l1=True, q1=4,
            option_type=DecisionOption.OptionType.LISTED_TARGET,
            evidence_score=.75,
            evidence_source=DecisionOption.EvidenceSource.TRACK_B_CANDIDATE_SCORE,
            parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
            scenario_version="review-page-synthetic-v1",
        )
        self.run = OptimizationResult.objects.create(
            planning_month=date(2026, 4, 1), scenario_version="review-page-synthetic-v1",
            run_version="review-page-run-v1", solver_status=OptimizationResult.SolverStatus.OPTIMAL,
            solver_metadata={"pipeline_context": {
                "mode": "RESEARCH_SYNTHETIC", "parameter_source": "SYNTHETIC_DEMO",
                "planning_month": "2026-04-01", "branch_code": "BIL01",
            }},
        )
        self.run.selected_options.add(self.option)
        check_and_persist_feasibility(self.run.pk)
        self.run.refresh_from_db()
        self.recommendation = Recommendation.objects.create(
            optimization_result=self.run, selected_option=self.option, branch=self.branch, product=self.product,
            action=Recommendation.Action.KEEP, status=Recommendation.Status.READY,
            reasons=[{"code": "SAVED_REASON", "text": "Saved backend explanation."}],
        )
        CandidateResult.objects.create(
            branch=self.branch, product=self.product, planning_month=date(2026, 4, 1),
            peer_score=.7, family_score=.6, similar_branch_score=.5, price_fit_score=.4,
            candidate_score=.65, rank=1, evidence_class=CandidateResult.EvidenceClass.ADEQUATE,
            run_version="candidate-review-test-v1",
        )
        self.url = reverse("reviews:review_run", args=[self.run.pk])
        self.client.force_login(self.reviewer)

    def _post_review(self, decision, comment, **extra):
        return self.client.post(self.url, {"decision": decision, "comment": comment, **extra})

    def test_get_shows_saved_evidence_before_review_controls(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        for expected in (
            "Review Recommendation Plan", "Review product", "KEEP", "2", "4",
            "Saved backend explanation.", "RESEARCH_SYNTHETIC", "review-page-run-v1",
            "BIL01", "April 2026", "0.650", "ADEQUATE", "Review Decision", "csrfmiddlewaretoken",
            "Current Human Review", "PENDING", "Not recorded",
        ):
            self.assertContains(response, expected)

    def test_anonymous_user_is_redirected_to_existing_admin_login(self):
        self.client.logout()

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith("/admin/login/"))

    def test_accept_creates_review_for_exact_run_and_uses_request_user(self):
        response = self._post_review(
            ReviewDecision.Decision.ACCEPT,
            "Reasonable for this research demonstration.",
            reviewer=self.other_user.pk,
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith(self.url + "?saved_review="))
        review = ReviewDecision.objects.get(reviewed_run=self.run)
        self.assertEqual(review.decision, ReviewDecision.Decision.ACCEPT)
        self.assertEqual(review.reviewer, self.reviewer)
        self.assertEqual(review.comment, "Reasonable for this research demonstration.")
        self.assertEqual(review.reviewed_run.run_version, "review-page-run-v1")
        confirmation = self.client.get(self.url)
        self.assertContains(confirmation, "Review saved successfully.")
        self.assertEqual(ReviewDecision.objects.count(), 1)

    def test_reject_creates_a_separate_persisted_review(self):
        response = self._post_review(ReviewDecision.Decision.REJECT, "Not suitable for this test branch.")

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            ReviewDecision.objects.get().decision,
            ReviewDecision.Decision.REJECT,
        )

    def test_machine_action_and_human_reject_remain_separate(self):
        machine_state = (
            self.recommendation.action,
            self.option.q0,
            self.option.q1,
            self.recommendation.reasons,
        )

        self._post_review(ReviewDecision.Decision.REJECT, "Not suitable for this test branch.")
        response = self.client.get(self.url)
        self.recommendation.refresh_from_db()
        self.option.refresh_from_db()

        self.assertContains(response, "KEEP")
        self.assertContains(response, "REJECT")
        self.assertContains(response, "REJECTED")
        self.assertEqual(
            (self.recommendation.action, self.option.q0, self.option.q1, self.recommendation.reasons),
            machine_state,
        )

    def test_modify_persists_comment_without_changing_recommendation_q1(self):
        original_q1 = self.option.q1
        comment = "Reduce q1 from 5 to 3 because branch display area is limited."

        response = self._post_review(ReviewDecision.Decision.MODIFY, comment)

        self.assertEqual(response.status_code, 302)
        review = ReviewDecision.objects.get()
        self.option.refresh_from_db()
        self.assertEqual(review.decision, ReviewDecision.Decision.MODIFY)
        self.assertEqual(review.comment, comment)
        self.assertEqual(self.option.q1, original_q1)

    def test_empty_or_invalid_form_input_creates_no_review(self):
        for payload in (
            {"decision": ReviewDecision.Decision.MODIFY, "comment": "   "},
            {"decision": "APPROVE_NOW", "comment": "Unsupported decision."},
        ):
            with self.subTest(payload=payload):
                response = self.client.post(self.url, payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(ReviewDecision.objects.count(), 0)
                self.assertContains(response, "PENDING")

    def test_later_run_does_not_change_review_target_and_refresh_does_not_duplicate(self):
        self._post_review(ReviewDecision.Decision.ACCEPT, "Exact run review.")
        later_run = OptimizationResult.objects.create(
            planning_month=date(2026, 5, 1), scenario_version="review-page-synthetic-v2",
            run_version="review-page-run-v2",
        )

        first = self.client.get(self.url)
        second = self.client.get(self.url)
        review = ReviewDecision.objects.get()
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(review.reviewed_run_id, self.run.pk)
        self.assertNotEqual(review.reviewed_run_id, later_run.pk)
        self.assertEqual(ReviewDecision.objects.count(), 1)

    def test_current_status_is_isolated_to_the_exact_run_version(self):
        self._post_review(ReviewDecision.Decision.ACCEPT, "Exact run review.")
        later_run = OptimizationResult.objects.create(
            planning_month=date(2026, 5, 1), scenario_version="review-page-synthetic-v2",
            run_version="review-page-run-v2",
        )

        first_status, first_latest = ReviewDecision.current_status_for_run(self.run)
        later_status, later_latest = ReviewDecision.current_status_for_run(later_run)

        self.assertEqual(first_status, ReviewDecision.CurrentStatus.ACCEPTED)
        self.assertEqual(first_latest.reviewed_run_id, self.run.pk)
        self.assertEqual(later_status, ReviewDecision.CurrentStatus.PENDING)
        self.assertIsNone(later_latest)

    def test_blocked_run_never_exposes_controls_or_creates_review(self):
        blocked_run = OptimizationResult.objects.create(
            planning_month=date(2026, 4, 1), scenario_version="blocked-review-v1",
            run_version="blocked-review-run-v1", solver_status=OptimizationResult.SolverStatus.INFEASIBLE,
            solver_metadata={"pipeline_context": {
                "mode": "RESEARCH_SYNTHETIC", "planning_month": "2026-04-01", "branch_code": "BIL01",
            }},
        )
        blocked_url = reverse("reviews:review_run", args=[blocked_run.pk])

        get_response = self.client.get(blocked_url)
        post_response = self.client.post(blocked_url, {
            "decision": ReviewDecision.Decision.ACCEPT, "comment": "Should not save.",
        })
        self.assertContains(get_response, "Review unavailable")
        self.assertNotContains(get_response, "Save Review")
        self.assertContains(post_response, "Review unavailable")
        self.assertEqual(ReviewDecision.objects.count(), 0)

    def test_review_view_contains_no_pipeline_execution_calls(self):
        from reviews.views import review_run
        import inspect

        source = inspect.getsource(review_run)
        for forbidden_operation in (
            "run_pipeline", "run_and_persist_optimization", "check_and_persist_feasibility",
            "create_recommendations_for_result", "train_model", "generate_candidates",
        ):
            self.assertNotIn(forbidden_operation, source)
