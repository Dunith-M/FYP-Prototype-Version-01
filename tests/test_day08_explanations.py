"""Known-answer tests for deterministic Day-08 explanation templates."""

import inspect
from datetime import date
from types import SimpleNamespace

import pytest

import planning.services.explanations as explanations
from analytics.models import CandidateResult
from catalog.models import Branch, Product
from optimization.checker import check_and_persist_feasibility
from planning.models import DecisionOption, OptimizationResult, Recommendation
from planning.services.action_engine import ActionDecision
from planning.services.explanations import (
    ConstraintContext,
    SYNTHETIC_STATUS_TEXT,
    generate_explanation,
    generate_explanations_for_optimization_result,
    ordinal,
)


def option(**overrides):
    values = {
        "pk": 101,
        "l0": True,
        "q0": 2,
        "l1": True,
        "q1": 3,
        "evidence_score": 0.625,
        "evidence_source": DecisionOption.EvidenceSource.HISTORICAL_LAG_1_NORMALIZED,
        "branch": SimpleNamespace(branch_code="BIL01"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def ready(action):
    return ActionDecision(action=action, status=Recommendation.Status.READY)


def context(*, listed=4, max_skus=150, introductions=4, max_introductions=5):
    return ConstraintContext(
        selected_listed_skus=listed,
        max_skus_per_branch=max_skus,
        selected_introductions=introductions,
        max_introductions_per_branch=max_introductions,
        checker_passed=True,
    )


def test_track_a_increase_has_exact_ordered_known_answer_text_without_forecast_claim():
    result = generate_explanation(
        selected_option=option(),
        action_result=ready(Recommendation.Action.INCREASE),
        constraint_context=context(),
    )

    assert result.reason_texts == (
        "Selected target increases from 2 to 3 units.",
        "Historical lag-1 normalized evidence score is 0.625.",
        "The selected plan uses 4 of the synthetic 150 listed-SKU capacity.",
        "The complete selected plan passed the independent feasibility checker.",
    )
    assert result.status_text == SYNTHETIC_STATUS_TEXT
    assert "forecast" not in " ".join(result.reason_texts).lower()


def test_track_b_rank_component_and_evidence_class_use_stored_values_in_order():
    candidate = SimpleNamespace(rank=2, peer_score=0.34, family_score=0.21, similar_branch_score=None, candidate_score=0.7, evidence_class="ADEQUATE")
    result = generate_explanation(
        selected_option=option(
            pk=102,
            l0=False,
            q0=None,
            l1=True,
            q1=1,
            evidence_source=DecisionOption.EvidenceSource.TRACK_B_CANDIDATE_SCORE,
        ),
        action_result=ready(Recommendation.Action.INTRODUCE),
        candidate_result=candidate,
        constraint_context=context(),
    )

    assert result.reason_texts == (
        "Selected option introduces the product with a target of 1 unit.",
        "Ranked 2nd among research-shortlisted candidates for BIL01.",
        "Peer-product research score is 0.34.",
        "Candidate evidence class is ADEQUATE.",
    )


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th"), (21, "21st")],
)
def test_ordinal_rank_formatting(rank, expected):
    assert ordinal(rank) == expected


def test_do_not_introduce_does_not_claim_candidate_is_bad_or_constraint_caused_it():
    candidate = SimpleNamespace(rank=2, peer_score=None, family_score=None, similar_branch_score=None, candidate_score=0.7, evidence_class="LIMITED")
    result = generate_explanation(
        selected_option=option(
            l0=False,
            q0=None,
            l1=False,
            q1=0,
            evidence_source=DecisionOption.EvidenceSource.TRACK_B_CANDIDATE_SCORE,
        ),
        action_result=ready(Recommendation.Action.DO_NOT_INTRODUCE),
        candidate_result=candidate,
        constraint_context=context(introductions=4, max_introductions=5),
    )
    text = " ".join(result.reason_texts).lower()
    assert result.reason_texts[0] == "Selected feasible option keeps the candidate unlisted in this scenario."
    assert "because" not in text
    assert "binding" not in text
    assert "profit" not in text


def test_withheld_reason_is_truthful_and_contains_no_positive_action_story():
    withheld = ActionDecision(
        action=None,
        status=Recommendation.Status.WITHHELD,
        reason_code="MISSING_REFERENCE_STATE",
    )
    result = generate_explanation(selected_option=option(q0=None), action_result=withheld)
    assert result.action is None
    assert result.action_status == Recommendation.Status.WITHHELD
    assert result.reason_texts == ("No action was generated because the required reference target q0 is unavailable.",)


def test_repeated_generation_is_deterministic_and_normal_results_have_two_to_four_reasons():
    kwargs = dict(
        selected_option=option(),
        action_result=ready(Recommendation.Action.INCREASE),
        constraint_context=context(),
    )
    first = generate_explanation(**kwargs)
    second = generate_explanation(**kwargs)
    assert first == second
    assert 2 <= len(first.reasons) <= 4


def test_null_budget_space_and_empty_category_produce_no_fake_constraint_text():
    result = generate_explanation(
        selected_option=option(),
        action_result=ready(Recommendation.Action.KEEP),
        constraint_context=context(),
    )
    text = " ".join(result.reason_texts).lower()
    assert "budget" not in text
    assert "space" not in text
    assert "category" not in text


def test_explanation_service_has_no_llm_solver_raw_data_or_financial_claim_dependency():
    source = inspect.getsource(explanations).lower()
    for forbidden in ("read_excel", "read_parquet", "ortools", "cpsolver", "profit", "revenue", "margin", "openai"):
        assert forbidden not in source


@pytest.mark.django_db
def test_orchestration_uses_persisted_candidate_evidence_without_creating_recommendations():
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Explanation branch")
    product = Product.objects.create(canonical_product_id="CP_EXPLAIN", product_code="EXPLAIN", product_desc="Explanation product")
    planning_month = date(2026, 4, 1)
    selected = DecisionOption.objects.create(
        branch=branch,
        product=product,
        planning_month=planning_month,
        scenario_version="explain_synthetic_v1",
        l0=False,
        q0=None,
        l1=False,
        q1=0,
        option_type=DecisionOption.OptionType.REMAIN_UNLISTED,
        evidence_score=0.8,
        evidence_source=DecisionOption.EvidenceSource.TRACK_B_CANDIDATE_SCORE,
        parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
    )
    CandidateResult.objects.create(
        branch=branch,
        product=product,
        planning_month=planning_month,
        peer_score=0.34,
        family_score=None,
        similar_branch_score=None,
        price_fit_score=None,
        candidate_score=0.8,
        rank=2,
        evidence_class=CandidateResult.EvidenceClass.LIMITED,
        run_version="day07_candidate_score_v1",
    )
    optimization_result = OptimizationResult.objects.create(
        planning_month=planning_month,
        scenario_version="explain_synthetic_v1",
        run_version="explain-run-v1",
        solver_status=OptimizationResult.SolverStatus.OPTIMAL,
    )
    optimization_result.selected_options.add(selected)
    check_and_persist_feasibility(optimization_result.pk)

    results = generate_explanations_for_optimization_result(optimization_result.pk)

    assert len(results) == 1
    assert results[0].reason_texts[:3] == (
        "Selected feasible option keeps the candidate unlisted in this scenario.",
        "Ranked 2nd among research-shortlisted candidates for BIL01.",
        "Peer-product research score is 0.34.",
    )
    assert Recommendation.objects.count() == 0
