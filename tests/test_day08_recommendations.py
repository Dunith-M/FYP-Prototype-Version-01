"""Gate-protected, transactional Day-08 Recommendation persistence tests."""

from datetime import date

import pytest

import planning.services.recommendation_service as recommendation_service
from catalog.models import Branch, Product
from optimization.checker import check_and_persist_feasibility
from planning.models import DecisionOption, OptimizationResult, Recommendation
from planning.services.explanations import ExplanationReason, ExplanationResult, SYNTHETIC_STATUS_TEXT
from planning.services.publication_gate import PublicationGateBlocked
from planning.services.recommendation_service import (
    RESEARCH_MODE,
    RecommendationPersistenceError,
    create_recommendations_for_result,
)


PLANNING_MONTH = date(2026, 4, 1)
SCENARIO_VERSION = "recommendation_synthetic_v1"


def make_option(branch, product, *, l0=True, q0=2, l1=True, q1=3):
    return DecisionOption.objects.create(
        branch=branch,
        product=product,
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        l0=l0,
        q0=q0,
        l1=l1,
        q1=q1,
        option_type=DecisionOption.OptionType.LISTED_TARGET if l1 and l0 else DecisionOption.OptionType.DELIST,
        evidence_score=0.7,
        evidence_source=DecisionOption.EvidenceSource.HISTORICAL_LAG_1_NORMALIZED,
        parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
    )


def checked_result(*, solver_status="OPTIMAL", option_count=1, run_version="recommendation-run-v1"):
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Recommendation branch")
    options = []
    for index in range(option_count):
        product = Product.objects.create(
            canonical_product_id=f"CP_RECOMMEND_{index}",
            product_code=f"RECOMMEND{index}",
            product_desc=f"Recommendation product {index}",
        )
        options.append(make_option(branch, product))
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version=run_version,
        solver_status=solver_status,
    )
    result.selected_options.add(*options)
    check_and_persist_feasibility(result.pk)
    result.refresh_from_db()
    return result, options


@pytest.mark.django_db
def test_optimal_checked_plan_persists_exact_selected_option_and_traceability():
    result, (selected,) = checked_result()

    persisted = create_recommendations_for_result(result.pk)
    recommendation = Recommendation.objects.select_related("optimization_result", "selected_option", "branch", "product").get()

    assert persisted.created_count == 1
    assert persisted.existing_count == 0
    assert recommendation.optimization_result_id == result.pk
    assert recommendation.selected_option_id == selected.pk
    assert recommendation.branch_id == selected.branch_id
    assert recommendation.product_id == selected.product_id
    assert recommendation.action == Recommendation.Action.INCREASE
    assert recommendation.status == Recommendation.Status.READY
    assert recommendation.reasons[0] == {"code": "TARGET_INCREASE", "text": "Selected target increases from 2 to 3 units."}
    assert recommendation.reasons[-1] == {
        "code": RESEARCH_MODE,
        "text": SYNTHETIC_STATUS_TEXT,
        "parameter_source": DecisionOption.ParameterSource.SYNTHETIC_DEMO,
    }
    assert recommendation.selected_option.parameter_source == DecisionOption.ParameterSource.SYNTHETIC_DEMO
    assert recommendation.optimization_result.feasibility_status == OptimizationResult.FeasibilityStatus.PASS


@pytest.mark.django_db
def test_feasible_checked_plan_is_allowed():
    result, _options = checked_result(solver_status=OptimizationResult.SolverStatus.FEASIBLE)
    persisted = create_recommendations_for_result(result.pk)
    assert persisted.created_count == 1
    assert Recommendation.objects.get().optimization_result_id == result.pk


@pytest.mark.django_db
@pytest.mark.parametrize("solver_status", ["INFEASIBLE", "UNKNOWN", "ERROR"])
def test_unusable_solver_states_create_zero_recommendations(solver_status):
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version=f"blocked-{solver_status}",
        solver_status=solver_status,
    )
    with pytest.raises(PublicationGateBlocked, match="PUB-GATE-001"):
        create_recommendations_for_result(result.pk)
    assert Recommendation.objects.count() == 0


@pytest.mark.django_db
def test_checker_failure_is_blocked_before_recommendation_persistence():
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Recommendation branch")
    first = make_option(branch, Product.objects.create(canonical_product_id="CP_FAIL_1", product_code="FAIL1", product_desc="Fail one"))
    make_option(branch, Product.objects.create(canonical_product_id="CP_FAIL_2", product_code="FAIL2", product_desc="Fail two"))
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version="checker-fail",
        solver_status=OptimizationResult.SolverStatus.OPTIMAL,
    )
    result.selected_options.add(first)
    check_and_persist_feasibility(result.pk)
    with pytest.raises(PublicationGateBlocked, match="PUB-GATE-003"):
        create_recommendations_for_result(result.pk)
    assert Recommendation.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("feasibility_status", "solver_metadata", "expected_code"),
    [
        (OptimizationResult.FeasibilityStatus.NOT_RUN, {}, "PUB-GATE-002"),
        (OptimizationResult.FeasibilityStatus.PASS, {}, "PUB-GATE-002"),
    ],
)
def test_checker_not_run_or_missing_evidence_is_blocked_before_recommendation_persistence(
    feasibility_status, solver_metadata, expected_code
):
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Recommendation branch")
    product = Product.objects.create(canonical_product_id="CP_NO_CHECK", product_code="NOCHECK", product_desc="No checker")
    selected = make_option(branch, product)
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version=f"checker-{feasibility_status}",
        solver_status=OptimizationResult.SolverStatus.OPTIMAL,
        feasibility_status=feasibility_status,
        solver_metadata=solver_metadata,
    )
    result.selected_options.add(selected)
    with pytest.raises(PublicationGateBlocked, match=expected_code):
        create_recommendations_for_result(result.pk)
    assert Recommendation.objects.count() == 0


@pytest.mark.django_db
def test_stale_checker_pass_is_blocked_before_recommendation_persistence():
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Recommendation branch")
    product = Product.objects.create(canonical_product_id="CP_STALE", product_code="STALE", product_desc="Stale product")
    first = make_option(branch, product, q1=2)
    second = make_option(branch, product, q1=3)
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version="stale-checker",
        solver_status=OptimizationResult.SolverStatus.OPTIMAL,
    )
    result.selected_options.add(first)
    check_and_persist_feasibility(result.pk)
    result.selected_options.set((second,))
    with pytest.raises(PublicationGateBlocked, match="PUB-GATE-004"):
        create_recommendations_for_result(result.pk)
    assert Recommendation.objects.count() == 0


@pytest.mark.django_db
def test_withheld_is_persisted_as_null_action_not_a_seventh_action(monkeypatch):
    result, (selected,) = checked_result()
    withheld = ExplanationResult(
        decision_option_id=selected.pk,
        action=None,
        action_status=Recommendation.Status.WITHHELD,
        reasons=(ExplanationReason("MISSING_REFERENCE_STATE", "No action was generated because the required reference target q0 is unavailable."),),
        status_text=SYNTHETIC_STATUS_TEXT,
    )
    monkeypatch.setattr(recommendation_service, "generate_explanations_for_optimization_result", lambda _id: (withheld,))

    create_recommendations_for_result(result.pk)
    recommendation = Recommendation.objects.get()
    assert recommendation.action is None
    assert recommendation.status == Recommendation.Status.WITHHELD
    assert recommendation.reasons[0]["code"] == "MISSING_REFERENCE_STATE"
    assert "WITHHELD" not in {value for value, _label in Recommendation.Action.choices}


@pytest.mark.django_db
def test_repeat_is_idempotent_and_completed_output_is_not_overwritten():
    result, _options = checked_result()
    first = create_recommendations_for_result(result.pk)
    second = create_recommendations_for_result(result.pk)
    assert first.created_count == 1
    assert second.created_count == 0
    assert second.existing_count == 1
    assert Recommendation.objects.count() == 1


@pytest.mark.django_db
def test_atomic_batch_rolls_back_when_second_persist_write_fails(monkeypatch):
    result, _options = checked_result(option_count=2)
    original = recommendation_service._save_new_recommendation
    calls = 0

    def fail_second(recommendation):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled persistence failure")
        return original(recommendation)

    monkeypatch.setattr(recommendation_service, "_save_new_recommendation", fail_second)
    with pytest.raises(RuntimeError, match="controlled persistence failure"):
        create_recommendations_for_result(result.pk)
    assert Recommendation.objects.count() == 0


@pytest.mark.django_db
def test_wrong_explanation_coverage_blocks_batch_without_writing_rows(monkeypatch):
    result, _options = checked_result(option_count=2)
    monkeypatch.setattr(recommendation_service, "generate_explanations_for_optimization_result", lambda _id: ())
    with pytest.raises(RecommendationPersistenceError, match="does not cover"):
        create_recommendations_for_result(result.pk)
    assert Recommendation.objects.count() == 0
