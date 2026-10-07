"""Truth-table, WITHHELD, and gate-integration tests for Day-08 actions."""

import inspect
from datetime import date

import pytest

import planning.services.action_engine as action_engine
from catalog.models import Branch, Product
from optimization.checker import check_and_persist_feasibility
from planning.models import DecisionOption, OptimizationResult, Recommendation
from planning.services.action_engine import (
    ActionDecision,
    ActionDerivationError,
    WithheldReason,
    derive_action,
    derive_actions_for_optimization_result,
    load_action_delta,
)
from planning.services.publication_gate import PublicationGateBlocked


PLANNING_MONTH = date(2026, 4, 1)
SCENARIO_VERSION = "action_synthetic_v1"


def assert_ready(decision, expected):
    assert decision.action == expected
    assert decision.status == Recommendation.Status.READY
    assert decision.reason_code is None


def assert_withheld(decision, reason_code):
    assert decision.action is None
    assert decision.status == Recommendation.Status.WITHHELD
    assert decision.reason_code == reason_code


@pytest.mark.parametrize(
    ("l0", "l1", "q0", "q1", "delta", "expected"),
    [
        (True, True, 2, 3, 0, "INCREASE"),
        (True, True, 2, 2, 0, "KEEP"),
        (True, True, 3, 2, 0, "REDUCE"),
        (True, False, 2, 0, 0, "REMOVE"),
        (False, True, None, 1, 0, "INTRODUCE"),
        (False, False, None, 0, 0, "DO_NOT_INTRODUCE"),
    ],
)
def test_six_action_truth_table(l0, l1, q0, q1, delta, expected):
    assert_ready(derive_action(l0=l0, l1=l1, q0=q0, q1=q1, delta=delta), expected)


@pytest.mark.parametrize(
    ("q1", "expected"),
    [(7, "INCREASE"), (6, "KEEP"), (5, "KEEP"), (4, "KEEP"), (3, "REDUCE")],
)
def test_delta_keep_band_is_inclusive_and_configuration_driven(q1, expected):
    assert_ready(derive_action(l0=True, l1=True, q0=5, q1=q1, delta=1), expected)


@pytest.mark.parametrize(
    ("state", "reason_code"),
    [
        ({"l0": None, "l1": True, "q0": None, "q1": 1, "delta": 0}, WithheldReason.MISSING_LISTING_STATE),
        ({"l0": True, "l1": None, "q0": 2, "q1": 2, "delta": 0}, WithheldReason.MISSING_PROPOSED_STATE),
        ({"l0": True, "l1": True, "q0": 2, "q1": None, "delta": 0}, WithheldReason.MISSING_PROPOSED_TARGET),
        ({"l0": True, "l1": True, "q0": None, "q1": 2, "delta": 0}, WithheldReason.MISSING_REFERENCE_STATE),
        ({"l0": True, "l1": True, "q0": 2, "q1": 2, "delta": -1}, WithheldReason.INVALID_STATE),
        ({"l0": True, "l1": False, "q0": 2, "q1": 1, "delta": 0}, WithheldReason.INVALID_STATE),
    ],
)
def test_invalid_or_missing_state_is_withheld_instead_of_fabricating_action(state, reason_code):
    assert_withheld(derive_action(**state), reason_code)


def test_t_action_02_missing_l0_q0_or_evidence_returns_withheld():
    assert_withheld(
        derive_action(l0=None, l1=True, q0=None, q1=2, delta=0),
        WithheldReason.MISSING_LISTING_STATE,
    )
    assert_withheld(
        derive_action(l0=True, l1=True, q0=None, q1=3, delta=0),
        WithheldReason.MISSING_REFERENCE_STATE,
    )
    assert_withheld(
        derive_action(l0=True, l1=True, q0=2, q1=3, delta=0, evidence_available=False),
        WithheldReason.INSUFFICIENT_EVIDENCE,
    )


def test_unknown_eligibility_and_unassessed_candidate_are_withheld_not_do_not_introduce():
    state = dict(l0=False, l1=False, q0=None, q1=0, delta=0)
    assert_withheld(derive_action(**state, eligibility=None), WithheldReason.UNKNOWN_ELIGIBILITY)
    assert_withheld(derive_action(**state, assessed=False), WithheldReason.INSUFFICIENT_EVIDENCE)


def test_known_ineligible_candidate_remains_distinct_from_withheld_or_do_not_introduce():
    with pytest.raises(ActionDerivationError, match="Excluded/ineligible"):
        derive_action(l0=False, l1=False, q0=None, q1=0, delta=0, eligibility=False)


def test_legitimate_candidate_q0_not_applicable_does_not_falsely_withhold():
    assert_ready(
        derive_action(l0=False, l1=True, q0=None, q1=1, delta=0, eligibility=True, assessed=True),
        "INTRODUCE",
    )
    assert_ready(
        derive_action(l0=False, l1=False, q0=None, q1=0, delta=0, eligibility=True, assessed=True),
        "DO_NOT_INTRODUCE",
    )


def test_listing_transition_precedes_quantity_comparison():
    assert_ready(derive_action(l0=True, l1=False, q0=5, q1=0, delta=0), "REMOVE")
    assert_ready(derive_action(l0=False, l1=True, q0=None, q1=1, delta=0), "INTRODUCE")


def test_delta_is_loaded_from_visible_synthetic_configuration_not_hard_coded():
    assert load_action_delta() == 0


def test_action_and_status_invariants_and_six_action_catalogue():
    actions = {value for value, _label in Recommendation.Action.choices}
    assert len(actions) == 6
    assert "WITHHELD" not in actions
    assert Recommendation.Status.WITHHELD in {value for value, _label in Recommendation.Status.choices}
    with pytest.raises(ValueError):
        ActionDecision(action="INCREASE", status=Recommendation.Status.WITHHELD, reason_code="INVALID_STATE")
    with pytest.raises(ValueError):
        ActionDecision(action=None, status=Recommendation.Status.READY)


def test_action_engine_has_no_raw_data_or_solver_dependency_and_truth_table_uses_no_score():
    source = inspect.getsource(action_engine)
    for forbidden_term in ("read_excel", "read_parquet", "ortools", "optimization.solver"):
        assert forbidden_term not in source
    assert "evidence_score" not in inspect.getsource(derive_action)
    assert "utility" not in inspect.getsource(derive_action)


def _option(branch, product, *, l0=False, q0=None, l1=False, q1=0):
    return DecisionOption.objects.create(
        branch=branch,
        product=product,
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        l0=l0,
        q0=q0,
        l1=l1,
        q1=q1,
        option_type=(DecisionOption.OptionType.LISTED_TARGET if l1 and l0 else DecisionOption.OptionType.LIST_TRIAL if l1 else DecisionOption.OptionType.DELIST if l0 else DecisionOption.OptionType.REMAIN_UNLISTED),
        evidence_score=0.5,
        evidence_source=DecisionOption.EvidenceSource.HISTORICAL_LAG_1_NORMALIZED,
        parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
    )


def _checked_result(*, solver_status="OPTIMAL", run_version="action-run-v1", alternate=False):
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Action branch")
    product = Product.objects.create(canonical_product_id="CP_ACTION", product_code="ACTION", product_desc="Action product")
    selected = _option(branch, product)
    if alternate:
        _option(branch, product, l0=False, q0=None, l1=True, q1=1)
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version=run_version,
        solver_status=solver_status,
    )
    result.selected_options.add(selected)
    check_and_persist_feasibility(result.pk)
    return result, selected


@pytest.mark.django_db
@pytest.mark.parametrize("solver_status", ["OPTIMAL", "FEASIBLE"])
def test_orchestration_requires_gate_and_derives_actions_for_ready_plan(solver_status):
    result, selected = _checked_result(solver_status=solver_status, run_version=f"action-{solver_status}")

    actions = derive_actions_for_optimization_result(result.pk)

    assert [(action.decision_option_id, action.action, action.status) for action in actions] == [
        (selected.pk, "DO_NOT_INTRODUCE", Recommendation.Status.READY)
    ]
    assert Recommendation.objects.count() == 0


@pytest.mark.django_db
def test_orchestration_blocks_checker_failure_without_creating_recommendations():
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Action branch")
    first = Product.objects.create(canonical_product_id="CP_ACTION_1", product_code="ACTION1", product_desc="Action one")
    second = Product.objects.create(canonical_product_id="CP_ACTION_2", product_code="ACTION2", product_desc="Action two")
    selected = _option(branch, first)
    _option(branch, second)
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version="action-checker-fail",
        solver_status=OptimizationResult.SolverStatus.OPTIMAL,
    )
    result.selected_options.add(selected)
    check_and_persist_feasibility(result.pk)

    with pytest.raises(PublicationGateBlocked, match="PUB-GATE-003"):
        derive_actions_for_optimization_result(result.pk)
    assert Recommendation.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize("solver_status", ["INFEASIBLE", "UNKNOWN", "ERROR"])
def test_orchestration_blocks_unusable_solver_states(solver_status):
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version=f"action-{solver_status}",
        solver_status=solver_status,
    )

    with pytest.raises(PublicationGateBlocked, match="PUB-GATE-001"):
        derive_actions_for_optimization_result(result.pk)


@pytest.mark.django_db
def test_orchestration_blocks_stale_checker_pass():
    result, selected = _checked_result(alternate=True)
    alternate = DecisionOption.objects.exclude(pk=selected.pk).get(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
    )
    result.selected_options.set((alternate,))

    with pytest.raises(PublicationGateBlocked, match="PUB-GATE-004"):
        derive_actions_for_optimization_result(result.pk)
