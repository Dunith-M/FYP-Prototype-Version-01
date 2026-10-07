"""Day-08 CP-SAT execution, status mapping, and result-persistence tests."""

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from catalog.models import Branch, Product
from optimization.runner import (
    SolverExecution,
    SolverRunError,
    execute_cp_sat,
    map_cp_sat_status,
    persist_solver_execution,
    run_and_persist_optimization,
)
from optimization.scenario_loader import DecisionUnit, OptimizationScenarioInput, SolverOption
from optimization.solver import add_exactly_one_option_constraints, add_proxy_utility_objective, build_option_variables
from optimization.validation import OptionValidationResult
from planning.models import DecisionOption, OptimizationResult, Recommendation


PLANNING_MONTH = date(2026, 4, 1)
SCENARIO_VERSION = "day07_synthetic_options_v1"
_CP_SAT_CONTEXT_KEEPALIVE: list[object] = []


def _scenario(*option_id_groups):
    units = []
    for product_id, option_ids in enumerate(option_id_groups, start=1):
        options = tuple(
            SolverOption(
                option_id=option_id,
                branch_id=1,
                branch_code="BIL01",
                product_id=product_id,
                canonical_product_id=f"CP_RUN_{product_id}",
                l0=True,
                q0=1,
                l1=True,
                q1=option_index,
                utility=0.1 * option_index,
                parameter_source="SYNTHETIC_DEMO",
            )
            for option_index, option_id in enumerate(option_ids, start=1)
        )
        units.append(DecisionUnit(1, "BIL01", product_id, f"CP_RUN_{product_id}", options))
    return OptimizationScenarioInput(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        parameter_source="SYNTHETIC_DEMO",
        units=tuple(units),
        validation_result=OptionValidationResult(
            planning_month=PLANNING_MONTH,
            scenario_version=SCENARIO_VERSION,
            decision_unit_count=len(units),
            option_count=sum(len(unit.options) for unit in units),
            errors=(),
        ),
    )


def _context_with_objective(scenario):
    context = build_option_variables(scenario)
    _CP_SAT_CONTEXT_KEEPALIVE.append(context)
    add_exactly_one_option_constraints(context)
    return context, add_proxy_utility_objective(context)


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        (cp_model.OPTIMAL, "OPTIMAL"),
        (cp_model.FEASIBLE, "FEASIBLE"),
        (cp_model.INFEASIBLE, "INFEASIBLE"),
        (cp_model.UNKNOWN, "UNKNOWN"),
    ],
)
def test_map_cp_sat_status_uses_the_stable_day08_contract(raw_status, expected):
    assert map_cp_sat_status(raw_status) == expected


def test_execute_cp_sat_extracts_exactly_one_option_per_unit_for_an_optimal_solution():
    context, objective = _context_with_objective(_scenario((101, 102), (103, 104)))

    execution = execute_cp_sat(context, objective)

    assert execution.solver_status == "OPTIMAL"
    assert execution.selected_option_ids == (102, 104)
    assert execution.scaled_objective_value is not None
    assert execution.original_objective_value is not None


def test_execute_cp_sat_keeps_infeasible_distinct_and_never_extracts_a_plan():
    context, objective = _context_with_objective(_scenario((201, 202)))
    context.model.Add(context.option_vars[201] == 0)
    context.model.Add(context.option_vars[202] == 0)

    execution = execute_cp_sat(context, objective)

    assert execution.solver_status == "INFEASIBLE"
    assert execution.selected_option_ids == ()
    assert execution.original_objective_value is None


def test_execute_cp_sat_maps_a_pipeline_exception_to_error_without_a_plan():
    context, objective = _context_with_objective(_scenario((301, 302)))

    class ExplodingSolver:
        parameters = SimpleNamespace(max_time_in_seconds=None, num_search_workers=None, random_seed=None)

        def Solve(self, _model):
            raise RuntimeError("controlled solver failure")

    execution = execute_cp_sat(context, objective, solver_factory=ExplodingSolver)

    assert execution.solver_status == "ERROR"
    assert execution.selected_option_ids == ()
    assert execution.original_objective_value is None
    assert "controlled solver failure" in execution.error_message


def _decision_option(branch, product, *, l0, q0, l1, q1, utility, option_type):
    return DecisionOption.objects.create(
        branch=branch,
        product=product,
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        l0=l0,
        q0=q0,
        l1=l1,
        q1=q1,
        option_type=option_type,
        evidence_score=utility,
        evidence_source=DecisionOption.EvidenceSource.HISTORICAL_LAG_1_NORMALIZED,
        parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
    )


@pytest.mark.django_db
def test_runner_persists_usable_status_selected_options_and_pending_feasibility_without_recommendations():
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Branch")
    existing_product = Product.objects.create(canonical_product_id="CP_EXIST", product_code="EXIST", product_desc="Existing")
    candidate_product = Product.objects.create(canonical_product_id="CP_CAND", product_code="CAND", product_desc="Candidate")
    _decision_option(
        branch,
        existing_product,
        l0=True,
        q0=1,
        l1=True,
        q1=1,
        utility=0.1,
        option_type=DecisionOption.OptionType.LISTED_TARGET,
    )
    _decision_option(
        branch,
        existing_product,
        l0=True,
        q0=1,
        l1=False,
        q1=0,
        utility=0.4,
        option_type=DecisionOption.OptionType.DELIST,
    )
    _decision_option(
        branch,
        candidate_product,
        l0=False,
        q0=None,
        l1=False,
        q1=0,
        utility=0.2,
        option_type=DecisionOption.OptionType.REMAIN_UNLISTED,
    )
    _decision_option(
        branch,
        candidate_product,
        l0=False,
        q0=None,
        l1=True,
        q1=1,
        utility=0.9,
        option_type=DecisionOption.OptionType.LIST_TRIAL,
    )

    outcome = run_and_persist_optimization(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version="test-run-v1",
    )
    result = OptimizationResult.objects.get(pk=outcome.optimization_result.pk)

    assert result.solver_status == OptimizationResult.SolverStatus.OPTIMAL
    assert result.selected_options.count() == 2
    assert set(result.selected_options.values_list("pk", flat=True)) == set(outcome.execution.selected_option_ids)
    assert result.selected_options.order_by().values("branch_id", "product_id").distinct().count() == 2
    assert result.objective_value == pytest.approx(1.3)
    assert result.feasibility_status == OptimizationResult.FeasibilityStatus.NOT_RUN
    assert result.solver_metadata["objective_name"] == "PROXY_UTILITY"
    assert result.solver_metadata["scaled_proxy_utility"] is not None
    states = list(
        result.selected_options.select_related("branch", "product").order_by("product__canonical_product_id")
    )
    assert {(state.l0, state.q0, state.l1, state.q1) for state in states} == {
        (True, 1, False, 0),
        (False, None, True, 1),
    }
    assert {state.parameter_source for state in states} == {DecisionOption.ParameterSource.SYNTHETIC_DEMO}
    assert Recommendation.objects.count() == 0

    rerun = run_and_persist_optimization(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version="test-run-v1",
    )
    assert rerun.optimization_result.pk == result.pk
    assert OptimizationResult.objects.count() == 1

    feasible = persist_solver_execution(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version="test-feasible-v1",
        execution=SolverExecution(
            "FEASIBLE",
            cp_model.FEASIBLE,
            outcome.execution.selected_option_ids,
            1300,
            Decimal("1.3"),
            0.01,
        ),
        objective=None,
        max_time_seconds=10.0,
        num_search_workers=1,
        random_seed=0,
    )
    assert feasible.solver_status == "FEASIBLE"
    assert set(feasible.selected_options.values_list("pk", flat=True)) == set(outcome.execution.selected_option_ids)


@pytest.mark.django_db
def test_persist_infeasible_execution_keeps_objective_and_selected_options_empty():
    execution = SolverExecution("INFEASIBLE", cp_model.INFEASIBLE, (), None, None, 0.01)

    result = persist_solver_execution(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version="test-infeasible-v1",
        execution=execution,
        objective=None,
        max_time_seconds=10.0,
        num_search_workers=1,
        random_seed=0,
    )

    assert result.solver_status == "INFEASIBLE"
    assert result.objective_value is None
    assert result.selected_options.count() == 0
    assert result.feasibility_status == "NOT_RUN"


@pytest.mark.django_db
@pytest.mark.parametrize("status", ["UNKNOWN", "ERROR"])
def test_unusable_statuses_persist_no_selected_options(status):
    result = persist_solver_execution(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version=f"test-{status.lower()}-v1",
        execution=SolverExecution(status, None, (), None, None, 0.01),
        objective=None,
        max_time_seconds=10.0,
        num_search_workers=1,
        random_seed=0,
    )

    assert result.solver_status == status
    assert result.selected_options.count() == 0
    assert result.objective_value is None


@pytest.mark.django_db
def test_persistence_rejects_selected_option_from_another_scenario():
    branch = Branch.objects.create(branch_code="BIL02", branch_name="Other branch")
    product = Product.objects.create(canonical_product_id="CP_OTHER", product_code="OTHER", product_desc="Other")
    other_option = DecisionOption.objects.create(
        branch=branch,
        product=product,
        planning_month=PLANNING_MONTH,
        scenario_version="another_synthetic_scenario",
        l0=False,
        q0=None,
        l1=False,
        q1=0,
        option_type=DecisionOption.OptionType.REMAIN_UNLISTED,
        evidence_score=0.1,
        evidence_source=DecisionOption.EvidenceSource.TRACK_B_CANDIDATE_SCORE,
        parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
    )

    with pytest.raises(SolverRunError, match="outside the optimization scenario"):
        persist_solver_execution(
            planning_month=PLANNING_MONTH,
            scenario_version=SCENARIO_VERSION,
            run_version="test-cross-scenario-v1",
            execution=SolverExecution("OPTIMAL", cp_model.OPTIMAL, (other_option.pk,), 1, Decimal("0.1"), 0.01),
            objective=None,
            max_time_seconds=10.0,
            num_search_workers=1,
            random_seed=0,
        )
    assert OptimizationResult.objects.count() == 0
