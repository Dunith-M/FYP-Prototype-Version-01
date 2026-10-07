"""Day-08 loader, validation, and option-variable construction tests."""

from datetime import date
from decimal import Decimal
import inspect
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

import optimization.scenario_loader as scenario_loader
from catalog.models import Branch, Product
from optimization.scenario_loader import (
    DecisionUnit,
    OptimizationScenarioInput,
    ScenarioInputBlocked,
    SolverOption,
    load_scenario_for_optimization,
)
from optimization.solver import (
    CpSatBuildContext,
    SyntheticConstraintConfig,
    VariableBuildError,
    add_exactly_one_option_constraints,
    add_permitted_synthetic_hard_constraints,
    add_proxy_utility_objective,
    build_option_variables,
    determine_utility_scale,
    load_synthetic_constraint_config,
    scale_proxy_utility,
)
from optimization.validation import OptionValidationIssue, OptionValidationResult, validate_option_set
from planning.models import DecisionOption


PLANNING_MONTH = date(2026, 4, 1)
SCENARIO_VERSION = "day07_synthetic_options_v1"
_CP_SAT_CONTEXT_KEEPALIVE: list[object] = []


def _option(branch, product, **overrides):
    values = {
        "planning_month": PLANNING_MONTH,
        "scenario_version": SCENARIO_VERSION,
        "l0": True,
        "q0": 2,
        "l1": True,
        "q1": 2,
        "option_type": DecisionOption.OptionType.LISTED_TARGET,
        "evidence_score": 0.75,
        "parameter_source": DecisionOption.ParameterSource.SYNTHETIC_DEMO,
    }
    return DecisionOption.objects.create(branch=branch, product=product, **(values | overrides))


def _row(**overrides):
    values = {
        "pk": 88,
        "planning_month": PLANNING_MONTH,
        "scenario_version": SCENARIO_VERSION,
        "branch_id": 1,
        "product_id": 2,
        "branch": SimpleNamespace(branch_code="BIL01"),
        "product": SimpleNamespace(canonical_product_id="CP_ROW"),
        "l0": True,
        "q0": 2,
        "l1": True,
        "q1": 2,
        "evidence_score": 0.75,
        "parameter_source": "SYNTHETIC_DEMO",
    }
    return SimpleNamespace(**(values | overrides))


def _validated_scenario(*option_id_groups, validation_errors=()):
    units = []
    for unit_index, option_ids in enumerate(option_id_groups, start=1):
        options = tuple(
            SolverOption(
                option_id=option_id,
                branch_id=unit_index,
                branch_code=f"BIL{unit_index:02d}",
                product_id=unit_index,
                canonical_product_id=f"CP_{unit_index}",
                l0=True,
                q0=2,
                l1=True,
                q1=2,
                utility=0.75,
                parameter_source="SYNTHETIC_DEMO",
            )
            for option_id in option_ids
        )
        units.append(
            DecisionUnit(
                branch_id=unit_index,
                branch_code=f"BIL{unit_index:02d}",
                product_id=unit_index,
                canonical_product_id=f"CP_{unit_index}",
                options=options,
            )
        )
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
            errors=tuple(validation_errors),
        ),
    )


def _build_variables_for_test(scenario):
    """Keep native CP-SAT models alive through this pytest process.

    OR-Tools 9.15 on this Windows/Python environment can corrupt the heap when
    a model is reclaimed before another model is constructed. Production keeps
    the build context for the rest of the solve workflow; this mirrors that
    lifecycle without changing solver behavior.
    """
    context = build_option_variables(scenario)
    _CP_SAT_CONTEXT_KEEPALIVE.append(context)
    return context


def _constraint_config(*, max_skus_per_branch=150, max_introductions_per_branch=5):
    return SyntheticConstraintConfig(
        mode="RESEARCH_SYNTHETIC",
        parameter_source="SYNTHETIC_DEMO",
        max_skus_per_branch=max_skus_per_branch,
        max_introductions_per_branch=max_introductions_per_branch,
        category_minimums={},
        budget_limit=None,
        space_limit=None,
    )


def _stateful_validated_scenario(*unit_option_states, branch_ids=None):
    """Build units from (option_id, L0, L1, q1) test-state tuples."""
    units = []
    for unit_index, states in enumerate(unit_option_states, start=1):
        branch_id = branch_ids[unit_index - 1] if branch_ids is not None else 1
        options = tuple(
            SolverOption(
                option_id=option_id,
                branch_id=branch_id,
                branch_code=f"BIL{branch_id:02d}",
                product_id=unit_index,
                canonical_product_id=f"CP_STATE_{unit_index}",
                l0=l0,
                q0=2 if l0 else None,
                l1=l1,
                q1=q1,
                utility=0.75,
                parameter_source="SYNTHETIC_DEMO",
            )
            for option_id, l0, l1, q1 in states
        )
        units.append(
            DecisionUnit(
                branch_id=branch_id,
                branch_code=f"BIL{branch_id:02d}",
                product_id=unit_index,
                canonical_product_id=f"CP_STATE_{unit_index}",
                options=options,
            )
        )
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


def _context_with_step08_constraints(scenario, config):
    context = _build_variables_for_test(scenario)
    add_exactly_one_option_constraints(context)
    return context, add_permitted_synthetic_hard_constraints(context, config)


def _objective_scenario(*unit_option_states):
    """Build units from (option_id, utility, L0, L1, q1) test-state tuples."""
    units = []
    for unit_index, states in enumerate(unit_option_states, start=1):
        options = tuple(
            SolverOption(
                option_id=option_id,
                branch_id=1,
                branch_code="BIL01",
                product_id=unit_index,
                canonical_product_id=f"CP_UTILITY_{unit_index}",
                l0=l0,
                q0=2 if l0 else None,
                l1=l1,
                q1=q1,
                utility=utility,
                parameter_source="SYNTHETIC_DEMO",
            )
            for option_id, utility, l0, l1, q1 in states
        )
        units.append(DecisionUnit(1, "BIL01", unit_index, f"CP_UTILITY_{unit_index}", options))
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


@pytest.mark.django_db
def test_loader_groups_valid_synthetic_options_and_preserves_identity():
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Branch")
    product = Product.objects.create(canonical_product_id="CP_LOAD", product_code="LOAD", product_desc="Loader product")
    selected = _option(branch, product)
    _option(branch, product, l1=False, q1=0, option_type=DecisionOption.OptionType.DELIST)

    validation = validate_option_set(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        option_rows=DecisionOption.objects.filter(planning_month=PLANNING_MONTH, scenario_version=SCENARIO_VERSION).select_related("branch", "product"),
    )
    scenario = load_scenario_for_optimization(planning_month=PLANNING_MONTH, scenario_version=SCENARIO_VERSION)

    assert validation.status == "PASS"
    assert scenario.parameter_source == "SYNTHETIC_DEMO"
    assert len(scenario.units) == 1
    assert scenario.decision_option_count == 2
    loaded = scenario.units[0]
    assert (loaded.branch_code, loaded.canonical_product_id) == ("BIL01", "CP_LOAD")
    assert loaded.options[1].option_id == selected.id
    assert {(option.l0, option.q0, option.l1, option.q1, option.utility) for option in loaded.options} == {
        (True, 2, True, 2, 0.75),
        (True, 2, False, 0, 0.75),
    }


@pytest.mark.django_db
def test_loader_rejects_a_scope_without_options():
    validation = validate_option_set(planning_month=PLANNING_MONTH, scenario_version="missing_scenario", option_rows=())

    assert validation.status == "BLOCKED"
    assert validation.error_counts == {"OPT-VAL-001": 1}
    with pytest.raises(ScenarioInputBlocked, match="OPT-VAL-001"):
        load_scenario_for_optimization(planning_month=PLANNING_MONTH, scenario_version="missing_scenario")


@pytest.mark.django_db
def test_loader_rejects_unexpected_parameter_source():
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Branch")
    product = Product.objects.create(canonical_product_id="CP_SOURCE", product_code="SOURCE", product_desc="Source product")
    _option(branch, product, parameter_source="UNEXPECTED")

    validation = validate_option_set(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        option_rows=DecisionOption.objects.select_related("branch", "product"),
    )

    assert validation.status == "BLOCKED"
    assert validation.error_counts == {"OPT-VAL-007": 1}
    with pytest.raises(ScenarioInputBlocked, match="OPT-VAL-007"):
        load_scenario_for_optimization(planning_month=PLANNING_MONTH, scenario_version=SCENARIO_VERSION)


@pytest.mark.django_db
def test_loader_rejects_missing_proxy_utility():
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Branch")
    product = Product.objects.create(canonical_product_id="CP_UTILITY", product_code="UTILITY", product_desc="Utility product")
    _option(branch, product, evidence_score=None)

    validation = validate_option_set(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        option_rows=DecisionOption.objects.select_related("branch", "product"),
    )

    assert validation.status == "BLOCKED"
    assert validation.error_counts == {"OPT-VAL-008": 1}
    with pytest.raises(ScenarioInputBlocked, match="OPT-VAL-008"):
        load_scenario_for_optimization(planning_month=PLANNING_MONTH, scenario_version=SCENARIO_VERSION)


@pytest.mark.django_db
def test_validation_blocks_duplicate_option_state_from_input_rows():
    row = _row()

    validation = validate_option_set(planning_month=PLANNING_MONTH, scenario_version=SCENARIO_VERSION, option_rows=(row, row))

    assert validation.status == "BLOCKED"
    assert validation.error_counts["OPT-VAL-010"] == 2


@pytest.mark.parametrize(
    ("overrides", "error_code"),
    [
        ({"branch": None, "branch_id": None}, "OPT-VAL-002"),
        ({"product": None, "product_id": None}, "OPT-VAL-003"),
        ({"l1": None}, "OPT-VAL-004"),
        ({"q0": None}, "OPT-VAL-004"),
        ({"q1": None}, "OPT-VAL-005"),
        ({"parameter_source": None}, "OPT-VAL-006"),
        ({"evidence_score": float("nan")}, "OPT-VAL-009"),
        ({"scenario_version": "another_scenario"}, "OPT-VAL-012"),
    ],
)
def test_validation_reports_malformed_row_without_repair(overrides, error_code):
    validation = validate_option_set(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        option_rows=(_row(**overrides),),
    )

    assert validation.status == "BLOCKED"
    assert error_code in validation.error_counts


@pytest.mark.django_db
def test_validation_blocks_inconsistent_reference_state_within_one_pair():
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Branch")
    product = Product.objects.create(canonical_product_id="CP_REFERENCE", product_code="REFERENCE", product_desc="Reference product")
    _option(branch, product, l1=False, q1=0, option_type=DecisionOption.OptionType.DELIST)
    _option(branch, product, l0=False, q0=None, l1=True, q1=1, option_type=DecisionOption.OptionType.LIST_TRIAL)

    validation = validate_option_set(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        option_rows=DecisionOption.objects.select_related("branch", "product"),
    )

    assert validation.status == "BLOCKED"
    assert validation.error_counts == {"OPT-VAL-011": 1}


def test_loader_module_has_no_upstream_data_reader_dependency():
    source = inspect.getsource(scenario_loader)
    for forbidden_term in ("read_excel", "read_parquet", "openpyxl", "data/raw/"):
        assert forbidden_term not in source


def test_variable_builder_creates_one_boolean_per_validated_option_with_id_mapping():
    scenario = _validated_scenario((101, 102, 103), (104, 105))

    context = _build_variables_for_test(scenario)

    assert len(context.option_vars) == scenario.decision_option_count == 5
    assert len(context.model.Proto().variables) == scenario.decision_option_count
    assert set(context.option_vars) == {101, 102, 103, 104, 105}
    assert str(context.option_vars[101]) == "option_101"
    assert all(list(variable.Proto().domain) == [0, 1] for variable in context.option_vars.values())


def test_variable_builder_preserves_multiple_decision_units():
    scenario = _validated_scenario((201, 202, 203, 204), (205, 206), (207, 208, 209))

    context = _build_variables_for_test(scenario)

    assert len(context.option_vars) == 9
    assert [len(context.variables_for_unit(unit)) for unit in scenario.units] == [4, 2, 3]


def test_variable_builder_rejects_duplicate_option_ids_without_overwriting_mapping():
    scenario = _validated_scenario((301, 302), (301,))

    with pytest.raises(VariableBuildError, match="Duplicate DecisionOption id 301"):
        build_option_variables(scenario)


def test_variable_builder_rejects_a_blocked_validation_result():
    scenario = _validated_scenario(
        (401,),
        validation_errors=(OptionValidationIssue("OPT-VAL-008", "MISSING_UTILITY", option_ids=(401,)),),
    )

    with pytest.raises(VariableBuildError, match="validation PASS"):
        build_option_variables(scenario)


def test_variable_builder_rejects_empty_validated_input():
    scenario = _validated_scenario()

    with pytest.raises(VariableBuildError, match="at least one validated DecisionOption"):
        build_option_variables(scenario)


def test_exactly_one_constraint_selects_one_option_for_a_four_option_unit():
    scenario = _validated_scenario((501, 502, 503, 504))
    context = _build_variables_for_test(scenario)

    constraint_count = add_exactly_one_option_constraints(context)
    solver = cp_model.CpSolver()
    status = solver.Solve(context.model)

    assert constraint_count == 1
    assert len(context.model.Proto().constraints) == 1
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert sum(solver.Value(context.option_vars[option_id]) for option_id in (501, 502, 503, 504)) == 1


def test_exactly_one_constraint_blocks_double_selection():
    scenario = _validated_scenario((601, 602))
    context = _build_variables_for_test(scenario)

    add_exactly_one_option_constraints(context)
    context.model.Add(context.option_vars[601] == 1)
    context.model.Add(context.option_vars[602] == 1)

    assert cp_model.CpSolver().Solve(context.model) == cp_model.INFEASIBLE


def test_exactly_one_constraint_blocks_zero_selection():
    scenario = _validated_scenario((701, 702))
    context = _build_variables_for_test(scenario)

    add_exactly_one_option_constraints(context)
    context.model.Add(context.option_vars[701] == 0)
    context.model.Add(context.option_vars[702] == 0)

    assert cp_model.CpSolver().Solve(context.model) == cp_model.INFEASIBLE


def test_exactly_one_constraints_are_independent_for_multiple_units():
    scenario = _validated_scenario((801, 802, 803, 804), (805, 806), (807, 808, 809))
    context = _build_variables_for_test(scenario)

    constraint_count = add_exactly_one_option_constraints(context)
    solver = cp_model.CpSolver()
    status = solver.Solve(context.model)

    assert constraint_count == len(scenario.units) == 3
    assert len(context.model.Proto().constraints) == 3
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert [sum(solver.Value(variable) for variable in context.variables_for_unit(unit)) for unit in scenario.units] == [1, 1, 1]


def test_exactly_one_constraint_forces_the_only_option_in_a_one_option_unit():
    scenario = _validated_scenario((901,))
    context = _build_variables_for_test(scenario)

    assert add_exactly_one_option_constraints(context) == 1
    solver = cp_model.CpSolver()
    assert solver.Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(context.option_vars[901]) == 1


def test_exactly_one_constraint_rejects_an_empty_unit():
    scenario = _validated_scenario(())
    context = CpSatBuildContext(model=cp_model.CpModel(), scenario=scenario, option_vars={})
    _CP_SAT_CONTEXT_KEEPALIVE.append(context)

    with pytest.raises(VariableBuildError, match="decision unit contains zero options"):
        add_exactly_one_option_constraints(context)


def test_exactly_one_constraint_rejects_a_blocked_validation_result():
    scenario = _validated_scenario(
        (1001,),
        validation_errors=(OptionValidationIssue("OPT-VAL-008", "MISSING_UTILITY", option_ids=(1001,)),),
    )
    model = cp_model.CpModel()
    context = CpSatBuildContext(model=model, scenario=scenario, option_vars={1001: model.NewBoolVar("option_1001")})
    _CP_SAT_CONTEXT_KEEPALIVE.append(context)

    with pytest.raises(VariableBuildError, match="validation PASS"):
        add_exactly_one_option_constraints(context)


def test_synthetic_sku_cap_allows_the_configured_boundary():
    scenario = _stateful_validated_scenario(
        ((1101, True, True, 1),),
        ((1102, True, True, 2),),
        ((1103, True, True, 5),),
    )
    context, counts = _context_with_step08_constraints(scenario, _constraint_config(max_skus_per_branch=3))

    assert cp_model.CpSolver().Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert counts.sku_cap_constraints == 1
    assert counts.introduction_cap_constraints == 0


def test_synthetic_sku_cap_blocks_more_than_the_configured_limit():
    scenario = _stateful_validated_scenario(
        ((1201, True, True, 1),),
        ((1202, True, True, 1),),
        ((1203, True, True, 1),),
        ((1204, True, True, 1),),
    )
    context, _ = _context_with_step08_constraints(scenario, _constraint_config(max_skus_per_branch=3))

    assert cp_model.CpSolver().Solve(context.model) == cp_model.INFEASIBLE


def test_synthetic_sku_cap_is_applied_per_branch_not_globally():
    scenario = _stateful_validated_scenario(
        ((1251, True, True, 1),),
        ((1252, True, True, 1),),
        branch_ids=(1, 2),
    )
    context, counts = _context_with_step08_constraints(scenario, _constraint_config(max_skus_per_branch=1))

    assert counts.sku_cap_constraints == 2
    assert cp_model.CpSolver().Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_synthetic_introduction_cap_allows_the_configured_boundary():
    scenario = _stateful_validated_scenario(
        ((1301, False, True, 1),),
        ((1302, False, True, 5),),
    )
    context, counts = _context_with_step08_constraints(scenario, _constraint_config(max_introductions_per_branch=2))

    assert cp_model.CpSolver().Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert counts.introduction_cap_constraints == 1


def test_synthetic_introduction_cap_blocks_more_than_the_configured_limit():
    scenario = _stateful_validated_scenario(
        ((1401, False, True, 1),),
        ((1402, False, True, 1),),
        ((1403, False, True, 1),),
    )
    context, _ = _context_with_step08_constraints(scenario, _constraint_config(max_introductions_per_branch=2))

    assert cp_model.CpSolver().Solve(context.model) == cp_model.INFEASIBLE


def test_do_not_introduce_options_do_not_consume_introduction_capacity():
    scenario = _stateful_validated_scenario(
        ((1501, False, True, 1),),
        ((1502, False, True, 1),),
        ((1503, False, False, 0),),
    )
    context, _ = _context_with_step08_constraints(scenario, _constraint_config(max_introductions_per_branch=2))

    assert cp_model.CpSolver().Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_existing_listed_options_do_not_consume_introduction_capacity():
    scenario = _stateful_validated_scenario(
        ((1601, True, True, 1),),
        ((1602, True, True, 2),),
        ((1603, True, True, 5),),
    )
    context, _ = _context_with_step08_constraints(scenario, _constraint_config(max_introductions_per_branch=0))

    assert cp_model.CpSolver().Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_q1_does_not_change_the_selected_sku_count():
    scenario = _stateful_validated_scenario(((1701, True, True, 5),))
    context, _ = _context_with_step08_constraints(scenario, _constraint_config(max_skus_per_branch=1))

    assert cp_model.CpSolver().Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_null_budget_configuration_creates_no_budget_constraint():
    config = load_synthetic_constraint_config()
    scenario = _stateful_validated_scenario(((1801, True, True, 1),))
    context, counts = _context_with_step08_constraints(scenario, config)

    assert config.budget_limit is None
    assert counts.budget_constraints == 0
    assert cp_model.CpSolver().Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_null_space_configuration_creates_no_space_constraint():
    config = load_synthetic_constraint_config()
    scenario = _stateful_validated_scenario(((1901, True, True, 1),))
    _context, counts = _context_with_step08_constraints(scenario, config)

    assert config.space_limit is None
    assert counts.space_constraints == 0


def test_empty_category_configuration_creates_no_category_constraint():
    config = load_synthetic_constraint_config()
    scenario = _stateful_validated_scenario(((2001, True, True, 1),))
    _context, counts = _context_with_step08_constraints(scenario, config)

    assert config.category_minimums == {}
    assert counts.category_constraints == 0


def test_proxy_utility_objective_selects_highest_utility_without_shared_conflict():
    scenario = _objective_scenario(
        ((2101, 0.10, True, True, 1), (2102, 0.20, True, True, 2), (2103, 0.05, True, False, 0)),
    )
    context = _build_variables_for_test(scenario)
    add_exactly_one_option_constraints(context)
    objective = add_proxy_utility_objective(context)
    solver = cp_model.CpSolver()

    assert solver.Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(context.option_vars[2102]) == 1
    assert objective.objective_name == "PROXY_UTILITY"


def test_proxy_utility_objective_optimizes_the_highest_total_feasible_introductions():
    scenario = _objective_scenario(
        ((2201, 0.01, False, False, 0), (2202, 0.90, False, True, 1)),
        ((2203, 0.01, False, False, 0), (2204, 0.85, False, True, 1)),
        ((2205, 0.01, False, False, 0), (2206, 0.80, False, True, 1)),
    )
    context, _ = _context_with_step08_constraints(scenario, _constraint_config(max_introductions_per_branch=2))
    add_proxy_utility_objective(context)
    solver = cp_model.CpSolver()

    assert solver.Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert [solver.Value(context.option_vars[option_id]) for option_id in (2202, 2204, 2206)] == [1, 1, 0]


def test_exact_integer_proxy_utilities_require_no_scaling():
    utilities = (Decimal("76"), Decimal("53"), Decimal("21"))
    scale = determine_utility_scale(utilities)

    assert scale == 1
    assert [scale_proxy_utility(utility, scale=scale) for utility in utilities] == [76, 53, 21]


def test_decimal_proxy_utilities_use_the_smallest_exact_power_of_ten_scale():
    utilities = (Decimal("0.761"), Decimal("0.534"), Decimal("0.214"))

    scale = determine_utility_scale(utilities)

    assert scale == 1000
    assert [scale_proxy_utility(utility, scale=scale) for utility in utilities] == [761, 534, 214]


def test_proxy_utility_rounding_is_deterministic_round_half_up():
    utility = Decimal("0.1235")
    scale = determine_utility_scale((utility,))

    assert scale == 10_000
    assert scale_proxy_utility(utility, scale=scale) == 1235


def test_proxy_utility_scaling_preserves_ordering():
    utilities = (Decimal("0.761"), Decimal("0.534"), Decimal("0.214"))
    scale = determine_utility_scale(utilities)
    coefficients = [scale_proxy_utility(utility, scale=scale) for utility in utilities]

    assert coefficients[0] > coefficients[1] > coefficients[2]


def test_zero_proxy_utility_remains_an_objective_term():
    scenario = _objective_scenario(((2401, 0, False, False, 0), (2402, 1, False, True, 1)))
    context = _build_variables_for_test(scenario)
    add_exactly_one_option_constraints(context)
    objective = add_proxy_utility_objective(context)

    assert objective.term_count == 2
    assert objective.scaled_coefficients[2401] == 0
    assert cp_model.CpSolver().Solve(context.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


@pytest.mark.parametrize("utility", [None, float("nan"), float("inf"), -0.1])
def test_missing_nonfinite_or_out_of_contract_proxy_utility_blocks_objective(utility):
    scenario = _objective_scenario(((2501, utility, True, True, 1),))
    context = _build_variables_for_test(scenario)

    with pytest.raises(VariableBuildError, match="proxy-utility objective"):
        add_proxy_utility_objective(context)


def test_proxy_utility_metadata_has_no_financial_objective_name():
    scenario = _objective_scenario(((2601, 0.5, True, True, 1),))
    context = _build_variables_for_test(scenario)
    objective = add_proxy_utility_objective(context)

    assert objective.objective_name == "PROXY_UTILITY"
    assert "PROFIT" not in objective.objective_name
