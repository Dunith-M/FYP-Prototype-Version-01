"""Status-matrix tests for the Day-08 server-side publication gate."""

import ast
from datetime import date
from pathlib import Path

import pytest

import planning.services.publication_gate as publication_gate
from catalog.models import Branch, Product
from optimization.checker import check_and_persist_feasibility
from planning.models import DecisionOption, OptimizationResult, Recommendation
from planning.services.publication_gate import (
    PublicationGateBlocked,
    evaluate_publication_gate,
    require_ready_for_action_generation,
)


PLANNING_MONTH = date(2026, 4, 1)
SCENARIO_VERSION = "publication_gate_synthetic_v1"


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


def _valid_result(*, solver_status="OPTIMAL", run_version="gate-run-v1", additional_option=False):
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Gate branch")
    product = Product.objects.create(canonical_product_id="CP_GATE", product_code="GATE", product_desc="Gate product")
    selected = _option(branch, product)
    if additional_option:
        _option(branch, product, l0=False, q0=None, l1=True, q1=1)
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version=run_version,
        solver_status=solver_status,
    )
    result.selected_options.add(selected)
    return result, selected


def test_publication_gate_has_no_checker_or_solver_runtime_imports():
    tree = ast.parse(Path(publication_gate.__file__).read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden = ("ortools", "optimization.checker", "optimization.solver")
    assert not any(module == name or module.startswith(f"{name}.") for module in imported_modules for name in forbidden)


@pytest.mark.django_db
@pytest.mark.parametrize("solver_status", ["OPTIMAL", "FEASIBLE"])
def test_usable_solver_with_checker_pass_and_exact_plan_is_ready(solver_status):
    result, _selected = _valid_result(solver_status=solver_status)
    check_and_persist_feasibility(result.pk)

    gate = require_ready_for_action_generation(result.pk)

    assert gate.allowed is True
    assert gate.code == "READY_FOR_ACTION_ENGINE"
    assert gate.plan_fingerprint
    assert Recommendation.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize("solver_status", ["INFEASIBLE", "UNKNOWN", "ERROR"])
def test_unusable_solver_statuses_are_blocked(solver_status):
    result, _selected = _valid_result(solver_status=solver_status)

    gate = evaluate_publication_gate(result.pk)

    assert gate.allowed is False
    assert gate.code == "PUB-GATE-001"


@pytest.mark.django_db
@pytest.mark.parametrize("solver_status", ["OPTIMAL", "FEASIBLE"])
def test_checker_failure_blocks_usable_solver_status(solver_status):
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Gate branch")
    first = Product.objects.create(canonical_product_id="CP_GATE_1", product_code="GATE1", product_desc="Gate one")
    second = Product.objects.create(canonical_product_id="CP_GATE_2", product_code="GATE2", product_desc="Gate two")
    selected = _option(branch, first)
    _option(branch, second)
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version=f"gate-fail-{solver_status}",
        solver_status=solver_status,
    )
    result.selected_options.add(selected)
    check_and_persist_feasibility(result.pk)
    result.refresh_from_db()

    gate = evaluate_publication_gate(result.pk)

    assert result.feasibility_status == "FAIL"
    assert gate.allowed is False
    assert gate.code == "PUB-GATE-003"


@pytest.mark.django_db
def test_missing_or_pending_checker_blocks_a_selected_usable_plan():
    result, _selected = _valid_result()

    gate = evaluate_publication_gate(result.pk)

    assert gate.allowed is False
    assert gate.code == "PUB-GATE-002"

    result.feasibility_status = OptimizationResult.FeasibilityStatus.PASS
    result.save(update_fields=("feasibility_status",))
    missing_evidence_gate = evaluate_publication_gate(result.pk)
    assert missing_evidence_gate.allowed is False
    assert missing_evidence_gate.code == "PUB-GATE-002"


@pytest.mark.django_db
def test_no_selected_plan_blocks_even_with_manually_set_pass_status():
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version="gate-no-selection",
        solver_status=OptimizationResult.SolverStatus.OPTIMAL,
        feasibility_status=OptimizationResult.FeasibilityStatus.PASS,
    )

    gate = evaluate_publication_gate(result.pk)

    assert gate.allowed is False
    assert gate.code == "PUB-GATE-006"
    with pytest.raises(PublicationGateBlocked, match="PUB-GATE-006"):
        require_ready_for_action_generation(result.pk)


@pytest.mark.django_db
def test_stale_checker_pass_blocks_when_selected_option_changes_after_checking():
    result, selected = _valid_result(additional_option=True)
    check_and_persist_feasibility(result.pk)
    alternate = DecisionOption.objects.exclude(pk=selected.pk).get(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
    )
    result.selected_options.set((alternate,))

    gate = evaluate_publication_gate(result.pk)

    assert gate.allowed is False
    assert gate.code == "PUB-GATE-004"


@pytest.mark.django_db
def test_checker_evidence_copied_from_another_result_is_blocked_as_wrong_plan():
    first, selected = _valid_result(run_version="gate-first")
    check_and_persist_feasibility(first.pk)
    first.refresh_from_db()
    second = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version="gate-second",
        solver_status=OptimizationResult.SolverStatus.OPTIMAL,
        feasibility_status=OptimizationResult.FeasibilityStatus.PASS,
        solver_metadata={"independent_checker": first.solver_metadata["independent_checker"]},
    )
    second.selected_options.add(selected)

    gate = evaluate_publication_gate(second.pk)

    assert gate.allowed is False
    assert gate.code == "PUB-GATE-005"
