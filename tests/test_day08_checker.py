"""Known-answer tests for the independent persisted-plan feasibility checker."""

import ast
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

import optimization.checker as checker
from catalog.models import Branch, Product
from optimization.checker import (
    CheckerConstraintConfig,
    CheckerInputError,
    RULE_FAIL,
    RULE_NOT_CONFIGURED,
    RULE_PASS,
    check_and_persist_feasibility,
    check_optimization_result,
    evaluate_persisted_selection,
)
from optimization.runner import run_and_persist_optimization
from planning.models import DecisionOption, OptimizationResult
from planning.services.publication_gate import evaluate_publication_gate
from planning.services.recommendation_service import create_recommendations_for_result
from planning.services.publication_gate import PublicationGateBlocked


PLANNING_MONTH = date(2026, 4, 1)
SCENARIO_VERSION = "checker_synthetic_v1"


def _config(*, max_skus=150, max_introductions=5):
    return CheckerConstraintConfig(
        mode="RESEARCH_SYNTHETIC",
        parameter_source="SYNTHETIC_DEMO",
        max_skus_per_branch=max_skus,
        max_introductions_per_branch=max_introductions,
        category_minimums={},
        budget_limit=None,
        space_limit=None,
    )


def _option(branch, product, *, l0, q0, l1, q1, source="SYNTHETIC_DEMO", scenario_version=SCENARIO_VERSION):
    return DecisionOption.objects.create(
        branch=branch,
        product=product,
        planning_month=PLANNING_MONTH,
        scenario_version=scenario_version,
        l0=l0,
        q0=q0,
        l1=l1,
        q1=q1,
        option_type=(DecisionOption.OptionType.LISTED_TARGET if l1 and l0 else DecisionOption.OptionType.LIST_TRIAL if l1 else DecisionOption.OptionType.DELIST if l0 else DecisionOption.OptionType.REMAIN_UNLISTED),
        evidence_score=0.5,
        evidence_source=DecisionOption.EvidenceSource.HISTORICAL_LAG_1_NORMALIZED,
        parameter_source=source,
    )


def _branch_and_products(count):
    branch = Branch.objects.create(branch_code="BIL01", branch_name="Checker branch")
    products = [
        Product.objects.create(canonical_product_id=f"CP_CHECK_{index}", product_code=f"CHECK{index}", product_desc=f"Checker {index}")
        for index in range(1, count + 1)
    ]
    return branch, products


def _result(selected_options, *, solver_status="OPTIMAL", run_version="checker-run-v1"):
    result = OptimizationResult.objects.create(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version=run_version,
        solver_status=solver_status,
    )
    result.selected_options.set(selected_options)
    return result


def _rules(report):
    return {rule.rule_code: rule for rule in report.rules}


def test_checker_has_no_solver_or_ortools_import_dependency():
    tree = ast.parse(Path(checker.__file__).read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert not any(module == "ortools" or module.startswith("ortools.") for module in imported_modules)
    assert "optimization.solver" not in imported_modules


@pytest.mark.django_db
def test_known_good_persisted_plan_passes_and_updates_only_feasibility_status():
    branch, products = _branch_and_products(2)
    existing = _option(branch, products[0], l0=True, q0=1, l1=False, q1=0)
    introduced = _option(branch, products[1], l0=False, q0=None, l1=True, q1=1)
    result = _result((existing, introduced))
    original_ids = set(result.selected_options.values_list("pk", flat=True))

    report = check_and_persist_feasibility(result.pk)
    result.refresh_from_db()

    assert report.status == RULE_PASS
    assert result.solver_status == "OPTIMAL"
    assert result.feasibility_status == "PASS"
    assert set(result.selected_options.values_list("pk", flat=True)) == original_ids
    assert all(rule.status != RULE_FAIL for rule in report.rules)
    assert _rules(report)["CHECK-CATEGORY-001"].status == RULE_NOT_CONFIGURED
    assert _rules(report)["CHECK-BUDGET-001"].status == RULE_NOT_CONFIGURED
    assert _rules(report)["CHECK-SPACE-001"].status == RULE_NOT_CONFIGURED
    assert _rules(report)["CHECK-SUPPLY-001"].status == RULE_NOT_CONFIGURED


@pytest.mark.django_db
def test_missing_selection_for_a_persisted_pair_fails_checker():
    branch, products = _branch_and_products(2)
    selected = _option(branch, products[0], l0=True, q0=1, l1=False, q1=0)
    _option(branch, products[1], l0=False, q0=None, l1=False, q1=0)
    result = _result((selected,))

    report = check_and_persist_feasibility(result.pk)
    result.refresh_from_db()

    assert report.status == RULE_FAIL
    assert _rules(report)["CHECK-EXACTLY-ONE-001"].status == RULE_FAIL
    assert result.feasibility_status == "FAIL"


@pytest.mark.django_db
def test_two_persisted_options_for_one_pair_fails_checker():
    branch, products = _branch_and_products(1)
    delisted = _option(branch, products[0], l0=True, q0=1, l1=False, q1=0)
    listed = _option(branch, products[0], l0=True, q0=1, l1=True, q1=1)
    result = _result((delisted, listed))

    report = check_optimization_result(result.pk)

    assert report.status == RULE_FAIL
    assert _rules(report)["CHECK-EXACTLY-ONE-001"].status == RULE_FAIL


@pytest.mark.django_db
def test_sku_and_introduction_violations_are_independently_detected():
    branch, products = _branch_and_products(4)
    selected = [_option(branch, product, l0=False, q0=None, l1=True, q1=1) for product in products]
    result = _result(selected)

    report = check_optimization_result(result.pk, constraint_config=_config(max_skus=3, max_introductions=2))
    rules = _rules(report)

    assert report.status == RULE_FAIL
    assert rules["CHECK-SKU-001"].status == RULE_FAIL
    assert rules["CHECK-INTRODUCTION-001"].status == RULE_FAIL


@pytest.mark.django_db
def test_boundary_sku_and_introduction_limits_pass():
    branch, products = _branch_and_products(2)
    selected = [_option(branch, product, l0=False, q0=None, l1=True, q1=1) for product in products]
    result = _result(selected)

    report = check_optimization_result(result.pk, constraint_config=_config(max_skus=2, max_introductions=2))

    assert report.status == RULE_PASS
    assert _rules(report)["CHECK-SKU-001"].status == RULE_PASS
    assert _rules(report)["CHECK-INTRODUCTION-001"].status == RULE_PASS


@pytest.mark.django_db
def test_wrong_scenario_option_and_unknown_option_fail_membership_check():
    branch, products = _branch_and_products(1)
    original = _option(branch, products[0], l0=True, q0=1, l1=False, q1=0)
    other = _option(branch, products[0], l0=True, q0=1, l1=True, q1=1, scenario_version="other_scenario")
    result = _result((other,))

    report = check_optimization_result(result.pk)
    rules = _rules(report)

    assert report.status == RULE_FAIL
    assert rules["CHECK-MEMBERSHIP-001"].status == RULE_FAIL
    assert original.pk not in set(result.selected_options.values_list("pk", flat=True))


@pytest.mark.django_db
def test_missing_or_unexpected_synthetic_provenance_fails_checker():
    branch, products = _branch_and_products(1)
    option = _option(branch, products[0], l0=False, q0=None, l1=False, q1=0, source="OPERATIONAL")
    result = _result((option,))

    report = check_optimization_result(result.pk)

    assert report.status == RULE_FAIL
    assert _rules(report)["CHECK-PROVENANCE-001"].status == RULE_FAIL


@pytest.mark.django_db
def test_invalid_unlisted_quantity_is_detected_from_a_mutated_plain_python_snapshot():
    branch, products = _branch_and_products(1)
    original = _option(branch, products[0], l0=True, q0=1, l1=False, q1=0)
    result = _result((original,))
    invalid_snapshot = SimpleNamespace(
        pk=original.pk,
        branch_id=original.branch_id,
        product_id=original.product_id,
        branch=original.branch,
        product=original.product,
        planning_month=original.planning_month,
        scenario_version=original.scenario_version,
        l0=True,
        q0=1,
        l1=False,
        q1=4,
        parameter_source="SYNTHETIC_DEMO",
    )

    report = evaluate_persisted_selection(
        optimization_result=result,
        selected_options=(invalid_snapshot,),
        original_options=(original,),
        constraint_config=_config(),
    )

    assert report.status == RULE_FAIL
    assert _rules(report)["CHECK-L1-Q1-001"].status == RULE_FAIL


@pytest.mark.django_db
def test_unusable_solver_status_is_blocked_before_checker_execution():
    result = _result((), solver_status="UNKNOWN")

    with pytest.raises(CheckerInputError, match="only OPTIMAL or FEASIBLE"):
        check_optimization_result(result.pk)


@pytest.mark.django_db
def test_solve_pass_corrupt_duplicate_selection_then_checker_fails_without_solver_rerun():
    """Critical independent-checker proof: persisted corruption defeats a valid solver plan."""
    branch, products = _branch_and_products(2)
    _option(branch, products[0], l0=True, q0=1, l1=True, q1=1)
    _option(branch, products[0], l0=True, q0=1, l1=False, q1=0)
    _option(branch, products[1], l0=False, q0=None, l1=False, q1=0)
    _option(branch, products[1], l0=False, q0=None, l1=True, q1=1)

    outcome = run_and_persist_optimization(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        run_version="checker-mutation-solve-pass-v1",
    )
    result = outcome.optimization_result
    assert outcome.execution.solver_status in {"OPTIMAL", "FEASIBLE"}
    assert result.selected_options.count() == 2

    before = check_and_persist_feasibility(result.pk)
    result.refresh_from_db()
    assert before.status == RULE_PASS
    assert result.solver_status == outcome.execution.solver_status
    assert result.feasibility_status == OptimizationResult.FeasibilityStatus.PASS

    # Test-database-only corruption after CP-SAT completed: select an unchosen
    # finite option from an already selected branch-product pair. Do not solve again.
    selected_ids = set(result.selected_options.values_list("pk", flat=True))
    duplicate = DecisionOption.objects.filter(
        planning_month=PLANNING_MONTH,
        scenario_version=SCENARIO_VERSION,
        branch=branch,
        product=products[0],
    ).exclude(pk__in=selected_ids).get()
    result.selected_options.add(duplicate)

    after = check_and_persist_feasibility(result.pk)
    result.refresh_from_db()
    assert after.status == RULE_FAIL
    assert _rules(after)["CHECK-EXACTLY-ONE-001"].status == RULE_FAIL
    assert result.solver_status == outcome.execution.solver_status
    assert result.feasibility_status == OptimizationResult.FeasibilityStatus.FAIL
    assert evaluate_publication_gate(result.pk).allowed is False
    with pytest.raises(PublicationGateBlocked):
        create_recommendations_for_result(result.pk)
    assert result.recommendations.count() == 0
