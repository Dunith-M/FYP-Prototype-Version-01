"""Independent ordinary-Python feasibility checks for persisted Day-08 plans.

This module reads the database result and finite option rows after execution.
It intentionally contains no model-building or solver-runtime dependency.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from planning.models import DecisionOption, OptimizationResult
from planning.services.plan_identity import build_selected_plan_fingerprint, constraint_signature_from_values
from planning.services.synthetic_state import SYNTHETIC_SOURCE, load_synthetic_options


RULE_PASS = "PASS"
RULE_FAIL = "FAIL"
RULE_NOT_CONFIGURED = "NOT_CONFIGURED"
USABLE_SOLVER_STATUSES = frozenset(
    {OptimizationResult.SolverStatus.OPTIMAL, OptimizationResult.SolverStatus.FEASIBLE}
)


class CheckerInputError(ValueError):
    """Raised when no persisted usable plan is eligible for independent checking."""


@dataclass(frozen=True)
class CheckerConstraintConfig:
    """Only the configured synthetic-demo limits to independently recalculate."""

    mode: str
    parameter_source: str
    max_skus_per_branch: int
    max_introductions_per_branch: int
    category_minimums: dict[str, object]
    budget_limit: object
    space_limit: object


@dataclass(frozen=True)
class RuleResult:
    """Deterministic evidence for one independently evaluated rule."""

    rule_code: str
    rule_name: str
    status: str
    scope: str
    observed_value: object
    allowed_value: object
    message: str


@dataclass(frozen=True)
class FeasibilityReport:
    """PASS/FAIL result and per-rule evidence for one persisted plan."""

    optimization_result_id: int
    planning_month: date
    scenario_version: str
    solver_status: str
    checked_option_count: int
    checked_unit_count: int
    rules: tuple[RuleResult, ...]

    @property
    def status(self) -> str:
        return RULE_FAIL if any(rule.status == RULE_FAIL for rule in self.rules) else RULE_PASS

    @property
    def errors(self) -> tuple[RuleResult, ...]:
        return tuple(rule for rule in self.rules if rule.status == RULE_FAIL)


def load_checker_constraint_config(config_path: Path | None = None) -> CheckerConstraintConfig:
    """Read the synthetic configuration without using any constraint builder."""
    path = config_path or Path(settings.BASE_DIR) / "config" / "synthetic" / "options.yaml"
    options = load_synthetic_options(path)
    constraints = options.get("constraints")
    if options.get("mode") != "RESEARCH_SYNTHETIC" or not isinstance(constraints, dict):
        raise CheckerInputError("Checker requires RESEARCH_SYNTHETIC configuration with a constraints mapping.")

    max_skus = constraints.get("max_skus_per_branch")
    max_introductions = constraints.get("max_introductions_per_branch")
    category_minimums = constraints.get("category_minimums")
    if (
        not isinstance(max_skus, int)
        or isinstance(max_skus, bool)
        or max_skus < 0
        or not isinstance(max_introductions, int)
        or isinstance(max_introductions, bool)
        or max_introductions < 0
        or not isinstance(category_minimums, dict)
    ):
        raise CheckerInputError("Synthetic checker configuration has invalid SKU, introduction, or category values.")

    return CheckerConstraintConfig(
        mode=options["mode"],
        parameter_source=options["parameter_source"],
        max_skus_per_branch=max_skus,
        max_introductions_per_branch=max_introductions,
        category_minimums=category_minimums,
        budget_limit=constraints.get("budget_limit"),
        space_limit=constraints.get("space_limit"),
    )


def checker_constraint_signature(constraint_config: CheckerConstraintConfig) -> dict[str, object]:
    """Return the same configured-constraint identity used in a checker run."""
    return constraint_signature_from_values(
        mode=constraint_config.mode,
        parameter_source=constraint_config.parameter_source,
        max_skus_per_branch=constraint_config.max_skus_per_branch,
        max_introductions_per_branch=constraint_config.max_introductions_per_branch,
        category_minimums=constraint_config.category_minimums,
        budget_limit=constraint_config.budget_limit,
        space_limit=constraint_config.space_limit,
    )


def _pair_key(option: object) -> tuple[int | None, int | None]:
    return getattr(option, "branch_id", None), getattr(option, "product_id", None)


def _pair_label(option: object) -> str:
    branch = getattr(getattr(option, "branch", None), "branch_code", None) or f"branch_id={getattr(option, 'branch_id', None)}"
    product = getattr(getattr(option, "product", None), "canonical_product_id", None) or f"product_id={getattr(option, 'product_id', None)}"
    return f"{branch}/{product}"


def _valid_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _original_units(original_options: Iterable[object]) -> dict[tuple[int | None, int | None], tuple[object, ...]]:
    grouped: dict[tuple[int | None, int | None], list[object]] = defaultdict(list)
    for option in original_options:
        grouped[_pair_key(option)].append(option)
    return {key: tuple(options) for key, options in grouped.items()}


def evaluate_persisted_selection(
    *,
    optimization_result: object,
    selected_options: Iterable[object],
    original_options: Iterable[object],
    constraint_config: CheckerConstraintConfig,
) -> FeasibilityReport:
    """Independently recalculate the persisted plan's configured hard rules.

    Inputs are plain option objects so deterministic tests can mutate a
    snapshot without weakening database constraints. The production entry
    point below supplies freshly queried persisted ORM rows.
    """
    selected = tuple(selected_options)
    original = tuple(original_options)
    units = _original_units(original)
    selected_by_pair: dict[tuple[int | None, int | None], list[object]] = defaultdict(list)
    for option in selected:
        selected_by_pair[_pair_key(option)].append(option)

    rules: list[RuleResult] = []
    expected_pairs = set(units)
    selected_pairs = set(selected_by_pair)
    pair_violations: list[str] = []
    for pair, options in sorted(units.items(), key=lambda item: _pair_label(item[1][0])):
        selected_count = len(selected_by_pair.get(pair, ()))
        if selected_count != 1:
            pair_violations.append(f"{_pair_label(options[0])}: selected_count={selected_count}")
    for pair in selected_pairs - expected_pairs:
        pair_violations.append(f"selected pair outside scenario: {pair}")
    rules.append(
        RuleResult(
            "CHECK-EXACTLY-ONE-001",
            "Exactly one selected option per branch-product pair",
            RULE_PASS if not pair_violations else RULE_FAIL,
            "scenario",
            {"units": len(units), "selected_options": len(selected)},
            "one selection per included pair",
            "All included pairs have exactly one persisted selection." if not pair_violations else "; ".join(pair_violations),
        )
    )

    original_ids = {getattr(option, "pk", None) for option in original}
    membership_violations: list[str] = []
    for option in selected:
        if (
            getattr(option, "pk", None) not in original_ids
            or getattr(option, "planning_month", None) != getattr(optimization_result, "planning_month", None)
            or getattr(option, "scenario_version", None) != getattr(optimization_result, "scenario_version", None)
            or _pair_key(option) not in expected_pairs
        ):
            membership_violations.append(f"option_id={getattr(option, 'pk', None)}")
    rules.append(
        RuleResult(
            "CHECK-MEMBERSHIP-001",
            "Selected option existed in original finite option set",
            RULE_PASS if not membership_violations else RULE_FAIL,
            "scenario",
            len(selected),
            len(original),
            "All selected options belong to the persisted scenario option set."
            if not membership_violations
            else "Invalid selections: " + ", ".join(membership_violations),
        )
    )

    provenance_violations = [
        f"option_id={getattr(option, 'pk', None)} source={getattr(option, 'parameter_source', None)!r}"
        for option in selected
        if getattr(option, "parameter_source", None) != constraint_config.parameter_source
    ]
    rules.append(
        RuleResult(
            "CHECK-PROVENANCE-001",
            "Synthetic provenance retained",
            RULE_PASS if not provenance_violations else RULE_FAIL,
            "selected options",
            Counter(getattr(option, "parameter_source", None) for option in selected),
            constraint_config.parameter_source,
            "Every selected option remains SYNTHETIC_DEMO."
            if not provenance_violations
            else "; ".join(provenance_violations),
        )
    )

    l1_q1_violations: list[str] = []
    reference_violations: list[str] = []
    for option in selected:
        l1, q1 = getattr(option, "l1", None), getattr(option, "q1", None)
        if not isinstance(l1, bool) or (not l1 and q1 != 0) or (l1 and not _valid_positive_integer(q1)):
            l1_q1_violations.append(f"option_id={getattr(option, 'pk', None)} L1={l1!r} q1={q1!r}")
        pair_options = units.get(_pair_key(option), ())
        reference_states = {(getattr(row, "l0", None), getattr(row, "q0", None)) for row in pair_options}
        current_state = (getattr(option, "l0", None), getattr(option, "q0", None))
        if (
            len(reference_states) != 1
            or current_state not in reference_states
            or (current_state[0] is True and not _valid_positive_integer(current_state[1]))
            or (current_state[0] is False and current_state[1] is not None)
            or not isinstance(current_state[0], bool)
        ):
            reference_violations.append(f"option_id={getattr(option, 'pk', None)} L0={current_state[0]!r} q0={current_state[1]!r}")
    rules.extend(
        (
            RuleResult(
                "CHECK-L1-Q1-001",
                "Proposed listing and target quantity validity",
                RULE_PASS if not l1_q1_violations else RULE_FAIL,
                "selected options",
                len(selected) - len(l1_q1_violations),
                len(selected),
                "All selected L1/q1 combinations follow the Day-07 option contract."
                if not l1_q1_violations
                else "; ".join(l1_q1_violations),
            ),
            RuleResult(
                "CHECK-REFERENCE-001",
                "Reference L0/q0 state integrity",
                RULE_PASS if not reference_violations else RULE_FAIL,
                "branch-product baseline",
                len(selected) - len(reference_violations),
                len(selected),
                "Selected options retain their original pair baseline."
                if not reference_violations
                else "; ".join(reference_violations),
            ),
        )
    )

    branches = {getattr(option, "branch_id", None) for option in original}
    branch_labels = {
        getattr(option, "branch_id", None): getattr(getattr(option, "branch", None), "branch_code", None)
        for option in (*original, *selected)
    }
    sku_observed = {
        branch_id: len({getattr(option, "product_id", None) for option in selected if getattr(option, "branch_id", None) == branch_id and getattr(option, "l1", None) is True})
        for branch_id in branches
    }
    sku_violations = {
        branch_labels.get(branch_id) or str(branch_id): count
        for branch_id, count in sku_observed.items()
        if count > constraint_config.max_skus_per_branch
    }
    rules.append(
        RuleResult(
            "CHECK-SKU-001",
            "Maximum listed SKUs per branch",
            RULE_PASS if not sku_violations else RULE_FAIL,
            "per branch",
            {branch_labels.get(branch_id) or str(branch_id): count for branch_id, count in sku_observed.items()},
            constraint_config.max_skus_per_branch,
            "Selected listed products are within the synthetic SKU cap."
            if not sku_violations
            else f"SKU cap violations: {sku_violations}",
        )
    )

    introduction_observed = {
        branch_id: sum(
            1
            for option in selected
            if getattr(option, "branch_id", None) == branch_id
            and getattr(option, "l0", None) is False
            and getattr(option, "l1", None) is True
        )
        for branch_id in branches
    }
    introduction_violations = {
        branch_labels.get(branch_id) or str(branch_id): count
        for branch_id, count in introduction_observed.items()
        if count > constraint_config.max_introductions_per_branch
    }
    rules.append(
        RuleResult(
            "CHECK-INTRODUCTION-001",
            "Maximum introductions per branch",
            RULE_PASS if not introduction_violations else RULE_FAIL,
            "per branch",
            {branch_labels.get(branch_id) or str(branch_id): count for branch_id, count in introduction_observed.items()},
            constraint_config.max_introductions_per_branch,
            "Selected unlisted-to-listed transitions are within the synthetic introduction cap."
            if not introduction_violations
            else f"Introduction cap violations: {introduction_violations}",
        )
    )

    for code, name, configured in (
        ("CHECK-CATEGORY-001", "Category constraints", bool(constraint_config.category_minimums)),
        ("CHECK-BUDGET-001", "Budget constraint", constraint_config.budget_limit is not None),
        ("CHECK-SPACE-001", "Space constraint", constraint_config.space_limit is not None),
        ("CHECK-SUPPLY-001", "Supply constraint", False),
    ):
        rules.append(
            RuleResult(
                code,
                name,
                RULE_FAIL if configured else RULE_NOT_CONFIGURED,
                "scenario configuration",
                "configured" if configured else None,
                None,
                "This constraint is not configured for the current synthetic demonstration."
                if not configured
                else "This constraint is configured but is outside the currently permitted independent-checker scope.",
            )
        )

    return FeasibilityReport(
        optimization_result_id=getattr(optimization_result, "pk"),
        planning_month=getattr(optimization_result, "planning_month"),
        scenario_version=getattr(optimization_result, "scenario_version"),
        solver_status=getattr(optimization_result, "solver_status"),
        checked_option_count=len(selected),
        checked_unit_count=len(units),
        rules=tuple(rules),
    )


def check_optimization_result(
    optimization_result_id: int,
    *,
    constraint_config: CheckerConstraintConfig | None = None,
) -> FeasibilityReport:
    """Read one persisted usable plan and independently evaluate it in Python."""
    result = OptimizationResult.objects.get(pk=optimization_result_id)
    if result.solver_status not in USABLE_SOLVER_STATUSES:
        raise CheckerInputError(
            f"OptimizationResult {result.pk} has solver_status={result.solver_status}; only OPTIMAL or FEASIBLE can enter the checker."
        )
    config = constraint_config or load_checker_constraint_config()
    selected_options, original_options = _persisted_plan_rows(result)
    return evaluate_persisted_selection(
        optimization_result=result,
        selected_options=selected_options,
        original_options=original_options,
        constraint_config=config,
    )


def _persisted_plan_rows(result: OptimizationResult) -> tuple[tuple[DecisionOption, ...], tuple[DecisionOption, ...]]:
    """Query the stored selection and original finite option set from PostgreSQL."""
    selected_options = tuple(result.selected_options.select_related("branch", "product").all())
    original_options = tuple(
        DecisionOption.objects.filter(
            planning_month=result.planning_month,
            scenario_version=result.scenario_version,
        ).select_related("branch", "product")
    )
    return selected_options, original_options


def check_and_persist_feasibility(optimization_result_id: int) -> FeasibilityReport:
    """Persist only PASS/FAIL feasibility; never alter selected options or solver state."""
    with transaction.atomic():
        result = OptimizationResult.objects.select_for_update().get(pk=optimization_result_id)
        if result.solver_status not in USABLE_SOLVER_STATUSES:
            raise CheckerInputError(
                f"OptimizationResult {result.pk} has solver_status={result.solver_status}; checker execution is blocked."
            )
        constraint_config = load_checker_constraint_config()
        selected_options, original_options = _persisted_plan_rows(result)
        report = evaluate_persisted_selection(
            optimization_result=result,
            selected_options=selected_options,
            original_options=original_options,
            constraint_config=constraint_config,
        )
        metadata = dict(result.solver_metadata or {})
        metadata["independent_checker"] = {
            "version": "day08_independent_checker_v1",
            "optimization_result_id": result.pk,
            "status": report.status,
            "plan_fingerprint": build_selected_plan_fingerprint(
                result,
                selected_options,
                constraint_signature=checker_constraint_signature(constraint_config),
            ),
            "checked_option_count": report.checked_option_count,
            "checked_unit_count": report.checked_unit_count,
            "rule_statuses": {rule.rule_code: rule.status for rule in report.rules},
            "completed_at": timezone.now().isoformat(),
        }
        result.feasibility_status = (
            OptimizationResult.FeasibilityStatus.PASS
            if report.status == RULE_PASS
            else OptimizationResult.FeasibilityStatus.FAIL
        )
        result.solver_metadata = metadata
        result.save(update_fields=("feasibility_status", "solver_metadata"))
    return report
