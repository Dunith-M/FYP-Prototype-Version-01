"""Validated ORM-to-plain-Python input boundary for the later CP-SAT solver."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from optimization.validation import OptionValidationResult, validate_option_set
from planning.models import DecisionOption


class ScenarioInputError(ValueError):
    """Base error for a Day-07 option scenario that cannot enter Day 08."""


class ScenarioInputBlocked(ScenarioInputError):
    """Raised only after the option-validation gate returns BLOCKED."""

    def __init__(self, result: OptionValidationResult):
        self.result = result
        super().__init__(result.summary())


@dataclass(frozen=True)
class SolverOption:
    """One traceable finite option for a future solver decision variable."""

    option_id: int
    branch_id: int
    branch_code: str
    product_id: int
    canonical_product_id: str
    l0: bool
    q0: int | None
    l1: bool
    q1: int
    utility: float
    parameter_source: str


@dataclass(frozen=True)
class DecisionUnit:
    """All finite alternatives for one branch-product decision."""

    branch_id: int
    branch_code: str
    product_id: int
    canonical_product_id: str
    options: tuple[SolverOption, ...]


@dataclass(frozen=True)
class OptimizationScenarioInput:
    """Stable, validated input for a later CP-SAT model."""

    planning_month: date
    scenario_version: str
    parameter_source: str
    units: tuple[DecisionUnit, ...]
    validation_result: OptionValidationResult

    @property
    def decision_option_count(self) -> int:
        return sum(len(unit.options) for unit in self.units)


def load_scenario_for_optimization(*, planning_month: date, scenario_version: str) -> OptimizationScenarioInput:
    """Load one persisted Day-07 synthetic option scenario through the ORM only."""
    if not scenario_version.strip():
        raise ScenarioInputError("Scenario version must be non-blank.")
    options = DecisionOption.objects.filter(
        planning_month=planning_month,
        scenario_version=scenario_version,
    ).select_related("branch", "product").order_by(
        "branch__branch_code",
        "product__canonical_product_id",
        "l1",
        "q1",
        "pk",
    )
    return build_optimization_scenario_input(
        planning_month=planning_month,
        scenario_version=scenario_version,
        option_rows=options,
    )


def build_optimization_scenario_input(
    *,
    planning_month: date,
    scenario_version: str,
    option_rows: Iterable[DecisionOption],
) -> OptimizationScenarioInput:
    """Validate already-scoped options and group only a PASS result."""
    rows = tuple(option_rows)
    validation = validate_option_set(
        planning_month=planning_month,
        scenario_version=scenario_version,
        option_rows=rows,
    )
    if validation.status != "PASS":
        raise ScenarioInputBlocked(validation)

    grouped: dict[tuple[int, int], list[SolverOption]] = {}
    identities: dict[tuple[int, int], tuple[str, str]] = {}
    for row in rows:
        key = (row.branch_id, row.product_id)
        identities[key] = (row.branch.branch_code, row.product.canonical_product_id)
        grouped.setdefault(key, []).append(
            SolverOption(
                option_id=row.pk,
                branch_id=row.branch_id,
                branch_code=row.branch.branch_code,
                product_id=row.product_id,
                canonical_product_id=row.product.canonical_product_id,
                l0=row.l0,
                q0=row.q0,
                l1=row.l1,
                q1=row.q1,
                utility=float(row.evidence_score),
                parameter_source=row.parameter_source,
            )
        )

    units = tuple(
        DecisionUnit(
            branch_id=branch_id,
            branch_code=identities[(branch_id, product_id)][0],
            product_id=product_id,
            canonical_product_id=identities[(branch_id, product_id)][1],
            options=tuple(options),
        )
        for (branch_id, product_id), options in grouped.items()
    )
    return OptimizationScenarioInput(
        planning_month=planning_month,
        scenario_version=scenario_version,
        parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
        units=units,
        validation_result=validation,
    )
