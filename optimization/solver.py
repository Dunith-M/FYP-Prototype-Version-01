"""CP-SAT option-variable construction from an already validated scenario."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from django.conf import settings
from ortools.sat.python import cp_model

from optimization.scenario_loader import DecisionUnit, OptimizationScenarioInput
from planning.services.synthetic_state import SYNTHETIC_SOURCE, load_synthetic_options


class VariableBuildError(ValueError):
    """Raised when something other than a valid finite scenario reaches CP-SAT."""


@dataclass(frozen=True)
class SyntheticConstraintConfig:
    """Enabled research-demo limits; never Singer operational policy."""

    mode: str
    parameter_source: str
    max_skus_per_branch: int
    max_introductions_per_branch: int
    category_minimums: dict[str, object]
    budget_limit: None
    space_limit: None


@dataclass(frozen=True)
class SyntheticConstraintCounts:
    """Traceable count of the explicitly enabled synthetic constraints."""

    sku_cap_constraints: int
    introduction_cap_constraints: int
    category_constraints: int = 0
    budget_constraints: int = 0
    space_constraints: int = 0
    supply_constraints: int = 0


@dataclass(frozen=True)
class ProxyUtilityObjective:
    """Recalculable metadata for the non-financial CP-SAT objective."""

    objective_name: str
    utility_scale: int
    rounding_mode: str
    term_count: int
    original_utilities: dict[int, Decimal]
    scaled_coefficients: dict[int, int]

    @property
    def original_utility_range(self) -> tuple[Decimal, Decimal]:
        values = tuple(self.original_utilities.values())
        return min(values), max(values)

    @property
    def scaled_utility_range(self) -> tuple[int, int]:
        values = tuple(self.scaled_coefficients.values())
        return min(values), max(values)


@dataclass(frozen=True)
class CpSatBuildContext:
    """Model and exact DecisionOption.id-to-BoolVar mapping for later stages."""

    model: cp_model.CpModel
    scenario: OptimizationScenarioInput
    option_vars: dict[int, cp_model.IntVar]

    def variables_for_unit(self, unit: DecisionUnit) -> tuple[cp_model.IntVar, ...]:
        """Return the already-created variables for one grouped decision unit."""
        return tuple(self.option_vars[option.option_id] for option in unit.options)


def load_synthetic_constraint_config(config_path: Path | None = None) -> SyntheticConstraintConfig:
    """Load only the currently permitted synthetic limits from YAML configuration."""
    path = config_path or Path(settings.BASE_DIR) / "config" / "synthetic" / "options.yaml"
    options = load_synthetic_options(path)
    constraints = options.get("constraints")
    if options.get("mode") != "RESEARCH_SYNTHETIC" or not isinstance(constraints, dict):
        raise VariableBuildError("Synthetic constraint configuration must declare RESEARCH_SYNTHETIC constraints.")

    max_skus = constraints.get("max_skus_per_branch")
    max_introductions = constraints.get("max_introductions_per_branch")
    category_minimums = constraints.get("category_minimums")
    budget_limit = constraints.get("budget_limit")
    space_limit = constraints.get("space_limit")
    if (
        not isinstance(max_skus, int)
        or isinstance(max_skus, bool)
        or max_skus < 0
        or not isinstance(max_introductions, int)
        or isinstance(max_introductions, bool)
        or max_introductions < 0
        or not isinstance(category_minimums, dict)
    ):
        raise VariableBuildError("Synthetic max-SKU, max-introduction, and category configuration is invalid.")
    if budget_limit is not None or space_limit is not None:
        raise VariableBuildError("Budget and space constraints are not enabled for the synthetic MVP scenario.")

    return SyntheticConstraintConfig(
        mode=options["mode"],
        parameter_source=options["parameter_source"],
        max_skus_per_branch=max_skus,
        max_introductions_per_branch=max_introductions,
        category_minimums=category_minimums,
        budget_limit=budget_limit,
        space_limit=space_limit,
    )


def build_option_variables(validated_scenario: OptimizationScenarioInput) -> CpSatBuildContext:
    """Create exactly one Boolean variable for every validated DecisionOption.

    This function intentionally adds no selection, feasibility, objective, or
    solver-execution behavior. Its only output is a traceable finite-option
    variable mapping for the next Day-08 step.
    """
    if not isinstance(validated_scenario, OptimizationScenarioInput):
        raise VariableBuildError("CP-SAT variable construction requires an OptimizationScenarioInput from the validation gate.")
    if validated_scenario.validation_result.status != "PASS":
        raise VariableBuildError("CP-SAT variable construction requires a validation PASS scenario input.")
    if not validated_scenario.units or validated_scenario.decision_option_count <= 0:
        raise VariableBuildError("CP-SAT variable construction requires at least one validated DecisionOption.")

    option_ids: set[int] = set()
    option_count = 0
    for unit in validated_scenario.units:
        if not unit.options:
            raise VariableBuildError(
                f"CP-SAT variable construction encountered an empty decision unit: branch_id={unit.branch_id}; product_id={unit.product_id}."
            )
        for option in unit.options:
            option_id = option.option_id
            if not isinstance(option_id, int) or isinstance(option_id, bool) or option_id <= 0:
                raise VariableBuildError(f"CP-SAT variable construction encountered an invalid DecisionOption id: {option_id!r}.")
            if option_id in option_ids:
                raise VariableBuildError(f"Duplicate DecisionOption id {option_id} encountered while constructing CP-SAT variables.")
            option_ids.add(option_id)
            option_count += 1

    if option_count != validated_scenario.decision_option_count:
        raise VariableBuildError(
            f"CP-SAT option-variable count mismatch: expected {validated_scenario.decision_option_count}, built {option_count}."
        )

    model = cp_model.CpModel()
    option_vars: dict[int, cp_model.IntVar] = {}
    for unit in validated_scenario.units:
        for option in unit.options:
            option_vars[option.option_id] = model.NewBoolVar(f"option_{option.option_id}")
    if len(option_vars) != option_count:
        raise VariableBuildError(
            f"CP-SAT option-variable mapping mismatch: expected {option_count}, built {len(option_vars)}."
        )
    return CpSatBuildContext(model=model, scenario=validated_scenario, option_vars=option_vars)


def add_exactly_one_option_constraints(build_context: CpSatBuildContext) -> int:
    """Require exactly one existing option for each branch-product decision unit.

    This deliberately reuses the Step-06 model and BoolVars. It adds neither
    shared constraints nor an objective and never creates an additional
    decision variable.
    """
    if not isinstance(build_context, CpSatBuildContext):
        raise VariableBuildError("Exactly-one constraint construction requires a CpSatBuildContext.")
    if build_context.scenario.validation_result.status != "PASS":
        raise VariableBuildError("Exactly-one constraint construction requires a validation PASS scenario input.")

    constraint_count = 0
    for unit in build_context.scenario.units:
        if not unit.options:
            raise VariableBuildError(
                "Cannot build exactly-one constraint for "
                f"{unit.branch_code}/{unit.canonical_product_id}: decision unit contains zero options."
            )
        try:
            unit_vars = build_context.variables_for_unit(unit)
        except KeyError as exc:
            raise VariableBuildError(
                "Cannot build exactly-one constraint for "
                f"{unit.branch_code}/{unit.canonical_product_id}: option_id={exc.args[0]} has no BoolVar mapping."
            ) from exc
        if len(unit_vars) != len(unit.options):
            raise VariableBuildError(
                "Cannot build exactly-one constraint for "
                f"{unit.branch_code}/{unit.canonical_product_id}: incomplete BoolVar mapping."
            )
        build_context.model.AddExactlyOne(unit_vars)
        constraint_count += 1

    if constraint_count != len(build_context.scenario.units):
        raise VariableBuildError(
            "Exactly-one constraint count mismatch: "
            f"expected {len(build_context.scenario.units)}, built {constraint_count}."
        )
    return constraint_count


def add_permitted_synthetic_hard_constraints(
    build_context: CpSatBuildContext,
    constraint_config: SyntheticConstraintConfig,
) -> SyntheticConstraintCounts:
    """Add only per-branch SKU and introduction caps from synthetic YAML.

    Budget, space, supply, and category constraints remain disabled here. A
    selected listed option counts as one SKU regardless of q1; an introduction
    is only the listing transition ``L0=False`` to ``L1=True``.
    """
    if not isinstance(build_context, CpSatBuildContext):
        raise VariableBuildError("Synthetic constraint construction requires a CpSatBuildContext.")
    if build_context.scenario.validation_result.status != "PASS":
        raise VariableBuildError("Synthetic constraint construction requires a validation PASS scenario input.")
    if not isinstance(constraint_config, SyntheticConstraintConfig):
        raise VariableBuildError("Synthetic constraint construction requires SyntheticConstraintConfig.")
    if (
        constraint_config.mode != "RESEARCH_SYNTHETIC"
        or constraint_config.parameter_source != SYNTHETIC_SOURCE
        or build_context.scenario.parameter_source != SYNTHETIC_SOURCE
    ):
        raise VariableBuildError("Synthetic constraints require RESEARCH_SYNTHETIC input marked SYNTHETIC_DEMO.")
    if constraint_config.budget_limit is not None or constraint_config.space_limit is not None:
        raise VariableBuildError("Budget and space constraints are outside the enabled synthetic MVP scope.")
    if constraint_config.category_minimums:
        raise VariableBuildError("Category constraints are not enabled for the current synthetic MVP configuration.")

    units_by_branch: dict[int, list[DecisionUnit]] = {}
    for unit in build_context.scenario.units:
        units_by_branch.setdefault(unit.branch_id, []).append(unit)

    sku_cap_constraints = 0
    introduction_cap_constraints = 0
    for branch_units in units_by_branch.values():
        listed_vars = tuple(
            build_context.option_vars[option.option_id]
            for unit in branch_units
            for option in unit.options
            if option.l1
        )
        if listed_vars:
            build_context.model.Add(sum(listed_vars) <= constraint_config.max_skus_per_branch)
            sku_cap_constraints += 1

        introduction_vars = tuple(
            build_context.option_vars[option.option_id]
            for unit in branch_units
            for option in unit.options
            if not option.l0 and option.l1
        )
        if introduction_vars:
            build_context.model.Add(sum(introduction_vars) <= constraint_config.max_introductions_per_branch)
            introduction_cap_constraints += 1

    return SyntheticConstraintCounts(
        sku_cap_constraints=sku_cap_constraints,
        introduction_cap_constraints=introduction_cap_constraints,
    )


def _proxy_utility_as_decimal(value: object, *, option_id: int) -> Decimal:
    """Parse one persisted Day-07 proxy utility without float arithmetic."""
    if isinstance(value, bool) or value is None:
        raise VariableBuildError(f"Cannot build proxy-utility objective: option_id={option_id} has missing or invalid utility.")
    try:
        utility = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise VariableBuildError(f"Cannot build proxy-utility objective: option_id={option_id} has non-numeric utility.") from exc
    if not utility.is_finite() or utility < 0 or utility > 1:
        raise VariableBuildError(f"Cannot build proxy-utility objective: option_id={option_id} utility must be finite and within the Day-07 0..1 contract.")
    return utility


def determine_utility_scale(utilities: tuple[Decimal, ...]) -> int:
    """Use the smallest power-of-ten scale preserving persisted decimal precision."""
    if not utilities:
        raise VariableBuildError("Cannot determine proxy-utility scale for an empty option set.")
    decimal_places = max(max(0, -utility.normalize().as_tuple().exponent) for utility in utilities)
    scale = 10**decimal_places
    max_int64 = (2**63) - 1
    if max(abs(utility) for utility in utilities) * scale * len(utilities) > max_int64:
        raise VariableBuildError("Proxy-utility coefficients exceed the safe CP-SAT integer range.")
    return scale


def scale_proxy_utility(utility: Decimal, *, scale: int) -> int:
    """Convert a Decimal proxy utility to an integer with ROUND_HALF_UP."""
    if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
        raise VariableBuildError("Proxy-utility scale must be a positive integer.")
    return int((utility * Decimal(scale)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def add_proxy_utility_objective(build_context: CpSatBuildContext) -> ProxyUtilityObjective:
    """Maximize the complete feasible plan's persisted Day-07 proxy utility.

    Utilities are read only from the validated scenario input. This function
    does not recalculate evidence or add economic, margin, revenue, or cost
    terms.
    """
    if not isinstance(build_context, CpSatBuildContext):
        raise VariableBuildError("Proxy-utility objective construction requires a CpSatBuildContext.")
    if build_context.scenario.validation_result.status != "PASS":
        raise VariableBuildError("Proxy-utility objective construction requires a validation PASS scenario input.")

    original_utilities: dict[int, Decimal] = {}
    for unit in build_context.scenario.units:
        for option in unit.options:
            option_id = option.option_id
            if option_id not in build_context.option_vars:
                raise VariableBuildError(f"Cannot build proxy-utility objective: option_id={option_id} has no BoolVar mapping.")
            if option_id in original_utilities:
                raise VariableBuildError(f"Cannot build proxy-utility objective: duplicate option_id={option_id}.")
            original_utilities[option_id] = _proxy_utility_as_decimal(option.utility, option_id=option_id)

    if len(original_utilities) != build_context.scenario.decision_option_count or len(original_utilities) != len(build_context.option_vars):
        raise VariableBuildError(
            "Proxy-utility objective requires exactly one utility, BoolVar, and objective term per validated DecisionOption."
        )
    utility_scale = determine_utility_scale(tuple(original_utilities.values()))
    scaled_coefficients = {
        option_id: scale_proxy_utility(utility, scale=utility_scale)
        for option_id, utility in original_utilities.items()
    }
    objective_terms = [
        scaled_coefficients[option_id] * build_context.option_vars[option_id]
        for option_id in original_utilities
    ]
    build_context.model.Maximize(sum(objective_terms))
    return ProxyUtilityObjective(
        objective_name="PROXY_UTILITY",
        utility_scale=utility_scale,
        rounding_mode="ROUND_HALF_UP",
        term_count=len(objective_terms),
        original_utilities=original_utilities,
        scaled_coefficients=scaled_coefficients,
    )
