"""CP-SAT execution and status persistence for the Day-08 research demo.

Model construction remains in :mod:`optimization.solver`; this module executes
that already-built model, extracts a usable selection, and persists only the
solver result.  It deliberately never runs the independent checker or creates
recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from time import perf_counter
from typing import Callable

from django.db import transaction
from django.utils import timezone
from ortools.sat.python import cp_model

from optimization.scenario_loader import load_scenario_for_optimization
from optimization.solver import (
    CpSatBuildContext,
    ProxyUtilityObjective,
    add_exactly_one_option_constraints,
    add_permitted_synthetic_hard_constraints,
    add_proxy_utility_objective,
    build_option_variables,
    load_synthetic_constraint_config,
)
from planning.models import DecisionOption, OptimizationResult


DEFAULT_MAX_TIME_SECONDS = 10.0
DEFAULT_NUM_SEARCH_WORKERS = 1
DEFAULT_RANDOM_SEED = 0
USABLE_SOLVER_STATUSES = frozenset(
    {OptimizationResult.SolverStatus.OPTIMAL, OptimizationResult.SolverStatus.FEASIBLE}
)


class SolverRunError(RuntimeError):
    """Raised when an existing published result would be overwritten."""


@dataclass(frozen=True)
class SolverExecution:
    """The safe application-level interpretation of one CP-SAT invocation."""

    solver_status: str
    raw_status: int | None
    selected_option_ids: tuple[int, ...]
    scaled_objective_value: int | None
    original_objective_value: Decimal | None
    wall_time_seconds: float | None
    error_message: str | None = None


@dataclass(frozen=True)
class SolverRunOutcome:
    """Persisted result plus the non-publishing execution details."""

    optimization_result: OptimizationResult
    execution: SolverExecution


def map_cp_sat_status(raw_status: int) -> str:
    """Map OR-Tools constants to the stable Day-08 solver-status contract."""
    if raw_status == cp_model.OPTIMAL:
        return OptimizationResult.SolverStatus.OPTIMAL
    if raw_status == cp_model.FEASIBLE:
        return OptimizationResult.SolverStatus.FEASIBLE
    if raw_status == cp_model.INFEASIBLE:
        return OptimizationResult.SolverStatus.INFEASIBLE
    return OptimizationResult.SolverStatus.UNKNOWN


def _extract_selected_option_ids(
    build_context: CpSatBuildContext,
    solver: cp_model.CpSolver,
) -> tuple[int, ...]:
    """Extract one selected persisted option per decision unit or fail loudly."""
    selected_option_ids: list[int] = []
    for unit in build_context.scenario.units:
        selected_for_unit = tuple(
            option.option_id
            for option in unit.options
            if solver.Value(build_context.option_vars[option.option_id]) == 1
        )
        if len(selected_for_unit) != 1:
            raise SolverRunError(
                "CP-SAT extraction invariant failed for "
                f"{unit.branch_code}/{unit.canonical_product_id}: "
                f"expected one selected option, found {len(selected_for_unit)}."
            )
        selected_option_ids.extend(selected_for_unit)
    return tuple(selected_option_ids)


def execute_cp_sat(
    build_context: CpSatBuildContext,
    objective: ProxyUtilityObjective,
    *,
    max_time_seconds: float = DEFAULT_MAX_TIME_SECONDS,
    num_search_workers: int = DEFAULT_NUM_SEARCH_WORKERS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    solver_factory: Callable[[], cp_model.CpSolver] = cp_model.CpSolver,
) -> SolverExecution:
    """Solve an already validated/built model without persisting it.

    Values are read only for ``OPTIMAL`` and ``FEASIBLE`` results.  An
    unexpected solver or extraction error remains distinct from INFEASIBLE and
    UNKNOWN as ``ERROR``.
    """
    if max_time_seconds <= 0:
        raise ValueError("max_time_seconds must be positive.")
    if num_search_workers <= 0:
        raise ValueError("num_search_workers must be positive.")
    if len(objective.scaled_coefficients) != len(build_context.option_vars):
        raise SolverRunError("Objective coefficient mapping does not match the CP-SAT option-variable mapping.")

    started_at = perf_counter()
    try:
        solver = solver_factory()
        solver.parameters.max_time_in_seconds = max_time_seconds
        solver.parameters.num_search_workers = num_search_workers
        solver.parameters.random_seed = random_seed
        raw_status = solver.Solve(build_context.model)
        raw_status_value = int(raw_status)
        wall_time_seconds = perf_counter() - started_at
        solver_status = map_cp_sat_status(raw_status)
        if solver_status not in USABLE_SOLVER_STATUSES:
            return SolverExecution(solver_status, raw_status_value, (), None, None, wall_time_seconds)

        selected_option_ids = _extract_selected_option_ids(build_context, solver)
        scaled_objective_value = sum(objective.scaled_coefficients[option_id] for option_id in selected_option_ids)
        original_objective_value = sum(
            (objective.original_utilities[option_id] for option_id in selected_option_ids),
            Decimal("0"),
        )
        return SolverExecution(
            solver_status,
            raw_status_value,
            selected_option_ids,
            scaled_objective_value,
            original_objective_value,
            wall_time_seconds,
        )
    except Exception as exc:  # Deliberately persists ERROR rather than claiming an OR-Tools status.
        return SolverExecution(
            OptimizationResult.SolverStatus.ERROR,
            None,
            (),
            None,
            None,
            perf_counter() - started_at,
            f"{type(exc).__name__}: {exc}",
        )


def _metadata(
    *,
    execution: SolverExecution,
    objective: ProxyUtilityObjective | None,
    max_time_seconds: float,
    num_search_workers: int,
    random_seed: int,
) -> dict[str, object]:
    """Return JSON-safe trace information, never financial terminology."""
    metadata: dict[str, object] = {
        "raw_cp_sat_status": int(execution.raw_status) if execution.raw_status is not None else None,
        "max_time_in_seconds": max_time_seconds,
        "num_search_workers": num_search_workers,
        "random_seed": random_seed,
        "solve_wall_time_seconds": execution.wall_time_seconds,
    }
    if execution.error_message:
        metadata["error_message"] = execution.error_message
    if objective is not None:
        metadata.update(
            {
                "objective_name": objective.objective_name,
                "utility_scale": objective.utility_scale,
                "rounding_mode": objective.rounding_mode,
                "objective_term_count": objective.term_count,
            }
        )
    if execution.scaled_objective_value is not None:
        metadata["scaled_proxy_utility"] = execution.scaled_objective_value
    return metadata


def persist_solver_execution(
    *,
    planning_month: date,
    scenario_version: str,
    run_version: str,
    execution: SolverExecution,
    objective: ProxyUtilityObjective | None,
    max_time_seconds: float,
    num_search_workers: int,
    random_seed: int,
) -> OptimizationResult:
    """Idempotently update an unpublished versioned solver result.

    A result that already has recommendations is considered downstream output
    and is never overwritten by a rerun of this Step-10 command.
    """
    if not run_version.strip():
        raise SolverRunError("run_version must be non-blank.")
    selected_option_ids = execution.selected_option_ids
    if execution.solver_status in USABLE_SOLVER_STATUSES:
        if not selected_option_ids:
            raise SolverRunError("A usable solver status requires selected DecisionOption ids.")
        if len(selected_option_ids) != len(set(selected_option_ids)):
            raise SolverRunError("Selected DecisionOption ids must be unique before persistence.")
    elif selected_option_ids:
        raise SolverRunError("INFEASIBLE, UNKNOWN, and ERROR solver results must not contain selected DecisionOptions.")

    with transaction.atomic():
        result, created = OptimizationResult.objects.select_for_update().get_or_create(
            planning_month=planning_month,
            scenario_version=scenario_version,
            run_version=run_version,
            defaults={"solver_status": OptimizationResult.SolverStatus.UNKNOWN},
        )
        if not created and result.recommendations.exists():
            raise SolverRunError(
                "Refusing to overwrite an OptimizationResult that already has recommendations; choose a new run_version."
            )

        result.solver_status = execution.solver_status
        result.objective_value = (
            float(execution.original_objective_value)
            if execution.solver_status in USABLE_SOLVER_STATUSES and execution.original_objective_value is not None
            else None
        )
        result.feasibility_status = OptimizationResult.FeasibilityStatus.NOT_RUN
        result.solver_metadata = _metadata(
            execution=execution,
            objective=objective,
            max_time_seconds=max_time_seconds,
            num_search_workers=num_search_workers,
            random_seed=random_seed,
        )
        result.completed_at = timezone.now()
        result.save()

        if execution.solver_status in USABLE_SOLVER_STATUSES:
            selected_options = list(
                DecisionOption.objects.filter(
                    pk__in=selected_option_ids,
                    planning_month=planning_month,
                    scenario_version=scenario_version,
                ).order_by("pk")
            )
            if len(selected_options) != len(selected_option_ids):
                raise SolverRunError(
                    "Selected DecisionOption records are missing or outside the optimization scenario/version."
                )
            selected_pairs = {(option.branch_id, option.product_id) for option in selected_options}
            expected_pairs = set(
                DecisionOption.objects.filter(
                    planning_month=planning_month,
                    scenario_version=scenario_version,
                )
                .order_by()
                .values_list("branch_id", "product_id")
                .distinct()
            )
            if len(selected_pairs) != len(selected_options) or selected_pairs != expected_pairs:
                raise SolverRunError(
                    "Selected DecisionOptions must contain exactly one option for every branch-product pair in the optimization scenario."
                )
            result.selected_options.set(selected_options)
        else:
            result.selected_options.clear()
    return result


def run_and_persist_optimization(
    *,
    planning_month: date,
    scenario_version: str,
    run_version: str,
    max_time_seconds: float = DEFAULT_MAX_TIME_SECONDS,
    num_search_workers: int = DEFAULT_NUM_SEARCH_WORKERS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SolverRunOutcome:
    """Run the complete pre-checker Day-08 model and persist its solver state."""
    objective: ProxyUtilityObjective | None = None
    try:
        scenario = load_scenario_for_optimization(
            planning_month=planning_month,
            scenario_version=scenario_version,
        )
        build_context = build_option_variables(scenario)
        add_exactly_one_option_constraints(build_context)
        constraint_config = load_synthetic_constraint_config()
        add_permitted_synthetic_hard_constraints(build_context, constraint_config)
        objective = add_proxy_utility_objective(build_context)
        execution = execute_cp_sat(
            build_context,
            objective,
            max_time_seconds=max_time_seconds,
            num_search_workers=num_search_workers,
            random_seed=random_seed,
        )
    except Exception as exc:
        execution = SolverExecution(
            OptimizationResult.SolverStatus.ERROR,
            None,
            (),
            None,
            None,
            None,
            f"{type(exc).__name__}: {exc}",
        )

    result = persist_solver_execution(
        planning_month=planning_month,
        scenario_version=scenario_version,
        run_version=run_version,
        execution=execution,
        objective=objective,
        max_time_seconds=max_time_seconds,
        num_search_workers=num_search_workers,
        random_seed=random_seed,
    )
    return SolverRunOutcome(result, execution)
