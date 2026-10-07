"""Strict finite-option validation gate between the ORM loader and CP-SAT."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from math import isfinite
from numbers import Real
from typing import Iterable

from planning.models import DecisionOption


@dataclass(frozen=True)
class OptionValidationIssue:
    """One deterministic reason that blocks a scenario from reaching CP-SAT."""

    code: str
    reason: str
    branch_id: int | None = None
    branch_code: str | None = None
    product_id: int | None = None
    canonical_product_id: str | None = None
    option_ids: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class OptionValidationResult:
    """Structured pre-solver gate outcome for one persisted option scenario."""

    planning_month: date
    scenario_version: str
    decision_unit_count: int
    option_count: int
    errors: tuple[OptionValidationIssue, ...]

    @property
    def status(self) -> str:
        return "PASS" if not self.errors else "BLOCKED"

    @property
    def error_counts(self) -> dict[str, int]:
        return dict(Counter(issue.code for issue in self.errors))

    def summary(self, *, limit: int = 5) -> str:
        """Return a compact, reproducible error summary for the loader boundary."""
        if self.status == "PASS":
            return "Option validation PASS."
        grouped: dict[tuple[str, str], int] = Counter((issue.code, issue.reason) for issue in self.errors)
        shown = list(grouped.items())[:limit]
        details = "; ".join(f"{code}: {reason} ({count})" for ((code, reason), count) in shown)
        remaining = len(grouped) - len(shown)
        suffix = "" if not remaining else f"; (+{remaining} more issue types)"
        return f"Option validation BLOCKED for planning_month={self.planning_month:%Y-%m}; scenario_version={self.scenario_version}: {details}{suffix}"


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))


def _non_empty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_option_set(
    *,
    planning_month: date,
    scenario_version: str,
    option_rows: Iterable[DecisionOption],
) -> OptionValidationResult:
    """Return PASS only when every supplied finite option is solver-safe.

    This function reports every detected malformed option or decision unit. It
    never alters, omits, fills, deduplicates, or infers a planning value.
    """
    rows = tuple(option_rows)
    errors: list[OptionValidationIssue] = []
    if not rows:
        errors.append(OptionValidationIssue("OPT-VAL-001", "EMPTY_DECISION_SET"))
        return OptionValidationResult(planning_month, scenario_version, 0, 0, tuple(errors))

    unit_keys: set[tuple[int | None, int | None]] = set()
    option_ids: defaultdict[int | None, list[object]] = defaultdict(list)
    option_states: defaultdict[tuple[object, ...], list[int | None]] = defaultdict(list)
    reference_states: defaultdict[tuple[int | None, int | None], set[tuple[object, object]]] = defaultdict(set)
    identities: dict[tuple[int | None, int | None], tuple[str | None, str | None]] = {}

    expected_source = DecisionOption.ParameterSource.SYNTHETIC_DEMO
    for row in rows:
        option_id = getattr(row, "pk", None)
        branch_id = getattr(row, "branch_id", None)
        product_id = getattr(row, "product_id", None)
        branch = getattr(row, "branch", None)
        product = getattr(row, "product", None)
        branch_code = getattr(branch, "branch_code", None)
        canonical_product_id = getattr(product, "canonical_product_id", None)
        l0 = getattr(row, "l0", None)
        q0 = getattr(row, "q0", None)
        l1 = getattr(row, "l1", None)
        q1 = getattr(row, "q1", None)
        utility = getattr(row, "evidence_score", None)
        parameter_source = getattr(row, "parameter_source", None)

        unit_key = (branch_id, product_id)
        unit_keys.add(unit_key)
        identities[unit_key] = (branch_code, canonical_product_id)
        option_ids[option_id].append(row)
        option_states[(branch_id, product_id, l1, q1, parameter_source)].append(option_id)

        if branch_id is None or branch is None or not _non_empty(branch_code):
            errors.append(OptionValidationIssue("OPT-VAL-002", "MISSING_BRANCH", branch_id, branch_code, product_id, canonical_product_id, (option_id,)))
        if product_id is None or product is None or not _non_empty(canonical_product_id):
            errors.append(OptionValidationIssue("OPT-VAL-003", "MISSING_CANONICAL_PRODUCT", branch_id, branch_code, product_id, canonical_product_id, (option_id,)))
        if getattr(row, "planning_month", None) != planning_month or getattr(row, "scenario_version", None) != scenario_version:
            errors.append(OptionValidationIssue("OPT-VAL-012", "WRONG_SCENARIO", branch_id, branch_code, product_id, canonical_product_id, (option_id,)))

        if not isinstance(l0, bool) or not isinstance(l1, bool) or (l0 and (q0 is None or not _finite_number(q0) or q0 <= 0)) or (not l0 and q0 is not None):
            errors.append(OptionValidationIssue("OPT-VAL-004", "MISSING_DECISION_STATE", branch_id, branch_code, product_id, canonical_product_id, (option_id,)))
        if not _finite_number(q1) or not float(q1).is_integer() or (not l1 and q1 != 0) or (l1 and q1 <= 0):
            errors.append(OptionValidationIssue("OPT-VAL-005", "INVALID_Q1", branch_id, branch_code, product_id, canonical_product_id, (option_id,)))

        if parameter_source is None or not _non_empty(parameter_source):
            errors.append(OptionValidationIssue("OPT-VAL-006", "MISSING_PARAMETER_SOURCE", branch_id, branch_code, product_id, canonical_product_id, (option_id,)))
        elif parameter_source != expected_source:
            errors.append(OptionValidationIssue("OPT-VAL-007", "UNEXPECTED_PARAMETER_SOURCE", branch_id, branch_code, product_id, canonical_product_id, (option_id,)))

        if utility is None:
            errors.append(OptionValidationIssue("OPT-VAL-008", "MISSING_UTILITY", branch_id, branch_code, product_id, canonical_product_id, (option_id,)))
        elif not _finite_number(utility):
            errors.append(OptionValidationIssue("OPT-VAL-009", "NONFINITE_UTILITY", branch_id, branch_code, product_id, canonical_product_id, (option_id,)))

        if isinstance(l0, bool) and (q0 is None or _finite_number(q0)):
            reference_states[unit_key].add((l0, q0))

    for option_id, duplicate_rows in option_ids.items():
        if len(duplicate_rows) > 1:
            errors.append(OptionValidationIssue("OPT-VAL-010", "DUPLICATE_OPTION_ID", option_ids=(option_id,)))
    for (branch_id, product_id, _l1, _q1, _source), duplicate_ids in option_states.items():
        if len(duplicate_ids) > 1:
            branch_code, canonical_product_id = identities[(branch_id, product_id)]
            errors.append(OptionValidationIssue("OPT-VAL-010", "DUPLICATE_OPTION_STATE", branch_id, branch_code, product_id, canonical_product_id, tuple(duplicate_ids)))
    for unit_key, states in reference_states.items():
        if len(states) > 1:
            branch_id, product_id = unit_key
            branch_code, canonical_product_id = identities[unit_key]
            errors.append(OptionValidationIssue("OPT-VAL-011", "INCONSISTENT_REFERENCE_STATE", branch_id, branch_code, product_id, canonical_product_id))

    return OptionValidationResult(
        planning_month=planning_month,
        scenario_version=scenario_version,
        decision_unit_count=len(unit_keys),
        option_count=len(rows),
        errors=tuple(errors),
    )
