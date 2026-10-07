"""Deterministic identity for one persisted finite Day-08 selected plan."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Iterable, Mapping


def constraint_signature_from_values(
    *,
    mode: object,
    parameter_source: object,
    max_skus_per_branch: object,
    max_introductions_per_branch: object,
    category_minimums: object,
    budget_limit: object,
    space_limit: object,
) -> dict[str, object]:
    """Return the small configured-constraint identity used by this MVP."""
    return {
        "mode": mode,
        "parameter_source": parameter_source,
        "max_skus_per_branch": max_skus_per_branch,
        "max_introductions_per_branch": max_introductions_per_branch,
        "category_minimums": category_minimums,
        "budget_limit": budget_limit,
        "space_limit": space_limit,
    }


def synthetic_constraint_signature(options: Mapping[str, object]) -> dict[str, object]:
    """Extract a canonical signature from the synthetic YAML configuration."""
    constraints = options.get("constraints")
    if not isinstance(constraints, Mapping):
        raise ValueError("Synthetic options must contain a constraints mapping.")
    return constraint_signature_from_values(
        mode=options.get("mode"),
        parameter_source=options.get("parameter_source"),
        max_skus_per_branch=constraints.get("max_skus_per_branch"),
        max_introductions_per_branch=constraints.get("max_introductions_per_branch"),
        category_minimums=constraints.get("category_minimums"),
        budget_limit=constraints.get("budget_limit"),
        space_limit=constraints.get("space_limit"),
    )


def build_selected_plan_fingerprint(
    optimization_result: object,
    selected_options: Iterable[object],
    *,
    constraint_signature: Mapping[str, object],
) -> str:
    """Hash exact selected option IDs/states, lineage, and active constraints."""
    selected_state = sorted(
        (
            {
                "decision_option_id": getattr(option, "pk", None),
                "branch_id": getattr(option, "branch_id", None),
                "product_id": getattr(option, "product_id", None),
                "l0": getattr(option, "l0", None),
                "q0": getattr(option, "q0", None),
                "l1": getattr(option, "l1", None),
                "q1": getattr(option, "q1", None),
                "parameter_source": getattr(option, "parameter_source", None),
            }
            for option in selected_options
        ),
        key=lambda item: (item["decision_option_id"] is None, item["decision_option_id"]),
    )
    planning_month = getattr(optimization_result, "planning_month", None)
    payload = {
        "planning_month": planning_month.isoformat() if planning_month is not None else None,
        "scenario_version": getattr(optimization_result, "scenario_version", None),
        "run_version": getattr(optimization_result, "run_version", None),
        "selected_options": selected_state,
        "constraints": dict(constraint_signature),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(encoded).hexdigest()
