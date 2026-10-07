"""Finite Day-07 synthetic planning possibilities; no solver decisions."""

from __future__ import annotations

from math import isfinite

import pandas as pd

from planning.services.synthetic_state import SYNTHETIC_SOURCE


OPTION_COLUMNS = ["option_id", "branch_code", "canonical_product_id", "L0", "q0", "L1", "q1", "option_type", "option_sequence", "parameter_source", "mode"]
EVIDENCE_SCORE_COLUMNS = ["branch_code", "canonical_product_id", "evidence_score", "evidence_source"]
TRACK_B_CANDIDATE_SCORE = "TRACK_B_CANDIDATE_SCORE"
HISTORICAL_LAG_1_NORMALIZED = "HISTORICAL_LAG_1_NORMALIZED"


def generate_finite_options(fixture: pd.DataFrame, options: dict[str, object]) -> pd.DataFrame:
    """Generate bounded L1/q1 possibilities from explicit synthetic L0/q0 only.

    Listed products receive valid q0-step/q0/q0+step targets plus a delist
    possibility. Unlisted products receive remain-unlisted and list-trial
    possibilities. This function never reads inventory, forecasts, or scores.
    """
    required = {"branch_code", "canonical_product_id", "L0", "q0", "parameter_source"}
    if not required.issubset(fixture.columns):
        raise ValueError("Synthetic fixture is missing required fields.")
    step, maximum, mode = options.get("default_quantity_step"), options.get("max_target_per_product"), options.get("mode")
    if not isinstance(step, int) or step < 1 or not isinstance(maximum, int) or maximum < step or mode != "RESEARCH_SYNTHETIC":
        raise ValueError("Synthetic option configuration is invalid.")
    rows = []
    for state in fixture.loc[fixture["parameter_source"].eq(SYNTHETIC_SOURCE)].sort_values(["branch_code", "canonical_product_id"], kind="stable").itertuples(index=False):
        q0 = None if pd.isna(state.q0) else int(state.q0)
        possibilities: list[tuple[int, int, str]]
        if state.L0 == 1:
            if q0 is None or q0 <= 0:
                raise ValueError("Listed synthetic state requires positive q0.")
            targets = sorted({value for value in (q0 - step, q0, q0 + step) if 0 < value <= maximum})
            possibilities = [(1, value, "LISTED_TARGET_OPTION") for value in targets] + [(0, 0, "DELIST_OPTION")]
        elif state.L0 == 0:
            if q0 is not None:
                raise ValueError("Unlisted synthetic state requires null q0.")
            possibilities = [(0, 0, "REMAIN_UNLISTED_OPTION"), (1, step, "LIST_TRIAL_OPTION")]
        else:
            raise ValueError("L0 must be 0 or 1.")
        for sequence, (l1, q1, option_type) in enumerate(possibilities, start=1):
            rows.append({"option_id": f"{state.branch_code}|{state.canonical_product_id}|L1_{l1}|Q1_{q1}", "branch_code": state.branch_code, "canonical_product_id": state.canonical_product_id, "L0": int(state.L0), "q0": q0, "L1": l1, "q1": q1, "option_type": option_type, "option_sequence": sequence, "parameter_source": SYNTHETIC_SOURCE, "mode": mode})
    result = pd.DataFrame(rows, columns=OPTION_COLUMNS)
    validate_finite_options(result, options)
    return result


def validate_finite_options(options_frame: pd.DataFrame, options: dict[str, object]) -> None:
    """Validate finite menu bounds, provenance, uniqueness, and non-selection."""
    if options_frame.columns.tolist() != OPTION_COLUMNS or options_frame.duplicated(["branch_code", "canonical_product_id", "L1", "q1"]).any():
        raise ValueError("Option schema or branch/product/L1/q1 uniqueness is invalid.")
    step, maximum = options["default_quantity_step"], options["max_target_per_product"]
    if not options_frame["L1"].isin([0, 1]).all() or not options_frame["q1"].ge(0).all() or not options_frame["parameter_source"].eq(SYNTHETIC_SOURCE).all() or not options_frame["mode"].eq("RESEARCH_SYNTHETIC").all():
        raise ValueError("Option state or provenance is invalid.")
    if not options_frame.loc[options_frame.L1.eq(0), "q1"].eq(0).all() or not options_frame.loc[options_frame.L1.eq(1), "q1"].between(1, maximum).all() or not options_frame.loc[options_frame.q1.gt(0), "q1"].mod(step).eq(0).all():
        raise ValueError("Option target bounds or quantity step is invalid.")


def build_option_evidence_scores(fixture: pd.DataFrame, candidate_results: pd.DataFrame, feature_month: pd.DataFrame) -> pd.DataFrame:
    """Build complete, traceable non-financial utility evidence for the demo menu.

    Track-B candidates retain their published candidate score. Existing listed
    products receive a historical research proxy of ``lag_1 / max(lag_1)``
    within the explicit synthetic fixture scope at the frozen latest month.
    Missing evidence blocks generation rather than being zero-filled.
    """
    fixture_columns = {"branch_code", "canonical_product_id", "L0", "parameter_source"}
    candidate_columns = {"branch_code", "canonical_product_id", "candidate_score"}
    feature_columns = {"branch_code", "canonical_product_id", "year_month", "lag_1"}
    if not fixture_columns.issubset(fixture.columns) or not candidate_columns.issubset(candidate_results.columns) or not feature_columns.issubset(feature_month.columns):
        raise ValueError("Cannot build option evidence: required fixture, candidate, or frozen-feature columns are missing.")

    scope = fixture.loc[fixture["parameter_source"].eq(SYNTHETIC_SOURCE), ["branch_code", "canonical_product_id", "L0"]].copy()
    if scope.empty or scope.duplicated(["branch_code", "canonical_product_id"]).any() or not scope["L0"].isin([0, 1]).all():
        raise ValueError("Cannot build option evidence: synthetic fixture scope is invalid.")

    candidate_scope = scope.loc[scope["L0"].eq(0), ["branch_code", "canonical_product_id"]]
    candidate_scores = candidate_scope.merge(
        candidate_results.loc[:, ["branch_code", "canonical_product_id", "candidate_score"]],
        on=["branch_code", "canonical_product_id"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"candidate_score": "evidence_score"})
    candidate_scores["evidence_source"] = TRACK_B_CANDIDATE_SCORE

    listed_scope = scope.loc[scope["L0"].eq(1), ["branch_code", "canonical_product_id"]]
    source_month = feature_month["year_month"].astype(str).max()
    latest_features = feature_month.loc[
        feature_month["year_month"].astype(str).eq(source_month),
        ["branch_code", "canonical_product_id", "lag_1"],
    ].copy()
    if latest_features.duplicated(["branch_code", "canonical_product_id"]).any():
        raise ValueError("Cannot build option evidence: frozen feature grain is not unique at the latest month.")
    listed_scores = listed_scope.merge(
        latest_features,
        on=["branch_code", "canonical_product_id"],
        how="left",
        validate="one_to_one",
    )
    lag_1 = pd.to_numeric(listed_scores["lag_1"], errors="coerce")
    if lag_1.isna().any() or (~lag_1.map(lambda value: isfinite(float(value)) and value >= 0)).any():
        raise ValueError("Cannot build option evidence: listed synthetic products need finite non-negative latest lag_1 evidence.")
    maximum_lag_1 = float(lag_1.max())
    if maximum_lag_1 <= 0:
        raise ValueError("Cannot build option evidence: listed synthetic products have no positive latest lag_1 evidence.")
    listed_scores = listed_scores.drop(columns="lag_1")
    listed_scores["evidence_score"] = lag_1 / maximum_lag_1
    listed_scores["evidence_source"] = HISTORICAL_LAG_1_NORMALIZED

    evidence_scores = pd.concat([candidate_scores, listed_scores], ignore_index=True).loc[:, EVIDENCE_SCORE_COLUMNS]
    numeric_scores = pd.to_numeric(evidence_scores["evidence_score"], errors="coerce")
    if (
        len(evidence_scores) != len(scope)
        or evidence_scores.duplicated(["branch_code", "canonical_product_id"]).any()
        or numeric_scores.isna().any()
        or ~numeric_scores.between(0.0, 1.0).all()
        or evidence_scores["evidence_source"].isna().any()
    ):
        raise ValueError("Cannot build option evidence: complete finite evidence is required for every synthetic decision pair.")
    evidence_scores["evidence_score"] = numeric_scores.astype(float)
    return evidence_scores.sort_values(["branch_code", "canonical_product_id"], kind="stable").reset_index(drop=True)


def persist_decision_options(options_frame: pd.DataFrame, *, planning_month, scenario_version: str, evidence_scores: pd.DataFrame) -> pd.DataFrame:
    """Persist an existing finite menu idempotently; never select an option."""
    from catalog.models import Branch, Product
    from planning.models import DecisionOption

    if not scenario_version.strip():
        raise ValueError("scenario_version must be non-blank.")
    frame = options_frame.copy()
    frame["planning_month"] = planning_month
    frame["scenario_version"] = scenario_version
    if not set(EVIDENCE_SCORE_COLUMNS).issubset(evidence_scores.columns):
        raise ValueError("Decision-option persistence requires complete evidence_score and evidence_source columns.")
    frame = frame.merge(evidence_scores[EVIDENCE_SCORE_COLUMNS], on=["branch_code", "canonical_product_id"], how="left", validate="many_to_one")
    if frame["evidence_score"].isna().any() or frame["evidence_source"].isna().any():
        raise ValueError("Decision-option persistence refuses options without complete evidence.")
    branches = {code: pk for code, pk in Branch.objects.values_list("branch_code", "pk")}
    products = {code: pk for code, pk in Product.objects.values_list("canonical_product_id", "pk")}
    for row in frame.itertuples(index=False):
        DecisionOption.objects.update_or_create(
            branch_id=branches[row.branch_code], product_id=products[row.canonical_product_id], planning_month=planning_month,
            scenario_version=scenario_version, l1=bool(row.L1), q1=int(row.q1),
            defaults={"l0": bool(row.L0), "q0": None if pd.isna(row.q0) else int(row.q0), "option_type": row.option_type, "evidence_score": float(row.evidence_score), "evidence_source": row.evidence_source, "parameter_source": SYNTHETIC_SOURCE},
        )
    return frame
