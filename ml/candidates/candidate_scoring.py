"""Transparent Day-07 Track-B research-proxy candidate scoring.

This module contains only deterministic scoring mathematics. It neither uses
Django nor creates actions. A candidate score is a research proxy score, not
a probability, profit estimate, listing decision, or recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.candidates.generate_candidates import RESEARCH_SHORTLIST


SCORE_METHOD_VERSION = "day07_candidate_score_v1"
BASE_WEIGHTS = {"peer": 0.40, "family": 0.30, "similarity": 0.20, "price": 0.10}
SIMILARITY_NOT_AVAILABLE_REASON = "NO_DEFENSIBLE_APPROVED_BRANCH_SIMILARITY_INPUT"


@dataclass(frozen=True)
class CandidateScoringResult:
    """Scored full evaluations, the Top-N shortlist, and branch summary."""

    evaluations: pd.DataFrame
    candidate_shortlist: pd.DataFrame
    branch_summary: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}.")


def _as_bool(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().upper() in {"TRUE", "T", "YES", "Y", "1"}
    return bool(value)


def _normalise_by_branch(raw: pd.Series, branches: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """Return branch-relative 0-1 min-max scores for available raw values.

    Identical non-missing raw values receive 0.5. This neutral deterministic
    treatment avoids divide-by-zero and does not create an artificial winner.
    """
    output = pd.Series(float("nan"), index=raw.index, dtype="float64")
    for _, indices in branches.groupby(branches, sort=False).groups.items():
        values = pd.to_numeric(raw.loc[indices], errors="coerce").dropna()
        if values.empty:
            continue
        minimum, maximum = float(values.min()), float(values.max())
        if minimum == maximum:
            output.loc[values.index] = 0.5
        elif higher_is_better:
            output.loc[values.index] = (values - minimum) / (maximum - minimum)
        else:
            output.loc[values.index] = (maximum - values) / (maximum - minimum)
    return output


def _positive_observations(feature_month: pd.DataFrame) -> pd.DataFrame:
    _require_columns(feature_month, {"branch_code", "canonical_product_id", "family_id", "fulfilled_positive_units_proxy"}, "Feature month")
    observed = feature_month[["branch_code", "canonical_product_id", "family_id", "fulfilled_positive_units_proxy"]].copy()
    observed["observed_positive_units"] = pd.to_numeric(observed["fulfilled_positive_units_proxy"], errors="coerce").fillna(0.0)
    return observed.loc[observed["observed_positive_units"].gt(0)].copy()


def _peer_and_family_raw(candidates: pd.DataFrame, feature_month: pd.DataFrame) -> pd.DataFrame:
    """Derive peer/family demand from frozen observations, excluding target peers.

    Peer raw is exact-product positive units in all branches except the target.
    Family raw is positive units for the candidate family in the target branch.
    Missing local history is never turned into imputed demand.
    """
    observed = _positive_observations(feature_month)
    product_all = observed.groupby("canonical_product_id", as_index=False)["observed_positive_units"].sum().rename(columns={"observed_positive_units": "_product_all_units"})
    product_target = observed.groupby(["branch_code", "canonical_product_id"], as_index=False)["observed_positive_units"].sum().rename(columns={"observed_positive_units": "_product_target_units"})
    family_target = observed.loc[observed["family_id"].notna() & observed["family_id"].astype("string").str.strip().ne("")]
    family_target = family_target.groupby(["branch_code", "family_id"], as_index=False)["observed_positive_units"].sum().rename(columns={"observed_positive_units": "target_branch_family_demand_raw"})
    output = candidates.merge(product_all, on="canonical_product_id", how="left", validate="many_to_one")
    output = output.merge(product_target, on=["branch_code", "canonical_product_id"], how="left", validate="one_to_one")
    output = output.merge(family_target, on=["branch_code", "family_id"], how="left", validate="many_to_one")
    output["peer_product_demand_raw"] = (output["_product_all_units"].fillna(0.0) - output["_product_target_units"].fillna(0.0)).clip(lower=0.0)
    output["target_branch_family_demand_raw"] = output["target_branch_family_demand_raw"].fillna(0.0)
    return output.drop(columns=["_product_all_units", "_product_target_units"])


def _price_fit_raw(candidates: pd.DataFrame, feature_month: pd.DataFrame, product_prices: pd.DataFrame) -> pd.DataFrame:
    """Distance from each branch's observed-product valid-price median.

    The candidate product itself is removed from that branch's price reference,
    so sparse target-branch candidate history cannot influence price evidence.
    """
    _require_columns(product_prices, {"canonical_product_id", "list_price", "price_valid"}, "Product prices")
    prices = product_prices[["canonical_product_id", "list_price", "price_valid"]].copy()
    prices["list_price"] = pd.to_numeric(prices["list_price"], errors="coerce")
    prices["price_valid"] = prices["price_valid"].map(_as_bool)
    prices = prices.loc[prices["price_valid"] & prices["list_price"].gt(0)].copy()
    observed = _positive_observations(feature_month)[["branch_code", "canonical_product_id"]].drop_duplicates()
    branch_prices = observed.merge(prices[["canonical_product_id", "list_price"]], on="canonical_product_id", how="inner")
    by_branch = {branch: group[["canonical_product_id", "list_price"]] for branch, group in branch_prices.groupby("branch_code", sort=False)}
    output = candidates.merge(prices[["canonical_product_id", "list_price"]], on="canonical_product_id", how="left", validate="many_to_one")
    raw_values: list[float | None] = []
    for row in output.itertuples(index=False):
        if pd.isna(row.list_price) or float(row.list_price) <= 0:
            raw_values.append(None)
            continue
        reference = by_branch.get(row.branch_code)
        if reference is None:
            raw_values.append(None)
            continue
        reference = reference.loc[reference["canonical_product_id"].ne(row.canonical_product_id), "list_price"]
        if reference.empty:
            raw_values.append(None)
            continue
        median_price = float(reference.median())
        raw_values.append(abs(float(row.list_price) - median_price) / median_price if median_price > 0 else None)
    output["price_fit_raw"] = pd.Series(raw_values, index=output.index, dtype="float64")
    return output


def _attach_similarity(candidates: pd.DataFrame, similarity_evidence: pd.DataFrame | None) -> pd.DataFrame:
    """Attach explicitly supplied similarity evidence, or preserve its absence."""
    output = candidates.copy()
    if similarity_evidence is None:
        output["similar_branch_product_raw"] = float("nan")
        output["similarity_component_available"] = False
        output["similarity_reason"] = SIMILARITY_NOT_AVAILABLE_REASON
        return output
    _require_columns(similarity_evidence, {"branch_code", "canonical_product_id", "similar_branch_product_raw"}, "Similarity evidence")
    if similarity_evidence.duplicated(["branch_code", "canonical_product_id"]).any():
        raise ValueError("Similarity evidence has duplicate branch/product keys.")
    output = output.merge(similarity_evidence[["branch_code", "canonical_product_id", "similar_branch_product_raw"]], on=["branch_code", "canonical_product_id"], how="left", validate="one_to_one")
    output["similar_branch_product_raw"] = pd.to_numeric(output["similar_branch_product_raw"], errors="coerce")
    output["similarity_component_available"] = output["similar_branch_product_raw"].notna()
    output["similarity_reason"] = output["similarity_component_available"].map(lambda available: "SUPPLIED_DEFENSIBLE_SIMILARITY_EVIDENCE" if available else SIMILARITY_NOT_AVAILABLE_REASON)
    return output


def _apply_weights_and_score(candidates: pd.DataFrame) -> pd.DataFrame:
    output = candidates.copy()
    output["peer_component_available"] = output["peer_product_demand_raw"].gt(0)
    output["family_component_available"] = output["target_branch_family_demand_raw"].gt(0)
    output["price_component_available"] = output["price_fit_raw"].notna()
    output["peer_product_demand_score"] = _normalise_by_branch(output["peer_product_demand_raw"].where(output["peer_component_available"]), output["branch_code"], higher_is_better=True)
    output["target_branch_family_demand_score"] = _normalise_by_branch(output["target_branch_family_demand_raw"].where(output["family_component_available"]), output["branch_code"], higher_is_better=True)
    output["similar_branch_product_score"] = _normalise_by_branch(output["similar_branch_product_raw"].where(output["similarity_component_available"]), output["branch_code"], higher_is_better=True)
    output["price_fit_score"] = _normalise_by_branch(output["price_fit_raw"].where(output["price_component_available"]), output["branch_code"], higher_is_better=False)
    components = {"peer": "peer_component_available", "family": "family_component_available", "similarity": "similarity_component_available", "price": "price_component_available"}
    for component in components:
        output[f"effective_{component}_weight"] = 0.0
    for index, row in output.iterrows():
        available = [component for component, column in components.items() if bool(row[column])]
        total_weight = sum(BASE_WEIGHTS[component] for component in available)
        if total_weight:
            for component in available:
                output.at[index, f"effective_{component}_weight"] = BASE_WEIGHTS[component] / total_weight
    output["effective_weight_sum"] = output[[f"effective_{component}_weight" for component in components]].sum(axis=1)
    output["weight_renormalization_reason"] = output.apply(lambda row: "NONE" if bool(row["similarity_component_available"]) and bool(row["price_component_available"]) else "|".join(reason for component, reason in (("similarity", "SIMILARITY_UNAVAILABLE"), ("price", "PRICE_UNAVAILABLE")) if not bool(row[f"{component}_component_available"])), axis=1)
    output["candidate_score"] = (output["effective_peer_weight"] * output["peer_product_demand_score"].fillna(0.0) + output["effective_family_weight"] * output["target_branch_family_demand_score"].fillna(0.0) + output["effective_similarity_weight"] * output["similar_branch_product_score"].fillna(0.0) + output["effective_price_weight"] * output["price_fit_score"].fillna(0.0)).where(output["effective_weight_sum"].gt(0))
    output["score_method_version"] = SCORE_METHOD_VERSION
    return output


def score_candidate_evaluations(evaluations: pd.DataFrame, feature_month: pd.DataFrame, product_prices: pd.DataFrame, *, top_n: int = 20, similarity_evidence: pd.DataFrame | None = None) -> CandidateScoringResult:
    """Score existing Track-B candidates and return a deterministic Top-N.

    The frozen feature artifact defines the historical cutoff. Similarity is
    included only when defensible evidence is explicitly supplied by the caller.
    """
    if not 10 <= top_n <= 20:
        raise ValueError("top_n must be between 10 and 20.")
    _require_columns(evaluations, {"branch_code", "canonical_product_id", "product_desc", "family_id", "family_desc", "candidate_status"}, "Candidate evaluations")
    if evaluations.duplicated(["branch_code", "canonical_product_id"]).any():
        raise ValueError("Candidate evaluations have duplicate branch/product keys.")
    pool = evaluations.loc[evaluations["candidate_status"].eq(RESEARCH_SHORTLIST)].copy()
    scored = _apply_weights_and_score(_attach_similarity(_price_fit_raw(_peer_and_family_raw(pool, feature_month), feature_month, product_prices), similarity_evidence))
    if scored["candidate_score"].isna().any():
        raise ValueError("A Track-B candidate has no available scoring component.")
    if not scored["candidate_score"].between(0.0, 1.0).all():
        raise ValueError("candidate_score must stay within 0 and 1.")
    if not scored["effective_weight_sum"].between(0.999999, 1.000001).all():
        raise ValueError("Effective component weights must sum to 1 for every scored candidate.")
    ordered = scored.sort_values(["branch_code", "candidate_score", "canonical_product_id"], ascending=[True, False, True], kind="stable")
    shortlist = ordered.groupby("branch_code", sort=False, group_keys=False).head(top_n).copy()
    shortlist["rank"] = shortlist.groupby("branch_code", sort=False).cumcount().add(1).astype("Int64")
    shortlist["in_shortlist"] = True
    output = evaluations.copy()
    score_columns = [column for column in scored.columns if column not in output.columns]
    output = output.merge(scored[["branch_code", "canonical_product_id", *score_columns]], on=["branch_code", "canonical_product_id"], how="left", validate="one_to_one")
    output["rank"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["in_shortlist"] = False
    lookup = shortlist.set_index(["branch_code", "canonical_product_id"])[["rank", "in_shortlist"]]
    keys = pd.MultiIndex.from_frame(output[["branch_code", "canonical_product_id"]])
    output["rank"] = lookup["rank"].reindex(keys).array
    output["in_shortlist"] = lookup["in_shortlist"].reindex(keys).fillna(False).to_numpy(dtype=bool)
    summary = output.loc[output["candidate_status"].eq(RESEARCH_SHORTLIST)].groupby("branch_code", as_index=False).agg(candidate_pool_count=("canonical_product_id", "size"), scored_candidate_count=("candidate_score", "count"), shortlisted_count=("in_shortlist", "sum"), similarity_component_used=("similarity_component_available", "any"), scoring_method_version=("score_method_version", "first")).sort_values("branch_code", kind="stable").reset_index(drop=True)
    return CandidateScoringResult(output.sort_values(["branch_code", "canonical_product_id"], kind="stable").reset_index(drop=True), shortlist.sort_values(["branch_code", "rank"], kind="stable").reset_index(drop=True), summary)
