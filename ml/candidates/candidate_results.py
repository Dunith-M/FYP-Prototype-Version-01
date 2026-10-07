"""Final Day-07 research candidate-result dataset construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.candidates.generate_candidates import RESEARCH_SHORTLIST


PARAMETER_SOURCE = "HISTORICAL_EVIDENCE"
FINAL_COLUMNS = [
    "branch_code", "planning_month", "canonical_product_id", "product_desc", "family_id", "family_desc",
    "peer_product_demand_raw", "peer_product_demand_score", "branch_family_demand_raw",
    "branch_family_demand_score", "similar_branch_product_raw", "similar_branch_product_score",
    "price_fit_raw", "price_fit_score", "peer_component_available", "family_component_available",
    "similarity_component_available", "price_component_available", "effective_peer_weight",
    "effective_family_weight", "effective_similarity_weight", "effective_price_weight", "candidate_score",
    "candidate_rank", "evidence_class", "status", "parameter_source", "score_method_version", "mapping_status",
    "candidate_pool_size",
]


@dataclass(frozen=True)
class FinalCandidateResult:
    results: pd.DataFrame
    summary: pd.DataFrame
    track_a_overlap_count: int


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}.")


def build_final_candidate_results(scored_evaluations: pd.DataFrame, feature_month: pd.DataFrame, *, planning_month: str, source_month: str, top_n: int = 20) -> FinalCandidateResult:
    """Finalize scored Track-B rows and reject overlap with source Track-A rows.

    Track-A is derived from the frozen feature row at ``source_month``: a
    non-null ``lag_1`` is the existing usable-local-history condition. Track-B
    candidates retain ``INSUFFICIENT`` as their local confidence class because
    no external-evidence confidence taxonomy has been approved.
    """
    if not 1 <= top_n <= 20:
        raise ValueError("top_n must be between 1 and 20.")
    _require(scored_evaluations, {"branch_code", "canonical_product_id", "candidate_status", "candidate_score", "product_desc", "family_id", "family_desc", "peer_product_demand_raw", "peer_product_demand_score", "target_branch_family_demand_raw", "target_branch_family_demand_score", "similar_branch_product_raw", "similar_branch_product_score", "price_fit_raw", "price_fit_score", "peer_component_available", "family_component_available", "similarity_component_available", "price_component_available", "effective_peer_weight", "effective_family_weight", "effective_similarity_weight", "effective_price_weight", "score_method_version", "mapping_status"}, "Scored evaluations")
    _require(feature_month, {"branch_code", "canonical_product_id", "year_month", "lag_1"}, "Feature month")
    pool = scored_evaluations.loc[scored_evaluations["candidate_status"].eq(RESEARCH_SHORTLIST)].copy()
    if pool.duplicated(["branch_code", "canonical_product_id"]).any():
        raise ValueError("Scored candidate pool has duplicate branch/product keys.")
    pool_sizes = pool.groupby("branch_code").size()
    ranked = pool.sort_values(["branch_code", "candidate_score", "canonical_product_id"], ascending=[True, False, True], kind="stable")
    final = ranked.groupby("branch_code", sort=False, group_keys=False).head(top_n).copy()
    final["candidate_rank"] = final.groupby("branch_code", sort=False).cumcount().add(1).astype("int64")
    track_a = feature_month.loc[feature_month["year_month"].astype(str).eq(source_month) & feature_month["lag_1"].notna(), ["branch_code", "canonical_product_id"]].drop_duplicates()
    overlap = final.merge(track_a, on=["branch_code", "canonical_product_id"], how="inner")
    if not overlap.empty:
        raise ValueError(f"Track-A overlap detected for final Track-B candidates: {len(overlap)} rows.")
    final["planning_month"] = planning_month
    final["branch_family_demand_raw"] = final.pop("target_branch_family_demand_raw")
    final["branch_family_demand_score"] = final.pop("target_branch_family_demand_score")
    final["evidence_class"] = "INSUFFICIENT"
    final["status"] = RESEARCH_SHORTLIST
    final["parameter_source"] = PARAMETER_SOURCE
    final["candidate_pool_size"] = final["branch_code"].map(pool_sizes).astype("int64")
    final = final.loc[:, FINAL_COLUMNS].copy()
    validate_final_candidate_results(final, track_a)
    counts = final.groupby("branch_code").agg(shortlisted_count=("canonical_product_id", "size"), adequate_count=("evidence_class", lambda value: int(value.eq("ADEQUATE").sum())), limited_count=("evidence_class", lambda value: int(value.eq("LIMITED").sum())), insufficient_count=("evidence_class", lambda value: int(value.eq("INSUFFICIENT").sum())), top_candidate_score=("candidate_score", "max")).reset_index()
    summary = counts.merge(pool_sizes.rename("candidate_pool_count"), on="branch_code", how="left")
    summary["scored_candidate_count"] = summary["candidate_pool_count"]
    summary["planning_month"] = planning_month
    summary["track_a_overlap_count"] = 0
    summary = summary[["planning_month", "branch_code", "candidate_pool_count", "scored_candidate_count", "shortlisted_count", "adequate_count", "limited_count", "insufficient_count", "top_candidate_score", "track_a_overlap_count"]].sort_values("branch_code", kind="stable")
    return FinalCandidateResult(final.sort_values(["branch_code", "candidate_rank"], kind="stable").reset_index(drop=True), summary.reset_index(drop=True), 0)


def validate_final_candidate_results(results: pd.DataFrame, track_a: pd.DataFrame | None = None) -> None:
    """Validate final Day-07 grain, bounds, labels, ranks, and anti-join."""
    _require(results, set(FINAL_COLUMNS), "Final candidate results")
    key = ["branch_code", "canonical_product_id", "planning_month"]
    if results.duplicated(key).any() or results[key].isna().any().any():
        raise ValueError("Final candidate keys must be non-null and unique.")
    if not results["candidate_score"].between(0, 1).all() or not results["candidate_rank"].between(1, 20).all():
        raise ValueError("Final candidate score or rank is outside its valid range.")
    if not results["status"].eq(RESEARCH_SHORTLIST).all() or not results["parameter_source"].eq(PARAMETER_SOURCE).all():
        raise ValueError("Final candidate labels are invalid.")
    if results.astype("string").apply(lambda column: column.str.contains("INTRODUCE", case=False, na=False).any()).any():
        raise ValueError("Final candidate results contain forbidden action text.")
    if results.duplicated(["branch_code", "planning_month", "candidate_rank"]).any() or results.groupby(["branch_code", "planning_month"]).size().gt(20).any():
        raise ValueError("Final candidate ranks are duplicate or exceed the shortlist limit.")
    if track_a is not None and not results.merge(track_a, on=["branch_code", "canonical_product_id"], how="inner").empty:
        raise ValueError("Final candidates overlap Track-A evidence.")
