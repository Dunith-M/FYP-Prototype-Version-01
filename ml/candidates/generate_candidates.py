"""Classify Day-07 research candidates without scoring or forecasting.

A Track-B candidate lacks usable local branch history but may be evaluated from
external research evidence.  It is a research shortlist entry, never an
operational Introduce recommendation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from ml.evaluation.confidence import classify_evidence


TRACK_A = "TRACK_A"
TRACK_B = "TRACK_B"
EXCLUDED = "EXCLUDED"
RESEARCH_SHORTLIST = "RESEARCH_SHORTLIST"
DEFAULT_TOP_N = 20
MIN_TOP_N = 10

UNRESOLVED_MAPPING = "UNRESOLVED_MAPPING"
MISSING_PRODUCT_IDENTITY = "MISSING_PRODUCT_IDENTITY"
NO_EXTERNAL_EVIDENCE = "NO_EXTERNAL_EVIDENCE"
INELIGIBLE = "INELIGIBLE"


@dataclass(frozen=True)
class CandidateClassification:
    """Auditable Track-A, Track-B, or exclusion result for one branch/product."""

    branch_code: str | None
    canonical_product_id: str | None
    track: str
    candidate_status: str | None
    local_usable_history: bool
    mapping_status: str
    peer_evidence_available: bool
    family_evidence_available: bool
    similar_branch_evidence_available: bool
    price_evidence_available: bool
    eligibility_status: str
    exclusion_reason: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a stable, tabular record suitable for a future artifact."""
        return asdict(self)


@dataclass(frozen=True)
class CandidatePoolBuildResult:
    """Traceable branch/product evaluations, candidate pool, shortlist, and totals."""

    evaluations: pd.DataFrame
    candidate_pool: pd.DataFrame
    candidate_shortlist: pd.DataFrame
    branch_summary: pd.DataFrame


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _as_bool(value: object) -> bool:
    """Accept existing boolean-like values; unknown/missing stays unavailable."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().upper() in {"TRUE", "T", "YES", "Y", "1"}
    return bool(value)


def _mapping_status(row: Mapping[str, object]) -> str:
    value = row.get("mapping_status", row.get("confidence_class"))
    return "UNKNOWN" if _is_blank(value) else str(value).strip().upper()


def _has_valid_product_identity(row: Mapping[str, object], mapping_status: str) -> bool:
    return (
        not _is_blank(row.get("canonical_product_id"))
        and not _as_bool(row.get("unresolved_product_flag"))
        and mapping_status != "UNRESOLVED"
    )


def _eligibility_status(row: Mapping[str, object]) -> str:
    """Use eligibility only when a trustworthy upstream field is supplied."""
    if "eligibility_status" in row and not _is_blank(row.get("eligibility_status")):
        return str(row["eligibility_status"]).strip().upper()
    if "eligible" in row and not _is_blank(row.get("eligible")):
        return "ELIGIBLE" if _as_bool(row["eligible"]) else INELIGIBLE
    return "NOT_AVAILABLE"


def _local_usable_history(row: Mapping[str, object]) -> bool:
    """Reuse Day-06 evidence semantics without invoking a forecasting path."""
    evidence_class = str(row.get("evidence_class", "")).strip().upper()
    reason_codes = str(row.get("evidence_reason_code", "")).upper().split("|")
    if evidence_class:
        return evidence_class != "INSUFFICIENT" or "NO_USABLE_HISTORY" not in reason_codes
    return "NO_USABLE_HISTORY" not in classify_evidence(row).reason_codes


def classify_track_b_candidate(row: Mapping[str, object]) -> CandidateClassification:
    """Classify one branch/product under the Day-07 Track-B definition.

    Track B requires no usable local history, a resolved canonical identity,
    and at least one independently supplied external-evidence flag.  A valid
    price may count as price evidence; family evidence cannot count without a
    family identity. No score, forecast, action, or recommendation is made.
    """
    mapping_status = _mapping_status(row)
    peer_evidence = _as_bool(row.get("peer_evidence_available"))
    family_evidence = _as_bool(row.get("family_evidence_available")) and not _is_blank(row.get("family_id"))
    similar_branch_evidence = _as_bool(row.get("similar_branch_evidence_available"))
    price_evidence = _as_bool(row.get("price_evidence_available", row.get("valid_price")))
    local_usable_history = _local_usable_history(row)
    eligibility_status = _eligibility_status(row)
    common = dict(
        branch_code=None if _is_blank(row.get("branch_code")) else str(row.get("branch_code")).strip(),
        canonical_product_id=(
            None if _is_blank(row.get("canonical_product_id")) else str(row.get("canonical_product_id")).strip()
        ),
        local_usable_history=local_usable_history,
        mapping_status=mapping_status,
        peer_evidence_available=peer_evidence,
        family_evidence_available=family_evidence,
        similar_branch_evidence_available=similar_branch_evidence,
        price_evidence_available=price_evidence,
        eligibility_status=eligibility_status,
    )

    if _is_blank(row.get("canonical_product_id")):
        return CandidateClassification(track=EXCLUDED, candidate_status=None, exclusion_reason=MISSING_PRODUCT_IDENTITY, **common)
    if not _has_valid_product_identity(row, mapping_status):
        return CandidateClassification(track=EXCLUDED, candidate_status=None, exclusion_reason=UNRESOLVED_MAPPING, **common)
    if local_usable_history:
        return CandidateClassification(track=TRACK_A, candidate_status=None, exclusion_reason=None, **common)
    if eligibility_status == INELIGIBLE:
        return CandidateClassification(track=EXCLUDED, candidate_status=None, exclusion_reason=INELIGIBLE, **common)
    if not any((peer_evidence, family_evidence, similar_branch_evidence, price_evidence)):
        return CandidateClassification(track=EXCLUDED, candidate_status=None, exclusion_reason=NO_EXTERNAL_EVIDENCE, **common)
    return CandidateClassification(
        track=TRACK_B,
        candidate_status=RESEARCH_SHORTLIST,
        exclusion_reason=None,
        **common,
    )


def classify_track_b_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Classify supplied branch/product evidence rows without changing inputs.

    This deliberately does not load artifacts, calculate candidate scores, or
    call the normal Track-A forecast pipeline.
    """
    required = {"branch_code", "canonical_product_id"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate input is missing required columns: {missing}.")
    records = [classify_track_b_candidate(row).to_dict() for row in candidates.to_dict("records")]
    return pd.DataFrame.from_records(records)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}.")


def _normalise_identifier(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip()


def _validate_top_n(top_n: int) -> None:
    if not MIN_TOP_N <= top_n <= DEFAULT_TOP_N:
        raise ValueError(f"top_n must be between {MIN_TOP_N} and {DEFAULT_TOP_N}.")


def _latest_local_features(feature_month: pd.DataFrame) -> pd.DataFrame:
    """Select the latest saved Day-04 feature row per branch/product only."""
    _require_columns(feature_month, {"branch_code", "canonical_product_id", "year_month", "lag_1"}, "Feature month")
    work = feature_month.copy(deep=True)
    work["canonical_product_id"] = _normalise_identifier(work["canonical_product_id"])
    work = work.loc[work["canonical_product_id"].notna() & work["canonical_product_id"].ne("")].copy()
    return (
        work.sort_values(["branch_code", "canonical_product_id", "year_month"], kind="stable")
        .drop_duplicates(["branch_code", "canonical_product_id"], keep="last")
    )


def _positive_evidence(feature_month: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Use observed positive-sales proxy only; absent sales are never imputed."""
    _require_columns(
        feature_month,
        {"branch_code", "canonical_product_id", "fulfilled_positive_units_proxy"},
        "Feature month",
    )
    observed = feature_month[["branch_code", "canonical_product_id", "fulfilled_positive_units_proxy"]].copy()
    observed["canonical_product_id"] = _normalise_identifier(observed["canonical_product_id"])
    observed["observed_positive_units"] = pd.to_numeric(
        observed["fulfilled_positive_units_proxy"], errors="coerce"
    ).fillna(0.0)
    observed = observed.loc[observed["observed_positive_units"].gt(0)].copy()
    families = products[["canonical_product_id", "family_id"]].drop_duplicates("canonical_product_id")
    return observed.merge(families, on="canonical_product_id", how="left", validate="many_to_one")


def build_candidate_pools(
    products: pd.DataFrame,
    branches: pd.DataFrame,
    confirmed_mappings: pd.DataFrame,
    feature_month: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> CandidatePoolBuildResult:
    """Build all Track-B evaluations plus a deterministic per-branch shortlist.

    The candidate pool starts with the supplied canonical product master for
    every supplied branch. A product enters it only when the existing Track-B
    classifier accepts it. The shortlist is an evidence ordering, not a
    weighted score, recommendation, listing decision, or planning action.
    """
    _validate_top_n(top_n)
    _require_columns(
        products,
        {"canonical_product_id", "product_desc", "family_id", "family_desc", "price_valid"},
        "Product master",
    )
    _require_columns(branches, {"branch_code"}, "Branch master")
    _require_columns(confirmed_mappings, {"canonical_product_id"}, "Confirmed mappings")

    product_rows = products.copy(deep=True)
    product_rows["canonical_product_id"] = _normalise_identifier(product_rows["canonical_product_id"])
    product_rows["product_desc"] = product_rows["product_desc"].astype("string").str.strip()
    product_rows["family_id"] = _normalise_identifier(product_rows["family_id"])
    product_rows["family_desc"] = product_rows["family_desc"].astype("string").str.strip()
    product_rows["product_record_usable"] = product_rows["product_desc"].notna() & product_rows["product_desc"].ne("")
    product_rows["valid_price"] = product_rows["price_valid"].map(_as_bool)
    if product_rows.loc[product_rows["canonical_product_id"].notna() & product_rows["canonical_product_id"].ne("")].duplicated(
        "canonical_product_id"
    ).any():
        raise ValueError("Product master has duplicate canonical_product_id values.")

    branch_rows = branches[["branch_code"]].copy()
    branch_rows["branch_code"] = _normalise_identifier(branch_rows["branch_code"])
    if branch_rows["branch_code"].isna().any() or branch_rows["branch_code"].eq("").any():
        raise ValueError("Branch master contains blank branch_code values.")
    if branch_rows.duplicated("branch_code").any():
        raise ValueError("Branch master has duplicate branch_code values.")

    confirmed_ids = set(_normalise_identifier(confirmed_mappings["canonical_product_id"]).dropna())
    product_rows["mapping_status"] = product_rows["canonical_product_id"].map(
        lambda value: "CONFIRMED" if value in confirmed_ids else "UNRESOLVED"
    )
    product_rows["_join_key"] = 1
    branch_rows["_join_key"] = 1
    base = branch_rows.merge(product_rows, on="_join_key", how="inner", validate="many_to_many").drop(columns="_join_key")

    latest_local = _latest_local_features(feature_month)
    local_columns = [column for column in latest_local.columns if column not in {"family_id", "family_desc", "valid_price"}]
    base = base.merge(
        latest_local[local_columns], on=["branch_code", "canonical_product_id"], how="left", validate="one_to_one"
    )
    observed = _positive_evidence(feature_month, product_rows)
    product_totals = observed.groupby("canonical_product_id", as_index=False)["observed_positive_units"].sum().rename(
        columns={"observed_positive_units": "product_positive_units_all_branches"}
    )
    local_product_totals = observed.groupby(["branch_code", "canonical_product_id"], as_index=False)[
        "observed_positive_units"
    ].sum().rename(columns={"observed_positive_units": "product_positive_units_target_branch"})
    family_totals = observed.loc[observed["family_id"].notna() & observed["family_id"].ne("")].groupby(
        ["branch_code", "family_id"], as_index=False
    )["observed_positive_units"].sum().rename(columns={"observed_positive_units": "family_positive_units_target_branch"})
    base = base.merge(product_totals, on="canonical_product_id", how="left", validate="many_to_one")
    base = base.merge(local_product_totals, on=["branch_code", "canonical_product_id"], how="left", validate="one_to_one")
    base = base.merge(family_totals, on=["branch_code", "family_id"], how="left", validate="many_to_one")
    for column in (
        "product_positive_units_all_branches",
        "product_positive_units_target_branch",
        "family_positive_units_target_branch",
    ):
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0)
    base["peer_product_evidence_strength"] = (
        base["product_positive_units_all_branches"] - base["product_positive_units_target_branch"]
    ).clip(lower=0.0)
    base["peer_evidence_available"] = base["peer_product_evidence_strength"].gt(0)
    base["family_evidence_available"] = base["family_positive_units_target_branch"].gt(0) & base["family_id"].notna() & base["family_id"].ne("")
    # No approved similarity input exists in Days 1-6; preserve that absence.
    base["similar_branch_evidence_available"] = False
    base["price_evidence_available"] = base["valid_price"]

    classified = classify_track_b_candidates(base)
    evaluations = pd.concat(
        [
            base.reset_index(drop=True),
            classified.loc[
                :,
                ["track", "candidate_status", "local_usable_history", "eligibility_status", "exclusion_reason"],
            ],
        ],
        axis=1,
    )
    evaluations["peer_product_evidence_available"] = evaluations["peer_evidence_available"]
    evaluations["target_branch_family_evidence_available"] = evaluations["family_evidence_available"]
    unusable = (~evaluations["product_record_usable"]) & ~evaluations["exclusion_reason"].isin(
        [MISSING_PRODUCT_IDENTITY, UNRESOLVED_MAPPING]
    )
    evaluations.loc[unusable, ["track", "candidate_status", "exclusion_reason"]] = [
        EXCLUDED,
        None,
        "UNUSABLE_PRODUCT_RECORD",
    ]
    evaluations.loc[
        evaluations["track"].eq(TRACK_A), "exclusion_reason"
    ] = "HAS_USABLE_LOCAL_HISTORY"
    evaluations["evidence_signal_count"] = evaluations[
        [
            "peer_product_evidence_available",
            "target_branch_family_evidence_available",
            "similar_branch_evidence_available",
            "price_evidence_available",
        ]
    ].sum(axis=1).astype(int)
    evaluations["rank"] = pd.Series(pd.NA, index=evaluations.index, dtype="Int64")
    evaluations["in_shortlist"] = False

    pool = evaluations.loc[evaluations["track"].eq(TRACK_B)].copy()
    ordered_pool = pool.sort_values(
        [
            "branch_code",
            "evidence_signal_count",
            "peer_product_evidence_strength",
            "family_positive_units_target_branch",
            "similar_branch_evidence_available",
            "price_evidence_available",
            "canonical_product_id",
        ],
        ascending=[True, False, False, False, False, False, True],
        kind="stable",
    )
    shortlist = ordered_pool.groupby("branch_code", sort=False, group_keys=False).head(top_n).copy()
    shortlist["rank"] = shortlist.groupby("branch_code", sort=False).cumcount().add(1).astype("Int64")
    shortlist["in_shortlist"] = True
    shortlist_lookup = shortlist.set_index(["branch_code", "canonical_product_id"])[["rank", "in_shortlist"]]
    lookup_index = pd.MultiIndex.from_frame(evaluations[["branch_code", "canonical_product_id"]])
    evaluations["rank"] = shortlist_lookup["rank"].reindex(lookup_index).array
    evaluations["in_shortlist"] = shortlist_lookup["in_shortlist"].reindex(lookup_index).fillna(False).to_numpy(dtype=bool)

    summary = branch_rows[["branch_code"]].copy()
    summary["total_valid_products"] = int(product_rows["canonical_product_id"].notna().sum() - product_rows["canonical_product_id"].eq("").sum())
    for label, mask in (
        ("excluded_track_a_count", evaluations["exclusion_reason"].eq("HAS_USABLE_LOCAL_HISTORY")),
        ("excluded_unresolved_count", evaluations["exclusion_reason"].isin([UNRESOLVED_MAPPING, MISSING_PRODUCT_IDENTITY])),
        ("excluded_unusable_count", evaluations["exclusion_reason"].eq("UNUSABLE_PRODUCT_RECORD")),
    ):
        counts = evaluations.loc[mask].groupby("branch_code").size()
        summary[label] = summary["branch_code"].map(counts).fillna(0).astype(int)
    pool_counts = pool.groupby("branch_code").size()
    shortlist_counts = shortlist.groupby("branch_code").size()
    summary["candidate_pool_count"] = summary["branch_code"].map(pool_counts).fillna(0).astype(int)
    summary["shortlisted_count"] = summary["branch_code"].map(shortlist_counts).fillna(0).astype(int)
    summary = summary.sort_values("branch_code", kind="stable").reset_index(drop=True)
    return CandidatePoolBuildResult(
        evaluations=evaluations.sort_values(["branch_code", "canonical_product_id"], kind="stable").reset_index(drop=True),
        candidate_pool=ordered_pool.reset_index(drop=True),
        candidate_shortlist=shortlist.sort_values(["branch_code", "rank"], kind="stable").reset_index(drop=True),
        branch_summary=summary,
    )


def write_candidate_pool_artifacts(result: CandidatePoolBuildResult, output_directory: Path) -> None:
    """Persist traceable evaluations and branch totals; never create planning options."""
    output_directory.mkdir(parents=True, exist_ok=True)
    result.evaluations.to_parquet(output_directory / "candidate_results.parquet", index=False)
    result.branch_summary.to_csv(output_directory / "candidate_summary.csv", index=False)
