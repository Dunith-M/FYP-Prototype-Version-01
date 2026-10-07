"""Load finalized Day-07 research evidence into ``CandidateResult`` records."""

from __future__ import annotations

from datetime import date

import pandas as pd
from django.db import transaction

from analytics.models import CandidateResult
from catalog.models import Branch, Product


REQUIRED_COLUMNS = {
    "branch_code",
    "planning_month",
    "canonical_product_id",
    "peer_product_demand_score",
    "branch_family_demand_score",
    "similar_branch_product_score",
    "price_fit_score",
    "candidate_score",
    "candidate_rank",
    "evidence_class",
    "status",
    "parameter_source",
    "score_method_version",
}


def _nullable_float(value: object) -> float | None:
    """Preserve unavailable evidence components as null rather than zero."""
    return None if pd.isna(value) else float(value)


def _planning_month(value: object) -> date:
    """Convert the artifact's YYYY-MM month key to the project's month date."""
    return date.fromisoformat(f"{str(value)}-01")


def load_candidate_results(frame: pd.DataFrame) -> tuple[int, int]:
    """Idempotently persist final Day-07 research results.

    The input must already be finalized (including the Track-A anti-join). This
    loader does not calculate or alter candidate scores; it only persists the
    evidence artifact for the Django evidence page.
    """
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate result artifact is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Candidate result artifact is empty; refusing to load no evidence.")
    if frame.duplicated(["branch_code", "canonical_product_id", "planning_month", "score_method_version"]).any():
        raise ValueError("Candidate result artifact has duplicate branch/product/month/run rows.")
    if not frame["status"].eq(CandidateResult.CandidateStatus.RESEARCH_SHORTLIST).all():
        raise ValueError("Only RESEARCH_SHORTLIST evidence may be loaded into CandidateResult.")
    if not frame["parameter_source"].eq(CandidateResult.ParameterSource.HISTORICAL_EVIDENCE).all():
        raise ValueError("CandidateResult requires HISTORICAL_EVIDENCE provenance.")

    branch_codes = set(frame["branch_code"].astype(str))
    product_ids = set(frame["canonical_product_id"].astype(str))
    branches = {code: pk for code, pk in Branch.objects.filter(branch_code__in=branch_codes).values_list("branch_code", "pk")}
    products = {
        code: pk
        for code, pk in Product.objects.filter(canonical_product_id__in=product_ids).values_list("canonical_product_id", "pk")
    }
    if set(branches) != branch_codes:
        raise ValueError(f"Unknown branch codes in candidate artifact: {sorted(branch_codes - set(branches))}")
    if set(products) != product_ids:
        raise ValueError(f"Unknown canonical product IDs in candidate artifact: {sorted(product_ids - set(products))}")

    created = updated = 0
    with transaction.atomic():
        for row in frame.itertuples(index=False):
            _, was_created = CandidateResult.objects.update_or_create(
                branch_id=branches[row.branch_code],
                product_id=products[row.canonical_product_id],
                planning_month=_planning_month(row.planning_month),
                run_version=str(row.score_method_version),
                defaults={
                    "peer_score": _nullable_float(row.peer_product_demand_score),
                    "family_score": _nullable_float(row.branch_family_demand_score),
                    "similar_branch_score": _nullable_float(row.similar_branch_product_score),
                    "price_fit_score": _nullable_float(row.price_fit_score),
                    "candidate_score": float(row.candidate_score),
                    "rank": int(row.candidate_rank),
                    "evidence_class": row.evidence_class,
                    "status": row.status,
                    "parameter_source": row.parameter_source,
                },
            )
            created += int(was_created)
            updated += int(not was_created)
    return created, updated
