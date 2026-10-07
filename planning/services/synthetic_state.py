"""Explicit synthetic L0/q0 fixture generation and validation for Day-07."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


FIXTURE_COLUMNS = ["branch_code", "canonical_product_id", "L0", "q0", "parameter_source"]
SYNTHETIC_SOURCE = "SYNTHETIC_DEMO"


def load_synthetic_options(path: Path) -> dict[str, object]:
    """Load the approved prototype-only configuration, never operational policy."""
    with path.open(encoding="utf-8") as stream:
        options = yaml.safe_load(stream)
    if not isinstance(options, dict) or options.get("parameter_source") != SYNTHETIC_SOURCE:
        raise ValueError("Synthetic options must be explicitly marked SYNTHETIC_DEMO.")
    return options


def build_synthetic_l0_q0_fixture(feature_month: pd.DataFrame, candidates: pd.DataFrame, options: dict[str, object], *, branch_code: str = "BIL01", listed_count: int = 15, candidate_count: int = 15) -> pd.DataFrame:
    """Build a reproducible demo state without reading physical inventory.

    Listed rows are the first canonical IDs with usable March local history.
    Their q0 cycles ``1..max_target_per_product`` in canonical-ID order.
    Candidate rows are final Track-B evidence rows and remain L0=0/q0=null.
    """
    maximum = options.get("max_target_per_product")
    if not isinstance(maximum, int) or maximum < 1:
        raise ValueError("Synthetic max_target_per_product must be a positive integer.")
    source_month = str(feature_month["year_month"].astype(str).max())
    listed = feature_month.loc[(feature_month["branch_code"].eq(branch_code)) & (feature_month["year_month"].astype(str).eq(source_month)) & feature_month["lag_1"].notna(), "canonical_product_id"].dropna().astype(str).sort_values().drop_duplicates().head(listed_count).tolist()
    candidate_ids = candidates.loc[(candidates["branch_code"].eq(branch_code)) & (candidates["candidate_rank"].notna()), "canonical_product_id"].astype(str).sort_values().drop_duplicates().head(candidate_count).tolist()
    candidate_ids = [product_id for product_id in candidate_ids if product_id not in set(listed)]
    if len(listed) != listed_count or len(candidate_ids) != candidate_count:
        raise ValueError("Insufficient source products for the requested synthetic fixture size.")
    rows = [{"branch_code": branch_code, "canonical_product_id": product_id, "L0": 1, "q0": (index % maximum) + 1, "parameter_source": SYNTHETIC_SOURCE} for index, product_id in enumerate(listed)]
    rows.extend({"branch_code": branch_code, "canonical_product_id": product_id, "L0": 0, "q0": None, "parameter_source": SYNTHETIC_SOURCE} for product_id in candidate_ids)
    return pd.DataFrame(rows, columns=FIXTURE_COLUMNS)


def validate_synthetic_l0_q0_fixture(fixture: pd.DataFrame, options: dict[str, object], *, branch_codes: set[str], product_ids: set[str], candidate_ids: set[str]) -> None:
    """Validate synthetic state semantics; inventory fields are intentionally absent."""
    if fixture.columns.tolist() != FIXTURE_COLUMNS or not 20 <= len(fixture) <= 50:
        raise ValueError("Fixture schema or demo row count is invalid.")
    if fixture["branch_code"].nunique() != 1 or not set(fixture["branch_code"]).issubset(branch_codes) or not set(fixture["canonical_product_id"]).issubset(product_ids):
        raise ValueError("Fixture branch or canonical product identity is invalid.")
    if fixture.duplicated(["branch_code", "canonical_product_id"]).any() or not fixture["L0"].isin([0, 1]).all() or not fixture["parameter_source"].eq(SYNTHETIC_SOURCE).all():
        raise ValueError("Fixture keys, L0 values, or provenance are invalid.")
    listed, unlisted = fixture["L0"].eq(1), fixture["L0"].eq(0)
    q0 = pd.to_numeric(fixture["q0"], errors="coerce")
    if q0.loc[listed].isna().any() or not q0.loc[listed].gt(0).all() or not q0.loc[listed].le(options["max_target_per_product"]).all() or q0.loc[unlisted].notna().any():
        raise ValueError("Fixture q0 values violate synthetic L0/q0 semantics.")
    candidate_rows = fixture[fixture["canonical_product_id"].isin(candidate_ids)]
    if not candidate_rows.empty and not (candidate_rows["L0"].eq(0) & q0.loc[candidate_rows.index].isna()).all():
        raise ValueError("Track-B candidates must be unlisted with null q0 in the synthetic fixture.")
