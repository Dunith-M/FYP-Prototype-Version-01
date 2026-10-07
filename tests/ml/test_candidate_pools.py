import pandas as pd

from ml.candidates.generate_candidates import (
    DEFAULT_TOP_N,
    EXCLUDED,
    RESEARCH_SHORTLIST,
    TRACK_B,
    build_candidate_pools,
)


def _inputs():
    products = pd.DataFrame(
        [
            {"canonical_product_id": "CP_A", "product_desc": "Track A", "family_id": "F1", "family_desc": "Family 1", "price_valid": True},
            {"canonical_product_id": "CP_B", "product_desc": "Candidate", "family_id": "F1", "family_desc": "Family 1", "price_valid": False},
            {"canonical_product_id": "CP_C", "product_desc": "Unmapped", "family_id": "F2", "family_desc": "Family 2", "price_valid": True},
            {"canonical_product_id": "CP_D", "product_desc": "", "family_id": "F1", "family_desc": "Family 1", "price_valid": True},
            {"canonical_product_id": None, "product_desc": "Missing ID", "family_id": "F3", "family_desc": "Family 3", "price_valid": True},
        ]
    )
    branches = pd.DataFrame({"branch_code": ["BIL01", "BIN01"]})
    mappings = pd.DataFrame({"canonical_product_id": ["CP_A", "CP_B", "CP_D"]})
    features = pd.DataFrame(
        [
            {"branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2026-02", "lag_1": 2.0, "fulfilled_positive_units_proxy": 5.0},
            {"branch_code": "BIN01", "canonical_product_id": "CP_B", "year_month": "2026-02", "lag_1": 3.0, "fulfilled_positive_units_proxy": 8.0},
            {"branch_code": "BIL01", "canonical_product_id": "CP_D", "year_month": "2026-02", "lag_1": None, "fulfilled_positive_units_proxy": 0.0},
        ]
    )
    return products, branches, mappings, features


def _result(top_n=10):
    return build_candidate_pools(*_inputs(), top_n=top_n)


def test_pool_excludes_track_a_unresolved_missing_and_unusable_records() -> None:
    result = _result()
    bil = result.evaluations.loc[result.evaluations.branch_code.eq("BIL01")].set_index("canonical_product_id")

    assert bil.loc["CP_A", "exclusion_reason"] == "HAS_USABLE_LOCAL_HISTORY"
    assert bil.loc["CP_C", "exclusion_reason"] == "UNRESOLVED_MAPPING"
    assert bil.loc["CP_D", "exclusion_reason"] == "UNUSABLE_PRODUCT_RECORD"
    assert bil.loc[pd.isna(bil.index), "exclusion_reason"].iloc[0] == "MISSING_PRODUCT_IDENTITY"
    assert result.candidate_pool["canonical_product_id"].eq("CP_B").any()
    assert (result.evaluations["track"].eq(EXCLUDED)).any()


def test_valid_track_b_enters_pool_with_research_status_and_never_introduce() -> None:
    result = _result()
    candidate = result.candidate_pool.loc[
        (result.candidate_pool.branch_code == "BIL01") & (result.candidate_pool.canonical_product_id == "CP_B")
    ].iloc[0]

    assert candidate.track == TRACK_B
    assert candidate.candidate_status == RESEARCH_SHORTLIST
    assert candidate.peer_product_evidence_available
    assert "INTRODUCE" not in " ".join(result.candidate_pool.astype("string").fillna("").to_numpy().ravel()).upper()


def test_shortlist_is_deterministic_and_never_exceeds_top_n_or_pool() -> None:
    first = _result()
    second = _result()

    assert len(first.candidate_shortlist) <= DEFAULT_TOP_N
    assert len(first.candidate_shortlist) <= len(first.candidate_pool)
    assert first.candidate_shortlist[["branch_code", "canonical_product_id", "rank"]].equals(
        second.candidate_shortlist[["branch_code", "canonical_product_id", "rank"]]
    )


def test_configured_top_n_caps_each_branch_shortlist() -> None:
    products, branches, mappings, features = _inputs()
    extra_ids = [f"CP_EXTRA_{number:02d}" for number in range(11)]
    extras = pd.DataFrame(
        [
            {"canonical_product_id": product_id, "product_desc": product_id, "family_id": "F1", "family_desc": "Family 1", "price_valid": True}
            for product_id in extra_ids
        ]
    )
    products = pd.concat([products, extras], ignore_index=True)
    mappings = pd.concat([mappings, pd.DataFrame({"canonical_product_id": extra_ids})], ignore_index=True)
    features = pd.concat(
        [
            features,
            pd.DataFrame(
                [
                    {"branch_code": "BIN01", "canonical_product_id": product_id, "year_month": "2026-02", "lag_1": 1.0, "fulfilled_positive_units_proxy": 1.0}
                    for product_id in extra_ids
                ]
            ),
        ],
        ignore_index=True,
    )

    result = build_candidate_pools(products, branches, mappings, features, top_n=10)
    bil_shortlist = result.candidate_shortlist.loc[result.candidate_shortlist.branch_code.eq("BIL01")]

    assert len(bil_shortlist) == 10
    assert len(bil_shortlist) <= 10
