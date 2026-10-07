import pandas as pd
import pytest

from ml.candidates.candidate_scoring import score_candidate_evaluations
from ml.candidates.generate_candidates import RESEARCH_SHORTLIST


def _evaluations():
    return pd.DataFrame([
        {"branch_code": "BIL01", "canonical_product_id": "CP_A", "product_desc": "A", "family_id": "F1", "family_desc": "Family 1", "candidate_status": RESEARCH_SHORTLIST},
        {"branch_code": "BIL01", "canonical_product_id": "CP_B", "product_desc": "B", "family_id": "F2", "family_desc": "Family 2", "candidate_status": RESEARCH_SHORTLIST},
    ])


def _features():
    return pd.DataFrame([
        {"branch_code": "BIL01", "canonical_product_id": "CP_A", "family_id": "F1", "fulfilled_positive_units_proxy": 100.0},
        {"branch_code": "BIN01", "canonical_product_id": "CP_A", "family_id": "F1", "fulfilled_positive_units_proxy": 20.0},
        {"branch_code": "BIN01", "canonical_product_id": "CP_B", "family_id": "F2", "fulfilled_positive_units_proxy": 80.0},
        {"branch_code": "BIL01", "canonical_product_id": "CP_X", "family_id": "F1", "fulfilled_positive_units_proxy": 90.0},
        {"branch_code": "BIL01", "canonical_product_id": "CP_Y", "family_id": "F2", "fulfilled_positive_units_proxy": 10.0},
    ])


def _prices():
    return pd.DataFrame([
        {"canonical_product_id": "CP_A", "list_price": 100.0, "price_valid": True},
        {"canonical_product_id": "CP_B", "list_price": 200.0, "price_valid": True},
        {"canonical_product_id": "CP_X", "list_price": 105.0, "price_valid": True},
        {"canonical_product_id": "CP_Y", "list_price": 95.0, "price_valid": True},
    ])


def _scored(**kwargs):
    return score_candidate_evaluations(_evaluations(), _features(), _prices(), top_n=10, **kwargs).candidate_shortlist


def test_peer_raw_excludes_target_branch_and_higher_peer_has_higher_score():
    scored = _scored().set_index("canonical_product_id")
    assert scored.loc["CP_A", "peer_product_demand_raw"] == 20.0
    assert scored.loc["CP_B", "peer_product_demand_raw"] == 80.0
    assert scored.loc["CP_B", "peer_product_demand_score"] > scored.loc["CP_A", "peer_product_demand_score"]


def test_stronger_target_branch_family_demand_has_higher_family_score():
    scored = _scored().set_index("canonical_product_id")
    assert scored.loc["CP_A", "target_branch_family_demand_score"] > scored.loc["CP_B", "target_branch_family_demand_score"]


def test_no_similarity_renormalizes_weights_and_invalid_price_removes_price_weight():
    prices = _prices()
    prices.loc[prices.canonical_product_id.eq("CP_A"), "price_valid"] = False
    result = score_candidate_evaluations(_evaluations(), _features(), prices, top_n=10).candidate_shortlist.set_index("canonical_product_id")
    assert not result["similarity_component_available"].any()
    assert result.loc["CP_B", "effective_peer_weight"] == pytest.approx(0.50)
    assert result.loc["CP_B", "effective_family_weight"] == pytest.approx(0.375)
    assert result.loc["CP_B", "effective_price_weight"] == pytest.approx(0.125)
    assert result.loc["CP_A", "price_component_available"] == False
    assert result.loc["CP_A", "effective_price_weight"] == 0.0


def test_supplied_defensible_similarity_keeps_original_weights():
    similarity = pd.DataFrame({"branch_code": ["BIL01", "BIL01"], "canonical_product_id": ["CP_A", "CP_B"], "similar_branch_product_raw": [2.0, 4.0]})
    scored = _scored(similarity_evidence=similarity).set_index("canonical_product_id")
    assert scored.loc["CP_A", "similarity_component_available"]
    assert scored.loc["CP_A", "effective_peer_weight"] == pytest.approx(0.40)
    assert scored.loc["CP_A", "effective_family_weight"] == pytest.approx(0.30)
    assert scored.loc["CP_A", "effective_similarity_weight"] == pytest.approx(0.20)
    assert scored.loc["CP_A", "effective_price_weight"] == pytest.approx(0.10)


def test_scores_weights_status_and_ranking_are_safe_and_deterministic():
    first = _scored()
    second = _scored()
    assert first["candidate_score"].between(0, 1).all()
    assert first[["peer_product_demand_score", "target_branch_family_demand_score", "price_fit_score"]].stack().between(0, 1).all()
    assert first["effective_weight_sum"].sub(1.0).abs().lt(1e-9).all()
    assert first["candidate_status"].eq(RESEARCH_SHORTLIST).all()
    assert "INTRODUCE" not in " ".join(first.astype("string").fillna("").to_numpy().ravel()).upper()
    assert first[["canonical_product_id", "rank"]].equals(second[["canonical_product_id", "rank"]])
