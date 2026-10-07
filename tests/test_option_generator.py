import pandas as pd
import pytest

from planning.services.option_generator import (
    HISTORICAL_LAG_1_NORMALIZED,
    TRACK_B_CANDIDATE_SCORE,
    build_option_evidence_scores,
    generate_finite_options,
)


def _options():
    return {"mode":"RESEARCH_SYNTHETIC", "default_quantity_step":1, "max_target_per_product":5}


def test_listed_unlisted_edges_and_provenance_create_only_finite_options():
    fixture = pd.DataFrame([
        {"branch_code":"BIL01","canonical_product_id":"CP_3","L0":1,"q0":3,"parameter_source":"SYNTHETIC_DEMO"},
        {"branch_code":"BIL01","canonical_product_id":"CP_1","L0":1,"q0":1,"parameter_source":"SYNTHETIC_DEMO"},
        {"branch_code":"BIL01","canonical_product_id":"CP_5","L0":1,"q0":5,"parameter_source":"SYNTHETIC_DEMO"},
        {"branch_code":"BIL01","canonical_product_id":"CP_C","L0":0,"q0":None,"parameter_source":"SYNTHETIC_DEMO"},
    ])
    result = generate_finite_options(fixture, _options())
    pairs = lambda product: set(map(tuple, result.loc[result.canonical_product_id.eq(product), ["L1","q1"]].to_numpy()))
    assert pairs("CP_3") == {(1,2),(1,3),(1,4),(0,0)}
    assert (1,0) not in pairs("CP_1")
    assert all(q1 <= 5 for _, q1 in pairs("CP_5"))
    assert pairs("CP_C") == {(0,0),(1,1)}
    assert not result.duplicated(["branch_code","canonical_product_id","L1","q1"]).any()
    assert result.parameter_source.eq("SYNTHETIC_DEMO").all()
    assert not {"selected", "winner", "action", "recommendation"} & set(result.columns)


def test_complete_option_evidence_uses_track_b_scores_and_normalized_historical_lag_1():
    fixture = pd.DataFrame([
        {"branch_code": "BIL01", "canonical_product_id": "CP_LISTED_LOW", "L0": 1, "parameter_source": "SYNTHETIC_DEMO"},
        {"branch_code": "BIL01", "canonical_product_id": "CP_LISTED_HIGH", "L0": 1, "parameter_source": "SYNTHETIC_DEMO"},
        {"branch_code": "BIL01", "canonical_product_id": "CP_CANDIDATE", "L0": 0, "parameter_source": "SYNTHETIC_DEMO"},
    ])
    candidates = pd.DataFrame([{"branch_code": "BIL01", "canonical_product_id": "CP_CANDIDATE", "candidate_score": 0.8}])
    features = pd.DataFrame([
        {"branch_code": "BIL01", "canonical_product_id": "CP_LISTED_LOW", "year_month": "2026-02", "lag_1": 1},
        {"branch_code": "BIL01", "canonical_product_id": "CP_LISTED_LOW", "year_month": "2026-03", "lag_1": 2},
        {"branch_code": "BIL01", "canonical_product_id": "CP_LISTED_HIGH", "year_month": "2026-03", "lag_1": 4},
    ])

    result = build_option_evidence_scores(fixture, candidates, features).set_index("canonical_product_id")

    assert result.loc["CP_CANDIDATE", "evidence_score"] == 0.8
    assert result.loc["CP_CANDIDATE", "evidence_source"] == TRACK_B_CANDIDATE_SCORE
    assert result.loc["CP_LISTED_LOW", "evidence_score"] == 0.5
    assert result.loc["CP_LISTED_HIGH", "evidence_score"] == 1.0
    assert set(result.loc[["CP_LISTED_LOW", "CP_LISTED_HIGH"], "evidence_source"]) == {HISTORICAL_LAG_1_NORMALIZED}


def test_option_evidence_refuses_missing_historical_or_candidate_evidence():
    fixture = pd.DataFrame([
        {"branch_code": "BIL01", "canonical_product_id": "CP_LISTED", "L0": 1, "parameter_source": "SYNTHETIC_DEMO"},
        {"branch_code": "BIL01", "canonical_product_id": "CP_CANDIDATE", "L0": 0, "parameter_source": "SYNTHETIC_DEMO"},
    ])
    features = pd.DataFrame(columns=["branch_code", "canonical_product_id", "year_month", "lag_1"])
    candidates = pd.DataFrame(columns=["branch_code", "canonical_product_id", "candidate_score"])

    with pytest.raises(ValueError, match="listed synthetic products need"):
        build_option_evidence_scores(fixture, candidates, features)
