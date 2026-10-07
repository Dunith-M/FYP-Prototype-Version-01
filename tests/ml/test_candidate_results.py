import pandas as pd
import pytest

from ml.candidates.candidate_results import PARAMETER_SOURCE, build_final_candidate_results
from ml.candidates.generate_candidates import RESEARCH_SHORTLIST


def _scored(count=7):
    return pd.DataFrame([{"branch_code":"BIL01","canonical_product_id":f"CP_{i:02d}","candidate_status":RESEARCH_SHORTLIST,"candidate_score":1-i/20,"product_desc":"P","family_id":"F","family_desc":"Family","peer_product_demand_raw":1.,"peer_product_demand_score":.5,"target_branch_family_demand_raw":2.,"target_branch_family_demand_score":.5,"similar_branch_product_raw":float("nan"),"similar_branch_product_score":float("nan"),"price_fit_raw":.1,"price_fit_score":.5,"peer_component_available":True,"family_component_available":True,"similarity_component_available":False,"price_component_available":True,"effective_peer_weight":.5,"effective_family_weight":.375,"effective_similarity_weight":0.,"effective_price_weight":.125,"score_method_version":"v1","mapping_status":"CONFIRMED"} for i in range(count)])


def _features():
    return pd.DataFrame([{"branch_code":"BIL01","canonical_product_id":"CP_TRACK_A","year_month":"2026-03","lag_1":1.0}])


def test_final_grain_labels_ranks_and_shortlist_limit():
    result = build_final_candidate_results(_scored(), _features(), planning_month="2026-04", source_month="2026-03")
    assert len(result.results) == 7
    assert result.results[["branch_code","canonical_product_id","planning_month"]].duplicated().sum() == 0
    assert result.results.status.eq(RESEARCH_SHORTLIST).all()
    assert result.results.parameter_source.eq(PARAMETER_SOURCE).all()
    assert result.results.candidate_rank.tolist() == list(range(1, 8))


def test_duplicate_keys_and_track_a_overlap_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build_final_candidate_results(pd.concat([_scored(1), _scored(1)]), _features(), planning_month="2026-04", source_month="2026-03")
    overlap = _scored(1)
    overlap.loc[0, "canonical_product_id"] = "CP_TRACK_A"
    with pytest.raises(ValueError, match="Track-A overlap"):
        build_final_candidate_results(overlap, _features(), planning_month="2026-04", source_month="2026-03")
