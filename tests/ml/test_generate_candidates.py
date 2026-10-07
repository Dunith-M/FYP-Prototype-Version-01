from unittest.mock import patch

import pandas as pd

from ml.candidates.generate_candidates import (
    EXCLUDED,
    RESEARCH_SHORTLIST,
    TRACK_A,
    TRACK_B,
    classify_track_b_candidate,
    classify_track_b_candidates,
)


def _row(**overrides):
    return {
        "branch_code": "BIL01",
        "canonical_product_id": "CP_P100",
        "lag_1": None,
        "mapping_status": "CONFIRMED",
        "peer_evidence_available": False,
        "family_evidence_available": False,
        "similar_branch_evidence_available": False,
        "price_evidence_available": False,
        "family_id": "F1",
    } | overrides


def test_usable_local_history_is_track_a_not_track_b() -> None:
    result = classify_track_b_candidate(_row(lag_1=4.0))

    assert result.track == TRACK_A
    assert result.local_usable_history is True
    assert result.candidate_status is None


def test_no_local_history_with_valid_external_evidence_is_research_shortlist() -> None:
    result = classify_track_b_candidate(_row(peer_evidence_available=True, family_evidence_available=True))

    assert result.track == TRACK_B
    assert result.candidate_status == RESEARCH_SHORTLIST
    assert result.local_usable_history is False
    assert result.peer_evidence_available is True
    assert result.family_evidence_available is True


def test_unresolved_mapping_is_excluded_before_track_b_evaluation() -> None:
    result = classify_track_b_candidate(_row(mapping_status="UNRESOLVED", peer_evidence_available=True))

    assert result.track == EXCLUDED
    assert result.exclusion_reason == "UNRESOLVED_MAPPING"


def test_no_local_history_alone_does_not_create_a_candidate() -> None:
    result = classify_track_b_candidate(_row())

    assert result.track == EXCLUDED
    assert result.exclusion_reason == "NO_EXTERNAL_EVIDENCE"


def test_records_never_use_introduce_and_do_not_invoke_track_a_forecasting() -> None:
    candidates = pd.DataFrame([_row(peer_evidence_available=True), _row(canonical_product_id="CP_P200", lag_1=2.0)])

    with patch("ml.evaluation.confidence.apply_confidence_to_forecasts") as forecast_path:
        results = classify_track_b_candidates(candidates)

    assert not results.astype("string").apply(lambda column: column.str.contains("INTRODUCE", case=False).any()).any()
    forecast_path.assert_not_called()
