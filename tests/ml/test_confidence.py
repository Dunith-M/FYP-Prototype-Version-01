from dataclasses import replace
from pathlib import Path

import pandas as pd

from ml.evaluation.confidence import (
    ADEQUATE,
    AVAILABLE,
    AVAILABLE_WITH_WARNING,
    ConfidenceConfig,
    DEFAULT_CONFIDENCE_CONFIG,
    INSUFFICIENT,
    LIMITED,
    WITHHELD,
    apply_confidence_to_forecasts,
    classify_evidence,
)


def _strong_row(**overrides):
    row = {
        "canonical_product_id": "CP_1",
        "lag_1": 4.0,
        "months_with_positive_sales": 3,
        "intermittency": 0.2,
        "inventory_observation_count": 5,
        "availability_unknown_rate": 0.1,
        "unresolved_product_flag": False,
        "reconciliation_difference_flag": False,
        "fallback_used": False,
    }
    return row | overrides


def test_strong_history_is_adequate() -> None:
    assessment = classify_evidence(_strong_row())
    assert assessment.evidence_class == ADEQUATE
    assert assessment.forecast_status == AVAILABLE


def test_sparse_history_high_intermittency_weak_inventory_and_fallback_are_limited() -> None:
    for row, reason in (
        (_strong_row(months_with_positive_sales=2), "SPARSE_HISTORY"),
        (_strong_row(intermittency=0.5), "HIGH_INTERMITTENCY"),
        (_strong_row(inventory_observation_count=0), "INVENTORY_EVIDENCE_LIMITED"),
        (_strong_row(fallback_used=True), "FALLBACK_MODEL_USED"),
    ):
        assessment = classify_evidence(row)
        assert assessment.evidence_class == LIMITED
        assert assessment.forecast_status == AVAILABLE_WITH_WARNING
        assert reason in assessment.reason_codes


def test_blocking_mapping_history_and_dq_conditions_withhold_forecasts() -> None:
    for row, reason in (
        (_strong_row(unresolved_product_flag=True), "UNRESOLVED_MAPPING"),
        (_strong_row(lag_1=None), "NO_USABLE_HISTORY"),
        (_strong_row(reconciliation_difference_flag=True), "CRITICAL_DQ_BLOCK"),
    ):
        assessment = classify_evidence(row)
        assert assessment.evidence_class == INSUFFICIENT
        assert assessment.forecast_status == WITHHELD
        assert reason in assessment.reason_codes


def test_insufficient_precedence_and_forecast_withholding() -> None:
    assessment = classify_evidence(_strong_row(unresolved_product_flag=True, months_with_positive_sales=1))
    assert assessment.evidence_class == INSUFFICIENT
    assert "UNRESOLVED_MAPPING" in assessment.reason_codes
    assert "SPARSE_HISTORY" not in assessment.reason_codes

    forecasts = pd.DataFrame(
        {
            "branch_code": ["BIL01"], "canonical_product_id": ["CP_1"],
            "target_month": ["2026-04"], "predicted_units": [8.4], "model_name": ["trailing_mean_3"],
        }
    )
    features = pd.DataFrame(
        [
            _strong_row(target_year_month="2026-04", branch_code="BIL01", unresolved_product_flag=True),
            _strong_row(target_year_month=None, branch_code="BIL01"),
        ]
    )
    output = apply_confidence_to_forecasts(forecasts, features)
    assert output.loc[0, "forecast_status"] == WITHHELD
    assert pd.isna(output.loc[0, "forecast_value"])
    assert output.loc[0, "raw_prediction"] == 8.4
    assert output.loc[0, "confidence_config_version"] == DEFAULT_CONFIDENCE_CONFIG.config_version
    assert output.loc[0, "confidence_config_mode"] == "RESEARCH_PROTOTYPE"


def test_research_configuration_changes_the_history_classification_without_magic_numbers() -> None:
    row = _strong_row(months_with_positive_sales=3)
    stricter_config = replace(DEFAULT_CONFIDENCE_CONFIG, min_positive_history_months=4)

    assert classify_evidence(row, DEFAULT_CONFIDENCE_CONFIG).evidence_class == ADEQUATE
    assert classify_evidence(row, stricter_config).evidence_class == LIMITED
    assert stricter_config.to_dict()["min_positive_history_months"] == 4


def test_thresholds_are_explicitly_marked_as_non_singer_approved_research_assumptions() -> None:
    source = Path(__file__).parents[2] / "ml" / "evaluation" / "confidence.py"
    text = source.read_text(encoding="utf-8").upper()

    assert "RESEARCH PROTOTYPE THRESHOLD" in text
    assert "NOT SINGER-APPROVED POLICY" in text
    assert "SINGER REQUIRES" not in text
    assert "SINGER CONFIDENCE THRESHOLD" not in text
    assert "APPROVED SINGER MINIMUM" not in text
