import pandas as pd
import pytest

from ml.evaluation.compare_lightgbm import LightGBMComparisonError, build_lightgbm_comparison


def _baseline_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": ["fold_01", "fold_02", "fold_03"],
            "validation_month": ["2025-12", "2026-01", "2026-02"],
            "target_month": ["2025-12", "2026-01", "2026-02"],
            "branch_code": ["BIL01", "BIL01", "BIL01"],
            "canonical_product_id": ["CP_1", "CP_1", "CP_1"],
            "actual_units": [2.0, 4.0, 6.0],
            "predicted_units": [1.0, 2.0, 3.0],
            "prediction_available": [True, True, True],
            "model_name": ["trailing_mean_3"] * 3,
        }
    )


def _lightgbm_predictions() -> pd.DataFrame:
    rows = _baseline_predictions().drop(columns="model_name").copy()
    rows["predicted_units"] = [2.0, 3.0, 5.0]
    return rows


def _fold_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": ["fold_01", "fold_02", "fold_03"],
            "training_period": ["2025-05 through 2025-11", "2025-05 through 2025-12", "2025-05 through 2026-01"],
            "training_cutoff_month": ["2025-11", "2025-12", "2026-01"],
            "validation_month": ["2025-12", "2026-01", "2026-02"],
        }
    )


def test_comparison_uses_shared_metrics_after_identical_key_validation() -> None:
    result = build_lightgbm_comparison(
        _baseline_predictions(), _lightgbm_predictions(), _fold_summary(), "trailing_mean_3"
    )

    assert len(result.predictions) == 3
    assert len(result.metrics) == 6
    assert result.metrics["observation_keys_identical"].all()
    lightgbm_fold_02 = result.metrics.loc[
        (result.metrics["fold_id"] == "fold_02")
        & (result.metrics["model_name"] == "lightgbm_challenger")
    ].iloc[0]
    assert lightgbm_fold_02.mae == 1.0
    assert lightgbm_fold_02.wape == pytest.approx(0.25)


def test_comparison_refuses_missing_or_extra_validation_keys() -> None:
    missing_key = _lightgbm_predictions().iloc[:2].copy()

    with pytest.raises(LightGBMComparisonError, match="Validation observation keys differ"):
        build_lightgbm_comparison(
            _baseline_predictions(), missing_key, _fold_summary(), "trailing_mean_3"
        )
