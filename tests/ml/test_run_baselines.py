import pandas as pd

from ml.evaluation.run_baselines import evaluate_all_folds


def _feature_rows() -> pd.DataFrame:
    months = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03"]
    return pd.DataFrame(
        {
            "branch_code": ["BIL01"] * len(months),
            "canonical_product_id": ["CP_1"] * len(months),
            "family_id": ["PF_1"] * len(months),
            "target_year_month": months,
            "target_next_month": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "lag_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "trailing_mean_3": [1.0, 2.0, 3.0, None, 5.0, 6.0],
            "target_available_flag": [True] * len(months),
        }
    )


def test_runner_uses_frozen_folds_and_generates_three_models_per_validation_row() -> None:
    result = evaluate_all_folds(_feature_rows())

    assert len(result.predictions) == 9
    assert set(result.predictions["target_month"]) == {"2025-12", "2026-01", "2026-02"}
    assert "2026-03" not in set(result.predictions["target_month"])
    assert set(result.predictions["model_name"]) == {
        "zero_forecast",
        "last_observation",
        "trailing_mean_3",
    }
    assert set(result.metrics["scope"]) == {"OVERALL", "FOLD", "BRANCH", "FAMILY"}
    assert len(result.metrics.loc[result.metrics["scope"] == "FOLD"]) == 9


def test_runner_retains_trailing_mean_fallback_metadata() -> None:
    result = evaluate_all_folds(_feature_rows())
    january_trailing = result.predictions.loc[
        (result.predictions["fold_id"] == "fold_02")
        & (result.predictions["model_name"] == "trailing_mean_3")
    ].iloc[0]

    assert january_trailing.predicted_units == 4.0
    assert bool(january_trailing.fallback_used) is True
    assert january_trailing.fallback_reason == "INSUFFICIENT_3_MONTH_HISTORY"
    assert january_trailing.absolute_error == 0.0
