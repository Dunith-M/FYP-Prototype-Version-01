import pandas as pd
import pytest

from ml.models.baselines import (
    last_observation_forecast,
    trailing_mean_3_forecast,
    trailing_mean_3_forecast_with_fallbacks,
    zero_forecast,
)


def test_zero_forecast_returns_zero_for_a_single_row() -> None:
    rows = pd.DataFrame({"target_next_month": [8]})

    prediction = zero_forecast(rows)

    assert prediction.tolist() == [0.0]
    assert prediction.name == "prediction"


def test_zero_forecast_returns_one_zero_per_input_row() -> None:
    rows = pd.DataFrame({"target_next_month": [0, 1, 8, 25, 3]})

    prediction = zero_forecast(rows)

    assert len(prediction) == len(rows)
    assert prediction.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_zero_forecast_preserves_input_row_order_and_index() -> None:
    rows = pd.DataFrame({"branch_code": ["B", "A", "C"]}, index=[8, 2, 11])

    prediction = zero_forecast(rows)

    assert prediction.index.tolist() == [8, 2, 11]
    assert prediction.tolist() == [0.0, 0.0, 0.0]


def test_zero_forecast_does_not_modify_input_data() -> None:
    rows = pd.DataFrame({"target_next_month": [0, 1, 8, 25], "feature": [4, 3, 2, 1]})
    original = rows.copy(deep=True)

    zero_forecast(rows)

    pd.testing.assert_frame_equal(rows, original)


def test_actual_target_values_do_not_change_zero_predictions() -> None:
    low_actuals = pd.DataFrame({"target_next_month": [0, 0, 0, 0]})
    high_actuals = pd.DataFrame({"target_next_month": [0, 3, 8, 20]})

    assert zero_forecast(low_actuals).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert zero_forecast(high_actuals).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_zero_forecast_is_deterministic_and_numeric() -> None:
    rows = pd.DataFrame({"feature": ["sparse", "positive", "zero"]})

    first = zero_forecast(rows)
    second = zero_forecast(rows)

    pd.testing.assert_series_equal(first, second)
    assert pd.api.types.is_numeric_dtype(first)


def test_zero_forecast_accepts_empty_input() -> None:
    rows = pd.DataFrame({"target_next_month": pd.Series(dtype="float64")})

    prediction = zero_forecast(rows)

    assert prediction.empty
    assert prediction.name == "prediction"
    assert pd.api.types.is_numeric_dtype(prediction)


def test_last_observation_forecast_returns_lag_1_for_a_single_row() -> None:
    rows = pd.DataFrame({"lag_1": [6.0], "target_next_month": [8.0]})

    prediction = last_observation_forecast(rows)

    assert prediction.tolist() == [6.0]
    assert prediction.name == "prediction"


def test_last_observation_forecast_returns_one_lag_1_value_per_row() -> None:
    rows = pd.DataFrame({"lag_1": [0.0, 2.0, 6.0, 15.0]})

    prediction = last_observation_forecast(rows)

    assert len(prediction) == len(rows)
    assert prediction.tolist() == [0.0, 2.0, 6.0, 15.0]


def test_last_observation_forecast_preserves_input_order_and_index() -> None:
    rows = pd.DataFrame({"lag_1": [3.0, 0.0, 12.0]}, index=[7, 1, 9])

    prediction = last_observation_forecast(rows)

    assert prediction.index.tolist() == [7, 1, 9]
    assert prediction.tolist() == [3.0, 0.0, 12.0]


def test_last_observation_uses_lag_1_not_deliberately_different_actuals() -> None:
    rows = pd.DataFrame({"lag_1": [1.0, 4.0, 7.0], "target_next_month": [50.0, 100.0, 0.0]})

    prediction = last_observation_forecast(rows)

    assert prediction.tolist() == [1.0, 4.0, 7.0]


def test_last_observation_forecast_does_not_modify_input_data() -> None:
    rows = pd.DataFrame({"lag_1": [2.0, 5.0, 10.0], "target_next_month": [100.0, 0.0, 999.0]})
    original = rows.copy(deep=True)

    last_observation_forecast(rows)

    pd.testing.assert_frame_equal(rows, original)


def test_last_observation_forecast_is_deterministic_and_numeric() -> None:
    rows = pd.DataFrame({"lag_1": [0.0, 3.0, 6.0, 12.0]})

    first = last_observation_forecast(rows)
    second = last_observation_forecast(rows)

    pd.testing.assert_series_equal(first, second)
    assert pd.api.types.is_numeric_dtype(first)


def test_last_observation_preserves_missing_lag_1_without_zero_fallback() -> None:
    rows = pd.DataFrame({"lag_1": [None, 3.0], "target_next_month": [8.0, 4.0]})

    prediction = last_observation_forecast(rows)

    assert pd.isna(prediction.iloc[0])
    assert prediction.iloc[1] == 3.0


def test_last_observation_rejects_rows_without_the_day_04_lag_feature() -> None:
    with pytest.raises(ValueError, match="lag_1"):
        last_observation_forecast(pd.DataFrame({"target_next_month": [8.0]}))


def test_trailing_mean_3_forecast_returns_feature_for_a_single_row() -> None:
    rows = pd.DataFrame({"trailing_mean_3": [6.0], "target_next_month": [8.0]})

    prediction = trailing_mean_3_forecast(rows)

    assert prediction.tolist() == [6.0]
    assert prediction.name == "prediction"


def test_trailing_mean_3_forecast_returns_one_feature_value_per_row() -> None:
    rows = pd.DataFrame({"trailing_mean_3": [0.0, 2.5, 6.0, 12.25]})

    prediction = trailing_mean_3_forecast(rows)

    assert len(prediction) == len(rows)
    assert prediction.tolist() == [0.0, 2.5, 6.0, 12.25]


def test_trailing_mean_3_forecast_preserves_input_order_and_index() -> None:
    rows = pd.DataFrame({"trailing_mean_3": [4.5, 0.0, 10.0]}, index=[8, 1, 11])

    prediction = trailing_mean_3_forecast(rows)

    assert prediction.index.tolist() == [8, 1, 11]
    assert prediction.tolist() == [4.5, 0.0, 10.0]


def test_trailing_mean_3_uses_feature_not_deliberately_different_actuals() -> None:
    rows = pd.DataFrame(
        {"trailing_mean_3": [2.0, 5.5, 8.0], "target_next_month": [50.0, 100.0, 0.0]}
    )

    prediction = trailing_mean_3_forecast(rows)

    assert prediction.tolist() == [2.0, 5.5, 8.0]


def test_trailing_mean_3_forecast_does_not_modify_input_data() -> None:
    rows = pd.DataFrame({"trailing_mean_3": [1.5, 4.0, 7.25], "target_next_month": [100.0, 0.0, 999.0]})
    original = rows.copy(deep=True)

    trailing_mean_3_forecast(rows)

    pd.testing.assert_frame_equal(rows, original)


def test_trailing_mean_3_forecast_is_deterministic_numeric_and_keeps_decimals() -> None:
    rows = pd.DataFrame({"trailing_mean_3": [3.6666667, 2.5]})

    first = trailing_mean_3_forecast(rows)
    second = trailing_mean_3_forecast(rows)

    pd.testing.assert_series_equal(first, second)
    assert first.tolist() == [3.6666667, 2.5]
    assert pd.api.types.is_numeric_dtype(first)


def test_trailing_mean_3_preserves_missing_history_without_fallback() -> None:
    rows = pd.DataFrame({"trailing_mean_3": [None, 3.0], "target_next_month": [8.0, 4.0]})

    prediction = trailing_mean_3_forecast(rows)

    assert pd.isna(prediction.iloc[0])
    assert prediction.iloc[1] == 3.0


def test_trailing_mean_3_rejects_rows_without_the_day_04_feature() -> None:
    with pytest.raises(ValueError, match="trailing_mean_3"):
        trailing_mean_3_forecast(pd.DataFrame({"target_next_month": [8.0]}))


def test_trailing_mean_fallbacks_prefer_a_valid_three_month_mean() -> None:
    rows = pd.DataFrame({"trailing_mean_3": [6.0], "lag_1": [9.0]})

    result = trailing_mean_3_forecast_with_fallbacks(rows)

    assert result.to_dict("records") == [
        {
            "model_name": "trailing_mean_3",
            "prediction": 6.0,
            "fallback_used": False,
            "fallback_reason": None,
        }
    ]


def test_trailing_mean_fallbacks_to_lag_1_when_history_is_insufficient() -> None:
    rows = pd.DataFrame({"trailing_mean_3": [None], "lag_1": [2.0]})

    result = trailing_mean_3_forecast_with_fallbacks(rows)

    assert result.loc[0, "prediction"] == 2.0
    assert bool(result.loc[0, "fallback_used"]) is True
    assert result.loc[0, "fallback_reason"] == "INSUFFICIENT_3_MONTH_HISTORY"


def test_trailing_mean_falls_back_to_zero_only_without_any_prior_observation() -> None:
    rows = pd.DataFrame({"trailing_mean_3": [None], "lag_1": [None]})

    result = trailing_mean_3_forecast_with_fallbacks(rows)

    assert result.loc[0, "prediction"] == 0.0
    assert bool(result.loc[0, "fallback_used"]) is True
    assert result.loc[0, "fallback_reason"] == "NO_PRIOR_OBSERVATION"


def test_zero_values_are_valid_history_not_missing_values() -> None:
    rows = pd.DataFrame(
        {"trailing_mean_3": [0.0, None], "lag_1": [8.0, 0.0]}
    )

    result = trailing_mean_3_forecast_with_fallbacks(rows)

    assert result["prediction"].tolist() == [0.0, 0.0]
    assert result["fallback_used"].tolist() == [False, True]
    assert result["fallback_reason"].tolist() == [None, "INSUFFICIENT_3_MONTH_HISTORY"]


def test_trailing_mean_fallbacks_return_expected_metadata_for_multiple_rows() -> None:
    rows = pd.DataFrame(
        {
            "trailing_mean_3": [6.0, None, None, 0.0],
            "lag_1": [9.0, 2.0, None, 7.0],
        }
    )

    result = trailing_mean_3_forecast_with_fallbacks(rows)

    assert result["model_name"].tolist() == ["trailing_mean_3"] * 4
    assert result["prediction"].tolist() == [6.0, 2.0, 0.0, 0.0]
    assert result["fallback_used"].tolist() == [False, True, True, False]
    assert result["fallback_reason"].tolist() == [
        None,
        "INSUFFICIENT_3_MONTH_HISTORY",
        "NO_PRIOR_OBSERVATION",
        None,
    ]


def test_trailing_mean_fallbacks_ignore_actual_targets_and_preserve_input() -> None:
    rows = pd.DataFrame(
        {
            "trailing_mean_3": [2.0, None, None, 0.0],
            "lag_1": [9.0, 5.5, None, 7.0],
            "target_next_month": [1000.0, 999.0, 500.0, 250.0],
        },
        index=[8, 1, 11, 3],
    )
    original = rows.copy(deep=True)

    result = trailing_mean_3_forecast_with_fallbacks(rows)

    assert result.index.tolist() == [8, 1, 11, 3]
    assert result["prediction"].tolist() == [2.0, 5.5, 0.0, 0.0]
    pd.testing.assert_frame_equal(rows, original)


def test_trailing_mean_fallbacks_are_deterministic() -> None:
    rows = pd.DataFrame({"trailing_mean_3": [None, 3.6666667], "lag_1": [0.0, 4.0]})

    first = trailing_mean_3_forecast_with_fallbacks(rows)
    second = trailing_mean_3_forecast_with_fallbacks(rows)

    pd.testing.assert_frame_equal(first, second)


def test_known_answer_normal_history_2_4_6_uses_precomputed_features() -> None:
    """The history is represented by Day-04 features, not recalculated here."""
    rows = pd.DataFrame(
        {"lag_1": [6.0], "trailing_mean_3": [4.0], "target_next_month": [999.0]}
    )

    zero_prediction = zero_forecast(rows)
    last_prediction = last_observation_forecast(rows)
    trailing_result = trailing_mean_3_forecast_with_fallbacks(rows)

    assert zero_prediction.tolist() == [0.0]
    assert last_prediction.tolist() == [6.0]
    assert trailing_result.loc[0, "prediction"] == 4.0
    assert bool(trailing_result.loc[0, "fallback_used"]) is False
    assert trailing_result.loc[0, "fallback_reason"] is None


def test_known_answer_empty_one_observation_and_all_zero_histories() -> None:
    rows = pd.DataFrame(
        {
            "lag_1": [None, 5.0, 0.0],
            "trailing_mean_3": [None, None, 0.0],
        }
    )

    zero_prediction = zero_forecast(rows)
    last_prediction = last_observation_forecast(rows)
    trailing_result = trailing_mean_3_forecast_with_fallbacks(rows)

    assert zero_prediction.tolist() == [0.0, 0.0, 0.0]
    assert pd.isna(last_prediction.iloc[0])
    assert last_prediction.iloc[1:].tolist() == [5.0, 0.0]
    assert trailing_result["prediction"].tolist() == [0.0, 5.0, 0.0]
    assert trailing_result["fallback_used"].tolist() == [True, True, False]
    assert trailing_result["fallback_reason"].tolist() == [
        "NO_PRIOR_OBSERVATION",
        "INSUFFICIENT_3_MONTH_HISTORY",
        None,
    ]


def test_absurd_actual_targets_cannot_change_any_baseline_prediction() -> None:
    rows = pd.DataFrame(
        {
            "lag_1": [2.0, 5.0, 8.0],
            "trailing_mean_3": [1.5, 4.0, 7.0],
            "target_next_month": [1000.0, 9999.0, 50000.0],
        }
    )

    assert zero_forecast(rows).tolist() == [0.0, 0.0, 0.0]
    assert last_observation_forecast(rows).tolist() == [2.0, 5.0, 8.0]
    assert trailing_mean_3_forecast(rows).tolist() == [1.5, 4.0, 7.0]
