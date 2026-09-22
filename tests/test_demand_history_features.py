import math

import pandas as pd
import pytest

from analytics.services.feature_engineering import (
    DemandHistoryFeatureValidationError,
    build_demand_history_features,
    validate_demand_history_feature_leakage,
)


def _row(month, target, branch="BIL01", product="CP_P001"):
    return {
        "branch_code": branch,
        "canonical_product_id": product,
        "year_month": month,
        "fulfilled_positive_units_proxy": target,
    }


def test_lags_and_shifted_rolling_values_do_not_use_current_target():
    source = pd.DataFrame([
        _row("2024-11", 5), _row("2024-12", 8), _row("2025-01", 4), _row("2025-02", 9),
    ])
    result = build_demand_history_features(source).features
    february = result.loc[result.year_month == "2025-02"].iloc[0]
    assert (february.lag_1, february.lag_2, february.lag_3) == (4, 8, 5)
    assert math.isclose(february.trailing_mean_3, 17 / 3)
    assert february.trailing_sum_3 == 17
    assert february.lag_1 != february.fulfilled_positive_units_proxy
    assert source.fulfilled_positive_units_proxy.tolist() == [5, 8, 4, 9]


def test_anti_leakage_current_and_future_target_changes_do_not_change_march_features():
    history = pd.DataFrame([
        _row("2024-12", 5), _row("2025-01", 8), _row("2025-02", 4),
        _row("2025-03", 9), _row("2025-04", 10),
    ])
    changed = history.copy(deep=True)
    changed.loc[changed.year_month == "2025-03", "fulfilled_positive_units_proxy"] = 9999
    changed.loc[changed.year_month == "2025-04", "fulfilled_positive_units_proxy"] = 50000

    baseline = build_demand_history_features(history).features
    altered = build_demand_history_features(changed).features
    columns = ["lag_1", "lag_2", "lag_3", "trailing_mean_3", "trailing_sum_3"]
    baseline_march = baseline.loc[baseline.year_month == "2025-03", columns].reset_index(drop=True)
    altered_march = altered.loc[altered.year_month == "2025-03", columns].reset_index(drop=True)

    pd.testing.assert_frame_equal(baseline_march, altered_march)
    assert altered_march.iloc[0].to_dict() == {
        "lag_1": 4.0, "lag_2": 8.0, "lag_3": 5.0,
        "trailing_mean_3": 17 / 3, "trailing_sum_3": 17.0,
    }


def test_early_history_remains_null_and_branch_product_series_are_isolated():
    source = pd.DataFrame([
        _row("2025-01", 5), _row("2025-02", 6), _row("2025-03", 7),
        _row("2025-01", 100, branch="BIN01"), _row("2025-01", 200, product="CP_P002"),
    ])
    result = build_demand_history_features(source).features
    bil = result[(result.branch_code == "BIL01") & (result.canonical_product_id == "CP_P001")]
    assert bil.iloc[0][["lag_1", "lag_2", "lag_3"]].isna().all()
    assert bil.iloc[1].lag_1 == 5 and pd.isna(bil.iloc[1].lag_2)
    assert bil.iloc[2].lag_1 == 6 and bil.iloc[2].lag_2 == 5 and pd.isna(bil.iloc[2].lag_3)
    assert pd.isna(result.loc[result.branch_code == "BIN01", "lag_1"].iloc[0])
    assert pd.isna(result.loc[result.canonical_product_id == "CP_P002", "lag_1"].iloc[0])


def test_prior_positive_counts_and_intermittency_exclude_current_month():
    source = pd.DataFrame([
        _row("2025-01", 5), _row("2025-02", 0), _row("2025-03", 3), _row("2025-04", 0), _row("2025-05", 9),
    ])
    result = build_demand_history_features(source).features
    april = result.loc[result.year_month == "2025-04"].iloc[0]
    may = result.loc[result.year_month == "2025-05"].iloc[0]
    assert april.months_with_positive_sales == 2
    assert math.isclose(may.intermittency, 0.5)
    assert may.months_with_positive_sales == 2


def test_all_positive_and_all_zero_prior_history_have_transparent_intermittency():
    positive = build_demand_history_features(pd.DataFrame([
        _row("2025-01", 5), _row("2025-02", 4), _row("2025-03", 2), _row("2025-04", 1),
    ])).features
    zeros = build_demand_history_features(pd.DataFrame([
        _row("2025-01", 0), _row("2025-02", 0), _row("2025-03", 0), _row("2025-04", 1),
    ])).features
    assert positive.iloc[-1].intermittency == 0
    assert zeros.iloc[-1].intermittency == 1


def test_missing_months_are_not_densified_and_unresolved_audit_is_preserved():
    source = pd.DataFrame([_row("2025-01", 5), _row("2025-03", 7)])
    unresolved = pd.DataFrame([{"canonical_product_id": None, "TOTAL_UNITS": 3}])
    first = build_demand_history_features(source, unresolved)
    second = build_demand_history_features(source, unresolved)
    assert len(first.features) == 2
    assert first.features.iloc[1].lag_1 == 5
    assert first.metrics.calendar_gap_edges == 1
    assert first.metrics.unresolved_rows_excluded == 1
    pd.testing.assert_frame_equal(first.features, second.features)


def test_sorting_duplicate_key_validation_and_reusable_leakage_audit():
    source = pd.DataFrame([
        _row("2025-03", 4), _row("2025-01", 5), _row("2025-02", 8), _row("2025-04", 9),
    ])
    features = build_demand_history_features(source).features
    april = features.loc[features.year_month == "2025-04"].iloc[0]
    assert (april.lag_1, april.lag_2, april.lag_3) == (4, 8, 5)
    validate_demand_history_feature_leakage(features)

    corrupted = features.copy(deep=True)
    corrupted.loc[corrupted.year_month == "2025-04", "lag_1"] = 9
    with pytest.raises(DemandHistoryFeatureValidationError, match="Leakage audit failed"):
        validate_demand_history_feature_leakage(corrupted)

    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(DemandHistoryFeatureValidationError, match="duplicate"):
        build_demand_history_features(duplicate)
