"""Mandatory Day-04 temporal leakage gate using the real feature services."""

from datetime import date
import math

import pandas as pd
import pytest

from analytics.services.feature_engineering import (
    DemandHistoryFeatureValidationError,
    build_demand_history_features,
    validate_demand_history_feature_leakage,
)
from analytics.services.inventory_features import build_inventory_evidence_features
from analytics.services.target_alignment import align_next_month_target


HISTORY_COLUMNS = (
    "lag_1", "lag_2", "lag_3", "trailing_mean_3", "trailing_sum_3",
    "months_with_positive_sales", "intermittency",
)
INVENTORY_COLUMNS = (
    "last_observed_qty", "mean_observed_qty_30d", "inventory_observation_count", "availability_unknown_rate",
)


def _monthly(month: str, units: float, branch: str = "BIL01", product: str = "CP_1") -> dict[str, object]:
    return {
        "branch_code": branch,
        "canonical_product_id": product,
        "year_month": month,
        "fulfilled_positive_units_proxy": units,
    }


def _inventory(day: date, qty: float | None, state: str = "AVAILABLE", branch: str = "BIL01", product: str = "CP_1") -> dict[str, object]:
    return {
        "branch_code": branch,
        "canonical_product_id": product,
        "business_date": day,
        "inventory_qty": qty,
        "availability_state": state,
    }


def _march_row(frame: pd.DataFrame) -> pd.Series:
    return frame.loc[(frame.branch_code == "BIL01") & (frame.canonical_product_id == "CP_1") & (frame.year_month == "2026-03")].iloc[0]


def test_lags_and_shifted_rollings_use_only_months_before_march():
    source = pd.DataFrame([
        _monthly("2025-12", 5), _monthly("2026-01", 8), _monthly("2026-02", 4), _monthly("2026-03", 9),
    ])
    features = build_demand_history_features(source).features
    march = _march_row(features)
    assert (march.lag_1, march.lag_2, march.lag_3) == (4, 8, 5)
    assert math.isclose(march.trailing_mean_3, 17 / 3)
    assert march.trailing_sum_3 == 17
    validate_demand_history_feature_leakage(features)


def test_current_and_future_target_mutations_cannot_change_march_history_features():
    source = pd.DataFrame([
        _monthly("2025-12", 5), _monthly("2026-01", 8), _monthly("2026-02", 4),
        _monthly("2026-03", 9), _monthly("2026-04", 10),
    ])
    changed = source.copy(deep=True)
    changed.loc[changed.year_month == "2026-03", "fulfilled_positive_units_proxy"] = 999999
    changed.loc[changed.year_month == "2026-04", "fulfilled_positive_units_proxy"] = 1000000
    first = build_demand_history_features(source).features
    second = build_demand_history_features(changed).features
    pd.testing.assert_frame_equal(
        _march_row(first).loc[list(HISTORY_COLUMNS)].to_frame().T.reset_index(drop=True),
        _march_row(second).loc[list(HISTORY_COLUMNS)].to_frame().T.reset_index(drop=True),
    )


def test_prior_positive_count_and_intermittency_exclude_prediction_month():
    source = pd.DataFrame([
        _monthly("2025-11", 5), _monthly("2025-12", 0), _monthly("2026-01", 4), _monthly("2026-02", 9),
    ])
    features = build_demand_history_features(source).features
    february = features.loc[features.year_month == "2026-02"].iloc[0]
    assert february.months_with_positive_sales == 2

    intermittency_source = pd.DataFrame([
        _monthly("2025-10", 5), _monthly("2025-11", 0), _monthly("2025-12", 3),
        _monthly("2026-01", 0), _monthly("2026-02", 9),
    ])
    changed = intermittency_source.copy(deep=True)
    changed.loc[changed.year_month == "2026-02", "fulfilled_positive_units_proxy"] = 999999
    baseline = build_demand_history_features(intermittency_source).features
    altered = build_demand_history_features(changed).features
    assert math.isclose(baseline.loc[baseline.year_month == "2026-02", "intermittency"].iloc[0], 0.5)
    assert baseline.loc[baseline.year_month == "2026-02", "intermittency"].iloc[0] == altered.loc[altered.year_month == "2026-02", "intermittency"].iloc[0]


def test_group_isolation_sorting_early_history_and_duplicate_key_protection():
    source = pd.DataFrame([
        _monthly("2026-03", 9), _monthly("2025-12", 5), _monthly("2026-02", 4), _monthly("2026-01", 8),
        _monthly("2025-12", 500, branch="BIN01"), _monthly("2026-01", 800, branch="BIN01"),
        _monthly("2025-12", 5000, product="CP_2"), _monthly("2026-01", 8000, product="CP_2"),
    ])
    features = build_demand_history_features(source).features
    march = _march_row(features)
    assert (march.lag_1, march.lag_2, march.lag_3) == (4, 8, 5)
    first = features.loc[(features.branch_code == "BIL01") & (features.canonical_product_id == "CP_1")].iloc[0]
    second = features.loc[(features.branch_code == "BIL01") & (features.canonical_product_id == "CP_1")].iloc[1]
    assert first[["lag_1", "lag_2", "lag_3"]].isna().all()
    assert second.lag_1 == 5 and pd.isna(second.lag_2) and pd.isna(second.lag_3)
    with pytest.raises(DemandHistoryFeatureValidationError, match="duplicate"):
        build_demand_history_features(pd.concat([source, source.iloc[[0]]], ignore_index=True))


def test_target_points_forward_only_to_same_series_next_consecutive_calendar_month():
    source = pd.DataFrame([
        _monthly("2025-12", 5), _monthly("2026-01", 8), _monthly("2026-02", 4), _monthly("2026-03", 9),
        _monthly("2026-01", 500, branch="BIN01"), _monthly("2026-02", 800, branch="BIN01"),
        _monthly("2026-01", 5000, product="CP_2"), _monthly("2026-02", 8000, product="CP_2"),
    ])
    history = build_demand_history_features(source).features
    assert "target_next_month" not in history.columns
    aligned = align_next_month_target(history).features
    january = aligned.loc[(aligned.branch_code == "BIL01") & (aligned.canonical_product_id == "CP_1") & (aligned.year_month == "2026-01")].iloc[0]
    february = aligned.loc[(aligned.branch_code == "BIL01") & (aligned.canonical_product_id == "CP_1") & (aligned.year_month == "2026-02")].iloc[0]
    assert (january.target_next_month, january.target_year_month) == (4, "2026-02")
    assert (february.target_next_month, february.target_year_month) == (9, "2026-03")
    assert january.target_next_month not in (5, 8)
    assert aligned.loc[(aligned.branch_code == "BIN01") & (aligned.year_month == "2026-01"), "target_next_month"].iloc[0] == 800
    assert aligned.loc[(aligned.canonical_product_id == "CP_2") & (aligned.year_month == "2026-01"), "target_next_month"].iloc[0] == 8000


def test_target_gap_is_unavailable_and_never_masquerades_as_next_month():
    history = build_demand_history_features(pd.DataFrame([_monthly("2026-01", 5), _monthly("2026-03", 9)])).features
    january = align_next_month_target(history).features.loc[lambda frame: frame.year_month == "2026-01"].iloc[0]
    assert pd.isna(january.target_next_month)
    assert pd.isna(january.target_year_month)
    assert bool(january.target_available_flag) is False


def test_inventory_features_use_only_dates_before_prediction_month_and_preserve_unknown():
    prediction = pd.DataFrame([_monthly("2026-03", 5)])
    prior = pd.DataFrame([
        _inventory(date(2026, 2, 26), 4), _inventory(date(2026, 2, 27), None, "UNKNOWN"),
        _inventory(date(2026, 2, 28), 8),
    ])
    contaminated = pd.concat([
        prior,
        pd.DataFrame([_inventory(date(2026, 3, 1), 999999), _inventory(date(2026, 4, 1), 1000000)]),
    ], ignore_index=True)
    first = build_inventory_evidence_features(prediction, prior).features
    second = build_inventory_evidence_features(prediction, contaminated).features
    pd.testing.assert_frame_equal(first.loc[:, INVENTORY_COLUMNS], second.loc[:, INVENTORY_COLUMNS])
    row = first.iloc[0]
    assert (row.last_observed_qty, row.mean_observed_qty_30d, row.inventory_observation_count) == (8, 6, 2)
    assert math.isclose(row.availability_unknown_rate, 1 / 3)
