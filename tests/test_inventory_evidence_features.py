from datetime import date

import math
import pandas as pd

from analytics.services.inventory_features import build_inventory_evidence_features


def _prediction(month="2026-03", branch="BIL01", product="CP_P001"):
    return {
        "branch_code": branch,
        "canonical_product_id": product,
        "year_month": month,
        "fulfilled_positive_units_proxy": 5,
    }


def _evidence(day, qty, state="AVAILABLE", branch="BIL01", product="CP_P001"):
    return {
        "branch_code": branch,
        "canonical_product_id": product,
        "business_date": day,
        "inventory_qty": qty,
        "availability_state": state,
    }


def test_last_mean_count_and_unknown_rate_use_only_prior_observed_evidence():
    prediction = pd.DataFrame([_prediction()])
    daily = pd.DataFrame([
        _evidence(date(2026, 2, 20), 5), _evidence(date(2026, 2, 25), 7),
        _evidence(date(2026, 2, 27), None, "UNKNOWN"), _evidence(date(2026, 2, 28), 4),
    ])
    row = build_inventory_evidence_features(prediction, daily).features.iloc[0]
    assert row.last_observed_qty == 4
    assert math.isclose(row.mean_observed_qty_30d, 16 / 3)
    assert row.inventory_observation_count == 3
    assert math.isclose(row.availability_unknown_rate, 1 / 4)
    assert pd.isna(daily.loc[2, "inventory_qty"])


def test_prediction_month_and_future_values_cannot_change_prediction_features():
    prediction = pd.DataFrame([_prediction()])
    base = pd.DataFrame([_evidence(date(2026, 2, 28), 4)])
    changed = pd.concat([
        base,
        pd.DataFrame([_evidence(date(2026, 3, 1), 999), _evidence(date(2026, 4, 1), 9999)]),
    ], ignore_index=True)
    first = build_inventory_evidence_features(prediction, base).features
    second = build_inventory_evidence_features(prediction, changed).features
    columns = ["last_observed_qty", "mean_observed_qty_30d", "inventory_observation_count", "availability_unknown_rate"]
    pd.testing.assert_frame_equal(first[columns], second[columns])


def test_unknown_is_not_zero_and_all_unknown_history_remains_unobserved():
    prediction = pd.DataFrame([_prediction()])
    mixed = pd.DataFrame([
        _evidence(date(2026, 2, 26), 4), _evidence(date(2026, 2, 27), None, "UNKNOWN"),
        _evidence(date(2026, 2, 28), 8),
    ])
    row = build_inventory_evidence_features(prediction, mixed).features.iloc[0]
    assert row.mean_observed_qty_30d == 6
    assert row.inventory_observation_count == 2
    assert math.isclose(row.availability_unknown_rate, 1 / 3)

    unknown = pd.DataFrame([
        _evidence(date(2026, 2, 27), None, "UNKNOWN"), _evidence(date(2026, 2, 28), None, "UNKNOWN"),
    ])
    all_unknown = build_inventory_evidence_features(prediction, unknown).features.iloc[0]
    assert pd.isna(all_unknown.last_observed_qty)
    assert pd.isna(all_unknown.mean_observed_qty_30d)
    assert all_unknown.inventory_observation_count == 0
    assert all_unknown.availability_unknown_rate == 1


def test_branch_product_isolation_and_row_preservation():
    predictions = pd.DataFrame([_prediction(), _prediction(branch="BIN01"), _prediction(product="CP_P002")])
    daily = pd.DataFrame([
        _evidence(date(2026, 2, 28), 4), _evidence(date(2026, 2, 28), 100, branch="BIN01"),
        _evidence(date(2026, 2, 28), 200, product="CP_P002"),
    ])
    first = build_inventory_evidence_features(predictions, daily).features
    second = build_inventory_evidence_features(predictions, daily).features
    assert len(first) == len(predictions)
    assert first.loc[
        (first.branch_code == "BIL01") & (first.canonical_product_id == "CP_P001"),
        "last_observed_qty",
    ].iloc[0] == 4
    assert first.loc[
        (first.branch_code == "BIN01") & (first.canonical_product_id == "CP_P001"),
        "last_observed_qty",
    ].iloc[0] == 100
    assert first.loc[first.canonical_product_id == "CP_P002", "last_observed_qty"].iloc[0] == 200
    pd.testing.assert_frame_equal(first, second)
