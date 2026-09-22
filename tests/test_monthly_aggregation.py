from datetime import date

import pandas as pd

from analytics.services.monthly_aggregation import build_branch_product_month


def _daily_row(**overrides):
    row = {
        "branch_code": "BIL01",
        "canonical_product_id": "CP_P001",
        "business_date": date(2025, 8, 1),
        "raw_total_units": 2,
        "fulfilled_positive_units_proxy_day": 2,
        "source_row_count": 1,
        "negative_unit_row_count": 0,
        "zero_unit_row_count": 0,
        "availability_state": "AVAILABLE",
        "inventory_qty": 4,
    }
    row.update(overrides)
    return row


def test_basic_monthly_sum_and_information_preserving_counts():
    daily = pd.DataFrame([
        _daily_row(business_date=date(2025, 8, 1), fulfilled_positive_units_proxy_day=2),
        _daily_row(business_date=date(2025, 8, 3), fulfilled_positive_units_proxy_day=1),
        _daily_row(business_date=date(2025, 8, 17), fulfilled_positive_units_proxy_day=4),
    ])
    result = build_branch_product_month(daily)
    row = result.branch_product_month.iloc[0]
    assert row.fulfilled_positive_units_proxy == 7
    assert row.sales_observation_count == 3
    assert row.year_month == "2025-08"
    assert daily.fulfilled_positive_units_proxy_day.tolist() == [2, 1, 4]


def test_negative_and_zero_events_do_not_change_proxy_and_remain_diagnostic():
    daily = pd.DataFrame([
        _daily_row(fulfilled_positive_units_proxy_day=5),
        _daily_row(business_date=date(2025, 8, 2), fulfilled_positive_units_proxy_day=0, negative_unit_row_count=1),
        _daily_row(business_date=date(2025, 8, 3), fulfilled_positive_units_proxy_day=0, zero_unit_row_count=2),
    ])
    result = build_branch_product_month(daily)
    row = result.branch_product_month.iloc[0]
    assert row.fulfilled_positive_units_proxy == 5
    assert row.negative_event_count == 1
    assert row.zero_event_count == 2


def test_month_branch_and_product_keys_are_separate_and_deterministic():
    daily = pd.DataFrame([
        _daily_row(fulfilled_positive_units_proxy_day=7),
        _daily_row(business_date=date(2025, 9, 1), fulfilled_positive_units_proxy_day=3),
        _daily_row(branch_code="BIN01", fulfilled_positive_units_proxy_day=4),
        _daily_row(canonical_product_id="CP_P002", fulfilled_positive_units_proxy_day=10),
    ])
    first = build_branch_product_month(daily)
    second = build_branch_product_month(daily)
    assert len(first.branch_product_month) == 4
    assert not first.branch_product_month.duplicated(
        ["branch_code", "canonical_product_id", "year_month"]
    ).any()
    pd.testing.assert_frame_equal(first.branch_product_month, second.branch_product_month)


def test_unknown_inventory_is_not_an_observation_or_zero_quantity():
    daily = pd.DataFrame([
        _daily_row(availability_state="AVAILABLE", inventory_qty=3),
        _daily_row(business_date=date(2025, 8, 2), availability_state="UNKNOWN", inventory_qty=None),
        _daily_row(business_date=date(2025, 8, 3), availability_state="AVAILABLE", inventory_qty=1),
    ])
    result = build_branch_product_month(daily)
    row = result.branch_product_month.iloc[0]
    assert row.inventory_observation_count == 2
    assert row.inventory_unknown_count == 1
    assert pd.isna(daily.loc[1, "inventory_qty"])


def test_unresolved_audit_is_explicitly_excluded_without_changing_audit_rows():
    daily = pd.DataFrame([_daily_row(fulfilled_positive_units_proxy_day=5)])
    unresolved = pd.DataFrame([
        {"canonical_product_id": None, "TOTAL_UNITS": 3, "source_product_code": "OLD001"}
    ])
    result = build_branch_product_month(daily, unresolved)
    assert len(result.branch_product_month) == 1
    assert result.metrics.unresolved_rows_excluded == 1
    assert pd.isna(unresolved.loc[0, "canonical_product_id"])
