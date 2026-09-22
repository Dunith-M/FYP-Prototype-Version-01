import math

import pandas as pd
import pytest

from analytics.services.target_alignment import NextMonthTargetValidationError, align_next_month_target


def _row(month, target, branch="BIL01", product="CP_1"):
    return {
        "branch_code": branch,
        "canonical_product_id": product,
        "year_month": month,
        "fulfilled_positive_units_proxy": target,
        "lag_1": 8,
        "lag_2": 5,
        "lag_3": None,
        "trailing_mean_3": None,
        "trailing_sum_3": None,
        "last_observed_qty": 4,
        "family_id": "F1",
    }


def test_next_month_target_uses_shift_minus_one_within_sorted_branch_product_series():
    source = pd.DataFrame([_row("2026-03", 9), _row("2025-12", 7), _row("2026-02", 4), _row("2026-01", 8)])
    output = align_next_month_target(source).features
    assert output.year_month.tolist() == ["2025-12", "2026-01", "2026-02", "2026-03"]
    assert output.target_next_month.tolist()[:3] == [8, 4, 9]
    assert pd.isna(output.target_next_month.iloc[3])
    assert output.target_year_month.tolist()[:3] == ["2026-01", "2026-02", "2026-03"]
    assert output.target_available_flag.tolist() == [True, True, True, False]


def test_branch_product_isolation_and_missing_calendar_month_are_safe():
    source = pd.DataFrame([
        _row("2026-01", 5), _row("2026-03", 9),
        _row("2026-01", 100, branch="BIN01"), _row("2026-02", 200, branch="BIN01"),
        _row("2026-01", 300, product="CP_2"), _row("2026-02", 400, product="CP_2"),
    ])
    result = align_next_month_target(source)
    output = result.features
    bil_january = output.loc[(output.branch_code == "BIL01") & (output.year_month == "2026-01")].iloc[0]
    assert pd.isna(bil_january.target_next_month)
    assert bool(bil_january.target_available_flag) is False
    assert output.loc[(output.branch_code == "BIN01") & (output.year_month == "2026-01"), "target_next_month"].iloc[0] == 200
    assert output.loc[(output.canonical_product_id == "CP_2") & (output.year_month == "2026-01"), "target_next_month"].iloc[0] == 400
    assert result.metrics.non_consecutive_month_pairs == 1


def test_target_alignment_preserves_feature_values_and_future_target_does_not_contaminate_features():
    source = pd.DataFrame([_row("2025-11", 5), _row("2025-12", 8), _row("2026-01", 4), _row("2026-02", 9)])
    changed = source.copy(deep=True)
    changed.loc[changed.year_month == "2026-02", "fulfilled_positive_units_proxy"] = 9999
    first = align_next_month_target(source).features
    second = align_next_month_target(changed).features
    columns = ["lag_1", "lag_2", "lag_3", "trailing_mean_3", "trailing_sum_3", "last_observed_qty", "family_id"]
    january_first = first.loc[first.year_month == "2026-01", columns].reset_index(drop=True)
    january_second = second.loc[second.year_month == "2026-01", columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(january_first, january_second)
    assert first.loc[first.year_month == "2026-01", "target_next_month"].iloc[0] == 9
    assert second.loc[second.year_month == "2026-01", "target_next_month"].iloc[0] == 9999
    assert source.fulfilled_positive_units_proxy.tolist() == [5, 8, 4, 9]


def test_duplicate_monthly_key_fails_and_output_is_deterministic():
    source = pd.DataFrame([_row("2026-01", 5), _row("2026-02", 7)])
    first = align_next_month_target(source).features
    second = align_next_month_target(source).features
    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(NextMonthTargetValidationError, match="duplicate branch/product/month"):
        align_next_month_target(pd.concat([source, source.iloc[[0]]], ignore_index=True))
    assert math.isnan(first.loc[first.year_month == "2026-02", "target_next_month"].iloc[0])
