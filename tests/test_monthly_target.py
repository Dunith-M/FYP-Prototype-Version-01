import pandas as pd

from analytics.services.targets import TARGET_NAME, build_monthly_fulfilled_positive_units_target


def _frame(rows):
    return pd.DataFrame(rows)


def test_positive_zero_and_negative_transactions_form_proxy_without_mutation():
    source = _frame([
        {"branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2025-01", "raw_total_units": 5},
        {"branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2025-01", "raw_total_units": 3},
        {"branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2025-01", "raw_total_units": 0},
        {"branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2025-01", "raw_total_units": -2},
    ])
    result = build_monthly_fulfilled_positive_units_target(source)
    assert result.monthly_target.loc[0, TARGET_NAME] == 8
    assert source["raw_total_units"].tolist() == [5, 3, 0, -2]
    assert (result.metrics.positive_source_observations, result.metrics.zero_source_observations, result.metrics.negative_source_observations) == (2, 1, 1)


def test_month_branch_and_product_keys_remain_separate_and_deterministic():
    source = _frame([
        {"branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2025-01", "raw_total_units": 2},
        {"branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2025-01", "raw_total_units": 4},
        {"branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2025-02", "raw_total_units": 7},
        {"branch_code": "BIN01", "canonical_product_id": "CP_A", "year_month": "2025-01", "raw_total_units": 8},
        {"branch_code": "BIL01", "canonical_product_id": "CP_B", "year_month": "2025-01", "raw_total_units": 9},
    ])
    first = build_monthly_fulfilled_positive_units_target(source)
    second = build_monthly_fulfilled_positive_units_target(source)
    assert len(first.monthly_target) == 4
    assert first.monthly_target[TARGET_NAME].tolist() == [6, 7, 9, 8]
    pd.testing.assert_frame_equal(first.monthly_target, second.monthly_target)
    assert "true_demand" not in first.monthly_target.columns


def test_daily_positive_contribution_preserves_mixed_transaction_evidence():
    daily = _frame([{
        "branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2025-01",
        "raw_total_units": 4, "fulfilled_positive_units_proxy_day": 5,
        "positive_unit_row_count": 1, "zero_unit_row_count": 0, "negative_unit_row_count": 1,
    }])
    result = build_monthly_fulfilled_positive_units_target(daily)
    assert result.monthly_target.loc[0, TARGET_NAME] == 5
    assert daily.loc[0, "raw_total_units"] == 4
    assert result.metrics.negative_source_observations == 1


def test_unresolved_audit_is_counted_but_not_mapped_or_included():
    mapped = _frame([{"branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2025-01", "raw_total_units": 5}])
    unresolved = _frame([
        {"source_product_code": "OLD001", "canonical_product_id": None, "TOTAL_UNITS": 3},
        {"source_product_code": "OLD001", "canonical_product_id": None, "TOTAL_UNITS": -1},
    ])
    result = build_monthly_fulfilled_positive_units_target(mapped, unresolved)
    assert result.monthly_target.loc[0, TARGET_NAME] == 5
    assert result.metrics.unresolved_excluded_rows == 2
    assert result.metrics.unresolved_excluded_positive_units == 3
    assert unresolved["canonical_product_id"].isna().all()


def test_unresolved_row_in_analytical_input_is_explicitly_excluded():
    source = _frame([
        {"branch_code": "BIL01", "canonical_product_id": "CP_A", "year_month": "2025-01", "raw_total_units": 5},
        {"branch_code": "BIL01", "canonical_product_id": None, "year_month": "2025-01", "raw_total_units": 3},
    ])

    result = build_monthly_fulfilled_positive_units_target(source)

    assert result.monthly_target.loc[0, TARGET_NAME] == 5
    assert result.metrics.unresolved_excluded_rows == 1
    assert result.metrics.unresolved_excluded_positive_units == 3
    assert pd.isna(source.loc[1, "canonical_product_id"])
