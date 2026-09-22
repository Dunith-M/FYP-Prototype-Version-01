from datetime import date

import pandas as pd
import pytest

from analytics.services.context_features import ContextFeatureValidationError, enrich_monthly_features


def _monthly():
    return pd.DataFrame([
        {"branch_code": "BIL01", "canonical_product_id": "CP_1", "year_month": "2026-03", "fulfilled_positive_units_proxy": 5},
        {"branch_code": "BIN01", "canonical_product_id": "CP_2", "year_month": "2026-04", "fulfilled_positive_units_proxy": 7},
    ])


def _products():
    return pd.DataFrame([
        {"canonical_product_id": "CP_1", "family_id": "F1", "family_desc": "Family 1", "list_price": 100, "price_valid": True},
        {"canonical_product_id": "CP_2", "family_id": None, "family_desc": None, "list_price": 0, "price_valid": False},
    ])


def _calendar():
    return pd.DataFrame([
        {"calendar_date": date(2026, 3, 1), "year_month": "2026-03", "IS_WEEKEND": True, "IS_PUBLIC_HOLIDAY": False},
        {"calendar_date": date(2026, 3, 2), "year_month": "2026-03", "IS_WEEKEND": False, "IS_PUBLIC_HOLIDAY": True},
        {"calendar_date": date(2026, 4, 4), "year_month": "2026-04", "IS_WEEKEND": True, "IS_PUBLIC_HOLIDAY": False},
        {"calendar_date": date(2026, 4, 5), "year_month": "2026-04", "IS_WEEKEND": True, "IS_PUBLIC_HOLIDAY": False},
    ])


def _branches():
    return pd.DataFrame([
        {"branch_code": "BIL01", "branch_name": "Galle"},
        {"branch_code": "BIN01", "branch_name": "Ambalangoda"},
    ])


def test_context_features_join_safely_and_preserve_monthly_rows():
    source = _monthly()
    result = enrich_monthly_features(source, _products(), _calendar(), _branches())
    output = result.features
    assert len(output) == len(source)
    assert output.duplicated(["branch_code", "canonical_product_id", "year_month"]).sum() == 0
    bil = output.loc[output.branch_code == "BIL01"].iloc[0]
    assert (bil.family_id, bil.family_desc, bil.valid_price, bil.branch_name) == ("F1", "Family 1", True, "Galle")
    assert bil.price_bucket.startswith("PRICE_Q")
    assert (bil.month, bil.quarter, bil.holiday_count, bil.weekend_count) == (3, 1, 1, 1)
    assert source.columns.tolist() == ["branch_code", "canonical_product_id", "year_month", "fulfilled_positive_units_proxy"]


def test_invalid_price_is_flagged_without_a_misleading_bucket_and_calendar_counts_are_monthly():
    result = enrich_monthly_features(_monthly(), _products(), _calendar(), _branches())
    invalid = result.features.loc[result.features.canonical_product_id == "CP_2"].iloc[0]
    assert bool(invalid.valid_price) is False
    assert invalid.price_bucket == "INVALID_OR_MISSING_PRICE"
    assert (invalid.month, invalid.quarter, invalid.holiday_count, invalid.weekend_count) == (4, 2, 0, 2)
    assert result.metrics.missing_family_rows == 1
    assert not {"cost", "margin", "profit"} & set(result.features.columns)


def test_duplicate_product_or_branch_masters_fail_instead_of_multiplying_rows():
    duplicate_product = pd.concat([_products(), _products().iloc[[0]]], ignore_index=True)
    with pytest.raises(ContextFeatureValidationError, match="Product master has duplicate"):
        enrich_monthly_features(_monthly(), duplicate_product, _calendar(), _branches())
    duplicate_branch = pd.concat([_branches(), _branches().iloc[[0]]], ignore_index=True)
    with pytest.raises(ContextFeatureValidationError, match="Branch master has duplicate"):
        enrich_monthly_features(_monthly(), _products(), _calendar(), duplicate_branch)


def test_daily_calendar_is_aggregated_before_join_and_duplicate_dates_fail():
    duplicate_day = pd.concat([_calendar(), _calendar().iloc[[0]]], ignore_index=True)
    with pytest.raises(ContextFeatureValidationError, match="Calendar has duplicate"):
        enrich_monthly_features(_monthly(), _products(), duplicate_day, _branches())
