import pandas as pd
from pandas.testing import assert_frame_equal

from ingestion.validators import (
    validate_branches,
    validate_calendar,
    validate_inventory,
    validate_products,
    validate_sales,
)


def result_for(results, rule_id):
    return next(result for result in results if result.rule_id == rule_id)


def valid_sales_frame():
    return pd.DataFrame(
        {
            "SALES_DATE": ["2025-04-01"],
            "SHOP_CODE": ["SBO01"],
            "SALES_PART": ["PART-001"],
            "TOTAL_UNITS": [1],
            "TOTAL_VALUE": [100.0],
        }
    )


def valid_product_frame():
    return pd.DataFrame(
        {
            "PRODUCT_CODE": ["PART-001"],
            "PRODUCT_DESC": ["Example product"],
            "PRODUCT_FAMILY_ID": ["PF001"],
            "PRODUCT_FAMILY_DESC": ["Example family"],
            "PRICE": [100.0],
        }
    )


def valid_inventory_frame():
    return pd.DataFrame(
        {
            "AS_AT_DATE": ["2025-04-01"],
            "SITE_ID": ["SBO01"],
            "PRODUCT_CODE": ["PART-001"],
            "QTY": [3],
        }
    )


def test_missing_required_sales_column_fails():
    frame = valid_sales_frame().drop(columns=["SHOP_CODE"])

    result = result_for(validate_sales(frame), "VAL-SALES-002")

    assert not result.passed
    assert result.count == 1


def test_non_numeric_total_units_fails_without_changing_source_frame():
    frame = valid_sales_frame().astype({"TOTAL_UNITS": "object"})
    frame.loc[0, "TOTAL_UNITS"] = "not-a-number"
    original = frame.copy(deep=True)

    result = result_for(validate_sales(frame), "VAL-SALES-005")

    assert not result.passed
    assert result.count == 1
    assert_frame_equal(frame, original)


def test_negative_total_units_is_still_numeric_and_valid_for_this_rule():
    frame = valid_sales_frame()
    frame.loc[0, "TOTAL_UNITS"] = -2

    result = result_for(validate_sales(frame), "VAL-SALES-005")

    assert result.passed
    assert result.count == 0


def test_duplicate_product_code_fails_without_deduplication():
    frame = pd.concat([valid_product_frame(), valid_product_frame()], ignore_index=True)

    result = result_for(validate_products(frame), "VAL-PROD-003")

    assert not result.passed
    assert result.count == 1
    assert len(frame) == 2


def test_non_numeric_price_where_present_fails():
    frame = valid_product_frame().astype({"PRICE": "object"})
    frame.loc[0, "PRICE"] = "unknown"

    result = result_for(validate_products(frame), "VAL-PROD-006")

    assert not result.passed
    assert result.count == 1


def test_non_numeric_inventory_qty_fails():
    frame = valid_inventory_frame().astype({"QTY": "object"})
    frame.loc[0, "QTY"] = "not-a-number"

    result = result_for(validate_inventory(frame), "VAL-INV-006")

    assert not result.passed
    assert result.count == 1


def test_missing_inventory_observation_is_not_converted_to_zero():
    frame = valid_inventory_frame()
    original = frame.copy(deep=True)

    validate_inventory(frame)

    assert_frame_equal(frame, original)
    assert (frame["QTY"] == 0).sum() == 0


def test_duplicate_calendar_date_fails_without_removing_rows():
    frame = pd.DataFrame({"DATE": ["2025-04-01", "2025-04-01"]})

    result = result_for(validate_calendar(frame), "VAL-CAL-004")

    assert not result.passed
    assert result.count == 1
    assert len(frame) == 2


def test_duplicate_branch_code_fails_without_modification():
    frame = pd.DataFrame({"SHOP_CODE": ["SBO01", "SBO01"]})

    result = result_for(validate_branches(frame), "VAL-BR-003")

    assert not result.passed
    assert result.count == 1
    assert len(frame) == 2
