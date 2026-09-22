import pandas as pd

from analytics.services.data_quality import FLAG_COLUMNS, apply_data_quality_flags


def _row(**overrides):
    values = {
        "raw_total_units": 5,
        "raw_total_value": 100,
        "hire_units": 2,
        "cash_units": 3,
        "hire_value": 40,
        "cash_value": 60,
        "family_id": "F01",
        "family_desc": "Family",
        "list_price": 100,
        "availability_state": "AVAILABLE",
        "mapping_status": "CONFIRMED",
        "canonical_product_id": "CP_P001",
        "inventory_qty": 4,
        "source_product_code": "P001",
    }
    values.update(overrides)
    return pd.DataFrame([values])


def test_unit_and_value_flags_preserve_source_values():
    source = pd.concat([_row(raw_total_units=-1), _row(raw_total_units=0), _row(raw_total_value=0)], ignore_index=True)
    result = apply_data_quality_flags(source)

    assert result["raw_total_units"].tolist() == [-1, 0, 5]
    assert result["negative_units_flag"].tolist() == [True, False, False]
    assert result["zero_units_flag"].tolist() == [False, True, False]
    assert result["zero_value_flag"].tolist() == [False, False, True]


def test_reconciliation_flags_only_evaluable_complete_component_rows():
    source = pd.concat([
        _row(raw_total_units=5, hire_units=2, cash_units=2),
        _row(raw_total_units=5, hire_units=2, cash_units=3),
        _row(hire_units=None),
    ], ignore_index=True)
    result = apply_data_quality_flags(source)

    assert result["reconciliation_difference_flag"].tolist() == [True, False, False]
    assert result["reconciliation_evaluable_flag"].tolist() == [True, True, False]
    assert result.loc[0, "raw_total_units"] == 5
    assert result.loc[0, "hire_units"] == 2
    assert result.loc[0, "cash_units"] == 2


def test_family_price_inventory_and_mapping_flags_preserve_rows_and_values():
    source = pd.concat([
        _row(family_id=None, list_price=None, availability_state="UNKNOWN", mapping_status="UNRESOLVED", canonical_product_id=None, inventory_qty=None, source_product_code="OLD001"),
        _row(family_desc="", list_price=0),
        _row(list_price=-100),
        _row(list_price=1),
    ], ignore_index=True)
    result = apply_data_quality_flags(source)

    assert len(result) == len(source)
    assert result["missing_family_flag"].tolist() == [True, True, False, False]
    assert result["invalid_price_flag"].tolist() == [True, True, True, False]
    assert result["inventory_unknown_flag"].tolist() == [True, False, False, False]
    assert result["unresolved_product_flag"].tolist() == [True, False, False, False]
    assert pd.isna(result.loc[0, "inventory_qty"])
    assert result.loc[0, "source_product_code"] == "OLD001"
    assert all(result[column].dtype == bool for column in FLAG_COLUMNS)


def test_missing_optional_inputs_are_explicitly_not_evaluable():
    result = apply_data_quality_flags(pd.DataFrame({"sales_units": [5], "sales_value": [10]}))

    assert result.loc[0, "reconciliation_evaluable_flag"] == False
    assert result.loc[0, "family_evaluable_flag"] == False
    assert result.loc[0, "price_evaluable_flag"] == False
