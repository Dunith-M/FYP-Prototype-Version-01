from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from analytics.branch_product_day import (
    INVENTORY_DAY_KEY,
    InventoryDayValidationError,
    build_branch_product_day,
    build_inventory_day,
    write_branch_product_day,
)
from catalog.models import Branch, Product


def _inventory_workbook(path: Path, rows: list[list[object]]) -> Path:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "inventory_data.csv"
    worksheet.append(["AS_AT_DATE", "PRODUCT_CODE", "SITE_ID", "QTY"])
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    return path


def _sales_day(path: Path, rows: list[list[object]]) -> Path:
    frame = pd.DataFrame(
        rows,
        columns=[
            "branch_code",
            "canonical_product_id",
            "business_date",
            "sales_units",
            "sales_value",
            "source_row_count",
            "negative_unit_row_count",
            "zero_unit_row_count",
        ],
    )
    frame.to_parquet(path, index=False)
    return path


def _branch(code: str = "BIL01") -> Branch:
    return Branch.objects.create(branch_code=code, branch_name=code)


def _product(code: str = "P001") -> Product:
    return Product.objects.create(
        canonical_product_id=f"CP_{code}", product_code=code, product_desc=code
    )


def _build_inventory(tmp_path, rows):
    source = _inventory_workbook(tmp_path / "inventory.xlsx", rows)
    output = tmp_path / "inventory_day.parquet"
    unresolved = tmp_path / "inventory_unresolved.parquet"
    result = build_inventory_day(source, output, unresolved)
    return result, output, unresolved


@pytest.mark.django_db
def test_matching_snapshot_joins_as_available(tmp_path):
    _branch()
    _product()
    _, inventory_path, _ = _build_inventory(
        tmp_path, [[date(2025, 5, 1), "P001", "BIL01", 8]]
    )
    sales_path = _sales_day(
        tmp_path / "sales.parquet",
        [["BIL01", "CP_P001", date(2025, 5, 1), 2, 200, 1, 0, 0]],
    )

    result = build_branch_product_day(sales_path, inventory_path)
    row = result.branch_product_day.iloc[0]

    assert row.sales_units == 2
    assert row.inventory_qty == 8
    assert row.availability_state == "AVAILABLE"


@pytest.mark.django_db
def test_missing_snapshot_is_unknown_with_null_quantity(tmp_path):
    _branch()
    _product()
    _, inventory_path, _ = _build_inventory(tmp_path, [])
    sales_path = _sales_day(
        tmp_path / "sales.parquet",
        [["BIL01", "CP_P001", date(2025, 5, 1), 2, 200, 1, 0, 0]],
    )

    result = build_branch_product_day(sales_path, inventory_path)
    row = result.branch_product_day.iloc[0]

    assert pd.isna(row.inventory_qty)
    assert row.availability_state == "UNKNOWN"
    assert row.inventory_qty != 0


@pytest.mark.django_db
def test_left_join_preserves_rows_and_sales_totals(tmp_path):
    _branch("BIL01")
    _branch("BIN01")
    _product("P001")
    _, inventory_path, _ = _build_inventory(
        tmp_path, [[date(2025, 5, 1), "P001", "BIL01", 8]]
    )
    sales_path = _sales_day(
        tmp_path / "sales.parquet",
        [
            ["BIL01", "CP_P001", date(2025, 5, 1), 2, 200, 1, 0, 0],
            ["BIN01", "CP_P001", date(2025, 5, 1), 3, 300, 1, 0, 0],
        ],
    )

    result = build_branch_product_day(sales_path, inventory_path)

    assert len(result.branch_product_day) == 2
    assert result.metrics.inventory_match_rows == 1
    assert result.metrics.unknown_inventory_rows == 1
    assert result.metrics.unit_difference == 0
    assert result.metrics.value_difference == 0


@pytest.mark.django_db
def test_duplicate_inventory_key_is_detected_and_not_summed(tmp_path):
    _branch()
    _product()
    source = _inventory_workbook(
        tmp_path / "inventory.xlsx",
        [
            [date(2025, 5, 1), "P001", "BIL01", 8],
            [date(2025, 5, 1), "P001", "BIL01", 9],
        ],
    )

    with pytest.raises(InventoryDayValidationError, match="duplicate"):
        build_inventory_day(
            source,
            tmp_path / "inventory_day.parquet",
            tmp_path / "inventory_unresolved.parquet",
        )


@pytest.mark.django_db
def test_branch_and_product_normalization_reuses_existing_rules(tmp_path):
    _branch()
    _product()
    result, inventory_path, _ = _build_inventory(
        tmp_path, [["2025-05-01 00:00:00", " p001 ", " bil01 ", 8]]
    )
    frame = pd.read_parquet(inventory_path)

    assert result.metrics.resolved_inventory_rows == 1
    assert frame.iloc[0].branch_code == "BIL01"
    assert frame.iloc[0].canonical_product_id == "CP_P001"
    assert frame.iloc[0].snapshot_date == date(2025, 5, 1)


@pytest.mark.django_db
def test_unmatched_branch_is_reported(tmp_path):
    _branch()
    _product()
    source = _inventory_workbook(
        tmp_path / "inventory.xlsx", [[date(2025, 5, 1), "P001", "OTHER", 8]]
    )

    with pytest.raises(InventoryDayValidationError, match="SITE_ID"):
        build_inventory_day(
            source,
            tmp_path / "inventory_day.parquet",
            tmp_path / "inventory_unresolved.parquet",
        )


@pytest.mark.django_db
def test_unmatched_product_is_preserved_separately(tmp_path):
    _branch()
    _product()
    result, inventory_path, unresolved_path = _build_inventory(
        tmp_path, [[date(2025, 5, 1), "UNKNOWN", "BIL01", 8]]
    )

    assert result.metrics.resolved_inventory_rows == 0
    assert result.metrics.unresolved_product_rows == 1
    assert pd.read_parquet(inventory_path).empty
    unresolved = pd.read_parquet(unresolved_path)
    assert unresolved.iloc[0].source_product_code == "UNKNOWN"


@pytest.mark.django_db
def test_nearby_snapshot_date_does_not_match(tmp_path):
    _branch()
    _product()
    _, inventory_path, _ = _build_inventory(
        tmp_path, [[date(2025, 5, 2), "P001", "BIL01", 8]]
    )
    sales_path = _sales_day(
        tmp_path / "sales.parquet",
        [["BIL01", "CP_P001", date(2025, 5, 1), 2, 200, 1, 0, 0]],
    )

    result = build_branch_product_day(sales_path, inventory_path)

    assert result.branch_product_day.iloc[0].availability_state == "UNKNOWN"
    assert pd.isna(result.branch_product_day.iloc[0].inventory_qty)


@pytest.mark.django_db
def test_inventory_day_key_is_unique_and_zero_is_explicit(tmp_path):
    _branch()
    _product()
    _, inventory_path, _ = _build_inventory(
        tmp_path, [[date(2025, 5, 1), "P001", "BIL01", 0]]
    )
    frame = pd.read_parquet(inventory_path)

    assert not frame.duplicated(list(INVENTORY_DAY_KEY)).any()
    assert frame.iloc[0].availability_state == "ZERO_CONFIRMED"


@pytest.mark.django_db
def test_branch_product_day_output_rebuild_is_idempotent(tmp_path):
    _branch()
    _product()
    _, inventory_path, _ = _build_inventory(
        tmp_path, [[date(2025, 5, 1), "P001", "BIL01", 8]]
    )
    sales_path = _sales_day(
        tmp_path / "sales.parquet",
        [["BIL01", "CP_P001", date(2025, 5, 1), 2, 200, 1, 0, 0]],
    )
    result = build_branch_product_day(sales_path, inventory_path)
    output = tmp_path / "branch_product_day.parquet"

    write_branch_product_day(result, output)
    first = pd.read_parquet(output)
    write_branch_product_day(result, output)
    second = pd.read_parquet(output)

    pd.testing.assert_frame_equal(first, second)
