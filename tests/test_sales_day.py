import ast
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from analytics.sales_day import SALES_DAY_KEY, build_sales_day, write_sales_day_outputs
from catalog.models import Branch, Product, SkuMapping


def _sales_workbook(path: Path, rows: list[list[object]]) -> Path:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "data"
    worksheet.append(
        ["SALES_DATE", "SHOP_CODE", "SALES_PART", "TOTAL_UNITS", "TOTAL_VALUE"]
    )
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    return path


def _branch(code: str) -> Branch:
    return Branch.objects.create(branch_code=code, branch_name=f"Branch {code}")


def _product_mapping(code: str) -> Product:
    product = Product.objects.create(
        canonical_product_id=f"CP_{code}",
        product_code=code,
        product_desc=f"Product {code}",
    )
    SkuMapping.objects.create(
        source_system=SkuMapping.SourceSystem.SALES,
        source_product_code=code,
        canonical_product=product,
        mapping_reason=SkuMapping.MappingReason.EXACT_CODE,
        confidence_class=SkuMapping.ConfidenceClass.CONFIRMED,
    )
    return product


def _unresolved_mapping(code: str) -> None:
    SkuMapping.objects.create(
        source_system=SkuMapping.SourceSystem.SALES,
        source_product_code=code,
        mapping_reason=SkuMapping.MappingReason.MANUAL_REVIEW,
        confidence_class=SkuMapping.ConfidenceClass.UNRESOLVED,
        notes="Historical sales code absent from current product master",
    )


@pytest.mark.django_db
def test_multiple_rows_same_key_aggregate_and_preserve_source_count(tmp_path):
    _branch("BIL01")
    _product_mapping("P001")
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx",
        [
            [datetime(2025, 1, 10, 9), "BIL01", "P001", 1, 100],
            [datetime(2025, 1, 10, 10), "BIL01", "P001", 2, 200],
            [datetime(2025, 1, 10, 11), "BIL01", "P001", 1, 100],
        ],
    )

    result = build_sales_day(sales_file)

    row = result.sales_day.iloc[0]
    assert len(result.sales_day) == 1
    assert row.sales_units == 4
    assert row.sales_value == 400
    assert row.source_row_count == 3


@pytest.mark.django_db
def test_dates_branches_and_products_make_separate_keys(tmp_path):
    _branch("BIL01")
    _branch("BIN01")
    _product_mapping("P001")
    _product_mapping("P002")
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx",
        [
            [date(2025, 1, 10), "BIL01", "P001", 1, 10],
            [date(2025, 1, 11), "BIL01", "P001", 1, 10],
            [date(2025, 1, 10), "BIN01", "P001", 1, 10],
            [date(2025, 1, 10), "BIL01", "P002", 1, 10],
        ],
    )

    result = build_sales_day(sales_file)

    assert len(result.sales_day) == 4
    assert not result.sales_day.duplicated(list(SALES_DAY_KEY)).any()


@pytest.mark.django_db
def test_negative_and_zero_units_remain_in_aggregation(tmp_path):
    _branch("BIL01")
    _product_mapping("P001")
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx",
        [
            [date(2025, 1, 10), "BIL01", "P001", 5, 500],
            [date(2025, 1, 10), "BIL01", "P001", -1, -100],
            [date(2025, 1, 10), "BIL01", "P001", 0, 0],
        ],
    )

    result = build_sales_day(sales_file)
    row = result.sales_day.iloc[0]

    assert row.sales_units == 4
    assert row.source_row_count == 3
    assert row.negative_unit_row_count == 1
    assert row.zero_unit_row_count == 1
    assert result.metrics.negative_source_rows == 1
    assert result.metrics.zero_unit_source_rows == 1


@pytest.mark.django_db
def test_unresolved_rows_are_separate_and_reconciled(tmp_path):
    _branch("BIL01")
    _product_mapping("P001")
    _unresolved_mapping("OLD001")
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx",
        [
            [date(2025, 1, 10), "BIL01", "P001", 2, 200],
            [date(2025, 1, 10), "BIL01", "OLD001", 3, 300],
        ],
    )

    result = build_sales_day(sales_file)

    assert len(result.sales_day) == 1
    assert len(result.unresolved_sales) == 1
    assert result.metrics.unresolved_product_rows == 1
    assert result.metrics.unresolved_source_codes == ("OLD001",)
    assert result.metrics.rows_reconciled
    assert result.unresolved_sales.iloc[0].canonical_product_id != result.unresolved_sales.iloc[0].canonical_product_id
    assert bool(result.unresolved_sales.iloc[0].unresolved_product_flag)


@pytest.mark.django_db
def test_daily_reconciliation_evidence_is_retained_without_null_imputation(tmp_path):
    _branch("BIL01")
    _product_mapping("P001")
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "data"
    worksheet.append(
        [
            "SALES_DATE", "SHOP_CODE", "SALES_PART", "HIRE_UNITS", "HIRE_VAL",
            "CASH_UNITS", "CASH_VALUE", "TOTAL_UNITS", "TOTAL_VALUE",
        ]
    )
    worksheet.append([date(2025, 1, 10), "BIL01", "P001", 2, 40, 3, 60, 5, 100])
    source = tmp_path / "sales_with_components.xlsx"
    workbook.save(source)

    result = build_sales_day(source)

    row = result.sales_day.iloc[0]
    assert row.raw_total_units == 5
    assert row.raw_total_value == 100
    assert row.hire_units == 2 and row.cash_units == 3
    assert row.hire_value == 40 and row.cash_value == 60
    assert bool(row.reconciliation_components_complete_flag)


@pytest.mark.django_db
def test_units_and_values_reconcile_and_repeated_rows_are_not_deduplicated(tmp_path):
    _branch("BIL01")
    _product_mapping("P001")
    source_row = [date(2025, 1, 10), "BIL01", "P001", 2, 125]
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [source_row, source_row])

    result = build_sales_day(sales_file)

    assert result.metrics.raw_resolved_signed_units == 4
    assert result.metrics.aggregated_sales_units == 4
    assert result.metrics.raw_resolved_total_value == 250
    assert result.metrics.aggregated_sales_value == 250
    assert result.sales_day.iloc[0].source_row_count == 2


@pytest.mark.django_db
def test_existing_normalizers_handle_branch_product_and_sales_date(tmp_path):
    _branch("BIL01")
    _product_mapping("P001")
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx",
        [["2025-01-10 14:35:21", " bil01 ", " p001 ", 1, 10]],
    )

    result = build_sales_day(sales_file)

    row = result.sales_day.iloc[0]
    assert row.branch_code == "BIL01"
    assert row.canonical_product_id == "CP_P001"
    assert row.business_date == date(2025, 1, 10)


@pytest.mark.django_db
def test_output_rebuild_is_idempotent(tmp_path):
    _branch("BIL01")
    _product_mapping("P001")
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx", [[date(2025, 1, 10), "BIL01", "P001", 1, 10]]
    )
    result = build_sales_day(sales_file)
    sales_day_path = tmp_path / "sales_day.parquet"
    unresolved_path = tmp_path / "sales_unresolved.parquet"

    write_sales_day_outputs(result, sales_day_path, unresolved_path)
    first = pd.read_parquet(sales_day_path)
    write_sales_day_outputs(result, sales_day_path, unresolved_path)
    second = pd.read_parquet(sales_day_path)

    pd.testing.assert_frame_equal(first, second)


def test_builder_has_no_inventory_dependency():
    source = Path("analytics/sales_day.py").read_text(encoding="utf-8")
    imports = [
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    ]
    imports.extend(
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any("inventory" in module.lower() for module in imports)
