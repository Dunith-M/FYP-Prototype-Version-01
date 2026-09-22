import csv
from pathlib import Path

import pytest
from django.core.management import call_command
from openpyxl import Workbook

from catalog.models import Product, SkuMapping
from catalog.sku_mapping import (
    build_unresolved_sales_queue,
    get_unresolved_sales_skus,
)


def _sales_workbook(path: Path, rows: list[list[object]]) -> Path:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "data"
    worksheet.append(["SALES_PART", "TOTAL_UNITS", "PART_DESC"])
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    return path


def _product(code: str, description: str = "Product") -> Product:
    return Product.objects.create(
        canonical_product_id=f"CP_{code}",
        product_code=code,
        product_desc=description,
    )


@pytest.mark.django_db
def test_unknown_sales_code_creates_unresolved_mapping(tmp_path):
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["UNKNOWN", 1]])

    call_command("build_sku_mappings", sales_file=sales_file)

    mapping = SkuMapping.objects.get(source_product_code="UNKNOWN")
    assert mapping.canonical_product is None
    assert mapping.confidence_class == SkuMapping.ConfidenceClass.UNRESOLVED
    assert mapping.mapping_reason == SkuMapping.MappingReason.MANUAL_REVIEW


@pytest.mark.django_db
def test_unresolved_mapping_is_preserved_after_rebuilding(tmp_path):
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["UNKNOWN", 1]])

    call_command("build_sku_mappings", sales_file=sales_file)
    call_command("build_sku_mappings", sales_file=sales_file)

    assert SkuMapping.objects.filter(source_product_code="UNKNOWN").count() == 1


@pytest.mark.django_db
def test_multiple_sales_rows_for_one_unknown_code_create_one_queue_record(tmp_path):
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx", [["UNKNOWN", 1], ["UNKNOWN", 2]]
    )

    call_command("build_sku_mappings", sales_file=sales_file)

    assert get_unresolved_sales_skus().count() == 1


@pytest.mark.django_db
def test_unresolved_mapping_is_returned_by_queue_query(tmp_path):
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["UNKNOWN", 1]])
    call_command("build_sku_mappings", sales_file=sales_file)

    assert list(
        get_unresolved_sales_skus().values_list("source_product_code", flat=True)
    ) == ["UNKNOWN"]


@pytest.mark.django_db
def test_confirmed_mapping_is_not_returned_by_queue_query(tmp_path):
    _product("KNOWN")
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["KNOWN", 1]])
    call_command("build_sku_mappings", sales_file=sales_file)

    assert get_unresolved_sales_skus().count() == 0


@pytest.mark.django_db
def test_unresolved_source_code_contributes_to_affected_row_count(tmp_path):
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx",
        [["UNKNOWN", 2], ["UNKNOWN", -1], ["OTHER", 4]],
    )
    call_command("build_sku_mappings", sales_file=sales_file)

    report = build_unresolved_sales_queue(sales_file)
    unknown = next(
        item for item in report.items if item.mapping.source_product_code == "UNKNOWN"
    )

    assert unknown.affected_sales_rows == 2
    assert unknown.affected_signed_units == 1
    assert report.affected_sales_rows == 3


@pytest.mark.django_db
def test_similar_description_does_not_remove_code_from_unresolved_queue(tmp_path):
    _product("NEW001", description="Samsung Refrigerator 250 L")
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx",
        [["OLD001", 1, "Samsung Fridge 250L"]],
    )
    call_command("build_sku_mappings", sales_file=sales_file)

    assert list(
        get_unresolved_sales_skus().values_list("source_product_code", flat=True)
    ) == ["OLD001"]


@pytest.mark.django_db
def test_exact_mapping_is_not_in_unresolved_queue(tmp_path):
    _product("KNOWN")
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["KNOWN", 1]])
    call_command("build_sku_mappings", sales_file=sales_file)

    report = build_unresolved_sales_queue(sales_file)

    assert report.unresolved_mapping_count == 0
    assert report.items == ()


@pytest.mark.django_db
def test_report_csv_keeps_canonical_product_id_blank_for_unresolved_mapping(tmp_path):
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["UNKNOWN", 1]])
    report_path = tmp_path / "unresolved.csv"
    call_command("build_sku_mappings", sales_file=sales_file)

    call_command(
        "report_unresolved_skus",
        sales_file=sales_file,
        output=report_path,
        top=0,
    )

    with report_path.open(newline="", encoding="utf-8") as report_file:
        rows = list(csv.DictReader(report_file))
    assert rows[0]["canonical_product_id"] == ""
    assert rows[0]["confidence_class"] == SkuMapping.ConfidenceClass.UNRESOLVED


@pytest.mark.django_db
def test_repeated_reporting_does_not_modify_mapping_state(tmp_path):
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["UNKNOWN", 1]])
    report_path = tmp_path / "unresolved.csv"
    call_command("build_sku_mappings", sales_file=sales_file)
    before = list(
        SkuMapping.objects.values_list(
            "source_system",
            "source_product_code",
            "canonical_product_id",
            "mapping_reason",
            "confidence_class",
            "reviewer",
            "reviewed_at",
            "notes",
        )
    )

    call_command("report_unresolved_skus", sales_file=sales_file, output=report_path)
    call_command("report_unresolved_skus", sales_file=sales_file, output=report_path)

    after = list(
        SkuMapping.objects.values_list(
            "source_system",
            "source_product_code",
            "canonical_product_id",
            "mapping_reason",
            "confidence_class",
            "reviewer",
            "reviewed_at",
            "notes",
        )
    )
    assert after == before
