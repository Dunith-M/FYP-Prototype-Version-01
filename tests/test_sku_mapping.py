from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import IntegrityError, transaction
from django.db.models.deletion import ProtectedError
from openpyxl import Workbook

from catalog.models import Product, SkuMapping
from catalog.product_master import normalize_product_code
from catalog.sku_mapping import (
    extract_sales_source_metrics,
    normalize_source_product_code,
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


@pytest.mark.parametrize(
    "source_code",
    ["ABC001", "abc001 ", " AbC001 ", " ab-001 "],
)
def test_sales_and_product_normalization_share_trim_and_uppercase_behavior(source_code):
    assert normalize_source_product_code(source_code) == normalize_product_code(
        source_code
    )


def test_blank_sales_part_is_detected(tmp_path):
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [[None, 1]])

    metrics = extract_sales_source_metrics(sales_file)

    assert metrics.blank_source_codes == 1
    assert metrics.distinct_codes == 0


@pytest.mark.django_db
@pytest.mark.parametrize("source_code", ["ABC001", "abc001 ", " AbC001 "])
def test_exact_source_code_maps_to_matching_product(tmp_path, source_code):
    product = _product("ABC001")
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [[source_code, 2]])

    call_command("build_sku_mappings", sales_file=sales_file)

    mapping = SkuMapping.objects.get(
        source_system=SkuMapping.SourceSystem.SALES,
        source_product_code="ABC001",
    )
    assert mapping.canonical_product == product
    assert mapping.mapping_reason == SkuMapping.MappingReason.EXACT_CODE
    assert mapping.confidence_class == SkuMapping.ConfidenceClass.CONFIRMED


@pytest.mark.django_db
def test_unknown_source_code_is_stored_unresolved_and_not_discarded(tmp_path):
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["UNKNOWN", 3]])

    call_command("build_sku_mappings", sales_file=sales_file)

    mapping = SkuMapping.objects.get(source_product_code="UNKNOWN")
    assert mapping.canonical_product is None
    assert mapping.confidence_class == SkuMapping.ConfidenceClass.UNRESOLVED
    assert SkuMapping.objects.count() == 1


@pytest.mark.django_db
def test_description_similarity_does_not_map_unknown_code(tmp_path):
    _product("NEW001", description="Samsung Refrigerator 250 L")
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx", [["OLD001", 1, "Samsung Fridge 250L"]]
    )

    call_command("build_sku_mappings", sales_file=sales_file)

    assert (
        SkuMapping.objects.get(source_product_code="OLD001").canonical_product is None
    )


@pytest.mark.django_db
def test_different_exact_codes_remain_unresolved(tmp_path):
    _product("ABC002")
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["ABC001", 1]])

    call_command("build_sku_mappings", sales_file=sales_file)

    mapping = SkuMapping.objects.get(source_product_code="ABC001")
    assert mapping.canonical_product is None
    assert mapping.confidence_class == SkuMapping.ConfidenceClass.UNRESOLVED


@pytest.mark.django_db
def test_internal_punctuation_difference_is_not_an_exact_match(tmp_path):
    _product("AB001")
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["AB-001", 1]])

    call_command("build_sku_mappings", sales_file=sales_file)

    assert (
        SkuMapping.objects.get(source_product_code="AB-001").canonical_product is None
    )


@pytest.mark.django_db
def test_whitespace_and_case_normalization_preserves_internal_punctuation(tmp_path):
    product = _product("AB-001")
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [[" ab-001 ", 1]])

    call_command("build_sku_mappings", sales_file=sales_file)

    mapping = SkuMapping.objects.get(source_product_code="AB-001")
    assert mapping.canonical_product == product
    assert mapping.mapping_reason == SkuMapping.MappingReason.EXACT_CODE
    assert mapping.confidence_class == SkuMapping.ConfidenceClass.CONFIRMED


@pytest.mark.django_db
def test_duplicate_source_system_and_code_is_prevented():
    SkuMapping.objects.create(
        source_system=SkuMapping.SourceSystem.SALES,
        source_product_code="ABC123",
        mapping_reason=SkuMapping.MappingReason.MANUAL_REVIEW,
        confidence_class=SkuMapping.ConfidenceClass.UNRESOLVED,
    )

    with pytest.raises(IntegrityError), transaction.atomic():
        SkuMapping.objects.create(
            source_system=SkuMapping.SourceSystem.SALES,
            source_product_code="ABC123",
            mapping_reason=SkuMapping.MappingReason.MANUAL_REVIEW,
            confidence_class=SkuMapping.ConfidenceClass.UNRESOLVED,
        )


@pytest.mark.django_db
def test_loader_is_idempotent(tmp_path):
    _product("ABC123")
    sales_file = _sales_workbook(
        tmp_path / "sales.xlsx", [["ABC123", 1], ["UNKNOWN", -1]]
    )

    call_command("build_sku_mappings", sales_file=sales_file)
    call_command("build_sku_mappings", sales_file=sales_file)

    assert SkuMapping.objects.count() == 2


@pytest.mark.django_db
def test_reviewed_or_approved_mapping_is_not_overwritten(tmp_path):
    target = _product("TARGET")
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [["ALIAS", 1]])
    mapping = SkuMapping.objects.create(
        source_system=SkuMapping.SourceSystem.SALES,
        source_product_code="ALIAS",
        canonical_product=target,
        mapping_reason=SkuMapping.MappingReason.APPROVED_ALIAS,
        confidence_class=SkuMapping.ConfidenceClass.CONFIRMED,
        reviewer="Data steward",
    )

    call_command("build_sku_mappings", sales_file=sales_file)

    mapping.refresh_from_db()
    assert mapping.canonical_product == target
    assert mapping.mapping_reason == SkuMapping.MappingReason.APPROVED_ALIAS


@pytest.mark.django_db
def test_mapped_product_is_protected_from_deletion():
    product = _product("ABC123")
    SkuMapping.objects.create(
        source_system=SkuMapping.SourceSystem.SALES,
        source_product_code="ABC123",
        canonical_product=product,
        mapping_reason=SkuMapping.MappingReason.EXACT_CODE,
        confidence_class=SkuMapping.ConfidenceClass.CONFIRMED,
    )

    with pytest.raises(ProtectedError):
        product.delete()


@pytest.mark.django_db
def test_loader_blocks_blank_sales_part(tmp_path):
    sales_file = _sales_workbook(tmp_path / "sales.xlsx", [[None, 1]])

    with pytest.raises(CommandError, match="blank_or_null_sales_part_rows=1"):
        call_command("build_sku_mappings", sales_file=sales_file)
