from pathlib import Path

import pytest
from django.core.management import call_command
from django.db import IntegrityError, transaction
from openpyxl import Workbook

from catalog.models import Product
from catalog.product_master import (
    canonical_product_id_for,
    inspect_product_source,
    normalize_product_code,
    parse_price,
)


def _product_workbook(path: Path, rows: list[list[object]]) -> Path:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "product_data.csv"
    worksheet.append(
        [
            "PRODUCT_CODE",
            "PRODUCT_DESC",
            "PRODUCT_FAMILY_ID",
            "PRODUCT_FAMILY_DESC",
            "PRICE",
        ]
    )
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    return path


def test_normalization_and_deterministic_canonical_id():
    assert normalize_product_code(" ab-c1 ") == "AB-C1"
    assert canonical_product_id_for(" ab-c1 ") == "CP_AB-C1"
    assert canonical_product_id_for("AB-C1") == "CP_AB-C1"


@pytest.mark.parametrize(
    ("source_value", "stored_value", "valid"),
    [
        (1250.5, "1250.5", True),
        (None, None, False),
        (0, "0", False),
        (-50, "-50", False),
    ],
)
def test_price_validity_rules(source_value, stored_value, valid):
    parsed = parse_price(source_value)

    assert parsed.valid is valid
    assert (None if parsed.value is None else str(parsed.value)) == stored_value


def test_conflicting_duplicate_product_codes_are_detected(tmp_path):
    product_file = _product_workbook(
        tmp_path / "products.xlsx",
        [
            ["abc1", "First description", "F1", "Family", 100],
            [" ABC1 ", "Conflicting description", "F1", "Family", 100],
        ],
    )

    validation = inspect_product_source(product_file)

    assert validation.duplicate_codes == {"ABC1": 2}
    assert validation.conflicting_duplicate_codes == ["ABC1"]


@pytest.mark.django_db
def test_product_code_and_canonical_id_are_unique():
    Product.objects.create(
        canonical_product_id="CP_ABC1",
        product_code="ABC1",
        product_desc="Product one",
    )

    with pytest.raises(IntegrityError), transaction.atomic():
        Product.objects.create(
            canonical_product_id="CP_ABC2",
            product_code="ABC1",
            product_desc="Duplicate code",
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        Product.objects.create(
            canonical_product_id="CP_ABC1",
            product_code="ABC2",
            product_desc="Duplicate canonical ID",
        )


@pytest.mark.django_db
def test_loader_is_idempotent_and_preserves_canonical_ids(tmp_path):
    product_file = _product_workbook(
        tmp_path / "products.xlsx",
        [
            [" abc1 ", "Product one", "F1", "Family one", 100],
            ["abc2", "Product two", "F2", "Family two", None],
            ["abc3", "Product three", "F3", "Family three", 0],
            ["abc4", "Product four", "F4", "Family four", -5],
        ],
    )

    call_command("load_products", product_file=product_file)
    first_ids = dict(
        Product.objects.values_list("product_code", "canonical_product_id")
    )
    call_command("load_products", product_file=product_file)

    assert Product.objects.count() == 4
    assert (
        dict(Product.objects.values_list("product_code", "canonical_product_id"))
        == first_ids
    )
    assert Product.objects.get(product_code="ABC1").price_valid is True
    assert Product.objects.get(product_code="ABC2").price_valid is False
    assert Product.objects.get(product_code="ABC3").price_valid is False
    assert Product.objects.get(product_code="ABC4").price_valid is False
