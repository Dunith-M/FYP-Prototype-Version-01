from pathlib import Path

import pytest
from django.core.management import call_command
from django.db import IntegrityError, transaction
from openpyxl import Workbook

from catalog.branch_master import (
    EXPECTED_BRANCH_CODES,
    coverage_report,
    inspect_branch_profile,
    normalize_branch_code,
)
from catalog.models import Branch


def _write_workbook(path: Path, headers: list[str], rows: list[list[object]], sheet_name: str) -> Path:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    worksheet.append(headers)
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    return path


def _branch_profile(path: Path, rows: list[list[object]]) -> Path:
    return _write_workbook(path, ["SHOP_CODE", "SHOP_NAME"], rows, "Branch Profile")


def _coverage_workbook(path: Path, column: str, values: list[object]) -> Path:
    return _write_workbook(path, [column], [[value] for value in values], "data")


def _expected_branch_rows() -> list[list[str]]:
    return [[code, f"Branch {code}"] for code in sorted(EXPECTED_BRANCH_CODES)]


def test_normalization_handles_whitespace_and_lowercase():
    assert normalize_branch_code(" bil01 ") == "BIL01"


def test_duplicate_input_branch_codes_are_detected(tmp_path):
    profile = _branch_profile(
        tmp_path / "branch_profile.xlsx",
        [["bil01", "Branch one"], [" BIL01 ", "Branch one duplicate"]],
    )

    validation = inspect_branch_profile(profile)

    assert validation.duplicate_codes == {"BIL01": 2}


def test_unknown_sales_branch_code_is_detected(tmp_path):
    sales = _coverage_workbook(tmp_path / "sales.xlsx", "SHOP_CODE", ["bil01", "UNKNOWN"])

    report = coverage_report(sales, "SHOP_CODE", {"BIL01"})

    assert report.unmatched_codes == {"UNKNOWN": 1}
    assert report.rows_affected_by_unmatched_codes == 1


def test_unknown_inventory_branch_code_is_detected(tmp_path):
    inventory = _coverage_workbook(tmp_path / "inventory.xlsx", "SITE_ID", [" bil01 ", "UNKNOWN"])

    report = coverage_report(inventory, "SITE_ID", {"BIL01"})

    assert report.unmatched_codes == {"UNKNOWN": 1}
    assert report.rows_affected_by_unmatched_codes == 1


@pytest.mark.django_db
def test_branch_code_is_unique():
    Branch.objects.create(branch_code="BIL01", branch_name="Branch one")

    with pytest.raises(IntegrityError), transaction.atomic():
        Branch.objects.create(branch_code="BIL01", branch_name="Duplicate branch")


@pytest.mark.django_db
def test_loader_creates_expected_master_and_is_idempotent(tmp_path):
    profile = _branch_profile(tmp_path / "branch_profile.xlsx", _expected_branch_rows())
    sales = _coverage_workbook(tmp_path / "sales.xlsx", "SHOP_CODE", [" bil01 ", "SBO01"])
    inventory = _coverage_workbook(tmp_path / "inventory.xlsx", "SITE_ID", ["BIL01", "sbo01"])

    call_command("load_branches", branch_profile=profile, sales=sales, inventory=inventory)
    call_command("load_branches", branch_profile=profile, sales=sales, inventory=inventory)

    assert Branch.objects.count() == 10
    assert set(Branch.objects.values_list("branch_code", flat=True)) == EXPECTED_BRANCH_CODES
