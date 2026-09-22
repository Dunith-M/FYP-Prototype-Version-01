from datetime import date, datetime, timezone
from pathlib import Path

from openpyxl import Workbook

from analytics.date_utils import (
    normalize_calendar_date,
    normalize_date_sources,
    normalize_inventory_date,
    normalize_sales_date,
    profile_workbook_date_column,
    year_month_from_calendar_date,
)


def _workbook(path: Path, sheet_name: str, column: str, values: list[object]) -> Path:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    worksheet.append([column])
    for value in values:
        worksheet.append([value])
    workbook.save(path)
    return path


def test_sales_timestamp_derives_business_date_and_preserves_original_value():
    timestamp = datetime(2025, 7, 18, 14, 35, 21)

    result = normalize_sales_date(timestamp)

    assert result.original_value == timestamp
    assert result.normalized_date == date(2025, 7, 18)
    assert result.status == "PARSED"


def test_datetime_and_iso_string_normalize_to_the_same_sales_date():
    assert normalize_sales_date(datetime(2025, 7, 18, 14, 35, 21)).normalized_date == date(
        2025, 7, 18
    )
    assert normalize_sales_date("2025-07-18 14:35:21").normalized_date == date(
        2025, 7, 18
    )
    assert normalize_sales_date("2025-07-18").normalized_date == date(2025, 7, 18)


def test_inventory_datetime_and_date_derive_snapshot_date():
    assert normalize_inventory_date(datetime(2025, 7, 18, 0, 0)).normalized_date == date(
        2025, 7, 18
    )
    assert normalize_inventory_date(date(2025, 7, 18)).normalized_date == date(
        2025, 7, 18
    )


def test_calendar_date_derives_calendar_date():
    assert normalize_calendar_date("2025-07-18").normalized_date == date(2025, 7, 18)


def test_normalized_dates_support_exact_date_equality():
    sales_date = normalize_sales_date("2025-07-18 14:35:21").normalized_date
    inventory_date = normalize_inventory_date("2025-07-18").normalized_date
    calendar_date = normalize_calendar_date("2025-07-18").normalized_date

    assert sales_date == calendar_date
    assert inventory_date == calendar_date


def test_invalid_and_null_dates_are_not_coerced_to_fake_dates():
    invalid = normalize_sales_date("not a date")
    null = normalize_sales_date(None)

    assert (invalid.status, invalid.normalized_date) == ("UNPARSEABLE", None)
    assert (null.status, null.normalized_date) == ("NULL", None)


def test_naive_timestamp_is_not_shifted_as_if_it_were_utc():
    naive = normalize_sales_date(datetime(2025, 7, 18, 0, 30))
    aware_utc = normalize_sales_date(datetime(2025, 7, 17, 19, 0, tzinfo=timezone.utc))

    assert naive.normalized_date == date(2025, 7, 18)
    assert aware_utc.normalized_date == date(2025, 7, 18)


def test_calendar_duplicate_dates_are_detected(tmp_path):
    calendar = _workbook(
        tmp_path / "calendar.xlsx",
        "Sheet1",
        "DATE",
        [date(2025, 7, 18), date(2025, 7, 18), None, "invalid"],
    )

    profile = profile_workbook_date_column(
        calendar, "Sheet1", "calendar", "DATE", normalize_calendar_date
    )

    assert profile.duplicate_date_rows == 1
    assert profile.null_dates == 1
    assert profile.unparseable_dates == 1


def test_year_month_is_derived_once_from_calendar_date():
    assert year_month_from_calendar_date(date(2025, 7, 18)) == "2025-07"


def test_repeated_normalization_is_identical_and_reports_exact_coverage(tmp_path):
    sales = _workbook(
        tmp_path / "sales.xlsx",
        "data",
        "SALES_DATE",
        [datetime(2025, 7, 18, 14, 35, 21), date(2025, 7, 19)],
    )
    inventory = _workbook(
        tmp_path / "inventory.xlsx",
        "inventory_data.csv",
        "AS_AT_DATE",
        [datetime(2025, 7, 18, 0, 0), date(2025, 7, 19)],
    )
    calendar = _workbook(
        tmp_path / "calendar.xlsx",
        "Sheet1",
        "DATE",
        [date(2025, 7, 18), date(2025, 7, 19)],
    )

    first = normalize_date_sources(sales, inventory, calendar)
    second = normalize_date_sources(sales, inventory, calendar)

    assert first.to_dict() == second.to_dict()
    assert first.sales_dates_missing_from_calendar == ()
    assert first.inventory_dates_missing_from_calendar == ()
