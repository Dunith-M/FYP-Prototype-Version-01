from pathlib import Path

import pandas as pd


# ============================================================
# FILE PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CALENDAR_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "calendar_data.xlsx"
)

SHEET_NAME = "Sheet1"


# ============================================================
# CHECK FILE
# ============================================================

print("=" * 80)
print("CALENDAR DATA PROFILING")
print("=" * 80)

print(f"\nFile path:\n{CALENDAR_FILE}")

if not CALENDAR_FILE.exists():
    raise FileNotFoundError(
        f"\nCalendar file not found:\n{CALENDAR_FILE}\n"
        "Make sure the Excel file is inside data/raw/"
    )


# ============================================================
# LOAD DATA
# ============================================================

calendar = pd.read_excel(
    CALENDAR_FILE,
    sheet_name=SHEET_NAME,
)


# ============================================================
# 1. BASIC STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("1. BASIC STRUCTURE")
print("=" * 80)

print(f"Sheet name: {SHEET_NAME}")
print(f"Rows: {len(calendar):,}")
print(f"Columns: {len(calendar.columns)}")

print("\nColumn names:")
for column in calendar.columns:
    print(f" - {column}")


# ============================================================
# 2. DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("2. DATA TYPES")
print("=" * 80)

print(calendar.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("3. MISSING VALUES")
print("=" * 80)

for column, count in calendar.isna().sum().items():
    print(f"{column}: {count:,}")


# ============================================================
# 4. DATE COLUMN CHECK
# ============================================================

print("\n" + "=" * 80)
print("4. DATE COLUMN CHECK")
print("=" * 80)

DATE_COLUMN = "DATE"

calendar_dates = pd.to_datetime(
    calendar[DATE_COLUMN],
    errors="coerce",
)

print(f"Date column: {DATE_COLUMN}")
print(f"Minimum date: {calendar_dates.min()}")
print(f"Maximum date: {calendar_dates.max()}")
print(f"Missing/unparseable dates: {calendar_dates.isna().sum():,}")
print(f"Unique dates: {calendar_dates.dt.date.nunique():,}")


# ============================================================
# 5. DUPLICATE DATE CHECK
# ============================================================

print("\n" + "=" * 80)
print("5. DUPLICATE DATE CHECK")
print("=" * 80)

duplicate_date_rows = calendar.duplicated(
    subset=[DATE_COLUMN],
    keep=False,
).sum()

excess_duplicate_dates = calendar.duplicated(
    subset=[DATE_COLUMN],
    keep="first",
).sum()

print(
    f"Rows involved in duplicate DATE values: {duplicate_date_rows:,}"
)

print(
    f"Excess duplicate DATE rows: {excess_duplicate_dates:,}"
)


# ============================================================
# 6. MISSING DATE CHECK
# ============================================================

print("\n" + "=" * 80)
print("6. MISSING CALENDAR DATES")
print("=" * 80)

valid_dates = calendar_dates.dropna().dt.normalize().drop_duplicates()

if not valid_dates.empty:
    expected_dates = pd.date_range(
        start=valid_dates.min(),
        end=valid_dates.max(),
        freq="D",
    )

    actual_dates = pd.DatetimeIndex(valid_dates)

    missing_calendar_dates = expected_dates.difference(actual_dates)

    print(f"Expected number of dates: {len(expected_dates):,}")
    print(f"Actual unique dates: {len(actual_dates):,}")
    print(f"Missing dates inside range: {len(missing_calendar_dates):,}")

    if len(missing_calendar_dates) > 0:
        print("\nMissing dates:")
        for date in missing_calendar_dates:
            print(f" - {date.date()}")
else:
    print("No valid calendar dates available.")


# ============================================================
# 7. HOLIDAY FIELDS
# ============================================================

print("\n" + "=" * 80)
print("7. HOLIDAY FIELD CHECKS")
print("=" * 80)

holiday_columns = [
    "IS_PUBLIC_HOLIDAY",
    "HOLIDAY_NAME",
    "EVENT_TYPE",
]

for column in holiday_columns:
    if column in calendar.columns:
        print(f"\n{column}")
        print(f"Missing: {calendar[column].isna().sum():,}")
        print(f"Unique values: {calendar[column].nunique(dropna=True):,}")
    else:
        print(f"\nWARNING: Expected holiday field not found: {column}")


if "IS_PUBLIC_HOLIDAY" in calendar.columns:
    public_holiday_rows = (
        calendar["IS_PUBLIC_HOLIDAY"] == 1
    ).sum()

    non_holiday_rows = (
        calendar["IS_PUBLIC_HOLIDAY"] == 0
    ).sum()

    print(f"\nPublic holiday rows: {public_holiday_rows:,}")
    print(f"Non-public-holiday rows: {non_holiday_rows:,}")


# ============================================================
# 8. MONTH CHECKS
# ============================================================

print("\n" + "=" * 80)
print("8. MONTH CHECKS")
print("=" * 80)

month_columns = [
    "MONTH",
    "MONTH_NAME",
]

for column in month_columns:
    if column in calendar.columns:
        print(f"\n{column}:")
        print(calendar[column].value_counts(dropna=False).sort_index())
    else:
        print(f"\nWARNING: Expected month field not found: {column}")


# ============================================================
# 9. WEEKDAY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("9. WEEKDAY CHECKS")
print("=" * 80)

weekday_columns = [
    "DAY_OF_WEEK",
    "DAY_OF_WEEK_NUM",
]

for column in weekday_columns:
    if column in calendar.columns:
        print(f"\n{column}:")
        print(calendar[column].value_counts(dropna=False).sort_index())
    else:
        print(f"\nWARNING: Expected weekday field not found: {column}")


# ============================================================
# 10. WEEKEND CHECKS
# ============================================================

print("\n" + "=" * 80)
print("10. WEEKEND CHECKS")
print("=" * 80)

if "IS_WEEKEND" in calendar.columns:
    print(calendar["IS_WEEKEND"].value_counts(dropna=False).sort_index())

    weekend_rows = (calendar["IS_WEEKEND"] == 1).sum()
    weekday_rows = (calendar["IS_WEEKEND"] == 0).sum()

    print(f"\nWeekend rows: {weekend_rows:,}")
    print(f"Non-weekend rows: {weekday_rows:,}")
else:
    print("WARNING: IS_WEEKEND column not found.")


# ============================================================
# 11. DATE-DERIVED CONSISTENCY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("11. DATE-DERIVED CONSISTENCY CHECKS")
print("=" * 80)

valid_mask = calendar_dates.notna()

if "MONTH" in calendar.columns:
    month_mismatches = (
        calendar.loc[valid_mask, "MONTH"]
        != calendar_dates.loc[valid_mask].dt.month
    ).sum()

    print(f"MONTH mismatches against DATE: {month_mismatches:,}")


if "MONTH_NAME" in calendar.columns:
    month_name_mismatches = (
        calendar.loc[valid_mask, "MONTH_NAME"].astype(str).str.strip().str.lower()
        != calendar_dates.loc[valid_mask].dt.month_name().str.lower()
    ).sum()

    print(
        f"MONTH_NAME mismatches against DATE: "
        f"{month_name_mismatches:,}"
    )


if "DAY_OF_WEEK" in calendar.columns:
    weekday_mismatches = (
        calendar.loc[valid_mask, "DAY_OF_WEEK"]
        .astype(str)
        .str.strip()
        .str.lower()
        != calendar_dates.loc[valid_mask]
        .dt.day_name()
        .str.lower()
    ).sum()

    print(
        f"DAY_OF_WEEK mismatches against DATE: "
        f"{weekday_mismatches:,}"
    )


if "IS_WEEKEND" in calendar.columns:
    derived_weekend = (
        calendar_dates.loc[valid_mask].dt.dayofweek >= 5
    ).astype(int)

    weekend_mismatches = (
        calendar.loc[valid_mask, "IS_WEEKEND"]
        != derived_weekend
    ).sum()

    print(
        f"IS_WEEKEND mismatches against DATE: "
        f"{weekend_mismatches:,}"
    )


# ============================================================
# 12. EXACT DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 80)
print("12. EXACT DUPLICATE ROWS")
print("=" * 80)

exact_duplicates = calendar.duplicated().sum()

print(f"Exact duplicate rows: {exact_duplicates:,}")


# ============================================================
# 13. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CALENDAR PROFILING SUMMARY")
print("=" * 80)

missing_date_count = calendar_dates.isna().sum()
unique_date_count = calendar_dates.dt.date.nunique()

print(f"""
File: {CALENDAR_FILE.name}
Sheet: {SHEET_NAME}

Rows: {len(calendar):,}
Columns: {len(calendar.columns)}

Date column: {DATE_COLUMN}
Minimum date: {calendar_dates.min()}
Maximum date: {calendar_dates.max()}

Unique dates: {unique_date_count:,}
Missing/unparseable DATE values: {missing_date_count:,}
Excess duplicate DATE rows: {excess_duplicate_dates:,}

Exact duplicate rows: {exact_duplicates:,}

IMPORTANT STRUCTURAL RULE:
One calendar row should represent one calendar date.

NO DATA WAS CHANGED.
NO ROWS WERE DELETED.
NO HOLIDAY VALUES WERE FILLED.
""")