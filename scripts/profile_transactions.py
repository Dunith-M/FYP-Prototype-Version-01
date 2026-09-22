from pathlib import Path
import pandas as pd

# ============================================================
# FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SALES_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "transactions_data.xlsx"
)

# The current file in data/raw is misspelled as "tansactions_data.xlsx".
# Use it as a fallback until the source file is renamed.
if not SALES_FILE.exists():
    misspelled_sales_file = SALES_FILE.with_name("tansactions_data.xlsx")
    if misspelled_sales_file.exists():
        SALES_FILE = misspelled_sales_file

OUTPUT_FILE = PROJECT_ROOT / "data" / "artifacts" / "transaction_profile_report.xlsx"

SHEET_NAME = "data"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("SALES DATA PROFILING")
print("=" * 80)

print(f"\nFile path:\n{SALES_FILE}")

if not SALES_FILE.exists():
    raise FileNotFoundError(
        f"\nSales file was not found:\n{SALES_FILE}\n"
        "Check that the Excel file is inside data/raw/"
    )

sales = pd.read_excel(
    SALES_FILE,
    sheet_name=SHEET_NAME,
)


# ============================================================
# BASIC STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("1. BASIC STRUCTURE")
print("=" * 80)

print(f"Sheet name: {SHEET_NAME}")
print(f"Rows: {len(sales):,}")
print(f"Columns: {len(sales.columns)}")

print("\nColumn names:")
for column in sales.columns:
    print(f" - {column}")


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("2. DATA TYPES")
print("=" * 80)

print(sales.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("3. MISSING VALUES")
print("=" * 80)

missing = sales.isna().sum()

for column, count in missing.items():
    print(f"{column}: {count:,}")


# ============================================================
# SHOP INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("4. SHOP INFORMATION")
print("=" * 80)

shop_count = sales["SHOP_CODE"].nunique(dropna=True)

print(f"Number of unique shops: {shop_count}")

print("\nShop codes:")
print(sorted(sales["SHOP_CODE"].dropna().astype(str).unique()))


# ============================================================
# PRODUCT INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("5. PRODUCT INFORMATION")
print("=" * 80)

product_count = sales["SALES_PART"].nunique(dropna=True)

print(f"Number of unique sales product codes: {product_count:,}")


# ============================================================
# DATE RANGE
# ============================================================

print("\n" + "=" * 80)
print("6. DATE RANGE")
print("=" * 80)

sales_dates = pd.to_datetime(
    sales["SALES_DATE"],
    errors="coerce",
)

print(f"Minimum SALES_DATE: {sales_dates.min()}")
print(f"Maximum SALES_DATE: {sales_dates.max()}")
print(f"Unique calendar dates: {sales_dates.dt.date.nunique():,}")


# ============================================================
# TOTAL_UNITS CHECKS
# ============================================================

print("\n" + "=" * 80)
print("7. TOTAL_UNITS CHECKS")
print("=" * 80)

negative_units = (sales["TOTAL_UNITS"] < 0).sum()
zero_units = (sales["TOTAL_UNITS"] == 0).sum()
positive_units = (sales["TOTAL_UNITS"] > 0).sum()

print(f"Negative TOTAL_UNITS rows: {negative_units:,}")
print(f"Zero TOTAL_UNITS rows: {zero_units:,}")
print(f"Positive TOTAL_UNITS rows: {positive_units:,}")


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("8. DUPLICATE INFORMATION")
print("=" * 80)

duplicate_rows = sales.duplicated().sum()

print(f"Exact duplicate / repeated rows: {duplicate_rows:,}")

print(
    "\nIMPORTANT: These rows are only being reported."
    "\nNothing is being deleted."
)


# ============================================================
# OPTIONAL EXTRA CHECKS
# ============================================================

print("\n" + "=" * 80)
print("9. EXTRA USEFUL CHECKS")
print("=" * 80)

missing_family = sales["FAM_DES"].isna().sum()

print(f"Missing FAM_DES rows: {missing_family:,}")

positive_units_zero_value = (
    (sales["TOTAL_UNITS"] > 0)
    & (sales["TOTAL_VALUE"] == 0)
).sum()

print(
    "Positive TOTAL_UNITS but zero TOTAL_VALUE rows: "
    f"{positive_units_zero_value:,}"
)


# ============================================================
# DOWNLOADABLE EXCEL REPORT
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

summary = pd.DataFrame(
    {
        "Metric": [
            "Source file",
            "Sheet",
            "Rows",
            "Columns",
            "Unique shops",
            "Unique product codes",
            "Minimum date",
            "Maximum date",
            "Unique calendar dates",
            "Negative TOTAL_UNITS rows",
            "Zero TOTAL_UNITS rows",
            "Positive TOTAL_UNITS rows",
            "Exact duplicate rows",
            "Missing FAM_DES rows",
            "Positive units with zero value",
        ],
        "Value": [
            SALES_FILE.name,
            SHEET_NAME,
            len(sales),
            len(sales.columns),
            shop_count,
            product_count,
            sales_dates.min(),
            sales_dates.max(),
            sales_dates.dt.date.nunique(),
            negative_units,
            zero_units,
            positive_units,
            duplicate_rows,
            missing_family,
            positive_units_zero_value,
        ],
    }
)

data_types = pd.DataFrame(
    {"Column": sales.columns, "Data type": sales.dtypes.astype(str).values}
)

missing_values = pd.DataFrame(
    {"Column": missing.index, "Missing values": missing.values}
)

checks = pd.DataFrame(
    {
        "Check": [
            "Negative TOTAL_UNITS",
            "Zero TOTAL_UNITS",
            "Positive TOTAL_UNITS",
            "Exact duplicate rows",
            "Missing FAM_DES",
            "Positive TOTAL_UNITS with zero TOTAL_VALUE",
        ],
        "Count": [
            negative_units,
            zero_units,
            positive_units,
            duplicate_rows,
            missing_family,
            positive_units_zero_value,
        ],
    }
)

shop_codes = pd.DataFrame(
    {"SHOP_CODE": sorted(sales["SHOP_CODE"].dropna().astype(str).unique())}
)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    report_sheets = {
        "Summary": summary,
        "Data_Types": data_types,
        "Missing_Values": missing_values,
        "Checks": checks,
        "Shop_Codes": shop_codes,
        "Sample_Rows": sales.head(100),
    }

    for sheet_name, report in report_sheets.items():
        report.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_cells in worksheet.columns:
            width = min(
                max(len(str(cell.value or "")) for cell in column_cells) + 2,
                40,
            )
            worksheet.column_dimensions[column_cells[0].column_letter].width = width

print(f"Downloadable Excel report created at:\n{OUTPUT_FILE}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SALES PROFILING SUMMARY")
print("=" * 80)

print(f"""
File: {SALES_FILE.name}
Sheet: {SHEET_NAME}

Rows: {len(sales):,}
Columns: {len(sales.columns)}

Unique shops: {shop_count}
Unique product codes: {product_count:,}

Minimum date: {sales_dates.min()}
Maximum date: {sales_dates.max()}

Negative TOTAL_UNITS: {negative_units:,}
Zero TOTAL_UNITS: {zero_units:,}

Exact repeated rows: {duplicate_rows:,}
Missing FAM_DES: {missing_family:,}

NO DATA WAS DELETED OR MODIFIED.
""")
