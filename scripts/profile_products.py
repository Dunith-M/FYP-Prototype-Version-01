from pathlib import Path

import pandas as pd


# ============================================================
# FILE PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRODUCT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "product_data.xlsx"
)

SHEET_NAME = "product_data.csv"


# ============================================================
# CHECK FILE EXISTS
# ============================================================

print("=" * 80)
print("PRODUCT MASTER PROFILING")
print("=" * 80)

print(f"\nFile path:\n{PRODUCT_FILE}")

if not PRODUCT_FILE.exists():
    raise FileNotFoundError(
        f"\nProduct file was not found:\n{PRODUCT_FILE}\n"
        "Make sure product_data.xlsx is inside data/raw/"
    )


# ============================================================
# LOAD PRODUCT MASTER
# ============================================================

products = pd.read_excel(
    PRODUCT_FILE,
    sheet_name=SHEET_NAME,
)


# ============================================================
# 1. BASIC STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("1. BASIC STRUCTURE")
print("=" * 80)

print(f"Sheet name: {SHEET_NAME}")
print(f"Rows: {len(products):,}")
print(f"Columns: {len(products.columns)}")

print("\nColumn names:")
for column in products.columns:
    print(f" - {column}")


# ============================================================
# 2. DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("2. DATA TYPES")
print("=" * 80)

print(products.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("3. MISSING VALUES")
print("=" * 80)

missing_values = products.isna().sum()

for column, count in missing_values.items():
    print(f"{column}: {count:,}")


# ============================================================
# 4. PRODUCT CODE CHECKS
# ============================================================

print("\n" + "=" * 80)
print("4. PRODUCT CODE CHECKS")
print("=" * 80)

number_of_products = products["PRODUCT_CODE"].nunique(dropna=True)

duplicate_product_codes = products["PRODUCT_CODE"].duplicated().sum()

missing_product_codes = products["PRODUCT_CODE"].isna().sum()

print(f"Number of unique products: {number_of_products:,}")
print(f"Duplicate PRODUCT_CODE rows: {duplicate_product_codes:,}")
print(f"Missing PRODUCT_CODE rows: {missing_product_codes:,}")


# ============================================================
# 5. PRICE CHECKS
# ============================================================

print("\n" + "=" * 80)
print("5. PRICE QUALITY CHECKS")
print("=" * 80)

missing_price = products["PRICE"].isna().sum()

zero_price = (products["PRICE"] == 0).sum()

negative_price = (products["PRICE"] < 0).sum()

positive_price = (products["PRICE"] > 0).sum()

print(f"Missing PRICE rows: {missing_price:,}")
print(f"PRICE = 0 rows: {zero_price:,}")
print(f"PRICE < 0 rows: {negative_price:,}")
print(f"PRICE > 0 rows: {positive_price:,}")


# ============================================================
# 6. PRODUCT FAMILY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("6. PRODUCT FAMILY CHECKS")
print("=" * 80)

family_id_count = products["PRODUCT_FAMILY_ID"].nunique(dropna=True)

family_desc_count = products["PRODUCT_FAMILY_DESC"].nunique(dropna=True)

missing_family_id = products["PRODUCT_FAMILY_ID"].isna().sum()

missing_family_desc = products["PRODUCT_FAMILY_DESC"].isna().sum()

print(f"Number of PRODUCT_FAMILY_ID values: {family_id_count:,}")
print(f"Number of PRODUCT_FAMILY_DESC values: {family_desc_count:,}")
print(f"Missing PRODUCT_FAMILY_ID rows: {missing_family_id:,}")
print(f"Missing PRODUCT_FAMILY_DESC rows: {missing_family_desc:,}")


# ============================================================
# 7. EXACT DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 80)
print("7. EXACT DUPLICATE ROWS")
print("=" * 80)

exact_duplicates = products.duplicated().sum()

print(f"Exact duplicate rows: {exact_duplicates:,}")


# ============================================================
# 8. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 80)
print("8. UNIQUE VALUES PER COLUMN")
print("=" * 80)

print(products.nunique(dropna=True))


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PRODUCT MASTER PROFILING SUMMARY")
print("=" * 80)

print(f"""
File: {PRODUCT_FILE.name}
Sheet: {SHEET_NAME}

Rows: {len(products):,}
Columns: {len(products.columns)}

Unique products: {number_of_products:,}
Duplicate PRODUCT_CODE rows: {duplicate_product_codes:,}
Missing PRODUCT_CODE rows: {missing_product_codes:,}

Missing PRICE rows: {missing_price:,}
PRICE = 0 rows: {zero_price:,}
PRICE < 0 rows: {negative_price:,}
PRICE > 0 rows: {positive_price:,}

Product families: {family_id_count:,}

Exact duplicate rows: {exact_duplicates:,}

NO DATA WAS CHANGED.
NO ROWS WERE DELETED.
NO PRICES WERE FIXED.
""")