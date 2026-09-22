from pathlib import Path

import pandas as pd


# ============================================================
# FILE PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BRANCH_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "branch_profile.xlsx"
)

SHEET_NAME = "Branch Profile"


# ============================================================
# EXPECTED PILOT BRANCH CODES
# ============================================================

EXPECTED_BRANCH_CODES = {
    "BIL01",
    "BIN01",
    "BIP01",
    "BIR01",
    "BIS01",
    "BIY01",
    "BKL01",
    "BMU01",
    "BOL01",
    "SBO01",
}


# ============================================================
# CHECK FILE
# ============================================================

print("=" * 80)
print("BRANCH PROFILE DATA PROFILING")
print("=" * 80)

print(f"\nFile path:\n{BRANCH_FILE}")

if not BRANCH_FILE.exists():
    raise FileNotFoundError(
        f"\nBranch profile file not found:\n{BRANCH_FILE}\n"
        "Make sure Singer_Galle_Branch_Profile(1).xlsx is inside data/raw/"
    )


# ============================================================
# LOAD DATA
# ============================================================

branches = pd.read_excel(
    BRANCH_FILE,
    sheet_name=SHEET_NAME,
)


# ============================================================
# 1. BASIC STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("1. BASIC STRUCTURE")
print("=" * 80)

print(f"Sheet name: {SHEET_NAME}")
print(f"Rows: {len(branches):,}")
print(f"Columns: {len(branches.columns)}")

print("\nColumn names:")
for column in branches.columns:
    print(f" - {column}")


# ============================================================
# 2. DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("2. DATA TYPES")
print("=" * 80)

print(branches.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("3. MISSING VALUES")
print("=" * 80)

for column, count in branches.isna().sum().items():
    print(f"{column}: {count:,}")


# ============================================================
# 4. SHOP_CODE CHECKS
# ============================================================

print("\n" + "=" * 80)
print("4. SHOP_CODE CHECKS")
print("=" * 80)

missing_shop_codes = branches["SHOP_CODE"].isna().sum()
unique_shop_codes = branches["SHOP_CODE"].nunique(dropna=True)

duplicate_shop_code_rows = branches.duplicated(
    subset=["SHOP_CODE"],
    keep=False,
).sum()

excess_duplicate_shop_codes = branches.duplicated(
    subset=["SHOP_CODE"],
    keep="first",
).sum()

print(f"Missing SHOP_CODE rows: {missing_shop_codes:,}")
print(f"Unique SHOP_CODE values: {unique_shop_codes:,}")
print(
    "Rows involved in duplicated SHOP_CODE values: "
    f"{duplicate_shop_code_rows:,}"
)
print(
    "Excess duplicate SHOP_CODE rows: "
    f"{excess_duplicate_shop_codes:,}"
)


# ============================================================
# 5. NORMALIZED BRANCH CODES
# ============================================================

print("\n" + "=" * 80)
print("5. BRANCH CODE VALUES")
print("=" * 80)

normalized_codes = (
    branches["SHOP_CODE"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
)

actual_branch_codes = set(normalized_codes)

print("Branch codes found:")
for code in sorted(actual_branch_codes):
    print(f" - {code}")


# ============================================================
# 6. EXPECTED PILOT CODE CHECK
# ============================================================

print("\n" + "=" * 80)
print("6. EXPECTED PILOT BRANCH CHECK")
print("=" * 80)

missing_expected_codes = EXPECTED_BRANCH_CODES - actual_branch_codes
unexpected_codes = actual_branch_codes - EXPECTED_BRANCH_CODES

print(f"Expected pilot branch codes: {len(EXPECTED_BRANCH_CODES)}")
print(f"Actual unique branch codes: {len(actual_branch_codes)}")

print("\nMissing expected branch codes:")
if missing_expected_codes:
    for code in sorted(missing_expected_codes):
        print(f" - {code}")
else:
    print(" None")

print("\nUnexpected branch codes:")
if unexpected_codes:
    for code in sorted(unexpected_codes):
        print(f" - {code}")
else:
    print(" None")


# ============================================================
# 7. BRANCH NAME CHECK
# ============================================================

print("\n" + "=" * 80)
print("7. BRANCH NAME CHECK")
print("=" * 80)

if "SHOP_NAME" in branches.columns:
    missing_branch_names = branches["SHOP_NAME"].isna().sum()

    print(f"Missing SHOP_NAME rows: {missing_branch_names:,}")

    print("\nBranch code -> branch name:")
    branch_names = branches[
        ["SHOP_CODE", "SHOP_NAME"]
    ].copy()

    for _, row in branch_names.iterrows():
        print(
            f" - {row['SHOP_CODE']} -> {row['SHOP_NAME']}"
        )
else:
    print("WARNING: SHOP_NAME column not found.")


# ============================================================
# 8. BRANCH ATTRIBUTE OVERVIEW
# ============================================================

print("\n" + "=" * 80)
print("8. BRANCH ATTRIBUTE OVERVIEW")
print("=" * 80)

attribute_columns = [
    column
    for column in branches.columns
    if column not in {"SHOP_CODE", "SHOP_NAME"}
]

print(f"Number of branch attribute columns: {len(attribute_columns)}")

for column in attribute_columns:
    missing = branches[column].isna().sum()
    unique = branches[column].nunique(dropna=True)

    print(
        f"{column}: "
        f"missing={missing:,}, "
        f"unique={unique:,}"
    )


# ============================================================
# 9. EXACT DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 80)
print("9. EXACT DUPLICATE ROWS")
print("=" * 80)

exact_duplicates = branches.duplicated().sum()

print(f"Exact duplicate rows: {exact_duplicates:,}")


# ============================================================
# 10. ACCEPTANCE CHECK
# ============================================================

print("\n" + "=" * 80)
print("10. STEP 9 ACCEPTANCE CHECK")
print("=" * 80)

passes_branch_count = unique_shop_codes == 10
passes_missing_codes = missing_shop_codes == 0
passes_duplicates = excess_duplicate_shop_codes == 0
passes_expected_set = (
    actual_branch_codes == EXPECTED_BRANCH_CODES
)

print(f"10 unique branch codes: {passes_branch_count}")
print(f"No missing SHOP_CODE: {passes_missing_codes}")
print(f"No duplicate SHOP_CODE: {passes_duplicates}")
print(f"Matches expected pilot codes: {passes_expected_set}")


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("BRANCH PROFILE SUMMARY")
print("=" * 80)

print(f"""
File: {BRANCH_FILE.name}
Sheet: {SHEET_NAME}

Rows: {len(branches):,}
Columns: {len(branches.columns)}

Unique branch codes: {unique_shop_codes:,}
Missing SHOP_CODE rows: {missing_shop_codes:,}
Duplicate SHOP_CODE rows: {excess_duplicate_shop_codes:,}

Expected pilot branches: {len(EXPECTED_BRANCH_CODES)}
Missing expected branches: {len(missing_expected_codes)}
Unexpected branches: {len(unexpected_codes)}

Exact duplicate rows: {exact_duplicates:,}

NO DATA WAS CHANGED.
NO BRANCH CODE WAS CORRECTED.
NO MISSING ATTRIBUTE WAS FILLED.
""")