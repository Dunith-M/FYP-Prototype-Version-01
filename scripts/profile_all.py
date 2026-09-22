from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/artifacts/data_profile_report.xlsx")


FILES = {
    "sales": RAW_DATA_DIR / "transactions_data.xlsx",
    "inventory": RAW_DATA_DIR / "inventory_data.xlsx",
    "products": RAW_DATA_DIR / "product_data.xlsx",
    "calendar": RAW_DATA_DIR / "calendar_data.xlsx",
    "branches": RAW_DATA_DIR / "branch_profile.xlsx",
}

# Support the existing misspelled sales filename if the corrected name is absent.
if not FILES["sales"].exists():
    misspelled_sales_file = RAW_DATA_DIR / "tansactions_data.xlsx"
    if misspelled_sales_file.exists():
        FILES["sales"] = misspelled_sales_file


def profile_dataset(name, file_path):
    print("\n" + "=" * 80)
    print(f"DATASET: {name.upper()}")
    print(f"FILE: {file_path}")
    print("=" * 80)

    df = pd.read_excel(file_path)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nCOLUMN NAMES")
    print(df.columns.tolist())

    print("\nDATA TYPES")
    print(df.dtypes)

    print("\nMISSING VALUES")
    print(df.isna().sum())

    print("\nUNIQUE VALUES PER COLUMN")
    print(df.nunique())

    print("\nFIRST 5 ROWS")
    print(df.head())

    return {
        "summary": pd.DataFrame(
            {
                "Metric": ["Dataset", "Source file", "Rows", "Columns"],
                "Value": [name, file_path.name, len(df), len(df.columns)],
            }
        ),
        "data_types": pd.DataFrame(
            {"Column": df.columns, "Data type": df.dtypes.astype(str).values}
        ),
        "missing_values": pd.DataFrame(
            {"Column": df.isna().sum().index, "Missing values": df.isna().sum().values}
        ),
        "unique_values": pd.DataFrame(
            {"Column": df.nunique().index, "Unique values": df.nunique().values}
        ),
        "sample_rows": df.head(100),
    }


def main():
    reports = {}

    for name, file_path in FILES.items():
        reports[name] = profile_dataset(name, file_path)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for dataset_name, report in reports.items():
            for report_name, data in report.items():
                sheet_name = f"{dataset_name}_{report_name}"[:31]
                data.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes = "A2"
                worksheet.auto_filter.ref = worksheet.dimensions

                for column_cells in worksheet.columns:
                    width = min(
                        max(len(str(cell.value or "")) for cell in column_cells) + 2,
                        40,
                    )
                    worksheet.column_dimensions[
                        column_cells[0].column_letter
                    ].width = width

    print(f"\nDownloadable Excel report created at:\n{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
