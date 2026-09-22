"""Small, read-only source-data validators for the Singer FYP MVP.

These functions validate incoming source data before later ingestion work. They
report problems but never modify the supplied DataFrames or source files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml


@dataclass(frozen=True)
class ValidationResult:
    """One structural validation outcome."""

    dataset: str
    rule_id: str
    severity: str
    passed: bool
    issue: str
    count: int
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


SOURCE_DATASETS = ("sales", "inventory", "products", "calendar", "branches")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_contract(
    dataset: str, contracts_root: Path | None = None
) -> dict[str, Any]:
    """Load the current v1 contract for one logical source dataset."""
    root = contracts_root or _project_root() / "contracts"
    contract_path = root / dataset / "v1.yaml"
    with contract_path.open(encoding="utf-8") as contract_file:
        contract = yaml.safe_load(contract_file)

    if not isinstance(contract, dict):
        raise ValueError(f"Invalid contract content: {contract_path}")
    if contract.get("dataset") != dataset:
        raise ValueError(
            f"Contract dataset mismatch in {contract_path}: "
            f"expected {dataset!r}, found {contract.get('dataset')!r}"
        )
    return contract


def _result(
    dataset: str,
    rule_id: str,
    severity: str,
    passed: bool,
    issue: str,
    count: int,
    notes: str = "",
) -> ValidationResult:
    return ValidationResult(
        dataset=dataset,
        rule_id=rule_id,
        severity=severity,
        passed=passed,
        issue=issue,
        count=int(count),
        notes=notes,
    )


def _missing_or_blank(series: pd.Series) -> pd.Series:
    """Return a temporary mask for null, empty, or whitespace-only values."""
    missing = series.isna()
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        missing = missing | series.astype("string").str.strip().eq("")
    return missing


def _required_column_result(
    dataset: str,
    rule_id: str,
    frame: pd.DataFrame,
    column: str,
    contract: Mapping[str, Any],
) -> ValidationResult:
    contract_requires_column = column in contract.get("required_columns", [])
    if not contract_requires_column:
        return _result(
            dataset,
            rule_id,
            "HIGH",
            False,
            f"{column} is not marked required in the current contract",
            1,
            "Validator/contract discrepancy; review the contract before ingestion.",
        )

    exists = column in frame.columns
    return _result(
        dataset,
        rule_id,
        "HIGH",
        exists,
        f"Required column {column} is missing",
        0 if exists else 1,
        "Required by the current v1 source contract.",
    )


def _key_not_missing_result(
    dataset: str,
    rule_id: str,
    frame: pd.DataFrame,
    column: str,
) -> ValidationResult:
    if column not in frame.columns:
        return _result(
            dataset,
            rule_id,
            "HIGH",
            False,
            f"Cannot check missing values because {column} is absent",
            1,
            "The required-column rule also reports this schema failure.",
        )

    count = int(_missing_or_blank(frame[column]).sum())
    return _result(
        dataset,
        rule_id,
        "HIGH",
        count == 0,
        f"{column} contains null, empty, or whitespace-only values",
        count,
        "Values are checked without trimming or filling the source column.",
    )


def _numeric_result(
    dataset: str,
    rule_id: str,
    frame: pd.DataFrame,
    column: str,
    severity: str = "HIGH",
    notes: str = "",
) -> ValidationResult:
    if column not in frame.columns:
        return _result(
            dataset,
            rule_id,
            severity,
            False,
            f"Cannot validate numeric values because {column} is absent",
            1,
            "The required-column rule also reports this schema failure.",
        )

    original_present = ~_missing_or_blank(frame[column])
    converted = pd.to_numeric(frame[column], errors="coerce")
    count = int((original_present & converted.isna()).sum())
    return _result(
        dataset,
        rule_id,
        severity,
        count == 0,
        f"{column} contains non-numeric values",
        count,
        notes or "Safe coercion is used for checking only; the source column is unchanged.",
    )


def _optional_numeric_result(
    dataset: str,
    rule_id: str,
    frame: pd.DataFrame,
    column: str,
    contract: Mapping[str, Any],
    notes: str,
) -> ValidationResult:
    """Validate a numeric field only when its optional source column exists."""
    if column not in frame.columns and column not in contract.get("required_columns", []):
        return _result(
            dataset,
            rule_id,
            "WARNING",
            True,
            f"Optional column {column} is absent",
            0,
            notes,
        )
    return _numeric_result(
        dataset,
        rule_id,
        frame,
        column,
        severity="WARNING",
        notes=notes,
    )


def _date_result(
    dataset: str,
    rule_id: str,
    frame: pd.DataFrame,
    column: str,
) -> ValidationResult:
    if column not in frame.columns:
        return _result(
            dataset,
            rule_id,
            "HIGH",
            False,
            f"Cannot validate dates because {column} is absent",
            1,
            "The required-column rule also reports this schema failure.",
        )

    original_present = ~_missing_or_blank(frame[column])
    converted = pd.to_datetime(frame[column], errors="coerce")
    count = int((original_present & converted.isna()).sum())
    return _result(
        dataset,
        rule_id,
        "HIGH",
        count == 0,
        f"{column} contains unparseable dates",
        count,
        "Safe parsing is used for checking only; the source column is unchanged.",
    )


def _unique_result(
    dataset: str,
    rule_id: str,
    frame: pd.DataFrame,
    column: str,
) -> ValidationResult:
    if column not in frame.columns:
        return _result(
            dataset,
            rule_id,
            "HIGH",
            False,
            f"Cannot check uniqueness because {column} is absent",
            1,
            "The required-column rule also reports this schema failure.",
        )

    count = int(frame.duplicated(subset=[column], keep="first").sum())
    return _result(
        dataset,
        rule_id,
        "HIGH",
        count == 0,
        f"{column} has duplicate values",
        count,
        "Duplicates are reported only; no source rows are removed.",
    )


def _contract_column_for_role(
    contract: Mapping[str, Any], role: str, fallback: str
) -> str:
    for name, definition in contract.get("columns", {}).items():
        if isinstance(definition, Mapping) and definition.get("role") == role:
            return name
    return fallback


def validate_sales(
    frame: pd.DataFrame, contract: Mapping[str, Any] | None = None
) -> list[ValidationResult]:
    """Validate the structural sales requirements without changing observations."""
    contract = contract or load_contract("sales")
    results = [
        _required_column_result("sales", "VAL-SALES-001", frame, "SALES_DATE", contract),
        _required_column_result("sales", "VAL-SALES-002", frame, "SHOP_CODE", contract),
        _required_column_result("sales", "VAL-SALES-003", frame, "SALES_PART", contract),
        _required_column_result("sales", "VAL-SALES-004", frame, "TOTAL_UNITS", contract),
        _numeric_result(
            "sales",
            "VAL-SALES-005",
            frame,
            "TOTAL_UNITS",
            notes=(
                "Negative and zero observations are numeric and remain valid for "
                "this structural check; values are not clipped or changed."
            ),
        ),
        _required_column_result("sales", "VAL-SALES-006", frame, "TOTAL_VALUE", contract),
        _numeric_result("sales", "VAL-SALES-007", frame, "TOTAL_VALUE"),
        _date_result("sales", "VAL-SALES-008", frame, "SALES_DATE"),
        _key_not_missing_result("sales", "VAL-SALES-009", frame, "SHOP_CODE"),
        _key_not_missing_result("sales", "VAL-SALES-010", frame, "SALES_PART"),
    ]
    return results


def validate_products(
    frame: pd.DataFrame, contract: Mapping[str, Any] | None = None
) -> list[ValidationResult]:
    """Validate product-master identity and price type without altering prices."""
    contract = contract or load_contract("products")
    return [
        _required_column_result("products", "VAL-PROD-001", frame, "PRODUCT_CODE", contract),
        _key_not_missing_result("products", "VAL-PROD-002", frame, "PRODUCT_CODE"),
        _unique_result("products", "VAL-PROD-003", frame, "PRODUCT_CODE"),
        _required_column_result("products", "VAL-PROD-004", frame, "PRODUCT_DESC", contract),
        _required_column_result(
            "products", "VAL-PROD-005", frame, "PRODUCT_FAMILY_ID", contract
        ),
        _optional_numeric_result(
            "products",
            "VAL-PROD-006",
            frame,
            "PRICE",
            contract,
            notes=(
                "Missing PRICE is allowed by the current contract. PRICE is "
                "list-price evidence, not cost, margin, or profit."
            ),
        ),
    ]


def validate_inventory(
    frame: pd.DataFrame, contract: Mapping[str, Any] | None = None
) -> list[ValidationResult]:
    """Validate the inventory snapshot schema without inferring stock values."""
    contract = contract or load_contract("inventory")
    key_columns = ("AS_AT_DATE", "SITE_ID", "PRODUCT_CODE")
    missing_key_cells = sum(
        int(_missing_or_blank(frame[column]).sum())
        for column in key_columns
        if column in frame.columns
    )
    absent_key_columns = [column for column in key_columns if column not in frame.columns]
    return [
        _required_column_result("inventory", "VAL-INV-001", frame, "AS_AT_DATE", contract),
        _required_column_result("inventory", "VAL-INV-002", frame, "SITE_ID", contract),
        _required_column_result("inventory", "VAL-INV-003", frame, "PRODUCT_CODE", contract),
        _required_column_result("inventory", "VAL-INV-004", frame, "QTY", contract),
        _date_result("inventory", "VAL-INV-005", frame, "AS_AT_DATE"),
        _numeric_result(
            "inventory",
            "VAL-INV-006",
            frame,
            "QTY",
            notes=(
                "Negative and zero quantities are not rejected by this structural "
                "validator; the source column is unchanged."
            ),
        ),
        _result(
            "inventory",
            "VAL-INV-007",
            "HIGH",
            not absent_key_columns and missing_key_cells == 0,
            "Critical inventory key fields contain missing values",
            len(absent_key_columns) if absent_key_columns else missing_key_cells,
            (
                "Missing inventory rows are not inferred as zero stock."
                if not absent_key_columns
                else "Required key columns are absent: " + ", ".join(absent_key_columns)
            ),
        ),
    ]


def validate_calendar(
    frame: pd.DataFrame, contract: Mapping[str, Any] | None = None
) -> list[ValidationResult]:
    """Validate the one-row-per-date calendar structure."""
    contract = contract or load_contract("calendar")
    date_column = _contract_column_for_role(contract, "primary_date_key", "DATE")
    return [
        _required_column_result("calendar", "VAL-CAL-001", frame, date_column, contract),
        _key_not_missing_result("calendar", "VAL-CAL-002", frame, date_column),
        _date_result("calendar", "VAL-CAL-003", frame, date_column),
        _unique_result("calendar", "VAL-CAL-004", frame, date_column),
    ]


def validate_branches(
    frame: pd.DataFrame, contract: Mapping[str, Any] | None = None
) -> list[ValidationResult]:
    """Validate the one-row-per-branch profile structure."""
    contract = contract or load_contract("branches")
    branch_column = _contract_column_for_role(contract, "branch_primary_key", "SHOP_CODE")
    return [
        _required_column_result("branches", "VAL-BR-001", frame, branch_column, contract),
        _key_not_missing_result("branches", "VAL-BR-002", frame, branch_column),
        _unique_result("branches", "VAL-BR-003", frame, branch_column),
    ]


def validate_all_datasets(
    *,
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    calendar: pd.DataFrame,
    branches: pd.DataFrame,
) -> list[ValidationResult]:
    """Run all five source validators and return one combined result list."""
    return [
        *validate_sales(sales),
        *validate_inventory(inventory),
        *validate_products(products),
        *validate_calendar(calendar),
        *validate_branches(branches),
    ]


def load_raw_source_datasets(raw_directory: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read the five contract-defined source workbooks without modifying them."""
    raw_directory = raw_directory or _project_root() / "data" / "raw"
    frames: dict[str, pd.DataFrame] = {}
    for dataset in SOURCE_DATASETS:
        contract = load_contract(dataset)
        frames[dataset] = pd.read_excel(
            raw_directory / contract["source_file"],
            sheet_name=contract.get("source_sheet", 0),
        )
    return frames


def validate_raw_sources(raw_directory: Path | None = None) -> list[ValidationResult]:
    """Load and validate all five current source workbooks in read-only mode."""
    frames = load_raw_source_datasets(raw_directory)
    return validate_all_datasets(**frames)


def validation_results_to_quality_rows(
    results: Iterable[ValidationResult],
) -> list[dict[str, Any]]:
    """Return rows compatible with the Step 12 data-quality-log column shape."""
    return [
        {
            "severity": result.severity,
            "dataset": result.dataset,
            "rule_id": result.rule_id,
            "issue": result.issue,
            "count": result.count,
            "status": "PASS" if result.passed else "OPEN",
            "notes": result.notes,
        }
        for result in results
    ]
