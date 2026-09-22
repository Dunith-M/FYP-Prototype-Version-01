"""Build the canonical daily sales fact before any inventory processing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.date_utils import normalize_sales_date
from analytics.services.data_quality import apply_data_quality_flags
from catalog.branch_master import normalize_branch_code
from catalog.models import Branch, SkuMapping
from catalog.sku_mapping import normalize_source_product_code


SALES_SHEET_NAME = "data"
REQUIRED_SALES_COLUMNS = (
    "SALES_DATE",
    "SHOP_CODE",
    "SALES_PART",
    "TOTAL_UNITS",
    "TOTAL_VALUE",
)
OPTIONAL_RECONCILIATION_COLUMNS = (
    "HIRE_UNITS",
    "HIRE_VAL",
    "CASH_UNITS",
    "CASH_VALUE",
)
SALES_DAY_KEY = ("branch_code", "canonical_product_id", "business_date")


class SalesDayValidationError(ValueError):
    """Raised when raw sales cannot safely form the canonical daily fact."""


@dataclass(frozen=True)
class SalesDayMetrics:
    total_source_rows: int
    resolved_product_rows: int
    unresolved_product_rows: int
    unresolved_source_codes: tuple[str, ...]
    negative_source_rows: int
    zero_unit_source_rows: int
    raw_resolved_signed_units: float
    aggregated_sales_units: float
    raw_resolved_total_value: float
    aggregated_sales_value: float
    duplicate_sales_day_keys: int

    @property
    def rows_reconciled(self) -> bool:
        return self.total_source_rows == (
            self.resolved_product_rows + self.unresolved_product_rows
        )

    @property
    def unit_difference(self) -> float:
        return self.raw_resolved_signed_units - self.aggregated_sales_units

    @property
    def value_difference(self) -> float:
        return self.raw_resolved_total_value - self.aggregated_sales_value


@dataclass(frozen=True)
class SalesDayBuildResult:
    sales_day: pd.DataFrame
    unresolved_sales: pd.DataFrame
    metrics: SalesDayMetrics


def _is_missing(value: object) -> bool:
    """Return a scalar missing-value check that also handles pandas values."""

    missing = pd.isna(value)
    return bool(missing) if not isinstance(missing, bool) else missing


def _normalized_branch_code(value: object) -> str:
    return "" if _is_missing(value) else normalize_branch_code(value)


def _normalized_source_product_code(value: object) -> str:
    return "" if _is_missing(value) else normalize_source_product_code(value)


def _normalized_business_date(value: object):
    result = normalize_sales_date(None if _is_missing(value) else value)
    return result.normalized_date, result.status


def _validate_required_columns(frame: pd.DataFrame, sales_file: Path) -> None:
    missing_columns = sorted(set(REQUIRED_SALES_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise SalesDayValidationError(
            f"{sales_file}: required sales columns are missing: {missing_columns}."
        )


def _validated_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    parsed = pd.to_numeric(frame[column], errors="coerce")
    invalid = parsed.isna() | ~pd.Series(parsed).map(
        lambda value: math.isfinite(float(value))
    )
    if invalid.any():
        raise SalesDayValidationError(
            f"Sales aggregation is blocked: {int(invalid.sum())} {column} values are "
            "missing, unparseable, or non-finite."
        )
    return parsed.astype(float)


def _validated_nullable_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Parse optional reconciliation evidence without replacing nulls with zero."""

    parsed = pd.to_numeric(frame[column], errors="coerce")
    supplied = frame[column].notna() & frame[column].astype("string").str.strip().ne("")
    invalid = supplied & parsed.isna()
    if invalid.any():
        raise SalesDayValidationError(
            f"Sales aggregation is blocked: {int(invalid.sum())} {column} values are unparseable."
        )
    return parsed.astype(float)


def _canonical_products_by_source_code() -> dict[str, str]:
    """Return only confirmed Sales mappings that have a canonical Product."""

    mappings = SkuMapping.objects.filter(
        source_system=SkuMapping.SourceSystem.SALES,
        confidence_class=SkuMapping.ConfidenceClass.CONFIRMED,
        canonical_product__isnull=False,
    ).select_related("canonical_product")
    return {
        mapping.source_product_code: mapping.canonical_product.canonical_product_id
        for mapping in mappings
    }


def _validate_source_dimensions(frame: pd.DataFrame) -> None:
    blank_branches = int((frame["branch_code"] == "").sum())
    if blank_branches:
        raise SalesDayValidationError(
            f"Sales aggregation is blocked: blank SHOP_CODE rows={blank_branches}."
        )

    official_branch_codes = set(Branch.objects.values_list("branch_code", flat=True))
    unmatched_branches = sorted(set(frame["branch_code"]) - official_branch_codes)
    if unmatched_branches:
        counts = {
            code: int((frame["branch_code"] == code).sum())
            for code in unmatched_branches
        }
        raise SalesDayValidationError(
            "Sales aggregation is blocked: SHOP_CODE values are absent from the "
            f"Branch master: {counts}."
        )

    blank_products = int((frame["source_product_code"] == "").sum())
    if blank_products:
        raise SalesDayValidationError(
            "Sales aggregation is blocked: "
            f"blank SALES_PART rows={blank_products}."
        )

    invalid_dates = frame["business_date"].isna()
    if invalid_dates.any():
        statuses = frame.loc[invalid_dates, "business_date_status"].value_counts().to_dict()
        raise SalesDayValidationError(
            "Sales aggregation is blocked: "
            f"invalid SALES_DATE rows={int(invalid_dates.sum())}; statuses={statuses}."
        )


def build_sales_day(sales_file: Path) -> SalesDayBuildResult:
    """Aggregate resolved historical sales at branch, canonical-product, day grain.

    Unresolved product rows are deliberately excluded from ``sales_day`` and
    returned unchanged enough for audit in ``unresolved_sales``.  This function
    neither reads nor combines any stock source.
    """

    raw_sales = pd.read_excel(sales_file, sheet_name=SALES_SHEET_NAME)
    _validate_required_columns(raw_sales, sales_file)
    working = raw_sales.copy()
    working.insert(0, "source_row_number", range(2, len(working) + 2))
    working["original_sales_timestamp"] = working["SALES_DATE"]
    working["branch_code"] = working["SHOP_CODE"].map(_normalized_branch_code)
    working["source_product_code"] = working["SALES_PART"].map(
        _normalized_source_product_code
    )

    dates = working["SALES_DATE"].map(_normalized_business_date)
    working["business_date"] = dates.map(lambda value: value[0])
    working["business_date_status"] = dates.map(lambda value: value[1])
    working["TOTAL_UNITS"] = _validated_numeric_column(working, "TOTAL_UNITS")
    working["TOTAL_VALUE"] = _validated_numeric_column(working, "TOTAL_VALUE")
    reconciliation_columns = [
        column for column in OPTIONAL_RECONCILIATION_COLUMNS if column in working.columns
    ]
    for column in reconciliation_columns:
        working[column] = _validated_nullable_numeric_column(working, column)
    _validate_source_dimensions(working)

    canonical_products = _canonical_products_by_source_code()
    working["canonical_product_id"] = working["source_product_code"].map(
        canonical_products
    )
    resolved_mask = working["canonical_product_id"].notna()
    resolved_sales = working.loc[resolved_mask].copy()
    unresolved_sales = working.loc[~resolved_mask].copy()

    aggregations: dict[str, tuple[str, object]] = {
        "sales_units": ("TOTAL_UNITS", "sum"),
        "sales_value": ("TOTAL_VALUE", "sum"),
        "source_row_count": ("source_row_number", "size"),
        "negative_unit_row_count": ("TOTAL_UNITS", lambda values: int((values < 0).sum())),
        "zero_unit_row_count": ("TOTAL_UNITS", lambda values: int((values == 0).sum())),
        "positive_unit_row_count": ("TOTAL_UNITS", lambda values: int((values > 0).sum())),
        "fulfilled_positive_units_proxy_day": (
            "TOTAL_UNITS",
            lambda values: values.where(values > 0, 0).sum(),
        ),
    }
    reconciliation_names = {
        "HIRE_UNITS": "hire_units",
        "HIRE_VAL": "hire_value",
        "CASH_UNITS": "cash_units",
        "CASH_VALUE": "cash_value",
    }
    for source_column in reconciliation_columns:
        canonical_name = reconciliation_names[source_column]
        aggregations[canonical_name] = (
            source_column,
            lambda values: values.sum(min_count=1),
        )
        aggregations[f"{canonical_name}_missing_source_row_count"] = (
            source_column,
            lambda values: int(values.isna().sum()),
        )
    sales_day = (
        resolved_sales.groupby(list(SALES_DAY_KEY), as_index=False, sort=True)
        .agg(**aggregations)
        .sort_values(list(SALES_DAY_KEY), kind="stable")
        .reset_index(drop=True)
    )
    sales_day["raw_total_units"] = sales_day["sales_units"]
    sales_day["raw_total_value"] = sales_day["sales_value"]
    if len(reconciliation_columns) == len(OPTIONAL_RECONCILIATION_COLUMNS):
        missing_count_columns = [
            f"{reconciliation_names[column]}_missing_source_row_count"
            for column in OPTIONAL_RECONCILIATION_COLUMNS
        ]
        sales_day["reconciliation_components_complete_flag"] = sales_day[
            missing_count_columns
        ].eq(0).all(axis=1)
    unresolved_sales = apply_data_quality_flags(unresolved_sales)
    duplicate_sales_day_keys = int(sales_day.duplicated(list(SALES_DAY_KEY)).sum())
    if duplicate_sales_day_keys:
        raise SalesDayValidationError(
            "Sales aggregation is blocked: duplicate SALES_DAY keys remain after "
            f"aggregation={duplicate_sales_day_keys}."
        )

    metrics = SalesDayMetrics(
        total_source_rows=len(working),
        resolved_product_rows=len(resolved_sales),
        unresolved_product_rows=len(unresolved_sales),
        unresolved_source_codes=tuple(
            sorted(unresolved_sales["source_product_code"].unique().tolist())
        ),
        negative_source_rows=int((working["TOTAL_UNITS"] < 0).sum()),
        zero_unit_source_rows=int((working["TOTAL_UNITS"] == 0).sum()),
        raw_resolved_signed_units=float(resolved_sales["TOTAL_UNITS"].sum()),
        aggregated_sales_units=float(sales_day["sales_units"].sum()),
        raw_resolved_total_value=float(resolved_sales["TOTAL_VALUE"].sum()),
        aggregated_sales_value=float(sales_day["sales_value"].sum()),
        duplicate_sales_day_keys=duplicate_sales_day_keys,
    )
    return SalesDayBuildResult(sales_day, unresolved_sales, metrics)


def write_sales_day_outputs(
    result: SalesDayBuildResult,
    sales_day_path: Path,
    unresolved_sales_path: Path,
) -> None:
    """Rebuild deterministic Parquet outputs without appending prior runs."""

    sales_day_path.parent.mkdir(parents=True, exist_ok=True)
    unresolved_sales_path.parent.mkdir(parents=True, exist_ok=True)
    result.sales_day.to_parquet(sales_day_path, index=False)
    result.unresolved_sales.to_parquet(unresolved_sales_path, index=False)
