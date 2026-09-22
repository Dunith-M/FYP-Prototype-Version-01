"""Validated product, calendar, and branch context features for monthly rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analytics.date_utils import normalize_calendar_date, year_month_from_calendar_date


MONTHLY_KEY = ("branch_code", "canonical_product_id", "year_month")


class ContextFeatureValidationError(ValueError):
    """Raised when a reference join cannot preserve the monthly analytical grain."""


@dataclass(frozen=True)
class ContextFeatureMetrics:
    input_rows: int
    output_rows: int
    unique_products: int
    unique_branches: int
    months: int
    invalid_price_rows: int
    missing_family_rows: int
    duplicate_product_keys: int
    duplicate_branch_keys: int
    duplicate_monthly_keys: int


@dataclass(frozen=True)
class ContextFeatureBuildResult:
    features: pd.DataFrame
    monthly_calendar: pd.DataFrame
    metrics: ContextFeatureMetrics


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ContextFeatureValidationError(f"{label} is missing columns: {missing}.")


def _require_unique(frame: pd.DataFrame, keys: tuple[str, ...], label: str) -> None:
    duplicates = int(frame.duplicated(list(keys)).sum())
    if duplicates:
        raise ContextFeatureValidationError(f"{label} has duplicate keys={duplicates}: {list(keys)}.")


def _as_boolean(values: pd.Series, label: str) -> pd.Series:
    normalized = values.astype("string").str.strip().str.upper()
    truthy = normalized.isin({"TRUE", "T", "YES", "Y", "1"})
    falsey = normalized.isin({"FALSE", "F", "NO", "N", "0", ""}) | normalized.isna()
    invalid = ~(truthy | falsey)
    if invalid.any():
        examples = sorted(normalized.loc[invalid].dropna().unique().tolist())[:3]
        raise ContextFeatureValidationError(f"{label} contains unsupported boolean values: {examples}.")
    return truthy.astype(bool)


def load_validated_calendar_daily(calendar_path: Path) -> pd.DataFrame:
    """Load the existing daily calendar reference without changing its source values."""

    try:
        calendar = pd.read_excel(calendar_path, sheet_name="Sheet1")
    except (OSError, ValueError) as error:
        raise ContextFeatureValidationError(f"Unable to read calendar reference: {calendar_path}.") from error
    _require_columns(calendar, ("DATE", "IS_WEEKEND", "IS_PUBLIC_HOLIDAY"), "Calendar")
    parsed = calendar["DATE"].map(normalize_calendar_date)
    calendar = calendar.copy(deep=True)
    calendar["calendar_date"] = parsed.map(lambda item: item.normalized_date)
    if calendar["calendar_date"].isna().any():
        raise ContextFeatureValidationError("Calendar contains null or invalid DATE values.")
    _require_unique(calendar, ("calendar_date",), "Calendar")
    calendar["year_month"] = calendar["calendar_date"].map(year_month_from_calendar_date)
    return calendar


def build_monthly_calendar_features(calendar_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the validated daily calendar to one safe row per ``year_month``."""

    _require_columns(calendar_daily, ("calendar_date", "year_month", "IS_WEEKEND", "IS_PUBLIC_HOLIDAY"), "Calendar")
    calendar = calendar_daily.copy(deep=True)
    _require_unique(calendar, ("calendar_date",), "Calendar")
    calendar["_weekend"] = _as_boolean(calendar["IS_WEEKEND"], "Calendar IS_WEEKEND")
    calendar["_holiday"] = _as_boolean(calendar["IS_PUBLIC_HOLIDAY"], "Calendar IS_PUBLIC_HOLIDAY")
    derived_month = pd.to_datetime(calendar["calendar_date"], errors="coerce").dt.month
    if derived_month.isna().any():
        raise ContextFeatureValidationError("Calendar contains invalid calendar_date values.")
    calendar["_month"] = derived_month.astype(int)
    calendar["_quarter"] = ((calendar["_month"] - 1) // 3 + 1).astype(int)
    monthly = (
        calendar.groupby("year_month", as_index=False, sort=True)
        .agg(
            month=("_month", "first"),
            quarter=("_quarter", "first"),
            holiday_count=("_holiday", "sum"),
            weekend_count=("_weekend", "sum"),
        )
        .sort_values("year_month", kind="stable")
        .reset_index(drop=True)
    )
    # Derivation from the approved calendar date is the single month/quarter rule.
    _require_unique(monthly, ("year_month",), "Monthly calendar")
    monthly["holiday_count"] = monthly["holiday_count"].astype(int)
    monthly["weekend_count"] = monthly["weekend_count"].astype(int)
    return monthly


def _price_buckets(product_master: pd.DataFrame) -> pd.Series:
    valid = product_master["valid_price"]
    buckets = pd.Series("INVALID_OR_MISSING_PRICE", index=product_master.index, dtype="string")
    valid_prices = pd.to_numeric(product_master.loc[valid, "list_price"], errors="coerce")
    if valid_prices.empty:
        return buckets
    if valid_prices.nunique() == 1:
        buckets.loc[valid] = "PRICE_Q1"
        return buckets
    # Research-only quartiles are deterministic and do not imply commercial price policy.
    codes = pd.qcut(valid_prices, q=4, labels=False, duplicates="drop")
    buckets.loc[valid] = codes.map(lambda value: f"PRICE_Q{int(value) + 1}").astype("string")
    return buckets


def _product_features(product_master: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        product_master,
        ("canonical_product_id", "family_id", "family_desc", "list_price", "price_valid"),
        "Product master",
    )
    product = product_master.copy(deep=True)
    _require_unique(product, ("canonical_product_id",), "Product master")
    product["valid_price"] = _as_boolean(product["price_valid"], "Product master price_valid")
    product["price_bucket"] = _price_buckets(product)
    return product[["canonical_product_id", "family_id", "family_desc", "valid_price", "price_bucket"]]


def _branch_features(branch_master: pd.DataFrame) -> pd.DataFrame:
    _require_columns(branch_master, ("branch_code", "branch_name"), "Branch master")
    branch = branch_master.copy(deep=True)
    _require_unique(branch, ("branch_code",), "Branch master")
    return branch[["branch_code", "branch_name"]]


def enrich_monthly_features(
    monthly_features: pd.DataFrame,
    product_master: pd.DataFrame,
    calendar_daily: pd.DataFrame,
    branch_master: pd.DataFrame,
) -> ContextFeatureBuildResult:
    """Safely add context features without altering the monthly input.

    Product, calendar and branch features use only validated fields with clear
    semantics. Untrusted financial fields and ambiguous branch attributes are excluded.
    """

    _require_columns(monthly_features, MONTHLY_KEY, "Monthly feature input")
    input_rows = len(monthly_features)
    output = monthly_features.copy(deep=True)
    _require_unique(output, MONTHLY_KEY, "Monthly feature input")
    if output[list(MONTHLY_KEY)].isna().any().any():
        raise ContextFeatureValidationError("Monthly feature input has null grain values.")

    product = _product_features(product_master)
    calendar = build_monthly_calendar_features(calendar_daily)
    branch = _branch_features(branch_master)
    output = output.merge(product, how="left", on="canonical_product_id", validate="many_to_one")
    if len(output) != input_rows:
        raise ContextFeatureValidationError("Product feature join changed row count.")
    output = output.merge(calendar, how="left", on="year_month", validate="many_to_one")
    if len(output) != input_rows:
        raise ContextFeatureValidationError("Calendar feature join changed row count.")
    output = output.merge(branch, how="left", on="branch_code", validate="many_to_one")
    if len(output) != input_rows:
        raise ContextFeatureValidationError("Branch feature join changed row count.")
    _require_unique(output, MONTHLY_KEY, "Enriched monthly features")
    output = output.sort_values(list(MONTHLY_KEY), kind="stable").reset_index(drop=True)

    metrics = ContextFeatureMetrics(
        input_rows=input_rows,
        output_rows=len(output),
        unique_products=int(output["canonical_product_id"].nunique()),
        unique_branches=int(output["branch_code"].nunique()),
        months=int(output["year_month"].nunique()),
        invalid_price_rows=int((~output["valid_price"].fillna(False)).sum()),
        missing_family_rows=int(
            output["family_id"].isna().sum()
            + output["family_id"].astype("string").str.strip().eq("").sum()
        ),
        duplicate_product_keys=0,
        duplicate_branch_keys=0,
        duplicate_monthly_keys=0,
    )
    return ContextFeatureBuildResult(features=output, monthly_calendar=calendar, metrics=metrics)


def write_context_features(result: ContextFeatureBuildResult, output_path: Path, calendar_output_path: Path) -> None:
    """Persist deterministic enriched features and the supporting monthly calendar layer."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_output_path.parent.mkdir(parents=True, exist_ok=True)
    result.features.to_parquet(output_path, index=False)
    result.monthly_calendar.to_parquet(calendar_output_path, index=False)
