"""Aggregate validated branch-product-day evidence to observed monthly evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analytics.date_utils import normalize_calendar_date, year_month_from_calendar_date
from analytics.services.targets import (
    DAILY_CONTRIBUTION_COLUMN,
    TARGET_KEY,
    TARGET_NAME,
    MonthlyTargetMetrics,
    build_monthly_fulfilled_positive_units_target,
)


MONTHLY_KEY = TARGET_KEY


class MonthlyAggregationValidationError(ValueError):
    """Raised when daily analytical evidence cannot safely be aggregated."""


@dataclass(frozen=True)
class BranchProductMonthMetrics:
    monthly_rows: int
    unique_branches: int
    unique_products: int
    minimum_year_month: str | None
    maximum_year_month: str | None
    total_positive_fulfilled_units: float
    total_sales_observation_count: int
    total_negative_event_count: int
    total_zero_event_count: int
    total_inventory_observation_count: int
    unresolved_rows_excluded: int
    duplicate_monthly_keys: int


@dataclass(frozen=True)
class BranchProductMonthBuildResult:
    branch_product_month: pd.DataFrame
    metrics: BranchProductMonthMetrics


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise MonthlyAggregationValidationError(
            f"BRANCH_PRODUCT_DAY is missing required columns: {missing}."
        )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise MonthlyAggregationValidationError(
            f"{column} contains null or non-numeric values."
        )
    return values


def _year_month(frame: pd.DataFrame) -> pd.Series:
    if "year_month" in frame.columns:
        values = frame["year_month"].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise MonthlyAggregationValidationError("year_month contains null or blank values.")
        return values

    normalized = frame["business_date"].map(normalize_calendar_date)
    invalid = normalized.map(lambda result: result.normalized_date is None)
    if invalid.any():
        raise MonthlyAggregationValidationError(
            f"business_date has invalid values={int(invalid.sum())}."
        )
    return normalized.map(lambda result: year_month_from_calendar_date(result.normalized_date))


def _event_count(frame: pd.DataFrame, count_column: str, flag_column: str) -> pd.Series:
    if count_column in frame.columns:
        return _numeric(frame, count_column).astype("int64")
    if flag_column in frame.columns:
        return frame[flag_column].fillna(False).astype(bool).astype("int64")
    return pd.Series(0, index=frame.index, dtype="int64")


def build_branch_product_month(
    branch_product_day: pd.DataFrame,
    unresolved_audit: pd.DataFrame | None = None,
) -> BranchProductMonthBuildResult:
    """Build one observed branch-product-month record from daily evidence.

    ``fulfilled_positive_units_proxy`` is observed positive fulfilled sales, not
    true or unconstrained demand. Unknown inventory remains a count of missing
    observation evidence, never zero inventory.
    """

    _require_columns(
        branch_product_day,
        ("branch_code", "canonical_product_id", "business_date", DAILY_CONTRIBUTION_COLUMN, "source_row_count", "availability_state"),
    )
    daily = branch_product_day.copy(deep=True)
    daily["year_month"] = _year_month(daily)

    target_result = build_monthly_fulfilled_positive_units_target(
        daily, unresolved_audit
    )
    target_metrics: MonthlyTargetMetrics = target_result.metrics
    unresolved = daily["canonical_product_id"].isna() | daily[
        "canonical_product_id"
    ].astype("string").str.strip().eq("").fillna(True)
    eligible = daily.loc[~unresolved].copy()
    if eligible[list(MONTHLY_KEY)].isna().any().any():
        raise MonthlyAggregationValidationError("Monthly grouping keys must be non-null for eligible rows.")

    eligible["sales_observation_count"] = _numeric(eligible, "source_row_count").astype("int64")
    # These are source transaction/event counts where available, not inferred return semantics.
    eligible["negative_event_count"] = _event_count(
        eligible, "negative_unit_row_count", "negative_units_flag"
    )
    eligible["zero_event_count"] = _event_count(
        eligible, "zero_unit_row_count", "zero_units_flag"
    )
    inventory_unknown = eligible["availability_state"].astype("string").str.strip().str.upper().eq("UNKNOWN")
    eligible["inventory_observation_count"] = (~inventory_unknown).astype("int64")
    eligible["inventory_unknown_count"] = inventory_unknown.astype("int64")
    eligible["positive_sales_day_count"] = (
        _numeric(eligible, DAILY_CONTRIBUTION_COLUMN) > 0
    ).astype("int64")

    diagnostics = (
        eligible.groupby(list(MONTHLY_KEY), as_index=False, sort=True)
        .agg(
            sales_observation_count=("sales_observation_count", "sum"),
            negative_event_count=("negative_event_count", "sum"),
            zero_event_count=("zero_event_count", "sum"),
            inventory_observation_count=("inventory_observation_count", "sum"),
            inventory_unknown_count=("inventory_unknown_count", "sum"),
            positive_sales_day_count=("positive_sales_day_count", "sum"),
        )
    )
    monthly = target_result.monthly_target.merge(
        diagnostics, how="left", on=list(MONTHLY_KEY), validate="one_to_one"
    ).sort_values(list(MONTHLY_KEY), kind="stable").reset_index(drop=True)
    if len(monthly) != len(target_result.monthly_target):
        raise MonthlyAggregationValidationError("Monthly diagnostic join changed target row count.")
    if monthly.isna().any().any():
        raise MonthlyAggregationValidationError("Monthly output contains unexpected null values.")
    duplicates = int(monthly.duplicated(list(MONTHLY_KEY)).sum())
    if duplicates:
        raise MonthlyAggregationValidationError(f"Monthly output duplicate keys={duplicates}.")
    if (monthly[TARGET_NAME] < 0).any():
        raise MonthlyAggregationValidationError("Monthly positive-fulfilled target cannot be negative.")

    metrics = BranchProductMonthMetrics(
        monthly_rows=len(monthly),
        unique_branches=int(monthly["branch_code"].nunique()),
        unique_products=int(monthly["canonical_product_id"].nunique()),
        minimum_year_month=(str(monthly["year_month"].min()) if len(monthly) else None),
        maximum_year_month=(str(monthly["year_month"].max()) if len(monthly) else None),
        total_positive_fulfilled_units=float(monthly[TARGET_NAME].sum()),
        total_sales_observation_count=int(monthly["sales_observation_count"].sum()),
        total_negative_event_count=int(monthly["negative_event_count"].sum()),
        total_zero_event_count=int(monthly["zero_event_count"].sum()),
        total_inventory_observation_count=int(monthly["inventory_observation_count"].sum()),
        unresolved_rows_excluded=target_metrics.unresolved_excluded_rows,
        duplicate_monthly_keys=duplicates,
    )
    return BranchProductMonthBuildResult(monthly, metrics)


def write_branch_product_month(
    result: BranchProductMonthBuildResult, output_path: Path
) -> None:
    """Persist the deterministic observed-monthly analytical dataset."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.branch_product_month.to_parquet(output_path, index=False)
