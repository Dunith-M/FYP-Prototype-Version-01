"""Deterministic monthly forecasting-target construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_NAME = "fulfilled_positive_units_proxy"
TARGET_KEY = ("branch_code", "canonical_product_id", "year_month")
DAILY_CONTRIBUTION_COLUMN = "fulfilled_positive_units_proxy_day"


class TargetBuildValidationError(ValueError):
    """Raised when the processed analytical layer cannot safely form a target."""


@dataclass(frozen=True)
class MonthlyTargetMetrics:
    target_rows: int
    minimum_target: float | None
    maximum_target: float | None
    zero_target_rows: int
    total_target_units: float
    positive_source_observations: int
    zero_source_observations: int
    negative_source_observations: int
    unresolved_excluded_rows: int
    unresolved_excluded_positive_units: float


@dataclass(frozen=True)
class MonthlyTargetBuildResult:
    monthly_target: pd.DataFrame
    metrics: MonthlyTargetMetrics


def _required_columns(frame: pd.DataFrame) -> None:
    missing = sorted(set(TARGET_KEY) - set(frame.columns))
    if missing:
        raise TargetBuildValidationError(
            f"Processed analytical input is missing required target columns: {missing}."
        )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise TargetBuildValidationError(
            f"{column} contains null or non-numeric values and cannot form a target."
        )
    return values.astype(float)


def _source_counts(frame: pd.DataFrame, units: pd.Series) -> tuple[int, int, int]:
    """Use retained transaction counts when present, otherwise inspect row values."""

    count_columns = (
        "positive_unit_row_count",
        "zero_unit_row_count",
        "negative_unit_row_count",
    )
    if all(column in frame.columns for column in count_columns):
        return tuple(int(pd.to_numeric(frame[column], errors="raise").sum()) for column in count_columns)
    return int((units > 0).sum()), int((units == 0).sum()), int((units < 0).sum())


def _unresolved_metrics(unresolved_audit: pd.DataFrame | None) -> tuple[int, float, int, int, int]:
    if unresolved_audit is None or unresolved_audit.empty:
        return 0, 0.0, 0, 0, 0
    unit_column = next(
        (column for column in ("raw_total_units", "TOTAL_UNITS", "sales_units") if column in unresolved_audit),
        None,
    )
    if unit_column is None:
        raise TargetBuildValidationError(
            "Unresolved audit input has no preserved sales-unit column."
        )
    units = _numeric(unresolved_audit, unit_column)
    return (
        len(unresolved_audit),
        float(units.where(units > 0, 0).sum()),
        int((units > 0).sum()),
        int((units == 0).sum()),
        int((units < 0).sum()),
    )


def build_monthly_fulfilled_positive_units_target(
    analytical_day: pd.DataFrame,
    unresolved_audit: pd.DataFrame | None = None,
) -> MonthlyTargetBuildResult:
    """Aggregate observed positive fulfilled units at branch-product-month grain.

    ``fulfilled_positive_units_proxy`` is not true, latent, unconstrained, or
    lost-sales-adjusted customer demand. Negative and zero source transactions
    remain in the processed/audit layers; they contribute zero only to this
    proxy target.
    """

    _required_columns(analytical_day)
    non_product_keys = ("branch_code", "year_month")
    if analytical_day[list(non_product_keys)].isna().any().any():
        raise TargetBuildValidationError("Target branch and month keys must be non-null.")

    raw_units_column = next(
        (column for column in ("raw_total_units", "sales_units", "TOTAL_UNITS") if column in analytical_day),
        None,
    )
    if raw_units_column is None:
        raise TargetBuildValidationError("Processed analytical input has no sales-unit column.")
    all_raw_units = _numeric(analytical_day, raw_units_column)
    eligible = analytical_day.copy(deep=True)
    unresolved_mask = eligible["canonical_product_id"].isna() | eligible[
        "canonical_product_id"
    ].astype("string").str.strip().eq("").fillna(True)
    inline_unresolved_positive_units = float(all_raw_units.loc[unresolved_mask].where(
        all_raw_units.loc[unresolved_mask] > 0, 0
    ).sum())
    eligible = eligible.loc[~unresolved_mask].copy()

    raw_units = _numeric(eligible, raw_units_column)
    if DAILY_CONTRIBUTION_COLUMN in eligible:
        contribution = _numeric(eligible, DAILY_CONTRIBUTION_COLUMN)
        if (contribution < 0).any():
            raise TargetBuildValidationError("Daily positive-unit contributions cannot be negative.")
    else:
        # Transaction-grain fallback. The pipeline retains the daily column so
        # mixed positive/negative transactions are not netted before targeting.
        contribution = raw_units.where(raw_units > 0, 0)

    eligible[DAILY_CONTRIBUTION_COLUMN] = contribution
    monthly_target = (
        eligible.groupby(list(TARGET_KEY), as_index=False, sort=True)[DAILY_CONTRIBUTION_COLUMN]
        .sum()
        .rename(columns={DAILY_CONTRIBUTION_COLUMN: TARGET_NAME})
        .sort_values(list(TARGET_KEY), kind="stable")
        .reset_index(drop=True)
    )
    if monthly_target.duplicated(list(TARGET_KEY)).any():
        raise TargetBuildValidationError("Monthly target has duplicate branch/product/month keys.")
    if (monthly_target[TARGET_NAME] < 0).any():
        raise TargetBuildValidationError("Monthly target cannot contain negative values.")

    positive, zero, negative = _source_counts(analytical_day, all_raw_units)
    unresolved_rows, unresolved_units, unresolved_positive, unresolved_zero, unresolved_negative = _unresolved_metrics(
        unresolved_audit
    )
    metrics = MonthlyTargetMetrics(
        target_rows=len(monthly_target),
        minimum_target=(float(monthly_target[TARGET_NAME].min()) if len(monthly_target) else None),
        maximum_target=(float(monthly_target[TARGET_NAME].max()) if len(monthly_target) else None),
        zero_target_rows=int((monthly_target[TARGET_NAME] == 0).sum()),
        total_target_units=float(monthly_target[TARGET_NAME].sum()),
        positive_source_observations=positive + unresolved_positive,
        zero_source_observations=zero + unresolved_zero,
        negative_source_observations=negative + unresolved_negative,
        unresolved_excluded_rows=int(unresolved_mask.sum()) + unresolved_rows,
        unresolved_excluded_positive_units=inline_unresolved_positive_units + unresolved_units,
    )
    return MonthlyTargetBuildResult(monthly_target=monthly_target, metrics=metrics)


def write_monthly_target(result: MonthlyTargetBuildResult, output_path) -> None:
    """Persist the deterministic target dataset without altering its inputs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.monthly_target.to_parquet(output_path, index=False)
