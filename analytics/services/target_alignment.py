"""Leakage-safe supervised next-month target alignment for monthly features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analytics.services.targets import TARGET_NAME


FEATURE_KEY = ("branch_code", "canonical_product_id", "year_month")
SUPERVISED_TARGET = "target_next_month"


class NextMonthTargetValidationError(ValueError):
    """Raised when monthly features cannot safely form next-month targets."""


@dataclass(frozen=True)
class NextMonthTargetMetrics:
    total_feature_rows: int
    valid_target_rows: int
    rows_without_target: int
    non_consecutive_month_pairs: int
    unique_branches: int
    unique_products: int
    minimum_feature_month: str | None
    maximum_feature_month: str | None
    minimum_target_month: str | None
    maximum_target_month: str | None
    duplicate_monthly_keys: int
    negative_target_values: int


@dataclass(frozen=True)
class NextMonthTargetBuildResult:
    features: pd.DataFrame
    metrics: NextMonthTargetMetrics


def _periods(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    if cleaned.isna().any() or cleaned.eq("").any():
        raise NextMonthTargetValidationError("year_month contains null or blank values.")
    try:
        return pd.Series(pd.PeriodIndex(cleaned, freq="M"), index=values.index)
    except (TypeError, ValueError) as error:
        raise NextMonthTargetValidationError("year_month must use YYYY-MM monthly values.") from error


def _require_input(frame: pd.DataFrame) -> None:
    missing = sorted(set((*FEATURE_KEY, TARGET_NAME)) - set(frame.columns))
    if missing:
        raise NextMonthTargetValidationError(f"Monthly feature input is missing columns: {missing}.")
    if frame[list(FEATURE_KEY)].isna().any().any():
        raise NextMonthTargetValidationError("Monthly feature keys must be non-null for model rows.")
    if frame["canonical_product_id"].astype("string").str.strip().eq("").any():
        raise NextMonthTargetValidationError("canonical_product_id must be non-blank for model rows.")
    duplicates = int(frame.duplicated(list(FEATURE_KEY)).sum())
    if duplicates:
        raise NextMonthTargetValidationError(
            f"Monthly feature input has duplicate branch/product/month keys={duplicates}."
        )


def validate_next_month_target_alignment(frame: pd.DataFrame) -> None:
    """Verify every available target is exactly the same-series calendar month M+1."""

    required = (*FEATURE_KEY, TARGET_NAME, SUPERVISED_TARGET, "target_year_month", "target_available_flag")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise NextMonthTargetValidationError(f"Target alignment audit requires columns: {missing}.")
    periods = _periods(frame["year_month"])
    for (_, _), group in frame.assign(_period=periods).groupby(
        ["branch_code", "canonical_product_id"], sort=False
    ):
        group = group.sort_values("_period", kind="stable").reset_index(drop=True)
        for position, row in group.iterrows():
            has_target = bool(row["target_available_flag"])
            if position == len(group) - 1:
                if has_target or pd.notna(row[SUPERVISED_TARGET]):
                    raise NextMonthTargetValidationError("Final observed month must not have a next-month target.")
                continue
            next_row = group.iloc[position + 1]
            consecutive = next_row["_period"] == row["_period"] + 1
            if consecutive:
                if not has_target or row["target_year_month"] != str(next_row["year_month"]):
                    raise NextMonthTargetValidationError("Consecutive month target alignment audit failed.")
                if row[SUPERVISED_TARGET] != next_row[TARGET_NAME]:
                    raise NextMonthTargetValidationError("Next-month target value alignment audit failed.")
            elif has_target or pd.notna(row[SUPERVISED_TARGET]) or pd.notna(row["target_year_month"]):
                raise NextMonthTargetValidationError("A non-consecutive monthly pair received a false next-month target.")


def align_next_month_target(monthly_features: pd.DataFrame) -> NextMonthTargetBuildResult:
    """Add the forward supervised target while preserving existing monthly feature values.

    For feature month M, ``target_next_month`` is the observed
    ``fulfilled_positive_units_proxy`` from calendar month M+1 for the same
    branch-product pair. Historical features point backward; the supervised
    target points forward. Non-consecutive observed months receive no target.
    """

    _require_input(monthly_features)
    output = monthly_features.copy(deep=True)
    output["_month_period"] = _periods(output["year_month"])
    source_target = pd.to_numeric(output[TARGET_NAME], errors="coerce")
    if source_target.isna().any() or (source_target < 0).any():
        raise NextMonthTargetValidationError(f"{TARGET_NAME} must be non-null, numeric, and non-negative.")
    output[TARGET_NAME] = source_target.astype(float)
    output = output.sort_values(
        ["branch_code", "canonical_product_id", "_month_period"], kind="stable"
    ).reset_index(drop=True)
    grouped = output.groupby(["branch_code", "canonical_product_id"], sort=False)
    next_target = grouped[TARGET_NAME].shift(-1)
    next_period = grouped["_month_period"].shift(-1)
    consecutive = next_period.eq(output["_month_period"] + 1)
    output[SUPERVISED_TARGET] = next_target.where(consecutive)
    output["target_year_month"] = next_period.astype("string").where(consecutive).astype("string")
    output["target_available_flag"] = consecutive.fillna(False).astype(bool)
    non_consecutive = int((next_period.notna() & ~consecutive.fillna(False)).sum())
    output = output.drop(columns="_month_period")
    validate_next_month_target_alignment(output)

    target_months = output.loc[output["target_available_flag"], "target_year_month"]
    metrics = NextMonthTargetMetrics(
        total_feature_rows=len(output),
        valid_target_rows=int(output["target_available_flag"].sum()),
        rows_without_target=int((~output["target_available_flag"]).sum()),
        non_consecutive_month_pairs=non_consecutive,
        unique_branches=int(output["branch_code"].nunique()),
        unique_products=int(output["canonical_product_id"].nunique()),
        minimum_feature_month=(str(output["year_month"].min()) if len(output) else None),
        maximum_feature_month=(str(output["year_month"].max()) if len(output) else None),
        minimum_target_month=(str(target_months.min()) if len(target_months) else None),
        maximum_target_month=(str(target_months.max()) if len(target_months) else None),
        duplicate_monthly_keys=int(output.duplicated(list(FEATURE_KEY)).sum()),
        negative_target_values=int((output[SUPERVISED_TARGET].dropna() < 0).sum()),
    )
    return NextMonthTargetBuildResult(features=output, metrics=metrics)


def write_next_month_targets(result: NextMonthTargetBuildResult, output_path: Path) -> None:
    """Persist the supervised monthly table without changing the source feature file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.features.to_parquet(output_path, index=False)
