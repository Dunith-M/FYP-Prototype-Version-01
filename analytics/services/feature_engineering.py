"""Leakage-safe demand-history features for observed monthly sales evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.services.targets import TARGET_NAME


FEATURE_KEY = ("branch_code", "canonical_product_id", "year_month")
FEATURE_COLUMNS = (
    "lag_1",
    "lag_2",
    "lag_3",
    "trailing_mean_3",
    "trailing_sum_3",
    "months_with_positive_sales",
    "intermittency",
)


class DemandHistoryFeatureValidationError(ValueError):
    """Raised when observed monthly input cannot safely form history features."""


@dataclass(frozen=True)
class DemandHistoryFeatureMetrics:
    feature_rows: int
    unique_branches: int
    unique_products: int
    minimum_year_month: str | None
    maximum_year_month: str | None
    lag_1_available_rows: int
    lag_2_available_rows: int
    lag_3_available_rows: int
    full_three_month_history_rows: int
    minimum_intermittency: float | None
    maximum_intermittency: float | None
    duplicate_keys: int
    calendar_gap_edges: int
    unresolved_rows_excluded: int


@dataclass(frozen=True)
class DemandHistoryFeatureBuildResult:
    features: pd.DataFrame
    metrics: DemandHistoryFeatureMetrics


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(set((*FEATURE_KEY, TARGET_NAME)) - set(frame.columns))
    if missing:
        raise DemandHistoryFeatureValidationError(
            f"BRANCH_PRODUCT_MONTH is missing required columns: {missing}."
        )


def _periods(year_month: pd.Series) -> pd.Series:
    values = year_month.astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        raise DemandHistoryFeatureValidationError("year_month contains null or blank values.")
    try:
        return pd.Series(pd.PeriodIndex(values, freq="M"), index=year_month.index)
    except (TypeError, ValueError) as error:
        raise DemandHistoryFeatureValidationError(
            "year_month must use YYYY-MM monthly values."
        ) from error


def _numeric_target(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame[TARGET_NAME], errors="coerce")
    if values.isna().any() or (values < 0).any():
        raise DemandHistoryFeatureValidationError(
            f"{TARGET_NAME} must be non-null, numeric, and non-negative."
        )
    return values.astype(float)


def _add_group_features(group: pd.DataFrame) -> pd.DataFrame:
    """Shift first, then roll: every feature is strictly prior to this row."""

    result = group.copy()
    target = result[TARGET_NAME]
    prior_target = target.shift(1)
    result["lag_1"] = prior_target
    result["lag_2"] = target.shift(2)
    result["lag_3"] = target.shift(3)
    result["trailing_mean_3"] = prior_target.rolling(window=3, min_periods=3).mean()
    result["trailing_sum_3"] = prior_target.rolling(window=3, min_periods=3).sum()

    prior_positive = target.gt(0).shift(1, fill_value=False)
    result["months_with_positive_sales"] = prior_positive.cumsum().astype("int64")
    prior_months = pd.Series(np.arange(len(result)), index=result.index, dtype="float64")
    result["intermittency"] = (1 - result["months_with_positive_sales"] / prior_months).where(
        prior_months > 0
    )
    return result


def validate_demand_history_feature_leakage(frame: pd.DataFrame) -> None:
    """Verify rolling features use target.shift(1), never month M or its future."""

    missing = sorted(set((*FEATURE_KEY, TARGET_NAME, *FEATURE_COLUMNS)) - set(frame.columns))
    if missing:
        raise DemandHistoryFeatureValidationError(
            f"Feature leakage audit requires columns: {missing}."
        )

    for _, group in frame.groupby(["branch_code", "canonical_product_id"], sort=False):
        targets = group[TARGET_NAME].tolist()
        for position, (_, row) in enumerate(group.iterrows()):
            prior = targets[:position]
            expected_lags = [
                prior[-offset] if len(prior) >= offset else np.nan
                for offset in (1, 2, 3)
            ]
            for column, expected in zip(("lag_1", "lag_2", "lag_3"), expected_lags):
                actual = row[column]
                if not ((pd.isna(actual) and pd.isna(expected)) or np.isclose(actual, expected)):
                    raise DemandHistoryFeatureValidationError(
                        f"Leakage audit failed for {column} at {row['year_month']}."
                    )
            expected_mean = np.mean(prior[-3:]) if len(prior) >= 3 else np.nan
            expected_sum = np.sum(prior[-3:]) if len(prior) >= 3 else np.nan
            for column, expected in (("trailing_mean_3", expected_mean), ("trailing_sum_3", expected_sum)):
                actual = row[column]
                if not ((pd.isna(actual) and pd.isna(expected)) or np.isclose(actual, expected)):
                    raise DemandHistoryFeatureValidationError(
                        f"Leakage audit failed for {column} at {row['year_month']}."
                    )
            expected_positive = sum(value > 0 for value in prior)
            if int(row["months_with_positive_sales"]) != expected_positive:
                raise DemandHistoryFeatureValidationError(
                    f"Leakage audit failed for months_with_positive_sales at {row['year_month']}."
                )
            expected_intermittency = 1 - expected_positive / len(prior) if prior else np.nan
            actual_intermittency = row["intermittency"]
            if not (
                (pd.isna(actual_intermittency) and pd.isna(expected_intermittency))
                or np.isclose(actual_intermittency, expected_intermittency)
            ):
                raise DemandHistoryFeatureValidationError(
                    f"Leakage audit failed for intermittency at {row['year_month']}."
                )


def build_demand_history_features(
    branch_product_month: pd.DataFrame,
    unresolved_audit: pd.DataFrame | None = None,
) -> DemandHistoryFeatureBuildResult:
    """Create demand-history features using only months strictly before month M.

    The input stays unchanged. Missing monthly records are not densified or
    interpreted as zero demand; lags reference only prior observed records.
    """

    _require_columns(branch_product_month)
    work = branch_product_month.copy(deep=True)
    if work[["branch_code", "year_month"]].isna().any().any():
        raise DemandHistoryFeatureValidationError("Branch and month keys must be non-null.")
    work["_month_period"] = _periods(work["year_month"])
    work[TARGET_NAME] = _numeric_target(work)
    unresolved = work["canonical_product_id"].isna() | work[
        "canonical_product_id"
    ].astype("string").str.strip().eq("").fillna(True)
    eligible = work.loc[~unresolved].copy()
    duplicates = int(eligible.duplicated(list(FEATURE_KEY)).sum())
    if duplicates:
        raise DemandHistoryFeatureValidationError(
            f"BRANCH_PRODUCT_MONTH has duplicate branch/product/month keys={duplicates}."
        )

    eligible = eligible.sort_values(
        ["branch_code", "canonical_product_id", "_month_period"], kind="stable"
    ).reset_index(drop=True)
    feature_groups = [
        _add_group_features(group)
        for _, group in eligible.groupby(["branch_code", "canonical_product_id"], sort=False)
    ]
    features = (
        pd.concat(feature_groups, ignore_index=True)
        if feature_groups
        else eligible.copy()
    )
    validate_demand_history_feature_leakage(features)

    gap_edges = 0
    for _, group in features.groupby(["branch_code", "canonical_product_id"], sort=False):
        gap_edges += int((group["_month_period"].astype("int64").diff().dropna() > 1).sum())
    features = features.drop(columns="_month_period")

    unresolved_audit_rows = 0 if unresolved_audit is None else len(unresolved_audit)
    metrics = DemandHistoryFeatureMetrics(
        feature_rows=len(features),
        unique_branches=int(features["branch_code"].nunique()),
        unique_products=int(features["canonical_product_id"].nunique()),
        minimum_year_month=(str(features["year_month"].min()) if len(features) else None),
        maximum_year_month=(str(features["year_month"].max()) if len(features) else None),
        lag_1_available_rows=int(features["lag_1"].notna().sum()),
        lag_2_available_rows=int(features["lag_2"].notna().sum()),
        lag_3_available_rows=int(features["lag_3"].notna().sum()),
        full_three_month_history_rows=int(features["trailing_mean_3"].notna().sum()),
        minimum_intermittency=(float(features["intermittency"].min()) if features["intermittency"].notna().any() else None),
        maximum_intermittency=(float(features["intermittency"].max()) if features["intermittency"].notna().any() else None),
        duplicate_keys=int(features.duplicated(list(FEATURE_KEY)).sum()),
        calendar_gap_edges=gap_edges,
        unresolved_rows_excluded=int(unresolved.sum()) + unresolved_audit_rows,
    )
    return DemandHistoryFeatureBuildResult(features, metrics)


def write_demand_history_features(
    result: DemandHistoryFeatureBuildResult, output_path: Path
) -> None:
    """Persist the deterministic, observed-month feature dataset."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.features.to_parquet(output_path, index=False)
