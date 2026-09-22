"""Leakage-safe inventory evidence features for prediction-month rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_KEY = ("branch_code", "canonical_product_id", "year_month")
INVENTORY_FEATURE_COLUMNS = (
    "last_observed_qty",
    "mean_observed_qty_30d",
    "inventory_observation_count",
    "availability_unknown_rate",
)
CURRENT_MONTH_AUDIT_RENAMES = {
    "inventory_observation_count": "current_month_inventory_observation_count_audit",
    "inventory_unknown_count": "current_month_inventory_unknown_count_audit",
}


class InventoryEvidenceFeatureValidationError(ValueError):
    """Raised when processed inventory evidence cannot safely form features."""


@dataclass(frozen=True)
class InventoryEvidenceFeatureMetrics:
    feature_rows: int
    last_observed_available_rows: int
    last_observed_missing_rows: int
    minimum_inventory_observation_count: int
    mean_inventory_observation_count: float
    maximum_inventory_observation_count: int
    minimum_availability_unknown_rate: float | None
    maximum_availability_unknown_rate: float | None
    fully_unknown_rows: int
    zero_observed_inventory_rows: int
    future_data_leakage_affected_rows: int
    unresolved_rows_excluded: int


@dataclass(frozen=True)
class InventoryEvidenceFeatureBuildResult:
    features: pd.DataFrame
    metrics: InventoryEvidenceFeatureMetrics


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise InventoryEvidenceFeatureValidationError(f"{label} is missing columns: {missing}.")


def _month_starts(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    if cleaned.isna().any() or cleaned.eq("").any():
        raise InventoryEvidenceFeatureValidationError("year_month contains null or blank values.")
    try:
        return pd.Series(pd.PeriodIndex(cleaned, freq="M").start_time.date, index=values.index)
    except (TypeError, ValueError) as error:
        raise InventoryEvidenceFeatureValidationError("year_month must use YYYY-MM values.") from error


def _inventory_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise InventoryEvidenceFeatureValidationError("business_date contains invalid inventory dates.")
    return parsed.dt.date


def _observed_mask(frame: pd.DataFrame) -> pd.Series:
    unknown = frame["availability_state"].astype("string").str.strip().str.upper().eq("UNKNOWN")
    qty = pd.to_numeric(frame["inventory_qty"], errors="coerce")
    return ~unknown & qty.notna()


def build_inventory_evidence_features(
    prediction_month_features: pd.DataFrame,
    branch_product_day: pd.DataFrame,
    unresolved_audit: pd.DataFrame | None = None,
) -> InventoryEvidenceFeatureBuildResult:
    """Add strictly-prior inventory evidence features without changing inputs.

    Inventory evidence for prediction month M uses dates before M only. UNKNOWN
    inventory stays unknown and is never converted to zero. Unknown-rate
    denominator is represented BRANCH_PRODUCT_DAY records in the prior 30-day
    window; missing calendar rows are not invented.
    """

    _require_columns(prediction_month_features, PREDICTION_KEY, "Prediction-month input")
    _require_columns(
        branch_product_day,
        ("branch_code", "canonical_product_id", "business_date", "inventory_qty", "availability_state"),
        "BRANCH_PRODUCT_DAY",
    )
    output = prediction_month_features.copy(deep=True)
    for old_name, audit_name in CURRENT_MONTH_AUDIT_RENAMES.items():
        if old_name in output.columns:
            output = output.rename(columns={old_name: audit_name})
    if output.duplicated(list(PREDICTION_KEY)).any():
        raise InventoryEvidenceFeatureValidationError("Prediction-month input has duplicate monthly keys.")
    if output[["branch_code", "year_month"]].isna().any().any():
        raise InventoryEvidenceFeatureValidationError("Prediction branch and month keys must be non-null.")

    unresolved = output["canonical_product_id"].isna() | output[
        "canonical_product_id"
    ].astype("string").str.strip().eq("").fillna(True)
    eligible = output.loc[~unresolved].copy()
    eligible["_month_start"] = _month_starts(eligible["year_month"])

    evidence = branch_product_day.copy(deep=True)
    if evidence.duplicated(["branch_code", "canonical_product_id", "business_date"]).any():
        raise InventoryEvidenceFeatureValidationError("BRANCH_PRODUCT_DAY has duplicate inventory evidence keys.")
    evidence["_inventory_date"] = _inventory_dates(evidence["business_date"])
    evidence["_observed"] = _observed_mask(evidence)
    unknown_with_qty = evidence.loc[
        evidence["availability_state"].astype("string").str.strip().str.upper().eq("UNKNOWN"),
        "inventory_qty",
    ].notna()
    if unknown_with_qty.any():
        raise InventoryEvidenceFeatureValidationError("UNKNOWN inventory evidence cannot have an observed quantity.")
    evidence_groups = {
        key: group.sort_values("_inventory_date", kind="stable")
        for key, group in evidence.groupby(["branch_code", "canonical_product_id"], sort=False)
    }

    rows: list[dict[str, object]] = []
    for row_index, row in eligible.iterrows():
        month_start = row["_month_start"]
        group = evidence_groups.get((row["branch_code"], row["canonical_product_id"]))
        if group is None:
            prior = pd.DataFrame(columns=evidence.columns)
            window = prior
        else:
            prior = group.loc[group["_inventory_date"] < month_start]
            window = prior.loc[prior["_inventory_date"] >= month_start - timedelta(days=30)]
        observed_prior = prior.loc[prior["_observed"]] if len(prior) else prior
        observed_window = window.loc[window["_observed"]] if len(window) else window
        rows.append(
            {
                "_row_index": row_index,
                "last_observed_qty": (
                    float(pd.to_numeric(observed_prior["inventory_qty"], errors="raise").iloc[-1])
                    if len(observed_prior)
                    else np.nan
                ),
                "mean_observed_qty_30d": (
                    float(pd.to_numeric(observed_window["inventory_qty"], errors="raise").mean())
                    if len(observed_window)
                    else np.nan
                ),
                "inventory_observation_count": int(len(observed_window)),
                "availability_unknown_rate": (
                    float(
                        window["availability_state"].astype("string").str.strip().str.upper().eq("UNKNOWN").mean()
                    )
                    if len(window)
                    else np.nan
                ),
            }
        )
    features = eligible.merge(
        pd.DataFrame(rows), left_index=True, right_on="_row_index", how="left", validate="one_to_one"
    ).drop(columns=["_row_index", "_month_start"])
    features = features.sort_values(list(PREDICTION_KEY), kind="stable").reset_index(drop=True)
    if len(features) != len(eligible):
        raise InventoryEvidenceFeatureValidationError("Inventory feature join changed prediction-row count.")
    rates = features["availability_unknown_rate"].dropna()
    if ((rates < 0) | (rates > 1)).any():
        raise InventoryEvidenceFeatureValidationError("availability_unknown_rate must remain between zero and one.")

    unresolved_audit_rows = 0 if unresolved_audit is None else len(unresolved_audit)
    counts = features["inventory_observation_count"]
    metrics = InventoryEvidenceFeatureMetrics(
        feature_rows=len(features),
        last_observed_available_rows=int(features["last_observed_qty"].notna().sum()),
        last_observed_missing_rows=int(features["last_observed_qty"].isna().sum()),
        minimum_inventory_observation_count=int(counts.min()) if len(counts) else 0,
        mean_inventory_observation_count=float(counts.mean()) if len(counts) else 0.0,
        maximum_inventory_observation_count=int(counts.max()) if len(counts) else 0,
        minimum_availability_unknown_rate=(float(rates.min()) if len(rates) else None),
        maximum_availability_unknown_rate=(float(rates.max()) if len(rates) else None),
        fully_unknown_rows=int((features["availability_unknown_rate"] == 1).sum()),
        zero_observed_inventory_rows=int((counts == 0).sum()),
        future_data_leakage_affected_rows=0,
        unresolved_rows_excluded=int(unresolved.sum()) + unresolved_audit_rows,
    )
    return InventoryEvidenceFeatureBuildResult(features, metrics)


def write_inventory_evidence_features(
    result: InventoryEvidenceFeatureBuildResult, output_path: Path
) -> None:
    """Persist deterministic inventory-evidence feature rows."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.features.to_parquet(output_path, index=False)
