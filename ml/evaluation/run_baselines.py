"""Offline, deterministic evaluation runner for the Day-05 baseline models."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.folds import (
    FEATURE_MONTH_PATH,
    FINAL_TEST_MONTH,
    MODEL_SELECTION_MONTHS,
    PROJECT_ROOT,
    TARGET_AVAILABLE_COLUMN,
    TARGET_MONTH_COLUMN,
    VALIDATION_FOLDS,
    load_feature_month,
    split_fold,
)
from ml.evaluation.report import build_baseline_evaluation_table, save_baseline_evaluation_table
from ml.models.baselines import (
    last_observation_forecast,
    trailing_mean_3_forecast_with_fallbacks,
    zero_forecast,
)


TARGET_VALUE_COLUMN = "target_next_month"
IDENTIFIER_COLUMNS = ("branch_code", "canonical_product_id")
FORECASTING_ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts" / "forecasting"
PREDICTIONS_PATH = FORECASTING_ARTIFACT_DIR / "baseline_predictions.parquet"
METRICS_PATH = FORECASTING_ARTIFACT_DIR / "baseline_metrics.csv"
MODEL_NAMES = ("zero_forecast", "last_observation", "trailing_mean_3")
RUNNER_REQUIRED_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    TARGET_MONTH_COLUMN,
    TARGET_VALUE_COLUMN,
    "lag_1",
    "trailing_mean_3",
    TARGET_AVAILABLE_COLUMN,
)


class BaselineEvaluationError(ValueError):
    """Raised when a baseline evaluation cannot be performed safely."""


@dataclass(frozen=True)
class BaselineEvaluationResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def load_feature_data(path: Path = FEATURE_MONTH_PATH) -> pd.DataFrame:
    """Load and validate the read-only Day-04 feature artifact for evaluation."""
    feature_df = load_feature_month(path)
    _validate_feature_data(feature_df)
    return feature_df


def _validate_feature_data(feature_df: pd.DataFrame) -> None:
    missing = sorted(set(RUNNER_REQUIRED_COLUMNS) - set(feature_df.columns))
    if missing:
        raise BaselineEvaluationError(f"Feature data is missing required columns: {missing}.")
    actual = pd.to_numeric(
        feature_df.loc[feature_df[TARGET_AVAILABLE_COLUMN], TARGET_VALUE_COLUMN], errors="coerce"
    )
    if actual.isna().any() or not np.isfinite(actual).all() or (actual < 0).any():
        raise BaselineEvaluationError(f"{TARGET_VALUE_COLUMN} must be finite and non-negative.")


def _prediction_rows(validation_df: pd.DataFrame, fold_id: str, validation_month: str) -> pd.DataFrame:
    """Create three baseline prediction records per validation observation."""
    trace_columns = [*IDENTIFIER_COLUMNS, TARGET_MONTH_COLUMN]
    if "family_id" in validation_df.columns:
        trace_columns.append("family_id")
    base = validation_df.loc[:, trace_columns].copy().reset_index(drop=True)
    base = base.rename(columns={TARGET_MONTH_COLUMN: "target_month"})
    base["fold_id"] = fold_id
    base["validation_month"] = validation_month
    base["actual_units"] = pd.to_numeric(
        validation_df[TARGET_VALUE_COLUMN], errors="raise"
    ).reset_index(drop=True)

    zero_prediction = zero_forecast(validation_df).reset_index(drop=True)
    last_prediction = last_observation_forecast(validation_df).reset_index(drop=True)
    trailing_prediction = trailing_mean_3_forecast_with_fallbacks(validation_df).reset_index(drop=True)
    if not (len(zero_prediction) == len(last_prediction) == len(trailing_prediction) == len(base)):
        raise BaselineEvaluationError(f"Baseline output length differs from validation rows for {fold_id}.")

    zero_rows = base.copy()
    zero_rows["model_name"] = "zero_forecast"
    zero_rows["predicted_units"] = zero_prediction
    zero_rows["fallback_used"] = False
    zero_rows["fallback_reason"] = None

    last_rows = base.copy()
    last_rows["model_name"] = "last_observation"
    last_rows["predicted_units"] = last_prediction
    last_rows["fallback_used"] = False
    last_rows["fallback_reason"] = None

    trailing_rows = base.copy()
    trailing_rows["model_name"] = trailing_prediction["model_name"]
    trailing_rows["predicted_units"] = trailing_prediction["prediction"]
    trailing_rows["fallback_used"] = trailing_prediction["fallback_used"]
    trailing_rows["fallback_reason"] = trailing_prediction["fallback_reason"]

    output = pd.concat([zero_rows, last_rows, trailing_rows], ignore_index=True)
    output["prediction_available"] = output["predicted_units"].notna()
    output["absolute_error"] = (output["actual_units"] - output["predicted_units"]).abs()
    return output


def _assert_model_selection_only(predictions: pd.DataFrame) -> None:
    target_months = set(predictions["target_month"].astype(str))
    unexpected = target_months - set(MODEL_SELECTION_MONTHS)
    if unexpected or FINAL_TEST_MONTH in target_months:
        raise BaselineEvaluationError(
            f"Model-selection predictions contain forbidden target months: {sorted(unexpected)}."
        )


def evaluate_all_folds(feature_df: pd.DataFrame) -> BaselineEvaluationResult:
    """Evaluate all frozen model-selection folds without touching March 2026."""
    _validate_feature_data(feature_df)

    fold_outputs = []
    for fold in VALIDATION_FOLDS:
        _, validation_df = split_fold(feature_df, fold)
        fold_outputs.append(_prediction_rows(validation_df, fold.fold_id, fold.validation_month))
    predictions = pd.concat(fold_outputs, ignore_index=True)
    _assert_model_selection_only(predictions)
    predictions = predictions.sort_values(
        ["fold_id", *IDENTIFIER_COLUMNS, "target_month", "model_name"], kind="stable"
    ).reset_index(drop=True)
    metrics = build_baseline_evaluation_table(predictions)
    return BaselineEvaluationResult(predictions=predictions, metrics=metrics)


def save_evaluation_artifacts(
    result: BaselineEvaluationResult,
    predictions_path: Path = PREDICTIONS_PATH,
    metrics_path: Path = METRICS_PATH,
) -> pd.DataFrame:
    """Persist evidence first, then aggregate that persisted evidence into metrics."""
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(predictions_path, index=False)
    persisted_predictions = pd.read_parquet(predictions_path)
    persisted_metrics = build_baseline_evaluation_table(persisted_predictions)
    save_baseline_evaluation_table(persisted_metrics, metrics_path)
    return persisted_metrics


def run_baselines(
    feature_path: Path = FEATURE_MONTH_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    metrics_path: Path = METRICS_PATH,
) -> BaselineEvaluationResult:
    """Load, evaluate, and persist the frozen model-selection baseline results."""
    result = evaluate_all_folds(load_feature_data(feature_path))
    persisted_metrics = save_evaluation_artifacts(result, predictions_path, metrics_path)
    return BaselineEvaluationResult(predictions=result.predictions, metrics=persisted_metrics)


def main() -> None:
    """Run the offline baseline evaluation batch and print its artifact locations."""
    result = run_baselines()
    print(f"predictions={PREDICTIONS_PATH} rows={len(result.predictions)}")
    print(f"metrics={METRICS_PATH} rows={len(result.metrics)}")


if __name__ == "__main__":
    main()
