"""Pooled LightGBM validation predictions on the frozen Day-05 folds.

This runner uses the existing Day-04 ``feature_month`` artifact and
``split_fold`` function.  It fits one pooled challenger per chronological
fold; it never creates a model per branch, product, or branch-product pair.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from ml.evaluation.folds import (
    FEATURE_MONTH_PATH,
    PROJECT_ROOT,
    TARGET_MONTH_COLUMN,
    VALIDATION_FOLDS,
    split_fold,
)
from ml.evaluation.run_baselines import TARGET_VALUE_COLUMN, load_feature_data
from ml.models.lightgbm_challenger import build_lightgbm_challenger


MODEL_NAME = "lightgbm_challenger"
IDENTIFIER_COLUMNS = ("branch_code", "canonical_product_id", "year_month")
CATEGORICAL_FEATURE_COLUMNS = (
    "branch_code",
    "canonical_product_id",
    "family_id",
    "family_desc",
    "price_bucket",
    "branch_name",
)
NUMERIC_FEATURE_COLUMNS = (
    "lag_1",
    "lag_2",
    "lag_3",
    "trailing_mean_3",
    "trailing_sum_3",
    "months_with_positive_sales",
    "intermittency",
    "last_observed_qty",
    "mean_observed_qty_30d",
    "inventory_observation_count",
    "availability_unknown_rate",
    "valid_price",
    "month",
    "quarter",
    "holiday_count",
    "weekend_count",
)
MODEL_FEATURE_COLUMNS = (*CATEGORICAL_FEATURE_COLUMNS, *NUMERIC_FEATURE_COLUMNS)
FORECASTING_ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts" / "forecasting"
PREDICTIONS_PATH = FORECASTING_ARTIFACT_DIR / "lightgbm_predictions.parquet"
FOLD_SUMMARY_PATH = FORECASTING_ARTIFACT_DIR / "lightgbm_fold_summary.csv"


class LightGBMEvaluationError(ValueError):
    """Raised when frozen feature rows cannot safely train the challenger."""


@dataclass(frozen=True)
class LightGBMEvaluationResult:
    """Validation prediction evidence and pooled training metadata."""

    predictions: pd.DataFrame
    fold_summary: pd.DataFrame


def _validate_model_columns(frame: pd.DataFrame) -> None:
    required = {*IDENTIFIER_COLUMNS, TARGET_MONTH_COLUMN, TARGET_VALUE_COLUMN, *MODEL_FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LightGBMEvaluationError(f"Feature data is missing required LightGBM columns: {missing}.")


def _categorical_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Return explicit category labels without dropping or filtering any rows."""
    return frame.loc[:, CATEGORICAL_FEATURE_COLUMNS].astype("string").fillna("__MISSING_CATEGORY__")


def _prepare_fold_features(
    train_df: pd.DataFrame, validation_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categories using training rows only and retain numeric missingness.

    LightGBM natively handles numeric ``NaN`` values.  Categorical values not
    seen during training are encoded as -1 in validation rather than causing
    row loss or a refit on validation/full data.
    """
    _validate_model_columns(train_df)
    _validate_model_columns(validation_df)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.float64,
    )
    train_categories = encoder.fit_transform(_categorical_values(train_df))
    validation_categories = encoder.transform(_categorical_values(validation_df))

    train_numeric = train_df.loc[:, NUMERIC_FEATURE_COLUMNS].copy()
    validation_numeric = validation_df.loc[:, NUMERIC_FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURE_COLUMNS:
        train_numeric[column] = pd.to_numeric(train_numeric[column], errors="raise")
        validation_numeric[column] = pd.to_numeric(validation_numeric[column], errors="raise")

    train_features = pd.concat(
        [
            pd.DataFrame(train_categories, columns=CATEGORICAL_FEATURE_COLUMNS, index=train_df.index),
            train_numeric,
        ],
        axis=1,
    )
    validation_features = pd.concat(
        [
            pd.DataFrame(validation_categories, columns=CATEGORICAL_FEATURE_COLUMNS, index=validation_df.index),
            validation_numeric,
        ],
        axis=1,
    )
    return train_features, validation_features


def generate_lightgbm_predictions(feature_df: pd.DataFrame) -> LightGBMEvaluationResult:
    """Fit one pooled challenger per frozen fold and return validation evidence.

    This function intentionally stops at predictions.  Metric comparison and
    model selection remain separate later workflow steps.
    """
    _validate_model_columns(feature_df)
    prediction_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for fold in VALIDATION_FOLDS:
        train_df, validation_df = split_fold(feature_df, fold)
        train_features, validation_features = _prepare_fold_features(train_df, validation_df)
        target = pd.to_numeric(train_df[TARGET_VALUE_COLUMN], errors="raise")

        model = build_lightgbm_challenger()
        model.fit(train_features, target, categorical_feature=list(CATEGORICAL_FEATURE_COLUMNS))
        predictions = model.predict(validation_features)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold_id": fold.fold_id,
                    "validation_month": fold.validation_month,
                    "target_month": validation_df[TARGET_MONTH_COLUMN].astype("string").to_numpy(),
                    "branch_code": validation_df["branch_code"].astype("string").to_numpy(),
                    "canonical_product_id": validation_df["canonical_product_id"].astype("string").to_numpy(),
                    "family_id": validation_df["family_id"].astype("string").to_numpy(),
                    "actual_units": pd.to_numeric(validation_df[TARGET_VALUE_COLUMN], errors="raise").to_numpy(),
                    "predicted_units": predictions,
                    "model_name": MODEL_NAME,
                    "prediction_available": np.isfinite(predictions),
                }
            )
        )
        train_target_months = train_df[TARGET_MONTH_COLUMN].astype("string")
        summaries.append(
            {
                "fold_id": fold.fold_id,
                "training_period": f"{train_target_months.min()} through {train_target_months.max()}",
                "training_cutoff_month": fold.training_cutoff_month,
                "validation_month": fold.validation_month,
                "training_row_count": len(train_df),
                "validation_row_count": len(validation_df),
                "training_branch_count": int(train_df["branch_code"].nunique()),
                "training_product_count": int(train_df["canonical_product_id"].nunique()),
                "model_type": type(model).__name__,
                "training_status": "PASS",
            }
        )

    output = pd.concat(prediction_frames, ignore_index=True)
    output = output.sort_values(
        ["fold_id", "branch_code", "canonical_product_id", "target_month"], kind="stable"
    ).reset_index(drop=True)
    return LightGBMEvaluationResult(
        predictions=output,
        fold_summary=pd.DataFrame(summaries),
    )


def save_lightgbm_prediction_artifacts(
    result: LightGBMEvaluationResult,
    predictions_path: Path = PREDICTIONS_PATH,
    fold_summary_path: Path = FOLD_SUMMARY_PATH,
) -> None:
    """Persist validation evidence only; fitted fold models are not saved."""
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    fold_summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(predictions_path, index=False)
    result.fold_summary.to_csv(fold_summary_path, index=False)


def run_lightgbm(
    feature_path: Path = FEATURE_MONTH_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    fold_summary_path: Path = FOLD_SUMMARY_PATH,
) -> LightGBMEvaluationResult:
    """Load the frozen artifact, create pooled fold predictions, and save them."""
    result = generate_lightgbm_predictions(load_feature_data(feature_path))
    save_lightgbm_prediction_artifacts(result, predictions_path, fold_summary_path)
    return result


def main() -> None:
    """Run pooled LightGBM fold training and print prediction artifact paths."""
    result = run_lightgbm()
    print(f"predictions={PREDICTIONS_PATH} rows={len(result.predictions)}")
    print(f"fold_summary={FOLD_SUMMARY_PATH} rows={len(result.fold_summary)}")


if __name__ == "__main__":
    main()
