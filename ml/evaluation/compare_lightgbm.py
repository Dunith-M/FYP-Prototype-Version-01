"""Fair Day-06 comparison of frozen baseline and LightGBM validation evidence."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from ml.evaluation.folds import PROJECT_ROOT, VALIDATION_FOLDS
from ml.evaluation.metrics import mae, wape_details


FORECASTING_ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts" / "forecasting"
BASELINE_PREDICTIONS_PATH = FORECASTING_ARTIFACT_DIR / "baseline_predictions.parquet"
LIGHTGBM_PREDICTIONS_PATH = FORECASTING_ARTIFACT_DIR / "lightgbm_predictions.parquet"
LIGHTGBM_FOLD_SUMMARY_PATH = FORECASTING_ARTIFACT_DIR / "lightgbm_fold_summary.csv"
SELECTED_BASELINE_PATH = FORECASTING_ARTIFACT_DIR / "selected_baseline.json"
COMPARISON_METRICS_PATH = FORECASTING_ARTIFACT_DIR / "day06_lightgbm_comparison_metrics.csv"
COMPARISON_PREDICTIONS_PATH = FORECASTING_ARTIFACT_DIR / "day06_lightgbm_comparison_predictions.parquet"

OBSERVATION_KEY = (
    "fold_id",
    "validation_month",
    "target_month",
    "branch_code",
    "canonical_product_id",
)


class LightGBMComparisonError(ValueError):
    """Raised when baseline and challenger evidence cannot be fairly compared."""


@dataclass(frozen=True)
class LightGBMComparisonResult:
    """Fold-level shared-metric results and aligned prediction evidence."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame


def _load_selected_baseline_name(path: Path = SELECTED_BASELINE_PATH) -> str:
    """Read the pre-existing Day-05 baseline selection without changing it."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        name = str(record["model_name"])
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise LightGBMComparisonError(f"Could not read the selected baseline record: {path}") from error
    if not name:
        raise LightGBMComparisonError("Selected baseline record has no model_name.")
    return name


def _select_complete_baseline_predictions(
    baseline_predictions: pd.DataFrame, baseline_model_name: str
) -> pd.DataFrame:
    required = {*OBSERVATION_KEY, "model_name", "actual_units", "predicted_units", "prediction_available"}
    missing = sorted(required - set(baseline_predictions.columns))
    if missing:
        raise LightGBMComparisonError(f"Baseline predictions are missing columns: {missing}.")
    selected = baseline_predictions.loc[baseline_predictions["model_name"].eq(baseline_model_name)].copy()
    if selected.empty:
        raise LightGBMComparisonError(f"Baseline prediction evidence has no rows for {baseline_model_name!r}.")
    if not selected["prediction_available"].all():
        raise LightGBMComparisonError(
            f"Selected baseline {baseline_model_name!r} has unavailable validation predictions."
        )
    return selected


def _validate_and_align_predictions(
    baseline_predictions: pd.DataFrame, lightgbm_predictions: pd.DataFrame
) -> pd.DataFrame:
    required = {*OBSERVATION_KEY, "actual_units", "predicted_units", "prediction_available"}
    missing = sorted(required - set(lightgbm_predictions.columns))
    if missing:
        raise LightGBMComparisonError(f"LightGBM predictions are missing columns: {missing}.")
    if not lightgbm_predictions["prediction_available"].all():
        raise LightGBMComparisonError("LightGBM has unavailable validation predictions.")
    for label, frame in (("baseline", baseline_predictions), ("LightGBM", lightgbm_predictions)):
        if frame.duplicated(list(OBSERVATION_KEY)).any():
            raise LightGBMComparisonError(f"{label} predictions have duplicate validation observation keys.")

    baseline_keys = set(baseline_predictions.loc[:, OBSERVATION_KEY].itertuples(index=False, name=None))
    lightgbm_keys = set(lightgbm_predictions.loc[:, OBSERVATION_KEY].itertuples(index=False, name=None))
    if baseline_keys != lightgbm_keys:
        missing_from_lightgbm = sorted(baseline_keys - lightgbm_keys)[:5]
        extra_in_lightgbm = sorted(lightgbm_keys - baseline_keys)[:5]
        raise LightGBMComparisonError(
            "Validation observation keys differ: "
            f"baseline_rows={len(baseline_predictions)}, lightgbm_rows={len(lightgbm_predictions)}, "
            f"missing_from_lightgbm={missing_from_lightgbm}, extra_in_lightgbm={extra_in_lightgbm}."
        )

    aligned = baseline_predictions.loc[:, [*OBSERVATION_KEY, "actual_units", "predicted_units"]].merge(
        lightgbm_predictions.loc[:, [*OBSERVATION_KEY, "actual_units", "predicted_units"]],
        on=list(OBSERVATION_KEY),
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_lightgbm"),
    )
    if not aligned["actual_units_baseline"].equals(aligned["actual_units_lightgbm"]):
        raise LightGBMComparisonError("Baseline and LightGBM actual target values differ for identical keys.")
    return aligned.rename(
        columns={
            "actual_units_baseline": "actual_units",
            "predicted_units_baseline": "baseline_predicted_units",
            "predicted_units_lightgbm": "lightgbm_predicted_units",
        }
    ).drop(columns="actual_units_lightgbm")


def build_lightgbm_comparison(
    baseline_predictions: pd.DataFrame,
    lightgbm_predictions: pd.DataFrame,
    fold_summary: pd.DataFrame,
    baseline_model_name: str,
) -> LightGBMComparisonResult:
    """Calculate shared MAE/WAPE only after exact observation-key alignment."""
    baseline = _select_complete_baseline_predictions(baseline_predictions, baseline_model_name)
    aligned = _validate_and_align_predictions(baseline, lightgbm_predictions)

    summary_required = {"fold_id", "training_period", "training_cutoff_month", "validation_month"}
    missing_summary = sorted(summary_required - set(fold_summary.columns))
    if missing_summary:
        raise LightGBMComparisonError(f"LightGBM fold summary is missing columns: {missing_summary}.")

    expected_folds = {fold.fold_id for fold in VALIDATION_FOLDS}
    if set(aligned["fold_id"]) != expected_folds:
        raise LightGBMComparisonError("Comparison evidence does not cover exactly the frozen Day-05 folds.")

    records: list[dict[str, object]] = []
    for fold in VALIDATION_FOLDS:
        rows = aligned.loc[aligned["fold_id"].eq(fold.fold_id)]
        fold_metadata = fold_summary.loc[fold_summary["fold_id"].eq(fold.fold_id)]
        if len(fold_metadata) != 1:
            raise LightGBMComparisonError(f"Expected exactly one LightGBM summary row for {fold.fold_id}.")
        metadata = fold_metadata.iloc[0]
        for model_name, prediction_column in (
            (baseline_model_name, "baseline_predicted_units"),
            ("lightgbm_challenger", "lightgbm_predicted_units"),
        ):
            metric_wape = wape_details(rows["actual_units"], rows[prediction_column])
            records.append(
                {
                    "model_name": model_name,
                    "fold_id": fold.fold_id,
                    "training_period": metadata["training_period"],
                    "training_cutoff_month": metadata["training_cutoff_month"],
                    "validation_month": fold.validation_month,
                    "validation_row_count": len(rows),
                    "mae": mae(rows["actual_units"], rows[prediction_column]),
                    "wape": metric_wape.wape,
                    "wape_available": metric_wape.denominator_available,
                    "actual_units_sum": metric_wape.actual_volume,
                    "absolute_error_sum": metric_wape.absolute_error_total,
                    "observation_keys_identical": True,
                }
            )

    metrics = pd.DataFrame(records).sort_values(["fold_id", "model_name"], kind="stable").reset_index(drop=True)
    return LightGBMComparisonResult(metrics=metrics, predictions=aligned.sort_values(list(OBSERVATION_KEY), kind="stable"))


def save_lightgbm_comparison_artifacts(
    result: LightGBMComparisonResult,
    metrics_path: Path = COMPARISON_METRICS_PATH,
    predictions_path: Path = COMPARISON_PREDICTIONS_PATH,
) -> None:
    """Save new Day-06 comparison artifacts without overwriting Day-05 evidence."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(metrics_path, index=False, na_rep="NaN")
    result.predictions.to_parquet(predictions_path, index=False)


def run_lightgbm_comparison() -> LightGBMComparisonResult:
    """Load saved Day-05/Step-5 evidence, align it, and persist Day-06 comparison evidence."""
    result = build_lightgbm_comparison(
        pd.read_parquet(BASELINE_PREDICTIONS_PATH),
        pd.read_parquet(LIGHTGBM_PREDICTIONS_PATH),
        pd.read_csv(LIGHTGBM_FOLD_SUMMARY_PATH),
        _load_selected_baseline_name(),
    )
    save_lightgbm_comparison_artifacts(result)
    return result


def main() -> None:
    """Create and report the comparison artifacts; do not select a winner."""
    result = run_lightgbm_comparison()
    print(f"metrics={COMPARISON_METRICS_PATH} rows={len(result.metrics)}")
    print(f"predictions={COMPARISON_PREDICTIONS_PATH} rows={len(result.predictions)}")


if __name__ == "__main__":
    main()
