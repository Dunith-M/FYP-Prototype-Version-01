"""Package completed Day-06 evidence without training or changing source artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from ml.evaluation.confidence import (
    ADEQUATE,
    AVAILABLE,
    AVAILABLE_WITH_WARNING,
    DEFAULT_CONFIDENCE_CONFIG,
    INSUFFICIENT,
    LIMITED,
    WITHHELD,
    apply_confidence_to_forecasts,
)
from ml.evaluation.folds import VALIDATION_FOLDS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECASTING_ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts" / "forecasting"
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_month.parquet"
FEATURE_MANIFEST_PATH = PROJECT_ROOT / "data" / "artifacts" / "feature_month_manifest.json"
DAY06_ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts" / "day06"

COMPARISON_SOURCE_PATH = FORECASTING_ARTIFACT_DIR / "day06_lightgbm_comparison_metrics.csv"
LIGHTGBM_PREDICTIONS_SOURCE_PATH = FORECASTING_ARTIFACT_DIR / "lightgbm_predictions.parquet"
BASELINE_PREDICTIONS_SOURCE_PATH = FORECASTING_ARTIFACT_DIR / "baseline_predictions.parquet"
SELECTION_SOURCE_PATH = FORECASTING_ARTIFACT_DIR / "selected_forecast_model.json"
OVERALL_COMPARISON_SOURCE_PATH = FORECASTING_ARTIFACT_DIR / "model_comparison_overall.csv"

MODEL_VERSION = "day06_v1"
FOLD_DEFINITION = "day05_chronological_folds"
MODE = "HISTORICAL_RESEARCH"
LIGHTGBM_MODEL_NAME = "lightgbm_challenger"


class Day06PackageError(ValueError):
    """Raised when completed Day-06 evidence cannot be packaged safely."""


@dataclass(frozen=True)
class Day06PackageResult:
    """Paths and counts returned after a validated, reproducible package write."""

    output_directory: Path
    comparison_rows: int
    lightgbm_prediction_rows: int
    forecast_rows: int
    evidence_counts: dict[str, int]
    manifest: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise Day06PackageError(f"Required JSON artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_to_project(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _check_existing_package(output_directory: Path, selected_model: str) -> None:
    """Permit only a verified repeat of the same named Day-06 package."""
    expected_paths = [
        output_directory / "model_comparison.csv",
        output_directory / "lightgbm_predictions.parquet",
        output_directory / "forecast_with_confidence.parquet",
        output_directory / "model_manifest.json",
    ]
    existing = [path for path in expected_paths if path.exists()]
    if not existing:
        return

    manifest_path = output_directory / "model_manifest.json"
    if not manifest_path.exists():
        raise Day06PackageError(
            "A partial Day-06 package already exists without a manifest; it will not be overwritten."
        )
    existing_manifest = _read_json(manifest_path)
    if (
        existing_manifest.get("model_version") != MODEL_VERSION
        or existing_manifest.get("selected_model") != selected_model
    ):
        raise Day06PackageError(
            "The existing Day-06 package represents a different run or selected model; "
            "it will not be overwritten."
        )


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Day06PackageError(f"{label} is missing required columns: {missing}")


def _build_manifest(
    *,
    comparison: pd.DataFrame,
    selection: dict[str, object],
    feature_manifest: dict[str, object],
) -> dict[str, object]:
    selected_model = str(selection["selected_model"])
    selected_overall = pd.read_csv(OVERALL_COMPARISON_SOURCE_PATH).iloc[0]
    if selected_model == str(selection["lightgbm_model"]):
        overall_mae = float(selected_overall["lightgbm_mae"])
        overall_wape = float(selected_overall["lightgbm_wape"])
        model_type = "LightGBM"
    else:
        overall_mae = float(selected_overall["baseline_mae"])
        overall_wape = float(selected_overall["baseline_wape"])
        model_type = f"Baseline: {selected_model}"

    metric_columns = [
        "model_name", "fold_id", "training_period", "training_cutoff_month",
        "validation_month", "validation_row_count", "mae", "wape",
        "wape_available", "actual_units_sum", "absolute_error_sum",
        "observation_keys_identical",
    ]
    return {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": feature_manifest["feature_version"],
        "feature_sha256": feature_manifest.get("sha256"),
        "target_definition": feature_manifest["target_column"],
        "fold_definition": FOLD_DEFINITION,
        "fold_count": len(VALIDATION_FOLDS),
        "selected_model": selected_model,
        "model_type": model_type,
        "selection_status": selection["status"],
        "selection_basis": selection["selection_scope"],
        "selection_reason": selection["selection_reason"],
        "baseline_model": selection["baseline_model"],
        "challenger_model": selection["lightgbm_model"],
        "comparison_metric": "WAPE with MAE reported alongside it",
        "overall_mae": overall_mae,
        "overall_wape": overall_wape,
        "fold_metrics": comparison.loc[:, metric_columns].to_dict(orient="records"),
        "confidence_policy_version": "confidence_rules_v1",
        "confidence_threshold_mode": "RESEARCH_PROTOTYPE",
        "confidence_configuration": {
            "policy_status": "NOT_SINGER_APPROVED_POLICY",
            **DEFAULT_CONFIDENCE_CONFIG.to_dict(),
        },
        "mode": MODE,
        "source_artifacts": {
            "comparison_metrics": _relative_to_project(COMPARISON_SOURCE_PATH),
            "lightgbm_predictions": _relative_to_project(LIGHTGBM_PREDICTIONS_SOURCE_PATH),
            "baseline_predictions": _relative_to_project(BASELINE_PREDICTIONS_SOURCE_PATH),
            "selected_model": _relative_to_project(SELECTION_SOURCE_PATH),
            "feature_table": _relative_to_project(FEATURE_PATH),
            "feature_manifest": _relative_to_project(FEATURE_MANIFEST_PATH),
        },
    }


def validate_day06_package(output_directory: Path = DAY06_ARTIFACT_DIR) -> Day06PackageResult:
    """Reopen and cross-check all four persisted Day-06 package artifacts."""
    comparison_path = output_directory / "model_comparison.csv"
    predictions_path = output_directory / "lightgbm_predictions.parquet"
    confidence_path = output_directory / "forecast_with_confidence.parquet"
    manifest_path = output_directory / "model_manifest.json"
    for path in (comparison_path, predictions_path, confidence_path, manifest_path):
        if not path.exists() or path.stat().st_size == 0:
            raise Day06PackageError(f"Package artifact is missing or empty: {path}")

    comparison = pd.read_csv(comparison_path)
    lightgbm_predictions = pd.read_parquet(predictions_path)
    forecasts = pd.read_parquet(confidence_path)
    manifest = _read_json(manifest_path)
    _require_columns(comparison, {"model_name", "fold_id", "validation_month", "mae", "wape"}, "model comparison")
    _require_columns(
        lightgbm_predictions,
        {"branch_code", "canonical_product_id", "validation_month", "fold_id", "actual_units", "predicted_units"},
        "LightGBM predictions",
    )
    _require_columns(
        forecasts,
        {
            "branch_code", "canonical_product_id", "planning_month", "raw_prediction",
            "forecast_value", "selected_model", "evidence_class", "evidence_reason_code",
            "evidence_warning", "forecast_status",
        },
        "confidence forecasts",
    )

    expected_folds = {fold.fold_id for fold in VALIDATION_FOLDS}
    comparison_folds = set(comparison["fold_id"])
    prediction_folds = set(lightgbm_predictions["fold_id"])
    lightgbm_comparison_folds = set(
        comparison.loc[comparison["model_name"].eq(LIGHTGBM_MODEL_NAME), "fold_id"]
    )
    if comparison_folds != expected_folds or prediction_folds != lightgbm_comparison_folds:
        raise Day06PackageError("Package fold coverage does not match the frozen Day-05 fold definition.")
    if not lightgbm_predictions["target_month"].eq(lightgbm_predictions["validation_month"]).all():
        raise Day06PackageError("LightGBM package predictions are not validation-month out-of-fold rows.")
    if not comparison["observation_keys_identical"].astype(bool).all():
        raise Day06PackageError("Comparison package contains unverified validation observation keys.")
    if int(manifest.get("fold_count", -1)) != len(VALIDATION_FOLDS):
        raise Day06PackageError("Manifest fold count does not match the frozen Day-05 folds.")
    if manifest.get("fold_definition") != FOLD_DEFINITION or manifest.get("mode") != MODE:
        raise Day06PackageError("Manifest fold definition or mode is not the required research provenance.")
    source_selection = _read_json(SELECTION_SOURCE_PATH)
    if manifest.get("selected_model") != source_selection.get("selected_model"):
        raise Day06PackageError("Manifest selected model does not match the Step-8 selection record.")

    valid_classes = {ADEQUATE, LIMITED, INSUFFICIENT}
    if not set(forecasts["evidence_class"]).issubset(valid_classes):
        raise Day06PackageError("Confidence artifact contains an unsupported evidence class.")
    status_by_class = {
        ADEQUATE: AVAILABLE,
        LIMITED: AVAILABLE_WITH_WARNING,
        INSUFFICIENT: WITHHELD,
    }
    expected_statuses = forecasts["evidence_class"].map(status_by_class)
    if not forecasts["forecast_status"].eq(expected_statuses).all():
        raise Day06PackageError("Confidence statuses do not match their evidence classes.")
    withheld = forecasts["forecast_status"].eq(WITHHELD)
    if not forecasts.loc[withheld, "forecast_value"].isna().all():
        raise Day06PackageError("WITHHELD forecasts must not retain a displayed forecast value.")

    return Day06PackageResult(
        output_directory=output_directory,
        comparison_rows=len(comparison),
        lightgbm_prediction_rows=len(lightgbm_predictions),
        forecast_rows=len(forecasts),
        evidence_counts={
            evidence_class: int(forecasts["evidence_class"].eq(evidence_class).sum())
            for evidence_class in (ADEQUATE, LIMITED, INSUFFICIENT)
        },
        manifest=manifest,
    )


def build_day06_package(output_directory: Path = DAY06_ARTIFACT_DIR) -> Day06PackageResult:
    """Copy completed Day-06 evidence into one reproducible artifact package."""
    selection = _read_json(SELECTION_SOURCE_PATH)
    feature_manifest = _read_json(FEATURE_MANIFEST_PATH)
    selected_model = str(selection["selected_model"])
    _check_existing_package(output_directory, selected_model)

    comparison = pd.read_csv(COMPARISON_SOURCE_PATH)
    lightgbm_predictions = pd.read_parquet(LIGHTGBM_PREDICTIONS_SOURCE_PATH)
    baseline_predictions = pd.read_parquet(BASELINE_PREDICTIONS_SOURCE_PATH)
    feature_month = pd.read_parquet(FEATURE_PATH)
    _require_columns(comparison, {"model_name", "fold_id", "validation_month", "mae", "wape"}, "source comparison")
    _require_columns(lightgbm_predictions, {"fold_id", "validation_month", "actual_units", "predicted_units"}, "source LightGBM predictions")
    _require_columns(baseline_predictions, {"model_name", "predicted_units"}, "source baseline predictions")

    selected_forecasts = baseline_predictions.loc[
        baseline_predictions["model_name"].eq(selected_model)
    ].copy()
    if selected_forecasts.empty:
        raise Day06PackageError(f"No saved baseline predictions exist for selected model {selected_model!r}.")
    confidence_forecasts = apply_confidence_to_forecasts(selected_forecasts, feature_month)
    keep_columns = [
        "branch_code", "canonical_product_id", "family_id", "fold_id", "validation_month",
        "target_month", "planning_month", "actual_units", "raw_prediction", "forecast_value",
        "selected_model", "evidence_class", "evidence_reason_code", "evidence_warning",
        "forecast_status", "confidence_rule_version", "confidence_config_version",
        "confidence_config_mode",
    ]
    confidence_forecasts = confidence_forecasts.loc[:, keep_columns]
    manifest = _build_manifest(
        comparison=comparison,
        selection=selection,
        feature_manifest=feature_manifest,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_directory / "model_comparison.csv", index=False)
    lightgbm_predictions.to_parquet(output_directory / "lightgbm_predictions.parquet", index=False)
    confidence_forecasts.to_parquet(output_directory / "forecast_with_confidence.parquet", index=False)
    (output_directory / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validate_day06_package(output_directory)


def main() -> None:
    """Write the Day-06 package from existing artifacts only; no model is retrained."""
    result = build_day06_package()
    print(f"directory={result.output_directory}")
    print(f"comparison_rows={result.comparison_rows}")
    print(f"lightgbm_prediction_rows={result.lightgbm_prediction_rows}")
    print(f"forecast_rows={result.forecast_rows}")
    print(f"evidence_counts={result.evidence_counts}")
    print(f"selected_model={result.manifest['selected_model']}")


if __name__ == "__main__":
    main()
