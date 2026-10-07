"""Transparent comparison of the Day-05 winning baseline and LightGBM."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.compare_lightgbm import (
    BASELINE_PREDICTIONS_PATH,
    COMPARISON_PREDICTIONS_PATH,
    OBSERVATION_KEY,
    SELECTED_BASELINE_PATH,
)
from ml.evaluation.metrics import mae, wape_details


FORECASTING_ARTIFACT_DIR = BASELINE_PREDICTIONS_PATH.parent
OVERALL_PATH = FORECASTING_ARTIFACT_DIR / "model_comparison_overall.csv"
BY_FOLD_PATH = FORECASTING_ARTIFACT_DIR / "model_comparison_by_fold.csv"
BY_BRANCH_PATH = FORECASTING_ARTIFACT_DIR / "model_comparison_by_branch.csv"
BY_FAMILY_PATH = FORECASTING_ARTIFACT_DIR / "model_comparison_by_family.csv"
LIGHTGBM_MODEL_NAME = "lightgbm_challenger"
TIE_TOLERANCE = 1e-12


class ModelComparisonError(ValueError):
    """Raised when saved evaluation evidence cannot support a fair comparison."""


@dataclass(frozen=True)
class ModelComparisonResult:
    """Comparison tables at the Day-05 reporting scopes."""

    overall: pd.DataFrame
    by_fold: pd.DataFrame
    by_branch: pd.DataFrame
    by_family: pd.DataFrame


def _selected_baseline_name(path: Path = SELECTED_BASELINE_PATH) -> str:
    """Return the Day-05 winner recorded by the established selection process."""
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["model_name"])
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ModelComparisonError(f"Cannot read Day-05 selected baseline: {path}") from error


def _add_family_metadata(
    aligned_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_model_name: str,
) -> pd.DataFrame:
    """Attach existing baseline family metadata after confirming matching keys."""
    required = {*OBSERVATION_KEY, "actual_units", "baseline_predicted_units", "lightgbm_predicted_units"}
    missing = sorted(required - set(aligned_predictions.columns))
    if missing:
        raise ModelComparisonError(f"Day-06 aligned predictions are missing columns: {missing}.")
    baseline = baseline_predictions.loc[
        baseline_predictions["model_name"].eq(baseline_model_name),
        [*OBSERVATION_KEY, "family_id", "actual_units"],
    ].copy()
    if baseline.duplicated(list(OBSERVATION_KEY)).any():
        raise ModelComparisonError("Day-05 selected-baseline predictions have duplicate observation keys.")
    aligned_keys = set(aligned_predictions.loc[:, OBSERVATION_KEY].itertuples(index=False, name=None))
    baseline_keys = set(baseline.loc[:, OBSERVATION_KEY].itertuples(index=False, name=None))
    if aligned_keys != baseline_keys:
        raise ModelComparisonError(
            "Day-06 and selected-baseline observation keys differ; cannot derive family comparison."
        )
    enriched = aligned_predictions.merge(
        baseline,
        on=list(OBSERVATION_KEY),
        how="inner",
        validate="one_to_one",
        suffixes=("", "_baseline_source"),
    )
    if not enriched["actual_units"].equals(enriched["actual_units_baseline_source"]):
        raise ModelComparisonError("Actual target values differ between Day-05 and Day-06 evidence.")
    return enriched.drop(columns="actual_units_baseline_source")


def _winner_and_metric(
    baseline_mae: float,
    baseline_wape_available: bool,
    baseline_wape: float,
    lightgbm_mae: float,
    lightgbm_wape_available: bool,
    lightgbm_wape: float,
) -> tuple[str, str]:
    """Use WAPE where defined; use MAE for zero-volume slices as Day-05 does."""
    if baseline_wape_available and lightgbm_wape_available:
        metric = "WAPE"
        difference = lightgbm_wape - baseline_wape
    else:
        metric = "MAE"
        difference = lightgbm_mae - baseline_mae
    if np.isclose(difference, 0.0, atol=TIE_TOLERANCE, rtol=0.0):
        return metric, "TIE"
    return metric, LIGHTGBM_MODEL_NAME if difference < 0 else "BASELINE"


def _comparison_record(
    rows: pd.DataFrame,
    baseline_model_name: str,
    *,
    scope: str,
    fold_id: str | None = None,
    validation_month: str | None = None,
    branch_code: str | None = None,
    family_id: str | None = None,
) -> dict[str, object]:
    baseline_wape = wape_details(rows["actual_units"], rows["baseline_predicted_units"])
    lightgbm_wape = wape_details(rows["actual_units"], rows["lightgbm_predicted_units"])
    baseline_mae = mae(rows["actual_units"], rows["baseline_predicted_units"])
    lightgbm_mae = mae(rows["actual_units"], rows["lightgbm_predicted_units"])
    comparison_metric, winner = _winner_and_metric(
        baseline_mae,
        baseline_wape.denominator_available,
        baseline_wape.wape,
        lightgbm_mae,
        lightgbm_wape.denominator_available,
        lightgbm_wape.wape,
    )
    wape_improvement_pp = (
        (baseline_wape.wape - lightgbm_wape.wape) * 100
        if baseline_wape.denominator_available and lightgbm_wape.denominator_available
        else float("nan")
    )
    relative_improvement = (
        (baseline_wape.wape - lightgbm_wape.wape) / baseline_wape.wape
        if baseline_wape.denominator_available and baseline_wape.wape != 0
        else float("nan")
    )
    return {
        "scope": scope,
        "fold_id": fold_id,
        "validation_month": validation_month,
        "branch_code": branch_code,
        "family_id": family_id,
        "observation_count": len(rows),
        "actual_units_sum": baseline_wape.actual_volume,
        "baseline_model_name": baseline_model_name,
        "baseline_mae": baseline_mae,
        "baseline_wape": baseline_wape.wape,
        "baseline_wape_available": baseline_wape.denominator_available,
        "lightgbm_mae": lightgbm_mae,
        "lightgbm_wape": lightgbm_wape.wape,
        "lightgbm_wape_available": lightgbm_wape.denominator_available,
        "mae_difference_lightgbm_minus_baseline": lightgbm_mae - baseline_mae,
        "wape_absolute_improvement_percentage_points": wape_improvement_pp,
        "wape_relative_improvement": relative_improvement,
        "comparison_metric": comparison_metric,
        "winner_on_comparison_metric": winner,
        "observation_keys_identical": True,
    }


def build_model_comparison(
    aligned_predictions: pd.DataFrame, baseline_predictions: pd.DataFrame, baseline_model_name: str
) -> ModelComparisonResult:
    """Build overall, fold, branch, and family comparisons from matched OOF rows."""
    evidence = _add_family_metadata(aligned_predictions, baseline_predictions, baseline_model_name)
    overall = pd.DataFrame([_comparison_record(evidence, baseline_model_name, scope="OVERALL")])
    by_fold = pd.DataFrame(
        [
            _comparison_record(
                rows,
                baseline_model_name,
                scope="FOLD",
                fold_id=str(fold_id),
                validation_month=str(rows["validation_month"].iloc[0]),
            )
            for fold_id, rows in evidence.groupby("fold_id", sort=True)
        ]
    )
    by_branch = pd.DataFrame(
        [
            _comparison_record(rows, baseline_model_name, scope="BRANCH", branch_code=str(branch_code))
            for branch_code, rows in evidence.groupby("branch_code", sort=True)
        ]
    )
    by_family = pd.DataFrame(
        [
            _comparison_record(
                rows,
                baseline_model_name,
                scope="FAMILY",
                family_id=None if pd.isna(family_id) else str(family_id),
            )
            for family_id, rows in evidence.groupby("family_id", dropna=False, sort=True)
        ]
    )
    return ModelComparisonResult(overall, by_fold, by_branch, by_family)


def save_model_comparison_artifacts(result: ModelComparisonResult) -> None:
    """Write new comparison-only artifacts without replacing Day-05/Day-06 evidence."""
    for table, path in (
        (result.overall, OVERALL_PATH),
        (result.by_fold, BY_FOLD_PATH),
        (result.by_branch, BY_BRANCH_PATH),
        (result.by_family, BY_FAMILY_PATH),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, na_rep="NaN")


def run_model_comparison() -> ModelComparisonResult:
    """Compare saved matched predictions; no model is retrained or selected here."""
    baseline_model_name = _selected_baseline_name()
    result = build_model_comparison(
        pd.read_parquet(COMPARISON_PREDICTIONS_PATH),
        pd.read_parquet(BASELINE_PREDICTIONS_PATH),
        baseline_model_name,
    )
    save_model_comparison_artifacts(result)
    return result


def main() -> None:
    """Create comparison artifacts and print their locations."""
    result = run_model_comparison()
    print(f"overall={OVERALL_PATH} rows={len(result.overall)}")
    print(f"by_fold={BY_FOLD_PATH} rows={len(result.by_fold)}")
    print(f"by_branch={BY_BRANCH_PATH} rows={len(result.by_branch)}")
    print(f"by_family={BY_FAMILY_PATH} rows={len(result.by_family)}")


if __name__ == "__main__":
    main()
