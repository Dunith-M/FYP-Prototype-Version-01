"""Aggregation and deterministic selection from baseline evaluation evidence."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from ml.evaluation.folds import FINAL_TEST_MONTH, MODEL_SELECTION_MONTHS, VALIDATION_FOLDS
from ml.evaluation.metrics import mae, wape_details


REPORT_COLUMNS = (
    "scope",
    "fold_id",
    "validation_month",
    "model_name",
    "branch_code",
    "family_id",
    "prediction_row_count",
    "number_of_rows",
    "unevaluable_row_count",
    "actual_units_sum",
    "absolute_error_sum",
    "mae",
    "wape",
    "wape_available",
)
REQUIRED_PREDICTION_COLUMNS = {
    "fold_id",
    "validation_month",
    "target_month",
    "branch_code",
    "family_id",
    "actual_units",
    "predicted_units",
    "model_name",
    "prediction_available",
}
EXPECTED_BASELINE_MODELS = ("zero_forecast", "last_observation", "trailing_mean_3")
TIE_BREAK_ORDER = {"zero_forecast": 0, "last_observation": 1, "trailing_mean_3": 2}
SELECTION_TOLERANCE = 1e-12


class BaselineReportError(ValueError):
    """Raised when persisted baseline evidence cannot form a safe report."""


@dataclass(frozen=True)
class BaselineSelection:
    """A reproducible simple-baseline decision from pooled validation evidence."""

    model_name: str
    selection_metric: str
    tie_break_used: bool


def _validate_predictions(predictions: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise BaselineReportError(f"Prediction evidence is missing required columns: {missing}.")
    target_months = set(predictions["target_month"].astype(str))
    unexpected = target_months - set(MODEL_SELECTION_MONTHS)
    if unexpected or FINAL_TEST_MONTH in target_months:
        raise BaselineReportError(
            f"Prediction evidence contains forbidden target months: {sorted(unexpected)}."
        )


def _metric_record(
    rows: pd.DataFrame,
    *,
    scope: str,
    fold_id: str = "ALL",
    validation_month: str = "ALL",
    branch_code: str | None = None,
    family_id: str | None = None,
) -> dict[str, object]:
    """Calculate a scope metric directly from its persisted evidence rows."""
    evaluable = rows.loc[rows["prediction_available"]]
    if evaluable.empty:
        metric_values = {
            "actual_units_sum": 0.0,
            "absolute_error_sum": 0.0,
            "mae": float("nan"),
            "wape": float("nan"),
            "wape_available": False,
        }
    else:
        actual = evaluable["actual_units"]
        predicted = evaluable["predicted_units"]
        wape_result = wape_details(actual, predicted)
        metric_values = {
            "actual_units_sum": wape_result.actual_volume,
            "absolute_error_sum": wape_result.absolute_error_total,
            "mae": mae(actual, predicted),
            "wape": wape_result.wape,
            "wape_available": wape_result.denominator_available,
        }
    return {
        "scope": scope,
        "fold_id": fold_id,
        "validation_month": validation_month,
        "model_name": str(rows["model_name"].iloc[0]),
        "branch_code": branch_code,
        "family_id": family_id,
        "prediction_row_count": len(rows),
        "number_of_rows": len(evaluable),
        "unevaluable_row_count": len(rows) - len(evaluable),
        **metric_values,
    }


def build_baseline_evaluation_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build OVERALL, FOLD, BRANCH, and FAMILY views from prediction evidence."""
    _validate_predictions(predictions)
    records: list[dict[str, object]] = []

    for _, rows in predictions.groupby("model_name", sort=True):
        records.append(_metric_record(rows, scope="OVERALL"))
    for (fold_id, validation_month, _), rows in predictions.groupby(
        ["fold_id", "validation_month", "model_name"], sort=True
    ):
        records.append(
            _metric_record(rows, scope="FOLD", fold_id=str(fold_id), validation_month=str(validation_month))
        )
    for (branch_code, _), rows in predictions.groupby(["branch_code", "model_name"], sort=True):
        records.append(_metric_record(rows, scope="BRANCH", branch_code=str(branch_code)))
    for (family_id, _), rows in predictions.groupby(
        ["family_id", "model_name"], dropna=False, sort=True
    ):
        family_value = None if pd.isna(family_id) else str(family_id)
        records.append(_metric_record(rows, scope="FAMILY", family_id=family_value))

    report = pd.DataFrame(records, columns=REPORT_COLUMNS)
    scope_order = {"OVERALL": 0, "FOLD": 1, "BRANCH": 2, "FAMILY": 3}
    report["_scope_order"] = report["scope"].map(scope_order)
    return report.sort_values(
        ["_scope_order", "fold_id", "branch_code", "family_id", "model_name"],
        kind="stable",
        na_position="last",
    ).drop(columns="_scope_order").reset_index(drop=True)


def save_baseline_evaluation_table(report: pd.DataFrame, path) -> None:
    """Persist the deterministic long-form evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False, na_rep="NaN")


def select_baseline(metrics: pd.DataFrame) -> BaselineSelection:
    """Select from complete Dec--Feb evidence using pooled WAPE then MAE.

    When WAPE is unavailable for every candidate, MAE becomes the primary
    criterion. Exact remaining ties use the documented simple-model order.
    """
    required = {"scope", "fold_id", "validation_month", "model_name", "mae", "wape", "wape_available"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise BaselineReportError(f"Metrics are missing required columns: {missing}.")

    expected_models = set(EXPECTED_BASELINE_MODELS)
    required_scopes = {"OVERALL", "FOLD", "BRANCH", "FAMILY"}
    if not required_scopes.issubset(set(metrics["scope"])):
        raise BaselineReportError("Metrics do not contain all required evaluation scopes.")
    overall = metrics.loc[metrics["scope"].eq("OVERALL")].copy()
    if set(overall["model_name"]) != expected_models or overall["model_name"].duplicated().any():
        raise BaselineReportError("OVERALL metrics must contain exactly one row for each baseline.")

    expected_fold_models = {
        (fold.fold_id, fold.validation_month, model_name)
        for fold in VALIDATION_FOLDS
        for model_name in expected_models
    }
    actual_fold_models = set(
        metrics.loc[metrics["scope"].eq("FOLD"), ["fold_id", "validation_month", "model_name"]]
        .itertuples(index=False, name=None)
    )
    if actual_fold_models != expected_fold_models:
        raise BaselineReportError("FOLD metrics do not cover every frozen fold/month and baseline.")
    if overall["mae"].isna().any():
        raise BaselineReportError("OVERALL MAE is unavailable for at least one baseline.")

    valid_wape = overall.loc[overall["wape_available"] & overall["wape"].notna()]
    if not valid_wape.empty:
        minimum_wape = valid_wape["wape"].min()
        candidates = valid_wape.loc[(valid_wape["wape"] - minimum_wape).abs() <= SELECTION_TOLERANCE]
        selection_metric = "WAPE"
    else:
        candidates = overall
        selection_metric = "MAE"

    minimum_mae = candidates["mae"].min()
    candidates = candidates.loc[(candidates["mae"] - minimum_mae).abs() <= SELECTION_TOLERANCE]
    tie_break_used = len(candidates) > 1
    chosen = candidates.sort_values(
        "model_name", key=lambda names: names.map(TIE_BREAK_ORDER), kind="stable"
    ).iloc[0]
    return BaselineSelection(
        model_name=str(chosen["model_name"]),
        selection_metric=selection_metric,
        tie_break_used=tie_break_used,
    )


def build_selected_baseline_record(
    metrics: pd.DataFrame,
    selection: BaselineSelection,
    *,
    selected_at: str,
    feature_version: str | None,
    feature_sha256: str | None,
) -> dict[str, object]:
    """Create a compact, JSON-safe record of the evidence-backed decision."""
    selected_overall = metrics.loc[
        metrics["scope"].eq("OVERALL") & metrics["model_name"].eq(selection.model_name)
    ].iloc[0]
    selected_folds = metrics.loc[
        metrics["scope"].eq("FOLD") & metrics["model_name"].eq(selection.model_name)
    ].sort_values("fold_id", kind="stable")
    fold_metrics = [
        {
            "fold_id": str(row.fold_id),
            "validation_month": str(row.validation_month),
            "mae": float(row.mae),
            "wape": None if pd.isna(row.wape) else float(row.wape),
            "wape_available": bool(row.wape_available),
            "number_of_rows": int(row.number_of_rows),
            "actual_units_sum": float(row.actual_units_sum),
        }
        for row in selected_folds.itertuples(index=False)
    ]
    return {
        "model_name": selection.model_name,
        "version": "1.0",
        "selection_metric": selection.selection_metric,
        "selection_scope": "POOLED_CHRONOLOGICAL_VALIDATION",
        "selection_rule": "lowest pooled validation WAPE; MAE used as tie-breaker",
        "tie_break_used": selection.tie_break_used,
        "validation_periods": list(MODEL_SELECTION_MONTHS),
        "final_test_month": FINAL_TEST_MONTH,
        "final_test_used_for_selection": False,
        "feature_version": feature_version,
        "feature_sha256": feature_sha256,
        "selected_at": selected_at,
        "source_predictions": "data/artifacts/forecasting/baseline_predictions.parquet",
        "source_metrics": "data/artifacts/forecasting/baseline_metrics.csv",
        "validation_metrics": {
            "mae": float(selected_overall.mae),
            "wape": None if pd.isna(selected_overall.wape) else float(selected_overall.wape),
            "wape_available": bool(selected_overall.wape_available),
            "actual_units_sum": float(selected_overall.actual_units_sum),
            "number_of_rows": int(selected_overall.number_of_rows),
        },
        "fold_metrics": fold_metrics,
    }


def save_selected_baseline_record(record: dict[str, object], path: Path) -> None:
    """Persist a deterministic, human-readable baseline-selection record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
