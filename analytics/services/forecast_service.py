"""Read-only access to persisted Day-05 forecasting evidence."""

import json
from pathlib import Path

import pandas as pd
from django.conf import settings


DISPLAY_ROW_LIMIT = 100
REQUIRED_PREDICTION_COLUMNS = {
    "branch_code",
    "canonical_product_id",
    "target_month",
    "family_id",
    "actual_units",
    "model_name",
    "predicted_units",
    "fallback_used",
    "fallback_reason",
    "absolute_error",
}
REQUIRED_METRIC_COLUMNS = {"scope", "model_name", "mae", "wape", "wape_available"}


class ForecastArtifactsUnavailable(RuntimeError):
    """Raised when persisted Day-05 research artifacts cannot be displayed."""


def forecast_artifact_paths() -> dict[str, Path]:
    """Return portable artifact locations rooted at Django's project directory."""
    artifact_directory = settings.BASE_DIR / "data" / "artifacts" / "forecasting"
    return {
        "predictions": artifact_directory / "baseline_predictions.parquet",
        "metrics": artifact_directory / "baseline_metrics.csv",
        "selection": artifact_directory / "selected_baseline.json",
    }


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ForecastArtifactsUnavailable(
            "Forecast artifacts are not available. Run the Day-05 forecasting pipeline first."
        )


def _load_selected_model(path: Path) -> str:
    _require_file(path)
    try:
        selected_model = json.loads(path.read_text(encoding="utf-8"))["model_name"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ForecastArtifactsUnavailable("The selected baseline record is invalid or incomplete.") from error
    if not isinstance(selected_model, str) or not selected_model:
        raise ForecastArtifactsUnavailable("The selected baseline record does not name a model.")
    return selected_model


def _load_predictions(path: Path) -> pd.DataFrame:
    _require_file(path)
    try:
        predictions = pd.read_parquet(path)
    except Exception as error:  # pragma: no cover - backend-specific parsing errors
        raise ForecastArtifactsUnavailable("Forecast prediction evidence could not be read.") from error
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ForecastArtifactsUnavailable(f"Forecast prediction evidence is missing columns: {missing}.")
    if predictions["target_month"].astype(str).eq("2026-03").any():
        raise ForecastArtifactsUnavailable("Forecast evidence unexpectedly includes the final test month.")
    return predictions


def _load_metrics(path: Path) -> pd.DataFrame:
    _require_file(path)
    try:
        metrics = pd.read_csv(path)
    except Exception as error:  # pragma: no cover - backend-specific parsing errors
        raise ForecastArtifactsUnavailable("Forecast metric evidence could not be read.") from error
    missing = sorted(REQUIRED_METRIC_COLUMNS - set(metrics.columns))
    if missing:
        raise ForecastArtifactsUnavailable(f"Forecast metric evidence is missing columns: {missing}.")
    return metrics


def _fallback_display(fallback_used: bool, fallback_reason: object) -> str:
    if not fallback_used:
        return "No"
    labels = {
        "INSUFFICIENT_3_MONTH_HISTORY": "Insufficient 3-month history",
        "NO_PRIOR_OBSERVATION": "No prior observation",
    }
    return f"Yes — {labels.get(str(fallback_reason), str(fallback_reason))}"


def _to_records(rows: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for row in rows.itertuples(index=False):
        records.append(
            {
                "product": row.canonical_product_id,
                "family": row.family_id,
                "predicted_units": row.predicted_units,
                "actual_units": row.actual_units,
                "model_name": row.model_name,
                "absolute_error": row.absolute_error,
                "fallback": _fallback_display(bool(row.fallback_used), row.fallback_reason),
            }
        )
    return records


def get_forecast_page_data(branch: str | None = None, month: str | None = None) -> dict[str, object]:
    """Prepare selected-baseline evidence for the historical research page."""
    paths = forecast_artifact_paths()
    selected_model = _load_selected_model(paths["selection"])
    predictions = _load_predictions(paths["predictions"])
    metrics = _load_metrics(paths["metrics"])

    branches = sorted(predictions["branch_code"].dropna().astype(str).unique())
    months = sorted(predictions["target_month"].dropna().astype(str).unique())
    selected_rows = predictions.loc[predictions["model_name"].eq(selected_model)].copy()
    if selected_rows.empty:
        raise ForecastArtifactsUnavailable("Selected baseline evidence is not available in the prediction artifact.")

    filter_message = None
    if branch:
        selected_rows = selected_rows.loc[selected_rows["branch_code"].eq(branch)]
        if branch not in branches:
            filter_message = f"No forecast evidence is available for branch '{branch}'."
    if month:
        selected_rows = selected_rows.loc[selected_rows["target_month"].astype(str).eq(month)]
        if month not in months:
            filter_message = f"No forecast evidence is available for month '{month}'."

    selected_rows = selected_rows.sort_values(
        ["branch_code", "target_month", "canonical_product_id"], kind="stable"
    )
    total_matching_rows = len(selected_rows)
    rows_truncated = total_matching_rows > DISPLAY_ROW_LIMIT
    displayed_rows = selected_rows.head(DISPLAY_ROW_LIMIT)

    overall = metrics.loc[
        metrics["scope"].eq("OVERALL") & metrics["model_name"].eq(selected_model)
    ]
    if len(overall) != 1:
        raise ForecastArtifactsUnavailable("Selected baseline overall metrics are unavailable or ambiguous.")
    overall_metric = overall.iloc[0]
    wape_available = bool(overall_metric["wape_available"])

    return {
        "artifact_available": True,
        "artifact_message": None,
        "selected_model": selected_model,
        "overall_mae": overall_metric["mae"],
        "overall_wape": overall_metric["wape"],
        "overall_wape_available": wape_available,
        "overall_wape_display": f"{float(overall_metric['wape']) * 100:.2f}%" if wape_available else "Unavailable",
        "branches": branches,
        "months": months,
        "selected_branch": branch or "",
        "selected_month": month or "",
        "forecast_rows": _to_records(displayed_rows),
        "total_matching_rows": total_matching_rows,
        "rows_truncated": rows_truncated,
        "filter_message": filter_message,
    }
