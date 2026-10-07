"""Read-only access to persisted Day-06 forecast evidence for the Django page."""

import json
from pathlib import Path

import pandas as pd
from django.conf import settings

from catalog.models import Product


DISPLAY_ROW_LIMIT = 100
REQUIRED_FORECAST_COLUMNS = {
    "branch_code", "canonical_product_id", "planning_month", "family_id", "actual_units",
    "raw_prediction", "forecast_value", "selected_model", "evidence_class",
    "evidence_reason_code", "evidence_warning", "forecast_status",
}
REQUIRED_MANIFEST_FIELDS = {"selected_model", "overall_mae", "overall_wape", "mode"}
EVIDENCE_BADGE_CLASSES = {
    "ADEQUATE": "bg-success",
    "LIMITED": "bg-warning text-dark",
    "INSUFFICIENT": "bg-danger",
}
EVIDENCE_REASON_LABELS = {
    "UNRESOLVED_MAPPING": "Unresolved product mapping",
    "NO_USABLE_HISTORY": "No usable historical observations",
    "CRITICAL_DQ_BLOCK": "Critical data-quality blocker",
    "SPARSE_HISTORY": "Sparse historical observations",
    "HIGH_INTERMITTENCY": "Highly intermittent history",
    "INVENTORY_EVIDENCE_LIMITED": "Inventory evidence limited",
    "FALLBACK_MODEL_USED": "Fallback model used",
}


class ForecastArtifactsUnavailable(RuntimeError):
    """Raised when persisted Day-06 research artifacts cannot be displayed."""


def forecast_artifact_paths() -> dict[str, Path]:
    """Return the persisted Day-06 artifact locations used by the Forecast page."""
    artifact_directory = settings.BASE_DIR / "data" / "artifacts" / "day06"
    return {
        "forecasts": artifact_directory / "forecast_with_confidence.parquet",
        "manifest": artifact_directory / "model_manifest.json",
    }


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ForecastArtifactsUnavailable(
            "Day-06 forecast artifacts are not available. Build the persisted Day-06 artifact package first."
        )


def _load_manifest(path: Path) -> dict[str, object]:
    _require_file(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ForecastArtifactsUnavailable("The Day-06 model manifest could not be read.") from error
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ForecastArtifactsUnavailable(f"The Day-06 model manifest is missing fields: {missing}.")
    if manifest["mode"] != "HISTORICAL_RESEARCH":
        raise ForecastArtifactsUnavailable("The forecast artifact is not marked as historical research evidence.")
    if not isinstance(manifest["selected_model"], str) or not manifest["selected_model"]:
        raise ForecastArtifactsUnavailable("The Day-06 manifest does not name a selected model.")
    return manifest


def _load_forecasts(path: Path) -> pd.DataFrame:
    _require_file(path)
    try:
        forecasts = pd.read_parquet(path)
    except Exception as error:  # pragma: no cover - backend-specific parsing errors
        raise ForecastArtifactsUnavailable("Day-06 forecast evidence could not be read.") from error
    missing = sorted(REQUIRED_FORECAST_COLUMNS - set(forecasts.columns))
    if missing:
        raise ForecastArtifactsUnavailable(f"Day-06 forecast evidence is missing columns: {missing}.")
    if forecasts["planning_month"].astype(str).eq("2026-03").any():
        raise ForecastArtifactsUnavailable("Forecast evidence unexpectedly includes the final test month.")
    return forecasts


def _display_text(value: object, default: str = "—") -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    return str(value)


def _display_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.2f}"


def _evidence_reason(row: pd.Series) -> str:
    """Present stored confidence reason codes without inventing UI-only rules."""
    codes = [code for code in _display_text(row["evidence_reason_code"], "").split("|") if code]
    labels = [EVIDENCE_REASON_LABELS.get(code, code.replace("_", " ").title()) for code in codes]
    if labels:
        return "; ".join(labels)
    return _display_text(row["evidence_warning"], "")


def _to_records(rows: pd.DataFrame) -> list[dict[str, object]]:
    """Format persisted values safely; withheld rows never expose raw predictions."""
    records = []
    for _, row in rows.iterrows():
        withheld = row["forecast_status"] == "WITHHELD"
        evidence_class = _display_text(row["evidence_class"])
        records.append(
            {
                "product": _display_text(row["canonical_product_id"]),
                "family": _display_text(row["family_id"]),
                "actual_units": _display_number(row["actual_units"]),
                "forecast": "WITHHELD" if withheld else _display_number(row["forecast_value"]),
                "model_name": "—" if withheld else _display_text(row["selected_model"]),
                "evidence_class": evidence_class,
                "evidence_badge_class": EVIDENCE_BADGE_CLASSES.get(evidence_class, "bg-secondary"),
                "evidence_reason": _evidence_reason(row),
                "absolute_error": (
                    "—"
                    if withheld or pd.isna(row["actual_units"]) or pd.isna(row["forecast_value"])
                    else _display_number(abs(float(row["actual_units"]) - float(row["forecast_value"])))
                ),
            }
        )
    return records


def _attach_product_details(records: list[dict[str, object]]) -> None:
    """Add persisted catalog display fields without changing forecast evidence."""

    product_ids = {record["product"] for record in records if record["product"] != "â€”"}
    details = {
        canonical_id: (product_code, product_desc)
        for canonical_id, product_code, product_desc in Product.objects.filter(
            canonical_product_id__in=product_ids
        ).values_list("canonical_product_id", "product_code", "product_desc")
    }
    for record in records:
        product_code, product_desc = details.get(record["product"], ("â€”", "â€”"))
        record["product_code"] = product_code
        record["product_description"] = product_desc


def get_forecast_page_data(branch: str | None = None, month: str | None = None) -> dict[str, object]:
    """Prepare persisted selected-model evidence for the existing research page."""
    paths = forecast_artifact_paths()
    manifest = _load_manifest(paths["manifest"])
    forecasts = _load_forecasts(paths["forecasts"])
    selected_model = str(manifest["selected_model"])
    if not forecasts["selected_model"].eq(selected_model).all():
        raise ForecastArtifactsUnavailable("Day-06 forecast rows do not match the selected model in the manifest.")

    branches = sorted(forecasts["branch_code"].dropna().astype(str).unique())
    months = sorted(forecasts["planning_month"].dropna().astype(str).unique())
    displayed = forecasts.copy()
    filter_message = None
    if branch:
        displayed = displayed.loc[displayed["branch_code"].eq(branch)]
        if branch not in branches:
            filter_message = f"No forecast evidence is available for branch '{branch}'."
    if month:
        displayed = displayed.loc[displayed["planning_month"].astype(str).eq(month)]
        if month not in months:
            filter_message = f"No forecast evidence is available for month '{month}'."

    displayed = displayed.sort_values(
        ["branch_code", "planning_month", "canonical_product_id"], kind="stable"
    )
    total_matching_rows = len(displayed)
    rows_truncated = total_matching_rows > DISPLAY_ROW_LIMIT
    displayed_rows = displayed.head(DISPLAY_ROW_LIMIT)

    records = _to_records(displayed_rows)
    _attach_product_details(records)
    return {
        "artifact_available": True,
        "artifact_message": None,
        "selected_model": selected_model,
        "overall_mae": _display_number(manifest["overall_mae"]),
        "overall_wape_display": f"{float(manifest['overall_wape']) * 100:.2f}%",
        "branches": branches,
        "months": months,
        "selected_branch": branch or "",
        "selected_month": month or "",
        "forecast_rows": records,
        "total_matching_rows": total_matching_rows,
        "rows_truncated": rows_truncated,
        "filter_message": filter_message,
        "evidence_counts": {
            evidence_class: int(displayed["evidence_class"].eq(evidence_class).sum())
            for evidence_class in ("ADEQUATE", "LIMITED", "INSUFFICIENT")
        },
    }


def get_forecast_evidence_for_context(*, branch: str, planning_month: str) -> dict[str, object]:
    """Read saved branch evidence for one planning context without generating forecasts.

    The frozen Day-06 artifact contains validation months. If its rows do not
    include the selected planning month, return the saved branch-level
    validation evidence while explicitly retaining that fact for the UI.
    """

    exact = get_forecast_page_data(branch=branch, month=planning_month)
    exact["context_forecast_available"] = bool(exact["total_matching_rows"])
    exact["artifact_validation_months"] = exact["months"]
    if exact["context_forecast_available"]:
        return exact

    branch_evidence = get_forecast_page_data(branch=branch)
    branch_evidence["context_forecast_available"] = False
    branch_evidence["artifact_validation_months"] = branch_evidence["months"]
    branch_evidence["filter_message"] = (
        f"No saved Day-06 forecast rows exist for planning month {planning_month}. "
        "The table shows saved historical validation evidence for the selected branch; "
        "it is not presented as an April forecast."
    )
    return branch_evidence
