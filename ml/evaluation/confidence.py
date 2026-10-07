"""Evidence-quality classification and abstention for research forecast rows."""

from dataclasses import asdict, dataclass
from typing import Mapping

import pandas as pd


ADEQUATE = "ADEQUATE"
LIMITED = "LIMITED"
INSUFFICIENT = "INSUFFICIENT"
AVAILABLE = "AVAILABLE"
AVAILABLE_WITH_WARNING = "AVAILABLE_WITH_WARNING"
WITHHELD = "WITHHELD"

CONFIDENCE_RULE_VERSION = "confidence_rules_v1"


@dataclass(frozen=True)
class ConfidenceConfig:
    """Explicit research assumptions for evidence classification.

    RESEARCH PROTOTYPE THRESHOLD
    NOT SINGER-APPROVED POLICY

    These values classify the quality of available evidence only.  They do not
    alter source data, features, targets, folds, or model predictions.
    """

    config_version: str = "confidence_research_config_v1"
    mode: str = "RESEARCH_PROTOTYPE"
    # Three positive-history months aligns with the existing trailing_mean_3
    # baseline's documented insufficient-history fallback condition.
    min_positive_history_months: int = 3
    high_intermittency_threshold: float = 0.5
    max_unknown_inventory_rate: float = 0.5
    min_inventory_observation_count: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return configuration metadata suitable for an auditable run record."""
        return asdict(self)


DEFAULT_CONFIDENCE_CONFIG = ConfidenceConfig()


@dataclass(frozen=True)
class EvidenceAssessment:
    """One deterministic evidence class with auditable reason codes."""

    evidence_class: str
    reason_codes: tuple[str, ...]
    warning: str | None
    forecast_status: str


def _is_true(row: Mapping[str, object], column: str) -> bool:
    value = row.get(column, False)
    return bool(value) if not pd.isna(value) else False


def _number(row: Mapping[str, object], column: str) -> float | None:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_blank(row: Mapping[str, object], column: str) -> bool:
    value = row.get(column)
    return value is None or pd.isna(value) or str(value).strip() == ""


def classify_evidence(
    row: Mapping[str, object], config: ConfidenceConfig = DEFAULT_CONFIDENCE_CONFIG
) -> EvidenceAssessment:
    """Classify evidence with INSUFFICIENT taking precedence over LIMITED.

    The inputs are existing Day-04/Day-05 field names.  Missing inventory
    evidence remains visible as a LIMITED warning; it is never converted to
    zero inventory.  A withheld forecast retains its raw prediction only for
    audit in the caller's returned frame.
    """
    blocking_codes: list[str] = []
    mapping_status = str(row.get("mapping_status", row.get("confidence_class", ""))).strip().upper()
    if _is_true(row, "unresolved_product_flag") or _is_blank(row, "canonical_product_id") or mapping_status == "UNRESOLVED":
        blocking_codes.append("UNRESOLVED_MAPPING")
    if _number(row, "lag_1") is None:
        blocking_codes.append("NO_USABLE_HISTORY")
    if _is_true(row, "reconciliation_difference_flag"):
        blocking_codes.append("CRITICAL_DQ_BLOCK")

    if blocking_codes:
        return EvidenceAssessment(
            evidence_class=INSUFFICIENT,
            reason_codes=tuple(blocking_codes),
            warning="Forecast withheld because required supporting evidence is insufficient.",
            forecast_status=WITHHELD,
        )

    limited_codes: list[str] = []
    positive_history = _number(row, "months_with_positive_sales")
    if positive_history is None or positive_history < config.min_positive_history_months:
        limited_codes.append("SPARSE_HISTORY")
    intermittency = _number(row, "intermittency")
    if intermittency is not None and intermittency >= config.high_intermittency_threshold:
        limited_codes.append("HIGH_INTERMITTENCY")
    inventory_count = _number(row, "inventory_observation_count")
    unknown_inventory_rate = _number(row, "availability_unknown_rate")
    if (
        inventory_count is None
        or inventory_count < config.min_inventory_observation_count
        or (unknown_inventory_rate is not None and unknown_inventory_rate > config.max_unknown_inventory_rate)
    ):
        limited_codes.append("INVENTORY_EVIDENCE_LIMITED")
    if _is_true(row, "fallback_used"):
        limited_codes.append("FALLBACK_MODEL_USED")

    if limited_codes:
        return EvidenceAssessment(
            evidence_class=LIMITED,
            reason_codes=tuple(limited_codes),
            warning="Forecast is available with limited supporting evidence.",
            forecast_status=AVAILABLE_WITH_WARNING,
        )
    return EvidenceAssessment(
        evidence_class=ADEQUATE,
        reason_codes=(),
        warning=None,
        forecast_status=AVAILABLE,
    )


def apply_confidence_to_forecasts(
    forecasts: pd.DataFrame,
    feature_month: pd.DataFrame,
    *,
    prediction_column: str = "predicted_units",
    config: ConfidenceConfig = DEFAULT_CONFIDENCE_CONFIG,
) -> pd.DataFrame:
    """Attach confidence fields to forecast evidence without mutating inputs.

    Forecast rows join to the existing feature row by its auditable
    ``target_year_month``.  ``raw_prediction`` is retained for audit; the
    user-facing ``forecast_value`` is null whenever evidence is INSUFFICIENT.
    """
    forecast_required = {"branch_code", "canonical_product_id", "target_month", prediction_column}
    feature_required = {"branch_code", "canonical_product_id", "target_year_month"}
    missing_forecast = sorted(forecast_required - set(forecasts.columns))
    missing_feature = sorted(feature_required - set(feature_month.columns))
    if missing_forecast:
        raise ValueError(f"Forecast rows are missing required columns: {missing_forecast}.")
    if missing_feature:
        raise ValueError(f"Feature rows are missing required columns: {missing_feature}.")

    evidence_columns = [
        "branch_code", "canonical_product_id", "target_year_month", "lag_1",
        "months_with_positive_sales", "intermittency", "inventory_observation_count",
        "availability_unknown_rate", "unresolved_product_flag", "reconciliation_difference_flag",
        "mapping_status", "confidence_class",
    ]
    available_evidence_columns = [column for column in evidence_columns if column in feature_month.columns]
    evidence = feature_month.loc[:, available_evidence_columns].copy().rename(
        columns={"target_year_month": "target_month"}
    )
    # Day-04 deliberately keeps rows without a consecutive next-month target.
    # They cannot correspond to a forecast target month and must not take part
    # in this lookup; unmatched forecast rows remain INSUFFICIENT below.
    evidence = evidence.loc[evidence["target_month"].notna()].copy()
    key = ["branch_code", "canonical_product_id", "target_month"]
    if evidence.duplicated(key).any():
        raise ValueError("Feature evidence has duplicate branch/product/target-month keys.")

    output = forecasts.copy(deep=True).merge(evidence, on=key, how="left", validate="many_to_one")
    assessments = output.apply(lambda row: classify_evidence(row, config), axis=1)
    output["evidence_class"] = [assessment.evidence_class for assessment in assessments]
    output["evidence_reason_code"] = ["|".join(assessment.reason_codes) for assessment in assessments]
    output["evidence_warning"] = [assessment.warning for assessment in assessments]
    output["forecast_status"] = [assessment.forecast_status for assessment in assessments]
    output["raw_prediction"] = output[prediction_column]
    output["forecast_value"] = output[prediction_column].where(output["forecast_status"] != WITHHELD)
    output["planning_month"] = output["target_month"]
    output["confidence_rule_version"] = CONFIDENCE_RULE_VERSION
    output["confidence_config_version"] = config.config_version
    output["confidence_config_mode"] = config.mode
    if "selected_model" not in output:
        output["selected_model"] = output["model_name"] if "model_name" in output else None
    return output
