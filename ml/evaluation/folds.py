"""Chronological row splits for the frozen forecasting evaluation timeline.

Splits are based on ``target_year_month``: the month a feature row predicts.
The feature creation month (``year_month``) is deliberately not used for
evaluation assignment.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_MONTH_PATH = PROJECT_ROOT / "data" / "processed" / "feature_month.parquet"
FOLDS_ARTIFACT_PATH = PROJECT_ROOT / "data" / "artifacts" / "forecasting" / "folds.csv"
TARGET_MONTH_COLUMN = "target_year_month"
TARGET_AVAILABLE_COLUMN = "target_available_flag"


class FoldDataValidationError(ValueError):
    """Raised when the feature data cannot safely support a frozen fold."""


def _as_month(value: str) -> date:
    """Validate a ``YYYY-MM`` constant and return its first calendar day."""
    return date.fromisoformat(f"{value}-01")


@dataclass(frozen=True)
class ValidationFold:
    """One chronological fold; the cutoff is its last eligible train month."""

    fold_id: str
    training_cutoff_month: str
    validation_month: str

    def __post_init__(self) -> None:
        if _as_month(self.training_cutoff_month) >= _as_month(self.validation_month):
            raise ValueError("training_cutoff_month must be before validation_month")

    def allows_training_month(self, month: str) -> bool:
        """Return whether a target month is eligible training history for this fold."""
        return _as_month(month) <= _as_month(self.training_cutoff_month)


# These are the only folds permitted for baseline/model selection.
VALIDATION_FOLDS: tuple[ValidationFold, ...] = (
    ValidationFold("fold_01", "2025-11", "2025-12"),
    ValidationFold("fold_02", "2025-12", "2026-01"),
    ValidationFold("fold_03", "2026-01", "2026-02"),
)

MODEL_SELECTION_MONTHS: tuple[str, ...] = tuple(
    fold.validation_month for fold in VALIDATION_FOLDS
)

# Keep the untouched test period separate from model-selection folds.
FINAL_TEST_MONTH = "2026-03"


def normalize_target_month(values: pd.Series) -> pd.Series:
    """Return target months as monthly periods, failing on null or invalid values."""
    if values.isna().any():
        raise FoldDataValidationError(f"{TARGET_MONTH_COLUMN} contains null values for eligible target rows.")
    try:
        periods = pd.Series(pd.PeriodIndex(values, freq="M"), index=values.index)
    except (TypeError, ValueError) as error:
        raise FoldDataValidationError(
            f"{TARGET_MONTH_COLUMN} contains values that cannot be parsed as calendar months."
        ) from error
    if periods.isna().any():
        raise FoldDataValidationError(f"{TARGET_MONTH_COLUMN} contains invalid calendar months.")
    return periods


def _eligible_target_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Select only Day-04 rows with a valid supervised next-month target."""
    required = {TARGET_MONTH_COLUMN, TARGET_AVAILABLE_COLUMN}
    missing = sorted(required - set(feature_df.columns))
    if missing:
        raise FoldDataValidationError(f"Feature data is missing required columns: {missing}.")

    availability = feature_df[TARGET_AVAILABLE_COLUMN]
    if availability.isna().any() or not pd.api.types.is_bool_dtype(availability):
        raise FoldDataValidationError(f"{TARGET_AVAILABLE_COLUMN} must contain non-null boolean values.")

    # Day-04 intentionally leaves the target month null when no consecutive
    # next-month target exists. A non-null value in those rows is inconsistent.
    unavailable_with_month = (~availability) & feature_df[TARGET_MONTH_COLUMN].notna()
    if unavailable_with_month.any():
        raise FoldDataValidationError(
            f"{TARGET_MONTH_COLUMN} must be null when {TARGET_AVAILABLE_COLUMN} is False."
        )

    eligible = feature_df.loc[availability].copy()
    normalize_target_month(eligible[TARGET_MONTH_COLUMN])
    return eligible


def get_validation_fold(fold_id: str) -> ValidationFold:
    """Return a frozen model-selection fold by ID."""
    for fold in VALIDATION_FOLDS:
        if fold.fold_id == fold_id:
            return fold
    raise FoldDataValidationError(f"Unknown validation fold_id: {fold_id!r}.")


def load_feature_month(path: Path = FEATURE_MONTH_PATH) -> pd.DataFrame:
    """Load and validate the read-only Day-04 feature artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"feature_month.parquet was not found: {path}")
    feature_df = pd.read_parquet(path)
    _eligible_target_rows(feature_df)
    return feature_df


def split_fold(feature_df: pd.DataFrame, fold: ValidationFold | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return chronological training and validation rows for one frozen fold."""
    selected_fold = get_validation_fold(fold) if isinstance(fold, str) else fold
    if selected_fold not in VALIDATION_FOLDS:
        raise FoldDataValidationError(f"Unknown validation fold_id: {selected_fold.fold_id!r}.")

    eligible = _eligible_target_rows(feature_df)
    target_months = normalize_target_month(eligible[TARGET_MONTH_COLUMN])
    validation_month = pd.Period(selected_fold.validation_month, freq="M")
    train_df = eligible.loc[target_months < validation_month].copy()
    validation_df = eligible.loc[target_months == validation_month].copy()

    if train_df.empty:
        raise FoldDataValidationError(f"Training data is empty for {selected_fold.fold_id}.")
    if validation_df.empty:
        raise FoldDataValidationError(f"Validation data is empty for {selected_fold.fold_id}.")
    return train_df, validation_df


def summarize_folds(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic fold metadata and actual row counts."""
    rows = []
    for fold in VALIDATION_FOLDS:
        train_df, validation_df = split_fold(feature_df, fold)
        rows.append(
            {
                "fold_id": fold.fold_id,
                "train_end": fold.training_cutoff_month,
                "validation_month": fold.validation_month,
                "train_row_count": len(train_df),
                "validation_row_count": len(validation_df),
            }
        )
    return pd.DataFrame(rows)


def write_folds_csv(feature_df: pd.DataFrame, path: Path = FOLDS_ARTIFACT_PATH) -> pd.DataFrame:
    """Write the reproducible three-row model-selection fold manifest."""
    summary = summarize_folds(feature_df)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary
