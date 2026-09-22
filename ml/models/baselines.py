"""Simple deterministic forecasting baselines."""

import pandas as pd


LAG_1_COLUMN = "lag_1"
TRAILING_MEAN_3_COLUMN = "trailing_mean_3"
TRAILING_MEAN_3_MODEL_NAME = "trailing_mean_3"


def zero_forecast(rows: pd.DataFrame) -> pd.Series:
    """Return a zero-unit prediction for every requested forecasting row.

    The baseline deliberately uses only the input row count and index.  It
    does not inspect targets, train, fit, or use future information.
    """
    return pd.Series(0.0, index=rows.index, name="prediction", dtype="float64")


def last_observation_forecast(rows: pd.DataFrame) -> pd.Series:
    """Return the existing Day-04 previous-observation feature as prediction.

    Missing ``lag_1`` values are intentionally preserved: they represent
    insufficient prior observed history, not zero demand.
    """
    if LAG_1_COLUMN not in rows:
        raise ValueError(f"Forecasting rows must contain {LAG_1_COLUMN!r}.")
    if not pd.api.types.is_numeric_dtype(rows[LAG_1_COLUMN]) and not rows[LAG_1_COLUMN].isna().all():
        raise ValueError(f"{LAG_1_COLUMN} must be numeric.")
    return rows[LAG_1_COLUMN].copy().rename("prediction")


def trailing_mean_3_forecast(rows: pd.DataFrame) -> pd.Series:
    """Return the existing Day-04 trailing-three-month feature as prediction.

    Missing values are intentionally preserved because they represent fewer
    than three prior observed monthly records, not zero demand.
    """
    if TRAILING_MEAN_3_COLUMN not in rows:
        raise ValueError(f"Forecasting rows must contain {TRAILING_MEAN_3_COLUMN!r}.")
    if not pd.api.types.is_numeric_dtype(rows[TRAILING_MEAN_3_COLUMN]) and not rows[TRAILING_MEAN_3_COLUMN].isna().all():
        raise ValueError(f"{TRAILING_MEAN_3_COLUMN} must be numeric.")
    return rows[TRAILING_MEAN_3_COLUMN].copy().rename("prediction")


def trailing_mean_3_forecast_with_fallbacks(rows: pd.DataFrame) -> pd.DataFrame:
    """Return trailing-mean predictions with explicit sparse-history provenance.

    The existing Day-04 features determine every prediction.  A missing mean
    falls back to ``lag_1``; only when both features are missing is zero used.
    """
    trailing_mean = trailing_mean_3_forecast(rows)
    lag_1 = last_observation_forecast(rows)
    has_trailing_mean = trailing_mean.notna()
    has_lag_1 = lag_1.notna()

    prediction = trailing_mean.where(has_trailing_mean, lag_1.where(has_lag_1, 0.0))
    fallback_used = ~has_trailing_mean
    fallback_reason = pd.Series([None] * len(rows), index=rows.index, dtype="object")
    fallback_reason.loc[~has_trailing_mean & has_lag_1] = "INSUFFICIENT_3_MONTH_HISTORY"
    fallback_reason.loc[~has_trailing_mean & ~has_lag_1] = "NO_PRIOR_OBSERVATION"

    return pd.DataFrame(
        {
            "model_name": TRAILING_MEAN_3_MODEL_NAME,
            "prediction": prediction,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        },
        index=rows.index,
    )
