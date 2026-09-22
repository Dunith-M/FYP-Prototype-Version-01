"""Forecast evaluation metrics."""

from dataclasses import dataclass

import numpy as np


def mae(actual, prediction) -> float:
    """Return mean absolute error for equally sized, finite numeric inputs."""
    try:
        actual_values = np.asarray(actual, dtype=float)
        prediction_values = np.asarray(prediction, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("actual and prediction must contain numeric values.") from error

    if actual_values.ndim != 1 or prediction_values.ndim != 1:
        raise ValueError("actual and prediction must be one-dimensional observations.")
    if len(actual_values) != len(prediction_values):
        raise ValueError("actual and prediction must have the same number of observations.")
    if len(actual_values) == 0:
        raise ValueError("MAE requires at least one observation.")
    if not np.isfinite(actual_values).all() or not np.isfinite(prediction_values).all():
        raise ValueError("actual and prediction must contain only finite values.")

    return float(np.abs(actual_values - prediction_values).mean())


@dataclass(frozen=True)
class WapeDetails:
    """WAPE value and denominator coverage needed for transparent reporting."""

    wape: float
    absolute_error_total: float
    actual_volume: float
    denominator_available: bool


def wape_details(actual, prediction) -> WapeDetails:
    """Return WAPE and its denominator coverage for finite forecasting inputs."""
    try:
        actual_values = np.asarray(actual, dtype=float)
        prediction_values = np.asarray(prediction, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("actual and prediction must contain numeric values.") from error

    if actual_values.ndim != 1 or prediction_values.ndim != 1:
        raise ValueError("actual and prediction must be one-dimensional observations.")
    if len(actual_values) != len(prediction_values):
        raise ValueError("actual and prediction must have the same number of observations.")
    if len(actual_values) == 0:
        raise ValueError("WAPE requires at least one observation.")
    if not np.isfinite(actual_values).all() or not np.isfinite(prediction_values).all():
        raise ValueError("actual and prediction must contain only finite values.")
    if (actual_values < 0).any():
        raise ValueError("actual values must be non-negative for WAPE.")

    absolute_error_total = float(np.abs(actual_values - prediction_values).sum())
    actual_volume = float(actual_values.sum())
    denominator_available = actual_volume > 0
    value = absolute_error_total / actual_volume if denominator_available else float("nan")
    return WapeDetails(value, absolute_error_total, actual_volume, denominator_available)


def wape(actual, prediction) -> float:
    """Return WAPE as a ratio, or ``NaN`` when actual volume is zero."""
    return wape_details(actual, prediction).wape
