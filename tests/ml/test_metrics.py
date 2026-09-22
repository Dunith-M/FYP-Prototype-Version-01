import math

import pytest

from ml.evaluation.metrics import mae, wape, wape_details


def test_mae_matches_the_provided_known_answer() -> None:
    assert mae([5, 10, 3], [4, 7, 5]) == 2.0


def test_mae_is_zero_for_a_perfect_forecast() -> None:
    assert mae([2, 4, 6], [2, 4, 6]) == 0.0


def test_mae_for_zero_forecast_is_correct() -> None:
    assert mae([2, 4, 6], [0, 0, 0]) == 4.0


def test_mae_preserves_a_decimal_result() -> None:
    result = mae([1, 2], [0, 0])

    assert result == 1.5
    assert isinstance(result, float)


@pytest.mark.parametrize("prediction", ([3, 7], [7, 3]))
def test_mae_treats_over_and_under_predictions_symmetrically(prediction) -> None:
    assert mae([5, 5], prediction) == 2.0


def test_mae_accepts_zero_values_as_valid_observations() -> None:
    assert mae([0, 0, 0], [0, 0, 0]) == 0.0


def test_mae_is_never_negative_for_valid_inputs() -> None:
    assert mae([0, 3, 10], [8, 0, 25]) >= 0.0


def test_mae_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same number"):
        mae([1, 2], [1])


def test_mae_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mae([], [])


@pytest.mark.parametrize("invalid", ([math.nan], [math.inf], [-math.inf], [None]))
def test_mae_rejects_missing_or_non_finite_values(invalid) -> None:
    with pytest.raises(ValueError, match="finite"):
        mae(invalid, [0])


def test_wape_matches_the_provided_known_answer_and_exposes_coverage() -> None:
    details = wape_details([5, 10, 3], [4, 7, 5])

    assert details.absolute_error_total == 6.0
    assert details.actual_volume == 18.0
    assert details.denominator_available is True
    assert details.wape == pytest.approx(6 / 18)
    assert wape([5, 10, 3], [4, 7, 5]) == pytest.approx(6 / 18)


def test_wape_is_zero_for_a_perfect_positive_volume_forecast() -> None:
    assert wape([2, 4, 6], [2, 4, 6]) == 0.0


def test_wape_for_zero_forecast_is_one() -> None:
    assert wape([2, 4, 6], [0, 0, 0]) == 1.0


def test_wape_handles_mixed_zero_and_positive_actuals() -> None:
    assert wape([0, 5, 0, 5], [1, 4, 2, 7]) == pytest.approx(0.6)


def test_wape_is_unavailable_for_zero_actual_volume_even_with_perfect_predictions() -> None:
    details = wape_details([0, 0, 0], [0, 0, 0])

    assert math.isnan(details.wape)
    assert details.actual_volume == 0.0
    assert details.denominator_available is False
    assert math.isnan(wape([0, 0, 0], [0, 0, 0]))


def test_wape_is_unavailable_for_zero_volume_with_nonzero_predictions() -> None:
    details = wape_details([0, 0], [5, 3])

    assert math.isnan(details.wape)
    assert details.absolute_error_total == 8.0
    assert details.denominator_available is False


def test_wape_handles_single_and_decimal_observations() -> None:
    assert wape([5], [3]) == pytest.approx(0.4)
    assert wape([2.5, 7.5], [2.0, 6.0]) == pytest.approx(0.2)


def test_wape_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same number"):
        wape([1, 2], [1])


def test_wape_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        wape([], [])


@pytest.mark.parametrize("invalid", ([math.nan], [math.inf], [-math.inf], [None]))
def test_wape_rejects_missing_or_non_finite_actual_values(invalid) -> None:
    with pytest.raises(ValueError, match="finite"):
        wape(invalid, [0])


@pytest.mark.parametrize("invalid", ([math.nan], [math.inf], [-math.inf], [None]))
def test_wape_rejects_missing_or_non_finite_prediction_values(invalid) -> None:
    with pytest.raises(ValueError, match="finite"):
        wape([1], invalid)


def test_wape_rejects_negative_actual_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        wape([-1, 2], [0, 0])


@pytest.mark.parametrize(
    ("actual", "prediction", "expected_mae", "expected_wape"),
    [
        ([1, 2], [0, 0], 1.5, 1.0),
        ([0, 5, 0, 5], [1, 4, 2, 7], 1.5, 0.6),
        ([0, 0], [5, 3], 4.0, None),
        ([5], [3], 2.0, 0.4),
        ([2.5, 7.5], [2.0, 6.0], 1.0, 0.2),
    ],
)
def test_mae_and_wape_combined_known_answer_cases(
    actual, prediction, expected_mae, expected_wape
) -> None:
    """Manual constants verify both production metrics without recreating formulas."""
    assert mae(actual, prediction) == pytest.approx(expected_mae)

    result = wape(actual, prediction)
    if expected_wape is None:
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected_wape)
