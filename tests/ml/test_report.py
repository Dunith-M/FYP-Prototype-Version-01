import pandas as pd
import pytest

from ml.evaluation.report import BaselineReportError, build_baseline_evaluation_table, select_baseline


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": ["fold_01", "fold_02"],
            "validation_month": ["2025-12", "2026-01"],
            "target_month": ["2025-12", "2026-01"],
            "branch_code": ["A", "B"],
            "canonical_product_id": ["CP_A", "CP_B"],
            "family_id": ["F_A", "F_B"],
            "actual_units": [100.0, 10.0],
            "predicted_units": [90.0, 0.0],
            "model_name": ["zero_forecast", "zero_forecast"],
            "prediction_available": [True, True],
        }
    )


def test_overall_wape_uses_pooled_rows_not_an_average_of_subgroup_wapes() -> None:
    report = build_baseline_evaluation_table(_prediction_rows())
    overall = report.loc[report["scope"] == "OVERALL"].iloc[0]

    # Branch WAPEs are 0.1 and 1.0, whose simple mean would incorrectly be 0.55.
    assert overall.actual_units_sum == 110.0
    assert overall.absolute_error_sum == 20.0
    assert overall.wape == pytest.approx(20 / 110)
    assert overall.wape != pytest.approx(0.55)


def test_report_preserves_unevaluable_rows_without_silent_loss() -> None:
    rows = _prediction_rows()
    rows.loc[1, "prediction_available"] = False
    rows.loc[1, "predicted_units"] = None
    report = build_baseline_evaluation_table(rows)
    overall = report.loc[report["scope"] == "OVERALL"].iloc[0]

    assert overall.prediction_row_count == 2
    assert overall.number_of_rows == 1
    assert overall.unevaluable_row_count == 1


def _selection_metrics(wapes: dict[str, float], maes: dict[str, float]) -> pd.DataFrame:
    models = ("zero_forecast", "last_observation", "trailing_mean_3")
    rows = []
    for model in models:
        rows.append({"scope": "OVERALL", "fold_id": "ALL", "validation_month": "ALL", "model_name": model, "mae": maes[model], "wape": wapes[model], "wape_available": True})
        rows.append({"scope": "BRANCH", "fold_id": "ALL", "validation_month": "ALL", "model_name": model, "mae": maes[model], "wape": wapes[model], "wape_available": True})
        rows.append({"scope": "FAMILY", "fold_id": "ALL", "validation_month": "ALL", "model_name": model, "mae": maes[model], "wape": wapes[model], "wape_available": True})
        for fold_id, validation_month in (("fold_01", "2025-12"), ("fold_02", "2026-01"), ("fold_03", "2026-02")):
            rows.append({"scope": "FOLD", "fold_id": fold_id, "validation_month": validation_month, "model_name": model, "mae": maes[model], "wape": wapes[model], "wape_available": True})
    return pd.DataFrame(rows)


def test_selection_uses_lowest_pooled_wape() -> None:
    selection = select_baseline(
        _selection_metrics(
            {"zero_forecast": 1.0, "last_observation": 0.7, "trailing_mean_3": 0.5},
            {"zero_forecast": 5.0, "last_observation": 3.0, "trailing_mean_3": 4.0},
        )
    )
    assert selection.model_name == "trailing_mean_3"
    assert selection.selection_metric == "WAPE"
    assert selection.tie_break_used is False


def test_selection_uses_mae_then_documented_order_for_wape_ties() -> None:
    selection = select_baseline(
        _selection_metrics(
            {"zero_forecast": 0.5, "last_observation": 0.5, "trailing_mean_3": 0.7},
            {"zero_forecast": 3.0, "last_observation": 2.0, "trailing_mean_3": 1.0},
        )
    )
    assert selection.model_name == "last_observation"
    assert selection.tie_break_used is False


def test_selection_is_deterministic_for_an_exact_wape_and_mae_tie() -> None:
    selection = select_baseline(
        _selection_metrics(
            {"zero_forecast": 0.5, "last_observation": 0.5, "trailing_mean_3": 0.7},
            {"zero_forecast": 2.0, "last_observation": 2.0, "trailing_mean_3": 1.0},
        )
    )
    assert selection.model_name == "zero_forecast"
    assert selection.tie_break_used is True


def test_selection_fails_without_all_model_evidence() -> None:
    metrics = _selection_metrics(
        {"zero_forecast": 1.0, "last_observation": 0.7, "trailing_mean_3": 0.5},
        {"zero_forecast": 5.0, "last_observation": 3.0, "trailing_mean_3": 4.0},
    )
    metrics = metrics.loc[metrics["model_name"] != "trailing_mean_3"]
    with pytest.raises(BaselineReportError, match="exactly one row for each baseline"):
        select_baseline(metrics)
