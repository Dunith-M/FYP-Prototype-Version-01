import pandas as pd

from ml.evaluation.record_selected_model import build_selection_record


def _overall(winner: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "baseline_model_name": ["trailing_mean_3"],
            "baseline_mae": [2.0],
            "baseline_wape": [0.5],
            "lightgbm_mae": [1.9],
            "lightgbm_wape": [0.45],
            "wape_relative_improvement": [0.1],
            "wape_absolute_improvement_percentage_points": [5.0],
            "winner_on_comparison_metric": [winner],
            "observation_keys_identical": [True],
        }
    )


def _scope(winners: list[str], wape_available: list[bool] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "winner_on_comparison_metric": winners,
            "baseline_wape_available": wape_available or [True] * len(winners),
        }
    )


def test_mixed_fold_or_branch_results_retain_the_baseline() -> None:
    selection = build_selection_record(
        _overall("lightgbm_challenger"),
        _scope(["lightgbm_challenger", "BASELINE"]),
        _scope(["lightgbm_challenger"]),
        _scope(["lightgbm_challenger", "BASELINE"], [True, False]),
    )

    assert selection.record["selected_model"] == "trailing_mean_3"
    assert selection.record["status"] == "BASELINE_RETAINED"
    assert selection.record["zero_volume_family_count"] == 1
    assert "additional ML complexity" in selection.text


def test_consistent_challenger_wins_can_be_recorded_without_hard_coding_outcome() -> None:
    selection = build_selection_record(
        _overall("lightgbm_challenger"),
        _scope(["lightgbm_challenger", "lightgbm_challenger"]),
        _scope(["lightgbm_challenger", "lightgbm_challenger"]),
        _scope(["lightgbm_challenger", "BASELINE"]),
    )

    assert selection.record["selected_model"] == "lightgbm_challenger"
    assert selection.record["status"] == "CHALLENGER_SELECTED"
