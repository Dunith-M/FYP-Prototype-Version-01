import pandas as pd

from ml.evaluation.compare_models import build_model_comparison


def _aligned_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": ["fold_01", "fold_02", "fold_03", "fold_01"],
            "validation_month": ["2025-12", "2026-01", "2026-02", "2025-12"],
            "target_month": ["2025-12", "2026-01", "2026-02", "2025-12"],
            "branch_code": ["BIL01", "BIL01", "BIN01", "BIN01"],
            "canonical_product_id": ["CP_1", "CP_1", "CP_2", "CP_2"],
            "actual_units": [2.0, 4.0, 6.0, 0.0],
            "baseline_predicted_units": [1.0, 2.0, 3.0, 1.0],
            "lightgbm_predicted_units": [2.0, 3.0, 5.0, 0.0],
        }
    )


def _baseline_predictions() -> pd.DataFrame:
    rows = _aligned_predictions().loc[:, [
        "fold_id", "validation_month", "target_month", "branch_code", "canonical_product_id", "actual_units"
    ]].copy()
    rows["family_id"] = ["PF_1", "PF_1", "PF_2", "PF_ZERO"]
    rows["model_name"] = "trailing_mean_3"
    return rows


def test_comparison_reports_pooled_and_segment_metrics_without_hiding_zero_volume_family() -> None:
    result = build_model_comparison(_aligned_predictions(), _baseline_predictions(), "trailing_mean_3")

    overall = result.overall.iloc[0]
    assert overall.baseline_mae == 1.75
    assert overall.lightgbm_mae == 0.5
    assert overall.winner_on_comparison_metric == "lightgbm_challenger"
    assert len(result.by_fold) == 3
    assert len(result.by_branch) == 2
    zero_family = result.by_family.loc[result.by_family["family_id"] == "PF_ZERO"].iloc[0]
    assert not bool(zero_family.baseline_wape_available)
    assert not bool(zero_family.lightgbm_wape_available)
    assert zero_family.comparison_metric == "MAE"
