import pandas as pd

from ml.evaluation.run_lightgbm import MODEL_NAME, generate_lightgbm_predictions


def _feature_rows() -> pd.DataFrame:
    rows = []
    for target_month in ("2025-10", "2025-11", "2025-12", "2026-01", "2026-02"):
        for number, (branch, product) in enumerate((("BIL01", "CP_1"), ("BIN01", "CP_2")), start=1):
            rows.append(
                {
                    "branch_code": branch,
                    "canonical_product_id": product,
                    "year_month": target_month,
                    "target_year_month": target_month,
                    "target_next_month": float(number),
                    "lag_1": float(number),
                    "lag_2": float(number),
                    "lag_3": float(number),
                    "trailing_mean_3": float(number),
                    "trailing_sum_3": float(number * 3),
                    "months_with_positive_sales": number,
                    "intermittency": 0.0,
                    "last_observed_qty": float(number),
                    "mean_observed_qty_30d": float(number),
                    "inventory_observation_count": number,
                    "availability_unknown_rate": 0.0,
                    "valid_price": True,
                    "month": int(target_month[-2:]),
                    "quarter": 4 if target_month.endswith(("10", "11", "12")) else 1,
                    "holiday_count": 0,
                    "weekend_count": 8,
                    "family_id": f"F_{number}",
                    "family_desc": f"Family {number}",
                    "price_bucket": "Q1",
                    "branch_name": branch,
                    "target_available_flag": True,
                }
            )
    return pd.DataFrame(rows)


def test_one_pooled_lightgbm_model_is_trained_for_each_frozen_fold() -> None:
    result = generate_lightgbm_predictions(_feature_rows())

    assert len(result.fold_summary) == 3
    assert result.fold_summary["fold_id"].tolist() == ["fold_01", "fold_02", "fold_03"]
    assert result.fold_summary["training_status"].tolist() == ["PASS", "PASS", "PASS"]
    assert result.fold_summary["training_branch_count"].tolist() == [2, 2, 2]
    assert result.fold_summary["training_product_count"].tolist() == [2, 2, 2]

    assert len(result.predictions) == 6
    assert set(result.predictions["target_month"]) == {"2025-12", "2026-01", "2026-02"}
    assert set(result.predictions["model_name"]) == {MODEL_NAME}
    assert result.predictions["prediction_available"].all()
