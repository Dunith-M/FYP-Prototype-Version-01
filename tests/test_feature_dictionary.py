from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "feature_name", "meaning", "source_dataset", "source_field_or_logic", "data_type", "unit",
    "feature_group", "leakage_safe", "available_at_prediction_time", "model_role",
    "missing_value_meaning", "transformation", "limitations_notes", "included_in_model",
}


def test_feature_dictionary_matches_the_frozen_feature_schema_and_has_required_metadata():
    root = Path(__file__).resolve().parents[1]
    dictionary = pd.read_csv(root / "docs" / "feature_dictionary.csv")
    feature_month = pd.read_parquet(root / "data" / "processed" / "feature_month.parquet")
    assert REQUIRED_COLUMNS <= set(dictionary.columns)
    assert dictionary.feature_name.is_unique
    assert set(dictionary.feature_name) == set(feature_month.columns)
    assert not set(feature_month.columns) - set(dictionary.feature_name)


def test_dictionary_records_actual_leakage_and_target_semantics_without_invented_financial_features():
    root = Path(__file__).resolve().parents[1]
    dictionary = pd.read_csv(root / "docs" / "feature_dictionary.csv").set_index("feature_name")
    for feature in ("lag_1", "lag_2", "lag_3", "trailing_mean_3", "trailing_sum_3", "months_with_positive_sales", "intermittency"):
        assert feature in dictionary.index
        assert dictionary.loc[feature, "leakage_safe"] == "Yes"
    assert "SHIFT FIRST THEN ROLLING" in dictionary.loc["trailing_mean_3", "transformation"]
    for feature in ("last_observed_qty", "mean_observed_qty_30d", "inventory_observation_count", "availability_unknown_rate"):
        assert dictionary.loc[feature, "leakage_safe"] == "Yes"
    assert "not zero" in dictionary.loc["availability_unknown_rate", "limitations_notes"].lower()
    assert dictionary.loc["target_next_month", "model_role"] == "Target"
    assert dictionary.loc["target_next_month", "included_in_model"] == "No"
    assert "not true demand" in dictionary.loc["fulfilled_positive_units_proxy", "limitations_notes"].lower()
    assert not {"cost", "margin", "profit", "contribution", "roi"} & set(dictionary.index.str.lower())
