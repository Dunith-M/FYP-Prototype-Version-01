import pandas as pd
import pytest
from unittest.mock import patch

from ml.evaluation.folds import (
    FINAL_TEST_MONTH,
    FOLDS_ARTIFACT_PATH,
    FoldDataValidationError,
    MODEL_SELECTION_MONTHS,
    VALIDATION_FOLDS,
    get_validation_fold,
    load_feature_month,
    normalize_target_month,
    split_fold,
    summarize_folds,
    write_folds_csv,
)


def _feature_rows() -> pd.DataFrame:
    """Small Day-04-shaped data with one intentionally unavailable target."""
    return pd.DataFrame(
        {
            "target_year_month": ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", None],
            "target_available_flag": [True, True, True, True, True, False],
            "row_label": ["nov", "dec", "jan", "feb", "mar", "unavailable"],
        }
    )


def test_exactly_three_model_selection_folds_are_frozen() -> None:
    assert len(VALIDATION_FOLDS) == 3
    assert [fold.fold_id for fold in VALIDATION_FOLDS] == ["fold_01", "fold_02", "fold_03"]


def test_validation_months_are_the_frozen_selection_months() -> None:
    assert MODEL_SELECTION_MONTHS == ("2025-12", "2026-01", "2026-02")


def test_march_is_a_separate_final_untouched_test_month() -> None:
    assert FINAL_TEST_MONTH == "2026-03"
    assert FINAL_TEST_MONTH not in MODEL_SELECTION_MONTHS


def test_every_training_cutoff_is_strictly_before_its_validation_month() -> None:
    for fold in VALIDATION_FOLDS:
        assert fold.training_cutoff_month < fold.validation_month


def test_folds_are_in_chronological_order() -> None:
    validation_months = [fold.validation_month for fold in VALIDATION_FOLDS]
    assert validation_months == sorted(validation_months)


@pytest.mark.parametrize("fold", VALIDATION_FOLDS)
def test_split_uses_target_month_and_excludes_future_rows(fold) -> None:
    train_df, validation_df = split_fold(_feature_rows(), fold)
    train_months = normalize_target_month(train_df["target_year_month"])
    validation_months = normalize_target_month(validation_df["target_year_month"])
    validation_month = pd.Period(fold.validation_month, freq="M")

    assert (train_months < validation_month).all()
    assert (validation_months == validation_month).all()
    assert not train_df["row_label"].isin(["mar", "unavailable"]).any()
    assert not validation_df["row_label"].isin(["mar", "unavailable"]).any()


@pytest.mark.parametrize("fold", VALIDATION_FOLDS)
def test_maximum_training_target_month_is_strictly_before_validation(fold) -> None:
    train_df, validation_df = split_fold(_feature_rows(), fold)
    train_months = normalize_target_month(train_df["target_year_month"])
    validation_month = pd.Period(fold.validation_month, freq="M")

    assert train_months.max() < validation_month
    assert (train_months < validation_month).all()
    assert train_df.index.intersection(validation_df.index).empty


def test_first_fold_never_leaks_later_target_months_into_training() -> None:
    train_df, validation_df = split_fold(_feature_rows(), "fold_01")

    assert train_df["row_label"].tolist() == ["nov"]
    assert validation_df["row_label"].tolist() == ["dec"]


def test_fold_01_excludes_all_deliberate_future_target_rows() -> None:
    feature_rows = pd.DataFrame(
        {
            "target_year_month": ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03"],
            "target_available_flag": [True, True, True, True, True, True],
        }
    )
    train_df, validation_df = split_fold(feature_rows, "fold_01")

    assert train_df["target_year_month"].tolist() == ["2025-10", "2025-11"]
    assert validation_df["target_year_month"].tolist() == ["2025-12"]
    selected_rows = pd.concat([train_df, validation_df])
    assert not selected_rows["target_year_month"].isin(["2026-01", "2026-02", "2026-03"]).any()


def test_invalid_fold_id_fails_clearly() -> None:
    with pytest.raises(FoldDataValidationError, match="Unknown validation fold_id"):
        get_validation_fold("fold_04")


def test_missing_target_month_column_fails_clearly() -> None:
    with pytest.raises(FoldDataValidationError, match="missing required columns"):
        split_fold(_feature_rows().drop(columns="target_year_month"), "fold_01")


def test_invalid_eligible_target_month_fails_clearly() -> None:
    feature_rows = _feature_rows()
    feature_rows.loc[0, "target_year_month"] = "not-a-month"

    with pytest.raises(FoldDataValidationError, match="cannot be parsed"):
        split_fold(feature_rows, "fold_01")


def test_manifest_contains_only_the_three_model_selection_folds() -> None:
    with patch.object(pd.DataFrame, "to_csv") as to_csv:
        summary = write_folds_csv(_feature_rows())

    assert summary.to_dict("records") == [
        {"fold_id": "fold_01", "train_end": "2025-11", "validation_month": "2025-12", "train_row_count": 1, "validation_row_count": 1},
        {"fold_id": "fold_02", "train_end": "2025-12", "validation_month": "2026-01", "train_row_count": 2, "validation_row_count": 1},
        {"fold_id": "fold_03", "train_end": "2026-01", "validation_month": "2026-02", "train_row_count": 3, "validation_row_count": 1},
    ]
    to_csv.assert_called_once_with(FOLDS_ARTIFACT_PATH, index=False)
    assert "2026-03" not in summary["validation_month"].tolist()


def test_final_test_month_occurs_after_all_validation_months() -> None:
    assert all(fold.validation_month < FINAL_TEST_MONTH for fold in VALIDATION_FOLDS)


def test_fold_summary_matches_the_frozen_fold_order() -> None:
    assert summarize_folds(_feature_rows())["fold_id"].tolist() == ["fold_01", "fold_02", "fold_03"]


def test_real_feature_artifact_has_leakage_safe_chronological_splits() -> None:
    feature_df = load_feature_month()

    for fold in VALIDATION_FOLDS:
        train_df, validation_df = split_fold(feature_df, fold)
        train_months = normalize_target_month(train_df["target_year_month"])
        validation_months = normalize_target_month(validation_df["target_year_month"])
        validation_month = pd.Period(fold.validation_month, freq="M")

        assert train_months.max() < validation_month
        assert (train_months < validation_month).all()
        assert (validation_months == validation_month).all()
        assert not (train_months == pd.Period(FINAL_TEST_MONTH, freq="M")).any()
        assert not (validation_months == pd.Period(FINAL_TEST_MONTH, freq="M")).any()
