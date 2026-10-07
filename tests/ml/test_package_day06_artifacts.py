import json

import pandas as pd
import pytest

from ml.evaluation.package_day06_artifacts import (
    Day06PackageError,
    MODEL_VERSION,
    build_day06_package,
    validate_day06_package,
)


def test_day06_package_reopens_with_frozen_folds_and_withheld_values(tmp_path) -> None:
    result = build_day06_package(tmp_path)

    assert result.comparison_rows == 6
    assert result.lightgbm_prediction_rows == 3532
    assert result.forecast_rows == 3532
    assert result.manifest["selected_model"] == "trailing_mean_3"
    assert result.manifest["mode"] == "HISTORICAL_RESEARCH"
    assert result.manifest["fold_count"] == 3

    forecasts = pd.read_parquet(tmp_path / "forecast_with_confidence.parquet")
    withheld = forecasts["forecast_status"].eq("WITHHELD")
    assert withheld.any()
    assert forecasts.loc[withheld, "forecast_value"].isna().all()
    assert validate_day06_package(tmp_path).evidence_counts == result.evidence_counts


def test_day06_package_rejects_a_different_existing_run(tmp_path) -> None:
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(
        json.dumps({"model_version": "different_version", "selected_model": "different_model"}),
        encoding="utf-8",
    )

    with pytest.raises(Day06PackageError, match="different run or selected model"):
        build_day06_package(tmp_path)

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["model_version"] != MODEL_VERSION
