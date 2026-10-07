"""Record the evidence-based research-model selection without changing forecasts."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from ml.evaluation.compare_models import (
    BY_BRANCH_PATH,
    BY_FAMILY_PATH,
    BY_FOLD_PATH,
    LIGHTGBM_MODEL_NAME,
    OVERALL_PATH,
)


FORECASTING_ARTIFACT_DIR = OVERALL_PATH.parent
SELECTION_JSON_PATH = FORECASTING_ARTIFACT_DIR / "selected_forecast_model.json"
SELECTION_TEXT_PATH = FORECASTING_ARTIFACT_DIR / "selected_forecast_model.txt"


class ModelSelectionRecordError(ValueError):
    """Raised when comparison evidence cannot support a selection record."""


@dataclass(frozen=True)
class SelectionRecord:
    """Machine-readable selection record and its human-readable explanation."""

    record: dict[str, object]
    text: str


def _winner_counts(table: pd.DataFrame) -> dict[str, int]:
    """Count visible comparison outcomes without suppressing losing segments."""
    winners = table["winner_on_comparison_metric"]
    return {
        "lightgbm_better": int(winners.eq(LIGHTGBM_MODEL_NAME).sum()),
        "baseline_better": int(winners.eq("BASELINE").sum()),
        "ties": int(winners.eq("TIE").sum()),
    }


def build_selection_record(
    overall: pd.DataFrame,
    by_fold: pd.DataFrame,
    by_branch: pd.DataFrame,
    by_family: pd.DataFrame,
) -> SelectionRecord:
    """Record a conservative choice from the completed chronological evidence.

    No formal promotion threshold exists in the project.  Therefore a
    challenger is selected only when it wins the pooled WAPE and has no worse
    fold or branch under the existing comparison metric.  Family outcomes are
    recorded for transparency but do not create a hidden numeric threshold.
    """
    if len(overall) != 1:
        raise ModelSelectionRecordError("Comparison evidence must contain exactly one overall row.")
    required = {
        "baseline_model_name", "baseline_mae", "baseline_wape", "lightgbm_mae", "lightgbm_wape",
        "wape_relative_improvement", "winner_on_comparison_metric", "observation_keys_identical",
    }
    missing = sorted(required - set(overall.columns))
    if missing:
        raise ModelSelectionRecordError(f"Overall comparison is missing columns: {missing}.")
    if not bool(overall.iloc[0]["observation_keys_identical"]):
        raise ModelSelectionRecordError("Comparison did not verify identical validation observations.")
    for label, table in (("fold", by_fold), ("branch", by_branch), ("family", by_family)):
        if "winner_on_comparison_metric" not in table:
            raise ModelSelectionRecordError(f"{label} comparison is missing winner evidence.")

    row = overall.iloc[0]
    fold_counts = _winner_counts(by_fold)
    branch_counts = _winner_counts(by_branch)
    family_counts = _winner_counts(by_family)
    zero_volume_family_count = int((~by_family["baseline_wape_available"].astype(bool)).sum())
    lightgbm_wins_overall = row["winner_on_comparison_metric"] == LIGHTGBM_MODEL_NAME
    stable_across_folds_and_branches = (
        fold_counts["baseline_better"] == 0 and branch_counts["baseline_better"] == 0
    )

    baseline_model = str(row["baseline_model_name"])
    if lightgbm_wins_overall and stable_across_folds_and_branches:
        selected_model = LIGHTGBM_MODEL_NAME
        status = "CHALLENGER_SELECTED"
        reason = (
            "LightGBM has lower pooled chronological-validation WAPE and no worse "
            "fold or branch under the shared comparison metric."
        )
    else:
        selected_model = baseline_model
        status = "BASELINE_RETAINED"
        reason = (
            "LightGBM's pooled WAPE improvement is not stable across the completed "
            "chronological folds and branches; the additional ML complexity is not "
            "justified by the currently available historical evidence."
        )

    record = {
        "schema_version": "selected_forecast_model_v1",
        "selection_scope": "chronological_out_of_fold_validation",
        "selected_model": selected_model,
        "status": status,
        "baseline_model": baseline_model,
        "lightgbm_model": LIGHTGBM_MODEL_NAME,
        "baseline_mae": float(row["baseline_mae"]),
        "baseline_wape": float(row["baseline_wape"]),
        "lightgbm_mae": float(row["lightgbm_mae"]),
        "lightgbm_wape": float(row["lightgbm_wape"]),
        "wape_relative_improvement": float(row["wape_relative_improvement"]),
        "wape_absolute_improvement_percentage_points": float(
            row["wape_absolute_improvement_percentage_points"]
        ),
        "selection_reason": reason,
        "promotion_rule": "existing project comparison logic; no formal numeric promotion rule implemented",
        "formal_promotion_rule_implemented": False,
        "fold_outcomes": fold_counts,
        "branch_outcomes": branch_counts,
        "family_outcomes": family_counts,
        "zero_volume_family_count": zero_volume_family_count,
        "observation_keys_identical": True,
        "source_artifacts": {
            "overall": str(OVERALL_PATH.relative_to(FORECASTING_ARTIFACT_DIR.parents[2])),
            "by_fold": str(BY_FOLD_PATH.relative_to(FORECASTING_ARTIFACT_DIR.parents[2])),
            "by_branch": str(BY_BRANCH_PATH.relative_to(FORECASTING_ARTIFACT_DIR.parents[2])),
            "by_family": str(BY_FAMILY_PATH.relative_to(FORECASTING_ARTIFACT_DIR.parents[2])),
        },
    }
    text = (
        "SELECTED MODEL\n"
        "==============\n\n"
        f"Model: {selected_model}\n\n"
        "Reason:\n"
        f"{reason}\n\n"
        f"Status: {status}\n"
    )
    return SelectionRecord(record=record, text=text)


def save_selection_record(
    selection: SelectionRecord,
    json_path: Path = SELECTION_JSON_PATH,
    text_path: Path = SELECTION_TEXT_PATH,
) -> None:
    """Persist separate selection records without modifying earlier evidence."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(selection.record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_path.write_text(selection.text, encoding="utf-8")


def run_selection_record() -> SelectionRecord:
    """Read saved comparison artifacts and write the selected-model record."""
    selection = build_selection_record(
        pd.read_csv(OVERALL_PATH),
        pd.read_csv(BY_FOLD_PATH),
        pd.read_csv(BY_BRANCH_PATH),
        pd.read_csv(BY_FAMILY_PATH),
    )
    save_selection_record(selection)
    return selection


def main() -> None:
    """Record selection evidence only; no model is retrained or integrated."""
    selection = run_selection_record()
    print(f"selected_model={selection.record['selected_model']}")
    print(f"status={selection.record['status']}")
    print(f"json={SELECTION_JSON_PATH}")
    print(f"text={SELECTION_TEXT_PATH}")


if __name__ == "__main__":
    main()
