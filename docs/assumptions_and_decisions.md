# MVP Assumptions and Decisions

## Prototype Mode
HISTORICAL / RESEARCH PROTOTYPE.

Historical Singer data is used for analytics.
The system must not claim that outputs are current operational advice.

## Forecast Target
Draft target:
next-month fulfilled positive sales units
at branch × canonical product × month.

## Negative / Zero Sales Events
Negative and zero values are preserved and flagged.
They will not be silently converted to zero or deleted.
Their exact business meaning remains unresolved.

## Optimization
Operational Singer constraints, L0 and q0 are not fully approved.

Synthetic planning values may therefore be used only for the
Synthetic Decision Demonstration.

Synthetic parameters must never be presented as Singer policy.

## Day 05 — Baseline selection

The simple baseline reference model was selected using pooled chronological
validation across December 2025, January 2026, and February 2026.

Selected baseline: `trailing_mean_3`

Primary selection metric: pooled validation WAPE. The selected model recorded
MAE `2.240091` and WAPE `0.588209` (ratio) across 3,532 validation rows with
13,451 actual units. It had the lowest valid pooled WAPE among the three simple
baselines; an MAE tie-break was not required.

March 2026 remains the untouched final test month and was not used for model
selection. The selected simple baseline is the reference for later challenger
model evaluation; this record does not select or train an advanced model.

Source evidence:

- `data/artifacts/forecasting/baseline_predictions.parquet`
- `data/artifacts/forecasting/baseline_metrics.csv`
- `data/artifacts/forecasting/selected_baseline.json`
