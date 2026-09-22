"""Build a standalone HTML walkthrough of the Day-05 Forecast page."""

from base64 import b64encode
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCREENSHOT = Path(r"C:\Users\DUNITH~1\AppData\Local\Temp\forecast_page_default.png")
FILTERED_SCREENSHOT = Path(r"C:\Users\DUNITH~1\AppData\Local\Temp\forecast_page_bil01_2026_02.png")
OUTPUT_PATH = PROJECT_ROOT / "docs" / "reports" / "Day_05_Historical_Forecast_Page_Report.html"


def image_data_uri(path: Path) -> str:
    """Return an embedded PNG data URI so the report is one downloadable file."""
    if not path.is_file():
        raise FileNotFoundError(f"Screenshot not found: {path}")
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    default_image = image_data_uri(DEFAULT_SCREENSHOT)
    filtered_image = image_data_uri(FILTERED_SCREENSHOT)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Day 05 Historical Forecast Page Report</title>
<style>
@page { size: A4; margin: 16mm; }
body { font-family: Arial, sans-serif; color: #1f2937; line-height: 1.5; max-width: 900px; margin: 28px auto; padding: 0 18px; }
h1 { font-size: 30px; margin-bottom: 6px; } h2 { font-size: 21px; margin-top: 30px; }
.subtitle { color: #5b6472; margin-top: 0; } .badge { display: inline-block; background: #ffc107; color: #212529; font-weight: 700; padding: 6px 10px; border-radius: 4px; }
.note { background: #fff8db; padding: 12px 15px; border-left: 4px solid #ffc107; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; } .metric { background: #eef6ff; border: 1px solid #cfe2ff; padding: 12px; border-radius: 6px; }
.metric strong { display:block; font-size: 19px; } table { width:100%; border-collapse: collapse; margin: 12px 0; } th { background: #173f5f; color: white; }
th, td { border: 1px solid #d9d9d9; padding: 8px; text-align: left; } img { width:100%; border: 1px solid #cbd5e1; border-radius: 5px; }
figure { margin: 18px 0 26px; } figcaption { color:#4b5563; font-size: 14px; margin-top: 7px; } ul { padding-left: 22px; } .small { font-size: 13px; color:#5b6472; }
@media print { body { margin: 0; max-width: none; } h2 { break-after: avoid; } figure { break-inside: avoid; } }
</style>
</head>
<body>
<p class="badge">HISTORICAL / RESEARCH PROTOTYPE</p>
<h1>Day 05 Historical Forecast Page Report</h1>
<p class="subtitle">Singer Product Mix DSS | Generated __GENERATED_AT__</p>
<p>This report explains the new <strong>Historical Forecast Evidence</strong> page added after the Day 05 baseline forecasting work. The page is evidence only: it shows saved historical validation results and does not provide current operational advice or stock recommendations.</p>
<div class="note"><strong>Important boundary.</strong> The Forecast page reads saved Day 05 artifacts. Opening or filtering the page does not run forecasting, train a model, recalculate metrics, or modify the data.</div>

<h2>What the page shows</h2>
<div class="grid">
<div class="metric"><span>Selected baseline</span><strong>trailing_mean_3</strong></div>
<div class="metric"><span>Overall validation MAE</span><strong>2.24 units</strong></div>
<div class="metric"><span>Overall validation WAPE</span><strong>58.82%</strong></div>
</div>
<p>The selected baseline was chosen from the three simple baselines using pooled chronological validation for December 2025, January 2026, and February 2026. March 2026 remains reserved as the untouched final test month.</p>

<h2>How the new page works</h2>
<table><thead><tr><th>Part</th><th>Meaning</th></tr></thead><tbody>
<tr><td>Research banner</td><td>Clearly states that the display is a historical research prototype, not current operational advice.</td></tr>
<tr><td>Branch and month filters</td><td>Allow a reader to narrow the saved evidence to one branch, one validation month, or both.</td></tr>
<tr><td>Selected baseline card</td><td>Shows the actual saved model decision and its pooled validation MAE and WAPE.</td></tr>
<tr><td>Forecast evidence table</td><td>Shows product, family, predicted units, actual units, model used, absolute error, and fallback status.</td></tr>
<tr><td>Fallback field</td><td>Makes sparse-history handling visible. It distinguishes normal trailing-mean predictions from lag-1 or zero fallbacks.</td></tr>
</tbody></table>

<h2>Default page view</h2>
<figure><img src="__DEFAULT_IMAGE__" alt="Default historical forecast evidence page"><figcaption>The default view shows all saved validation evidence for the selected baseline. The page labels that only the first 100 of 3,532 matching rows are displayed, so no evidence is silently presented as complete on screen.</figcaption></figure>

<h2>Filtered evidence example</h2>
<figure><img src="__FILTERED_IMAGE__" alt="Forecast page filtered to branch BIL01 and validation month 2026-02"><figcaption>This example applies Branch <strong>BIL01</strong> and Validation month <strong>2026-02</strong>. The page reports 102 matching persisted evidence rows and preserves the selected model metrics and fallback column.</figcaption></figure>

<h2>How to read the table</h2>
<ul>
<li><strong>Predicted units</strong> is the saved forecast from the selected baseline.</li>
<li><strong>Actual units</strong> is the historical next-month fulfilled positive sales target used for validation.</li>
<li><strong>Absolute error</strong> is the size of the difference between the prediction and the actual value.</li>
<li><strong>Fallback</strong> shows whether sparse history required an explicit alternative to the primary trailing three-month mean.</li>
</ul>

<h2>Data and safety notes</h2>
<ul>
<li>Displayed artifacts: <code>baseline_predictions.parquet</code>, <code>baseline_metrics.csv</code>, and <code>selected_baseline.json</code>.</li>
<li>Validation months shown: 2025-12, 2026-01, and 2026-02.</li>
<li>March 2026 is not displayed as model-selection evidence.</li>
<li>The page does not present Increase, Reduce, stock, profit, or allocation recommendations.</li>
</ul>
<p class="small">Open the page locally at http://127.0.0.1:8000/analytics/forecast/ while the Django development server is running.</p>
</body></html>"""
    report = report.replace("__GENERATED_AT__", generated_at)
    report = report.replace("__DEFAULT_IMAGE__", default_image)
    report = report.replace("__FILTERED_IMAGE__", filtered_image)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(OUTPUT_PATH)
    print(f"bytes={OUTPUT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
