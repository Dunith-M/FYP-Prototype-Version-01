"""Read lightweight persisted mapping and data-quality evidence for the UI."""

from __future__ import annotations

import csv
from pathlib import Path

from django.conf import settings

from catalog.branch_master import EXPECTED_BRANCH_CODES
from catalog.models import Branch


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as report_file:
            return list(csv.DictReader(report_file))
    except OSError:
        return []


def _integer(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _display_count(value: str | None) -> str:
    count = _integer(value)
    return f"{count:,}" if count is not None else "Not available"


def _badge_class(status: str) -> str:
    if status == "PASS":
        return "bg-success"
    if status in {"REVIEW", "OPEN", "WARNING"}:
        return "bg-warning text-dark"
    return "bg-secondary"


def _metric(logs: dict[str, dict[str, str]], rule_id: str) -> dict[str, str] | None:
    row = logs.get(rule_id)
    if row is None:
        return None
    status = row.get("status", "NOT AVAILABLE")
    return {
        "label": row.get("issue", rule_id),
        "value": _display_count(row.get("count")),
        "status": status,
        "badge_class": _badge_class(status),
        "notes": row.get("notes", ""),
    }


def _combined_metric(
    logs: dict[str, dict[str, str]],
    *,
    label: str,
    rule_ids: tuple[str, ...],
) -> dict[str, str]:
    rows = [logs[rule_id] for rule_id in rule_ids if rule_id in logs]
    if not rows:
        return {
            "label": label,
            "value": "Not available",
            "status": "NOT AVAILABLE",
            "badge_class": _badge_class("NOT AVAILABLE"),
            "notes": "Not available in persisted DQ results.",
        }
    parts = [
        f"{logs[rule_id].get('issue', rule_id)}: {_display_count(logs[rule_id].get('count'))}"
        for rule_id in rule_ids
        if rule_id in logs
    ]
    status = next((row.get("status", "NOT AVAILABLE") for row in rows if row.get("status") != "PASS"), "PASS")
    return {
        "label": label,
        "value": "; ".join(parts),
        "status": status,
        "badge_class": _badge_class(status),
        "notes": "Persisted DQ log counts are diagnostics and may overlap.",
    }


def get_data_quality_page_data(*, unresolved_limit: int = 20) -> dict[str, object]:
    """Return saved reports and database metadata without opening raw datasets."""
    root = Path(settings.BASE_DIR)
    mapping_rows = _read_csv(root / "data" / "artifacts" / "mapping" / "mapping_coverage.csv")
    unresolved_rows = _read_csv(root / "data" / "artifacts" / "mapping" / "unresolved_sku_queue.csv")
    dq_rows = _read_csv(root / "docs" / "data_quality_log.csv")
    disposition_rows = _read_csv(root / "data" / "artifacts" / "day04_dq_disposition.csv")

    mapping_by_metric = {row.get("metric"): row for row in mapping_rows}
    product_mapping = mapping_by_metric.get("distinct_product_codes")
    logs = {row.get("rule_id"): row for row in dq_rows if row.get("rule_id")}
    disposition_by_dataset = {row.get("dataset"): row for row in disposition_rows if row.get("dataset")}

    expected_branch_count = len(EXPECTED_BRANCH_CODES)
    mapped_branch_count = Branch.objects.count()
    branch_coverage = (
        f"{mapped_branch_count / expected_branch_count * 100:.1f}%"
        if expected_branch_count
        else "Not available"
    )
    branch_mapping = _metric(logs, "DQ-BR-003")
    branch_unmatched = _display_count(logs.get("DQ-BR-003", {}).get("count"))

    quality_metrics = [
        metric
        for metric in (
            _metric(logs, "DQ-SALES-002"),
            _metric(logs, "DQ-SALES-003"),
            _combined_metric(logs, label="Sales component reconciliation mismatches", rule_ids=("DQ-SALES-009", "DQ-SALES-010")),
            _metric(logs, "DQ-SALES-008"),
            _combined_metric(logs, label="Invalid/non-positive product prices", rule_ids=("DQ-PROD-001", "DQ-PROD-002", "DQ-PROD-003")),
            _combined_metric(logs, label="Calendar validation issues", rule_ids=("DQ-CAL-001", "DQ-CAL-002", "DQ-CAL-003", "DQ-CAL-004", "DQ-CAL-005")),
            _metric(logs, "DQ-BR-004"),
        )
        if metric is not None
    ]

    safety_checks = [
        {
            "label": "Branch mapping",
            "status": branch_mapping["status"] if branch_mapping else "NOT AVAILABLE",
            "badge_class": branch_mapping["badge_class"] if branch_mapping else _badge_class("NOT AVAILABLE"),
            "detail": branch_mapping["notes"] if branch_mapping else "Not available in persisted DQ results.",
        },
        {
            "label": "Processed row-count reconciliation",
            "status": disposition_by_dataset.get("sales", {}).get("reconciliation_status", "NOT AVAILABLE"),
            "badge_class": _badge_class(disposition_by_dataset.get("sales", {}).get("reconciliation_status", "NOT AVAILABLE")),
            "detail": disposition_by_dataset.get("sales", {}).get("notes", "Not available in persisted DQ results."),
        },
        {
            "label": "Missing inventory remains UNKNOWN, not zero",
            "status": disposition_by_dataset.get("inventory", {}).get("reconciliation_status", "NOT AVAILABLE"),
            "badge_class": _badge_class(disposition_by_dataset.get("inventory", {}).get("reconciliation_status", "NOT AVAILABLE")),
            "detail": disposition_by_dataset.get("inventory", {}).get("notes", "Not available in persisted DQ results."),
        },
        {
            "label": "Sales/inventory row-explosion guard",
            "status": "NOT AVAILABLE",
            "badge_class": _badge_class("NOT AVAILABLE"),
            "detail": "Verified by existing Day-03 tests, but no standalone persisted page metric is available.",
        },
    ]

    return {
        "mode": "HISTORICAL_RESEARCH",
        "branch_mapping": {
            "expected": _display_count(str(expected_branch_count)),
            "mapped": _display_count(str(mapped_branch_count)),
            "unmatched": branch_unmatched,
            "coverage": branch_coverage,
            "status": branch_mapping["status"] if branch_mapping else "NOT AVAILABLE",
            "badge_class": branch_mapping["badge_class"] if branch_mapping else _badge_class("NOT AVAILABLE"),
        },
        "product_mapping": (
            {
                "total": _display_count(product_mapping.get("total")),
                "mapped": _display_count(product_mapping.get("mapped")),
                "unresolved": _display_count(product_mapping.get("unresolved")),
                "coverage": f"{float(product_mapping['coverage_pct']):.1f}%" if product_mapping.get("coverage_pct") else "Not available",
            }
            if product_mapping
            else None
        ),
        "quality_metrics": quality_metrics,
        "unknown_inventory": "Not available in persisted DQ results.",
        "unresolved_rows": unresolved_rows[:unresolved_limit],
        "unresolved_total": len(unresolved_rows),
        "safety_checks": safety_checks,
        "data_available": bool(mapping_rows or dq_rows or disposition_rows),
    }
