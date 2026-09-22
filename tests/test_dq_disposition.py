import hashlib
from pathlib import Path

import pandas as pd
import pytest

from analytics.services.dq_disposition import (
    DqDispositionValidationError,
    build_dq_disposition_report,
    validate_dq_disposition_report,
    write_dq_disposition_report,
)


def _manifest():
    return {
        "artifact_hash": "a" * 64,
        "created_at": "2026-09-18T19:49:31+05:30",
        "sources": {name: {"hash": hashlib.sha256(name.encode()).hexdigest()} for name in ("sales", "inventory", "products", "calendar", "branches")},
    }


def _report():
    return build_dq_disposition_report(
        sales_day=pd.DataFrame([{"source_row_count": 90, "negative_unit_row_count": 2, "zero_unit_row_count": 1}]),
        sales_unresolved=pd.DataFrame([{"negative_units_flag": True, "zero_units_flag": False}] * 8),
        inventory_day=pd.DataFrame([{"branch_code": "BIL01", "canonical_product_id": "CP1", "snapshot_date": "2026-01-01", "availability_state": "AVAILABLE"}]),
        inventory_unresolved=pd.DataFrame(columns=["source_product_code"]),
        products=pd.DataFrame([{"product_code": "P1", "price_valid": True}, {"product_code": "P2", "price_valid": False}]),
        calendar=pd.DataFrame([{"DATE": "2026-01-01"}]),
        branches=pd.DataFrame([{"SHOP_CODE": "BIL01"}]),
        feature_manifest=_manifest(),
    )


def test_disposition_reconciles_overlapping_diagnostics_and_writes_csv(tmp_path):
    result = _report()
    sales = result.report.set_index("dataset").loc["sales"]
    assert sales.total_rows == 98
    assert sales.accepted_rows == 90
    assert sales.quarantined_rows == 8
    assert sales.negative_unit_rows == 10
    assert sales.unmatched_product_rows == 8
    assert result.run_id == "dq-" + "a" * 16
    output = tmp_path / "day04_dq_disposition.csv"
    write_dq_disposition_report(result, output)
    assert output.exists()
    assert set(result.report.columns)
    assert pd.read_csv(output).run_id.notna().all()


def test_reconciliation_failure_is_not_silenced():
    report = _report().report.copy()
    report.loc[report.dataset == "sales", "quarantined_rows"] = 5
    with pytest.raises(DqDispositionValidationError, match="does not reconcile"):
        validate_dq_disposition_report(report)


def test_existing_day02_log_is_not_written_by_disposition_report(tmp_path):
    root = Path(__file__).resolve().parents[1]
    log_path = root / "docs" / "data_quality_log.csv"
    before = hashlib.sha256(log_path.read_bytes()).hexdigest()
    write_dq_disposition_report(_report(), tmp_path / "report.csv")
    assert hashlib.sha256(log_path.read_bytes()).hexdigest() == before
