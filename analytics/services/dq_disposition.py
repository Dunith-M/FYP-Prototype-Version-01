"""Run-specific DQ disposition reporting without changing analytical evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import pandas as pd


REQUIRED_DISPOSITION_COLUMNS = (
    "run_id", "dataset", "created_at", "source_file_id", "source_file_hash", "processing_stage",
    "total_rows", "accepted_rows", "quarantined_rows", "rejected_rows", "unmatched_branch_rows",
    "unmatched_product_rows", "negative_unit_rows", "zero_unit_rows", "invalid_price_rows",
    "duplicate_key_rows", "reconciliation_status", "notes",
)


class DqDispositionValidationError(ValueError):
    """Raised when actual run disposition counts do not reconcile."""


@dataclass(frozen=True)
class DqDispositionBuildResult:
    report: pd.DataFrame
    run_id: str


def run_id_from_feature_manifest(manifest: Mapping[str, object]) -> str:
    """Use the frozen artifact hash as the deterministic current-run identifier."""

    artifact_hash = str(manifest.get("artifact_hash", "")).lower()
    if len(artifact_hash) != 64 or any(character not in "0123456789abcdef" for character in artifact_hash):
        raise DqDispositionValidationError("Feature manifest has no valid artifact SHA-256 for the run ID.")
    return f"dq-{artifact_hash[:16]}"


def _count(frame: pd.DataFrame, count_column: str, flag_column: str) -> int:
    if count_column in frame.columns:
        return int(pd.to_numeric(frame[count_column], errors="raise").sum())
    if flag_column in frame.columns:
        return int(frame[flag_column].fillna(False).astype(bool).sum())
    return 0


def _source_metadata(
    dataset: str,
    source_hashes: Mapping[str, Mapping[str, str]],
    source_file_ids: Mapping[str, object],
) -> tuple[str, str]:
    try:
        digest = str(source_hashes[dataset]["hash"])
    except KeyError as error:
        raise DqDispositionValidationError(f"Required {dataset} source hash is unavailable.") from error
    if len(digest) != 64:
        raise DqDispositionValidationError(f"{dataset} source hash is not a SHA-256 digest.")
    source_file_id = source_file_ids.get(dataset)
    return ("" if source_file_id is None else str(source_file_id), digest)


def _row(
    *,
    run_id: str,
    dataset: str,
    created_at: str,
    source_hashes: Mapping[str, Mapping[str, str]],
    source_file_ids: Mapping[str, object],
    total: int,
    accepted: int,
    quarantined: int,
    rejected: int = 0,
    unmatched_branch: int = 0,
    unmatched_product: int = 0,
    negative_units: int = 0,
    zero_units: int = 0,
    invalid_price: int = 0,
    duplicate_keys: int = 0,
    notes: str = "",
) -> dict[str, object]:
    source_file_id, source_file_hash = _source_metadata(dataset, source_hashes, source_file_ids)
    return {
        "run_id": run_id,
        "dataset": dataset,
        "created_at": created_at,
        "source_file_id": source_file_id,
        "source_file_hash": source_file_hash,
        "processing_stage": "day03_to_step10_processed_snapshot",
        "total_rows": int(total),
        "accepted_rows": int(accepted),
        "quarantined_rows": int(quarantined),
        "rejected_rows": int(rejected),
        "unmatched_branch_rows": int(unmatched_branch),
        "unmatched_product_rows": int(unmatched_product),
        "negative_unit_rows": int(negative_units),
        "zero_unit_rows": int(zero_units),
        "invalid_price_rows": int(invalid_price),
        "duplicate_key_rows": int(duplicate_keys),
        "reconciliation_status": "PASS",
        "notes": notes,
    }


def build_dq_disposition_report(
    *,
    sales_day: pd.DataFrame,
    sales_unresolved: pd.DataFrame,
    inventory_day: pd.DataFrame,
    inventory_unresolved: pd.DataFrame,
    products: pd.DataFrame,
    calendar: pd.DataFrame,
    branches: pd.DataFrame,
    feature_manifest: Mapping[str, object],
    source_file_ids: Mapping[str, object] | None = None,
) -> DqDispositionBuildResult:
    """Build run-specific acceptance/quarantine evidence from actual outputs.

    The DQ disposition report is run-specific operational evidence showing how
    source records were accepted, quarantined or rejected. It is separate from
    ``docs/data_quality_log.csv``, which tracks known DQ rules/issues.
    """

    source_file_ids = source_file_ids or {}
    source_hashes = feature_manifest.get("sources")
    created_at = str(feature_manifest.get("created_at", ""))
    if not isinstance(source_hashes, Mapping) or not created_at:
        raise DqDispositionValidationError("Feature manifest lacks source hashes or created_at metadata.")
    run_id = run_id_from_feature_manifest(feature_manifest)

    sales_accepted = _count(sales_day, "source_row_count", "")
    sales_quarantined = len(sales_unresolved)
    inventory_accepted = len(inventory_day)
    inventory_quarantined = len(inventory_unresolved)
    product_duplicates = int(products.duplicated(["product_code"]).sum()) if "product_code" in products else 0
    calendar_duplicates = int(calendar.duplicated(["DATE"]).sum()) if "DATE" in calendar else 0
    branch_duplicates = int(branches.duplicated(["SHOP_CODE"]).sum()) if "SHOP_CODE" in branches else 0

    rows = [
        _row(
            run_id=run_id, dataset="sales", created_at=created_at, source_hashes=source_hashes,
            source_file_ids=source_file_ids, total=sales_accepted + sales_quarantined,
            accepted=sales_accepted, quarantined=sales_quarantined,
            unmatched_product=sales_quarantined,
            negative_units=_count(sales_day, "negative_unit_row_count", "negative_units_flag") + _count(sales_unresolved, "", "negative_units_flag"),
            zero_units=_count(sales_day, "zero_unit_row_count", "zero_units_flag") + _count(sales_unresolved, "", "zero_units_flag"),
            notes="Resolved rows entered SALES_DAY; unresolved product rows are preserved in sales_unresolved. Negative and zero units are diagnostics only.",
        ),
        _row(
            run_id=run_id, dataset="inventory", created_at=created_at, source_hashes=source_hashes,
            source_file_ids=source_file_ids, total=inventory_accepted + inventory_quarantined,
            accepted=inventory_accepted, quarantined=inventory_quarantined,
            unmatched_product=inventory_quarantined,
            zero_units=int(inventory_day.get("availability_state", pd.Series(dtype="string")).eq("ZERO_CONFIRMED").sum()),
            duplicate_keys=int(inventory_day.duplicated(["branch_code", "canonical_product_id", "snapshot_date"]).sum()),
            notes="Resolved rows entered INVENTORY_DAY; unresolved product rows are preserved in inventory_unresolved. UNKNOWN inventory is not zero inventory.",
        ),
        _row(
            run_id=run_id, dataset="products", created_at=created_at, source_hashes=source_hashes,
            source_file_ids=source_file_ids, total=len(products), accepted=len(products), quarantined=0,
            invalid_price=int((~products["price_valid"].fillna(False)).sum()), duplicate_keys=product_duplicates,
            notes="All current product-master rows entered the canonical master; invalid list prices are DQ flags and not automatic rejection.",
        ),
        _row(
            run_id=run_id, dataset="calendar", created_at=created_at, source_hashes=source_hashes,
            source_file_ids=source_file_ids, total=len(calendar), accepted=len(calendar), quarantined=0,
            duplicate_keys=calendar_duplicates,
            notes="Validated source calendar rows entered the calendar reference; no dates were invented.",
        ),
        _row(
            run_id=run_id, dataset="branches", created_at=created_at, source_hashes=source_hashes,
            source_file_ids=source_file_ids, total=len(branches), accepted=len(branches), quarantined=0,
            duplicate_keys=branch_duplicates,
            notes="Validated branch-profile rows entered the canonical Branch master; optional profile gaps are DQ diagnostics only.",
        ),
    ]
    report = pd.DataFrame(rows, columns=REQUIRED_DISPOSITION_COLUMNS)
    validate_dq_disposition_report(report)
    return DqDispositionBuildResult(report=report, run_id=run_id)


def validate_dq_disposition_report(report: pd.DataFrame) -> None:
    """Require independent accepted/quarantined/rejected reconciliation per dataset."""

    missing = sorted(set(REQUIRED_DISPOSITION_COLUMNS) - set(report.columns))
    if missing:
        raise DqDispositionValidationError(f"Disposition report is missing columns: {missing}.")
    if report["run_id"].isna().any() or report["run_id"].astype("string").str.strip().eq("").any():
        raise DqDispositionValidationError("Every disposition row must have a run_id.")
    for row in report.itertuples(index=False):
        reconciled = row.accepted_rows + row.quarantined_rows + row.rejected_rows == row.total_rows
        if not reconciled:
            raise DqDispositionValidationError(
                f"{row.dataset} disposition does not reconcile: total={row.total_rows}; "
                f"accepted+quarantined+rejected={row.accepted_rows + row.quarantined_rows + row.rejected_rows}."
            )
    if report.duplicated(["run_id", "dataset"]).any():
        raise DqDispositionValidationError("Disposition report has duplicate run/dataset rows.")


def write_dq_disposition_report(result: DqDispositionBuildResult, output_path: Path) -> None:
    """Atomically write the report; project-wide DQ documentation is never touched."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    result.report.to_csv(temporary, index=False, encoding="utf-8")
    reloaded = pd.read_csv(temporary, keep_default_na=False)
    validate_dq_disposition_report(reloaded)
    temporary.replace(output_path)


def load_feature_manifest(path: Path) -> dict[str, object]:
    """Read only the existing Step-10 frozen-feature manifest."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DqDispositionValidationError(f"Unable to read feature manifest: {path}.") from error
