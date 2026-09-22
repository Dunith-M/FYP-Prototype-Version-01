"""Safe persistence of the frozen monthly analytical feature evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Mapping

import pandas as pd

from analytics.services.targets import TARGET_NAME
from analytics.services.target_alignment import SUPERVISED_TARGET


FEATURE_KEY = ("branch_code", "canonical_product_id", "year_month")
FEATURE_VERSION = "feature_month_schema_v1"
SCHEMA_VERSION = "feature_month_manifest_v1"
REQUIRED_SOURCES = ("sales", "inventory", "products", "calendar", "branches")
FORBIDDEN_COLUMNS = {
    "true_demand", "cost", "margin", "profit", "landed_cost", "contribution_margin", "expected_profit"
}


class FeatureSnapshotValidationError(ValueError):
    """Raised when the supervised monthly data cannot safely be frozen."""


@dataclass(frozen=True)
class FeatureSnapshotMetrics:
    row_count: int
    column_count: int
    unique_branches: int
    unique_products: int
    min_feature_month: str | None
    max_feature_month: str | None
    valid_target_row_count: int
    missing_target_row_count: int
    duplicate_monthly_keys: int
    artifact_hash: str


@dataclass(frozen=True)
class FeatureSnapshotResult:
    artifact_path: Path
    manifest_path: Path
    manifest: dict[str, object]
    metrics: FeatureSnapshotMetrics


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest of exact file bytes, streamed without loading it all."""

    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes_from_raw_files(source_paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    """Resolve exact raw-file hashes when SourceFile metadata is unavailable.

    Hashes are calculated from original source-file bytes, never transformed
    DataFrames. Callers must supply all five declared source datasets.
    """

    missing = sorted(set(REQUIRED_SOURCES) - set(source_paths))
    if missing:
        raise FeatureSnapshotValidationError(f"Required source paths are missing: {missing}.")
    resolved: dict[str, dict[str, str]] = {}
    for source_name in REQUIRED_SOURCES:
        path = Path(source_paths[source_name])
        if not path.is_file():
            raise FeatureSnapshotValidationError(f"Required raw source is unavailable: {source_name} ({path}).")
        resolved[source_name] = {
            "hash": sha256_file(path),
            "hash_algorithm": "SHA-256",
            "hash_provenance": "computed_from_original_raw_file_bytes",
            "filename": path.name,
        }
    return resolved


def mapping_snapshot_version(mapping_records: list[dict[str, object]]) -> str:
    """Create an exact, deterministic canonical-SKU mapping snapshot identifier."""

    canonical = json.dumps(mapping_records, sort_keys=True, default=str, separators=(",", ":"))
    return f"sha256:{sha256(canonical.encode('utf-8')).hexdigest()}"


def _periods(values: pd.Series) -> pd.PeriodIndex:
    try:
        periods = pd.PeriodIndex(values.astype("string").str.strip(), freq="M")
    except (TypeError, ValueError) as error:
        raise FeatureSnapshotValidationError("year_month must use parseable YYYY-MM values.") from error
    if periods.isna().any():
        raise FeatureSnapshotValidationError("year_month contains null or blank values.")
    return periods


def _validate_features(frame: pd.DataFrame) -> None:
    required = (*FEATURE_KEY, TARGET_NAME, SUPERVISED_TARGET, "target_available_flag")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise FeatureSnapshotValidationError(f"Feature dataset is missing columns: {missing}.")
    forbidden = sorted({column.casefold() for column in frame.columns} & FORBIDDEN_COLUMNS)
    if forbidden:
        raise FeatureSnapshotValidationError(f"Feature dataset contains forbidden fields: {forbidden}.")
    if frame[list(FEATURE_KEY)].isna().any().any():
        raise FeatureSnapshotValidationError("Feature monthly keys must be non-null.")
    if frame["canonical_product_id"].astype("string").str.strip().eq("").any():
        raise FeatureSnapshotValidationError("canonical_product_id must be non-blank.")
    _periods(frame["year_month"])
    duplicates = int(frame.duplicated(list(FEATURE_KEY)).sum())
    if duplicates:
        raise FeatureSnapshotValidationError(f"Feature dataset has duplicate monthly keys={duplicates}.")
    for column in (TARGET_NAME, SUPERVISED_TARGET):
        values = pd.to_numeric(frame[column], errors="coerce")
        if column == TARGET_NAME and values.isna().any():
            raise FeatureSnapshotValidationError(f"{column} must be non-null and numeric.")
        if (values.dropna() < 0).any():
            raise FeatureSnapshotValidationError(f"{column} must be non-negative where present.")
    available = frame["target_available_flag"].astype(bool)
    if frame.loc[available, SUPERVISED_TARGET].isna().any() or frame.loc[~available, SUPERVISED_TARGET].notna().any():
        raise FeatureSnapshotValidationError("target_available_flag and target_next_month do not agree.")


def _git_metadata(repository_path: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_path, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], cwd=repository_path, check=True, capture_output=True, text=True).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _feature_columns(columns: list[str]) -> list[str]:
    excluded = {*FEATURE_KEY, TARGET_NAME, SUPERVISED_TARGET, "target_year_month", "target_available_flag"}
    return [column for column in columns if column not in excluded]


def freeze_feature_month(
    features: pd.DataFrame,
    *,
    artifact_path: Path,
    manifest_path: Path,
    source_hashes: Mapping[str, Mapping[str, str]],
    mapping_version: str,
    planning_cutoff: str,
    repository_path: Path,
    created_at: str | None = None,
) -> FeatureSnapshotResult:
    """Write and verify the frozen feature snapshot plus traceability manifest.

    ``feature_month.parquet`` is the frozen reproducible monthly analytical
    feature snapshot. Its manifest records exact source hashes, cutoff,
    mapping/feature versions, target semantics and artifact integrity metadata.
    """

    _validate_features(features)
    if not mapping_version.strip():
        raise FeatureSnapshotValidationError("mapping_version must be a real non-blank snapshot identifier.")
    missing_sources = sorted(set(REQUIRED_SOURCES) - set(source_hashes))
    if missing_sources:
        raise FeatureSnapshotValidationError(f"Required source hashes are missing: {missing_sources}.")
    for source in REQUIRED_SOURCES:
        digest = source_hashes[source].get("hash", "")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
            raise FeatureSnapshotValidationError(f"{source} source hash is not a SHA-256 digest.")

    artifact_path = Path(artifact_path)
    manifest_path = Path(manifest_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = artifact_path.with_name(f".{artifact_path.name}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    features.to_parquet(temporary_artifact, index=False, engine="pyarrow")
    reloaded = pd.read_parquet(temporary_artifact, engine="pyarrow")
    try:
        pd.testing.assert_frame_equal(features.reset_index(drop=True), reloaded.reset_index(drop=True), check_dtype=True)
    except AssertionError as error:
        temporary_artifact.unlink(missing_ok=True)
        raise FeatureSnapshotValidationError("Parquet round-trip changed feature values or column types.") from error
    artifact_hash = sha256_file(temporary_artifact)
    target_available = reloaded["target_available_flag"].astype(bool)
    git_commit, working_tree_dirty = _git_metadata(repository_path)
    try:
        manifest_artifact_path = artifact_path.relative_to(repository_path).as_posix()
    except ValueError:
        manifest_artifact_path = artifact_path.as_posix()
    manifest: dict[str, object] = {
        "artifact_name": "feature_month",
        "artifact_path": manifest_artifact_path,
        "created_at": created_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "HISTORICAL_RESEARCH",
        "planning_cutoff": planning_cutoff,
        "sources": {name: dict(source_hashes[name]) for name in REQUIRED_SOURCES},
        "mapping_version": mapping_version,
        "feature_version": FEATURE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "target_definition": {
            "source": TARGET_NAME,
            "supervised_target": SUPERVISED_TARGET,
            "meaning": "next-calendar-month observed positive fulfilled sales units proxy for the same branch-product pair",
            "not_claimed": ["true demand", "unconstrained demand", "lost-sales adjusted demand"],
        },
        "key_columns": list(FEATURE_KEY),
        "row_count": len(reloaded),
        "column_count": len(reloaded.columns),
        "feature_columns": _feature_columns(reloaded.columns.tolist()),
        "target_column": SUPERVISED_TARGET,
        "min_feature_month": str(reloaded["year_month"].min()) if len(reloaded) else None,
        "max_feature_month": str(reloaded["year_month"].max()) if len(reloaded) else None,
        "valid_target_row_count": int(target_available.sum()),
        "missing_target_row_count": int((~target_available).sum()),
        "artifact_hash": artifact_hash,
        "artifact_hash_algorithm": "SHA-256",
        "git_commit": git_commit,
        "working_tree_dirty": working_tree_dirty,
    }
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_artifact.replace(artifact_path)
    temporary_manifest.replace(manifest_path)
    metrics = FeatureSnapshotMetrics(
        row_count=len(reloaded),
        column_count=len(reloaded.columns),
        unique_branches=int(reloaded["branch_code"].nunique()),
        unique_products=int(reloaded["canonical_product_id"].nunique()),
        min_feature_month=(str(reloaded["year_month"].min()) if len(reloaded) else None),
        max_feature_month=(str(reloaded["year_month"].max()) if len(reloaded) else None),
        valid_target_row_count=int(target_available.sum()),
        missing_target_row_count=int((~target_available).sum()),
        duplicate_monthly_keys=int(reloaded.duplicated(list(FEATURE_KEY)).sum()),
        artifact_hash=artifact_hash,
    )
    return FeatureSnapshotResult(artifact_path, manifest_path, manifest, metrics)
