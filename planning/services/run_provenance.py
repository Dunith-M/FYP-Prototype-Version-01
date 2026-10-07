"""Freeze existing artifact/config provenance for a newly persisted run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import transaction

from ingestion.models import SourceFile
from planning.models import OptimizationResult, RunProvenance
from planning.modes import ScenarioMode


class RunProvenanceCaptureError(RuntimeError):
    """Raised when a new run cannot be linked to its existing persisted inputs."""


REQUIRED_SOURCE_DATASETS = frozenset(
    {
        SourceFile.Dataset.SALES,
        SourceFile.Dataset.INVENTORY,
        SourceFile.Dataset.PRODUCTS,
        SourceFile.Dataset.CALENDAR,
        SourceFile.Dataset.BRANCHES,
    }
)


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RunProvenanceCaptureError(f"Cannot read persisted provenance artifact {path}: {error}") from error


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunProvenanceCaptureError(f"Cannot read persisted JSON provenance artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RunProvenanceCaptureError(f"Persisted provenance artifact {path} is not a JSON object.")
    return payload


def _required_text(payload: dict[str, Any], key: str, *, artifact: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RunProvenanceCaptureError(f"Persisted provenance artifact {artifact} has no usable {key!r} value.")
    return value


def _source_snapshot(feature_manifest: dict[str, Any], *, artifact: Path) -> dict[str, dict[str, str]]:
    sources = feature_manifest.get("sources")
    if not isinstance(sources, dict):
        raise RunProvenanceCaptureError(f"Persisted feature manifest {artifact} has no source snapshot.")

    snapshot: dict[str, dict[str, str]] = {}
    for dataset in sorted(REQUIRED_SOURCE_DATASETS):
        entry = sources.get(dataset)
        if not isinstance(entry, dict):
            raise RunProvenanceCaptureError(f"Persisted feature manifest {artifact} has no {dataset!r} source entry.")
        filename = entry.get("filename")
        file_hash = entry.get("hash")
        if not isinstance(filename, str) or not filename or not isinstance(file_hash, str) or len(file_hash) != 64:
            raise RunProvenanceCaptureError(f"Persisted feature manifest {artifact} has an invalid {dataset!r} source entry.")
        snapshot[dataset] = {"filename": filename, "file_hash": file_hash}
    return snapshot


def _default_paths() -> tuple[Path, Path, Path]:
    base = Path(settings.BASE_DIR)
    return (
        base / "data" / "artifacts" / "feature_month_manifest.json",
        base / "data" / "artifacts" / "day06" / "model_manifest.json",
        base / "config" / "synthetic" / "options.yaml",
    )


def capture_run_provenance(
    *,
    optimization_result: OptimizationResult,
    branch,
    mode: str,
    feature_manifest_path: Path | None = None,
    model_manifest_path: Path | None = None,
    synthetic_config_path: Path | None = None,
) -> RunProvenance:
    """Persist the immutable existing inputs for a new optimization result.

    This reads only small, already-persisted manifests/configuration at run
    time. It never reads or hashes the raw Excel source data. Existing runs
    without a captured snapshot are deliberately left as legacy/incomplete
    rather than backfilled with unprovable current metadata.
    """

    if mode not in ScenarioMode.values:
        raise RunProvenanceCaptureError(f"Unsupported mode for provenance capture: {mode!r}.")

    default_feature, default_model, default_config = _default_paths()
    feature_manifest_path = feature_manifest_path or default_feature
    model_manifest_path = model_manifest_path or default_model
    synthetic_config_path = synthetic_config_path or default_config

    feature_manifest = _read_json(feature_manifest_path)
    model_manifest = _read_json(model_manifest_path)
    source_snapshot = _source_snapshot(feature_manifest, artifact=feature_manifest_path)
    expected = {
        "optimization_result_id": optimization_result.pk,
        "branch_id": branch.pk,
        "mode": mode,
        "source_manifest_path": str(feature_manifest_path),
        "source_manifest_hash": _sha256_file(feature_manifest_path),
        "source_snapshot": source_snapshot,
        "mapping_version": _required_text(feature_manifest, "mapping_version", artifact=feature_manifest_path),
        "feature_version": _required_text(feature_manifest, "feature_version", artifact=feature_manifest_path),
        "feature_artifact_hash": _required_text(feature_manifest, "artifact_hash", artifact=feature_manifest_path),
        "model_manifest_path": str(model_manifest_path),
        "model_manifest_hash": _sha256_file(model_manifest_path),
        "model_type": _required_text(model_manifest, "model_type", artifact=model_manifest_path),
        "model_version": _required_text(model_manifest, "model_version", artifact=model_manifest_path),
        "selected_model": _required_text(model_manifest, "selected_model", artifact=model_manifest_path),
        "scenario_config_path": "",
        "scenario_config_hash": "",
    }
    if mode == ScenarioMode.RESEARCH_SYNTHETIC:
        expected["scenario_config_path"] = str(synthetic_config_path)
        expected["scenario_config_hash"] = _sha256_file(synthetic_config_path)

    with transaction.atomic():
        provenance, created = RunProvenance.objects.get_or_create(
            optimization_result=optimization_result,
            defaults={key: value for key, value in expected.items() if key != "optimization_result_id"},
        )
        if not created:
            mismatches = [
                key for key, value in expected.items()
                if key != "optimization_result_id" and getattr(provenance, key) != value
            ]
            if mismatches:
                raise RunProvenanceCaptureError(
                    f"Run {optimization_result.pk} already has a different immutable provenance snapshot: "
                    f"{', '.join(mismatches)}."
                )

        exact_source_ids: list[int] = []
        for dataset, entry in source_snapshot.items():
            exact_source_ids.extend(
                SourceFile.objects.filter(dataset=dataset, file_hash=entry["file_hash"]).values_list("pk", flat=True)
            )
        provenance.source_files.set(exact_source_ids)
    return provenance
