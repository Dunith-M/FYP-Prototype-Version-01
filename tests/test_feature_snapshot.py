import json
from hashlib import sha256

import pandas as pd
import pytest

from analytics.services.feature_snapshot import (
    FeatureSnapshotValidationError,
    freeze_feature_month,
    sha256_file,
    source_hashes_from_raw_files,
)


def _features():
    return pd.DataFrame([
        {"branch_code": "BIL01", "canonical_product_id": "CP_1", "year_month": "2026-01", "fulfilled_positive_units_proxy": 5.0, "lag_1": 2.0, "valid_price": True, "target_next_month": 7.0, "target_year_month": "2026-02", "target_available_flag": True},
        {"branch_code": "BIL01", "canonical_product_id": "CP_1", "year_month": "2026-02", "fulfilled_positive_units_proxy": 7.0, "lag_1": 5.0, "valid_price": True, "target_next_month": None, "target_year_month": None, "target_available_flag": False},
    ])


def _hashes():
    return {name: {"hash": sha256(name.encode()).hexdigest()} for name in ("sales", "inventory", "products", "calendar", "branches")}


def test_snapshot_round_trip_manifest_and_hash_are_exact(tmp_path):
    artifact = tmp_path / "processed" / "feature_month.parquet"
    manifest_path = tmp_path / "artifacts" / "feature_month_manifest.json"
    result = freeze_feature_month(_features(), artifact_path=artifact, manifest_path=manifest_path, source_hashes=_hashes(), mapping_version="sha256:" + "a" * 64, planning_cutoff="2026-02-28", repository_path=tmp_path, created_at="2026-09-18T19:30:00+05:30")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    restored = pd.read_parquet(artifact)
    assert artifact.exists() and manifest_path.exists()
    assert len(restored) == len(_features()) == manifest["row_count"]
    assert restored.columns.tolist() == _features().columns.tolist()
    assert pd.isna(restored.loc[1, "target_next_month"])
    assert manifest["artifact_hash"] == sha256_file(artifact) == result.metrics.artifact_hash
    assert manifest["sources"] == _hashes()
    assert manifest["target_definition"]["source"] == "fulfilled_positive_units_proxy"
    assert manifest["target_definition"]["supervised_target"] == "target_next_month"
    assert "true demand" in manifest["target_definition"]["not_claimed"]


def test_duplicate_keys_missing_hash_or_forbidden_fields_block_freeze(tmp_path):
    paths = dict(artifact_path=tmp_path / "a.parquet", manifest_path=tmp_path / "m.json", source_hashes=_hashes(), mapping_version="sha256:" + "a" * 64, planning_cutoff="2026-02-28", repository_path=tmp_path)
    with pytest.raises(FeatureSnapshotValidationError, match="duplicate monthly"):
        freeze_feature_month(pd.concat([_features(), _features().iloc[[0]]], ignore_index=True), **paths)
    missing = _hashes(); missing.pop("sales")
    with pytest.raises(FeatureSnapshotValidationError, match="source hashes are missing"):
        freeze_feature_month(_features(), **(paths | {"source_hashes": missing}))
    with pytest.raises(FeatureSnapshotValidationError, match="forbidden"):
        freeze_feature_month(_features().assign(profit=1), **paths)


def test_raw_source_hashes_are_calculated_from_exact_file_bytes(tmp_path):
    source_paths = {}
    for name in ("sales", "inventory", "products", "calendar", "branches"):
        path = tmp_path / f"{name}.xlsx"
        path.write_bytes(name.encode("utf-8"))
        source_paths[name] = path
    resolved = source_hashes_from_raw_files(source_paths)
    assert resolved["sales"]["hash"] == sha256(b"sales").hexdigest()
    assert resolved["sales"]["hash_provenance"] == "computed_from_original_raw_file_bytes"
