"""Freeze the validated Step-09 supervised monthly feature dataset."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.services.feature_snapshot import (
    FeatureSnapshotValidationError,
    freeze_feature_month,
    mapping_snapshot_version,
    source_hashes_from_raw_files,
)
from catalog.models import SkuMapping


class Command(BaseCommand):
    help = "Freeze feature_month.parquet and its reproducibility manifest."

    def add_arguments(self, parser):
        processed = settings.BASE_DIR / "data" / "processed"
        artifacts = settings.BASE_DIR / "data" / "artifacts"
        parser.add_argument("--input", type=Path, default=processed / "branch_product_month_supervised.parquet")
        parser.add_argument("--artifact", type=Path, default=processed / "feature_month.parquet")
        parser.add_argument("--manifest", type=Path, default=artifacts / "feature_month_manifest.json")
        parser.add_argument("--daily-evidence", type=Path, default=processed / "branch_product_day.parquet")

    def handle(self, *args, **options):
        raw = settings.BASE_DIR / "data" / "raw"
        source_paths = {
            "sales": raw / "transactions_data.xlsx",
            "inventory": raw / "inventory_data.xlsx",
            "products": raw / "product_data.xlsx",
            "calendar": raw / "calendar_data.xlsx",
            "branches": raw / "branch_profile.xlsx",
        }
        mapping_records = list(
            SkuMapping.objects.values(
                "source_system", "source_product_code", "canonical_product_id", "mapping_reason",
                "confidence_class", "effective_from", "effective_to", "reviewer", "reviewed_at", "notes",
            ).order_by("source_system", "source_product_code")
        )
        try:
            daily = pd.read_parquet(options["daily_evidence"])
            planning_cutoff = str(pd.to_datetime(daily["business_date"], errors="raise").max().date())
            result = freeze_feature_month(
                pd.read_parquet(options["input"]),
                artifact_path=options["artifact"],
                manifest_path=options["manifest"],
                source_hashes=source_hashes_from_raw_files(source_paths),
                mapping_version=mapping_snapshot_version(mapping_records),
                planning_cutoff=planning_cutoff,
                repository_path=settings.BASE_DIR,
            )
        except (OSError, ValueError, FeatureSnapshotValidationError) as error:
            raise CommandError(str(error)) from error
        metrics = result.metrics
        self.stdout.write(self.style.SUCCESS("Frozen feature snapshot complete."))
        self.stdout.write(
            f"rows={metrics.row_count}; columns={metrics.column_count}; valid_target={metrics.valid_target_row_count}; "
            f"missing_target={metrics.missing_target_row_count}; sha256={metrics.artifact_hash}"
        )
        self.stdout.write(f"artifact={result.artifact_path}; manifest={result.manifest_path}")
