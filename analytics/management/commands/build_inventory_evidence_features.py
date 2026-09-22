"""Build Step-07 inventory evidence features from processed Parquet layers."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.services.inventory_features import (
    InventoryEvidenceFeatureValidationError,
    build_inventory_evidence_features,
    write_inventory_evidence_features,
)


class Command(BaseCommand):
    help = "Build strictly-prior inventory evidence features for prediction months."

    def add_arguments(self, parser):
        processed = settings.BASE_DIR / "data" / "processed"
        parser.add_argument(
            "--input", type=Path, default=processed / "branch_product_month_demand_history_features.parquet"
        )
        parser.add_argument("--daily-evidence", type=Path, default=processed / "branch_product_day.parquet")
        parser.add_argument("--unresolved-audit", type=Path, default=processed / "sales_unresolved.parquet")
        parser.add_argument(
            "--output", type=Path, default=processed / "branch_product_month_inventory_evidence_features.parquet"
        )

    def handle(self, *args, **options):
        try:
            result = build_inventory_evidence_features(
                pd.read_parquet(options["input"]),
                pd.read_parquet(options["daily_evidence"]),
                pd.read_parquet(options["unresolved_audit"]),
            )
        except (OSError, ValueError, InventoryEvidenceFeatureValidationError) as error:
            raise CommandError(str(error)) from error

        write_inventory_evidence_features(result, options["output"])
        metrics = result.metrics
        self.stdout.write(self.style.SUCCESS("Inventory evidence feature build complete."))
        self.stdout.write(
            f"rows={metrics.feature_rows}; last_observed_available={metrics.last_observed_available_rows}; "
            f"last_observed_missing={metrics.last_observed_missing_rows}; "
            f"inventory_count={metrics.minimum_inventory_observation_count}/"
            f"{metrics.mean_inventory_observation_count:.2f}/{metrics.maximum_inventory_observation_count}; "
            f"unknown_rate={metrics.minimum_availability_unknown_rate}..{metrics.maximum_availability_unknown_rate}"
        )
        self.stdout.write(
            f"fully_unknown={metrics.fully_unknown_rows}; zero_observed={metrics.zero_observed_inventory_rows}; "
            f"future_leakage_affected={metrics.future_data_leakage_affected_rows}; "
            f"unresolved_excluded={metrics.unresolved_rows_excluded}"
        )
        self.stdout.write(f"output={options['output']}")
