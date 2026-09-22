"""Build Step-05 demand-history features from processed monthly evidence."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.services.feature_engineering import (
    DemandHistoryFeatureValidationError,
    build_demand_history_features,
    write_demand_history_features,
)


class Command(BaseCommand):
    help = "Build leakage-safe demand-history features from BRANCH_PRODUCT_MONTH."

    def add_arguments(self, parser):
        processed = settings.BASE_DIR / "data" / "processed"
        parser.add_argument("--input", type=Path, default=processed / "branch_product_month.parquet")
        parser.add_argument("--unresolved-audit", type=Path, default=processed / "sales_unresolved.parquet")
        parser.add_argument(
            "--output",
            type=Path,
            default=processed / "branch_product_month_demand_history_features.parquet",
        )

    def handle(self, *args, **options):
        try:
            result = build_demand_history_features(
                pd.read_parquet(options["input"]), pd.read_parquet(options["unresolved_audit"])
            )
        except (OSError, ValueError, DemandHistoryFeatureValidationError) as error:
            raise CommandError(str(error)) from error

        write_demand_history_features(result, options["output"])
        metrics = result.metrics
        self.stdout.write(self.style.SUCCESS("Demand-history feature build complete."))
        self.stdout.write(
            f"rows={metrics.feature_rows}; branches={metrics.unique_branches}; products={metrics.unique_products}; "
            f"months={metrics.minimum_year_month}..{metrics.maximum_year_month}"
        )
        self.stdout.write(
            f"lag_1={metrics.lag_1_available_rows}; lag_2={metrics.lag_2_available_rows}; "
            f"lag_3={metrics.lag_3_available_rows}; full_history={metrics.full_three_month_history_rows}; "
            f"intermittency={metrics.minimum_intermittency}..{metrics.maximum_intermittency}; "
            f"calendar_gap_edges={metrics.calendar_gap_edges}; unresolved_excluded={metrics.unresolved_rows_excluded}"
        )
        self.stdout.write(f"output={options['output']}")
