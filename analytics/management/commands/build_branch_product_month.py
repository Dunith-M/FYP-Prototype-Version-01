"""Build the Step-04 branch-product-month analytical dataset."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.services.monthly_aggregation import (
    MonthlyAggregationValidationError,
    build_branch_product_month,
    write_branch_product_month,
)


class Command(BaseCommand):
    help = "Aggregate processed BRANCH_PRODUCT_DAY evidence to observed monthly rows."

    def add_arguments(self, parser):
        processed = settings.BASE_DIR / "data" / "processed"
        parser.add_argument("--input", type=Path, default=processed / "branch_product_day.parquet")
        parser.add_argument("--unresolved-audit", type=Path, default=processed / "sales_unresolved.parquet")
        parser.add_argument("--output", type=Path, default=processed / "branch_product_month.parquet")

    def handle(self, *args, **options):
        try:
            result = build_branch_product_month(
                pd.read_parquet(options["input"]), pd.read_parquet(options["unresolved_audit"])
            )
        except (OSError, ValueError, MonthlyAggregationValidationError) as error:
            raise CommandError(str(error)) from error

        write_branch_product_month(result, options["output"])
        metrics = result.metrics
        self.stdout.write(self.style.SUCCESS("BRANCH_PRODUCT_MONTH build complete."))
        self.stdout.write(
            f"rows={metrics.monthly_rows}; branches={metrics.unique_branches}; "
            f"products={metrics.unique_products}; months={metrics.minimum_year_month}..{metrics.maximum_year_month}"
        )
        self.stdout.write(
            f"positive_units={metrics.total_positive_fulfilled_units}; "
            f"sales_observations={metrics.total_sales_observation_count}; "
            f"negative_events={metrics.total_negative_event_count}; "
            f"zero_events={metrics.total_zero_event_count}; "
            f"inventory_observations={metrics.total_inventory_observation_count}; "
            f"unresolved_excluded={metrics.unresolved_rows_excluded}; duplicates={metrics.duplicate_monthly_keys}"
        )
        self.stdout.write(f"output={options['output']}")
