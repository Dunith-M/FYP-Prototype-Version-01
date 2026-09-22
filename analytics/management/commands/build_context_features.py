"""Build Step-08 product, calendar, and branch context features."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.services.context_features import (
    ContextFeatureValidationError,
    enrich_monthly_features,
    load_validated_calendar_daily,
    write_context_features,
)
from catalog.models import Branch, Product


class Command(BaseCommand):
    help = "Enrich monthly features with validated product, calendar, and branch context."

    def add_arguments(self, parser):
        processed = settings.BASE_DIR / "data" / "processed"
        raw = settings.BASE_DIR / "data" / "raw"
        parser.add_argument("--input", type=Path, default=processed / "branch_product_month_inventory_evidence_features.parquet")
        parser.add_argument("--calendar", type=Path, default=raw / "calendar_data.xlsx")
        parser.add_argument("--output", type=Path, default=processed / "branch_product_month_context_features.parquet")
        parser.add_argument("--calendar-output", type=Path, default=processed / "calendar_month_features.parquet")

    def handle(self, *args, **options):
        product_master = pd.DataFrame.from_records(
            Product.objects.values("canonical_product_id", "family_id", "family_desc", "list_price", "price_valid")
        )
        branch_master = pd.DataFrame.from_records(Branch.objects.values("branch_code", "branch_name"))
        try:
            result = enrich_monthly_features(
                pd.read_parquet(options["input"]),
                product_master,
                load_validated_calendar_daily(options["calendar"]),
                branch_master,
            )
        except (OSError, ValueError, ContextFeatureValidationError) as error:
            raise CommandError(str(error)) from error
        write_context_features(result, options["output"], options["calendar_output"])
        metrics = result.metrics
        self.stdout.write(self.style.SUCCESS("Context feature build complete."))
        self.stdout.write(
            f"rows={metrics.input_rows}->{metrics.output_rows}; products={metrics.unique_products}; "
            f"branches={metrics.unique_branches}; months={metrics.months}; invalid_price={metrics.invalid_price_rows}; "
            f"missing_family={metrics.missing_family_rows}"
        )
        self.stdout.write(f"output={options['output']}; calendar_output={options['calendar_output']}")
