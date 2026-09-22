"""Generate the current run-specific data-quality disposition report."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.services.dq_disposition import (
    DqDispositionValidationError,
    build_dq_disposition_report,
    load_feature_manifest,
    write_dq_disposition_report,
)
from catalog.models import Product
from ingestion.models import SourceFile


class Command(BaseCommand):
    help = "Create the run-specific Day-04 DQ disposition report from existing outputs."

    def add_arguments(self, parser):
        processed = settings.BASE_DIR / "data" / "processed"
        artifacts = settings.BASE_DIR / "data" / "artifacts"
        parser.add_argument("--sales-day", type=Path, default=processed / "sales_day.parquet")
        parser.add_argument("--sales-unresolved", type=Path, default=processed / "sales_unresolved.parquet")
        parser.add_argument("--inventory-day", type=Path, default=processed / "inventory_day.parquet")
        parser.add_argument("--inventory-unresolved", type=Path, default=processed / "inventory_unresolved.parquet")
        parser.add_argument("--manifest", type=Path, default=artifacts / "feature_month_manifest.json")
        parser.add_argument("--output", type=Path, default=artifacts / "day04_dq_disposition.csv")

    def handle(self, *args, **options):
        raw = settings.BASE_DIR / "data" / "raw"
        source_file_ids = {
            dataset: SourceFile.objects.filter(dataset=dataset, status=SourceFile.Status.VALIDATED)
            .order_by("-uploaded_at").values_list("id", flat=True).first()
            for dataset in SourceFile.Dataset.values
        }
        try:
            result = build_dq_disposition_report(
                sales_day=pd.read_parquet(options["sales_day"]),
                sales_unresolved=pd.read_parquet(options["sales_unresolved"]),
                inventory_day=pd.read_parquet(options["inventory_day"]),
                inventory_unresolved=pd.read_parquet(options["inventory_unresolved"]),
                products=pd.DataFrame.from_records(Product.objects.values("product_code", "price_valid")),
                calendar=pd.read_excel(raw / "calendar_data.xlsx", sheet_name="Sheet1"),
                branches=pd.read_excel(raw / "branch_profile.xlsx", sheet_name="Branch Profile"),
                feature_manifest=load_feature_manifest(options["manifest"]),
                source_file_ids=source_file_ids,
            )
            write_dq_disposition_report(result, options["output"])
        except (OSError, ValueError, DqDispositionValidationError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(self.style.SUCCESS(f"DQ disposition report created: {options['output']}"))
        self.stdout.write(f"run_id={result.run_id}; datasets={len(result.report)}")
