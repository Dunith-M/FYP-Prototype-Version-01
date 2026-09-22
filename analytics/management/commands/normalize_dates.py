"""Report reproducible, source-preserving normalized-date metrics."""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.date_utils import DateSourceValidationError, normalize_date_sources


class Command(BaseCommand):
    help = (
        "Normalize and validate Sales, Inventory, and Calendar dates without "
        "changing raw workbooks or creating analytical joins."
    )

    def add_arguments(self, parser):
        raw_dir = settings.BASE_DIR / "data" / "raw"
        parser.add_argument(
            "--sales-file", type=Path, default=raw_dir / "transactions_data.xlsx"
        )
        parser.add_argument(
            "--inventory-file", type=Path, default=raw_dir / "inventory_data.xlsx"
        )
        parser.add_argument(
            "--calendar-file", type=Path, default=raw_dir / "calendar_data.xlsx"
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=settings.BASE_DIR
            / "data"
            / "artifacts"
            / "date_normalization"
            / "date_normalization_report.json",
            help="Deterministic JSON validation report path.",
        )

    def handle(self, *args, **options):
        try:
            report = normalize_date_sources(
                options["sales_file"],
                options["inventory_file"],
                options["calendar_file"],
            )
        except (DateSourceValidationError, OSError) as error:
            raise CommandError(str(error)) from error

        output_path = options["output"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        self.stdout.write(self.style.SUCCESS("Date normalization validation complete."))
        for profile, label in (
            (report.sales, "Sales / business_date"),
            (report.inventory, "Inventory / snapshot_date"),
            (report.calendar, "Calendar / calendar_date"),
        ):
            self.stdout.write(
                f"{label}: rows={profile.total_rows}; parsed={profile.parsed_dates}; "
                f"null={profile.null_dates}; unparseable={profile.unparseable_dates}; "
                f"range={profile.minimum_date}..{profile.maximum_date}; "
                f"unique_dates={profile.unique_date_count}"
            )
        self.stdout.write(
            "Calendar coverage: "
            f"sales_missing_dates={len(report.sales_dates_missing_from_calendar)}; "
            f"sales_rows_affected={report.sales_rows_missing_from_calendar}; "
            f"inventory_missing_dates={len(report.inventory_dates_missing_from_calendar)}; "
            f"inventory_rows_affected={report.inventory_rows_missing_from_calendar}"
        )
        self.stdout.write(f"Report written: {output_path}")
