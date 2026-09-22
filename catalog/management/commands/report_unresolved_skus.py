import csv
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from catalog.sku_mapping import (
    SalesSkuDataValidationError,
    build_unresolved_sales_queue,
)


class Command(BaseCommand):
    help = "Report unresolved Sales SKU mappings without changing their review state."

    def add_arguments(self, parser):
        raw_directory = settings.BASE_DIR / "data" / "raw"
        parser.add_argument(
            "--sales-file",
            type=Path,
            default=raw_directory / "transactions_data.xlsx",
            help="Path to the historical sales workbook.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=settings.BASE_DIR
            / "data"
            / "artifacts"
            / "mapping"
            / "unresolved_sku_queue.csv",
            help="CSV path for the unresolved queue report.",
        )
        parser.add_argument(
            "--top",
            type=int,
            default=10,
            help="Number of highest-impact unresolved codes to print.",
        )

    @staticmethod
    def _write_csv(output_path: Path, report) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "source_system",
            "source_product_code",
            "mapping_reason",
            "confidence_class",
            "canonical_product_id",
            "affected_sales_rows",
            "affected_signed_units",
            "affected_absolute_units",
            "reviewer",
            "reviewed_at",
            "notes",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=fieldnames)
            writer.writeheader()
            for item in report.items:
                mapping = item.mapping
                writer.writerow(
                    {
                        "source_system": mapping.source_system,
                        "source_product_code": mapping.source_product_code,
                        "mapping_reason": mapping.mapping_reason,
                        "confidence_class": mapping.confidence_class,
                        "canonical_product_id": "",
                        "affected_sales_rows": item.affected_sales_rows,
                        "affected_signed_units": item.affected_signed_units,
                        "affected_absolute_units": item.affected_absolute_units,
                        "reviewer": mapping.reviewer,
                        "reviewed_at": mapping.reviewed_at or "",
                        "notes": mapping.notes,
                    }
                )

    def handle(self, *args, **options):
        try:
            report = build_unresolved_sales_queue(options["sales_file"])
        except (SalesSkuDataValidationError, OSError) as error:
            raise CommandError(str(error)) from error

        output_path = options["output"]
        self._write_csv(output_path, report)

        self.stdout.write(
            self.style.SUCCESS("Unresolved Sales SKU queue report created.")
        )
        self.stdout.write(f"Unresolved mappings: {report.unresolved_mapping_count}")
        self.stdout.write(f"Affected sales rows: {report.affected_sales_rows}")
        self.stdout.write(f"Affected signed units: {report.affected_signed_units}")
        self.stdout.write(
            f"Missing unresolved source codes: {list(report.missing_unresolved_source_codes)}"
        )
        self.stdout.write(
            f"Extra unresolved mapping codes: {list(report.extra_unresolved_mapping_codes)}"
        )
        self.stdout.write(f"CSV: {output_path}")
        for item in report.items[: max(options["top"], 0)]:
            self.stdout.write(
                f"{item.mapping.source_product_code}: rows={item.affected_sales_rows}; "
                f"signed_units={item.affected_signed_units}; "
                f"absolute_units={item.affected_absolute_units}"
            )
