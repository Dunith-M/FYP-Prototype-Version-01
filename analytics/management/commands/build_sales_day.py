"""Build the canonical daily sales fact without any inventory processing."""

from __future__ import annotations

import math
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.sales_day import (
    SalesDayValidationError,
    build_sales_day,
    write_sales_day_outputs,
)


class Command(BaseCommand):
    help = "Aggregate resolved Sales rows to branch, canonical-product, business-date grain."

    def add_arguments(self, parser):
        parser.add_argument(
            "--sales-file",
            type=Path,
            default=settings.BASE_DIR / "data" / "raw" / "transactions_data.xlsx",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=settings.BASE_DIR / "data" / "processed" / "sales_day.parquet",
        )
        parser.add_argument(
            "--unresolved-output",
            type=Path,
            default=settings.BASE_DIR / "data" / "processed" / "sales_unresolved.parquet",
        )

    def handle(self, *args, **options):
        try:
            result = build_sales_day(options["sales_file"])
        except (SalesDayValidationError, OSError, ValueError) as error:
            raise CommandError(str(error)) from error

        metrics = result.metrics
        if not metrics.rows_reconciled:
            raise CommandError("Sales aggregation row reconciliation failed.")
        if not math.isclose(metrics.unit_difference, 0.0, abs_tol=1e-9):
            raise CommandError(
                f"Sales aggregation unit reconciliation failed: {metrics.unit_difference}."
            )
        if not math.isclose(metrics.value_difference, 0.0, abs_tol=1e-9):
            raise CommandError(
                f"Sales aggregation value reconciliation failed: {metrics.value_difference}."
            )

        write_sales_day_outputs(
            result,
            options["output"],
            options["unresolved_output"],
        )
        self.stdout.write(self.style.SUCCESS("SALES_DAY build complete."))
        self.stdout.write(
            f"Source rows: total={metrics.total_source_rows}; "
            f"resolved={metrics.resolved_product_rows}; "
            f"unresolved={metrics.unresolved_product_rows}"
        )
        self.stdout.write(
            f"SALES_DAY rows={len(result.sales_day)}; "
            f"duplicate_keys={metrics.duplicate_sales_day_keys}"
        )
        self.stdout.write(
            f"Signed units: raw_resolved={metrics.raw_resolved_signed_units}; "
            f"sales_day={metrics.aggregated_sales_units}; "
            f"difference={metrics.unit_difference}"
        )
        self.stdout.write(
            f"Total value: raw_resolved={metrics.raw_resolved_total_value}; "
            f"sales_day={metrics.aggregated_sales_value}; "
            f"difference={metrics.value_difference}"
        )
        self.stdout.write(f"SALES_DAY output: {options['output']}")
        self.stdout.write(f"Unresolved sales audit output: {options['unresolved_output']}")
