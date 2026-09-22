"""Build INVENTORY_DAY and BRANCH_PRODUCT_DAY with safe unknown-inventory handling."""

from __future__ import annotations

import math
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.branch_product_day import (
    InventoryDayValidationError,
    build_branch_product_day,
    build_inventory_day,
    write_branch_product_day,
)


class Command(BaseCommand):
    help = "Build validated inventory snapshots and left-join them to SALES_DAY."

    def add_arguments(self, parser):
        processed = settings.BASE_DIR / "data" / "processed"
        raw = settings.BASE_DIR / "data" / "raw"
        parser.add_argument("--inventory-file", type=Path, default=raw / "inventory_data.xlsx")
        parser.add_argument("--sales-day", type=Path, default=processed / "sales_day.parquet")
        parser.add_argument("--inventory-day", type=Path, default=processed / "inventory_day.parquet")
        parser.add_argument(
            "--inventory-unresolved",
            type=Path,
            default=processed / "inventory_unresolved.parquet",
        )
        parser.add_argument(
            "--output", type=Path, default=processed / "branch_product_day.parquet"
        )

    def handle(self, *args, **options):
        try:
            inventory = build_inventory_day(
                options["inventory_file"],
                options["inventory_day"],
                options["inventory_unresolved"],
            )
            result = build_branch_product_day(options["sales_day"], options["inventory_day"])
        except (InventoryDayValidationError, OSError, ValueError) as error:
            raise CommandError(str(error)) from error

        metrics = result.metrics
        if len(result.branch_product_day) != metrics.sales_day_rows:
            raise CommandError("Left join did not preserve SALES_DAY row count.")
        if not math.isclose(metrics.unit_difference, 0.0, abs_tol=1e-9):
            raise CommandError(f"Sales-unit reconciliation failed: {metrics.unit_difference}.")
        if not math.isclose(metrics.value_difference, 0.0, abs_tol=1e-9):
            raise CommandError(f"Sales-value reconciliation failed: {metrics.value_difference}.")

        write_branch_product_day(result, options["output"])
        self.stdout.write(self.style.SUCCESS("BRANCH_PRODUCT_DAY build complete."))
        self.stdout.write(
            f"Inventory rows: source={inventory.metrics.source_inventory_rows}; "
            f"resolved={inventory.metrics.resolved_inventory_rows}; "
            f"unresolved_products={inventory.metrics.unresolved_product_rows}; "
            f"duplicate_keys={inventory.metrics.duplicate_inventory_keys}"
        )
        self.stdout.write(
            f"Join rows: sales_day={metrics.sales_day_rows}; "
            f"branch_product_day={metrics.branch_product_day_rows}; "
            f"matched={metrics.inventory_match_rows}; unknown={metrics.unknown_inventory_rows}"
        )
        self.stdout.write(
            f"Sales reconciliation: units_difference={metrics.unit_difference}; "
            f"value_difference={metrics.value_difference}"
        )
        self.stdout.write(f"INVENTORY_DAY output: {options['inventory_day']}")
        self.stdout.write(f"BRANCH_PRODUCT_DAY output: {options['output']}")
