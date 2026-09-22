"""Build Step-09 next-month supervised targets from processed monthly features."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.services.target_alignment import (
    NextMonthTargetValidationError,
    align_next_month_target,
    write_next_month_targets,
)


class Command(BaseCommand):
    help = "Create safe calendar-consecutive next-month targets for monthly feature rows."

    def add_arguments(self, parser):
        processed = settings.BASE_DIR / "data" / "processed"
        parser.add_argument("--input", type=Path, default=processed / "branch_product_month_context_features.parquet")
        parser.add_argument("--output", type=Path, default=processed / "branch_product_month_supervised.parquet")

    def handle(self, *args, **options):
        try:
            result = align_next_month_target(pd.read_parquet(options["input"]))
        except (OSError, ValueError, NextMonthTargetValidationError) as error:
            raise CommandError(str(error)) from error
        write_next_month_targets(result, options["output"])
        metrics = result.metrics
        self.stdout.write(self.style.SUCCESS("Next-month target alignment complete."))
        self.stdout.write(
            f"rows={metrics.total_feature_rows}; valid_target={metrics.valid_target_rows}; "
            f"without_target={metrics.rows_without_target}; non_consecutive={metrics.non_consecutive_month_pairs}; "
            f"negative_target={metrics.negative_target_values}"
        )
        self.stdout.write(f"output={options['output']}")
