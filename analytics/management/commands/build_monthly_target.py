"""Build the Step-03 monthly fulfilled-sales proxy target."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.services.targets import (
    TARGET_NAME,
    TargetBuildValidationError,
    build_monthly_fulfilled_positive_units_target,
    write_monthly_target,
)


class Command(BaseCommand):
    help = "Build the monthly observed-positive-fulfilled-sales proxy target."

    def add_arguments(self, parser):
        processed = settings.BASE_DIR / "data" / "processed"
        parser.add_argument("--input", type=Path, default=processed / "analytical_day.parquet")
        parser.add_argument("--unresolved-audit", type=Path, default=processed / "sales_unresolved.parquet")
        parser.add_argument("--output", type=Path, default=processed / "monthly_fulfilled_positive_units_target.parquet")

    def handle(self, *args, **options):
        try:
            result = build_monthly_fulfilled_positive_units_target(
                pd.read_parquet(options["input"]), pd.read_parquet(options["unresolved_audit"])
            )
        except (OSError, ValueError, TargetBuildValidationError) as error:
            raise CommandError(str(error)) from error

        write_monthly_target(result, options["output"])
        metrics = result.metrics
        self.stdout.write(self.style.SUCCESS("Monthly target build complete."))
        self.stdout.write(
            f"rows={metrics.target_rows}; total_{TARGET_NAME}={metrics.total_target_units}; "
            f"range={metrics.minimum_target}..{metrics.maximum_target}; zeros={metrics.zero_target_rows}"
        )
        self.stdout.write(
            f"source_observations: positive={metrics.positive_source_observations}; "
            f"zero={metrics.zero_source_observations}; negative={metrics.negative_source_observations}"
        )
        self.stdout.write(
            f"unresolved_excluded: rows={metrics.unresolved_excluded_rows}; "
            f"positive_units={metrics.unresolved_excluded_positive_units}"
        )
        self.stdout.write(f"output={options['output']}")
