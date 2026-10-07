"""Load the finalized Day-07 candidate artifact into PostgreSQL."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from analytics.services.candidate_result_loader import load_candidate_results


class Command(BaseCommand):
    help = "Idempotently load finalized Day-07 research candidates into CandidateResult."

    def add_arguments(self, parser):
        parser.add_argument(
            "--artifact",
            type=Path,
            default=settings.BASE_DIR / "data" / "artifacts" / "day07" / "candidate_results.parquet",
        )

    def handle(self, *args, **options):
        artifact = options["artifact"]
        if not artifact.is_file():
            raise CommandError(f"Final candidate artifact was not found: {artifact}")
        try:
            created, updated = load_candidate_results(pd.read_parquet(artifact))
        except (ValueError, TypeError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(f"CandidateResult load complete: created={created}; updated={updated}")
