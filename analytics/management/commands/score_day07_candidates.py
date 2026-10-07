"""Score persisted Day-07 Track-B research candidates without planning actions."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from catalog.models import Product
from ml.candidates.candidate_scoring import score_candidate_evaluations


class Command(BaseCommand):
    help = "Score existing Day-07 research candidates and persist deterministic Top-N ranks."

    def add_arguments(self, parser):
        parser.add_argument("--top-n", type=int, default=20)
        parser.add_argument("--candidate-results", type=Path, default=settings.BASE_DIR / "data" / "artifacts" / "day07" / "candidate_results.parquet")
        parser.add_argument("--feature-month", type=Path, default=settings.BASE_DIR / "data" / "processed" / "feature_month.parquet")
        parser.add_argument("--output-directory", type=Path, default=settings.BASE_DIR / "data" / "artifacts" / "day07")

    def handle(self, *args, **options):
        for label, path in (("Candidate results", options["candidate_results"]), ("Frozen feature artifact", options["feature_month"])):
            if not path.is_file():
                raise CommandError(f"{label} was not found: {path}")
        product_prices = pd.DataFrame.from_records(Product.objects.values("canonical_product_id", "list_price", "price_valid"))
        result = score_candidate_evaluations(
            pd.read_parquet(options["candidate_results"]),
            pd.read_parquet(options["feature_month"]),
            product_prices,
            top_n=options["top_n"],
        )
        output_directory = options["output_directory"]
        output_directory.mkdir(parents=True, exist_ok=True)
        result.evaluations.to_parquet(output_directory / "candidate_results.parquet", index=False)
        result.branch_summary.to_csv(output_directory / "candidate_summary.csv", index=False)
        self.stdout.write("DAY 07 — CANDIDATE SCORING CHECKPOINT")
        for row in result.branch_summary.itertuples(index=False):
            self.stdout.write(f"{row.branch_code}: scored_candidate_count={row.scored_candidate_count}; shortlisted_count={row.shortlisted_count}")
