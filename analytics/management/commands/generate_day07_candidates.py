"""Build Day-07 research candidate pools from existing persisted evidence."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from catalog.models import Branch, Product, SkuMapping
from ml.candidates.generate_candidates import build_candidate_pools, write_candidate_pool_artifacts


class Command(BaseCommand):
    help = "Generate Day-07 Track-B research candidate pools and Top-N shortlists."

    def add_arguments(self, parser):
        parser.add_argument("--top-n", type=int, default=20)
        parser.add_argument(
            "--feature-month",
            type=Path,
            default=settings.BASE_DIR / "data" / "processed" / "feature_month.parquet",
        )
        parser.add_argument(
            "--output-directory",
            type=Path,
            default=settings.BASE_DIR / "data" / "artifacts" / "day07",
        )

    def handle(self, *args, **options):
        feature_path = options["feature_month"]
        if not feature_path.is_file():
            raise CommandError(f"Frozen feature artifact was not found: {feature_path}")
        products = pd.DataFrame.from_records(
            Product.objects.values("canonical_product_id", "product_desc", "family_id", "family_desc", "price_valid")
        )
        branches = pd.DataFrame.from_records(Branch.objects.values("branch_code"))
        confirmed_mappings = pd.DataFrame.from_records(
            SkuMapping.objects.filter(
                confidence_class=SkuMapping.ConfidenceClass.CONFIRMED,
                canonical_product__isnull=False,
            ).values("canonical_product__canonical_product_id")
        ).rename(columns={"canonical_product__canonical_product_id": "canonical_product_id"})
        result = build_candidate_pools(
            products,
            branches,
            confirmed_mappings,
            pd.read_parquet(feature_path),
            top_n=options["top_n"],
        )
        write_candidate_pool_artifacts(result, options["output_directory"])
        self.stdout.write("DAY 07 — CANDIDATE POOL CHECKPOINT")
        for row in result.branch_summary.itertuples(index=False):
            self.stdout.write(
                f"{row.branch_code}: candidate_pool_count={row.candidate_pool_count}; shortlisted_count={row.shortlisted_count}"
            )
