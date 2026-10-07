"""Generate the explicit Day-07 synthetic L0/q0 demo fixture."""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from catalog.models import Branch, Product
from planning.services.synthetic_state import build_synthetic_l0_q0_fixture, load_synthetic_options, validate_synthetic_l0_q0_fixture


class Command(BaseCommand):
    help = "Generate a deterministic synthetic L0/q0 demo fixture; never uses inventory."

    def handle(self, *args, **options):
        root = settings.BASE_DIR
        config = root / "config" / "synthetic"
        fixture_path = config / "l0_q0_demo.csv"
        candidate_results = pd.read_parquet(root / "data" / "artifacts" / "day07" / "candidate_results.parquet")
        fixture = build_synthetic_l0_q0_fixture(pd.read_parquet(root / "data" / "processed" / "feature_month.parquet"), candidate_results, load_synthetic_options(config / "options.yaml"))
        validate_synthetic_l0_q0_fixture(fixture, load_synthetic_options(config / "options.yaml"), branch_codes=set(Branch.objects.values_list("branch_code", flat=True)), product_ids=set(Product.objects.values_list("canonical_product_id", flat=True)), candidate_ids=set(candidate_results.loc[candidate_results["branch_code"].eq("BIL01"), "canonical_product_id"]))
        fixture.to_csv(fixture_path, index=False)
        self.stdout.write(f"fixture={fixture_path}; rows={len(fixture)}")
