"""Persist final Day-07 branch/product/planning-month research results."""

import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from ml.candidates.candidate_results import build_final_candidate_results


class Command(BaseCommand):
    help = "Finalize scored Day-07 candidates with a Track-A anti-join."

    def handle(self, *args, **options):
        root = settings.BASE_DIR
        output = root / "data" / "artifacts" / "day07"
        results_path = output / "candidate_results.parquet"
        feature_path = root / "data" / "processed" / "feature_month.parquet"
        manifest_path = root / "data" / "artifacts" / "feature_month_manifest.json"
        if not all(path.is_file() for path in (results_path, feature_path, manifest_path)):
            raise CommandError("Candidate results, feature artifact, or feature manifest is missing.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cutoff = pd.Period(str(manifest["planning_cutoff"])[:7], freq="M")
        final = build_final_candidate_results(pd.read_parquet(results_path), pd.read_parquet(feature_path), planning_month=str(cutoff + 1), source_month=str(cutoff), top_n=20)
        pd.read_parquet(results_path).to_parquet(output / "candidate_scoring_evaluations.parquet", index=False)
        final.results.to_parquet(results_path, index=False)
        final.summary.to_csv(output / "candidate_summary.csv", index=False)
        self.stdout.write(f"planning_month={cutoff + 1}; rows={len(final.results)}; track_a_overlap_count=0")
