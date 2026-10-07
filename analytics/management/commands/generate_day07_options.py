"""Persist the finite synthetic Day-07 option menu without optimization."""

from datetime import date

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from planning.services.option_generator import build_option_evidence_scores, generate_finite_options, persist_decision_options
from planning.services.synthetic_state import load_synthetic_options


class Command(BaseCommand):
    help = "Generate finite synthetic L1/q1 planning possibilities; never selects one."

    def handle(self, *args, **options):
        root = settings.BASE_DIR
        fixture = pd.read_csv(root / "config" / "synthetic" / "l0_q0_demo.csv")
        config = load_synthetic_options(root / "config" / "synthetic" / "options.yaml")
        result = generate_finite_options(fixture, config)
        candidates = pd.read_parquet(root / "data" / "artifacts" / "day07" / "candidate_results.parquet")
        feature_month = pd.read_parquet(root / "data" / "processed" / "feature_month.parquet")
        planning_month = date.fromisoformat(str(candidates["planning_month"].iloc[0]) + "-01")
        scenario_version = "day07_synthetic_options_v1"
        evidence_scores = build_option_evidence_scores(fixture, candidates, feature_month)
        result = persist_decision_options(result, planning_month=planning_month, scenario_version=scenario_version, evidence_scores=evidence_scores)
        path = root / "data" / "artifacts" / "day07" / "decision_options.parquet"
        result.to_parquet(path, index=False)
        self.stdout.write(f"options={len(result)}; evidence_pairs={len(evidence_scores)}; scenario_version={scenario_version}; path={path}")
