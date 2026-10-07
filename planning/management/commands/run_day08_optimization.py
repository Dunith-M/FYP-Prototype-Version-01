"""Run and persist the bounded Day-08 synthetic CP-SAT demo."""

from __future__ import annotations

from datetime import date

from django.core.management.base import BaseCommand, CommandError

from optimization.runner import DEFAULT_MAX_TIME_SECONDS, run_and_persist_optimization


class Command(BaseCommand):
    help = "Run the validated Day-08 synthetic CP-SAT model and persist only OptimizationResult."

    def add_arguments(self, parser):
        parser.add_argument("--planning-month", required=True, help="Planning month in YYYY-MM format.")
        parser.add_argument("--scenario-version", required=True, help="Persisted DecisionOption scenario version.")
        parser.add_argument("--run-version", required=True, help="Explicit idempotent OptimizationResult run version.")
        parser.add_argument("--max-time-seconds", type=float, default=DEFAULT_MAX_TIME_SECONDS)

    def handle(self, *args, **options):
        try:
            planning_month = date.fromisoformat(f"{options['planning_month']}-01")
        except ValueError as exc:
            raise CommandError("--planning-month must use YYYY-MM format.") from exc
        if options["max_time_seconds"] <= 0:
            raise CommandError("--max-time-seconds must be positive.")

        outcome = run_and_persist_optimization(
            planning_month=planning_month,
            scenario_version=options["scenario_version"],
            run_version=options["run_version"],
            max_time_seconds=options["max_time_seconds"],
        )
        result = outcome.optimization_result
        self.stdout.write(
            self.style.SUCCESS(
                "optimization_result_id={id} status={status} selected_options={selected} "
                "objective_value={objective} feasibility_status={feasibility}".format(
                    id=result.pk,
                    status=result.solver_status,
                    selected=result.selected_options.count(),
                    objective=result.objective_value,
                    feasibility=result.feasibility_status,
                )
            )
        )
