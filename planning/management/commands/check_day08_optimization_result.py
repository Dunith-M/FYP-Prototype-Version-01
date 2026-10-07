"""Run the independent Python feasibility checker for one persisted result."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from optimization.checker import CheckerInputError, check_and_persist_feasibility


class Command(BaseCommand):
    help = "Independently check persisted Day-08 selected options and persist PASS or FAIL."

    def add_arguments(self, parser):
        parser.add_argument("--result-id", required=True, type=int)

    def handle(self, *args, **options):
        try:
            report = check_and_persist_feasibility(options["result_id"])
        except CheckerInputError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            "result_id={id} solver_status={solver_status} feasibility_status={status} "
            "selected_options={options} decision_units={units}".format(
                id=report.optimization_result_id,
                solver_status=report.solver_status,
                status=report.status,
                options=report.checked_option_count,
                units=report.checked_unit_count,
            )
        )
        for rule in report.rules:
            self.stdout.write(
                "{code} | {status} | {name} | observed={observed!r} | allowed={allowed!r} | {message}".format(
                    code=rule.rule_code,
                    status=rule.status,
                    name=rule.rule_name,
                    observed=rule.observed_value,
                    allowed=rule.allowed_value,
                    message=rule.message,
                )
            )
