"""Display persisted Day-08 selected decision states without deriving actions."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from planning.models import OptimizationResult


class Command(BaseCommand):
    help = "Show exact selected DecisionOptions for one persisted Day-08 OptimizationResult."

    def add_arguments(self, parser):
        parser.add_argument("--result-id", required=True, type=int)
        parser.add_argument("--limit", type=int, default=10, help="Maximum selected states to print; default 10.")

    def handle(self, *args, **options):
        if options["limit"] < 0:
            raise CommandError("--limit must be zero or greater.")
        try:
            result = OptimizationResult.objects.get(pk=options["result_id"])
        except OptimizationResult.DoesNotExist as exc:
            raise CommandError(f"OptimizationResult {options['result_id']} does not exist.") from exc

        selected_options = result.selected_options.select_related("branch", "product").order_by(
            "branch__branch_code", "product__canonical_product_id", "pk"
        )
        self.stdout.write(
            "result_id={id} solver_status={status} feasibility_status={feasibility} selected_options={count}".format(
                id=result.pk,
                status=result.solver_status,
                feasibility=result.feasibility_status,
                count=selected_options.count(),
            )
        )
        self.stdout.write("branch | product | option_id | L0 | q0 | L1 | q1 | proxy_utility | parameter_source")
        for option in selected_options[: options["limit"]]:
            self.stdout.write(
                "{branch} | {product} | {option_id} | {l0} | {q0} | {l1} | {q1} | {utility} | {source}".format(
                    branch=option.branch.branch_code,
                    product=option.product.canonical_product_id,
                    option_id=option.pk,
                    l0=int(option.l0),
                    q0="NULL" if option.q0 is None else option.q0,
                    l1=int(option.l1),
                    q1=option.q1,
                    utility=option.evidence_score,
                    source=option.parameter_source,
                )
            )
