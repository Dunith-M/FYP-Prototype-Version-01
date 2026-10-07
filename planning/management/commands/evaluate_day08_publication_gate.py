"""Evaluate the server-side Day-08 action/publication gate without publishing."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from planning.services.publication_gate import evaluate_publication_gate


class Command(BaseCommand):
    help = "Evaluate whether an OptimizationResult may proceed to the action engine."

    def add_arguments(self, parser):
        parser.add_argument("--result-id", required=True, type=int)

    def handle(self, *args, **options):
        try:
            gate = evaluate_publication_gate(options["result_id"])
        except Exception as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(
            "result_id={id} allowed={allowed} code={code} reason={reason} plan_fingerprint={fingerprint}".format(
                id=gate.optimization_result_id,
                allowed=gate.allowed,
                code=gate.code,
                reason=" ".join(gate.reasons),
                fingerprint=gate.plan_fingerprint,
            )
        )
