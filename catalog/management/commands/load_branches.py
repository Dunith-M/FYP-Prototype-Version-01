from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from catalog.branch_master import (
    EXPECTED_BRANCH_CODES,
    BranchDataValidationError,
    coverage_report,
    inspect_branch_profile,
)
from catalog.models import Branch


class Command(BaseCommand):
    help = "Load and validate the official Branch master from the raw branch profile."

    def add_arguments(self, parser):
        raw_directory = settings.BASE_DIR / "data" / "raw"
        parser.add_argument(
            "--branch-profile",
            type=Path,
            default=raw_directory / "branch_profile.xlsx",
            help="Path to the branch profile workbook.",
        )
        parser.add_argument(
            "--sales",
            type=Path,
            default=raw_directory / "transactions_data.xlsx",
            help="Path to the sales workbook used for SHOP_CODE coverage validation.",
        )
        parser.add_argument(
            "--inventory",
            type=Path,
            default=raw_directory / "inventory_data.xlsx",
            help="Path to the inventory workbook used for SITE_ID coverage validation.",
        )

    @staticmethod
    def _fail_for_branch_profile(validation) -> None:
        if validation.is_valid:
            return
        raise CommandError(
            "Branch profile validation failed: "
            f"null_codes={validation.null_codes}; "
            f"duplicate_codes={validation.duplicate_codes}; "
            f"missing_expected_codes={validation.missing_expected_codes}; "
            f"unexpected_codes={validation.unexpected_codes}; "
            f"blank_names={validation.blank_names}."
        )

    @staticmethod
    def _fail_for_coverage(dataset: str, report) -> None:
        if report.is_valid:
            return
        raise CommandError(
            f"{dataset} branch coverage failed: unmatched_codes={report.unmatched_codes}; "
            f"rows_affected={report.rows_affected_by_unmatched_codes}."
        )

    def handle(self, *args, **options):
        profile_path = options["branch_profile"]
        sales_path = options["sales"]
        inventory_path = options["inventory"]

        try:
            validation = inspect_branch_profile(profile_path)
            self._fail_for_branch_profile(validation)

            # Validate source coverage before any Branch rows are written.
            source_codes = {record.branch_code for record in validation.records}
            sales_preflight = coverage_report(sales_path, "SHOP_CODE", source_codes)
            inventory_preflight = coverage_report(inventory_path, "SITE_ID", source_codes)
            self._fail_for_coverage("Sales", sales_preflight)
            self._fail_for_coverage("Inventory", inventory_preflight)
        except (BranchDataValidationError, OSError) as error:
            raise CommandError(str(error)) from error

        with transaction.atomic():
            existing_codes = set(Branch.objects.values_list("branch_code", flat=True))
            existing_unexpected = sorted(existing_codes - EXPECTED_BRANCH_CODES)
            if existing_unexpected:
                raise CommandError(
                    "Branch master already contains unexpected codes and will not be changed: "
                    f"{existing_unexpected}."
                )

            for record in validation.records:
                Branch.objects.update_or_create(
                    branch_code=record.branch_code,
                    defaults={"branch_name": record.branch_name},
                )

            master_codes = set(Branch.objects.values_list("branch_code", flat=True))
            missing_codes = sorted(EXPECTED_BRANCH_CODES - master_codes)
            unexpected_codes = sorted(master_codes - EXPECTED_BRANCH_CODES)
            if len(master_codes) != 10 or missing_codes or unexpected_codes:
                raise CommandError(
                    "Branch master validation failed after load: "
                    f"count={len(master_codes)}; missing_codes={missing_codes}; "
                    f"unexpected_codes={unexpected_codes}."
                )

            # The source validation requires source_codes to be the exact expected
            # master set, and the check above proves the persisted Branch set is
            # identical. The preflight reports therefore validate coverage against
            # Branch.branch_code without a second full scan of the large raw files.
            if master_codes != source_codes:
                raise CommandError(
                    "Persisted Branch codes differ from the validated source codes; "
                    "coverage cannot be accepted."
                )
            sales_coverage = sales_preflight
            inventory_coverage = inventory_preflight

        self.stdout.write(self.style.SUCCESS("Branch master loaded and validated."))
        self.stdout.write(f"Branch count: {len(master_codes)}")
        self.stdout.write(f"Sales coverage: {sales_coverage}")
        self.stdout.write(f"Inventory coverage: {inventory_coverage}")
