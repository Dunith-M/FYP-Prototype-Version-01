"""Register immutable metadata for the current contracted raw source files."""

from django.core.management.base import BaseCommand

from ingestion.services.source_registration import register_contracted_source_files


class Command(BaseCommand):
    help = (
        "Register actual file hash, workbook row count, and contract version for "
        "each contracted raw source. This does not alter raw files or validate DQ."
    )

    def handle(self, *args, **options):
        results = register_contracted_source_files()
        for result in results:
            action = "registered" if result.created else "already registered"
            source = result.source_file
            self.stdout.write(
                f"{result.dataset}: {action}; {source.filename}; "
                f"rows={source.row_count}; sha256={source.file_hash}; "
                f"contract={source.contract_version}; status={source.status}"
            )
