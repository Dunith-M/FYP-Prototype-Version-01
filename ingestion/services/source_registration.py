"""Register immutable metadata for the contracted raw source files.

This service is deliberately for an explicit management operation.  Browser
requests to the Data Status page continue to read only ``SourceFile`` records;
they never open raw workbooks or calculate hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.db import transaction

from ingestion.models import SourceFile
from ingestion.validators import SOURCE_DATASETS, load_contract


@dataclass(frozen=True)
class RegisteredSource:
    """The persisted result for one actual, contracted raw source file."""

    dataset: str
    source_file: SourceFile
    created: bool


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_version(dataset: str, contract: dict[str, object]) -> str:
    version = contract.get("version")
    if version is None:
        raise ValueError(f"Contract for {dataset!r} does not declare a version.")
    return f"{dataset}-v{version}"


def register_contracted_source_files(
    raw_root: Path | None = None,
) -> list[RegisteredSource]:
    """Persist source metadata calculated from the real contracted workbooks.

    Metadata is gathered before database writes so a missing/unreadable source
    cannot leave a partial registration set.  Existing dataset/hash versions
    are preserved untouched, making repeated registrations idempotent.
    """

    source_root = raw_root or Path(settings.BASE_DIR) / "data" / "raw"
    collected: list[tuple[str, str, str, int, str]] = []
    for dataset in SOURCE_DATASETS:
        contract = load_contract(dataset)
        filename = contract.get("source_file")
        sheet_name = contract.get("source_sheet")
        if not isinstance(filename, str) or not isinstance(sheet_name, str):
            raise ValueError(f"Contract for {dataset!r} must define source_file and source_sheet.")

        source_path = source_root / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Contracted {dataset} source is missing: {source_path}")

        # This is a deliberate command-time read, never a request-time read.
        row_count = len(pd.read_excel(source_path, sheet_name=sheet_name))
        collected.append(
            (
                dataset,
                filename,
                _sha256_file(source_path),
                row_count,
                _contract_version(dataset, contract),
            )
        )

    registered: list[RegisteredSource] = []
    with transaction.atomic():
        for dataset, filename, file_hash, row_count, contract_version in collected:
            source_file, created = SourceFile.objects.get_or_create(
                dataset=dataset,
                file_hash=file_hash,
                defaults={
                    "filename": filename,
                    "row_count": row_count,
                    "contract_version": contract_version,
                    # Registration identifies a real source version; it does
                    # not claim that all downstream data-quality checks passed.
                    "status": SourceFile.Status.REGISTERED,
                },
            )
            registered.append(RegisteredSource(dataset, source_file, created))
    return registered
