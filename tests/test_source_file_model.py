import pytest

from ingestion.models import SourceFile


@pytest.mark.django_db
def test_source_file_stores_traceability_metadata():
    source_file = SourceFile.objects.create(
        dataset=SourceFile.Dataset.SALES,
        filename="transactions_data.xlsx",
        file_hash="a" * 64,
        row_count=67_385,
        contract_version="v1",
        status=SourceFile.Status.VALIDATED,
    )

    assert source_file.filename == "transactions_data.xlsx"
    assert source_file.file_hash == "a" * 64
    assert source_file.dataset == SourceFile.Dataset.SALES
    assert source_file.row_count == 67_385
    assert source_file.contract_version == "v1"
    assert source_file.status == SourceFile.Status.VALIDATED
    assert source_file.uploaded_at is not None
    assert str(source_file) == "sales: transactions_data.xlsx"
