from django.shortcuts import render

from .models import SourceFile


def data_status(request):
    """Render current persisted source-file metadata without reading raw files."""
    source_files = list(SourceFile.objects.order_by("dataset", "-uploaded_at", "-pk"))
    latest_by_dataset = {}
    version_counts = {}
    for source_file in source_files:
        version_counts[source_file.dataset] = version_counts.get(source_file.dataset, 0) + 1
        latest_by_dataset.setdefault(source_file.dataset, source_file)

    dataset_rows = [
        {
            "dataset": dataset,
            "dataset_label": label,
            "source_file": latest_by_dataset.get(dataset),
            "version_count": version_counts.get(dataset, 0),
        }
        for dataset, label in SourceFile.Dataset.choices
    ]
    return render(
        request,
        "ingestion/data_status.html",
        {
            "dataset_rows": dataset_rows,
            "registered_source_file_count": len(source_files),
            "mode": "HISTORICAL_RESEARCH",
        },
    )
