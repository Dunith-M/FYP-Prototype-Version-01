from django.contrib import admin

from .models import SourceFile


@admin.register(SourceFile)
class SourceFileAdmin(admin.ModelAdmin):
    list_display = (
        "dataset",
        "filename",
        "row_count",
        "contract_version",
        "status",
        "uploaded_at",
    )
    list_filter = ("dataset", "status")
    search_fields = ("filename", "file_hash")
