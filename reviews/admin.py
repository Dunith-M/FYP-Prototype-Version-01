from django.contrib import admin

from .models import ReviewDecision


@admin.register(ReviewDecision)
class ReviewDecisionAdmin(admin.ModelAdmin):
    list_display = ("reviewed_run", "reviewer", "decision", "reviewed_at")
    list_filter = ("decision",)
    search_fields = ("reviewed_run__run_version", "reviewer__username")
    list_select_related = ("reviewed_run", "reviewer")
