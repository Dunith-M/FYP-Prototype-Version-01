from django.contrib import admin

from .models import CandidateResult


@admin.register(CandidateResult)
class CandidateResultAdmin(admin.ModelAdmin):
    list_display = ("branch", "product", "planning_month", "rank", "candidate_score", "evidence_class", "status", "run_version")
    list_filter = ("planning_month", "evidence_class", "status", "run_version")
    search_fields = ("branch__branch_code", "product__canonical_product_id", "product__product_desc", "run_version")
