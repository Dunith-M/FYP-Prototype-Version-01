from django.contrib import admin

from .models import Branch, Product, SkuMapping


@admin.register(Branch)
class BranchAdmin(admin.ModelAdmin):
    list_display = ("branch_code", "branch_name")
    search_fields = ("branch_code", "branch_name")
    ordering = ("branch_code",)


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = (
        "canonical_product_id",
        "product_code",
        "product_desc",
        "family_desc",
        "list_price",
        "price_valid",
    )
    list_filter = ("price_valid", "family_desc")
    search_fields = (
        "canonical_product_id",
        "product_code",
        "product_desc",
        "family_desc",
    )
    ordering = ("product_code",)


@admin.register(SkuMapping)
class SkuMappingAdmin(admin.ModelAdmin):
    list_display = (
        "source_system",
        "source_product_code",
        "canonical_product",
        "mapping_reason",
        "confidence_class",
        "reviewer",
        "reviewed_at",
    )
    list_filter = ("source_system", "mapping_reason", "confidence_class")
    search_fields = (
        "source_product_code",
        "canonical_product__product_code",
        "canonical_product__product_desc",
    )
    ordering = ("source_system", "source_product_code")
