from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from catalog.models import Product, SkuMapping
from catalog.sku_mapping import (
    SalesSkuDataValidationError,
    extract_sales_source_metrics,
)


class Command(BaseCommand):
    help = "Build exact Sales SKU mappings and preserve unmatched codes as unresolved."

    def add_arguments(self, parser):
        parser.add_argument(
            "--sales-file",
            type=Path,
            default=settings.BASE_DIR / "data" / "raw" / "transactions_data.xlsx",
            help="Path to the historical sales workbook.",
        )

    @staticmethod
    def _is_review_protected(mapping: SkuMapping) -> bool:
        return (
            mapping.mapping_reason == SkuMapping.MappingReason.APPROVED_ALIAS
            or bool(mapping.reviewer.strip())
            or mapping.reviewed_at is not None
        )

    def handle(self, *args, **options):
        sales_file = options["sales_file"]
        try:
            metrics = extract_sales_source_metrics(sales_file)
        except (SalesSkuDataValidationError, OSError) as error:
            raise CommandError(str(error)) from error

        if metrics.blank_source_codes:
            raise CommandError(
                "Sales SKU mapping is blocked: "
                f"blank_or_null_sales_part_rows={metrics.blank_source_codes}."
            )

        products_by_code = {
            product.product_code: product
            for product in Product.objects.only("id", "product_code")
        }
        preserved_reviewed_mappings = 0

        with transaction.atomic():
            existing_mappings = {
                mapping.source_product_code: mapping
                for mapping in SkuMapping.objects.filter(
                    source_system=SkuMapping.SourceSystem.SALES
                ).select_related("canonical_product")
            }
            for source_code in metrics.codes:
                existing = existing_mappings.get(source_code)
                if existing is not None and self._is_review_protected(existing):
                    preserved_reviewed_mappings += 1
                    continue

                product = products_by_code.get(source_code)
                if product is not None:
                    defaults = {
                        "canonical_product": product,
                        "mapping_reason": SkuMapping.MappingReason.EXACT_CODE,
                        "confidence_class": SkuMapping.ConfidenceClass.CONFIRMED,
                        "reviewer": "",
                        "reviewed_at": None,
                        "notes": "",
                    }
                else:
                    defaults = {
                        "canonical_product": None,
                        "mapping_reason": SkuMapping.MappingReason.MANUAL_REVIEW,
                        "confidence_class": SkuMapping.ConfidenceClass.UNRESOLVED,
                        "reviewer": "",
                        "reviewed_at": None,
                        "notes": "Historical sales code absent from current product master",
                    }

                SkuMapping.objects.update_or_create(
                    source_system=SkuMapping.SourceSystem.SALES,
                    source_product_code=source_code,
                    defaults=defaults,
                )

        sales_mappings = SkuMapping.objects.filter(
            source_system=SkuMapping.SourceSystem.SALES
        )
        exact_mapping_count = sales_mappings.filter(
            mapping_reason=SkuMapping.MappingReason.EXACT_CODE
        ).count()
        unresolved_mapping_count = sales_mappings.filter(
            confidence_class=SkuMapping.ConfidenceClass.UNRESOLVED
        ).count()
        mappings_by_code = {
            mapping.source_product_code: mapping
            for mapping in sales_mappings.only(
                "source_product_code", "canonical_product"
            )
        }
        mapped_codes = {
            code
            for code, mapping in mappings_by_code.items()
            if mapping.canonical_product_id is not None
        }
        unresolved_codes = set(metrics.codes) - mapped_codes
        mapped_rows = sum(metrics.codes[code].row_count for code in mapped_codes)
        unresolved_rows = sum(
            metrics.codes[code].row_count for code in unresolved_codes
        )
        mapped_signed_units = sum(
            (metrics.codes[code].signed_units for code in mapped_codes), start=0
        )
        unresolved_signed_units = sum(
            (metrics.codes[code].signed_units for code in unresolved_codes), start=0
        )
        total_signed_units = mapped_signed_units + unresolved_signed_units

        def percentage(numerator, denominator) -> Decimal:
            return (
                Decimal(0)
                if denominator == 0
                else numerator * Decimal(100) / denominator
            )

        self.stdout.write(self.style.SUCCESS("Sales SKU mappings built."))
        self.stdout.write(f"Distinct SALES_PART codes: {metrics.distinct_codes}")
        self.stdout.write(f"EXACT_CODE mappings: {exact_mapping_count}")
        self.stdout.write(f"UNRESOLVED mappings: {unresolved_mapping_count}")
        self.stdout.write(
            "Distinct-code coverage: "
            f"{len(mapped_codes)}/{metrics.distinct_codes} "
            f"({percentage(len(mapped_codes), metrics.distinct_codes):.2f}%)"
        )
        self.stdout.write(
            "Row coverage: "
            f"{mapped_rows}/{metrics.total_rows} "
            f"({percentage(mapped_rows, metrics.total_rows):.2f}%)"
        )
        self.stdout.write(
            "Signed-unit coverage: "
            f"mapped={mapped_signed_units}; unresolved={unresolved_signed_units}; "
            f"({percentage(mapped_signed_units, total_signed_units):.2f}%)"
        )
        self.stdout.write(
            f"Unresolved sales rows: {unresolved_rows}; "
            f"blank SALES_PART rows: {metrics.blank_source_codes}"
        )
        self.stdout.write(f"Reviewed mappings preserved: {preserved_reviewed_mappings}")
