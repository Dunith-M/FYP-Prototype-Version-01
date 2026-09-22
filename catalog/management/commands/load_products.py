from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from catalog.models import Product
from catalog.product_master import ProductDataValidationError, inspect_product_source


class Command(BaseCommand):
    help = "Load the Product master from the raw product workbook."

    def add_arguments(self, parser):
        parser.add_argument(
            "--product-file",
            type=Path,
            default=settings.BASE_DIR / "data" / "raw" / "product_data.xlsx",
            help="Path to the product-master workbook.",
        )

    def handle(self, *args, **options):
        product_file = options["product_file"]
        try:
            validation = inspect_product_source(product_file)
        except (ProductDataValidationError, OSError) as error:
            raise CommandError(str(error)) from error

        if not validation.is_loadable:
            raise CommandError(
                "Product source validation failed: "
                f"missing_product_codes={validation.missing_product_codes}; "
                f"conflicting_duplicate_codes={validation.conflicting_duplicate_codes}."
            )

        with transaction.atomic():
            for record in validation.unique_records:
                conflicting_canonical_id = Product.objects.filter(
                    canonical_product_id=record.canonical_product_id
                ).exclude(product_code=record.product_code)
                if conflicting_canonical_id.exists():
                    raise CommandError(
                        "Canonical product ID collision: "
                        f"{record.canonical_product_id} is already assigned to a different product code."
                    )

                Product.objects.update_or_create(
                    product_code=record.product_code,
                    defaults={
                        "canonical_product_id": record.canonical_product_id,
                        "product_desc": record.product_desc,
                        "family_id": record.family_id,
                        "family_desc": record.family_desc,
                        "list_price": record.list_price,
                        "price_valid": record.price_valid,
                    },
                )

        self.stdout.write(self.style.SUCCESS("Product master loaded."))
        self.stdout.write(f"Rows read: {validation.row_count}")
        self.stdout.write(f"Unique product codes: {len(validation.unique_records)}")
        self.stdout.write(f"Duplicate source codes: {validation.duplicate_codes}")
        self.stdout.write(f"Price counts: {validation.price_counts}")
