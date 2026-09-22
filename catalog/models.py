from django.db import models


class Branch(models.Model):
    """Official pilot-branch reference loaded from the branch profile source."""

    branch_code = models.CharField(max_length=20, unique=True)
    branch_name = models.CharField(max_length=255)

    class Meta:
        ordering = ["branch_code"]

    def __str__(self) -> str:
        return f"{self.branch_code} - {self.branch_name}"


class Product(models.Model):
    """Official product-master record for one normalized source product code."""

    canonical_product_id = models.CharField(max_length=120, unique=True, editable=False)
    product_code = models.CharField(max_length=100, unique=True)
    product_desc = models.CharField(max_length=500)
    family_id = models.CharField(max_length=100, blank=True)
    family_desc = models.CharField(max_length=255, blank=True)
    list_price = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        null=True,
        blank=True,
    )
    price_valid = models.BooleanField(default=False)

    class Meta:
        ordering = ("product_code",)

    def __str__(self) -> str:
        return f"{self.product_code} - {self.product_desc}"


class SkuMapping(models.Model):
    """Maps one normalized external SKU to its canonical Product when known."""

    class SourceSystem(models.TextChoices):
        SALES = "SALES", "Sales"
        INVENTORY = "INVENTORY", "Inventory"
        PRODUCT = "PRODUCT", "Product"

    class MappingReason(models.TextChoices):
        EXACT_CODE = "EXACT_CODE", "Exact Code"
        APPROVED_ALIAS = "APPROVED_ALIAS", "Approved Alias"
        MANUAL_REVIEW = "MANUAL_REVIEW", "Manual Review"

    class ConfidenceClass(models.TextChoices):
        CONFIRMED = "CONFIRMED", "Confirmed"
        SUGGESTED = "SUGGESTED", "Suggested"
        UNRESOLVED = "UNRESOLVED", "Unresolved"

    source_system = models.CharField(max_length=20, choices=SourceSystem.choices)
    source_product_code = models.CharField(max_length=100)
    canonical_product = models.ForeignKey(
        Product,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="sku_mappings",
    )
    mapping_reason = models.CharField(max_length=30, choices=MappingReason.choices)
    confidence_class = models.CharField(max_length=20, choices=ConfidenceClass.choices)
    effective_from = models.DateField(null=True, blank=True)
    effective_to = models.DateField(null=True, blank=True)
    reviewer = models.CharField(max_length=255, blank=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ("source_system", "source_product_code")
        constraints = [
            models.UniqueConstraint(
                fields=("source_system", "source_product_code"),
                name="catalog_skumapping_source_system_code_uniq",
            )
        ]

    def __str__(self) -> str:
        return f"{self.source_system}: {self.source_product_code}"
