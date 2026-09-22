from django.core.validators import RegexValidator
from django.db import models


class SourceFile(models.Model):
    """Immutable traceability metadata for one registered source-file version."""

    class Dataset(models.TextChoices):
        SALES = "sales", "Sales"
        INVENTORY = "inventory", "Inventory"
        PRODUCTS = "products", "Products"
        CALENDAR = "calendar", "Calendar"
        BRANCHES = "branches", "Branches"

    class Status(models.TextChoices):
        REGISTERED = "REGISTERED", "Registered"
        VALIDATED = "VALIDATED", "Validated"
        REJECTED = "REJECTED", "Rejected"
        QUARANTINED = "QUARANTINED", "Quarantined"

    filename = models.CharField(max_length=255)
    file_hash = models.CharField(
        max_length=64,
        validators=[
            RegexValidator(
                regex=r"^[0-9a-fA-F]{64}$",
                message="file_hash must be a 64-character SHA-256 hexadecimal digest.",
            )
        ],
        help_text="SHA-256 hexadecimal digest calculated by a future ingestion service.",
    )
    dataset = models.CharField(max_length=20, choices=Dataset.choices)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    row_count = models.PositiveIntegerField()
    contract_version = models.CharField(max_length=50)
    status = models.CharField(
        max_length=50,
        choices=Status.choices,
        default=Status.REGISTERED,
    )

    class Meta:
        ordering = ["-uploaded_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "file_hash"],
                name="ingestion_sourcefile_dataset_hash_uniq",
            )
        ]

    def __str__(self) -> str:
        return f"{self.dataset}: {self.filename}"
