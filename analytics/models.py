from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

from catalog.models import Branch, Product


class CandidateResult(models.Model):
    """Persisted Day-07 Track-B research evidence, not a planning decision.

    ``planning_month`` stores the first calendar day of the represented month.
    ``run_version`` follows the existing artifact-version approach because the
    project does not yet have a Django analytical-run/snapshot model.
    """

    class EvidenceClass(models.TextChoices):
        ADEQUATE = "ADEQUATE", "Adequate"
        LIMITED = "LIMITED", "Limited"
        INSUFFICIENT = "INSUFFICIENT", "Insufficient"

    class CandidateStatus(models.TextChoices):
        RESEARCH_SHORTLIST = "RESEARCH_SHORTLIST", "Research shortlist"

    class ParameterSource(models.TextChoices):
        HISTORICAL_EVIDENCE = "HISTORICAL_EVIDENCE", "Historical evidence"

    branch = models.ForeignKey(Branch, on_delete=models.PROTECT, related_name="candidate_results")
    product = models.ForeignKey(Product, on_delete=models.PROTECT, related_name="candidate_results")
    planning_month = models.DateField()
    peer_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    family_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    similar_branch_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    price_fit_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    candidate_score = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    rank = models.PositiveIntegerField(validators=[MinValueValidator(1)])
    evidence_class = models.CharField(max_length=20, choices=EvidenceClass.choices)
    status = models.CharField(max_length=30, choices=CandidateStatus.choices, default=CandidateStatus.RESEARCH_SHORTLIST)
    run_version = models.CharField(max_length=100)
    parameter_source = models.CharField(max_length=30, choices=ParameterSource.choices, default=ParameterSource.HISTORICAL_EVIDENCE)

    class Meta:
        ordering = ("planning_month", "branch__branch_code", "rank", "product__canonical_product_id")
        constraints = [
            models.UniqueConstraint(fields=("branch", "product", "planning_month", "run_version"), name="analytics_candidate_result_run_uniq"),
            models.CheckConstraint(condition=models.Q(candidate_score__gte=0.0) & models.Q(candidate_score__lte=1.0), name="analytics_candidate_score_0_1"),
            models.CheckConstraint(condition=models.Q(rank__gt=0), name="analytics_candidate_rank_positive"),
        ]
        indexes = [
            models.Index(fields=("branch", "planning_month"), name="analytics_cand_branch_month_ix"),
            models.Index(fields=("planning_month", "rank"), name="analytics_cand_month_rank_ix"),
        ]

    def __str__(self) -> str:
        return f"{self.branch.branch_code} | {self.product.canonical_product_id} | {self.planning_month:%Y-%m} | Rank {self.rank}"
