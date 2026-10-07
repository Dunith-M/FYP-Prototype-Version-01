from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

from catalog.models import Branch, Product
from ingestion.models import SourceFile


class DecisionOption(models.Model):
    """One synthetic planning possibility for a later optimizer; never a decision."""

    class OptionType(models.TextChoices):
        LISTED_TARGET = "LISTED_TARGET_OPTION", "Listed target option"
        DELIST = "DELIST_OPTION", "Delist option"
        REMAIN_UNLISTED = "REMAIN_UNLISTED_OPTION", "Remain unlisted option"
        LIST_TRIAL = "LIST_TRIAL_OPTION", "List trial option"

    class ParameterSource(models.TextChoices):
        SYNTHETIC_DEMO = "SYNTHETIC_DEMO", "Synthetic demo"

    class EvidenceSource(models.TextChoices):
        UNASSIGNED = "UNASSIGNED", "Unassigned"
        TRACK_B_CANDIDATE_SCORE = "TRACK_B_CANDIDATE_SCORE", "Track-B candidate score"
        HISTORICAL_LAG_1_NORMALIZED = "HISTORICAL_LAG_1_NORMALIZED", "Historical lag-1 normalized proxy"

    branch = models.ForeignKey(Branch, on_delete=models.PROTECT, related_name="decision_options")
    product = models.ForeignKey(Product, on_delete=models.PROTECT, related_name="decision_options")
    planning_month = models.DateField()
    l0 = models.BooleanField(help_text="Synthetic starting listing state.")
    q0 = models.PositiveIntegerField(null=True, blank=True, help_text="Synthetic reference target; not physical stock.")
    l1 = models.BooleanField(help_text="Proposed listing state for this option.")
    q1 = models.PositiveIntegerField(help_text="Proposed target allocation, not a purchase quantity.")
    option_type = models.CharField(max_length=30, choices=OptionType.choices)
    evidence_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    evidence_source = models.CharField(
        max_length=40,
        choices=EvidenceSource.choices,
        default=EvidenceSource.UNASSIGNED,
        help_text="Research evidence origin for the non-financial proxy utility.",
    )
    parameter_source = models.CharField(max_length=30, choices=ParameterSource.choices, default=ParameterSource.SYNTHETIC_DEMO)
    scenario_version = models.CharField(max_length=100)

    class Meta:
        ordering = ("planning_month", "branch__branch_code", "product__canonical_product_id", "l1", "q1")
        constraints = [
            models.UniqueConstraint(fields=("branch", "product", "planning_month", "scenario_version", "l1", "q1"), name="planning_option_scenario_uniq"),
            models.CheckConstraint(condition=(models.Q(l1=False, q1=0) | models.Q(l1=True, q1__gt=0)), name="planning_option_l1_q1_valid"),
            models.CheckConstraint(condition=(models.Q(l0=False, q0__isnull=True) | models.Q(l0=True, q0__gt=0)), name="planning_option_l0_q0_valid"),
        ]
        indexes = [
            models.Index(fields=("branch", "planning_month"), name="planning_opt_branch_month_ix"),
            models.Index(fields=("scenario_version", "option_type"), name="planning_opt_scenario_type_ix"),
        ]

    def __str__(self) -> str:
        return f"{self.branch.branch_code} | {self.product.canonical_product_id} | {self.planning_month:%Y-%m} | L1={int(self.l1)} q1={self.q1}"


class OptimizationResult(models.Model):
    """One Day-08 solver run over a versioned finite option scenario."""

    class SolverStatus(models.TextChoices):
        OPTIMAL = "OPTIMAL", "Optimal"
        FEASIBLE = "FEASIBLE", "Feasible"
        INFEASIBLE = "INFEASIBLE", "Infeasible"
        UNKNOWN = "UNKNOWN", "Unknown"
        ERROR = "ERROR", "Error"

    class FeasibilityStatus(models.TextChoices):
        NOT_RUN = "NOT_RUN", "Not run"
        PASS = "PASS", "Pass"
        FAIL = "FAIL", "Fail"

    planning_month = models.DateField()
    scenario_version = models.CharField(max_length=100)
    run_version = models.CharField(max_length=100)
    solver_status = models.CharField(max_length=20, choices=SolverStatus.choices, default=SolverStatus.UNKNOWN)
    objective_value = models.FloatField(null=True, blank=True, help_text="Non-financial optimization proxy utility.")
    solver_metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Reproducibility metadata for the non-financial CP-SAT run.",
    )
    feasibility_status = models.CharField(max_length=20, choices=FeasibilityStatus.choices, default=FeasibilityStatus.NOT_RUN)
    selected_options = models.ManyToManyField(DecisionOption, blank=True, related_name="optimization_results")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-created_at",)
        constraints = [
            models.UniqueConstraint(
                fields=("planning_month", "scenario_version", "run_version"),
                name="planning_result_scenario_run_uniq",
            )
        ]
        indexes = [
            models.Index(fields=("planning_month", "scenario_version"), name="planning_res_month_scen_ix"),
        ]

    def __str__(self) -> str:
        return f"{self.scenario_version} | {self.run_version} | {self.solver_status}"


class RunProvenance(models.Model):
    """Immutable input/version snapshot for one persisted optimization run.

    Recommendation rows reach their provenance through ``optimization_result``.
    The snapshot intentionally stores source identifiers once per run rather
    than copying them into every branch/product recommendation.
    """

    optimization_result = models.OneToOneField(
        OptimizationResult,
        on_delete=models.PROTECT,
        related_name="provenance",
    )
    branch = models.ForeignKey(Branch, on_delete=models.PROTECT, related_name="run_provenance")
    mode = models.CharField(
        max_length=30,
        choices=(
            ("HISTORICAL", "Historical / Research Analytics"),
            ("RESEARCH_SYNTHETIC", "Synthetic Decision Demo"),
        ),
    )
    source_manifest_path = models.CharField(max_length=255)
    source_manifest_hash = models.CharField(max_length=64)
    source_snapshot = models.JSONField(
        help_text="Frozen dataset filename/hash entries copied from the persisted feature manifest."
    )
    source_files = models.ManyToManyField(
        SourceFile,
        blank=True,
        related_name="run_provenance_records",
        help_text="Registered SourceFile versions that exactly match the frozen manifest hashes.",
    )
    mapping_version = models.CharField(max_length=255)
    feature_version = models.CharField(max_length=100)
    feature_artifact_hash = models.CharField(max_length=64)
    model_manifest_path = models.CharField(max_length=255)
    model_manifest_hash = models.CharField(max_length=64)
    model_type = models.CharField(max_length=255)
    model_version = models.CharField(max_length=100)
    selected_model = models.CharField(max_length=100)
    scenario_config_path = models.CharField(max_length=255, blank=True)
    scenario_config_hash = models.CharField(max_length=64, blank=True)
    captured_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-captured_at",)

    def __str__(self) -> str:
        return f"{self.optimization_result.run_version} | {self.mode}"


class Recommendation(models.Model):
    """One Day-08 branch/product output, never a Day-09 review decision."""

    class Action(models.TextChoices):
        INCREASE = "INCREASE", "Increase"
        KEEP = "KEEP", "Keep"
        REDUCE = "REDUCE", "Reduce"
        REMOVE = "REMOVE", "Remove"
        INTRODUCE = "INTRODUCE", "Introduce"
        DO_NOT_INTRODUCE = "DO_NOT_INTRODUCE", "Do not introduce"

    class Status(models.TextChoices):
        READY = "READY", "Ready"
        WITHHELD = "WITHHELD", "Withheld"

    optimization_result = models.ForeignKey(OptimizationResult, on_delete=models.PROTECT, related_name="recommendations")
    selected_option = models.ForeignKey(DecisionOption, null=True, blank=True, on_delete=models.PROTECT, related_name="recommendations")
    branch = models.ForeignKey(Branch, on_delete=models.PROTECT, related_name="recommendations")
    product = models.ForeignKey(Product, on_delete=models.PROTECT, related_name="recommendations")
    action = models.CharField(max_length=30, choices=Action.choices, null=True, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.WITHHELD)
    reasons = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("optimization_result", "branch__branch_code", "product__canonical_product_id")
        constraints = [
            models.UniqueConstraint(
                fields=("optimization_result", "branch", "product"),
                name="planning_recommendation_result_pair_uniq",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(status="WITHHELD", action__isnull=True)
                    | models.Q(status="READY", action__isnull=False)
                ),
                name="planning_recommendation_action_status_valid",
            ),
        ]

    def clean(self) -> None:
        """Require a checked selected option before a Day-08 output is ready."""
        super().clean()
        if self.status != self.Status.READY:
            return

        errors = {}
        if self.optimization_result.feasibility_status != OptimizationResult.FeasibilityStatus.PASS:
            errors["status"] = "A ready recommendation requires a feasibility PASS result."
        if self.selected_option is None:
            errors["selected_option"] = "A ready recommendation requires a selected decision option."
        else:
            if self.selected_option.branch_id != self.branch_id or self.selected_option.product_id != self.product_id:
                errors["selected_option"] = "Selected option must match the recommendation branch and product."
            elif not self.optimization_result.selected_options.filter(pk=self.selected_option.pk).exists():
                errors["selected_option"] = "Selected option must belong to the optimization result."
        if errors:
            raise ValidationError(errors)

    def __str__(self) -> str:
        return f"{self.branch.branch_code} | {self.product.canonical_product_id} | {self.status}"
