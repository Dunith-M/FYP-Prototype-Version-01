from django.conf import settings
from django.db import models

from planning.models import OptimizationResult


class ReviewDecision(models.Model):
    """One immutable human review event for an exact persisted recommendation plan."""

    class Decision(models.TextChoices):
        ACCEPT = "ACCEPT", "Accept"
        REJECT = "REJECT", "Reject"
        MODIFY = "MODIFY", "Modify"

    class CurrentStatus(models.TextChoices):
        """Current display state derived from immutable review events."""

        PENDING = "PENDING", "Pending"
        ACCEPTED = "ACCEPTED", "Accepted"
        REJECTED = "REJECTED", "Rejected"
        MODIFICATION_REQUESTED = "MODIFICATION_REQUESTED", "Modification requested"

    reviewed_run = models.ForeignKey(
        OptimizationResult,
        on_delete=models.PROTECT,
        related_name="review_decisions",
        help_text="The exact persisted optimization run/version reviewed by the human.",
    )
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="review_decisions",
        help_text="Authenticated user who recorded this review event.",
    )
    decision = models.CharField(max_length=10, choices=Decision.choices)
    comment = models.TextField(help_text="Required reviewer rationale; not a Singer operational approval.")
    reviewed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-reviewed_at",)

    @classmethod
    def current_status_for_run(cls, reviewed_run: OptimizationResult):
        """Return the current status and newest review for one exact run.

        ReviewDecision remains the audit source of truth. This status is
        derived rather than stored on a machine Recommendation, so a human
        decision cannot overwrite Day-08 action or evidence values.
        """

        latest_review = (
            cls.objects.filter(reviewed_run=reviewed_run)
            .select_related("reviewer")
            .order_by("-reviewed_at", "-pk")
            .first()
        )
        if latest_review is None:
            return cls.CurrentStatus.PENDING, None

        status_by_decision = {
            cls.Decision.ACCEPT: cls.CurrentStatus.ACCEPTED,
            cls.Decision.REJECT: cls.CurrentStatus.REJECTED,
            cls.Decision.MODIFY: cls.CurrentStatus.MODIFICATION_REQUESTED,
        }
        return status_by_decision[latest_review.decision], latest_review

    def __str__(self) -> str:
        return f"{self.reviewed_run.run_version} — {self.decision} — {self.reviewer}"
