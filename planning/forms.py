"""Validated planning-context input for the Day-09 prototype workflow."""

from __future__ import annotations

from datetime import date

from django import forms

from analytics.models import CandidateResult
from catalog.branch_master import EXPECTED_BRANCH_CODES
from catalog.models import Branch
from .modes import ScenarioMode


class PilotBranchChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, branch: Branch) -> str:
        return f"{branch.branch_code} — {branch.branch_name}"


class RunScenarioForm(forms.Form):
    """Collect a supported month, registered pilot branch, and prototype mode."""

    planning_month = forms.TypedChoiceField(
        choices=(),
        coerce=date.fromisoformat,
        empty_value=None,
        label="Planning month",
        help_text="Only months with persisted candidate evidence are available.",
    )
    branch = PilotBranchChoiceField(
        queryset=Branch.objects.none(),
        empty_label="Select a pilot branch",
        label="Branch",
    )
    mode = forms.ChoiceField(choices=ScenarioMode.choices, label="Mode")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        supported_months = list(
            CandidateResult.objects.order_by("planning_month")
            .values_list("planning_month", flat=True)
            .distinct()
        )
        self.fields["planning_month"].choices = [
            (month.isoformat(), month.strftime("%B %Y")) for month in supported_months
        ]
        self.fields["branch"].queryset = Branch.objects.filter(
            branch_code__in=EXPECTED_BRANCH_CODES
        ).order_by("branch_code")
        self.has_planning_months = bool(supported_months)
        self.has_branches = self.fields["branch"].queryset.exists()
