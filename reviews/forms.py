from django import forms

from .models import ReviewDecision


class ReviewDecisionForm(forms.ModelForm):
    """Collect only the human decision and rationale for an already-known run."""

    class Meta:
        model = ReviewDecision
        fields = ("decision", "comment")
        widgets = {
            "comment": forms.Textarea(attrs={"rows": 4}),
        }

    def clean_comment(self):
        comment = self.cleaned_data["comment"].strip()
        if not comment:
            raise forms.ValidationError("A meaningful review comment is required.")
        return comment
