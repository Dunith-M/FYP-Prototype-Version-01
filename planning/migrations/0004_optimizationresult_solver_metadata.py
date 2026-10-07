# Generated manually for the Day-08 solver-run metadata boundary.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("planning", "0003_decisionoption_evidence_source"),
    ]

    operations = [
        migrations.AddField(
            model_name="optimizationresult",
            name="solver_metadata",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Reproducibility metadata for the non-financial CP-SAT run.",
            ),
        ),
    ]
