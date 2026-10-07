"""Single mode contract for the Day-09 research workflow."""

from django.db import models


class ScenarioMode(models.TextChoices):
    HISTORICAL = "HISTORICAL", "Historical / Research Analytics"
    RESEARCH_SYNTHETIC = "RESEARCH_SYNTHETIC", "Synthetic Decision Demo"
