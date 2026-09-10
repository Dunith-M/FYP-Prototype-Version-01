# MVP Assumptions and Decisions

## Prototype Mode
HISTORICAL / RESEARCH PROTOTYPE.

Historical Singer data is used for analytics.
The system must not claim that outputs are current operational advice.

## Forecast Target
Draft target:
next-month fulfilled positive sales units
at branch × canonical product × month.

## Negative / Zero Sales Events
Negative and zero values are preserved and flagged.
They will not be silently converted to zero or deleted.
Their exact business meaning remains unresolved.

## Optimization
Operational Singer constraints, L0 and q0 are not fully approved.

Synthetic planning values may therefore be used only for the
Synthetic Decision Demonstration.

Synthetic parameters must never be presented as Singer policy.
