"""Disposition flags for analytical observations.

Flags identify questionable observations while preserving original evidence. A
flag does not automatically mean that a row should be removed.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import pandas as pd


RECONCILIATION_ABS_TOLERANCE = 1e-9

FLAG_COLUMNS = (
    "negative_units_flag",
    "zero_units_flag",
    "zero_value_flag",
    "reconciliation_difference_flag",
    "missing_family_flag",
    "invalid_price_flag",
    "inventory_unknown_flag",
    "unresolved_product_flag",
)


def _first_present(columns: Iterable[str], frame: pd.DataFrame) -> str | None:
    return next((column for column in columns if column in frame.columns), None)


def _numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _blank_or_missing(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype("string").str.strip().eq("")


def _boolean(values: pd.Series) -> pd.Series:
    """Return an explicit, non-null pandas boolean column."""

    return values.fillna(False).astype(bool)


def apply_data_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with evidence-preserving disposition flags.

    The function never changes a supplied source/canonical value or drops a
    row. When source columns needed for reconciliation, product attributes, or
    inventory state are absent, the related flag is ``False`` and a companion
    ``*_evaluable_flag`` records that the condition was not assessable.
    """

    output = frame.copy(deep=True)

    units_column = _first_present(("raw_total_units", "sales_units", "TOTAL_UNITS"), output)
    value_column = _first_present(("raw_total_value", "sales_value", "TOTAL_VALUE"), output)
    units = _numeric(output, units_column)
    values = _numeric(output, value_column)
    output["negative_units_flag"] = _boolean(units.lt(0))
    output["zero_units_flag"] = _boolean(units.eq(0))
    output["zero_value_flag"] = _boolean(values.eq(0))

    reconciliation_columns = {
        "total_units": _first_present(("raw_total_units", "TOTAL_UNITS"), output),
        "hire_units": _first_present(("hire_units", "HIRE_UNITS"), output),
        "cash_units": _first_present(("cash_units", "CASH_UNITS"), output),
        "total_value": _first_present(("raw_total_value", "TOTAL_VALUE"), output),
        "hire_value": _first_present(("hire_value", "HIRE_VAL"), output),
        "cash_value": _first_present(("cash_value", "CASH_VALUE"), output),
    }
    reconciliation_available = all(reconciliation_columns.values())
    if reconciliation_available:
        numeric = {
            name: _numeric(output, column)
            for name, column in reconciliation_columns.items()
        }
        evaluable = pd.Series(True, index=output.index)
        for series in numeric.values():
            evaluable &= series.notna()
        if "reconciliation_components_complete_flag" in output:
            evaluable &= _boolean(output["reconciliation_components_complete_flag"])
        units_match = (numeric["total_units"] - numeric["hire_units"] - numeric["cash_units"]).abs().le(
            RECONCILIATION_ABS_TOLERANCE
        )
        values_match = (numeric["total_value"] - numeric["hire_value"] - numeric["cash_value"]).abs().le(
            RECONCILIATION_ABS_TOLERANCE
        )
        output["reconciliation_evaluable_flag"] = _boolean(evaluable)
        output["reconciliation_difference_flag"] = _boolean(evaluable & ~(units_match & values_match))
    else:
        output["reconciliation_evaluable_flag"] = False
        output["reconciliation_difference_flag"] = False

    family_columns = [column for column in ("family_id", "family_desc") if column in output]
    output["family_evaluable_flag"] = bool(family_columns)
    output["missing_family_flag"] = (
        _boolean(pd.concat([_blank_or_missing(output[column]) for column in family_columns], axis=1).any(axis=1))
        if family_columns
        else False
    )

    price_column = _first_present(("list_price", "price", "PRICE"), output)
    output["price_evaluable_flag"] = price_column is not None
    output["invalid_price_flag"] = _boolean(_numeric(output, price_column).isna() | _numeric(output, price_column).le(0)) if price_column else False

    availability_column = _first_present(("availability_state",), output)
    output["inventory_evaluable_flag"] = availability_column is not None
    output["inventory_unknown_flag"] = (
        _boolean(output[availability_column].astype("string").str.strip().str.upper().eq("UNKNOWN"))
        if availability_column
        else False
    )

    mapping_status_column = _first_present(("mapping_status", "confidence_class"), output)
    canonical_column = _first_present(("canonical_product_id",), output)
    output["product_mapping_evaluable_flag"] = (
        mapping_status_column is not None or canonical_column is not None
    )
    unresolved_by_status = (
        output[mapping_status_column].astype("string").str.strip().str.upper().eq("UNRESOLVED")
        if mapping_status_column
        else pd.Series(False, index=output.index)
    )
    unresolved_by_canonical_id = (
        _blank_or_missing(output[canonical_column])
        if canonical_column
        else pd.Series(False, index=output.index)
    )
    output["unresolved_product_flag"] = _boolean(
        unresolved_by_status | unresolved_by_canonical_id
    )

    return output
