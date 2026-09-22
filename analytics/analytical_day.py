"""Reference-data enrichment for BRANCH_PRODUCT_DAY without feature creation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from analytics.date_utils import normalize_calendar_date, year_month_from_calendar_date
from analytics.sales_day import SALES_DAY_KEY
from analytics.services.data_quality import apply_data_quality_flags
from catalog.models import Branch, Product

class AnalyticalDayValidationError(ValueError): pass

@dataclass(frozen=True)
class AnalyticalDayMetrics:
    base_rows:int; output_rows:int; product_matched:int; product_unmatched:int; calendar_matched:int; calendar_unmatched:int; branch_matched:int; branch_unmatched:int; unknown_before:int; unknown_after:int; units_before:float; units_after:float; value_before:float; value_after:float

def _dupes(frame, keys): return int(frame.duplicated(keys).sum())
def _require_unique(frame, keys, label):
    count=_dupes(frame, keys)
    if count: raise AnalyticalDayValidationError(f"{label} duplicate keys={count}.")

def _calendar_reference(path:Path)->pd.DataFrame:
    raw=pd.read_excel(path, sheet_name="Sheet1")
    if "DATE" not in raw: raise AnalyticalDayValidationError("Calendar DATE column is missing.")
    parsed=raw["DATE"].map(normalize_calendar_date)
    raw["calendar_date"]=parsed.map(lambda r:r.normalized_date)
    if raw.calendar_date.isna().any(): raise AnalyticalDayValidationError("Calendar contains null or invalid DATE values.")
    _require_unique(raw,["calendar_date"],"Calendar")
    rename={"YEAR":"calendar_year","MONTH":"calendar_month","MONTH_NAME":"month_name","DAY":"day","DAY_OF_WEEK":"day_of_week","DAY_OF_WEEK_NUM":"day_of_week_num","WEEK_OF_YEAR":"week_of_year","QUARTER":"quarter","IS_WEEKEND":"is_weekend","IS_PUBLIC_HOLIDAY":"is_public_holiday","HOLIDAY_NAME":"holiday_name","EVENT_TYPE":"event_type","REGION":"calendar_region","SOURCE":"calendar_source","KNOWN_AT_DATE":"known_at_date"}
    raw=raw.rename(columns={k:v for k,v in rename.items() if k in raw})
    raw["year_month"]=raw.calendar_date.map(year_month_from_calendar_date)
    return raw[["calendar_date",*list(rename.values()),"year_month"]]

def _product_reference()->pd.DataFrame:
    frame=pd.DataFrame.from_records(Product.objects.values("canonical_product_id","product_code","product_desc","family_id","family_desc","list_price","price_valid"))
    _require_unique(frame,["canonical_product_id"],"Product")
    return frame

def _branch_reference()->pd.DataFrame:
    frame=pd.DataFrame.from_records(Branch.objects.values("branch_code","branch_name"))
    _require_unique(frame,["branch_code"],"Branch")
    return frame

def build_analytical_day(base_path:Path, calendar_path:Path):
    base=pd.read_parquet(base_path)
    _require_unique(base,list(SALES_DAY_KEY),"BRANCH_PRODUCT_DAY")
    if base[list(SALES_DAY_KEY)].isna().any().any(): raise AnalyticalDayValidationError("BRANCH_PRODUCT_DAY has null grain values.")
    products=_product_reference(); calendar=_calendar_reference(calendar_path); branches=_branch_reference()
    start=len(base); units=float(base.sales_units.sum()); value=float(base.sales_value.sum()); unknown=int((base.availability_state=="UNKNOWN").sum())
    out=base.merge(products,how="left",on="canonical_product_id",validate="many_to_one",indicator="_product")
    if len(out)!=start: raise AnalyticalDayValidationError("Product join multiplied rows.")
    product_unmatched=int((out._product!="both").sum())
    if product_unmatched: raise AnalyticalDayValidationError(f"Product join has unmatched rows={product_unmatched}.")
    out=out.drop(columns="_product")
    out=out.merge(calendar,how="left",left_on="business_date",right_on="calendar_date",validate="many_to_one",indicator="_calendar")
    if len(out)!=start: raise AnalyticalDayValidationError("Calendar join multiplied rows.")
    calendar_matched=int((out._calendar=="both").sum()); out=out.drop(columns="_calendar")
    out=out.merge(branches,how="left",on="branch_code",validate="many_to_one",indicator="_branch")
    if len(out)!=start: raise AnalyticalDayValidationError("Branch join multiplied rows.")
    branch_unmatched=int((out._branch!="both").sum())
    if branch_unmatched: raise AnalyticalDayValidationError(f"Branch join has unmatched rows={branch_unmatched}.")
    out=out.drop(columns="_branch").sort_values(list(SALES_DAY_KEY),kind="stable").reset_index(drop=True)
    out=apply_data_quality_flags(out)
    _require_unique(out,list(SALES_DAY_KEY),"ANALYTICAL_DAY")
    if (out.loc[out.availability_state=="UNKNOWN","inventory_qty"].notna().any()): raise AnalyticalDayValidationError("UNKNOWN inventory quantity became non-null.")
    return out, AnalyticalDayMetrics(start,len(out),start-product_unmatched,product_unmatched,calendar_matched,start-calendar_matched,start-branch_unmatched,branch_unmatched,unknown,int((out.availability_state=="UNKNOWN").sum()),units,float(out.sales_units.sum()),value,float(out.sales_value.sum()))

def write_analytical_day(frame:pd.DataFrame,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(path,index=False)
