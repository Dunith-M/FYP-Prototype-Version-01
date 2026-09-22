"""Cross-pipeline Day 3 guards; source-count assertions target this fixed MVP data."""
from pathlib import Path
import pandas as pd
from catalog.branch_master import EXPECTED_BRANCH_CODES,coverage_report
from catalog.sku_mapping import extract_sales_source_metrics

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data'/'raw'; PROCESSED=ROOT/'data'/'processed'
KEY=['branch_code','canonical_product_id','business_date']
def test_t_data_map_01_all_branch_codes_map_once():
    sales=coverage_report(RAW/'transactions_data.xlsx','SHOP_CODE',EXPECTED_BRANCH_CODES)
    inventory=coverage_report(RAW/'inventory_data.xlsx','SITE_ID',EXPECTED_BRANCH_CODES)
    assert sales.unmatched_codes=={} and inventory.unmatched_codes=={} and len(EXPECTED_BRANCH_CODES)==10
def test_t_data_map_02_no_many_to_many_product_join_and_unresolved_counted():
    sales=pd.read_parquet(PROCESSED/'sales_day.parquet'); inv=pd.read_parquet(PROCESSED/'inventory_day.parquet'); joined=pd.read_parquet(PROCESSED/'branch_product_day.parquet'); unresolved=pd.read_parquet(PROCESSED/'sales_unresolved.parquet'); raw=extract_sales_source_metrics(RAW/'transactions_data.xlsx')
    assert not sales.duplicated(KEY).any() and not inv.duplicated(['branch_code','canonical_product_id','snapshot_date']).any()
    assert len(joined)==len(sales) and not joined.duplicated(KEY).any()
    assert raw.total_rows==int(sales.source_row_count.sum())+len(unresolved)
    assert len(unresolved)>0 and unresolved.source_product_code.nunique()>0
def test_day3_reconciliations_hold_across_outputs():
    sales=pd.read_parquet(PROCESSED/'sales_day.parquet'); joined=pd.read_parquet(PROCESSED/'branch_product_day.parquet'); analytical=pd.read_parquet(PROCESSED/'analytical_day.parquet')
    assert sales.sales_units.sum()==joined.sales_units.sum()==analytical.sales_units.sum()
    assert sales.sales_value.sum()==joined.sales_value.sum()==analytical.sales_value.sum()
def test_t_data_avail_01_unknown_never_has_zero_quantity_in_current_output():
    joined=pd.read_parquet(PROCESSED/'branch_product_day.parquet'); unknown=joined[joined.availability_state=='UNKNOWN']
    assert len(unknown)>0 and unknown.inventory_qty.isna().all() and not (unknown.inventory_qty==0).any()
