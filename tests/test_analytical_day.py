from datetime import date
from pathlib import Path
import pandas as pd
import pytest
from openpyxl import Workbook
from analytics.analytical_day import AnalyticalDayValidationError,build_analytical_day,write_analytical_day
from catalog.models import Branch,Product

def _base(path,rows):
    pd.DataFrame(rows,columns=['branch_code','canonical_product_id','business_date','sales_units','sales_value','source_row_count','inventory_qty','availability_state']).to_parquet(path,index=False);return path
def _calendar(path,rows):
    w=Workbook();s=w.active;s.title='Sheet1';s.append(['DATE','YEAR','MONTH','MONTH_NAME','DAY','DAY_OF_WEEK','DAY_OF_WEEK_NUM','WEEK_OF_YEAR','QUARTER','IS_WEEKEND','IS_PUBLIC_HOLIDAY','HOLIDAY_NAME','EVENT_TYPE','REGION','SOURCE','KNOWN_AT_DATE'])
    for r in rows:s.append([r,2025,5,'May',r.day,'Thursday',4,18,2,0,0,'','','Sri Lanka','Test',''])
    w.save(path);return path
def _refs():
    Branch.objects.create(branch_code='BIL01',branch_name='Galle');Product.objects.create(canonical_product_id='CP_P001',product_code='P001',product_desc='Product',family_id='F',family_desc='Family',list_price='100.00',price_valid=True)
@pytest.mark.django_db
def test_attributes_attach_and_preserve_unknown(tmp_path):
    _refs();base=_base(tmp_path/'b.parquet',[['BIL01','CP_P001',date(2025,5,1),2,200,1,None,'UNKNOWN']]);cal=_calendar(tmp_path/'c.xlsx',[date(2025,5,1)])
    out,m=build_analytical_day(base,cal);r=out.iloc[0]
    assert r.product_code=='P001' and r.branch_name=='Galle' and r.year_month=='2025-05' and r.availability_state=='UNKNOWN' and pd.isna(r.inventory_qty) and m.units_before==m.units_after
    assert bool(r.inventory_unknown_flag) and not bool(r.unresolved_product_flag)
@pytest.mark.django_db
def test_missing_calendar_keeps_row(tmp_path):
    _refs();base=_base(tmp_path/'b.parquet',[['BIL01','CP_P001',date(2025,5,2),2,200,1,8,'AVAILABLE']]);cal=_calendar(tmp_path/'c.xlsx',[date(2025,5,1)])
    out,m=build_analytical_day(base,cal);assert len(out)==1 and m.calendar_unmatched==1 and pd.isna(out.iloc[0].calendar_date)
@pytest.mark.django_db
def test_missing_product_and_branch_are_detected(tmp_path):
    _refs();cal=_calendar(tmp_path/'c.xlsx',[date(2025,5,1)])
    with pytest.raises(AnalyticalDayValidationError,match='Product'):build_analytical_day(_base(tmp_path/'p.parquet',[['BIL01','CP_BAD',date(2025,5,1),1,1,1,1,'AVAILABLE']]),cal)
    with pytest.raises(AnalyticalDayValidationError,match='Branch'):build_analytical_day(_base(tmp_path/'b.parquet',[['BAD','CP_P001',date(2025,5,1),1,1,1,1,'AVAILABLE']]),cal)
@pytest.mark.django_db
def test_duplicate_calendar_is_detected(tmp_path):
    _refs();base=_base(tmp_path/'b.parquet',[['BIL01','CP_P001',date(2025,5,1),1,1,1,1,'AVAILABLE']]);cal=_calendar(tmp_path/'c.xlsx',[date(2025,5,1),date(2025,5,1)])
    with pytest.raises(AnalyticalDayValidationError,match='Calendar duplicate'):build_analytical_day(base,cal)
@pytest.mark.django_db
def test_output_is_idempotent_and_no_lag_fields(tmp_path):
    _refs();base=_base(tmp_path/'b.parquet',[['BIL01','CP_P001',date(2025,5,1),1,1,1,1,'AVAILABLE']]);cal=_calendar(tmp_path/'c.xlsx',[date(2025,5,1)]);out,_=build_analytical_day(base,cal);path=tmp_path/'a.parquet';write_analytical_day(out,path);one=pd.read_parquet(path);write_analytical_day(out,path);two=pd.read_parquet(path)
    pd.testing.assert_frame_equal(one,two);assert not any(c.startswith(('lag_','rolling_')) for c in two.columns)
