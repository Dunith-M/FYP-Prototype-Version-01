from decimal import Decimal
from analytics.mapping_coverage import Coverage,percentage
import pytest
from pathlib import Path
from openpyxl import Workbook
from catalog.models import Product,SkuMapping
from analytics.mapping_coverage import calculate_mapping_coverage
def test_coverage_formulas_and_reconciliation():
    c=Coverage(10,7,3);assert c.pct==Decimal(70) and c.reconciles
def test_signed_and_absolute_values_preserve_negative_units():
    signed=Coverage(Decimal(4),Decimal(5),Decimal(-1));absolute=Coverage(Decimal(6),Decimal(5),Decimal(1));assert signed.reconciles and absolute.reconciles and signed.pct==Decimal(125)
def test_zero_denominator_is_safe(): assert percentage(0,0) is None
def _sales(path,rows):
    w=Workbook();s=w.active;s.title='data';s.append(['SALES_PART','TOTAL_UNITS'])
    for r in rows:s.append(r)
    w.save(path);return path
def _mapped(code='A'):
    p=Product.objects.create(canonical_product_id='CP_'+code,product_code=code,product_desc=code);SkuMapping.objects.create(source_system='SALES',source_product_code=code,canonical_product=p,mapping_reason='EXACT_CODE',confidence_class='CONFIRMED')
def _unresolved(code='B'): SkuMapping.objects.create(source_system='SALES',source_product_code=code,mapping_reason='MANUAL_REVIEW',confidence_class='UNRESOLVED')
@pytest.mark.django_db
def test_confirmed_canonical_mapping_counts_as_mapped(tmp_path):
    _mapped();_unresolved();r=calculate_mapping_coverage(_sales(tmp_path/'s.xlsx',[['A',2],['B',1]]));assert (r.distinct_codes.mapped,r.rows.mapped)==(1,1)
@pytest.mark.django_db
def test_unresolved_is_not_counted_as_mapped(tmp_path):
    _unresolved();r=calculate_mapping_coverage(_sales(tmp_path/'s.xlsx',[['B',2]]));assert r.rows.mapped==0 and r.rows.unresolved==1
@pytest.mark.django_db
def test_negative_and_zero_source_units_are_preserved(tmp_path):
    _mapped();r=calculate_mapping_coverage(_sales(tmp_path/'s.xlsx',[['A',5],['A',-1],['A',0]]));assert r.signed_units.mapped==Decimal(4) and r.absolute_units.mapped==Decimal(6) and r.rows.mapped==3
@pytest.mark.django_db
def test_report_calculation_does_not_modify_mappings(tmp_path):
    _mapped();before=list(SkuMapping.objects.values_list('id','confidence_class','canonical_product_id'));calculate_mapping_coverage(_sales(tmp_path/'s.xlsx',[['A',1]]));assert list(SkuMapping.objects.values_list('id','confidence_class','canonical_product_id'))==before
