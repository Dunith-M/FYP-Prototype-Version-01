"""Read-only product-mapping coverage metrics for historical Sales."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from catalog.models import SkuMapping
from catalog.sku_mapping import extract_sales_source_metrics

def percentage(mapped, total):
    return None if total == 0 else Decimal(mapped) * Decimal(100) / Decimal(total)
@dataclass(frozen=True)
class Coverage:
    total: Decimal|int; mapped: Decimal|int; unresolved: Decimal|int
    @property
    def pct(self): return percentage(self.mapped,self.total)
    @property
    def reconciles(self): return self.total == self.mapped + self.unresolved
@dataclass(frozen=True)
class MappingCoverageReport:
    distinct_codes:Coverage; rows:Coverage; signed_units:Coverage; absolute_units:Coverage; unresolved_impact:tuple[tuple[str,int,Decimal,Decimal],...]
def calculate_mapping_coverage(sales_file:Path)->MappingCoverageReport:
    source=extract_sales_source_metrics(sales_file); codes=set(source.codes)
    mapped_codes=set(SkuMapping.objects.filter(source_system=SkuMapping.SourceSystem.SALES,confidence_class=SkuMapping.ConfidenceClass.CONFIRMED,canonical_product__isnull=False).values_list('source_product_code',flat=True)) & codes
    unresolved=codes-mapped_codes
    def total(attr, selected): return sum((getattr(source.codes[c],attr) for c in selected),start=Decimal(0))
    impact=tuple(sorted(((c,source.codes[c].row_count,source.codes[c].signed_units,source.codes[c].absolute_units) for c in unresolved),key=lambda x:(-x[1],-x[3],x[0])))
    return MappingCoverageReport(Coverage(len(codes),len(mapped_codes),len(unresolved)),Coverage(source.total_rows,sum(source.codes[c].row_count for c in mapped_codes),sum(source.codes[c].row_count for c in unresolved)),Coverage(total('signed_units',codes),total('signed_units',mapped_codes),total('signed_units',unresolved)),Coverage(total('absolute_units',codes),total('absolute_units',mapped_codes),total('absolute_units',unresolved)),impact)
