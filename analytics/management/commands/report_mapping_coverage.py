import csv
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand
from analytics.mapping_coverage import calculate_mapping_coverage
class Command(BaseCommand):
    help='Report Sales product-mapping coverage without changing mappings.'
    def add_arguments(self,p):
        p.add_argument('--sales-file',type=Path,default=settings.BASE_DIR/'data'/'raw'/'transactions_data.xlsx');p.add_argument('--output',type=Path,default=settings.BASE_DIR/'data'/'artifacts'/'mapping'/'mapping_coverage.csv');p.add_argument('--impact-output',type=Path,default=settings.BASE_DIR/'data'/'artifacts'/'mapping'/'unresolved_impact.csv')
    def handle(self,*a,**o):
        r=calculate_mapping_coverage(o['sales_file']);o['output'].parent.mkdir(parents=True,exist_ok=True)
        with o['output'].open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['metric','total','mapped','unresolved','coverage_pct'])
            for n,c in [('distinct_product_codes',r.distinct_codes),('sales_rows',r.rows),('signed_sales_units',r.signed_units),('absolute_sales_units',r.absolute_units)]:w.writerow([n,c.total,c.mapped,c.unresolved,'' if c.pct is None else f'{c.pct:.6f}'])
        with o['impact_output'].open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['source_product_code','affected_sales_rows','affected_signed_units','affected_absolute_units']);w.writerows(r.unresolved_impact)
        for n,c in [('Distinct codes',r.distinct_codes),('Sales rows',r.rows),('Signed units',r.signed_units),('Absolute units',r.absolute_units)]:self.stdout.write(f'{n}: {c.mapped}/{c.total}; unresolved={c.unresolved}; coverage={"N/A" if c.pct is None else f"{c.pct:.2f}%"}')
        self.stdout.write(f'Coverage report: {o["output"]}');self.stdout.write(f'Impact report: {o["impact_output"]}')
