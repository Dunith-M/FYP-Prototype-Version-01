from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand,CommandError
from analytics.analytical_day import AnalyticalDayValidationError,build_analytical_day,write_analytical_day
class Command(BaseCommand):
    help="Enrich BRANCH_PRODUCT_DAY with validated Product, Calendar, and Branch references."
    def add_arguments(self,p):
        processed=settings.BASE_DIR/"data"/"processed"; raw=settings.BASE_DIR/"data"/"raw"
        p.add_argument("--base",type=Path,default=processed/"branch_product_day.parquet");p.add_argument("--calendar",type=Path,default=raw/"calendar_data.xlsx");p.add_argument("--output",type=Path,default=processed/"analytical_day.parquet")
    def handle(self,*a,**o):
        try: frame,m=build_analytical_day(o["base"],o["calendar"])
        except (AnalyticalDayValidationError,OSError,ValueError) as e: raise CommandError(str(e)) from e
        if m.units_before!=m.units_after or m.value_before!=m.value_after: raise CommandError("Reference joins changed sales totals.")
        write_analytical_day(frame,o["output"])
        self.stdout.write(self.style.SUCCESS("ANALYTICAL_DAY build complete."));self.stdout.write(f"rows={m.output_rows}; product={m.product_matched}/{m.base_rows}; calendar={m.calendar_matched}/{m.base_rows}; branch={m.branch_matched}/{m.base_rows}; unknown={m.unknown_after}");self.stdout.write(f"output={o['output']}")
