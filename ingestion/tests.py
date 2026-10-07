import inspect
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.test import TestCase
from django.urls import reverse

from .models import SourceFile
from .services.source_registration import register_contracted_source_files
from .views import data_status


class DataStatusPageTests(TestCase):
    def test_page_displays_persisted_latest_metadata_without_writes(self):
        SourceFile.objects.create(
            dataset=SourceFile.Dataset.SALES,
            filename="older_sales.xlsx",
            file_hash="a" * 64,
            row_count=10,
            contract_version="sales-v1",
            status=SourceFile.Status.REGISTERED,
        )
        current = SourceFile.objects.create(
            dataset=SourceFile.Dataset.SALES,
            filename="current_sales.xlsx",
            file_hash="b" * 64,
            row_count=67385,
            contract_version="sales-v2",
            status=SourceFile.Status.VALIDATED,
        )

        response = self.client.get(reverse("ingestion:data_status"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Data Status")
        self.assertContains(response, "current_sales.xlsx")
        self.assertNotContains(response, "older_sales.xlsx")
        self.assertContains(response, str(current.row_count))
        self.assertContains(response, current.contract_version)
        self.assertContains(response, current.status)
        self.assertContains(response, current.file_hash[:12])
        self.assertContains(response, "Latest of 2 registered versions")
        self.assertContains(response, "Inventory")
        self.assertContains(response, "Not registered")
        self.assertContains(response, reverse("data_quality"))
        self.assertEqual(SourceFile.objects.count(), 2)

    def test_page_handles_an_empty_sourcefile_table(self):
        response = self.client.get(reverse("ingestion:data_status"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No source files have been registered yet.")
        for label in ("Sales", "Inventory", "Products", "Calendar", "Branches"):
            self.assertContains(response, label)

    def test_view_does_not_open_or_process_raw_source_files(self):
        source = inspect.getsource(data_status)
        for forbidden in ("read_excel", "read_parquet", "openpyxl", "pandas", "data/raw"):
            self.assertNotIn(forbidden, source)


class SourceRegistrationTests(TestCase):
    def test_registration_persists_real_hashes_and_is_idempotent(self):
        contracts = {
            dataset: {
                "dataset": dataset,
                "source_file": f"{dataset}.xlsx",
                "source_sheet": "data",
                "version": 1,
            }
            for dataset in ("sales", "inventory", "products", "calendar", "branches")
        }
        raw_root = Path("contracted-test-source-root")
        with patch("ingestion.services.source_registration.load_contract", side_effect=contracts.get), patch(
            "ingestion.services.source_registration.Path.is_file", return_value=True
        ), patch(
            "ingestion.services.source_registration._sha256_file",
            side_effect=lambda path: sha256(f"real bytes for {path.name}".encode()).hexdigest(),
        ), patch(
            "ingestion.services.source_registration.pd.read_excel",
            return_value=pd.DataFrame({"evidence": [1, 2, 3]}),
        ):
            first = register_contracted_source_files(raw_root)
            second = register_contracted_source_files(raw_root)

        self.assertEqual(SourceFile.objects.count(), 5)
        self.assertTrue(all(result.created for result in first))
        self.assertTrue(all(not result.created for result in second))
        sales = SourceFile.objects.get(dataset=SourceFile.Dataset.SALES)
        self.assertEqual(sales.filename, "sales.xlsx")
        self.assertEqual(sales.row_count, 3)
        self.assertEqual(sales.contract_version, "sales-v1")
        self.assertEqual(sales.status, SourceFile.Status.REGISTERED)
        self.assertEqual(len(sales.file_hash), 64)
