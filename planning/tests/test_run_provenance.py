import hashlib
import json
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import TestCase

from catalog.models import Branch, Product
from ingestion.models import SourceFile
from planning.models import DecisionOption, OptimizationResult, Recommendation, RunProvenance
from planning.modes import ScenarioMode
from planning.services.run_provenance import capture_run_provenance


class RunProvenanceTests(TestCase):
    source_hashes = {
        "sales": "1" * 64,
        "inventory": "2" * 64,
        "products": "3" * 64,
        "calendar": "4" * 64,
        "branches": "5" * 64,
    }

    def setUp(self):
        self.branch = Branch.objects.create(branch_code="BIL01", branch_name="GALLE-A")
        self.product = Product.objects.create(
            canonical_product_id="CP_TRACE", product_code="TRACE", product_desc="Trace product",
            family_id="F1", family_desc="Trace family",
        )
        self.result = OptimizationResult.objects.create(
            planning_month=date(2026, 4, 1), scenario_version="synthetic-test-v1",
            run_version="run-trace-test-001", solver_status=OptimizationResult.SolverStatus.OPTIMAL,
            feasibility_status=OptimizationResult.FeasibilityStatus.PASS,
        )
        self.option = DecisionOption.objects.create(
            branch=self.branch, product=self.product, planning_month=date(2026, 4, 1),
            l0=True, q0=1, l1=True, q1=2,
            option_type=DecisionOption.OptionType.LISTED_TARGET,
            parameter_source=DecisionOption.ParameterSource.SYNTHETIC_DEMO,
            scenario_version="synthetic-test-v1",
        )
        self.result.selected_options.add(self.option)
        self.recommendation = Recommendation.objects.create(
            optimization_result=self.result, selected_option=self.option,
            branch=self.branch, product=self.product,
            action=Recommendation.Action.KEEP, status=Recommendation.Status.READY,
        )
        self.source_files = {
            dataset: SourceFile.objects.create(
                dataset=dataset, filename=f"{dataset}-v1.xlsx", file_hash=file_hash,
                row_count=1, contract_version="contract-v1", status=SourceFile.Status.VALIDATED,
            )
            for dataset, file_hash in self.source_hashes.items()
        }

    def _artifact_paths(self, directory: Path):
        feature_manifest = directory / "feature_manifest.json"
        model_manifest = directory / "model_manifest.json"
        synthetic_config = directory / "options.yaml"
        feature_manifest.write_text(json.dumps({
            "artifact_hash": "a" * 64,
            "feature_version": "feature-v1",
            "mapping_version": "mapping-v1",
            "sources": {
                dataset: {"filename": f"{dataset}-v1.xlsx", "hash": file_hash}
                for dataset, file_hash in self.source_hashes.items()
            },
        }), encoding="utf-8")
        model_manifest.write_text(json.dumps({
            "model_type": "Baseline: trailing_mean_3",
            "model_version": "model-v1",
            "selected_model": "trailing_mean_3",
        }), encoding="utf-8")
        synthetic_config.write_text("mode: RESEARCH_SYNTHETIC\nparameter_source: SYNTHETIC_DEMO\n", encoding="utf-8")
        return feature_manifest, model_manifest, synthetic_config

    def _capture(self, *, mode=ScenarioMode.RESEARCH_SYNTHETIC, return_paths=False):
        with TemporaryDirectory() as temporary_directory:
            paths = self._artifact_paths(Path(temporary_directory))
            provenance = capture_run_provenance(
                optimization_result=self.result, branch=self.branch, mode=mode,
                feature_manifest_path=paths[0], model_manifest_path=paths[1], synthetic_config_path=paths[2],
            )
            if return_paths:
                return provenance, hashlib.sha256(paths[2].read_bytes()).hexdigest()
            return provenance

    def test_recommendation_is_traceable_to_exact_run_and_frozen_versions(self):
        provenance, expected_config_hash = self._capture(return_paths=True)
        self.recommendation.refresh_from_db()

        self.assertEqual(self.recommendation.optimization_result_id, self.result.pk)
        self.assertEqual(self.recommendation.optimization_result.provenance.pk, provenance.pk)
        self.assertEqual(provenance.branch, self.branch)
        self.assertEqual(provenance.mode, ScenarioMode.RESEARCH_SYNTHETIC)
        self.assertEqual(provenance.mapping_version, "mapping-v1")
        self.assertEqual(provenance.model_version, "model-v1")
        self.assertEqual(provenance.selected_model, "trailing_mean_3")
        self.assertEqual(provenance.scenario_config_hash, expected_config_hash)
        self.assertEqual(provenance.source_files.count(), 5)

    def test_old_run_keeps_exact_source_versions_when_new_source_is_registered(self):
        provenance = self._capture()
        original_ids = set(provenance.source_files.values_list("pk", flat=True))
        SourceFile.objects.create(
            dataset=SourceFile.Dataset.SALES, filename="sales-v2.xlsx", file_hash="6" * 64,
            row_count=2, contract_version="contract-v2", status=SourceFile.Status.VALIDATED,
        )

        provenance.refresh_from_db()
        self.assertSetEqual(set(provenance.source_files.values_list("pk", flat=True)), original_ids)
        self.assertEqual(provenance.source_snapshot["sales"]["file_hash"], self.source_hashes["sales"])

    def test_historical_run_has_no_false_synthetic_configuration_provenance(self):
        provenance = self._capture(mode=ScenarioMode.HISTORICAL)

        self.assertEqual(provenance.mode, ScenarioMode.HISTORICAL)
        self.assertEqual(provenance.scenario_config_path, "")
        self.assertEqual(provenance.scenario_config_hash, "")

    def test_existing_provenance_is_not_replaced_by_different_snapshot(self):
        self._capture()
        with self.assertRaisesMessage(Exception, "different immutable provenance snapshot"):
            with TemporaryDirectory() as temporary_directory:
                paths = self._artifact_paths(Path(temporary_directory))
                Path(paths[0]).write_text(json.dumps({
                    "artifact_hash": "b" * 64,
                    "feature_version": "feature-v2",
                    "mapping_version": "mapping-v2",
                    "sources": {
                        dataset: {"filename": f"{dataset}-v1.xlsx", "hash": file_hash}
                        for dataset, file_hash in self.source_hashes.items()
                    },
                }), encoding="utf-8")
                capture_run_provenance(
                    optimization_result=self.result, branch=self.branch, mode=ScenarioMode.RESEARCH_SYNTHETIC,
                    feature_manifest_path=paths[0], model_manifest_path=paths[1], synthetic_config_path=paths[2],
                )
