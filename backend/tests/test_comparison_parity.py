"""Parity checks for every model shown on the comparison page."""

import csv

import numpy as np

from app.config import settings
from app.repositories.sample_repository import SampleRepository
from app.services.comparison_service import ComparisonService
from app.services.inference_service import InferenceService


def _rows_by_index(path, *, checkpoint: str | None = None):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if checkpoint is not None:
        rows = [row for row in rows if row.get("checkpoint") == checkpoint]
    return {
        int(
            row["original_sample_index"]
            if "original_sample_index" in row
            else row["sample_index"]
        ): row
        for row in rows
    }


def test_comparison_models_reproduce_research_outputs() -> None:
    research_root = settings.deployment_root.parent / "PIE"
    precomputed_root = (
        research_root
        / "outputs/BinaryPrediction/FusionThreeClassEnvSkeletonLogitCorrection"
        / "precomputed_logits"
    )
    fusion_rows = _rows_by_index(
        research_root
        / "outputs/BinaryPrediction/FusionThreeClassEnvSkeletonLogitCorrection"
        / "uncertainty_gated_residual_correction/three_class_predictions_test.csv"
    )
    risk_rows = _rows_by_index(
        research_root
        / "outputs/BinaryPrediction/RelationRiskGraphThreeClass"
        / "three_class_predictions_test.csv",
        checkpoint="mapped_macro_f1",
    )
    environment_reference = np.load(precomputed_root / "env_logits.npy")
    skeleton_reference = np.load(precomputed_root / "skeleton_logits.npy")

    repository = SampleRepository(settings.manifest_path, settings.tensors_path)
    primary = InferenceService(
        repository,
        settings.trajectory_checkpoint,
        settings.social_checkpoint,
    )
    service = ComparisonService(
        repository,
        primary,
        settings.environment_checkpoint,
        settings.skeleton_checkpoint,
        settings.environment_skeleton_checkpoint,
        settings.relation_risk_checkpoint,
    )

    for sample in repository.list_samples():
        result = service.predict(sample["id"])
        assert result["pedestrian_count"] == sample["pedestrian_count"]
        for pedestrian in result["pedestrians"]:
            source_index = pedestrian["sample_index"]
            methods = {
                method["id"]: method for method in pedestrian["methods"]
            }
            environment = np.array(
                list(methods["environment"]["logits"].values())
            )
            skeleton = np.array(list(methods["skeleton"]["logits"].values()))
            np.testing.assert_allclose(
                environment,
                environment_reference[source_index],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                skeleton,
                skeleton_reference[source_index],
                atol=1e-5,
            )

            fusion_row = fusion_rows[source_index]
            expected_fusion = np.array(
                [
                    fusion_row["final_logit_no_crossing"],
                    fusion_row["final_logit_crossing"],
                    fusion_row["final_logit_irrelevant"],
                ],
                dtype=np.float64,
            )
            fusion = np.array(
                list(methods["environment_skeleton"]["logits"].values())
            )
            np.testing.assert_allclose(fusion, expected_fusion, atol=1e-5)

            risk_row = risk_rows[source_index]
            expected_risk = np.array(
                [
                    risk_row["logit_no_crossing"],
                    risk_row["logit_crossing"],
                    risk_row["logit_irrelevant"],
                ],
                dtype=np.float64,
            )
            risk = np.array(list(methods["relation_risk"]["logits"].values()))
            np.testing.assert_allclose(risk, expected_risk, atol=1e-5)
