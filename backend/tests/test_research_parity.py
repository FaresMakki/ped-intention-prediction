"""Optional parity test against the original experiment artifacts."""

import csv

import numpy as np
import pytest

from app.config import settings
from app.repositories.sample_repository import SampleRepository
from app.services.inference_service import InferenceService


def test_prepared_samples_reproduce_research_outputs() -> None:
    research_root = settings.deployment_root.parent / "PIE"
    precomputed = (
        research_root
        / "BinaryPrediction/FusionThreeClassTrajectorySocialClasswiseLocked"
        / "assets/source_model_outputs"
    )
    prediction_csv = (
        research_root
        / "outputs/BinaryPrediction/FusionThreeClassTrajectorySocialClasswiseLocked"
        / "predictions_test_rule_B.csv"
    )
    if not precomputed.exists() or not prediction_csv.exists():
        pytest.skip("Original research artifacts are not part of this deployment.")

    repository = SampleRepository(settings.manifest_path, settings.tensors_path)
    service = InferenceService(
        repository,
        settings.trajectory_checkpoint,
        settings.social_checkpoint,
    )
    trajectory_reference = np.load(precomputed / "traj_logits.npy")
    social_reference = np.load(precomputed / "social_logits.npy")
    rows = {
        int(row["sample_index"]): row
        for row in csv.DictReader(prediction_csv.open(encoding="utf-8-sig"))
    }

    for sample in repository.list_samples():
        result = service.predict(sample["id"])
        source_index = sample["sample_index"]
        trajectory = np.array(list(result["trajectory"]["logits"].values()))
        social = np.array(list(result["social"]["logits"].values()))
        np.testing.assert_allclose(
            trajectory,
            trajectory_reference[source_index],
            atol=1e-5,
        )
        np.testing.assert_allclose(social, social_reference[source_index], atol=1e-5)
        assert result["fusion"]["binary_prediction"] == int(
            rows[source_index]["fixed_prediction"]
        )
