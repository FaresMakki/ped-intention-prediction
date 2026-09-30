"""Contract tests that also exercise both real checkpoints."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_and_prediction_contract() -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["models_loaded"] is True
        assert health.json()["sample_count"] == 15

        samples = client.get("/api/samples").json()
        assert len(samples) == 15
        assert all(sample["pedestrian_count"] >= 2 for sample in samples)
        response = client.post(f"/api/samples/{samples[0]['id']}/predict")
        assert response.status_code == 200
        result = response.json()
        assert result["trajectory"]["checkpoint"] == "trajectory_model.pt"
        assert result["social"]["checkpoint"] == "social_model.pt"
        assert result["fusion"]["weights"] == {
            "no_crossing": 0.75,
            "crossing": 1.0,
            "irrelevant": 0.8,
        }

        comparison = client.post(f"/api/comparison/samples/{samples[0]['id']}/predict")
        assert comparison.status_code == 200
        comparison_result = comparison.json()
        assert comparison_result["pedestrian_count"] == samples[0][
            "pedestrian_count"
        ]
        assert len(comparison_result["pedestrians"]) == samples[0][
            "pedestrian_count"
        ]
        assert [method["id"] for method in comparison_result["global_methods"]] == [
            "trajectory_social",
            "relation_risk",
            "environment",
            "environment_skeleton",
            "skeleton",
        ]
        assert comparison_result["agreement"]["total_methods"] == 5
        for global_method in comparison_result["global_methods"]:
            individual = [
                next(
                    method
                    for method in person["methods"]
                    if method["id"] == global_method["id"]
                )["binary_prediction"]
                for person in comparison_result["pedestrians"]
            ]
            assert global_method["binary_prediction"] == int(any(individual))


def test_every_scene_is_parallel_and_uses_global_or() -> None:
    with TestClient(app) as client:
        summaries = client.get("/api/samples").json()
        for summary in summaries:
            sample = client.get(f"/api/samples/{summary['id']}").json()
            pedestrian_count = sample["pedestrian_count"]
            assert 2 <= pedestrian_count <= 4
            assert len(sample["pedestrians"]) == pedestrian_count
            assert all(
                len(frame["pedestrian_boxes"]) == pedestrian_count
                for frame in sample["frames"]
            )
            assert all(
                person["box"] is not None
                for frame in sample["frames"]
                for person in frame["pedestrian_boxes"]
            )

        response = client.post("/api/samples/sample-06/predict-scene")
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["pedestrian_count"] == 3
        individual = [
            prediction["fusion"]["binary_prediction"]
            for prediction in result["predictions"]
        ]
        assert result["global_decision"]["binary_prediction"] == int(any(individual))
        assert len(result["global_decision"]["crossing_pedestrian_ids"]) == sum(
            individual
        )

        batch = client.post("/api/predict-all")
        assert batch.status_code == 200, batch.text
        batch_result = batch.json()
        assert batch_result["total"] == 15
        assert all(
            prediction["pedestrian_count"] >= 2
            for prediction in batch_result["predictions"]
        )


def test_unknown_sample_is_not_found() -> None:
    with TestClient(app) as client:
        assert client.get("/api/samples/does-not-exist").status_code == 404
