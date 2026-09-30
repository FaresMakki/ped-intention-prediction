"""Contract tests that also exercise both real checkpoints."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_and_prediction_contract() -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["models_loaded"] is True
        assert health.json()["sample_count"] == 16

        samples = client.get("/api/samples").json()
        assert len(samples) == 16
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
        assert [method["id"] for method in comparison_result["methods"]] == [
            "trajectory_social",
            "relation_risk",
            "environment",
            "environment_skeleton",
            "skeleton",
        ]
        assert comparison_result["agreement"]["total_methods"] == 5


def test_parallel_scene_runs_each_pedestrian_and_uses_global_or() -> None:
    with TestClient(app) as client:
        sample = client.get("/api/samples/sample-16").json()
        assert sample["pedestrian_count"] == 2
        assert [person["pedestrian_id"] for person in sample["pedestrians"]] == [
            "3_8_547",
            "3_8_548",
        ]
        assert all(len(frame["pedestrian_boxes"]) == 2 for frame in sample["frames"])
        assert all(
            person["box"] is not None
            for frame in sample["frames"]
            for person in frame["pedestrian_boxes"]
        )

        response = client.post("/api/samples/sample-16/predict-scene")
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["pedestrian_count"] == 2
        assert [
            prediction["fusion"]["binary_prediction"]
            for prediction in result["predictions"]
        ] == [1, 0]
        assert result["global_decision"]["binary_prediction"] == 1
        assert result["global_decision"]["binary_label"] == "Crossing"
        assert result["global_decision"]["crossing_pedestrian_ids"] == ["3_8_547"]
        assert result["global_decision"]["not_crossing_pedestrian_ids"] == [
            "3_8_548"
        ]


def test_unknown_sample_is_not_found() -> None:
    with TestClient(app) as client:
        assert client.get("/api/samples/does-not-exist").status_code == 404
