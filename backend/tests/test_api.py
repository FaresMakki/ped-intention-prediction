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


def test_unknown_sample_is_not_found() -> None:
    with TestClient(app) as client:
        assert client.get("/api/samples/does-not-exist").status_code == 404
