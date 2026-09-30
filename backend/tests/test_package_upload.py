"""End-to-end checks for portable package upload and inference."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "pie_no_crossing_example.zip": "sample-01",
    "pie_crossing_example.zip": "sample-07",
    "pie_irrelevant_example.zip": "sample-12",
}


def test_example_packages_upload_and_reproduce_prepared_inference() -> None:
    with TestClient(app) as client:
        uploaded_ids: dict[str, str] = {}
        for filename, prepared_id in EXAMPLES.items():
            content = (DEPLOYMENT_ROOT / "example_packages" / filename).read_bytes()
            response = client.post(
                "/api/uploads",
                content=content,
                headers={"Content-Type": "application/zip"},
            )
            assert response.status_code == 201, response.text
            payload = response.json()
            sample = payload["sample"]
            uploaded_ids[prepared_id] = sample["id"]
            assert sample["source"] == "upload"
            assert payload["summary"]["source"] == "upload"
            prepared_sample = client.get(f"/api/samples/{prepared_id}").json()
            assert sample["pedestrian_count"] == prepared_sample["pedestrian_count"]
            assert sample["pedestrian_count"] >= 2
            assert len(sample["frames"]) == 15
            assert all(
                len(frame["pedestrian_boxes"]) == sample["pedestrian_count"]
                for frame in sample["frames"]
            )

            frame = client.get(sample["frames"][7]["url"])
            assert frame.status_code == 200
            assert frame.headers["content-type"] == "image/jpeg"
            assert frame.content.startswith(b"\xff\xd8\xff")

            uploaded = client.post(f"/api/samples/{sample['id']}/predict").json()
            prepared = client.post(f"/api/samples/{prepared_id}/predict").json()
            assert uploaded["sample_index"] == prepared["sample_index"]
            assert uploaded["trajectory"]["logits"] == prepared["trajectory"]["logits"]
            assert uploaded["social"]["logits"] == prepared["social"]["logits"]
            assert uploaded["fusion"] == prepared["fusion"]

            uploaded_scene = client.post(
                f"/api/samples/{sample['id']}/predict-scene"
            ).json()
            prepared_scene = client.post(
                f"/api/samples/{prepared_id}/predict-scene"
            ).json()
            assert uploaded_scene["pedestrian_count"] == sample["pedestrian_count"]
            assert uploaded_scene["global_decision"] == prepared_scene["global_decision"]
            for uploaded_person, prepared_person in zip(
                uploaded_scene["predictions"],
                prepared_scene["predictions"],
                strict=True,
            ):
                assert uploaded_person["pedestrian_id"] == prepared_person["pedestrian_id"]
                assert uploaded_person["trajectory"] == prepared_person["trajectory"]
                assert uploaded_person["social"] == prepared_person["social"]
                assert uploaded_person["fusion"] == prepared_person["fusion"]

        samples = client.get("/api/samples").json()
        assert len(samples) == 18
        assert all(sample["source"] == "upload" for sample in samples[:3])

        uploaded_comparison = client.post(
            f"/api/comparison/samples/{uploaded_ids['sample-07']}/predict"
        ).json()
        prepared_comparison = client.post(
            "/api/comparison/samples/sample-07/predict"
        ).json()
        assert uploaded_comparison["methods"] == prepared_comparison["methods"]
        assert uploaded_comparison["agreement"] == prepared_comparison["agreement"]


def test_invalid_package_is_rejected_without_registration() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            content=b"this is not a zip file",
            headers={"Content-Type": "application/zip"},
        )
        assert response.status_code == 400
        assert "valid ZIP" in response.json()["detail"]
        assert len(client.get("/api/samples").json()) == 15


def test_legacy_version_one_package_remains_supported() -> None:
    content = (
        DEPLOYMENT_ROOT
        / "example_packages"
        / "pie_legacy_v1_single_person.zip"
    ).read_bytes()
    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            content=content,
            headers={"Content-Type": "application/zip"},
        )
        assert response.status_code == 201, response.text
        sample = response.json()["sample"]
        assert sample["pedestrian_count"] == 1
        scene = client.post(f"/api/samples/{sample['id']}/predict-scene").json()
        assert scene["pedestrian_count"] == 1
