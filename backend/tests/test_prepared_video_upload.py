"""End-to-end checks for matched prepared-video and data uploads."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = DEPLOYMENT_ROOT / "prepared_video_examples"
EXAMPLE_FOLDERS = (
    "01_no_crossing_4ped",
    "02_crossing_4ped",
    "03_crossing_4ped",
    "04_irrelevant_4ped",
    "05_no_crossing_4ped",
)


def _files(folder_name: str) -> dict:
    folder = EXAMPLE_ROOT / folder_name
    return {
        "video": ("clip.mp4", (folder / "clip.mp4").read_bytes(), "video/mp4"),
        "package": (
            "clip_data.zip",
            (folder / "clip_data.zip").read_bytes(),
            "application/zip",
        ),
    }


def test_all_prepared_video_examples_upload_and_run_parallel_inference() -> None:
    with TestClient(app) as client:
        assert len(client.get("/api/samples").json()) == 15
        for folder_name in EXAMPLE_FOLDERS:
            download = client.get(
                f"/prepared-video-examples/{folder_name}/clip.mp4"
            )
            assert download.status_code == 200
            assert download.headers["content-type"] == "video/mp4"
            response = client.post(
                "/api/prepared-video/uploads",
                files=_files(folder_name),
            )
            assert response.status_code == 201, response.text
            payload = response.json()
            sample = payload["sample"]
            video = payload["video"]
            assert sample["source"] == "prepared-video"
            assert sample["pedestrian_count"] == 4
            assert video["file_name"] == "clip.mp4"
            assert video["frame_count"] == 15
            assert video["playback_fps"] == 10.0
            assert len(video["sha256"]) == 64

            served = client.get(video["url"])
            assert served.status_code == 200
            assert served.headers["content-type"] == "video/mp4"
            assert served.content[4:8] == b"ftyp"

            prediction = client.post(
                f"/api/samples/{sample['id']}/predict-scene"
            ).json()
            assert prediction["pedestrian_count"] == 4
            assert len(prediction["predictions"]) == 4
            assert {
                person["pedestrian_id"] for person in prediction["predictions"]
            } == {
                person["pedestrian_id"] for person in sample["pedestrians"]
            }

        assert len(client.get("/api/samples").json()) == 20


def test_mismatched_video_and_data_are_rejected_before_registration() -> None:
    first = EXAMPLE_ROOT / EXAMPLE_FOLDERS[0]
    second = EXAMPLE_ROOT / EXAMPLE_FOLDERS[1]
    files = {
        "video": ("clip.mp4", (first / "clip.mp4").read_bytes(), "video/mp4"),
        "package": (
            "clip_data.zip",
            (second / "clip_data.zip").read_bytes(),
            "application/zip",
        ),
    }
    with TestClient(app) as client:
        response = client.post("/api/prepared-video/uploads", files=files)
        assert response.status_code == 400
        assert "do not match" in response.json()["detail"]
        assert len(client.get("/api/samples").json()) == 15


def test_regular_package_without_video_identity_is_rejected() -> None:
    regular = DEPLOYMENT_ROOT / "example_packages" / "pie_crossing_example.zip"
    files = {
        "video": ("clip.mp4", b"\x00\x00\x00\x18ftypisom", "video/mp4"),
        "package": ("clip_data.zip", regular.read_bytes(), "application/zip"),
    }
    with TestClient(app) as client:
        response = client.post("/api/prepared-video/uploads", files=files)
        assert response.status_code == 400
        assert "no prepared-video metadata" in response.json()["detail"]
