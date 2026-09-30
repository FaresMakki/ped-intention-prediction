#!/usr/bin/env python3
"""Export deterministic v2 multi-pedestrian and legacy v1 example packages."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = DEPLOYMENT_ROOT / "data"
OUTPUT_ROOT = DEPLOYMENT_ROOT / "example_packages"
EXAMPLES = {
    "sample-01": "pie_no_crossing_example.zip",
    "sample-07": "pie_crossing_example.zip",
    "sample-12": "pie_irrelevant_example.zip",
}
LEGACY_EXAMPLE = ("sample-01", "pie_legacy_v1_single_person.zip")
ARRAY_NAMES = (
    "trajectory_nodes",
    "trajectory_adjacency",
    "trajectory_mask",
    "social_nodes",
    "social_adjacency",
    "social_mask",
    "environment_nodes",
    "environment_adjacency",
    "environment_mask",
    "skeleton_input",
    "skeleton_keypoint_mask",
    "skeleton_frame_mask",
    "relation_risk_nodes",
    "relation_risk_adjacency",
    "relation_risk_mask",
)


def zip_info(name: str, compression: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.external_attr = 0o100644 << 16
    return info


def tensor_archive(
    arrays: dict[str, np.ndarray],
    indices: list[int],
    *,
    batched: bool,
) -> bytes:
    stream = io.BytesIO()
    selected = {
        name: np.array(
            arrays[name][indices] if batched else arrays[name][indices[0]],
            copy=True,
        )
        for name in ARRAY_NAMES
    }
    np.savez_compressed(
        stream,
        **selected,
    )
    return stream.getvalue()


def package_manifest_v2(sample: dict) -> dict:
    return {
        "format": "pie-crossing-package",
        "version": 2,
        "frame_count": 15,
        "tensors_file": "tensors.npz",
        "display_name": f"Uploaded multi-ped {sample['class_name']} scene",
        "set_id": sample["set_id"],
        "video_id": sample["video_id"],
        "pedestrians": [
            {
                "sample_index": pedestrian["sample_index"],
                "pedestrian_id": pedestrian["pedestrian_id"],
                "class_name": pedestrian["class_name"],
                "raw_class": pedestrian["raw_class"],
                "binary_target": pedestrian["binary_target"],
                "color": pedestrian["color"],
            }
            for pedestrian in sample["pedestrians"]
        ],
        "frames": [
            {
                "ordinal": frame["ordinal"],
                "source_frame": frame["source_frame"],
                "file": f"frames/frame_{frame['ordinal']:02d}.jpg",
                "pedestrian_boxes": [
                    {
                        "pedestrian_id": box["pedestrian_id"],
                        "box": box["box"],
                    }
                    for box in frame["pedestrian_boxes"]
                ],
            }
            for frame in sample["frames"]
        ],
    }


def package_manifest_v1(sample: dict) -> dict:
    pedestrian = sample["pedestrians"][0]
    return {
        "format": "pie-crossing-package",
        "version": 1,
        "frame_count": 15,
        "tensors_file": "tensors.npz",
        "display_name": "Legacy single-pedestrian example",
        "sample_index": pedestrian["sample_index"],
        "set_id": sample["set_id"],
        "video_id": sample["video_id"],
        "pedestrian_id": pedestrian["pedestrian_id"],
        "class_name": pedestrian["class_name"],
        "raw_class": pedestrian["raw_class"],
        "binary_target": pedestrian["binary_target"],
        "frames": [
            {
                "ordinal": frame["ordinal"],
                "source_frame": frame["source_frame"],
                "file": f"frames/frame_{frame['ordinal']:02d}.jpg",
                "target_box": frame["pedestrian_boxes"][0]["box"],
            }
            for frame in sample["frames"]
        ],
    }


def write_package(
    destination: Path,
    sample: dict,
    arrays: dict[str, np.ndarray],
    *,
    legacy: bool = False,
) -> None:
    manifest = package_manifest_v1(sample) if legacy else package_manifest_v2(sample)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    tensor_indices = [
        int(pedestrian["tensor_index"])
        for pedestrian in sample["pedestrians"]
    ]
    tensors = tensor_archive(
        arrays,
        tensor_indices[:1] if legacy else tensor_indices,
        batched=not legacy,
    )
    with zipfile.ZipFile(destination, "w", allowZip64=False) as archive:
        archive.writestr(
            zip_info("manifest.json", zipfile.ZIP_DEFLATED),
            manifest_bytes,
        )
        archive.writestr(
            zip_info("tensors.npz", zipfile.ZIP_STORED),
            tensors,
        )
        for frame in sample["frames"]:
            name = f"frame_{frame['ordinal']:02d}.jpg"
            content = (DATA_ROOT / "frames" / sample["id"] / name).read_bytes()
            archive.writestr(
                zip_info(f"frames/{name}", zipfile.ZIP_STORED),
                content,
            )


def main() -> None:
    payload = json.loads((DATA_ROOT / "samples.json").read_text(encoding="utf-8"))
    samples = {sample["id"]: sample for sample in payload["samples"]}
    with np.load(DATA_ROOT / "prepared_samples.npz", allow_pickle=False) as source:
        arrays = {name: source[name] for name in ARRAY_NAMES}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sample_id, filename in EXAMPLES.items():
        destination = OUTPUT_ROOT / filename
        write_package(destination, samples[sample_id], arrays)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        print(f"{destination.name}: {destination.stat().st_size} bytes, sha256={digest}")
    sample_id, filename = LEGACY_EXAMPLE
    destination = OUTPUT_ROOT / filename
    write_package(destination, samples[sample_id], arrays, legacy=True)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"{destination.name}: {destination.stat().st_size} bytes, sha256={digest}")


if __name__ == "__main__":
    main()
