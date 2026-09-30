#!/usr/bin/env python3
"""Export three deterministic, upload-ready packages from the demo samples."""

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
    "sample-06": "pie_crossing_example.zip",
    "sample-11": "pie_irrelevant_example.zip",
}
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


def tensor_archive(arrays: dict[str, np.ndarray], index: int) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        **{name: np.array(arrays[name][index], copy=True) for name in ARRAY_NAMES},
    )
    return stream.getvalue()


def package_manifest(sample: dict) -> dict:
    return {
        "format": "pie-crossing-package",
        "version": 1,
        "frame_count": 15,
        "tensors_file": "tensors.npz",
        "display_name": f"Uploaded {sample['class_name']} example",
        "sample_index": sample["sample_index"],
        "set_id": sample["set_id"],
        "video_id": sample["video_id"],
        "pedestrian_id": sample["pedestrian_id"],
        "class_name": sample["class_name"],
        "raw_class": sample["raw_class"],
        "binary_target": sample["binary_target"],
        "frames": [
            {
                "ordinal": frame["ordinal"],
                "source_frame": frame["source_frame"],
                "file": f"frames/frame_{frame['ordinal']:02d}.jpg",
                "target_box": frame["target_box"],
            }
            for frame in sample["frames"]
        ],
    }


def write_package(
    destination: Path,
    sample: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    manifest = package_manifest(sample)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    tensors = tensor_archive(arrays, int(sample["tensor_index"]))
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


if __name__ == "__main__":
    main()
