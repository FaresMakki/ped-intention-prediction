#!/usr/bin/env python3
"""Export deterministic v2 multi-pedestrian and legacy v1 example packages."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from prepare_demo_data import (
    CLASS_NAMES,
    PEDESTRIAN_COLORS,
    RESEARCH_ROOT,
    export_frames,
    load_target_boxes,
    read_index,
    scene_class,
)

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = DEPLOYMENT_ROOT / "data"
OUTPUT_ROOT = DEPLOYMENT_ROOT / "example_packages"
PREPARED_VIDEO_ROOT = DEPLOYMENT_ROOT / "prepared_video_examples"
PREPARED_VIDEO_FPS = 10.0
PREPARED_EXAMPLES = {
    "sample-01": "pie_no_crossing_example.zip",
    "sample-07": "pie_crossing_example.zip",
    "sample-12": "pie_irrelevant_example.zip",
}
LEGACY_EXAMPLE = ("sample-01", "pie_legacy_v1_single_person.zip")
EXTERNAL_EXAMPLES = {
    "pie_extra_no_crossing_4ped.zip": (584, 585, 586, 587),
    "pie_extra_crossing_4ped_01.zip": (683, 684, 686, 687),
    "pie_extra_crossing_4ped_02.zip": (758, 759, 760, 762),
}
PREPARED_VIDEO_EXAMPLES = {
    "01_no_crossing_4ped": (584, 585, 586, 587),
    "02_crossing_4ped": (683, 684, 686, 687),
    "03_crossing_4ped": (758, 759, 760, 762),
    "04_irrelevant_4ped": (1043, 1044, 1045, 1046),
    "05_no_crossing_4ped": (1106, 1107, 1108, 1109),
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


def package_manifest_v2(sample: dict, video: dict | None = None) -> dict:
    manifest = {
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
    if video is not None:
        manifest["video"] = video
    return manifest


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
    frame_root: Path = DATA_ROOT / "frames",
    legacy: bool = False,
    video: dict | None = None,
) -> None:
    manifest = (
        package_manifest_v1(sample)
        if legacy
        else package_manifest_v2(sample, video)
    )
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
            content = (frame_root / sample["id"] / name).read_bytes()
            archive.writestr(
                zip_info(f"frames/{name}", zipfile.ZIP_STORED),
                content,
            )


def write_video(destination: Path, frame_directory: Path) -> dict:
    """Encode the exact 15 annotated observation frames for presentation."""
    paths = [frame_directory / f"frame_{ordinal:02d}.jpg" for ordinal in range(1, 16)]
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read {paths[0]} for video export.")
    height, width = first.shape[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        PREPARED_VIDEO_FPS,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}.")
    try:
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(f"Invalid prepared-video frame {path}.")
            writer.write(image)
    finally:
        writer.release()
    payload = destination.read_bytes()
    if len(payload) < 12 or payload[4:8] != b"ftyp":
        raise RuntimeError(f"Video encoder did not create a valid MP4: {destination}")
    return {
        "file_name": destination.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "frame_count": 15,
        "playback_fps": PREPARED_VIDEO_FPS,
        "source_fps": 30.0,
        "width": width,
        "height": height,
    }


def source_arrays() -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    trajectory_data = (
        RESEARCH_ROOT
        / "outputs/BinaryPrediction/PedTrajectoryContextGraphThreeClass"
        / "ped_trajectory_context_dataset"
    )
    social_data = (
        RESEARCH_ROOT
        / "outputs/BinaryPrediction/SocialInteractionGraphThreeClass"
        / "social_interaction_dataset"
    )
    environment_data = (
        RESEARCH_ROOT
        / "BinaryPrediction/EnvGraphThreeClassNodeMotionFeatures"
        / "assets/protected_baseline/env_graph_dataset"
    )
    skeleton_data = (
        RESEARCH_ROOT
        / "BinaryPrediction/SkeletonThreeClassPedGraphPlusGCN"
        / "assets/skeleton_19joint_dataset"
    )
    relation_risk_data = (
        RESEARCH_ROOT
        / "outputs/BinaryPrediction/RelationRiskGraphThreeClass"
        / "relation_risk_graph_dataset"
    )
    arrays = {
        "trajectory_nodes": np.load(trajectory_data / "X_traj_nodes.npy", mmap_mode="r"),
        "trajectory_adjacency": np.load(
            trajectory_data / "X_traj_adj.npy", mmap_mode="r"
        ),
        "trajectory_mask": np.load(
            trajectory_data / "X_traj_node_mask.npy", mmap_mode="r"
        ),
        "social_nodes": np.load(social_data / "X_social_nodes.npy", mmap_mode="r"),
        "social_adjacency": np.load(
            social_data / "X_social_adj.npy", mmap_mode="r"
        ),
        "social_mask": np.load(
            social_data / "X_social_node_mask.npy", mmap_mode="r"
        ),
        "environment_nodes": np.load(
            environment_data / "X_env_nodes.npy", mmap_mode="r"
        ),
        "environment_adjacency": np.load(
            environment_data / "X_env_adj.npy", mmap_mode="r"
        ),
        "environment_mask": np.load(
            environment_data / "X_env_node_mask.npy", mmap_mode="r"
        ),
        "skeleton_input": np.load(
            skeleton_data / "X_skeleton_input_19.npy", mmap_mode="r"
        ),
        "skeleton_keypoint_mask": np.load(
            skeleton_data / "X_keypoint_valid_mask_19.npy", mmap_mode="r"
        ),
        "skeleton_frame_mask": np.load(
            skeleton_data / "X_frame_pose_valid.npy", mmap_mode="r"
        ),
        "relation_risk_nodes": np.load(
            relation_risk_data / "X_relation_nodes.npy", mmap_mode="r"
        ),
        "relation_risk_adjacency": np.load(
            relation_risk_data / "X_relation_adj.npy", mmap_mode="r"
        ),
        "relation_risk_mask": np.load(
            relation_risk_data / "X_relation_node_mask.npy", mmap_mode="r"
        ),
    }
    return read_index(trajectory_data / "traj_graph_index.csv"), arrays


def external_sample(
    rows: list[dict[str, str]],
    sample_id: str,
    frame_root: Path,
    built_in_keys: set[tuple[str, str, tuple[int, ...]]],
) -> dict:
    primary = rows[0]
    frames = tuple(json.loads(primary["frames"]))
    scene_key = (primary["set_id"], primary["video_id"], frames)
    if scene_key in built_in_keys:
        raise RuntimeError(f"External package {sample_id} duplicates a built-in scene.")
    if any(
        (row["set_id"], row["video_id"], tuple(json.loads(row["frames"])))
        != scene_key
        for row in rows
    ):
        raise RuntimeError(f"External package {sample_id} is not frame-aligned.")
    if any(
        row["target_ped_found_all_15_frames"] not in {"1", "True", "true"}
        for row in rows
    ):
        raise RuntimeError(f"External package {sample_id} has incomplete annotations.")
    requested = {
        row["ped_id"]: set(json.loads(row["frames"]))
        for row in rows
    }
    boxes = load_target_boxes(
        RESEARCH_ROOT
        / "annotations"
        / primary["set_id"]
        / f"{primary['video_id']}_annt.xml",
        requested,
    )
    if len(boxes) != 15 * len(rows):
        raise RuntimeError(f"External package {sample_id} is missing target boxes.")
    frame_info = export_frames(
        rows,
        sample_id,
        boxes,
        frame_root=frame_root,
        url_prefix="/external",
    )
    pedestrians = []
    for index, row in enumerate(rows):
        raw_class = int(row["raw_crossing"])
        pedestrians.append(
            {
                "tensor_index": int(row["sample_index"]),
                "sample_index": int(row["sample_index"]),
                "pedestrian_id": row["ped_id"],
                "class_name": CLASS_NAMES[raw_class],
                "raw_class": raw_class,
                "binary_target": int(row["binary_crossing"]),
                "color": PEDESTRIAN_COLORS[index],
            }
        )
    raw_class = scene_class(rows)
    return {
        "id": sample_id,
        "display_name": f"External multi-ped {CLASS_NAMES[raw_class]} scene",
        "set_id": primary["set_id"],
        "video_id": primary["video_id"],
        "class_name": CLASS_NAMES[raw_class],
        "raw_class": raw_class,
        "binary_target": int(raw_class == 1),
        "pedestrians": pedestrians,
        "frames": frame_info,
    }


def print_package(destination: Path) -> None:
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"{destination.name}: {destination.stat().st_size} bytes, sha256={digest}")


def main() -> None:
    payload = json.loads((DATA_ROOT / "samples.json").read_text(encoding="utf-8"))
    samples = {sample["id"]: sample for sample in payload["samples"]}
    with np.load(DATA_ROOT / "prepared_samples.npz", allow_pickle=False) as source:
        arrays = {name: source[name] for name in ARRAY_NAMES}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sample_id, filename in PREPARED_EXAMPLES.items():
        destination = OUTPUT_ROOT / filename
        write_package(destination, samples[sample_id], arrays)
        print_package(destination)
    sample_id, filename = LEGACY_EXAMPLE
    destination = OUTPUT_ROOT / filename
    write_package(destination, samples[sample_id], arrays, legacy=True)
    print_package(destination)

    source_rows, all_arrays = source_arrays()
    rows_by_index = {int(row["sample_index"]): row for row in source_rows}
    built_in_keys = {
        (
            sample["set_id"],
            sample["video_id"],
            tuple(frame["source_frame"] for frame in sample["frames"]),
        )
        for sample in samples.values()
    }
    with tempfile.TemporaryDirectory(prefix="pie-upload-examples-") as directory:
        frame_root = Path(directory)
        all_index_groups = list(EXTERNAL_EXAMPLES.values())
        all_index_groups.extend(PREPARED_VIDEO_EXAMPLES.values())
        unique_index_groups = list(dict.fromkeys(all_index_groups))
        external_samples: dict[tuple[int, ...], dict] = {}
        for number, indices in enumerate(unique_index_groups, start=1):
            rows = [rows_by_index[index] for index in indices]
            sample = external_sample(
                rows,
                f"external-{number:02d}",
                frame_root,
                built_in_keys,
            )
            external_samples[indices] = sample

        for filename, indices in EXTERNAL_EXAMPLES.items():
            sample = external_samples[indices]
            destination = OUTPUT_ROOT / filename
            write_package(
                destination,
                sample,
                all_arrays,
                frame_root=frame_root,
            )
            print_package(destination)

        PREPARED_VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
        for folder_name, indices in PREPARED_VIDEO_EXAMPLES.items():
            sample = external_samples[indices]
            destination = PREPARED_VIDEO_ROOT / folder_name
            video = write_video(
                destination / "clip.mp4",
                frame_root / sample["id"],
            )
            package = destination / "clip_data.zip"
            write_package(
                package,
                sample,
                all_arrays,
                frame_root=frame_root,
                video=video,
            )
            print_package(destination / "clip.mp4")
            print_package(package)


if __name__ == "__main__":
    main()
