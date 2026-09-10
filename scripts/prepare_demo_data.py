#!/usr/bin/env python3
"""Prepare 15 deterministic PIE test sequences and their model tensors."""

from __future__ import annotations

import csv
import json
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = DEPLOYMENT_ROOT.parent / "PIE"
DATA_ROOT = DEPLOYMENT_ROOT / "data"
FRAME_ROOT = DATA_ROOT / "frames"
CLASS_NAMES = {0: "No Crossing", 1: "Crossing", -1: "Irrelevant"}
CLASS_ORDER = (0, 1, -1)
SAMPLES_PER_CLASS = 5


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def select_samples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Take the first test sample from five distinct videos per raw class."""
    test_rows = [row for row in rows if row["split"] == "test"]
    selected: list[dict[str, str]] = []
    for raw_class in CLASS_ORDER:
        class_rows = [row for row in test_rows if int(row["raw_crossing"]) == raw_class]
        used_videos: set[str] = set()
        chosen: list[dict[str, str]] = []
        for row in class_rows:
            if row["video_id"] in used_videos:
                continue
            chosen.append(row)
            used_videos.add(row["video_id"])
            if len(chosen) == SAMPLES_PER_CLASS:
                break
        if len(chosen) != SAMPLES_PER_CLASS:
            raise RuntimeError(f"Not enough distinct videos for raw class {raw_class}.")
        selected.extend(chosen)
    return selected


def load_target_boxes(
    annotation_path: Path,
    requested: dict[str, set[int]],
) -> dict[tuple[str, int], list[float]]:
    boxes: dict[tuple[str, int], list[float]] = {}
    for _, element in ET.iterparse(annotation_path, events=("end",)):
        if element.tag != "box":
            continue
        identifier = next(
            (
                child.text
                for child in element.findall("attribute")
                if child.attrib.get("name") == "id"
            ),
            None,
        )
        frame = int(element.attrib["frame"])
        if identifier in requested and frame in requested[identifier]:
            boxes[(identifier, frame)] = [
                float(element.attrib["xtl"]),
                float(element.attrib["ytl"]),
                float(element.attrib["xbr"]),
                float(element.attrib["ybr"]),
            ]
        element.clear()
    return boxes


def find_video(set_id: str, video_id: str) -> Path:
    matches = list((RESEARCH_ROOT / "PIE_clips" / set_id).rglob(f"{video_id}.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one clip for {set_id}/{video_id}, got {matches}"
        )
    return matches[0]


def annotate_frame(
    image: np.ndarray,
    pedestrian_id: str,
    box: list[float] | None,
) -> np.ndarray:
    if box is None:
        return image
    x1, y1, x2, y2 = (round(value) for value in box)
    color = (104, 239, 177)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
    label = f"TARGET {pedestrian_id}"
    (width, height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        2,
    )
    label_top = max(0, y1 - height - baseline - 12)
    cv2.rectangle(
        image,
        (x1, label_top),
        (x1 + width + 16, label_top + height + baseline + 12),
        (12, 27, 23),
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 8, label_top + height + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    return image


def export_frames(
    row: dict[str, str],
    sample_id: str,
    boxes: dict[tuple[str, int], list[float]],
) -> list[dict]:
    frames = json.loads(row["frames"])
    destination = FRAME_ROOT / sample_id
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(find_video(row["set_id"], row["video_id"])))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for {sample_id}.")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    output: list[dict] = []
    try:
        for ordinal, source_frame in enumerate(frames, start=1):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not decode frame {source_frame} for {sample_id}."
                )
            box = boxes.get((row["ped_id"], source_frame))
            image = annotate_frame(image, row["ped_id"], box)
            height, width = image.shape[:2]
            target_width = 960
            if width > target_width:
                image = cv2.resize(
                    image,
                    (target_width, round(height * target_width / width)),
                    interpolation=cv2.INTER_AREA,
                )
            filename = f"frame_{ordinal:02d}.jpg"
            if not cv2.imwrite(
                str(destination / filename),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 84],
            ):
                raise RuntimeError(f"Could not write {destination / filename}.")
            output.append(
                {
                    "ordinal": ordinal,
                    "source_frame": source_frame,
                    "url": f"/media/{sample_id}/{filename}",
                    "target_box": box,
                }
            )
    finally:
        capture.release()
    return output


def main() -> None:
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
        / "BinaryPrediction/outputs/BinaryPrediction/EnvGraphThreeClass"
        / "env_graph_dataset"
    )
    skeleton_data = (
        RESEARCH_ROOT
        / "outputs/BinaryPrediction/SkeletonThreeClassBBox15x19JointsSTGCN"
        / "skeleton_19joint_dataset"
    )
    relation_risk_data = (
        RESEARCH_ROOT
        / "outputs/BinaryPrediction/RelationRiskGraphThreeClass"
        / "relation_risk_graph_dataset"
    )
    trajectory_rows = read_index(trajectory_data / "traj_graph_index.csv")
    social_rows = read_index(social_data / "social_graph_index.csv")
    if trajectory_rows != social_rows:
        raise RuntimeError("Trajectory and social sample indexes are not identical.")
    comparison_indexes = {
        "environment": read_index(environment_data / "env_three_class_index.csv"),
        "skeleton": read_index(skeleton_data / "skeleton_19joint_index.csv"),
        "relation-risk": read_index(relation_risk_data / "relation_graph_index.csv"),
    }
    alignment_fields = (
        "sample_index",
        "set_id",
        "video_id",
        "ped_id",
        "split",
        "raw_crossing",
        "binary_crossing",
    )
    for name, rows in comparison_indexes.items():
        if len(rows) != len(trajectory_rows) or any(
            any(source[field] != candidate[field] for field in alignment_fields)
            for source, candidate in zip(trajectory_rows, rows, strict=True)
        ):
            raise RuntimeError(f"{name} samples are not aligned to the source index.")
    selected = select_samples(trajectory_rows)
    selected_indices = np.array([int(row["sample_index"]) for row in selected])

    arrays = {
        "trajectory_nodes": np.load(
            trajectory_data / "X_traj_nodes.npy",
            mmap_mode="r",
        ),
        "trajectory_adjacency": np.load(
            trajectory_data / "X_traj_adj.npy",
            mmap_mode="r",
        ),
        "trajectory_mask": np.load(
            trajectory_data / "X_traj_node_mask.npy",
            mmap_mode="r",
        ),
        "social_nodes": np.load(social_data / "X_social_nodes.npy", mmap_mode="r"),
        "social_adjacency": np.load(social_data / "X_social_adj.npy", mmap_mode="r"),
        "social_mask": np.load(social_data / "X_social_node_mask.npy", mmap_mode="r"),
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

    requests_by_video: dict[tuple[str, str], dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in selected:
        requests_by_video[(row["set_id"], row["video_id"])][row["ped_id"]].update(
            json.loads(row["frames"])
        )
    boxes_by_video = {
        key: load_target_boxes(
            RESEARCH_ROOT / "annotations" / key[0] / f"{key[1]}_annt.xml",
            request,
        )
        for key, request in requests_by_video.items()
    }

    resolved_frame_root = FRAME_ROOT.resolve()
    resolved_data_root = DATA_ROOT.resolve()
    if resolved_frame_root.parent != resolved_data_root:
        raise RuntimeError(
            "Refusing to replace frames outside the deployment data folder."
        )
    if resolved_frame_root.exists():
        shutil.rmtree(resolved_frame_root)
    FRAME_ROOT.mkdir(parents=True)
    samples = []
    for tensor_index, row in enumerate(selected):
        sample_id = f"sample-{tensor_index + 1:02d}"
        raw_class = int(row["raw_crossing"])
        frame_info = export_frames(
            row,
            sample_id,
            boxes_by_video[(row["set_id"], row["video_id"])],
        )
        samples.append(
            {
                "id": sample_id,
                "display_name": f"Example {tensor_index + 1:02d}",
                "tensor_index": tensor_index,
                "sample_index": int(row["sample_index"]),
                "set_id": row["set_id"],
                "video_id": row["video_id"],
                "pedestrian_id": row["ped_id"],
                "class_name": CLASS_NAMES[raw_class],
                "raw_class": raw_class,
                "binary_target": int(row["binary_crossing"]),
                "frames": frame_info,
            }
        )

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_ROOT / "prepared_samples.npz",
        **{name: array[selected_indices] for name, array in arrays.items()},
    )
    manifest = {
        "version": 2,
        "frame_count": 15,
        "selection": {
            "split": "test",
            "method": (
                "First indexed sample from five distinct videos per raw class; "
                "no correctness filtering."
            ),
            "class_balance": {name: SAMPLES_PER_CLASS for name in CLASS_NAMES.values()},
        },
        "samples": samples,
    }
    (DATA_ROOT / "samples.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(samples)} samples in {DATA_ROOT}")


if __name__ == "__main__":
    main()
