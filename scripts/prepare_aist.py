#!/usr/bin/env python3
"""Extract a small synchronized AIST++ sequence and create person masks."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", default="gBR_sBM_cAll_d04_mBR0_ch01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--masks-only", action="store_true")
    return parser.parse_args()


def video_paths(dataset_root: Path, sequence: str) -> dict[str, Path]:
    paths = {}
    for camera_number in range(1, 10):
        camera = f"c{camera_number:02d}"
        name = sequence.replace("cAll", camera) + ".mp4"
        path = dataset_root / "videos" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[camera] = path
    return paths


def inspect_videos(paths: dict[str, Path]) -> tuple[int, dict[str, dict[str, float]]]:
    metadata = {}
    frame_counts = []
    for camera, path in paths.items():
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata[camera] = {
            "frames": frame_count,
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        frame_counts.append(frame_count)
        capture.release()
    return min(frame_counts), metadata


def validate_motion(dataset_root: Path, sequence: str, video_frames: int) -> Path:
    motion_path = dataset_root / "annotations" / "motions" / f"{sequence}.pkl"
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    with motion_path.open("rb") as handle:
        motion = pickle.load(handle)
    motion_frames = len(motion["smpl_poses"])
    if motion_frames < video_frames:
        raise RuntimeError(
            f"Motion has {motion_frames} frames but videos have {video_frames}"
        )
    return motion_path


def sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if num_frames < 1 or num_frames > total_frames:
        raise ValueError(f"num_frames must be between 1 and {total_frames}")
    return np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)


def extract_frames(
    paths: dict[str, Path],
    indices: np.ndarray,
    output: Path,
    size: tuple[int, int],
) -> None:
    selected = set(indices.tolist())
    for camera, path in paths.items():
        camera_output = output / "images" / camera
        camera_output.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(path))
        saved = 0
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in selected:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                destination = camera_output / f"{frame_index:06d}.jpg"
                if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not write {destination}")
                saved += 1
            frame_index += 1
        capture.release()
        if saved != len(indices):
            raise RuntimeError(f"Saved {saved}/{len(indices)} frames for {camera}")
        print(f"Extracted {saved} frames from {camera}", flush=True)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask * 255


def generate_masks(output: Path, batch_size: int) -> None:
    image_paths = sorted((output / "images").glob("c*/*.jpg"))
    if not image_paths:
        raise RuntimeError("No extracted images found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    categories = [category.lower() for category in weights.meta["categories"]]
    person_class = categories.index("person")
    model = deeplabv3_resnet50(weights=weights).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    print(f"Generating {len(image_paths)} masks on {device}", flush=True)
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for path in batch_paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(image).permute(2, 0, 1).to(device)
                tensor = tensor.float().div_(255.0)
                tensors.append((tensor - mean) / std)
            logits = model(torch.stack(tensors))["out"]
            predictions = logits.argmax(1).eq(person_class).cpu().numpy()
            for path, prediction in zip(batch_paths, predictions):
                relative = path.relative_to(output / "images").with_suffix(".png")
                destination = output / "masks" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), clean_mask(prediction)):
                    raise RuntimeError(f"Could not write {destination}")
            completed = min(start + batch_size, len(image_paths))
            if completed % 50 == 0 or completed == len(image_paths):
                print(f"Generated {completed}/{len(image_paths)} masks", flush=True)


def main() -> None:
    args = parse_args()
    if args.extract_only and args.masks_only:
        raise ValueError("--extract-only and --masks-only cannot be combined")

    paths = video_paths(args.dataset_root, args.sequence)
    total_frames, videos = inspect_videos(paths)
    motion_path = validate_motion(args.dataset_root, args.sequence, total_frames)
    indices = sample_indices(total_frames, args.num_frames)
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.masks_only:
        extract_frames(paths, indices, args.output, (args.width, args.height))
        metadata = {
            "sequence": args.sequence,
            "motion_file": str(motion_path),
            "sample_indices": indices.tolist(),
            "cameras": videos,
            "output_resolution": [args.width, args.height],
        }
        with (args.output / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    if not args.extract_only:
        generate_masks(args.output, args.batch_size)


if __name__ == "__main__":
    main()
