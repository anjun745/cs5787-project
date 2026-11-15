#!/usr/bin/env python3
"""Render a procedural raise-hand-and-wave motion with a trained D3GA avatar."""

from __future__ import annotations

import argparse
import copy
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import AttrDict
from datasets.aist_dataset import AISTDataset
from lib.batch import Batcher
from lib.smplman import Smplman
from models.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aist_quick.yml")
    parser.add_argument("--iteration", type=int, default=5000)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def make_wave_motion(root_orient, n_frames=120):
    """Create standing -> raise right hand -> wave -> lower hand."""
    if n_frames < 60:
        raise ValueError("Use at least 60 frames for a readable wave")

    poses = np.zeros((n_frames, 72), dtype=np.float32)
    poses[:, :3] = root_orient

    left_shoulder = 16
    right_shoulder = 17
    right_elbow = 19

    # SMPL's zero pose is a T-pose. Start with both arms relaxed downward.
    left_down = -1.35
    right_down = 1.35
    right_up = -1.57
    poses[:, left_shoulder * 3 + 2] = left_down

    raise_end = int(round(n_frames * 0.25))
    wave_end = int(round(n_frames * 0.75))
    for frame in range(n_frames):
        if frame < raise_end:
            amount = smoothstep(frame / max(raise_end - 1, 1))
            shoulder = (1.0 - amount) * right_down + amount * right_up
            elbow = 0.0
        elif frame < wave_end:
            phase = (frame - raise_end) / max(wave_end - raise_end - 1, 1)
            shoulder = right_up
            # Three friendly side-to-side waves of the raised forearm.
            elbow = 0.55 * np.sin(phase * 6.0 * np.pi)
        else:
            amount = smoothstep((frame - wave_end) / max(n_frames - wave_end - 1, 1))
            shoulder = (1.0 - amount) * right_up + amount * right_down
            elbow = 0.0

        poses[frame, right_shoulder * 3 + 2] = shoulder
        poses[frame, right_elbow * 3 + 2] = elbow

    return poses


class CustomMotionDataset(Dataset):
    def __init__(self, reference_dataset, poses, translation):
        self.reference = reference_dataset[0]
        self.poses = poses
        self.translation = np.asarray(translation, dtype=np.float32)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        sample = copy.deepcopy(self.reference)
        pose = self.poses[index]
        sample["smplx"]["poses"] = pose
        sample["smplx"]["Rh"] = np.zeros(3, dtype=np.float32)
        sample["smplx"]["Th"] = self.translation
        sample["lbs_motion"] = pose
        sample["_index"]["frame"] = index
        sample["_index"]["order_frame_idx"] = index
        return sample


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    if not os.path.isabs(config.assets):
        config.assets = os.path.abspath(config.assets)
    config.train.use_background = False
    config.train.use_blur = False
    config.train.use_pixel_cal = False
    config.train.erode_mask = False
    config.train.close_holes = False

    reference = AISTDataset(
        test_camera=config.train.test_camera,
        eval=True,
        **config.data,
    )
    # Use a known, centered root orientation/position from the source capture,
    # while replacing every articulated body joint with our custom motion.
    source_frame = min(621, len(reference.poses) - 1)
    root_orient = reference.poses[source_frame, :3].copy()
    translation = reference.trans[source_frame] / reference.scale
    poses = make_wave_motion(root_orient, args.frames)
    dataset = CustomMotionDataset(reference, poses, translation)

    assets = AttrDict({"camera_ids": list(reference.cameras)})
    body_model = Smplman(config, assets)
    batcher = Batcher(config, body_model)
    trainer = Trainer(config, assets, is_eval=True)
    trainer.eval()
    trainer.restore(args.iteration, strict=False)

    output = Path(config.train.run_dir) / f"novel_wave_{args.iteration:06d}"
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output / "wave_motion.npz", poses=poses, translation=translation, fps=args.fps)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc="Rendering custom wave")):
            frame = batcher.process(batch)[0]
            frame["iteration"] = float("inf")
            rendering = trainer.fit(frame)["pred_image"]
            torchvision.utils.save_image(rendering, frame_dir / f"{index:05d}.png")

    frame_paths = sorted(frame_dir.glob("*.png"))
    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    video_path = output / "raise_hand_and_wave.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {video_path}")
    for frame_path in frame_paths:
        writer.write(cv2.imread(str(frame_path)))
    writer.release()
    print(f"Saved {len(frame_paths)} frames to {video_path}")


if __name__ == "__main__":
    main()
