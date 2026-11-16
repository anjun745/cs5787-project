import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class AISTDataset(Dataset):
    """Small AIST++ adapter for D3GA's ActorsHQ-style training path."""

    def __init__(
        self,
        prepared_root,
        camera_file,
        n_testing_frames=20,
        test_camera="c01",
        depth_root=None,
        depth_format="npy",
        eval=False,
        warmup=False,
        **kwargs,
    ):
        self.root = Path(prepared_root)
        self.eval = eval
        self.warmup = warmup
        self.test_camera = test_camera
        self.depth_root = Path(depth_root) if depth_root else None
        self.depth_format = depth_format.lstrip(".")
        self.internal_counter = 0
        self.warmup_idx = 0

        with open(self.root / "metadata.json") as f:
            metadata = json.load(f)
        self.sequence = metadata["sequence"]
        self.frames = [int(x) for x in metadata["sample_indices"]]
        with open(metadata["motion_file"], "rb") as f:
            motion = pickle.load(f)
        self.poses = np.asarray(motion["smpl_poses"], dtype=np.float32).reshape(-1, 72)
        self.trans = np.asarray(motion["smpl_trans"], dtype=np.float32)
        self.scale = float(np.asarray(motion["smpl_scaling"]).reshape(-1)[0])

        # Hold out the last sampled poses, just like the ActorsHQ loader.
        split = max(1, len(self.frames) - int(n_testing_frames))
        self.frame_list = self.frames[split:] if eval else self.frames[:split]

        with open(camera_file) as f:
            camera_params = json.load(f)
        self.cameras = {}
        for camera in camera_params:
            camera_id = camera["name"]
            K = np.asarray(camera["matrix"], dtype=np.float32)
            source_w, source_h = camera["size"]
            image = cv2.imread(str(self.root / "images" / camera_id / f"{self.frames[0]:06d}.jpg"))
            if image is None:
                raise FileNotFoundError(f"No prepared images for AIST++ camera {camera_id}")
            height, width = image.shape[:2]
            K[0] *= width / source_w
            K[1] *= height / source_h
            R = cv2.Rodrigues(np.asarray(camera["rotation"], dtype=np.float32))[0]
            # AIST++ coordinates are scaled SMPL coordinates. Dividing the whole
            # scene by the fitted scale lets D3GA use the ordinary SMPL template.
            T = np.asarray(camera["translation"], dtype=np.float32).reshape(3, 1) / self.scale
            self.cameras[camera_id] = {
                "K": K,
                "Rt": np.concatenate([R, T], axis=1).astype(np.float32),
            }
        self.cam2idx = {name: i for i, name in enumerate(self.cameras)}
        if self.test_camera not in self.cameras:
            raise ValueError(f"Unknown test camera {self.test_camera}; choose from {list(self.cameras)}")

    def n_cameras(self):
        return len(self.cameras)

    def n_frames(self):
        return len(self.frame_list)

    def random_frames(self, n=200):
        n = min(n, len(self.frame_list))
        self.frame_list = np.random.choice(self.frame_list, n, replace=False).tolist()

    def __len__(self):
        return len(self.frame_list)

    @staticmethod
    def get_boundary_mask(mask, kernel_size=3):
        binary = (mask >= 128).astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        boundary = cv2.dilate(binary, kernel) != cv2.erode(binary, kernel)
        return boundary, binary.astype(bool)

    def __getitem__(self, idx):
        if self.warmup and not self.eval:
            if (self.internal_counter + 1) % 100 == 0:
                self.warmup_idx = np.random.randint(len(self))
            idx = self.warmup_idx
        self.internal_counter += 1

        frame = self.frame_list[idx]
        camera_id = self.test_camera if self.eval else np.random.choice(list(self.cameras))
        image_path = self.root / "images" / camera_id / f"{frame:06d}.jpg"
        mask_path = self.root / "masks" / camera_id / f"{frame:06d}.png"
        image_bgr = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            raise FileNotFoundError(f"Missing prepared AIST++ frame: {image_path}")

        boundary, foreground = self.get_boundary_mask(mask)
        pose = self.poses[frame].copy()
        smpl = {
            # Keep global orientation in the 72-D pose. D3GA applies Th by
            # moving the camera, so Rh must remain zero here.
            "poses": pose,
            "shapes": np.zeros(10, dtype=np.float32),
            "expression": np.zeros(10, dtype=np.float32),
            "Rh": np.zeros(3, dtype=np.float32),
            "Th": self.trans[frame] / self.scale,
        }
        sample = {
            "smplx": smpl,
            "lbs_motion": pose,
            "image": image_bgr[..., ::-1].transpose(2, 0, 1).astype(np.float32),
            "seg_fg": foreground[None],
            "boundary_fg": boundary[None],
            # A body-only run only needs foreground/background labels.
            "seg_part": foreground[None].astype(np.int32) * 4,
            "_index": {
                "frame": idx,
                "seq": self.sequence,
                "order_frame_idx": frame,
                "order_cam_idx": self.cam2idx[camera_id],
            },
            "camera_ids": [camera_id],
        }
        sample.update(self.cameras[camera_id])

        if self.depth_root is not None:
            depth_path = self.depth_root / camera_id / f"{frame:06d}.{self.depth_format}"
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing depth prior: {depth_path}")
            if self.depth_format == "npy":
                depth = np.load(depth_path).astype(np.float32)
            else:
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth is None:
                    raise FileNotFoundError(f"Could not read depth prior: {depth_path}")
                depth = depth.astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.shape != foreground.shape[-2:]:
                depth = cv2.resize(
                    depth,
                    (foreground.shape[-1], foreground.shape[-2]),
                    interpolation=cv2.INTER_LINEAR,
                )
            sample["depth_prior"] = depth[None]

        return sample
