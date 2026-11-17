#!/usr/bin/env python3
"""Create a baseline-vs-depth video and foreground metrics."""

from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "dngaussian" / "outputs"
BASELINE = next((OUTPUTS / "aist_3cam_baseline").glob("*/body/cinema_005000/cam_test"))
DEPTH = next((OUTPUTS / "aist_3cam_depth").glob("*/body/cinema_005000/cam_test"))


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255


def foreground_crop(gt_path, baseline_path, depth_path):
    rgba = np.asarray(Image.open(gt_path), dtype=np.float32) / 255
    mask = rgba[..., 3] > 0.5
    gt = rgba[..., :3] * mask[..., None] + ~mask[..., None]
    baseline = load_rgb(baseline_path)
    depth = load_rgb(depth_path)

    y, x = np.where(mask)
    pad = 20
    y0, y1 = max(0, y.min() - pad), min(mask.shape[0], y.max() + pad + 1)
    x0, x1 = max(0, x.min() - pad), min(mask.shape[1], x.max() + pad + 1)
    crop_mask = mask[y0:y1, x0:x1, None]

    def crop(image):
        image = image[y0:y1, x0:x1]
        return image * crop_mask + ~crop_mask

    return gt, baseline, depth, mask, crop(gt), crop(baseline), crop(depth)


def psnr(gt, prediction, mask):
    mse = np.square(gt - prediction)[mask].mean()
    return -10 * np.log10(max(mse, 1e-10))


def lpips_tensor(image):
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None]
    return F.interpolate(tensor, (256, 256), mode="bilinear", align_corners=False)[0] * 2 - 1


def label(image, text):
    image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(image, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def main():
    gt_paths = sorted((BASELINE / "ground_truth").glob("*.png"))
    baseline_paths = sorted((BASELINE / "prediction").glob("*.png"))
    depth_paths = sorted((DEPTH / "prediction").glob("*.png"))
    if not (len(gt_paths) == len(baseline_paths) == len(depth_paths)):
        raise RuntimeError("Baseline and depth evaluations are incomplete")

    destination = OUTPUTS / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(destination / "baseline_vs_depth.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (1440, 270),
    )

    baseline_psnr, depth_psnr = [], []
    baseline_ssim, depth_ssim = [], []
    baseline_lpips, depth_lpips = [], []

    for gt_path, baseline_path, depth_path in zip(gt_paths, baseline_paths, depth_paths):
        gt, baseline, depth, mask, gt_crop, baseline_crop, depth_crop = foreground_crop(
            gt_path, baseline_path, depth_path
        )
        baseline_psnr.append(psnr(gt, baseline, mask))
        depth_psnr.append(psnr(gt, depth, mask))
        baseline_ssim.append(structural_similarity(gt_crop, baseline_crop, channel_axis=2, data_range=1))
        depth_ssim.append(structural_similarity(gt_crop, depth_crop, channel_axis=2, data_range=1))
        baseline_lpips.append(lpips_tensor(baseline_crop))
        depth_lpips.append(lpips_tensor(depth_crop))
        video.write(np.concatenate([label(gt, "Ground truth"), label(baseline, "Baseline"), label(depth, "Ours + depth")], axis=1))

    video.release()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = lpips.LPIPS(net="alex").to(device).eval()
    gt_batch = torch.stack([lpips_tensor(foreground_crop(g, b, d)[4]) for g, b, d in zip(gt_paths, baseline_paths, depth_paths)]).to(device)
    with torch.inference_mode():
        baseline_lpips = metric(gt_batch, torch.stack(baseline_lpips).to(device)).mean().item()
        depth_lpips = metric(gt_batch, torch.stack(depth_lpips).to(device)).mean().item()

    values = {
        "Baseline": (np.mean(baseline_psnr), np.mean(baseline_ssim), baseline_lpips),
        "Depth": (np.mean(depth_psnr), np.mean(depth_ssim), depth_lpips),
    }
    with (destination / "foreground_metrics.txt").open("w") as file:
        for name, (psnr_value, ssim_value, lpips_value) in values.items():
            file.write(f"{name}: PSNR {psnr_value:.5f}, SSIM {ssim_value:.5f}, LPIPS {lpips_value:.5f}\n")

    print((destination / "baseline_vs_depth.mp4").relative_to(ROOT))
    print((destination / "foreground_metrics.txt").read_text(), end="")


if __name__ == "__main__":
    main()
