"""
Evaluate a trained D3GA AIST++ model on the held-out frames.

Usage:
    python evaluate.py configs/aist_quick.yml
    python evaluate.py configs/aist_quick.yml 5000

The script uses the same held-out AIST++ split, rendering path, background
compositing, and compute_errors() metric implementation as test.py. It writes
per-frame metrics and their averages to CSV/TXT files.
"""

import argparse
import csv
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch as th
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import AttrDict
from datasets.aist_dataset import AISTDataset
from lib.batch import Batcher
from lib.smplman import Smplman
from models.trainer import Trainer
from recorder.heatmap import compute_errors


def create_aist_dataset(config):
    """Create the AIST++ held-out dataset and static camera assets."""
    dataset = AISTDataset(
        test_camera=config.train.test_camera,
        eval=True,
        **config.data,
    )
    assets = AttrDict({"camera_ids": list(dataset.cameras)})
    return dataset, assets


def configure_evaluation(config):
    """Match the evaluation-time switches used by the project's test.py."""
    config.train.use_background = False
    config.train.use_blur = False
    config.train.use_pixel_cal = False
    config.train.erode_mask = False
    config.train.close_holes = False


def to_float(value):
    """Convert a scalar tensor or NumPy value to a Python float."""
    if th.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def evaluate(config_path, iteration=None, output_dir=None, num_workers=None):
    if not th.cuda.is_available():
        raise RuntimeError("CUDA is not available. D3GA evaluation requires a CUDA device.")

    config = OmegaConf.load(config_path)
    if not os.path.isabs(config.assets):
        config.assets = os.path.abspath(config.assets)

    configure_evaluation(config)

    run_dir = Path(config.train.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(glob(str(checkpoint_dir / "*.pth")))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}. Train the model first."
        )

    # If an iteration is requested, Trainer.restore() selects that checkpoint
    # when present; otherwise it falls back to the latest checkpoint.
    dataset, assets = create_aist_dataset(config)
    logger.info(f"Evaluation dataset contains {len(dataset)} held-out frames")
    logger.info(f"Evaluation camera: {config.train.test_camera}")

    body_model = Smplman(config, assets)
    batcher = Batcher(config, body_model)

    trainer = Trainer(config, assets, is_eval=True)
    trainer.eval()
    _, restored_iteration = trainer.restore(
        iteration=iteration,
        strict=False,
        return_state=True,
    )
    logger.info(f"Evaluating checkpoint at iteration {restored_iteration}")

    if output_dir is None:
        output_dir = run_dir / f"evaluation_{int(restored_iteration):06d}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = int(config.train.get("num_workers", 2))
    num_workers = max(0, int(num_workers))

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )

    bg_color = config.train.get("background", "white")
    white_background = float(bg_color == "white")

    rows = []
    ssim_values = []
    psnr_values = []
    lpips_values = []

    with th.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="Evaluating")):
            if batch is None:
                logger.warning(f"Skipping empty batch {batch_index}")
                continue

            frames = batcher.process(batch)
            for frame in frames:
                # This matches test.py's test trajectory behavior.
                frame["iteration"] = float("inf")
                pkg = trainer.fit(frame)

                prediction = pkg["pred_image"]
                ground_truth = frame["image"].to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )

                # Match the exact foreground/boundary compositing used in test.py.
                boundary_fg = 1.0 - frame["boundary_fg"].float()
                alpha_gt = frame["alpha"][0:1, :, :] * boundary_fg
                ground_truth_composited = (
                    ground_truth * alpha_gt
                    + (1.0 - alpha_gt) * white_background
                )

                _, frame_ssim, frame_psnr, frame_lpips = compute_errors(
                    ground_truth_composited,
                    prediction,
                    use_npc=False,
                    pkg=(config, batch_index, "test"),
                )

                frame_ssim = to_float(frame_ssim)
                frame_psnr = to_float(frame_psnr)
                frame_lpips = to_float(frame_lpips)

                ssim_values.append(frame_ssim)
                psnr_values.append(frame_psnr)
                lpips_values.append(frame_lpips)

                rows.append({
                    "frame_index": len(rows),
                    "ssim": frame_ssim,
                    "psnr": frame_psnr,
                    "lpips": frame_lpips,
                })

    if not rows:
        raise RuntimeError(
            "No frames were evaluated. Check the AIST++ held-out split and dataset paths."
        )

    mean_ssim = float(np.mean(ssim_values))
    mean_psnr = float(np.mean(psnr_values))
    mean_lpips = float(np.mean(lpips_values))

    csv_path = output_dir / "metrics_per_frame.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["frame_index", "ssim", "psnr", "lpips"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "metrics_summary.txt"
    with summary_path.open("w") as summary_file:
        summary_file.write(f"Checkpoint iteration: {restored_iteration}\n")
        summary_file.write(f"Dataset: {config.dataset_name}\n")
        summary_file.write(f"Camera: {config.train.test_camera}\n")
        summary_file.write(f"Number of evaluated frames: {len(rows)}\n")
        summary_file.write(f"SSIM: {mean_ssim:.6f}\n")
        summary_file.write(f"PSNR: {mean_psnr:.6f} dB\n")
        summary_file.write(f"LPIPS: {mean_lpips:.6f}\n")

    logger.info("Evaluation complete")
    logger.info(f"Frames evaluated: {len(rows)}")
    logger.info(f"Mean SSIM:  {mean_ssim:.6f} (higher is better)")
    logger.info(f"Mean PSNR:  {mean_psnr:.6f} dB (higher is better)")
    logger.info(f"Mean LPIPS: {mean_lpips:.6f} (lower is better)")
    logger.info(f"Per-frame metrics: {csv_path}")
    logger.info(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained D3GA AIST++ checkpoint using PSNR, SSIM, and LPIPS."
    )
    parser.add_argument(
        "config",
        help="Path to the training YAML configuration, e.g. configs/aist_quick.yml",
    )
    parser.add_argument(
        "iteration",
        nargs="?",
        default=None,
        help="Checkpoint iteration to evaluate. Defaults to the latest checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for the CSV and summary files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Defaults to train.num_workers from the YAML.",
    )
    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        iteration=args.iteration,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
