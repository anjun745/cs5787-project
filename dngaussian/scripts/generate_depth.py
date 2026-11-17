#!/usr/bin/env python3
"""Generate inverse-depth priors for prepared AIST++ images."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_root", type=Path)
    args = parser.parse_args()

    image_paths = sorted((args.prepared_root / "images").glob("c*/*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.prepared_root / 'images'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(device).eval()

    print(f"Generating {len(image_paths)} depth maps on {device}")
    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, 1):
            output_path = (
                args.prepared_root
                / "depth"
                / image_path.relative_to(args.prepared_root / "images").with_suffix(".npy")
            )
            if output_path.exists():
                continue

            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            prediction = model(**inputs).predicted_depth[:, None]
            prediction = F.interpolate(
                prediction,
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, prediction.float().cpu().numpy())
            if index % 50 == 0 or index == len(image_paths):
                print(f"Generated {index}/{len(image_paths)}")


if __name__ == "__main__":
    main()
