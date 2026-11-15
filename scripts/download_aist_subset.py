#!/usr/bin/env python3
"""Download the exact nine AIST videos and AIST++ annotations used here."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


TERMS_URL = "https://aistdancedb.ongaaccel.jp/terms_of_use/"
VIDEO_LIST_URL = "https://storage.googleapis.com/aist_plusplus_public/20121228/video_list.txt"
VIDEO_BASE_URL = "https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M"
ANNOTATION_URLS = {
    "motions.zip": "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/motions.zip",
    "cameras.zip": "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/cameras.zip",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / "datasets/AIST++")
    parser.add_argument("--sequence", default="gBR_sBM_cAll_d04_mBR0_ch01")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--agree-to-terms",
        action="store_true",
        help=f"Confirm that you read and agree to {TERMS_URL}",
    )
    parser.add_argument("--skip-annotations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def download(url: str, destination: Path):
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Already present: {destination}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)
    print(f"Saved {destination}", flush=True)
    return destination


def safe_extract(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def main():
    args = parse_args()
    if not args.agree_to_terms:
        raise SystemExit(
            f"Read {TERMS_URL}, then rerun with --agree-to-terms if you agree."
        )
    if "cAll" not in args.sequence:
        raise SystemExit("Sequence must contain the AIST++ cAll camera marker")

    names = [args.sequence.replace("cAll", f"c{camera:02d}") for camera in range(1, 10)]
    video_urls = [f"{VIDEO_BASE_URL}/{name}.mp4" for name in names]

    print("Videos:")
    for url in video_urls:
        print(f"  {url}")
    if not args.skip_annotations:
        print("Annotations:")
        for url in ANNOTATION_URLS.values():
            print(f"  {url}")
    if args.dry_run:
        return

    # Validate the requested names against Google's official AIST++ video list.
    with urllib.request.urlopen(VIDEO_LIST_URL) as response:
        available = {line.decode("utf-8").strip() for line in response if line.strip()}
    missing = [name for name in names if name not in available]
    if missing:
        raise RuntimeError(f"Videos absent from the official AIST++ list: {missing}")

    video_dir = args.dataset_root / "videos"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(download, url, video_dir / f"{name}.mp4")
            for name, url in zip(names, video_urls)
        ]
        for future in futures:
            future.result()

    if not args.skip_annotations:
        archive_dir = args.dataset_root / ".archives"
        annotation_dir = args.dataset_root / "annotations"
        for filename, url in ANNOTATION_URLS.items():
            archive = download(url, archive_dir / filename)
            safe_extract(archive, annotation_dir)
            print(f"Extracted {archive} into {annotation_dir}", flush=True)

    print(f"AIST++ subset is ready under {args.dataset_root}")


if __name__ == "__main__":
    main()
