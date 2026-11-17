#!/usr/bin/env python3
"""Convert the licensed legacy SMPL pickle to a NumPy-backed local pickle."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    # Chumpy 0.70 imports NumPy aliases removed in modern NumPy. They are only
    # needed while unpickling the legacy, user-downloaded SMPL model.
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    with args.input.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")

    converted = {}
    for key, value in model.items():
        if value.__class__.__module__.startswith("chumpy"):
            converted[key] = np.asarray(value.r)
        else:
            converted[key] = value

    # tetra-sampler registers these buffers even for body-only SMPL models.
    converted.setdefault("lmk_faces_idx", np.empty(0, dtype=np.int64))
    converted.setdefault("lmk_bary_coords", np.empty((0, 3), dtype=np.float32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Converted SMPL model written to {args.output}")


if __name__ == "__main__":
    main()
