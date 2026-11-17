"""Proximity-guided Gaussian unpooling for tetrahedral-cage avatars.

This is a D3GA-compatible adaptation of the Gaussian-unpooling idea from
FSGS. Nearest neighbors are found in canonical 3D space. The caller is
responsible for embedding each new point in a valid cage tetrahedron.
"""

from __future__ import annotations

from typing import Dict

import torch


def select_unpool_pairs(
    canonical_positions: torch.Tensor,
    knn_k: int = 4,
    proximity_percentile: float = 90.0,
    max_new_gaussians: int = 250,
    chunk_size: int = 1024,
) -> Dict[str, torch.Tensor]:
    """Select Gaussian pairs in locally sparse canonical-space regions.

    The proximity score is the mean distance to the K nearest neighbors.
    Each Gaussian above the requested score percentile is paired with its
    nearest neighbor. Pair indices are undirected and deduplicated.
    """
    if canonical_positions.ndim != 2 or canonical_positions.shape[-1] != 3:
        raise ValueError("canonical_positions must have shape (N, 3)")
    if not 0.0 <= proximity_percentile <= 100.0:
        raise ValueError("proximity_percentile must be between 0 and 100")

    count = canonical_positions.shape[0]
    empty_pairs = torch.empty((0, 2), dtype=torch.long, device=canonical_positions.device)
    empty_scores = canonical_positions.new_empty((0,))
    if count < 2 or max_new_gaussians <= 0:
        return {"pairs": empty_pairs, "scores": empty_scores}

    k = min(max(int(knn_k), 1), count - 1)
    all_scores = canonical_positions.new_empty((count,))
    nearest_global = torch.empty((count,), dtype=torch.long, device=canonical_positions.device)

    # Chunk the distance calculation to avoid allocating a full N x N matrix.
    for start in range(0, count, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), count)
        distances = torch.cdist(
            canonical_positions[start:end].unsqueeze(0),
            canonical_positions.unsqueeze(0),
        ).squeeze(0)
        rows = torch.arange(end - start, device=canonical_positions.device)
        cols = torch.arange(start, end, device=canonical_positions.device)
        distances[rows, cols] = float("inf")
        nearest_distances, nearest_indices = torch.topk(
            distances, k=k, dim=1, largest=False, sorted=True
        )
        all_scores[start:end] = nearest_distances.mean(dim=1)
        nearest_global[start:end] = nearest_indices[:, 0]

    threshold = torch.quantile(all_scores, proximity_percentile / 100.0)
    selected = torch.nonzero(all_scores >= threshold, as_tuple=False).squeeze(-1)

    pair_to_score = {}
    for source in selected.tolist():
        target = int(nearest_global[source].item())
        pair = (min(source, target), max(source, target))
        score = max(float(all_scores[pair[0]].item()), float(all_scores[pair[1]].item()))
        pair_to_score[pair] = max(pair_to_score.get(pair, float("-inf")), score)

    if not pair_to_score:
        return {"pairs": empty_pairs, "scores": empty_scores}

    ranked = sorted(pair_to_score.items(), key=lambda item: item[1], reverse=True)
    ranked = ranked[:int(max_new_gaussians)]
    pairs = torch.tensor([item[0] for item in ranked], dtype=torch.long,
                         device=canonical_positions.device)
    scores = torch.tensor([item[1] for item in ranked], dtype=canonical_positions.dtype,
                          device=canonical_positions.device)
    return {"pairs": pairs, "scores": scores}
