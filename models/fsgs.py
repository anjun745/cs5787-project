"""FSGS proximity-guided Gaussian unpooling for D3GA cage models.

This module isolates the Gaussian densification/unpooling component from
CageNet. New Gaussians are initialized in canonical space and assigned
barycentric coordinates in the tetrahedral cage so they follow deformation.

Note: this implements proximity-guided unpooling, not every component of the
original FSGS method (such as virtual-view augmentation).
"""

import torch as th
import torch.nn.functional as F

from loguru import logger
from utils.gaussian_unpooling import select_unpool_pairs


def unpool_gaussians(model, fsgs_config):
    """Add Gaussians in locally sparse regions of the canonical cage.

    This first implementation is intentionally restricted to pairs in
    the same tetrahedron, which makes barycentric interpolation valid.
    It implements proximity-guided unpooling; pseudo-view/depth guidance
    are separate FSGS components and are not implemented here.
    """
    if not bool(fsgs_config.get("enabled", False)):
        return []

    geometry = model.geometry
    tetra_ids = geometry.tetra_id
    barys = geometry.barys
    if barys.numel() == 0:
        return []

    tetra_vertices = geometry.canonical_tetras[tetra_ids]
    canonical_positions = th.einsum("nij,ni->nj", tetra_vertices, barys)

    selection = select_unpool_pairs(
        canonical_positions=canonical_positions.detach(),
        knn_k=int(fsgs_config.get("knn_k", 4)),
        proximity_percentile=float(fsgs_config.get("proximity_percentile", 90.0)),
        max_new_gaussians=int(fsgs_config.get("max_new_gaussians", 250)),
    )
    pairs = selection["pairs"]
    if pairs.numel() == 0:
        return []

    parent_a, parent_b = pairs[:, 0], pairs[:, 1]
    midpoint_positions = 0.5 * (
        canonical_positions[parent_a] + canonical_positions[parent_b]
    )

    def barycentric_coordinates(points, tetra_vertices):
        origin = tetra_vertices[:, 0]
        basis = (tetra_vertices[:, 1:] - origin[:, None, :]).transpose(1, 2)
        coefficients = th.linalg.solve(
            basis, (points - origin).unsqueeze(-1)
        ).squeeze(-1)
        return th.cat([1.0 - coefficients.sum(dim=-1, keepdim=True), coefficients], dim=-1)

    # Prefer either parent's tetrahedron if it contains the midpoint.
    parent_a_tetra = tetra_ids[parent_a]
    parent_b_tetra = tetra_ids[parent_b]
    bary_a = barycentric_coordinates(
        midpoint_positions, geometry.canonical_tetras[parent_a_tetra]
    )
    bary_b = barycentric_coordinates(
        midpoint_positions, geometry.canonical_tetras[parent_b_tetra]
    )
    tolerance = 1e-4
    valid_a = (bary_a >= -tolerance).all(dim=-1) & (bary_a <= 1.0 + tolerance).all(dim=-1)
    valid_b = (bary_b >= -tolerance).all(dim=-1) & (bary_b <= 1.0 + tolerance).all(dim=-1)

    choose_b = (~valid_a) & valid_b
    new_tetra_ids = th.where(choose_b, parent_b_tetra, parent_a_tetra)
    new_barys = th.where(choose_b[:, None], bary_b, bary_a)

    # If neither parent tetrahedron contains the midpoint, use the nearest
    # tetrahedron centroid and project barycentric weights onto its simplex.
    needs_fallback = (~valid_a) & (~valid_b)
    if needs_fallback.any():
        all_tetra = geometry.canonical_tetras
        centroids = all_tetra.mean(dim=1)
        fallback_points = midpoint_positions[needs_fallback]
        nearest_tetra = th.cdist(
            fallback_points.unsqueeze(0), centroids.unsqueeze(0)
        ).squeeze(0).argmin(dim=1)
        fallback_bary = barycentric_coordinates(
            fallback_points, all_tetra[nearest_tetra]
        ).clamp_min(0.0)
        fallback_bary = fallback_bary / fallback_bary.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        new_tetra_ids[needs_fallback] = nearest_tetra
        new_barys[needs_fallback] = fallback_bary

    # Clamp tiny numerical negatives and renormalize to keep barycentrics
    # inside the tetrahedron.
    new_barys = new_barys.clamp_min(0.0)
    new_barys = new_barys / new_barys.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    new_points = th.einsum(
        "nij,ni->nj", geometry.canonical_tetras[new_tetra_ids], new_barys
    )
    rotation_a = model.rotation.detach()[parent_a]
    rotation_b = model.rotation.detach()[parent_b]
    # q and -q represent the same rotation; align signs before averaging.
    opposite = (rotation_a * rotation_b).sum(dim=-1, keepdim=True) < 0
    rotation_b = th.where(opposite, -rotation_b, rotation_b)
    new_rotations = F.normalize(0.5 * (rotation_a + rotation_b), dim=-1)
    new_scaling = 0.5 * (model.scaling.detach()[parent_a] + model.scaling.detach()[parent_b])
    new_colors = 0.5 * (model.colors_feat.detach()[parent_a] + model.colors_feat.detach()[parent_b])

    replacements = []
    for name, appended in (
        ("colors_feat", new_colors),
        ("rotation", new_rotations),
        ("scaling", new_scaling),
    ):
        old_parameter, new_parameter = model._expanded_parameter(
            getattr(model, name), appended
        )
        setattr(model, name, new_parameter)
        if old_parameter is not new_parameter:
            replacements.append((old_parameter, new_parameter))

    if model.use_SHS():
        for name in ("opacities", "features_dc", "features_rest"):
            parameter = getattr(model, name)
            appended = 0.5 * (
                parameter.detach()[parent_a] + parameter.detach()[parent_b]
            )
            old_parameter, new_parameter = model._expanded_parameter(
                parameter, appended
            )
            setattr(model, name, new_parameter)
            replacements.append((old_parameter, new_parameter))

    # These tensors are registered buffers on CageBase. Reassigning them
    # preserves buffer registration while extending the Gaussian dimension.
    geometry.barys = th.cat([geometry.barys, new_barys], dim=0)
    geometry.tetra_id = th.cat([geometry.tetra_id, new_tetra_ids], dim=0)
    geometry.init_points = th.cat([geometry.init_points, new_points], dim=0)
    geometry.init_rotations = th.cat([geometry.init_rotations, new_rotations], dim=0)

    new_canonical_tetras = geometry.canonical_tetras[new_tetra_ids]
    new_gradient = th.linalg.inv(geometry.cage.gradient(new_canonical_tetras))
    geometry.canonical_gradient = th.cat([geometry.canonical_gradient, new_gradient], dim=0)

    logger.info(
        f"[FSGS] Added {pairs.shape[0]} Gaussians to {model.name}; "
        f"total={model.colors_feat.shape[0]}"
    )
    return replacements

