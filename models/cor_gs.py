# CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization
# (Zhang et al., ECCV 2024, https://arxiv.org/abs/2405.12110), adapted to D3GA.
#
# Two Gaussian avatars are trained side by side on the same views. Where they
# disagree, at least one of them is probably wrong, so disagreement is suppressed
# in two ways:
#
#   * Co-pruning (point disagreement): a Gaussian whose nearest neighbour in the
#     other field is farther than tau is pruned.
#   * Pseudo-view co-regularization (rendering disagreement): both fields render
#     an unseen view near the training cameras, and the two renderings are pulled
#     together with (1 - lambda) * L1 + lambda * D-SSIM.
#
# D3GA differs from the static 3DGS setting the paper was written for, so the
# following adaptations are made (see README, "CoR-GS"):
#
#   * Gaussians move with the body. Point disagreement is measured on the posed
#     Gaussian centres of the current training frame, so both fields are
#     compared in the same body state.
#   * D3GA has no densification and a fixed Gaussian budget. Pruning therefore
#     sets a persistent per-Gaussian mask (CageNet.active_mask) instead of
#     deleting optimizer rows. The cap on how much one event may prune is
#     max_ratio.
#   * Scene units differ between datasets (AIST++ ~ metres, Goliath mm), so
#     tau and the pseudo-camera noise are relative by default: tau is a multiple
#     of the median Gaussian spacing, and the noise std is a fraction of the
#     distance between the two cameras being interpolated.

import numpy as np
import torch as th
from loguru import logger
from pytorch3d.ops import knn_points
from scipy.spatial.transform import Rotation, Slerp

from utils.loss_utils import l1_loss, ssim


def _to_numpy(x):
    if th.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _homogeneous(Rt):
    Rt = _to_numpy(Rt).astype(np.float64).reshape(-1, 4)[:3]
    M = np.eye(4)
    M[:3] = Rt
    return M


class PseudoViewSampler:
    """Samples unseen cameras between neighbouring training cameras.

    The paper writes the pseudo pose as P' = (t + eps, q), where t is a training
    camera position, eps ~ N(0, sigma^2) and q is the rotation averaged from the
    two nearest training cameras (following FSGS). Here, for a training camera
    i and one of its k nearest cameras j:

        u      ~ U(0, 1)          (u = 0.5 if interpolate is False)
        centre = lerp(c_i, c_j, u) + eps,  eps ~ N(0, (noise * |c_i - c_j|)^2 I)
        R      = slerp(R_i, R_j, u)

    With interpolate=False the camera stays at c_i + eps and uses the averaged
    (u = 0.5) rotation, which is the literal form in the paper.

    D3GA moves the cameras rather than the body (Batcher bakes the global body
    transform B into each frame's c2w). Because c2w'_k = B^-1 (w2c_k)^-1, the
    neighbour camera in the frame's space is c2w'_j = c2w'_i @ w2c_i @ c2w_j.
    That expression does not depend on B, so the raw dataset extrinsics are
    enough.
    """

    def __init__(self, cameras, k_nearest=2, noise=0.05, interpolate=True, seed=0):
        self.names = list(cameras.keys())
        if len(self.names) < 2:
            raise ValueError("CoR-GS pseudo views need at least two training cameras")
        for name in self.names:
            if "Rt" not in cameras[name]:
                raise ValueError(f"Camera {name} has no 'Rt' extrinsics; pseudo views are unsupported for this dataset")

        self.w2c = {n: _homogeneous(cameras[n]["Rt"]) for n in self.names}
        self.c2w = {n: np.linalg.inv(m) for n, m in self.w2c.items()}
        centers = np.stack([self.c2w[n][:3, 3] for n in self.names])

        k = max(1, min(int(k_nearest), len(self.names) - 1))
        dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1)[:, :k]
        self.neighbours = {self.names[i]: [self.names[j] for j in order[i]] for i in range(len(self.names))}

        self.noise = float(noise)
        self.interpolate = bool(interpolate)
        self.rng = np.random.default_rng(seed)

    def sample_c2w(self, c2w_i, camera_i, w2c_i=None):
        """Returns a pseudo camera-to-world matrix in the same space as c2w_i."""
        c2w_i = _to_numpy(c2w_i).astype(np.float64)
        w2c_i = self.w2c[camera_i] if w2c_i is None else _homogeneous(w2c_i)
        camera_j = self.neighbours[camera_i][self.rng.integers(len(self.neighbours[camera_i]))]
        c2w_j = c2w_i @ w2c_i @ self.c2w[camera_j]

        u = float(self.rng.uniform()) if self.interpolate else 0.5
        rotations = Rotation.from_matrix(np.stack([c2w_i[:3, :3], c2w_j[:3, :3]]))
        R = Slerp([0.0, 1.0], rotations)([u]).as_matrix()[0]

        c_i, c_j = c2w_i[:3, 3], c2w_j[:3, 3]
        baseline = np.linalg.norm(c_j - c_i)
        anchor = (1.0 - u) * c_i + u * c_j if self.interpolate else c_i
        center = anchor + self.rng.normal(scale=self.noise * baseline, size=3)

        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = center
        return c2w

    def __call__(self, frame):
        """Returns a shallow copy of a Batcher frame, seen from a pseudo camera.

        Intrinsics, crop, pose and the source camera id (used for per-camera
        embeddings or calibration, if enabled) are inherited from the
        training frame.
        """
        c2w = self.sample_c2w(frame["c2w"], frame["camera_id"], frame.get("Rt"))
        w2c = np.linalg.inv(c2w)
        pseudo = dict(frame)
        pseudo["c2w"] = c2w.astype(np.float32)
        # Same convention as lib/batch.py
        pseudo["R"] = np.transpose(w2c[:3, :3]).astype(np.float32)
        pseudo["T"] = w2c[:3, 3].astype(np.float32)
        return pseudo


def photometric_disagreement(img_a, img_b, lambda_dssim):
    """R_pcolor = (1 - lambda) * L1(I'_1, I'_2) + lambda * D-SSIM(I'_1, I'_2)."""
    return (1.0 - lambda_dssim) * l1_loss(img_a, img_b) + lambda_dssim * (1.0 - ssim(img_a, img_b))


class CoRGS:
    """Co-regularization for two D3GA fields (models.trainer.Trainer instances)."""

    def __init__(self, cor_config, cameras, rgb_weight=1.0, seed=0):
        self.cfg = cor_config
        self.pseudo_cfg = cor_config.pseudo
        self.prune_cfg = cor_config.co_prune
        self.rgb_weight = float(rgb_weight)
        self.sampler = PseudoViewSampler(
            cameras,
            k_nearest=self.pseudo_cfg.get("k_nearest", 2),
            noise=self.pseudo_cfg.get("noise", 0.05),
            interpolate=self.pseudo_cfg.get("interpolate", True),
            seed=seed,
        )
        logger.info(f"CoR-GS pseudo-view neighbours: {self.sampler.neighbours}")

    # ------------------------------------------------------------------ #
    # Pseudo-view co-regularization
    # ------------------------------------------------------------------ #
    def pseudo_weight(self, iteration):
        """lambda_p, scaled to D3GA's rgb_weight and linearly ramped in.

        The paper's total loss is L_color + lambda_p * R_pcolor. D3GA multiplies
        its colour loss by rgb_weight, so lambda_p is scaled by the same weight
        to keep the paper's 1:1 balance.
        """
        start = self.pseudo_cfg.get("start", 500)
        end = self.pseudo_cfg.get("end", -1)
        if iteration < start or (end > 0 and iteration > end):
            return 0.0
        every = max(1, self.pseudo_cfg.get("every", 1))
        if (iteration - start) % every != 0:
            return 0.0
        ramp = self.pseudo_cfg.get("ramp", 0)
        scale = min(1.0, (iteration - start + 1) / ramp) if ramp > 0 else 1.0
        return self.pseudo_cfg.get("weight", 1.0) * self.rgb_weight * scale

    def pseudo_loss(self, trainers, frames, iteration):
        """Returns (weighted loss, unweighted disagreement) or (0, None) when inactive."""
        weight = self.pseudo_weight(iteration)
        if weight == 0.0:
            return 0.0, None

        lambda_dssim = self.pseudo_cfg.get("lambda_dssim", 0.2)
        terms = []
        for frame in frames:
            pseudo = self.sampler(frame)
            # Both fields share one random background, so any disagreement in
            # transparency shows up as a colour difference.
            bg = th.rand(3, device="cuda", dtype=th.float32)
            img_a = trainers[0].render_view(pseudo, bg)
            img_b = trainers[1].render_view(pseudo, bg)
            terms.append(photometric_disagreement(img_a, img_b, lambda_dssim))

        disagreement = th.stack(terms).mean()
        return weight * disagreement, disagreement.detach()

    # ------------------------------------------------------------------ #
    # Co-pruning
    # ------------------------------------------------------------------ #
    def should_prune(self, iteration):
        start = self.prune_cfg.get("start", 1000)
        until = self.prune_cfg.get("until", -1)
        every = max(1, self.prune_cfg.get("every", 500))
        if iteration < start or (until > 0 and iteration > until):
            return False
        return (iteration - start) % every == 0

    @th.no_grad()
    def co_prune(self, trainers, frame):
        """Prunes the Gaussians of each field that have no match in the other.

        M_i = 1 if ||mu_i^1 - KNN(mu_i^1, Theta^2)|| > tau (and vice versa).
        """
        centers = [t.model.gaussian_centers(frame) for t in trainers]
        points = [th.cat(c, dim=0) for c in centers]

        tau = float(self.prune_cfg.get("tau", 5.0))
        if self.prune_cfg.get("relative_tau", True):
            spacing = th.cat([_nn_distance(p, p, k=2) for p in points]).median()
            threshold = tau * spacing.item()
        else:
            threshold = tau

        max_ratio = float(self.prune_cfg.get("max_ratio", 1.0))
        stats = {"cor/threshold": threshold}
        for k, (field_centers, own, other) in enumerate(
            [(centers[0], points[0], points[1]), (centers[1], points[1], points[0])]
        ):
            distance = _nn_distance(own, other, k=1)
            drop = distance > threshold
            n_flagged = int(drop.sum().item())

            cap = int(max_ratio * own.shape[0])
            if n_flagged > cap:
                # Only the most disagreeing Gaussians go; the rest wait for the next event.
                worst = th.topk(distance, cap).indices
                drop = th.zeros_like(drop)
                drop[worst] = True

            sizes = [c.shape[0] for c in field_centers]
            for garment, garment_drop in zip(trainers[k].model.garments, th.split(drop, sizes)):
                garment.prune(garment_drop)

            stats[f"cor/flagged_f{k}"] = n_flagged
            stats[f"cor/pruned_f{k}"] = int(drop.sum().item())
            stats[f"cor/active_f{k}"] = sum(g.n_active() for g in trainers[k].model.garments)

        return stats


def _nn_distance(query, reference, k=1):
    """Euclidean distance from each query point to its k-th nearest reference point."""
    dists = knn_points(query[None].float(), reference[None].float(), K=k).dists[0, :, k - 1]
    return dists.clamp_min(0.0).sqrt()
