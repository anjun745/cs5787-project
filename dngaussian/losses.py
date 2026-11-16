"""Scale-invariant global/local depth losses.

This adapts the normalization idea from DNGaussian to D3GA. Both inputs are
expected to be inverse-depth maps; invalid/background pixels are excluded.
"""

import torch
import torch.nn.functional as F


def _as_bchw(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[None]
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"Expected a single-channel depth map, got {tuple(value.shape)}")
    return value


def _masked_stats(
    values: torch.Tensor, mask: torch.Tensor, dim: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.to(values.dtype)
    count = weights.sum(dim=dim, keepdim=True).clamp_min(1.0)
    mean = (values * weights).sum(dim=dim, keepdim=True) / count
    variance = ((values - mean).square() * weights).sum(dim=dim, keepdim=True) / count
    return mean, variance.sqrt().clamp_min(eps)


def global_local_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 32,
    local_weight: float = 0.1,
    min_valid_fraction: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compare depth structure without trusting the prior's scale or offset.

    The global term normalizes each full foreground map. The local term uses
    non-overlapping patches and retains only patches with enough valid pixels.
    """
    prediction = _as_bchw(prediction)
    target = _as_bchw(target).to(prediction)
    mask = _as_bchw(mask).bool()
    mask = mask & torch.isfinite(prediction) & torch.isfinite(target)

    if not mask.any():
        return prediction.sum() * 0.0

    height, width = prediction.shape[-2:]
    size = min(int(patch_size), height, width)
    if size < 2:
        pred_mean, pred_std = _masked_stats(prediction, mask, dim=(2, 3), eps=eps)
        target_mean, target_std = _masked_stats(target, mask, dim=(2, 3), eps=eps)
        error = (
            (prediction - pred_mean) / pred_std
            - (target - target_mean) / target_std
        ).square()
        return error[mask].mean()

    pred_patches = F.unfold(prediction, kernel_size=size, stride=size).transpose(1, 2)
    target_patches = F.unfold(target, kernel_size=size, stride=size).transpose(1, 2)
    mask_patches = F.unfold(mask.to(prediction.dtype), kernel_size=size, stride=size).transpose(1, 2) > 0.5
    valid_patches = mask_patches.sum(dim=-1) >= size * size * min_valid_fraction
    if not valid_patches.any():
        return prediction.sum() * 0.0

    pred_patch_mean, pred_patch_std = _masked_stats(pred_patches, mask_patches, dim=2, eps=eps)
    target_patch_mean, target_patch_std = _masked_stats(target_patches, mask_patches, dim=2, eps=eps)
    _, pred_global_std = _masked_stats(prediction, mask, dim=(2, 3), eps=eps)
    _, target_global_std = _masked_stats(target, mask, dim=(2, 3), eps=eps)
    pred_global_std = pred_global_std.flatten(1)[:, None]
    target_global_std = target_global_std.flatten(1)[:, None]

    # DNGaussian global normalization centers each patch locally but uses the
    # full image's standard deviation. This keeps a common scale across all
    # patches while the local term magnifies fine variation within each patch.
    global_error = (
        (pred_patches - pred_patch_mean) / pred_global_std
        - (target_patches - target_patch_mean) / target_global_std
    ).square()
    local_error = (
        (pred_patches - pred_patch_mean) / pred_patch_std
        - (target_patches - target_patch_mean) / target_patch_std
    ).square()
    local_mask = mask_patches & valid_patches[..., None]
    global_loss = global_error[local_mask].mean()
    local_loss = local_error[local_mask].mean()

    return global_loss + float(local_weight) * local_loss
