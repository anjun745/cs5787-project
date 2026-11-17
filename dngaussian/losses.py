"""DNGaussian-style global/local inverse-depth loss."""

import torch
import torch.nn.functional as F


def _stats(x, mask):
    mask = mask.to(x.dtype)
    count = mask.sum(-1, keepdim=True).clamp_min(1)
    mean = (x * mask).sum(-1, keepdim=True) / count
    variance = ((x - mean).square() * mask).sum(-1, keepdim=True) / count
    return mean, (variance + 1e-6).sqrt()


def _patches(x, size):
    return F.unfold(x[None], size, stride=size).transpose(1, 2)


def global_local_depth_loss(prediction, target, mask, patch_size=32, local_weight=0.1):
    """Compare inverse-depth structure inside the foreground mask."""
    mask = mask & torch.isfinite(prediction) & torch.isfinite(target)
    if not mask.any():
        return prediction.sum() * 0

    pred_patches = _patches(prediction, patch_size)
    target_patches = _patches(target, patch_size)
    mask_patches = _patches(mask.float(), patch_size).bool()
    valid = mask_patches.sum(-1) >= patch_size * patch_size / 4
    loss_mask = mask_patches & valid[..., None]
    if not loss_mask.any():
        return prediction.sum() * 0

    pred_mean, pred_std = _stats(pred_patches, mask_patches)
    target_mean, target_std = _stats(target_patches, mask_patches)
    _, pred_global_std = _stats(prediction.flatten(1), mask.flatten(1))
    _, target_global_std = _stats(target.flatten(1), mask.flatten(1))

    global_error = (
        (pred_patches - pred_mean) / pred_global_std[..., None]
        - (target_patches - target_mean) / target_global_std[..., None]
    ).square()
    local_error = (
        (pred_patches - pred_mean) / pred_std
        - (target_patches - target_mean) / target_std
    ).square()

    return global_error[loss_mask].mean() + local_weight * local_error[loss_mask].mean()
