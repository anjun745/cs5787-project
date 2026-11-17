import torch

from dngaussian.losses import global_local_depth_loss


def test_affine_depth_is_normalized():
    target = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8) + 1
    prediction = target * 4.0 + 17.0
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = global_local_depth_loss(prediction, target, mask, patch_size=4)
    assert loss.item() < 1e-10


def test_background_is_ignored_and_gradients_flow():
    target = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8) + 1
    prediction = target.clone().requires_grad_(True)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, 2:6, 2:6] = True
    with torch.no_grad():
        prediction[:, :2] = 10000
    loss = global_local_depth_loss(prediction, target, mask, patch_size=4)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, :2].abs().sum().item() == 0


def test_constant_prediction_has_finite_gradients():
    prediction = torch.zeros(1, 64, 64, requires_grad=True)
    target = torch.rand_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    global_local_depth_loss(prediction, target, mask).backward()
    assert torch.isfinite(prediction.grad).all()
