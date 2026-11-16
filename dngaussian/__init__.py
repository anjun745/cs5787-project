"""DNGaussian-inspired depth regularization for the D3GA training path."""

from .losses import global_local_depth_loss

__all__ = ["global_local_depth_loss"]
