# DNGaussian depth regularization for D3GA

This directory contains the first-stage adaptation of DNGaussian's
global/local depth normalization to the dynamic, cage-controlled D3GA model.
It deliberately keeps D3GA's SMPL motion, tetrahedral cage, deformation MLPs,
and neural color renderer.

## Depth-prior layout

Generate one single-channel depth or inverse-depth array per prepared AIST++
image. The loader expects:

```text
<depth_root>/
  c01/000000.npy
  c01/000010.npy
  c02/000000.npy
  ...
```

The filename is the original AIST++ frame number used under `images/`.
Floating-point `.npy` files are preferred. PNG priors are also supported by
setting `data.depth_format: png`.

Add the root to the AIST config:

```yaml
data:
  depth_root: ${oc.env:AIST_ROOT}/prepared/gBR_sBM_cAll_d04_mBR0_ch01/depth
  depth_format: npy
```

Then set `train.depth_regularization.enabled: true`. Use `target_type: depth`
for metric/forward depth, or `target_type: inverse_depth` for disparity-like
monocular predictions. Training fails early if an enabled prior is missing.

## Current scope

The renderer returns its differentiable alpha-weighted inverse-depth map and
the loss compares it with the prior inside the foreground mask. Global
normalization constrains broad body shape; non-overlapping local patches
constrain smaller surface changes without trusting monocular scale or offset.

This stage does not implement DNGaussian's hard-opacity and opacity-only
passes. D3GA centers are outputs of a shared cage/deformation network rather
than independent parameters, so those passes require explicit gradient
routing or bounded per-Gaussian canonical offsets before they are safe.
