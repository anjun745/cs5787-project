# DNGaussian depth regularization for D3GA

This directory contains the first-stage adaptation of DNGaussian's
global/local depth normalization to the dynamic, cage-controlled D3GA model.
It deliberately keeps D3GA's SMPL motion, tetrahedral cage, deformation MLPs,
and neural color renderer.

## Depth-prior layout

Generate one single-channel inverse-depth `.npy` file per prepared image:

```text
<depth_root>/
  c01/000000.npy
  c01/000010.npy
  c02/000000.npy
  ...
```

The filename is the original AIST++ frame number. Set the root and loss weight:

```bash
python dngaussian/scripts/generate_depth.py "$AIST_ROOT/prepared/gBR_sBM_cAll_d04_mBR0_ch01"
```

```yaml
data:
  depth_root: ${oc.env:AIST_ROOT}/prepared/gBR_sBM_cAll_d04_mBR0_ch01/depth
train:
  depth_weight: 0.05
```

The current AIST config trains on `c01`, `c04`, and `c07`, then evaluates on
the unseen `c02` camera.

Run the matching baseline and depth experiments with:

```bash
./dngaussian/scripts/train_baseline.sh
./dngaussian/scripts/train_depth.sh
```

After both runs finish, render the held-out camera and poses:

```bash
./dngaussian/scripts/evaluate.sh baseline
./dngaussian/scripts/evaluate.sh depth
python dngaussian/scripts/compare.py
```

Both runs write under `dngaussian/outputs/`:

```text
dngaussian/outputs/
  aist_3cam_baseline/gBR_sBM_cAll_d04_mBR0_ch01/body/
  aist_3cam_depth/gBR_sBM_cAll_d04_mBR0_ch01/body/
```

The generated depth maps stay under the AIST++ dataset directory because they
are cached input data, not model output.

## Current scope

The renderer returns its differentiable alpha-weighted inverse-depth map and
the loss compares it with the prior inside the foreground mask. Global
normalization constrains broad body shape; non-overlapping local patches
constrain smaller surface changes without trusting monocular scale or offset.

Hard/soft optimization is intentionally out of scope for this first test.
