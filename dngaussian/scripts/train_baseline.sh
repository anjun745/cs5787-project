#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${ROOT}/scripts/train_aist_quick.sh" \
    train.tag=aist_3cam_baseline \
    train.depth_weight=0 \
    "$@"
